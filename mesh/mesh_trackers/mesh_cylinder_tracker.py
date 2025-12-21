import cv2
import numpy as np
from copy import deepcopy
from class_proccesors.detection import Detection
from mesh.mesh_shapes.mesh_object import MeshObject
from mesh.mesh_trackers.mesh_tracker import MeshTracker, State


class MeshCylinderTracker(MeshTracker):
    
    def __init__(self):
        super().__init__()
        # Hill climbing parameters
        self.step_trans = 2.0  # pixels
        self.step_rot = np.radians(2.0)  # radians
        self.step_scale = 1.02  # factor
        self.max_iterations = 20
        self.iou_threshold = 0.9

    def can_track(self, det: Detection) -> bool:
        return det.label in ("bottle", "cup", "can") and det.mask is not None
    

    def track(self, det: Detection, curr_frame) -> MeshObject:
        if not self.last_state or not self.last_state.last_mesh:

            return None


        current_mesh = deepcopy(self.last_state.last_mesh)
        
 
        current_mesh.frame_index = det.frame_index
        current_mesh.mask = det.mask
        current_mesh.bbox_xyxy = det.bbox_xyxy

        # --- 1. Estimate Motion (Z-Rotation & Scale) from Image Data ---
        if self.last_state.last_frame is not None and curr_frame is not None:
            scale, angle = self._estimate_motion(
                self.last_state.last_frame, 
                self.last_state.last_mask if hasattr(self.last_state, 'last_mask') else self.last_state.last_det.mask, 
                curr_frame
            )
            if scale != 1.0:
                if scale > 1.0:
                    self.enlarge(current_mesh, scale)
                else:
                    self.shrink(current_mesh, 1.0/scale)
            
            # if angle != 0.0:
            #     self.rotate_z(current_mesh, angle)

        best_score = self._compute_score(current_mesh, det.mask)
        
        # Hill Climbing / Greedy Local Search
        for _ in range(self.max_iterations):
            improved = False
            best_op = None
            best_arg = None
            
            # Define candidate moves
            # (Function Name, Argument, Inverse Function Name, Inverse Argument)
            moves = [
                ('translate_x', self.step_trans),
                ('translate_x', -self.step_trans),
                ('translate_y', self.step_trans),
                ('translate_y', -self.step_trans),
                ('rotate_x', self.step_rot),
                ('rotate_x', -self.step_rot),
                ('rotate_y', self.step_rot),
                ('rotate_y', -self.step_rot),
                ('enlarge', self.step_scale),
                ('shrink', self.step_scale), # shrink uses factor > 1 divisor logic in base
            ]

            # Try all moves
            for op_name, arg in moves:
                # Create a temporary candidate
                candidate = deepcopy(current_mesh)
                
                # Apply transform
                getattr(self, op_name)(candidate, arg)
                
                # Score
                score = self._compute_score(candidate, det.mask)
                
                if score > best_score:
                    best_score = score
                    best_op = op_name
                    best_arg = arg
                    improved = True

            # If we found an improvement, apply it to the main mesh and continue
            if improved:
                getattr(self, best_op)(current_mesh, best_arg)
            else:
                # Local maximum reached
                break

        # --- 2. Estimate Roll (Z-Rotation) from Optical Flow ---
        roll_angle = self._estimate_roll_from_flow(det, curr_frame)
        if roll_angle is not None and abs(roll_angle) > 1e-4:
             self.rotate_z(current_mesh, roll_angle)

        if best_score < self.iou_threshold:
            return None

        # Update state
        self.last_state = State(
            last_frame=curr_frame,
            last_det=det,
            last_mesh=current_mesh
        )

        return current_mesh

    def _compute_score(self, mesh: MeshObject, target_mask: np.ndarray) -> float:
        """
        Projects the mesh to 2D and computes Intersection over Union (IoU) 
        or Coverage with the target mask.
        """
        if mesh.vertices is None or len(mesh.vertices) == 0:
            return 0.0

        h, w = target_mask.shape[:2]
        
        pts = mesh.vertices[:, :2].astype(np.int32)
        
        if len(pts) < 3:
            return 0.0
            
        hull = cv2.convexHull(pts)
        mesh_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mesh_mask, hull, 255)

        t_mask = (target_mask > 0).astype(np.uint8)
        m_mask = (mesh_mask > 0).astype(np.uint8)

        intersection = np.logical_and(t_mask, m_mask).sum()
        union = np.logical_or(t_mask, m_mask).sum()

        if union == 0:
            return 0.0
            
        return float(intersection) / float(union)

    def _estimate_motion(self, prev_frame, prev_mask, curr_frame):
        """
        Estimates scale and rotation (Z-axis) change between frames using Optical Flow.
        Returns (scale, angle_radians).
        """
        if prev_frame is None or curr_frame is None or prev_mask is None:
            return 1.0, 0.0

        # Convert to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # Find features in previous frame within the mask
        # Ensure mask is uint8
        mask = prev_mask
        if mask.dtype != np.uint8:
            mask = (mask > 0).astype(np.uint8) * 255
        
        # Erode mask slightly to avoid border features
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)

        p0 = cv2.goodFeaturesToTrack(prev_gray, mask=mask, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

        if p0 is None or len(p0) < 4:
            return 1.0, 0.0

        # Calculate Optical Flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Select good points
        if p1 is None:
            return 1.0, 0.0
            
        good_new = p1[st==1]
        good_old = p0[st==1]

        if len(good_new) < 4:
            return 1.0, 0.0

        # Estimate Affine Partial 2D (Translation + Rotation + Uniform Scale)
        # [ s*cos(a)  -s*sin(a)  tx ]
        # [ s*sin(a)   s*cos(a)  ty ]
        M, inliers = cv2.estimateAffinePartial2D(good_old, good_new)

        if M is None:
            return 1.0, 0.0

        # Extract Scale and Rotation
        # s*cos(a) = M[0,0]
        # s*sin(a) = M[1,0]
        s_cos = M[0, 0]
        s_sin = M[1, 0]

        scale = np.sqrt(s_cos**2 + s_sin**2)
        angle = np.arctan2(s_sin, s_cos)

        return scale, angle

    def _estimate_roll_from_flow(self, det: Detection, curr_frame) -> float | None:
        """
        Estimate in-plane rotation (roll / rotate_z) between last_frame and curr_frame
        using tracked features inside ROI/mask. Uses both object texture and background
        cues around it (via ROI), but masks to prefer points on/near the object.
        """
        prev_frame = self.last_state.last_frame
        prev_det = self.last_state.last_det
        if prev_frame is None or prev_det is None or prev_det.mask is None:
            return None

        # grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY) if prev_frame.ndim == 3 else prev_frame
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY) if curr_frame.ndim == 3 else curr_frame

        x1, y1, x2, y2 = det.bbox_xyxy
        x1, y1 = int(max(0, x1)), int(max(0, y1))
        x2, y2 = int(min(curr_gray.shape[1] - 1, x2)), int(min(curr_gray.shape[0] - 1, y2))
        if x2 <= x1 or y2 <= y1:
            return None

        # ROI
        prev_roi = prev_gray[y1:y2, x1:x2]
        curr_roi = curr_gray[y1:y2, x1:x2]

        # mask ROI (prefer object area, but allow a thin band to capture rotation cues)
        mask = (det.mask[y1:y2, x1:x2] > 0).astype(np.uint8) * 255
        if mask.sum() == 0:
            return None

        # good features to track (within mask)
        p0 = cv2.goodFeaturesToTrack(
            prev_roi,
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            mask=mask
        )
        if p0 is None or len(p0) < 8:
            return None

        # optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            prev_roi, curr_roi, p0, None,
            winSize=(15, 15), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        if p1 is None:
            return None

        st = st.reshape(-1)
        p0 = p0.reshape(-1, 2)[st == 1]
        p1 = p1.reshape(-1, 2)[st == 1]
        if len(p0) < 8:
            return None

        # Estimate rotation around ROI center using angle differences
        cx = (x2 - x1) * 0.5
        cy = (y2 - y1) * 0.5

        v0 = p0 - np.array([cx, cy], dtype=np.float32)
        v1 = p1 - np.array([cx, cy], dtype=np.float32)

        # Avoid near-center points (unstable angle)
        r0 = np.linalg.norm(v0, axis=1)
        r1 = np.linalg.norm(v1, axis=1)
        keep = (r0 > 10.0) & (r1 > 10.0)
        v0, v1 = v0[keep], v1[keep]
        if len(v0) < 6:
            return None

        a0 = np.arctan2(v0[:, 1], v0[:, 0])
        a1 = np.arctan2(v1[:, 1], v1[:, 0])

        da = a1 - a0
        # wrap to [-pi, pi]
        da = (da + np.pi) % (2 * np.pi) - np.pi

        # robust estimate: median
        angle = float(np.median(da))

        # clamp (avoid huge jumps)
        angle = float(np.clip(angle, -np.radians(15.0), np.radians(15.0)))
        return angle