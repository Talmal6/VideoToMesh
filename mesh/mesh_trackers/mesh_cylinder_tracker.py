import cv2
import numpy as np
from typing import Optional
from copy import deepcopy
from detection.detection import Detection
from mesh.mesh_shapes.mesh_object import MeshObject
from mesh.mesh_trackers.mesh_tracker import MeshTracker, State


class MeshCylinderTracker(MeshTracker):
    """
    Fast tracker:
    1) Predict motion between frames using LK + estimateAffinePartial2D:
       translation + uniform scale + in-plane rotation (roll).
    2) Apply that transform to last mesh vertices (in image coords).
    3) Optional fast refinement with coordinate ascent on (tx, ty, rz, scale).
       - No deepcopy per candidate
       - ROI-only IoU scoring
    """

    def __init__(self):
        super().__init__()

        # Refinement parameters (keep small if you want speed)
        self.max_refine_iters = 6

        self.step_trans = 2.0                 # pixels
        self.step_rot = np.radians(2.0)       # radians
        self.step_scale = 1.02                # multiplicative

        self.iou_threshold = 0.25

        # LK / features parameters
        self.max_corners = 80
        self.quality_level = 0.3
        self.min_distance = 7
        self.lk_win = (15, 15)
        self.lk_max_level = 2

    def can_track(self, det: Detection) -> bool:
        return det.label in ("bottle", "cup", "can", "box", "book", "tv", "monitor", "phone") and det.mask is not None

    def track(self, det: Detection, curr_frame) -> Optional[MeshObject]:
        if (not self.last_state) or (self.last_state.last_mesh is None) or (self.last_state.last_frame is None):
            return None
        if curr_frame is None:
            return None

        prev_frame = self.last_state.last_frame
        prev_det = self.last_state.last_det
        prev_mask = None
        if prev_det is not None and getattr(prev_det, "mask", None) is not None:
            prev_mask = prev_det.mask
        else:
            return None

        # Start from last mesh (copy once, outside loops)
        mesh = deepcopy(self.last_state.last_mesh)
        mesh.frame_index = det.frame_index
        mesh.mask = det.mask
        mesh.bbox_xyxy = det.bbox_xyxy

        base_vertices = mesh.vertices.astype(np.float32)  # (N,3) in image coords
        if base_vertices.size == 0:
            return None

        # 1) Predict motion: (tx, ty, scale, rz) using optical flow affine
        tx, ty, scale, rz = self._estimate_affine_motion(prev_frame, prev_mask, curr_frame)
        # Apply predicted transform
        pred_vertices = self._apply_pose_2d(base_vertices, tx=tx, ty=ty, rz=rz, scale=scale)

        # Score prediction
        best_score = self._compute_iou_score_roi(pred_vertices[:, :2], det.mask)
        best_pose = (tx, ty, rz, scale)

        # 2) Optional fast refinement (single-parameter moves only)
        # Coordinate ascent: each iteration chooses best +/- step for each param independently.
        # No combinations; no deepcopy; only cheap vertex transform + ROI IoU.
        cur_tx, cur_ty, cur_rz, cur_scale = best_pose

        for _ in range(self.max_refine_iters):
            improved = False

            # refine translation X
            cur_tx, best_score, improved = self._refine_param(
                base_vertices, det.mask,
                cur_tx, cur_ty, cur_rz, cur_scale,
                param="tx", step=self.step_trans,
                best_score=best_score,
                improved=improved
            )

            # refine translation Y
            cur_ty, best_score, improved = self._refine_param(
                base_vertices, det.mask,
                cur_tx, cur_ty, cur_rz, cur_scale,
                param="ty", step=self.step_trans,
                best_score=best_score,
                improved=improved
            )

            # refine rotation Z
            cur_rz, best_score, improved = self._refine_param(
                base_vertices, det.mask,
                cur_tx, cur_ty, cur_rz, cur_scale,
                param="rz", step=self.step_rot,
                best_score=best_score,
                improved=improved
            )

            # refine scale (multiplicative)
            cur_scale, best_score, improved = self._refine_param_scale(
                base_vertices, det.mask,
                cur_tx, cur_ty, cur_rz, cur_scale,
                step=self.step_scale,
                best_score=best_score,
                improved=improved
            )

            if not improved:
                break

        if best_score < self.iou_threshold:
            return None

        # Commit final vertices to mesh
        final_vertices = self._apply_pose_2d(base_vertices, tx=cur_tx, ty=cur_ty, rz=cur_rz, scale=cur_scale)
        mesh.vertices = final_vertices
        mesh.confidence = float(best_score)

        # Update state
        self.last_state = State(
            last_frame=curr_frame,
            last_det=det,
            last_mesh=mesh
        )
        return mesh

    # -------------------------
    # Motion estimation
    # -------------------------
    def _estimate_affine_motion(self, prev_frame, prev_mask, curr_frame):
        """
        Estimates (tx, ty, scale, rz) that maps prev->curr in full image coordinates.
        Uses features inside prev_mask.
        Returns identity if insufficient signal.
        """
        if prev_frame is None or curr_frame is None or prev_mask is None:
            return 0.0, 0.0, 1.0, 0.0

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY) if prev_frame.ndim == 3 else prev_frame
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY) if curr_frame.ndim == 3 else curr_frame

        # Ensure mask aligns with prev_gray
        mask = prev_mask
        if mask.dtype != np.uint8:
            mask = (mask > 0).astype(np.uint8) * 255
        if mask.shape[:2] != prev_gray.shape[:2]:
            mask = cv2.resize(mask, (prev_gray.shape[1], prev_gray.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Erode to avoid border points
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
        if int(mask.sum()) == 0:
            return 0.0, 0.0, 1.0, 0.0

        p0 = cv2.goodFeaturesToTrack(
            prev_gray,
            mask=mask,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            blockSize=7
        )
        if p0 is None or len(p0) < 8:
            return 0.0, 0.0, 1.0, 0.0

        p1, st, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, p0, None,
            winSize=self.lk_win,
            maxLevel=self.lk_max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        if p1 is None or st is None:
            return 0.0, 0.0, 1.0, 0.0

        st = st.reshape(-1)
        good_old = p0.reshape(-1, 2)[st == 1]
        good_new = p1.reshape(-1, 2)[st == 1]
        if len(good_old) < 8:
            return 0.0, 0.0, 1.0, 0.0

        M, inliers = cv2.estimateAffinePartial2D(good_old, good_new)
        if M is None:
            return 0.0, 0.0, 1.0, 0.0

        # Extract (scale, rz) from rotation+scale part; translation from last column
        a00, a01, tx = float(M[0, 0]), float(M[0, 1]), float(M[0, 2])
        a10, a11, ty = float(M[1, 0]), float(M[1, 1]), float(M[1, 2])

        # For estimateAffinePartial2D with uniform scale:
        # [ s*cos  -s*sin ]
        # [ s*sin   s*cos ]
        scale = float(np.sqrt(a00 * a00 + a10 * a10))
        if scale <= 1e-6:
            return 0.0, 0.0, 1.0, 0.0

        rz = float(np.arctan2(a10, a00))

        # Clamp to avoid insane jumps (keeps tracker stable and fast)
        rz = float(np.clip(rz, -np.radians(15.0), np.radians(15.0)))
        scale = float(np.clip(scale, 0.85, 1.15))
        tx = float(np.clip(tx, -50.0, 50.0))
        ty = float(np.clip(ty, -50.0, 50.0))

        return tx, ty, scale, rz

    # -------------------------
    # Pose application
    # -------------------------
    def _apply_pose_2d(self, vertices_xyz: np.ndarray, tx: float, ty: float, rz: float, scale: float) -> np.ndarray:
        """
        Apply uniform scale + in-plane rotation around centroid (x,y), plus translation.
        Keeps Z scaled as well (optional; you can keep Z unchanged if you prefer).
        """
        v = vertices_xyz.copy()  # Copy array only (cheap vs deepcopy object)
        xy = v[:, :2]

        # rotate/scale about centroid in image coordinates
        c = xy.mean(axis=0)
        xy0 = xy - c

        cos_r = float(np.cos(rz))
        sin_r = float(np.sin(rz))

        R = np.array([[cos_r, -sin_r],
                      [sin_r,  cos_r]], dtype=np.float32)

        xy1 = (xy0 @ R.T) * float(scale)
        xy2 = xy1 + c + np.array([tx, ty], dtype=np.float32)

        v[:, 0] = xy2[:, 0]
        v[:, 1] = xy2[:, 1]
        v[:, 2] *= float(scale)

        return v

    # -------------------------
    # Fast ROI IoU scoring
    # -------------------------
    def _compute_iou_score_roi(self, pts2d: np.ndarray, target_mask: np.ndarray) -> float:
        """
        IoU between convex hull of pts2d and target mask, computed only in the ROI bbox of the hull.
        """
        if pts2d is None or len(pts2d) < 3 or target_mask is None:
            return 0.0

        h, w = target_mask.shape[:2]
        pts = pts2d.astype(np.int32)

        hull = cv2.convexHull(pts)
        x, y, rw, rh = cv2.boundingRect(hull)
        if rw <= 0 or rh <= 0:
            return 0.0

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w, x + rw)
        y2 = min(h, y + rh)
        if x2 <= x1 or y2 <= y1:
            return 0.0

        # Local hull coords
        hull_roi = hull.copy()
        hull_roi[:, 0, 0] -= x1
        hull_roi[:, 0, 1] -= y1

        roi_h = y2 - y1
        roi_w = x2 - x1

        mesh_roi = np.zeros((roi_h, roi_w), dtype=np.uint8)
        cv2.fillConvexPoly(mesh_roi, hull_roi, 1)

        t = (target_mask[y1:y2, x1:x2] > 0).astype(np.uint8)

        inter = int(np.logical_and(t, mesh_roi).sum())
        union = int(np.logical_or(t, mesh_roi).sum())
        if union == 0:
            return 0.0
        return float(inter) / float(union)

    # -------------------------
    # Refinement helpers (single-parameter)
    # -------------------------
    def _refine_param(self, base_vertices, mask, tx, ty, rz, scale, param, step, best_score, improved):
        """
        Try +/- step for one parameter and accept best.
        Returns (new_param_value, new_best_score, improved_flag).
        """
        candidates = []
        if param == "tx":
            candidates = [tx + step, tx - step]
        elif param == "ty":
            candidates = [ty + step, ty - step]
        elif param == "rz":
            candidates = [rz + step, rz - step]
        else:
            return (tx if param == "tx" else ty if param == "ty" else rz), best_score, improved

        best_val = (tx if param == "tx" else ty if param == "ty" else rz)

        for val in candidates:
            if param == "tx":
                v = self._apply_pose_2d(base_vertices, tx=val, ty=ty, rz=rz, scale=scale)
            elif param == "ty":
                v = self._apply_pose_2d(base_vertices, tx=tx, ty=val, rz=rz, scale=scale)
            else:  # rz
                v = self._apply_pose_2d(base_vertices, tx=tx, ty=ty, rz=val, scale=scale)

            score = self._compute_iou_score_roi(v[:, :2], mask)
            if score > best_score:
                best_score = score
                best_val = val
                improved = True

        return best_val, best_score, improved

    def _refine_param_scale(self, base_vertices, mask, tx, ty, rz, scale, step, best_score, improved):
        """
        Try scale * step and scale / step and accept best.
        """
        candidates = [scale * step, scale / step]
        best_val = scale

        for val in candidates:
            v = self._apply_pose_2d(base_vertices, tx=tx, ty=ty, rz=rz, scale=val)
            score = self._compute_iou_score_roi(v[:, :2], mask)
            if score > best_score:
                best_score = score
                best_val = val
                improved = True

        return best_val, best_score, improved
