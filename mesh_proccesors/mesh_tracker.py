"""
MeshTracker: Tracks MeshData across frames using similarity transform.
Applies transformation (translation + rotation + scale) to all mesh primitives.

Dependencies: cv2, numpy only.
"""

import cv2
import numpy as np
from copy import deepcopy
from mesh_proccesors.mesh_factory import MeshData


class MeshTracker:
    """
    Tracks MeshData using KLT optical flow + RANSAC similarity estimation.
    Updates vertices, poly2d, rings2d, centerline with the transformation.
    """

    def __init__(
        self,
        max_corners=200,
        quality_level=0.01,
        min_distance=5,
        lk_win=(21, 21),
        lk_max_level=3,
        ransac_reproj_thresh=3.0,
    ):
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.ransac_reproj_thresh = ransac_reproj_thresh

        self.lk_params = dict(
            winSize=lk_win,
            maxLevel=lk_max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

    @staticmethod
    def _to_gray(frame):
        if frame.ndim == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    @staticmethod
    def _poly_mask(poly2d, h, w):
        """Create uint8 mask from polygon (full-frame)."""
        if poly2d is None or len(poly2d) < 3:
            return None
        m = np.zeros((h, w), dtype=np.uint8)
        pts = np.asarray(poly2d, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(m, [pts], 255)
        return m

    @staticmethod
    def _affine_to_similarity_params(A2x3):
        """
        Extract similarity parameters from affine matrix.
        For A = [[a,b,tx],[c,d,ty]] from estimateAffinePartial2D:
        scale ~ sqrt(a^2 + c^2), rotation ~ atan2(c,a)
        """
        a, b, tx = A2x3[0]
        c, d, ty = A2x3[1]
        scale = float(np.sqrt(a * a + c * c))
        angle = float(np.degrees(np.arctan2(c, a)))
        return scale, angle, float(tx), float(ty)

    @staticmethod
    def _apply_affine_to_points(A2x3, pts_xy):
        """Apply affine transform to 2D points. pts_xy: (N,2) -> (N,2)"""
        pts = np.asarray(pts_xy, dtype=np.float32).reshape(-1, 2)
        ones = np.ones((pts.shape[0], 1), dtype=np.float32)
        P = np.hstack([pts, ones])  # (N,3)
        out = (P @ A2x3.T).astype(np.float32)  # (N,2)
        return out

    def _track_meshdata(self, old_gray, new_gray, mesh_data):
        """
        Track a single MeshData between frames.
        Returns updated MeshData or None if tracking failed.
        """
        h, w = old_gray.shape[:2]

        poly2d = mesh_data.poly2d
        if poly2d is None or len(poly2d) < 3:
            return None

        # 1) Create mask from polygon, find features inside object
        mask = self._poly_mask(poly2d, h, w)
        if mask is None:
            return None

        pts = cv2.goodFeaturesToTrack(
            old_gray,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            mask=mask,
        )
        if pts is None or len(pts) < 10:
            return None

        # 2) KLT optical flow
        new_pts, st, _ = cv2.calcOpticalFlowPyrLK(
            old_gray, new_gray, pts, None, **self.lk_params
        )
        if new_pts is None or st is None:
            return None

        st = st.reshape(-1).astype(bool)
        if st.sum() < 10:
            return None

        old_good = pts[st].reshape(-1, 2)
        new_good = new_pts[st].reshape(-1, 2)

        # 3) Robust similarity transform (rotation + scale + translation)
        A2x3, inliers = cv2.estimateAffinePartial2D(
            old_good,
            new_good,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_reproj_thresh,
            confidence=0.99,
            maxIters=2000,
        )
        if A2x3 is None:
            return None

        inl = inliers.reshape(-1).astype(bool) if inliers is not None else np.ones(len(old_good), dtype=bool)
        inlier_ratio = float(inl.mean()) if len(inl) else 0.0
        if inlier_ratio < 0.4:
            return None

        scale, angle_deg, dx, dy = self._affine_to_similarity_params(A2x3)

        # 4) Apply transform to ALL render primitives + vertices
        # Make a copy to avoid mutating original
        updated = deepcopy(mesh_data)

        # poly2d: reshape for transform, then back
        poly_flat = mesh_data.poly2d.reshape(-1, 2)
        new_poly = self._apply_affine_to_points(A2x3, poly_flat)
        updated.poly2d = new_poly.round().astype(np.int32).reshape(-1, 1, 2)

        # centerline (2 endpoints)
        x1, y1, x2, y2 = mesh_data.centerline
        cl = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
        cl2 = self._apply_affine_to_points(A2x3, cl).round().astype(np.int32)
        updated.centerline = (int(cl2[0, 0]), int(cl2[0, 1]), int(cl2[1, 0]), int(cl2[1, 1]))

        # rings2d: [cx, cy, ax, ay, angle]
        rings = mesh_data.rings2d
        if rings is not None and len(rings) > 0:
            centers = rings[:, 0:2].astype(np.float32)
            centers2 = self._apply_affine_to_points(A2x3, centers)

            ax = np.maximum(1, np.round(rings[:, 2].astype(np.float32) * scale)).astype(np.int32)
            ay = np.maximum(1, np.round(rings[:, 3].astype(np.float32) * scale)).astype(np.int32)
            ang = (rings[:, 4].astype(np.int32) + int(round(angle_deg)))

            new_rings = np.zeros_like(rings, dtype=np.int32)
            new_rings[:, 0:2] = np.round(centers2).astype(np.int32)
            new_rings[:, 2] = ax
            new_rings[:, 3] = ay
            new_rings[:, 4] = ang
            updated.rings2d = new_rings

        # vertices: apply to x,y; scale z for consistency
        V = mesh_data.vertices.astype(np.float32)
        xy = V[:, 0:2]
        xy2 = self._apply_affine_to_points(A2x3, xy)
        V2 = V.copy()
        V2[:, 0:2] = xy2
        V2[:, 2] *= scale
        updated.vertices = V2

        # Box-specific fields: transform if present
        if mesh_data.box_front2d is not None and len(mesh_data.box_front2d) > 0:
            front = mesh_data.box_front2d.reshape(-1, 2).astype(np.float32)
            updated.box_front2d = self._apply_affine_to_points(A2x3, front).round().astype(np.int32)

        if mesh_data.box_back2d is not None and len(mesh_data.box_back2d) > 0:
            back = mesh_data.box_back2d.reshape(-1, 2).astype(np.float32)
            updated.box_back2d = self._apply_affine_to_points(A2x3, back).round().astype(np.int32)

        if mesh_data.box_edges2d is not None and len(mesh_data.box_edges2d) > 0:
            # box_edges2d is shape (12, 2, 2) - flatten to (24, 2), transform, reshape back
            edges_flat = mesh_data.box_edges2d.reshape(-1, 2).astype(np.float32)
            edges_transformed = self._apply_affine_to_points(A2x3, edges_flat)
            updated.box_edges2d = edges_transformed.round().astype(np.int32).reshape(-1, 2, 2)

        # 5) Update tracking metadata
        updated.dx = dx
        updated.dy = dy
        updated.scale = scale
        updated.angle_deg = mesh_data.angle_deg + angle_deg
        updated.tracking_confidence = inlier_ratio

        return updated

    def track(self, old_frame, new_frame, mesh_list, conf_threshold=0.25):
        """
        Track list of MeshData between frames.
        
        Args:
            old_frame: Previous frame (BGR)
            new_frame: Current frame (BGR)
            mesh_list: List[MeshData] from previous frame
            conf_threshold: Minimum combined confidence
            
        Returns:
            List[MeshData] with updated geometry
        """
        old_gray = self._to_gray(old_frame)
        new_gray = self._to_gray(new_frame)

        out = []
        for m in mesh_list:
            updated = self._track_meshdata(old_gray, new_gray, m)
            if updated is None:
                continue

            # Gating by combined confidence
            base_conf = m.confidence
            track_conf = updated.tracking_confidence

            if base_conf * track_conf < conf_threshold:
                continue

            # Propagate combined confidence
            updated.confidence = base_conf * track_conf
            out.append(updated)

        return out
