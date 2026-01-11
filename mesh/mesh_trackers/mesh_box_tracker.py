from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from detection.detection import Detection
from mesh.mesh_shapes.mesh_object import MeshObject
from mesh.mesh_trackers.mesh_tracker import MeshTracker
from utils.depth_estimator import DepthEstimator

logger = logging.getLogger(__name__)


class MeshBoxTracker(MeshTracker):
    """
    Depth-aware tracker for box-like objects.
    Uses a heavy MiDaS model to derive per-pixel depth and orient the back face accordingly.
    """

    def __init__(self, labels: Tuple[str, ...] = ("box", "book", "tv", "monitor", "remote", "cell phone")):
        super().__init__()
        self.labels = labels

        # Perspective + geometry config
        self.scale_back_face = 0.85        # Shrink back quad for perspective
        self.expansion_factor = 1.10       # Expand front quad slightly for stability
        self.depth_multiplier = 5.0        # Sensitivity to depth gradient (pixels)
        self.smoothing_alpha = 0.5         # EMA for vertex updates
        self.min_size_px = 25

        # Heavy depth estimator (singleton)
        self.depth_estimator = DepthEstimator()

    def can_track(self, det: Detection) -> bool:
        return det.label in self.labels

    def track(self, det: Detection, curr_frame: np.ndarray) -> Optional[MeshObject]:
        if self.last_state.last_mesh is None:
            return None

        mesh = self.last_state.last_mesh
        h, w = curr_frame.shape[:2]

        # 1) Geometry from mask
        mask01 = self._prepare_mask(det.mask, (h, w))
        raw_front_poly = self._mask_to_rect_quad(mask01)
        if raw_front_poly is None:
            return None

        raw_front_poly = self._canonicalize_polygon(raw_front_poly)
        front_poly = self._apply_expansion(raw_front_poly, self.expansion_factor)

        # 2) Heavy depth inference
        depth_map = self.depth_estimator.get_depth_map(curr_frame)
        grad_vector = self.depth_estimator.get_local_gradient(depth_map, det.bbox_xyxy)

        # 3) Back face from depth gradient
        front_xy, back_xy = self._calculate_ai_perspective(front_poly, grad_vector)

        # 4) Update mesh vertices
        self._update_mesh_vertices(mesh, front_xy, back_xy)

        # 5) Update state/bbox
        self.last_state.last_det = det
        self.last_state.last_frame = curr_frame
        self.last_state.last_mesh = mesh
        mesh.bbox_xyxy = det.bbox_xyxy

        return mesh

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _calculate_ai_perspective(self, front_poly: np.ndarray, grad_vector: np.ndarray):
        """
        Use depth gradient to orient the back face.
        MiDaS: higher depth value = closer; we push the back face toward the far side.
        """
        mag = np.linalg.norm(grad_vector)
        if mag > 0:
            shift_vector = -grad_vector * self.depth_multiplier
        else:
            shift_vector = np.array([0.0, 30.0], dtype=np.float32)

        max_shift = 100.0
        norm = np.linalg.norm(shift_vector)
        if norm > max_shift and norm > 1e-6:
            shift_vector = (shift_vector / norm) * max_shift

        back_poly = front_poly.copy()
        back_poly += shift_vector

        back_center = back_poly.mean(axis=0)
        back_poly = back_center + (back_poly - back_center) * self.scale_back_face

        return front_poly, back_poly

    def _update_mesh_vertices(self, mesh: MeshObject, front_xy: np.ndarray, back_xy: np.ndarray):
        n = 4
        depth_z = 50.0  # Visual depth scalar

        front_v = np.hstack([front_xy, np.zeros((n, 1), dtype=np.float32)])
        back_v = np.hstack([back_xy, np.full((n, 1), depth_z, dtype=np.float32)])
        new_vertices = np.vstack([front_v, back_v]).astype(np.float32)

        if mesh.vertices is not None and mesh.vertices.shape == new_vertices.shape:
            alpha = self.smoothing_alpha
            mesh.vertices = alpha * new_vertices + (1 - alpha) * mesh.vertices
        else:
            mesh.vertices = new_vertices

    # ------------------------------------------------------------------
    # Mask/quad helpers
    # ------------------------------------------------------------------

    def _apply_expansion(self, poly: np.ndarray, factor: float) -> np.ndarray:
        if factor == 1.0:
            return poly
        center = poly.mean(axis=0)
        return center + (poly - center) * factor

    def _prepare_mask(self, mask: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        h, w = shape
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        return (mask > 0).astype(np.uint8)

    def _mask_to_rect_quad(self, mask01: np.ndarray) -> Optional[np.ndarray]:
        contours, _ = cv2.findContours(mask01, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        valid = [c for c in contours if cv2.contourArea(c) > self.min_size_px**2]
        if not valid:
            return None
        largest = max(valid, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest)
        return np.asarray(cv2.boxPoints(rect), dtype=np.float32)

    def _canonicalize_polygon(self, poly: np.ndarray) -> np.ndarray:
        c = poly.mean(axis=0)
        angles = np.arctan2(poly[:, 1] - c[1], poly[:, 0] - c[0])
        order = np.argsort(angles)
        poly = poly[order]
        tl_idx = np.argmin(poly.sum(axis=1))
        return np.roll(poly, -tl_idx, axis=0)
