from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, List
import logging

import cv2
import numpy as np

from detection.detection import Detection
from mesh.mesh_shapes.mesh_object import MeshObject
from mesh.mesh_proccesors.mesh_handler import Handler

logger = logging.getLogger(__name__)

@dataclass
class State:
    last_frame: Optional[np.ndarray] = None


class BoxHandler(Handler):
    def __init__(
        self,
        labels: Tuple[str, ...] = ("box", "book", "tv", "monitor", "remote","phone"),
        depth_ratio: float = 0.35,
        min_size_px: int = 8,
        # silhouette fidelity
        approx_eps_frac: float = 0.01,     # smaller => more points
        use_convex_hull: bool = True,      # True = stable + fast, but loses concavities
        max_poly_points: int = 64,         # cap complexity
    ):
        self.labels = labels
        self.depth_ratio = float(max(0.05, depth_ratio))
        self.min_size_px = int(max(1, min_size_px))
        self.approx_eps_frac = float(np.clip(approx_eps_frac, 0.001, 0.05))
        self.use_convex_hull = bool(use_convex_hull)
        self.max_poly_points = int(max(8, max_poly_points))
        self.last_state = State()

    def can_handle(self, det: Detection) -> bool:
        return det.label in self.labels and det.mask is not None

    def process(self, obj: MeshObject, det: Detection, frame: np.ndarray) -> Optional[MeshObject]:
        mesh = self._convert_to_mesh(det, frame)
        if mesh is None:
            logger.warning(f"Failed to create box mesh for {det.label}")
            return None

        mesh.object_id = det.object_id
        mesh.label = det.label
        mesh.confidence = det.confidence
        mesh.frame_index = det.frame_index
        mesh.bbox_xyxy = det.bbox_xyxy
        mesh.mask = det.mask
        return mesh

    # ---------------------------------------------------------------------
    # Mask helpers
    # ---------------------------------------------------------------------

    def _ensure_mask_size(self, mask: np.ndarray, frame_shape: Tuple[int, int]) -> np.ndarray:
        h, w = frame_shape
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        return (mask > 0).astype(np.uint8)

    def _largest_contour(self, mask01: np.ndarray) -> Optional[np.ndarray]:
        contours, _ = cv2.findContours(mask01.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contours = [c for c in contours if cv2.contourArea(c) >= float(self.min_size_px * self.min_size_px)]
        if not contours:
            return None
        return max(contours, key=cv2.contourArea)

    def _robust_width_from_mask(self, mask01: np.ndarray, step: int = 4) -> Optional[float]:
        ys, xs = np.where(mask01 > 0)
        if ys.size == 0:
            return None
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        h = y2 - y1
        if h <= 0:
            return None
        ys0 = y1 + int(0.2 * h)
        ys1 = y1 + int(0.8 * h)

        widths = []
        for y in range(ys0, ys1, max(1, step)):
            row = np.where(mask01[y] > 0)[0]
            if row.size:
                widths.append(int(row.max() - row.min() + 1))
        if len(widths) < 5:
            return float(xs.max() - xs.min() + 1)
        return float(np.median(np.asarray(widths, dtype=np.float32)))

    # ---------------------------------------------------------------------
    # Polygon extraction from mask
    # ---------------------------------------------------------------------

    def _mask_polygon(self, mask01: np.ndarray) -> Optional[np.ndarray]:
        cnt = self._largest_contour(mask01)
        if cnt is None:
            logger.info("No valid contour found in mask.")
            return None

        peri = cv2.arcLength(cnt, True)
        eps = self.approx_eps_frac * peri
        poly = cv2.approxPolyDP(cnt, eps, True)  # (N,1,2)

        if poly is None or len(poly) < 3:
            return None

        poly = poly.reshape(-1, 2).astype(np.float32)

        if self.use_convex_hull:
            hull = cv2.convexHull(poly.astype(np.float32))
            poly = hull.reshape(-1, 2).astype(np.float32)

        # cap number of points (uniform downsample)
        if len(poly) > self.max_poly_points:
            idxs = np.linspace(0, len(poly) - 1, self.max_poly_points).astype(int)
            poly = poly[idxs]

        # ensure CCW order (nice for side faces)
        if self._polygon_area(poly) < 0:
            poly = poly[::-1].copy()

        return poly

    def _polygon_area(self, poly: np.ndarray) -> float:
        x = poly[:, 0]
        y = poly[:, 1]
        return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))

    # ---------------------------------------------------------------------
    # Triangulation (fast path for convex polygon)
    # ---------------------------------------------------------------------

    def _triangulate_convex_fan(self, n: int) -> np.ndarray:
        """
        For convex polygon vertices [0..n-1], triangulate as fan from vertex 0:
        (0,i,i+1) for i=1..n-2
        """
        faces = []
        for i in range(1, n - 1):
            faces.append([0, i, i + 1])
        return np.asarray(faces, dtype=np.int32)

    # ---------------------------------------------------------------------
    # Build extruded prism mesh
    # ---------------------------------------------------------------------

    def _build_prism_mesh(
        self,
        poly2d: np.ndarray,
        depth_px: float,
        bbox_xyxy: Tuple[float, float, float, float],
    ) -> Optional[MeshObject]:
        n = len(poly2d)
        if n < 3:
            return None

        # vertices: front (z=0) + back (z=depth)
        front = np.hstack([poly2d, np.zeros((n, 1), dtype=np.float32)])
        back = front.copy()
        back[:, 2] += float(depth_px)

        vertices = np.vstack([front, back]).astype(np.float32)

        # Triangulate front/back (convex fan)
        front_faces = self._triangulate_convex_fan(n)  # indices in [0..n-1]
        back_faces = front_faces.copy()
        # reverse winding for back
        back_faces = back_faces[:, [0, 2, 1]] + n

        # side faces: for each edge i->j, build quad as two triangles:
        # front i, front j, back j, back i
        side_faces = []
        for i in range(n):
            j = (i + 1) % n
            fi, fj = i, j
            bi, bj = i + n, j + n
            side_faces.append([fi, fj, bj])
            side_faces.append([fi, bj, bi])

        faces = np.vstack([front_faces, back_faces, np.asarray(side_faces, dtype=np.int32)])

        return MeshObject(
            object_id=0,
            label="",
            confidence=0.0,
            frame_index=0,
            bbox_xyxy=bbox_xyxy,
            vertices=vertices,
            faces=faces,
        )

    # ---------------------------------------------------------------------
    # Main conversion
    # ---------------------------------------------------------------------

    def _convert_to_mesh(self, det: Detection, frame: np.ndarray) -> Optional[MeshObject]:
        frame_shape = frame.shape[:2]
        mask01 = self._ensure_mask_size(det.mask, frame_shape)

        poly = self._mask_polygon(mask01)
        if poly is None:
            return None

        # depth from robust width average (median widths)
        w_med = self._robust_width_from_mask(mask01, step=4)
        if w_med is None:
            return None

        depth_px = float(self.depth_ratio) * float(w_med)
        depth_px = max(1.0, depth_px)

        return self._build_prism_mesh(poly, depth_px, det.bbox_xyxy)
