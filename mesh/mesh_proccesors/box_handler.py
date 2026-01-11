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
    last_front_xy: Optional[np.ndarray] = None   # (N,2)
    last_depth: Optional[float] = None


class MeshBoxHandler(Handler):
    """
    Extract a polygon from det.mask and build a 3D mesh by extrusion.
    Works on CPU and matches the mask much better than quad/rect hacks.
    """

    def __init__(
        self,
        labels: Tuple[str, ...] = ("box", "book", "tv", "monitor", "remote", "cell phone"),
        min_area_px2: int = 25 * 25,
        # Polygon control:
        max_vertices: int = 48,          # cap complexity for realtime
        rdp_epsilon_ratio: float = 0.01,  # relative to contour perimeter
        # Depth control:
        depth_ratio: float = 0.18,       # thickness relative to sqrt(area) or min bbox side
        depth_min_px: float = 8.0,
        depth_max_px: float = 120.0,
        # Mask cleaning:
        morph_kernel: int = 5,
        # Smoothing:
        smooth_alpha: float = 0.65,      # higher = faster response, lower = smoother
    ):
        self.labels = labels
        self.min_area_px2 = int(min_area_px2)
        self.max_vertices = int(max_vertices)
        self.rdp_epsilon_ratio = float(rdp_epsilon_ratio)
        self.depth_ratio = float(depth_ratio)
        self.depth_min_px = float(depth_min_px)
        self.depth_max_px = float(depth_max_px)
        self.morph_kernel = int(morph_kernel)
        self.smooth_alpha = float(smooth_alpha)

        self.state = State()

    def can_handle(self, det: Detection) -> bool:
        return det.label in self.labels and det.mask is not None

    def process(self, obj: MeshObject, det: Detection, frame: np.ndarray) -> Optional[MeshObject]:
        mesh = self._generate_mesh(det, frame)
        if mesh is None:
            return None

        mesh.object_id = det.object_id
        mesh.label = det.label
        mesh.confidence = det.confidence
        mesh.frame_index = det.frame_index
        mesh.bbox_xyxy = det.bbox_xyxy
        mesh.mask = det.mask
        return mesh

    # ------------------------ Core pipeline ------------------------

    def _generate_mesh(self, det: Detection, frame: np.ndarray) -> Optional[MeshObject]:
        h, w = frame.shape[:2]
        mask01 = self._prepare_mask(det.mask, (h, w))
        poly = self._mask_to_polygon(mask01)
        if poly is None:
            return None

        # Ensure CCW (for triangulation)
        poly = self._ensure_ccw(poly)

        # Depth from object scale (no MiDaS)
        depth_z = self._estimate_depth_z(poly, det.bbox_xyxy)

        # Smooth (optional)
        poly, depth_z = self._smooth(poly, depth_z)

        # Triangulate front face
        tris = self._ear_clip_triangulate(poly)
        if tris is None or len(tris) == 0:
            return None

        return self._build_extruded_mesh(poly, tris, depth_z, det.bbox_xyxy)

    # ------------------------ Mask utils ------------------------

    def _prepare_mask(self, mask: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        h, w = shape
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.uint8)

        k = max(3, self.morph_kernel)
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        # Close holes + remove small noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    def _mask_to_polygon(self, mask01: np.ndarray) -> Optional[np.ndarray]:
        contours, _ = cv2.findContours(mask01, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None

        # pick largest area contour
        c = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(c))
        if area < float(self.min_area_px2):
            return None

        peri = float(cv2.arcLength(c, True))
        eps = max(1.0, self.rdp_epsilon_ratio * peri)

        approx = cv2.approxPolyDP(c, eps, True)  # (M,1,2)
        poly = approx.reshape(-1, 2).astype(np.float32)

        # If too many vertices, simplify further (increase epsilon gradually)
        if poly.shape[0] > self.max_vertices:
            scale = 1.0
            for _ in range(8):
                scale *= 1.35
                approx2 = cv2.approxPolyDP(c, eps * scale, True)
                poly2 = approx2.reshape(-1, 2).astype(np.float32)
                if 3 <= poly2.shape[0] <= self.max_vertices:
                    poly = poly2
                    break

        # Must be at least a triangle
        if poly.shape[0] < 3:
            return None

        # Avoid self-intersections risk by falling back to convex hull if needed
        # (safer for ear clipping; you can disable if you want concave objects)
        if not self._is_simple_polygon(poly):
            hull = cv2.convexHull(poly.astype(np.float32))
            poly = hull.reshape(-1, 2).astype(np.float32)

        # Cap again
        if poly.shape[0] > self.max_vertices:
            poly = poly[: self.max_vertices].copy()

        return poly

    # ------------------------ Depth estimate ------------------------

    def _estimate_depth_z(self, poly_xy: np.ndarray, bbox_xyxy: np.ndarray) -> float:
        # Robust object scale: sqrt(area of polygon)
        area = abs(self._polygon_area(poly_xy))
        scale = float(np.sqrt(max(area, 1.0)))

        depth = scale * self.depth_ratio
        depth = float(np.clip(depth, self.depth_min_px, self.depth_max_px))
        return depth

    # ------------------------ Smoothing ------------------------

    def _smooth(self, poly_xy: np.ndarray, depth_z: float) -> Tuple[np.ndarray, float]:
        if self.state.last_front_xy is None:
            self.state.last_front_xy = poly_xy
            self.state.last_depth = depth_z
            return poly_xy, depth_z

        # If vertex count changes (common after RDP), smoothing becomes invalid.
        # In that case, reset smoothing to current.
        if self.state.last_front_xy.shape[0] != poly_xy.shape[0]:
            self.state.last_front_xy = poly_xy
            self.state.last_depth = depth_z
            return poly_xy, depth_z

        a = self.smooth_alpha
        sm_xy = a * poly_xy + (1.0 - a) * self.state.last_front_xy
        sm_d = a * depth_z + (1.0 - a) * float(self.state.last_depth)

        self.state.last_front_xy = sm_xy
        self.state.last_depth = float(sm_d)
        return sm_xy.astype(np.float32), float(sm_d)

    # ------------------------ Triangulation (Ear clipping) ------------------------

    def _ear_clip_triangulate(self, poly: np.ndarray) -> Optional[np.ndarray]:
        """
        Ear clipping triangulation for a simple polygon in CCW order.
        Returns triangles as (T,3) indices into poly.
        """
        n = poly.shape[0]
        if n < 3:
            return None
        if n == 3:
            return np.array([[0, 1, 2]], dtype=np.int32)

        idx = list(range(n))
        tris: List[List[int]] = []

        # Precompute
        def is_convex(a, b, c) -> bool:
            return self._cross(poly[b] - poly[a], poly[c] - poly[b]) > 1e-6  # CCW convex

        def point_in_tri(p, a, b, c) -> bool:
            # barycentric sign method
            v0 = c - a
            v1 = b - a
            v2 = p - a
            den = self._cross(v1, v0)
            if abs(den) < 1e-9:
                return False
            u = self._cross(v2, v0) / den
            v = self._cross(v1, v2) / den
            return (u >= -1e-6) and (v >= -1e-6) and (u + v <= 1.0 + 1e-6)

        guard = 0
        while len(idx) > 3 and guard < 5000:
            guard += 1
            ear_found = False

            m = len(idx)
            for i in range(m):
                i0 = idx[(i - 1) % m]
                i1 = idx[i]
                i2 = idx[(i + 1) % m]

                if not is_convex(i0, i1, i2):
                    continue

                a, b, c = poly[i0], poly[i1], poly[i2]

                # check no other point inside triangle
                any_inside = False
                for j in range(m):
                    ij = idx[j]
                    if ij in (i0, i1, i2):
                        continue
                    if point_in_tri(poly[ij], a, b, c):
                        any_inside = True
                        break
                if any_inside:
                    continue

                # it's an ear
                tris.append([i0, i1, i2])
                del idx[i]
                ear_found = True
                break

            if not ear_found:
                # fallback: convex hull triangulation to avoid total failure
                hull = cv2.convexHull(poly.astype(np.float32), returnPoints=False).reshape(-1).tolist()
                if len(hull) >= 3:
                    base = hull[0]
                    fallback = []
                    for k in range(1, len(hull) - 1):
                        fallback.append([base, hull[k], hull[k + 1]])
                    return np.array(fallback, dtype=np.int32)
                return None

        if len(idx) == 3:
            tris.append([idx[0], idx[1], idx[2]])

        return np.array(tris, dtype=np.int32)

    # ------------------------ Build extruded mesh ------------------------

    def _build_extruded_mesh(self, front_xy: np.ndarray, front_tris: np.ndarray, depth_z: float, bbox_xyxy):
        n = front_xy.shape[0]

        front_v = np.hstack([front_xy, np.zeros((n, 1), dtype=np.float32)])
        back_xy = front_xy.copy()
        back_v = np.hstack([back_xy, np.full((n, 1), depth_z, dtype=np.float32)])
        vertices = np.vstack([front_v, back_v]).astype(np.float32)

        # Front faces as-is (CCW)
        front_faces = front_tris.astype(np.int32)

        # Back faces reversed winding (to keep normals consistent)
        back_faces = (front_tris[:, ::-1] + n).astype(np.int32)

        # Side faces: connect each edge i->j with i+n, j+n
        side_faces = []
        for i in range(n):
            j = (i + 1) % n
            side_faces.append([i, j, j + n])
            side_faces.append([i, j + n, i + n])

        faces = np.vstack([
            front_faces,
            back_faces,
            np.asarray(side_faces, dtype=np.int32)
        ]).astype(np.int32)

        return MeshObject(
            object_id=0, label="", confidence=0.0, frame_index=0,
            bbox_xyxy=bbox_xyxy,
            vertices=vertices,
            faces=faces
        )

    # ------------------------ Geometry helpers ------------------------

    @staticmethod
    def _cross(a: np.ndarray, b: np.ndarray) -> float:
        return float(a[0] * b[1] - a[1] * b[0])

    def _polygon_area(self, poly: np.ndarray) -> float:
        x = poly[:, 0]
        y = poly[:, 1]
        return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    def _ensure_ccw(self, poly: np.ndarray) -> np.ndarray:
        if self._polygon_area(poly) < 0:
            return poly[::-1].copy()
        return poly

    def _is_simple_polygon(self, poly: np.ndarray) -> bool:
        # Lightweight sanity: reject obviously degenerate (repeated points / too small edges)
        if poly.shape[0] < 3:
            return False
        d = np.linalg.norm(poly - np.roll(poly, -1, axis=0), axis=1)
        if np.any(d < 1e-3):
            return False
        return True
