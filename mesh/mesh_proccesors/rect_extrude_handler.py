from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional
import logging

import cv2
import numpy as np

from detection.detection import Detection
from mesh.mesh_shapes.mesh_object import MeshObject
from mesh.mesh_proccesors.mesh_handler import Handler

logger = logging.getLogger(__name__)


@dataclass
class State:
    last_front: Optional[np.ndarray] = None
    last_back: Optional[np.ndarray] = None
    last_depth: Optional[float] = None


class RectExtrudeHandler(Handler):
    """
    Force a rectangular box from the mask using minAreaRect, then extrude in-image
    (XY shift) to build a stable box. Z is only used to close volume.
    """

    def __init__(
        self,
        labels: Tuple[str, ...] = ("box", "book", "tv", "monitor", "remote", "cell phone"),
        smooth_alpha: float = 0.55,
        depth_scale_xy: float = 0.35,   # how far to shift back face in XY (fraction of shorter edge)
        depth_scale_z: float = 0.15,    # Z thickness as fraction of shorter edge
        min_area_px2: float = 20 * 20,
    ):
        self.labels = labels
        self.smooth_alpha = float(np.clip(smooth_alpha, 0.0, 1.0))
        self.depth_scale_xy = float(depth_scale_xy)
        self.depth_scale_z = float(depth_scale_z)
        self.min_area_px2 = float(min_area_px2)
        self.state = State()

    def can_handle(self, det: Detection) -> bool:
        return det.label in self.labels and det.mask is not None

    def process(self, obj: MeshObject, det: Detection, frame: np.ndarray) -> Optional[MeshObject]:
        mesh = self._build_mesh(det)
        if mesh is None:
            return None

        mesh.object_id = det.object_id
        mesh.label = det.label
        mesh.confidence = det.confidence
        mesh.frame_index = det.frame_index
        mesh.bbox_xyxy = det.bbox_xyxy
        mesh.mask = det.mask
        return mesh

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------
    def _build_mesh(self, det: Detection) -> Optional[MeshObject]:
        mask = det.mask
        if mask is None:
            return None
        mask01 = (mask > 0).astype(np.uint8)

        box = self._rect_from_mask(mask01)
        if box is None:
            return None

        front = self._canonical_rect(box)
        depth_dir, edge_scale = self._depth_direction_and_scale(front)

        shift_xy = depth_dir * (self.depth_scale_xy * edge_scale)
        back = front + shift_xy
        depth_z = self.depth_scale_z * edge_scale

        # Smooth if previous exists and same vertex count (always 4)
        if self.state.last_front is not None:
            a = self.smooth_alpha
            front = a * front + (1.0 - a) * self.state.last_front
            back = a * back + (1.0 - a) * self.state.last_back
            depth_z = a * depth_z + (1.0 - a) * (self.state.last_depth if self.state.last_depth else depth_z)

        self.state.last_front = front
        self.state.last_back = back
        self.state.last_depth = depth_z

        return self._assemble_mesh(front, back, depth_z, det.bbox_xyxy)

    def _rect_from_mask(self, mask01: np.ndarray) -> Optional[np.ndarray]:
        contours, _ = cv2.findContours(mask01, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < self.min_area_px2:
            return None
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        return np.asarray(box, dtype=np.float32)

    def _canonical_rect(self, box: np.ndarray) -> np.ndarray:
        c = box.mean(axis=0)
        ang = np.arctan2(box[:, 1] - c[1], box[:, 0] - c[0])
        box = box[np.argsort(ang)]
        tl = np.argmin(box.sum(axis=1))
        return np.roll(box, -tl, axis=0).astype(np.float32)

    def _depth_direction_and_scale(self, rect: np.ndarray):
        edge_top = rect[1] - rect[0]
        edge_side = rect[3] - rect[0]
        len_top = np.linalg.norm(edge_top)
        len_side = np.linalg.norm(edge_side)
        if len_top < len_side:
            depth_dir = edge_top
            scale = len_top
        else:
            depth_dir = edge_side
            scale = len_side
        n = np.linalg.norm(depth_dir)
        if n < 1e-6:
            depth_dir = np.array([0.0, -1.0], dtype=np.float32)
        else:
            depth_dir = depth_dir / n
        # force downward/in-image direction
        if depth_dir[1] < 0:
            depth_dir = -depth_dir
        return depth_dir.astype(np.float32), float(scale)

    def _assemble_mesh(self, front: np.ndarray, back: np.ndarray, depth_z: float, bbox_xyxy) -> MeshObject:
        # front/back are quads, keep order TL, TR, BR, BL
        front_v = np.hstack([front, np.zeros((4, 1), dtype=np.float32)])
        back_v = np.hstack([back, np.full((4, 1), depth_z, dtype=np.float32)])
        vertices = np.vstack([front_v, back_v]).astype(np.float32)

        front_faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        back_faces = np.array([[4, 6, 5], [4, 7, 6]], dtype=np.int32)

        side_faces = [
            [0, 1, 5], [0, 5, 4],
            [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6],
            [3, 0, 4], [3, 4, 7],
        ]
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
