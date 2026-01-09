from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, List
import copy
import logging

import numpy as np
import cv2


from detection.detection import Detection
from mesh.mesh_shapes.mesh_object import MeshObject
from mesh.mesh_proccesors.mesh_handler import Handler
from mesh.mesh_shapes.mesh_object import MeshObject

logger = logging.getLogger(__name__)

@dataclass
class State:
    last_frame : Optional[np.ndarray] = None
    rotation_angle_y: float = 0.0
    rotation_angle_x: float = 0.0
    

# ------------------------------------------------------------
# Handler
# ------------------------------------------------------------
class CylinderHandler(Handler):
    def __init__(
        self,
        labels: Tuple[str, ...] = ("bottle", "cup", "can"),
        y_step: int = 3,
        sides: int = 32,
        min_row_width_px: int = 8,
        min_rings: int = 10,
        smooth_radius_alpha: float = 0.25,  
        smooth_center_alpha: float = 0.25,
        close_kernel: int = 3,
    ):
        self.labels = labels
        self.y_step = int(max(1, y_step))
        self.sides = int(max(8, sides))
        self.min_row_width_px = int(max(1, min_row_width_px))
        self.min_rings = int(max(2, min_rings))
        self.smooth_radius_alpha = float(np.clip(smooth_radius_alpha, 0.0, 1.0))
        self.smooth_center_alpha = float(np.clip(smooth_center_alpha, 0.0, 1.0))
        self.close_kernel = int(max(0, close_kernel))
        self.last_state = State()
        

    # --------------------------
    # Handler contract
    # --------------------------
    def can_handle(self, det: Detection) -> bool:
        return det.label in self.labels and det.mask is not None


   

    def process(self, obj: MeshObject, det: Detection, frame: np.ndarray) -> None:
        """
        build mesh from det.mask and det.bbox_xyxy.
        """
        # Resize mask if needed
        mask = det.mask
        h, w = frame.shape[:2]
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

        rows = self._split_mask_to_rows(mask, self.y_step)
        radiuses = [width_px / 2.0 for row in rows if (width_px := self._get_row_width_px(row)) >= self.min_row_width_px]
        smoothed_radiuses = self._smooth_values(radiuses, self.smooth_radius_alpha)
        discs = [self._make_disc(radius, self.sides, y) for y, radius in enumerate(smoothed_radiuses)]
        if len(discs) < self.min_rings:
            logger.warning("Not enough rings to build CylinderMesh. Aborting.")
            return
        vertices, faces = self._convert_discs_to_mesh(discs)

        # Apply rotation from previous object if available
        new_pose = None
        if obj and obj.pose:
            new_pose = copy.deepcopy(obj.pose)
            # Rotate around centroid
            center = vertices.mean(axis=0)
            vertices -= center
            
            # Apply X rotation
            rx = new_pose.rotation[0]
            if rx != 0:
                c, s = np.cos(rx), np.sin(rx)
                y = vertices[:, 1] * c - vertices[:, 2] * s
                z = vertices[:, 1] * s + vertices[:, 2] * c
                vertices[:, 1] = y
                vertices[:, 2] = z
                
            # Apply Y rotation
            ry = new_pose.rotation[1]
            if ry != 0:
                c, s = np.cos(ry), np.sin(ry)
                x = vertices[:, 0] * c + vertices[:, 2] * s
                z = -vertices[:, 0] * s + vertices[:, 2] * c
                vertices[:, 0] = x
                vertices[:, 2] = z
                
            # Apply Z rotation
            rz = new_pose.rotation[2]
            if rz != 0:
                c, s = np.cos(rz), np.sin(rz)
                x = vertices[:, 0] * c - vertices[:, 1] * s
                y = vertices[:, 0] * s + vertices[:, 1] * c
                vertices[:, 0] = x
                vertices[:, 1] = y
            
            vertices += center

        vertices = self._move_mesh_to_mask(vertices, mask)

        mesh_obj = MeshObject(
            object_id=det.object_id,
            label=det.label,
            confidence=det.confidence,
            frame_index=det.frame_index,
            bbox_xyxy=det.bbox_xyxy,
            mask=mask,
            vertices=vertices,
            faces=faces,
        )
        
        if new_pose:
            mesh_obj.pose = new_pose
            
        return mesh_obj
        

        
    def _split_mask_to_rows(self, mask: np.ndarray, y_step: int) -> List[np.ndarray]:
        """
        returns list of mask rows (each of height y_step)
        """

        rows = []
        h, w = mask.shape
        for y in range(0, h, y_step):
            row = mask[y : min(y + y_step, h), :]
            rows.append(row)
        return rows
    
    def _get_row_width_px(self, row: np.ndarray) -> int:
        """
        returns width in pixels of the mask row
        """
        cols = np.any(row > 0, axis=0)
        width_px = int(np.sum(cols))
        return width_px
    
    def _smooth_values(self, values: List[float], alpha: float) -> List[float]:
        """
        applies exponential moving average smoothing to the list of values
        """
        if alpha <= 0.0 or len(values) == 0:
            return values
        
        smoothed = [values[0]]
        for v in values[1:]:
            smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])
        return smoothed

    def _make_disc(self, radius: float, sides: int, y: float) -> np.ndarray:
        """
        builds a disc of given radius at height y with right height to be stacked into a mesh
        """
        angles = np.linspace(0.0, 2.0 * np.pi, num=sides, endpoint=False, dtype=np.float32)
        xs = np.cos(angles) * radius
        zs = np.sin(angles) * radius
        ys = np.full_like(xs, y * self.y_step)
        disc = np.stack([xs, ys, zs], axis=1)
        return disc

    def _convert_discs_to_mesh(self, discs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        builds vertices and faces from list of discs
        """
        if len(discs) < 2:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32)
        
        vertices = np.vstack(discs).astype(np.float32)
        faces = []

        sides = discs[0].shape[0]
        for i in range(len(discs) - 1):
            for j in range(sides):
                b0 = i * sides + j
                b1 = i * sides + (j + 1) % sides
                t0 = (i + 1) * sides + j
                t1 = (i + 1) * sides + (j + 1) % sides
                faces.append((b0, t0, t1))
                faces.append((b0, t1, b1))

        return vertices, np.asarray(faces, dtype=np.int32)
    

    
    def _move_mesh_to_mask(self, vertices: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        move the mesh to align with the mask so there will be maximal overlap
        """
        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            return vertices
        
        min_y = np.min(ys)
        min_x = np.min(xs)
        max_x = np.max(xs)
        center_x = (min_x + max_x) / 2.0

        # Shift vertices
        # vertices is (N, 3) -> (x, y, z)
        vertices[:, 0] += center_x
        vertices[:, 1] += min_y
        
        return vertices
    


        
    

    