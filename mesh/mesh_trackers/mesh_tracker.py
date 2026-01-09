# mesh/mesh_trackers/mesh_tracker.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

from detection.detection import Detection
from mesh.mesh_shapes.mesh_object import MeshObject


@dataclass
class State:
    last_frame: Optional[np.ndarray] = None
    last_mesh: Optional[MeshObject] = None
    last_det: Optional[Detection] = None


class MeshTracker(ABC):
    def __init__(self):
        self.last_state = State()

    @abstractmethod
    def track(self, det: Detection, curr_frame: np.ndarray) -> Optional[MeshObject]:
        raise NotImplementedError

    def translate_x(self, mesh: MeshObject, dx: float) -> None:
        """Moves the mesh along the X axis."""
        if mesh.vertices is not None:
            mesh.vertices[:, 0] += dx

    def translate_y(self, mesh: MeshObject, dy: float) -> None:
        """Moves the mesh along the Y axis."""
        if mesh.vertices is not None:
            mesh.vertices[:, 1] += dy

    def rotate_x(self, mesh: MeshObject, angle: float) -> None:
        """Rotates the mesh around the X axis (around its centroid). like a nodding motion."""
        if mesh.vertices is not None and len(mesh.vertices) > 0:
            center = mesh.vertices.mean(axis=0)
            mesh.vertices -= center
            c, s = np.cos(angle), np.sin(angle)
            y = mesh.vertices[:, 1] * c - mesh.vertices[:, 2] * s
            z = mesh.vertices[:, 1] * s + mesh.vertices[:, 2] * c
            mesh.vertices[:, 1] = y
            mesh.vertices[:, 2] = z
            mesh.vertices += center
            
            # Update pose rotation X
            mesh.pose.rotation[0] += angle

    def rotate_y(self, mesh: MeshObject, angle: float) -> None:
        """Rotates the mesh around the Y axis (around its centroid). Like a vertical spinning top."""
        if mesh.vertices is not None and len(mesh.vertices) > 0:
            center = mesh.vertices.mean(axis=0)
            mesh.vertices -= center
            c, s = np.cos(angle), np.sin(angle)
            x = mesh.vertices[:, 0] * c + mesh.vertices[:, 2] * s
            z = -mesh.vertices[:, 0] * s + mesh.vertices[:, 2] * c
            mesh.vertices[:, 0] = x
            mesh.vertices[:, 2] = z
            mesh.vertices += center

            # Update pose rotation Y
            mesh.pose.rotation[1] += angle

    def rotate_z(self, mesh: MeshObject, angle: float) -> None:
        """Rotates the mesh around the Z axis (around its centroid). Like a screwdriver."""
        if mesh.vertices is not None and len(mesh.vertices) > 0:
            center = mesh.vertices.mean(axis=0)
            mesh.vertices -= center
            c, s = np.cos(angle), np.sin(angle)
            x = mesh.vertices[:, 0] * c - mesh.vertices[:, 1] * s
            y = mesh.vertices[:, 0] * s + mesh.vertices[:, 1] * c
            mesh.vertices[:, 0] = x
            mesh.vertices[:, 1] = y
            mesh.vertices += center
            
            # Update pose rotation Z
            mesh.pose.rotation[2] += angle


    def enlarge(self, mesh: MeshObject, factor: float) -> None:
        """Enlarges the mesh by the given factor (around its centroid)."""
        if mesh.vertices is not None and len(mesh.vertices) > 0:
            center = mesh.vertices.mean(axis=0)
            mesh.vertices[:] = (mesh.vertices - center) * factor + center

    def shrink(self, mesh: MeshObject, factor: float) -> None:
        """Shrinks the mesh by the given factor (around its centroid)."""
        if mesh.vertices is not None and len(mesh.vertices) > 0 and factor != 0:
            center = mesh.vertices.mean(axis=0)
            mesh.vertices[:] = (mesh.vertices - center) / factor + center