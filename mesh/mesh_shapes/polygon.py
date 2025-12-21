# ----------------------------
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from mesh_shapes.mesh_object import MeshObject


@dataclass
class Polygon(MeshObject):
    """
    Extruded polygon mesh:
    - points2d: Nx2 vertices in local XY plane (must be ordered, non-self-intersecting)
    - height: extrusion along Z
    """
    points2d: Optional[np.ndarray] = None   # shape (N,2) float32
    height: float = 1.0
    capped: bool = True
