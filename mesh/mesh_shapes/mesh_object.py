from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple
import numpy as np
import copy

@dataclass
class Pose:
    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    rotation: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    scale: np.ndarray    = field(default_factory=lambda: np.ones(3, dtype=np.float32))

@dataclass
class MeshObject:
    object_id: int
    label: str
    confidence: float
    frame_index: int
    bbox_xyxy: Tuple[float, float, float, float]
    mask: Optional[np.ndarray] = None
    color_bgr: Tuple[int, int, int] = (0, 255, 0)

    pose: "Pose" = field(default_factory=lambda: Pose())
    cache: Dict[str, Any] = field(default_factory=dict)

    # --- Mesh State ---
    vertices: Optional[np.ndarray] = None   
    faces: Optional[np.ndarray] = None
    
    # --- Optimization State ---
    vertices_base: Optional[np.ndarray] = None 
    
    base_height: float = 1.0
    base_width: float = 1.0

    def apply_rotation(self, rotation_angle: float) -> None:
        """ מעדכן את הסיבוב של האובייקט סביב ציר ה-Y """
        self.pose.rotation[1] += rotation_angle

    def copy(self) -> "MeshObject":
        return copy.deepcopy(self)