import numpy as np
from typing import Dict, List, Tuple
from class_proccesors.detection import Detection
from mesh.mesh_factory import MeshFactory
from mesh.mesh_shapes.mesh_object import MeshObject

class MeshTracker:
    def __init__(self, factory: MeshFactory, max_missed_frames: int = 30, iou_threshold: float = 0.35):
        self.factory = factory
        self.max_missed_frames = max_missed_frames
        self.iou_threshold = iou_threshold
        
        
        self.tracked_objects: Dict[int, MeshObject] = {}
        self.missed_counts: Dict[int, int] = {}
        self.next_object_id = 0 

    def update(self, detection: Detection, frame) -> List[MeshObject]:
        return None # for future implementation