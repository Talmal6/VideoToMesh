from typing import List, Optional
import numpy as np

from class_proccesors.detection import Detection
from mesh.mesh_shapes.mesh_object import MeshObject
from mesh.mesh_proccesors.mesh_handler import Handler
from mesh.mesh_trackers.mesh_cylinder_tracker import MeshCylinderTracker
from mesh.mesh_trackers.mesh_tracker import MeshTracker, State

class MeshManager:
    def __init__(self, handlers: List[Handler]):
        if not handlers:
            raise ValueError("MeshManager requires at least one handler.")
        self.handlers = handlers
        self.trackers =  [MeshCylinderTracker()]

    def _pick_handler(self, det: Detection) -> Handler:
        for h in self.handlers:
            if h.can_handle(det):
                return h
        raise ValueError(f"No handler found for label='{det.label}'")
    
    def _pick_tracker(self, det: Detection) -> Optional[MeshTracker]:
        for t in self.trackers:
            if t.can_track(det):
                return t
        return None
        

    def get_mesh(self, det: Detection, frame: np.ndarray) -> Optional[MeshObject]:
        # 1. Pick Handler
        try:
            handler = self._pick_handler(det)
        except ValueError:
            return None

        tracker = self._pick_tracker(det)
        
        mesh = None
        
        # 3. Try Tracking
        if tracker:
            mesh = tracker.track(det, frame)
            
        # 4. If Tracking failed (or no tracker), Create New
        if mesh is None:
            print("Creating new mesh using Handler.")
            # Create new mesh using Handler
            mesh = handler.process(None, det, frame.shape[:2])
            
            # Initialize tracker state with the new mesh
            if tracker and mesh:
                tracker.last_state = State(
                    last_frame=frame,
                    last_det=det,
                    last_mesh=mesh
                )
        else:
            print("Mesh tracked successfully.")
        
        return mesh