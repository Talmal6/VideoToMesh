from typing import List, Optional
import numpy as np
import logging

from detection.detection import Detection
from mesh.mesh_shapes.mesh_object import MeshObject
from mesh.mesh_proccesors.mesh_handler import Handler
from mesh.mesh_trackers.mesh_cylinder_tracker import MeshCylinderTracker
from mesh.mesh_trackers.mesh_tracker import MeshTracker, State

logger = logging.getLogger(__name__)

class MeshManager:
    def __init__(self, handlers: List[Handler]):
        if not handlers:
            raise ValueError("MeshManager requires at least one handler.")
        self.handlers = handlers
        self.trackers = [MeshCylinderTracker()]

    def _pick_handler(self, det: Detection) -> Handler:
        for h in self.handlers:
            if h.can_handle(det):
                return h
        logger.warning(f"No handler found for label='{det.label}'")
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

        tracker = None #self._pick_tracker(det)
        
        mesh = None
        
        # 3. Try Tracking, if tracking failed then go to processing
        if tracker:
            mesh = tracker.track(det, frame)
        
            
        # 4. If Tracking failed (or no tracker), Create New
        if mesh is None:
            logger.info("Cannot track mesh, creating new mesh using Handler.")
            # Create new mesh using Handler
            last_mesh = tracker.last_state.last_mesh if tracker and tracker.last_state else None
            mesh = handler.process(last_mesh, det, frame)
            
            # Initialize tracker state with the new mesh
            if tracker and mesh:
                tracker.last_state = State(
                    last_frame=frame,
                    last_det=det,
                    last_mesh=mesh
                )
        else:
            logger.info(f"Mesh tracked successfully for {det.label}")
            
        return mesh
