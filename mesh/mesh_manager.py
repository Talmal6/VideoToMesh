from typing import List, Optional
import logging
import numpy as np

from detection.detection import Detection
from mesh.mesh_shapes.mesh_object import MeshObject
from mesh.mesh_proccesors.mesh_handler import Handler

logger = logging.getLogger(__name__)

class MeshManager:
    def __init__(self, handlers: List[Handler]):
        if not handlers:
            raise ValueError("MeshManager requires at least one handler.")
        self.handlers = handlers

    def _pick_handler(self, det: Detection) -> Handler:
        for h in self.handlers:
            if h.can_handle(det):
                return h
        logger.warning(f"No handler found for label='{det.label}'")
        raise ValueError(f"No handler found for label='{det.label}'")
        
    def get_mesh(self, det: Detection, frame: np.ndarray) -> Optional[MeshObject]:
        # 1. Select Handler (Fallback/Creator)
        try:
            handler = self._pick_handler(det)
        except ValueError:
            return None

        return handler.process(None, det, frame)
