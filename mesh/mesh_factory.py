from mesh.mesh_proccesors.mesh_handler import Handler
from typing import List, Tuple
from class_proccesors.detection import Detection
from mesh.mesh_shapes.mesh_object import MeshObject


class MeshFactory:
    def __init__(self, handlers: List[Handler]):
        if not handlers:
            raise ValueError("MeshFactory requires at least one handler.")
        self.handlers = handlers

    def _pick(self, det: Detection) -> Handler:
        for h in self.handlers:
            if h.can_handle(det):
                return h
        raise ValueError(f"No handler found for label='{det.label}'")

    def create(self, det: Detection, curr_frame, last_frame, last_mesh: MeshObject) -> MeshObject:
        h : Handler = self._pick(det)
        obj = h.create_object(det)
        h.process(obj, det, curr_frame, last_frame, last_mesh)
        
            
        return obj