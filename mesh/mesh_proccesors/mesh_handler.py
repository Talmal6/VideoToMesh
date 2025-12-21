from typing import Tuple
from class_proccesors.detection import Detection
from mesh.mesh_shapes.mesh_object import MeshObject



class Handler:
    """
    Contract:
    - can_handle(det) selects handler
    - create_object(det) creates the concrete MeshObject
    - process(obj, det, frame_shape_hw) performs: mask/bbox -> params + build_mesh
    """
    def can_handle(self, det: Detection) -> bool:
        raise NotImplementedError

    def create_object(self, det: Detection) -> MeshObject:
        raise NotImplementedError

    def process(self, obj: MeshObject, det: Detection, frame_shape_hw: Tuple[int, int]) -> None:
        raise NotImplementedError
