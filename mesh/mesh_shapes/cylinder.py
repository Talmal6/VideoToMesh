from dataclasses import dataclass
import numpy as np
from mesh.mesh_shapes.mesh_object import MeshObject


@dataclass
class Cylinder(MeshObject):
    radius: float = 1.0
    height: float = 1.0
    sides: int = 24
    capped: bool = True

    def build_mesh(self) -> None:
        sides = max(3, int(self.sides))
        r = float(self.radius)
        h2 = 0.5 * float(self.height)

        angles = np.linspace(0.0, 2.0 * np.pi, num=sides, endpoint=False, dtype=np.float32)
        xs = np.cos(angles) * r
        ys = np.sin(angles) * r

        bottom = np.stack([xs, ys, np.full_like(xs, -h2)], axis=1)
        top    = np.stack([xs, ys, np.full_like(xs, +h2)], axis=1)

        vertices = np.vstack([bottom, top]).astype(np.float32)
        faces = []

        # sides
        for i in range(sides):
            j = (i + 1) % sides
            b0, b1 = i, j
            t0, t1 = i + sides, j + sides
            faces.append((b0, t0, t1))
            faces.append((b0, t1, b1))

        # caps
        if self.capped:
            bc = len(vertices)
            tc = bc + 1
            vertices = np.vstack([
                vertices,
                np.array([[0.0, 0.0, -h2],
                          [0.0, 0.0, +h2]], dtype=np.float32)
            ])

            for i in range(sides):
                j = (i + 1) % sides
                faces.append((bc, j, i))                   # bottom
                faces.append((tc, i + sides, j + sides))   # top

        self.vertices = vertices
        self.faces = np.asarray(faces, dtype=np.int32)
