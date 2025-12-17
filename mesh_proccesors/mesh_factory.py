import numpy as np
from dataclasses import dataclass


@dataclass
class Mesh:
    vertices: np.ndarray  # (N,3) float32
    faces: np.ndarray     # (M,3) int32


class MeshFactory:
    """
    Geometry-only factory.
    Creates canonical meshes centered at origin, aligned to axes.
    NO pose, NO fitting logic.
    """

    @staticmethod
    def cylinder(
        radius: float,
        height: float,
        segments: int = 32,
        capped: bool = True
    ) -> Mesh:
        """
        Cylinder aligned with +Z axis, centered at origin.
        radius in XY, height along Z.
        """
        r = float(radius)
        h = float(height)
        seg = int(max(3, segments))

        angles = np.linspace(0.0, 2.0 * np.pi, seg, endpoint=False)
        x = r * np.cos(angles)
        y = r * np.sin(angles)

        z_bot = -h * 0.5
        z_top =  h * 0.5

        bottom = np.stack([x, y, np.full_like(x, z_bot)], axis=1)
        top    = np.stack([x, y, np.full_like(x, z_top)], axis=1)

        vertices = [bottom, top]
        faces = []

        # side faces
        for i in range(seg):
            j = (i + 1) % seg
            b0, b1 = i, j
            t0, t1 = i + seg, j + seg
            faces.append([b0, t0, t1])
            faces.append([b0, t1, b1])

        if capped:
            bottom_center = 2 * seg
            top_center = 2 * seg + 1
            vertices.append(
                np.array([[0.0, 0.0, z_bot], [0.0, 0.0, z_top]], dtype=np.float32)
            )

            for i in range(seg):
                j = (i + 1) % seg
                faces.append([bottom_center, j, i])
                faces.append([top_center, i + seg, j + seg])

        V = np.concatenate(vertices, axis=0).astype(np.float32)
        F = np.array(faces, dtype=np.int32)

        return Mesh(vertices=V, faces=F)

    @staticmethod
    def box(width: float, height: float, depth: float) -> Mesh:
        """
        Axis-aligned box centered at origin.
        width=X, depth=Y, height=Z
        """
        w = float(width) * 0.5
        d = float(depth) * 0.5
        h = float(height) * 0.5

        V = np.array([
            [-w, -d, -h],
            [ w, -d, -h],
            [ w,  d, -h],
            [-w,  d, -h],
            [-w, -d,  h],
            [ w, -d,  h],
            [ w,  d,  h],
            [-w,  d,  h],
        ], dtype=np.float32)

        F = np.array([
            [0, 1, 2], [0, 2, 3],
            [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1],
            [1, 5, 6], [1, 6, 2],
            [2, 6, 7], [2, 7, 3],
            [3, 7, 4], [3, 4, 0],
        ], dtype=np.int32)

        return Mesh(vertices=V, faces=F)


def save_obj(mesh: Mesh, path: str):
    """
    Save mesh to Wavefront OBJ (triangles).
    """
    with open(path, "w") as f:
        for v in mesh.vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for tri in mesh.faces:
            a, b, c = tri + 1
            f.write(f"f {a} {b} {c}\n")
