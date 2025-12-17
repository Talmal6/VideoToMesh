import numpy as np
from mesh_factory import Mesh, MeshFactory
def _rot_z(deg: float) -> np.ndarray:
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([
        [ c, -s, 0.0],
        [ s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

def apply_pose_2p5d(mesh: Mesh, cx: float, cy: float, angle_deg: float) -> Mesh:
    """
    2.5D pose: rotate around Z (image plane), then translate in XY by (cx,cy).
    Note: units are in 'pixel space' if your dims are in pixels.
    """
    R = _rot_z(angle_deg)
    V = (mesh.vertices @ R.T).astype(np.float32)
    V[:, 0] += float(cx)
    V[:, 1] += float(cy)
    return Mesh(vertices=V, faces=mesh.faces)

def mesh_from_fit(fit: dict, cylinder_segments: int = 32) -> Mesh:
    """
    fit dict expected:
      cylinder: {'shape':'cylinder','cx','cy','angle_deg','r','height'}
      box:      {'shape':'box','cx','cy','angle_deg','box_w','box_h','box_d'}
    Returns a mesh already posed in 2D (XY) according to the fit.
    """
    shape = fit["shape"]
    cx, cy = fit["cx"], fit["cy"]
    angle = fit.get("angle_deg", 0.0)

    if shape == "cylinder":
        r = float(fit["r"])
        h = float(fit["height"])
        base = MeshFactory.cylinder(radius=r, height=h, segments=cylinder_segments, capped=True)
        return apply_pose_2p5d(base, cx, cy, angle)

    if shape == "box":
        w = float(fit["box_w"])
        h = float(fit["box_h"])
        d = float(fit["box_d"])
        base = MeshFactory.box(width=w, height=h, depth=d)
        return apply_pose_2p5d(base, cx, cy, angle)

    raise ValueError(f"Unknown shape: {shape}")
