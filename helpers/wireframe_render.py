import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pyrender
import trimesh

from mesh_proccesors.mesh_factory import Mesh, MeshFactory


def project_mesh_to_image(mesh: Mesh, fit: dict) -> np.ndarray:
    """
    Project 3D mesh vertices to 2D image coordinates using the fit parameters.
    This is a 'fake' projection assuming orthographic-like view aligned with image plane.
    
    fit: dict from SymmetricShapeFitter
    """
    cx, cy = fit["center"]
    angle_deg = fit["angle_deg"]
    
    # 1. Get vertices (N, 3)
    # In MeshFactory:
    # Cylinder: Z is height axis, XY is radius
    # Box: Z is height axis, X is width, Y is depth
    
    # We want to map:
    # Mesh Z -> Image Y (aligned with long axis of object)
    # Mesh X -> Image X (aligned with short axis/width)
    # Mesh Y -> Depth (into screen)
    
    # Let's rotate the mesh so its 'up' (Z) aligns with image Y
    # and its 'width' (X) aligns with image X.
    
    # Original vertices
    V = mesh.vertices.copy()
    
    # Swap axes to align with image plane logic before rotation
    # Mesh Z (height) -> Image Y
    # Mesh X (width)  -> Image X
    # Mesh Y (depth)  -> Z depth
    
    # V_img = [x, z, y]
    x = V[:, 0]
    y = V[:, 2] # height becomes Y
    z = V[:, 1] # depth becomes Z
    
    # 2. Rotate in 2D plane by 'angle'
    # The angle from minAreaRect is usually the angle of the long side (height).
    # If angle=0, the object is vertical? Or horizontal?
    # In minAreaRect, angle is often relative to X axis.
    # Let's assume standard rotation matrix.
    
    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    
    # Rotation matrix for 2D points (x, y)
    # x_new = x * cos - y * sin
    # y_new = x * sin + y * cos
    
    x_rot = x * c - y * s
    y_rot = x * s + y * c
    
    # 3. Translate to center
    x_final = x_rot + cx
    y_final = y_rot + cy
    
    return np.stack([x_final, y_final], axis=1)

def draw_mesh_wireframe(frame: np.ndarray, mesh: Mesh, fit: dict, color=(0, 255, 255), thickness=1):
    """
    Draws the wireframe of the mesh on the frame.
    """
    if mesh is None or fit is None:
        return

    pts_2d = project_mesh_to_image(mesh, fit)
    pts_2d = pts_2d.astype(np.int32)
    
    # Draw edges
    # Naive approach: draw all edges from faces
    # To avoid drawing same edge twice, we could use a set of edges
    
    edges = set()
    for f in mesh.faces:
        # f is [v1, v2, v3]
        for i in range(3):
            v1 = f[i]
            v2 = f[(i+1)%3]
            if v1 > v2:
                v1, v2 = v2, v1
            edges.add((v1, v2))
            
    h, w = frame.shape[:2]
    
    for v1_idx, v2_idx in edges:
        pt1 = pts_2d[v1_idx]
        pt2 = pts_2d[v2_idx]
        
        # Simple bounds check to avoid drawing way off screen
        if (0 <= pt1[0] < w and 0 <= pt1[1] < h) or (0 <= pt2[0] < w and 0 <= pt2[1] < h):
            cv2.line(frame, tuple(pt1), tuple(pt2), color, thickness, cv2.LINE_AA)

def generate_mesh_from_fit(fit: dict) -> Optional[Mesh]:
    if not fit:
        return None

    shape = fit.get("shape")
    dims = fit.get("dimensions", {})

    if shape == "cylinder":
        r = dims.get("radius", 10.0)
        h = dims.get("height", 10.0)
        return MeshFactory.cylinder(radius=r, height=h, segments=16)

    w = dims.get("width", 10.0)
    h = dims.get("height", 10.0)
    d = dims.get("depth", 10.0)
    return MeshFactory.box(width=w, height=h, depth=d)

def render_mesh_pretty(mesh_path, out_png="mesh.png", W=1280, H=720,
                       bg=(1.0, 1.0, 1.0, 1.0), wireframe=False):
    mesh = trimesh.load(mesh_path, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate(mesh.dump())

    # נורמליזציה לגודל מסודר + מרכז
    mesh = mesh.copy()
    mesh.vertices -= mesh.vertices.mean(axis=0)
    scale = np.max(np.linalg.norm(mesh.vertices, axis=1))
    if scale > 0:
        mesh.vertices /= scale

    scene = pyrender.Scene(bg_color=bg, ambient_light=(0.15, 0.15, 0.15))

    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=(0.85, 0.85, 0.88, 1.0),
        metallicFactor=0.05,
        roughnessFactor=0.45
    )

    pm = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True, wireframe=wireframe)
    scene.add(pm)

    # תאורות טובות (3-point lighting)
    key = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    fill = pyrender.DirectionalLight(color=np.ones(3), intensity=1.5)
    back = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)

    scene.add(key, pose=_look_at(light_dir=(+1, -1, -1)))
    scene.add(fill, pose=_look_at(light_dir=(-1, -0.3, -1)))
    scene.add(back, pose=_look_at(light_dir=(0, +1, -1)))

    # מצלמה
    cam = pyrender.PerspectiveCamera(yfov=np.deg2rad(35.0))
    cam_pose = _camera_pose(dist=2.2, elev_deg=20, azim_deg=35)
    scene.add(cam, pose=cam_pose)

    r = pyrender.OffscreenRenderer(W, H)
    color, depth = r.render(scene)
    r.delete()

    import imageio.v2 as imageio
    imageio.imwrite(out_png, color)
    return out_png


@dataclass
class MeshRenderProcessor:
    out_dir: Path
    width: int = 1280
    height: int = 720
    bg_rgba: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    wireframe: bool = False
    elev_deg: float = 20.0
    azim_deg: float = 35.0
    dist: float = 2.2

    def process(self, mesh_path: str | Path, out_name: Optional[str] = None) -> Path:
        """Load OBJ/mesh, normalize, render with pyrender offscreen, return PNG path."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        mesh_path = Path(mesh_path)

        if out_name is None:
            suffix = "_wire" if self.wireframe else ""
            out_name = f"{mesh_path.stem}{suffix}.png"
        out_png = self.out_dir / out_name

        mesh = trimesh.load(mesh_path, force="mesh")
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = trimesh.util.concatenate(mesh.dump())

        mesh = _normalize_mesh(mesh)

        scene = pyrender.Scene(bg_color=self.bg_rgba, ambient_light=(0.15, 0.15, 0.15))

        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=(0.85, 0.85, 0.88, 1.0),
            metallicFactor=0.05,
            roughnessFactor=0.45,
        )

        pm = pyrender.Mesh.from_trimesh(
            mesh, material=material, smooth=True, wireframe=self.wireframe
        )
        scene.add(pm)

        scene.add(
            pyrender.DirectionalLight(color=np.ones(3), intensity=3.0),
            pose=_look_at(light_dir=(+1, -1, -1)),
        )
        scene.add(
            pyrender.DirectionalLight(color=np.ones(3), intensity=1.5),
            pose=_look_at(light_dir=(-1, -0.3, -1)),
        )
        scene.add(
            pyrender.DirectionalLight(color=np.ones(3), intensity=2.0),
            pose=_look_at(light_dir=(0, +1, -1)),
        )

        cam = pyrender.PerspectiveCamera(yfov=np.deg2rad(35.0))
        scene.add(
            cam,
            pose=_camera_pose(
                dist=self.dist, elev_deg=self.elev_deg, azim_deg=self.azim_deg
            ),
        )

        renderer = pyrender.OffscreenRenderer(self.width, self.height)
        color, _depth = renderer.render(scene)
        renderer.delete()

        import imageio.v2 as imageio

        imageio.imwrite(out_png.as_posix(), color)
        return out_png

def _camera_pose(dist=2.2, elev_deg=20, azim_deg=35):
    elev = np.deg2rad(elev_deg)
    azim = np.deg2rad(azim_deg)
    x = dist * np.cos(elev) * np.sin(azim)
    y = dist * np.sin(elev)
    z = dist * np.cos(elev) * np.cos(azim)
    eye = np.array([x, y, z])
    target = np.array([0.0, 0.0, 0.0])
    up = np.array([0.0, 1.0, 0.0])

    return _view_matrix(eye, target, up)

def _view_matrix(eye, target, up):
    f = (target - eye)
    f /= np.linalg.norm(f)
    u = up / np.linalg.norm(up)
    s = np.cross(f, u)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)

    M = np.eye(4)
    M[0, :3] = s
    M[1, :3] = u
    M[2, :3] = -f
    T = np.eye(4)
    T[:3, 3] = -eye
    return M @ T

def _look_at(light_dir=(1, -1, -1)):
    d = np.array(light_dir, dtype=float)
    d /= np.linalg.norm(d)
    # “ממקמים” את האור כאילו הוא מצלמה שמסתכלת למרכז
    eye = -2.0 * d
    return _view_matrix(eye, np.zeros(3), np.array([0, 1, 0], dtype=float))


def _normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = mesh.copy()
    mesh.vertices -= mesh.vertices.mean(axis=0)
    scale = np.max(np.linalg.norm(mesh.vertices, axis=1))
    if scale > 0:
        mesh.vertices /= scale
    return mesh

# שימוש:
# render_mesh_pretty("mesh.obj", out_png="mesh.png", wireframe=False)
# render_mesh_pretty("mesh.obj", out_png="mesh_wire.png", wireframe=True)
