from __future__ import annotations

from typing import Optional, Tuple, Union
import logging

import numpy as np

from detection.detection import Detection
from mesh.mesh_proccesors.mesh_handler import Handler
from mesh.mesh_shapes.mesh_object import MeshObject

logger = logging.getLogger(__name__)


class SAM3DMeshHandler(Handler):
    """
    Builds a mesh from (frame + det.mask).
    Supports:
      - RGB-D: if you pass depth in det.frame as a tuple (rgb, depth) or extend your pipeline accordingly.
      - RGB-only: uses a monocular depth estimator (demo-quality).

    IMPORTANT: Your MeshManager currently passes `frame` as the 3rd arg (not frame_shape_hw). :contentReference[oaicite:4]{index=4}
    So we accept either ndarray frame or (H,W) tuple.
    """
    def __init__(
        self,
        fx: float = 600.0,
        fy: float = 600.0,
        cx: float = 320.0,
        cy: float = 240.0,
        use_mono_depth: bool = True,
    ):
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.use_mono_depth = use_mono_depth
        self._depth_model = None

    def can_handle(self, det: Detection) -> bool:
        has_mask = det.mask is not None
        logger.info(f"[SAM3D] can_handle check: label={det.label}, has_mask={has_mask}, mask_shape={det.mask.shape if has_mask else None}")
        return has_mask

    # ---- Depth estimation (RGB-only) ----
    def _lazy_init_depth(self):
        if self._depth_model is not None:
            return
        try:
            import torch
        except ImportError as e:
            raise RuntimeError("RGB-only mode requires torch.") from e

        # This will download weights the first time unless cached.
        # If you cannot download from the environment, you must provide depth from an RGB-D camera instead.
        self._depth_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        self._depth_model.eval()

        self._midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

        self._torch = torch

    def _estimate_depth_midas(self, bgr: np.ndarray) -> np.ndarray:
        self._lazy_init_depth()

        torch = self._torch
        img_rgb = bgr[:, :, ::-1].copy()  # BGR -> RGB

        inp = self._midas_transforms(img_rgb).to(next(self._depth_model.parameters()).device)

        with torch.no_grad():
            pred = self._depth_model(inp)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = pred.detach().cpu().numpy().astype(np.float32)
        # Normalize to [0,1] for pseudo-depth
        dmin, dmax = np.percentile(depth, 5), np.percentile(depth, 95)
        depth = np.clip((depth - dmin) / max(1e-6, (dmax - dmin)), 0.0, 1.0)
        # Map to an arbitrary metric-ish range (demo only)
        depth = 0.3 + depth * 1.7  # ~[0.3m, 2.0m]
        return depth

    # ---- Geometry ----
    def _backproject(self, depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
        h, w = depth.shape[:2]
        ys, xs = np.where(mask.astype(bool))
        if len(xs) < 200:
            return np.zeros((0, 3), dtype=np.float32)

        z = depth[ys, xs].astype(np.float32)
        x = (xs.astype(np.float32) - self.cx) * z / self.fx
        y = (ys.astype(np.float32) - self.cy) * z / self.fy
        pts = np.stack([x, y, z], axis=1)
        return pts

    def _mesh_from_pointcloud(self, pts: np.ndarray, det: Detection) -> Optional[MeshObject]:
        if pts.shape[0] < 2000:
            return None

        try:
            import open3d as o3d
        except ImportError as e:
            raise RuntimeError("Mesh extraction requires open3d.") from e

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))

        # Normals (required for Poisson)
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(50)

        mesh, _dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)

        # Crop to the point cloud bounds to remove Poisson “balloon”
        bbox = pcd.get_axis_aligned_bounding_box()
        mesh = mesh.crop(bbox)

        v = np.asarray(mesh.vertices, dtype=np.float32)
        f = np.asarray(mesh.triangles, dtype=np.int32)
        if v.shape[0] == 0 or f.shape[0] == 0:
            return None
        
        return MeshObject(
            object_id=det.object_id,
            label=det.label,
            confidence=det.confidence,
            frame_index=det.frame_index,
            bbox_xyxy=det.bbox_xyxy,
            mask=det.mask,
            vertices=v,
            faces=f
        )

    def process(self, obj, det: Detection, frame_or_shape: Union[np.ndarray, Tuple[int, int]]):
        logger.info(f"[SAM3D] process called: label={det.label}")
        
        frame = det.frame if det.frame is not None else (frame_or_shape if isinstance(frame_or_shape, np.ndarray) else None)
        if frame is None:
            logger.warning("[SAM3D] No frame available")
            return None

        if det.mask is None:
            logger.warning("[SAM3D] No mask available")
            return None

        logger.info(f"[SAM3D] Starting depth estimation, frame shape: {frame.shape}, mask shape: {det.mask.shape}")
        
        if not self.use_mono_depth:
            raise RuntimeError("No depth provided. Enable mono depth or use an RGB-D source.")

        depth = self._estimate_depth_midas(frame)
        logger.info(f"[SAM3D] Depth estimated, shape: {depth.shape}")
        
        pts = self._backproject(depth, det.mask)
        logger.info(f"[SAM3D] Backprojected {pts.shape[0]} points")

        mesh = self._mesh_from_pointcloud(pts, det)
        if mesh:
            logger.info(f"[SAM3D] Mesh created: {mesh.vertices.shape[0]} vertices, {mesh.faces.shape[0]} faces")
            
            # Convert metric vertices back to pixel space for Renderer consistency
            # Renderer expects X,Y in pixels. We scale Z to be comparable for rotation.
            v = mesh.vertices
            z_metric = v[:, 2]
            
            # Avoid divide by zero
            z_safe = np.maximum(z_metric, 0.1)
            
            x_pix = (v[:, 0] * self.fx / z_safe) + self.cx
            y_pix = (v[:, 1] * self.fy / z_safe) + self.cy
            z_pix = z_metric * 100.0  # Scale depth for visualization
            
            mesh.vertices = np.stack([x_pix, y_pix, z_pix], axis=1).astype(np.float32)
        else:
            logger.warning("[SAM3D] Failed to create mesh from point cloud")
        
        return mesh
