
import cv2
import numpy as np
from typing import Optional, Tuple
from mesh.mesh_shapes.mesh_object import MeshObject


class Renderer:
    """
    Renders:
    1) Mesh filled triangles overlay on the main frame (proper transparency)
    2) Optional mask highlight
    3) Small embedded interactive preview (click + drag to rotate)

    Key fix vs previous versions:
    - Mesh is drawn into a separate layer + a mesh_mask.
    - Alpha blending is applied ONLY where mesh_mask is true.
      => The object will actually look transparent and consistent.
    """

    def __init__(
        self,
        base_color=(0, 255, 255),        # yellow
        highlight_color=(0, 165, 255),   # orange
        alpha=0.6,                      # mesh overlay alpha
        mask_color=(0, 180, 0),
        mask_alpha=0.35,
        preview_angle: float = -np.pi / 3,
        max_faces_to_draw: int = 1500,
        preview_every: int = 6,
        drag_sensitivity: float = 0.01,  # radians per pixel
    ):
        # Visual config
        self.base_color = base_color
        self.highlight_color = highlight_color
        self.alpha = float(alpha)
        self.mask_color = mask_color
        self.mask_alpha = float(mask_alpha)

        # Preview config
        self.preview_angle = float(preview_angle)
        self.max_faces_to_draw = int(max(1, max_faces_to_draw))
        self.preview_every = int(max(1, preview_every))

        # Cached preview (performance)
        self._preview_counter = 0
        self._last_preview: Optional[np.ndarray] = None

        # Mouse interaction state for preview
        self._window_name: Optional[str] = None
        self._preview_rect: Optional[Tuple[int, int, int, int]] = None  # (x1,y1,x2,y2)
        self._mouse_down = False
        self._mouse_last_x = 0
        self._mouse_last_y = 0
        self._drag_sensitivity = float(drag_sensitivity)

    # ---------------------------------------------------------------------
    # Public: attach mouse interaction to a cv2 window
    # ---------------------------------------------------------------------

    def attach_mouse(self, window_name: str) -> None:
        """
        Call once after cv2.namedWindow(window_name).
        Enables: click+drag inside the embedded preview to rotate it.
        """
        self._window_name = window_name
        cv2.setMouseCallback(window_name, self._on_mouse)

    def set_preview_angle(self, angle: float) -> None:
        self.preview_angle = float(angle)

    def adjust_preview_angle(self, delta_angle: float) -> None:
        self.preview_angle += float(delta_angle)

    def get_preview_angle(self) -> float:
        return float(self.preview_angle)

    # ---------------------------------------------------------------------
    # Main render
    # ---------------------------------------------------------------------

    def render_frame(self, frame: np.ndarray, mesh_object: Optional[MeshObject]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Returns:
            output_frame, preview_image (or None)

        Transparency behavior:
        - Mesh overlay is blended ONLY where mesh pixels exist (mesh_mask).
        - Mask highlight is blended only where mask exists.
        """
        if mesh_object is None or mesh_object.vertices is None or mesh_object.faces is None:
            self._last_preview = None
            self._preview_rect = None
            return frame, None

        if len(mesh_object.vertices) == 0 or len(mesh_object.faces) == 0:
            self._last_preview = None
            self._preview_rect = None
            return frame, None

        out = frame.copy()

        # 1) Build optional mask highlight layer
        mask_layer, mask_bool = self._build_mask_layer(frame.shape, mesh_object)

        # 2) Draw mesh into separate layer + mesh mask
        mesh_layer, mesh_mask = self._draw_mesh_layer(frame.shape, mesh_object)

        # 3) Blend mesh ONLY where mesh exists (this fixes "not transparent")
        out = self._blend_where(out, mesh_layer, mesh_mask, self.alpha)

        # 4) Blend segmentation mask highlight (optional)
        if mask_bool is not None and mask_bool.any():
            out = self._blend_where(out, mask_layer, mask_bool, self.mask_alpha)

        # 5) Draw rotation indicator arrow on bbox (optional)
        out = self._draw_rotation_indicator(out, mesh_object)

        # 6) Preview (cached, unless dragging -> refresh every frame)
        self._preview_counter += 1
        must_refresh_preview = self._mouse_down
        if must_refresh_preview or (self._preview_counter % self.preview_every == 0):
            prev = self._render_mesh_preview(mesh_object)
            if prev is not None:
                self._last_preview = prev

        preview = self._last_preview
        if preview is not None:
            out, rect = self._embed_preview(out, preview)
            self._preview_rect = rect
        else:
            self._preview_rect = None

        return out, preview

    # ---------------------------------------------------------------------
    # Layer builders / blending
    # ---------------------------------------------------------------------

    def _build_mask_layer(self, frame_shape: Tuple[int, int, int], mesh: MeshObject) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Returns:
            mask_layer: BGR layer with mask_color where mask is true
            mask_bool: boolean mask (H,W) where mask is true (or None if no mask)
        """
        h, w = frame_shape[:2]
        layer = np.zeros((h, w, 3), dtype=np.uint8)

        if getattr(mesh, "mask", None) is None:
            return layer, None

        mask = mesh.mask
        if mask is None:
            return layer, None

        # Normalize to uint8 0/255
        if mask.dtype != np.uint8:
            mask = (mask > 0).astype(np.uint8) * 255
        elif mask.max() <= 1:
            mask = (mask > 0).astype(np.uint8) * 255

        if mask.ndim == 3:
            mask = mask.squeeze()

        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        mb = mask > 0
        if mb.any():
            layer[mb] = self.mask_color
            return layer, mb

        return layer, mb  # empty

    def _blend_where(self, base: np.ndarray, overlay: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
        """
        Blend overlay onto base only where mask is True.
        base and overlay are uint8 BGR.
        mask is bool (H,W) or uint8 (0/255).
        """
        if mask.dtype != np.bool_:
            mask_bool = mask > 0
        else:
            mask_bool = mask

        if not mask_bool.any():
            return base

        out = base.copy()
        # float blend only on masked pixels
        a = float(alpha)
        out[mask_bool] = (a * overlay[mask_bool] + (1.0 - a) * base[mask_bool]).astype(np.uint8)
        return out

    # ---------------------------------------------------------------------
    # Mesh drawing (separate layer + mask)
    # ---------------------------------------------------------------------

    def _sample_faces(self, faces: np.ndarray) -> np.ndarray:
        F = len(faces)
        if F <= self.max_faces_to_draw:
            return faces
        step = int(np.ceil(F / self.max_faces_to_draw))
        return faces[::step]

    def _draw_mesh_layer(self, frame_shape: Tuple[int, int, int], mesh: MeshObject) -> Tuple[np.ndarray, np.ndarray]:
        """
        Draw mesh into an offscreen layer (opaque) and a mask of where it drew.
        Returns:
            mesh_layer: uint8 BGR
            mesh_mask:  uint8 (0/255)
        """
        h, w = frame_shape[:2]
        mesh_layer = np.zeros((h, w, 3), dtype=np.uint8)
        mesh_mask = np.zeros((h, w), dtype=np.uint8)

        verts = mesh.vertices
        faces_all = mesh.faces

        # Main overlay uses XY directly
        points_2d = verts[:, :2].astype(np.int32)
        z_values = verts[:, 2].astype(np.float32)

        z_min = float(z_values.min())
        z_max = float(z_values.max())
        depth_range = (z_max - z_min) if (z_max - z_min) != 0 else 1.0

        faces = self._sample_faces(faces_all)
        tris_2d = points_2d[faces]  # (F',3,2)

        for face_idx, face in enumerate(faces):
            tri_pts = tris_2d[face_idx].reshape(-1, 1, 2)

            stripe_color = self.base_color if ((face_idx // 2) % 2 == 0) else self.highlight_color

            avg_depth = float(np.mean(z_values[face]))
            depth_norm = (avg_depth - z_min) / depth_range
            shade = 0.6 + 0.4 * (1.0 - depth_norm)

            shaded_color = (
                int(stripe_color[0] * shade),
                int(stripe_color[1] * shade),
                int(stripe_color[2] * shade),
            )

            cv2.fillPoly(mesh_layer, [tri_pts], shaded_color)
            cv2.fillPoly(mesh_mask, [tri_pts], 255)

        return mesh_layer, mesh_mask

    # ---------------------------------------------------------------------
    # Rotation indicator arrow
    # ---------------------------------------------------------------------

    def _draw_rotation_indicator(self, frame: np.ndarray, mesh: MeshObject) -> np.ndarray:
        if getattr(mesh, "bbox_xyxy", None) is None:
            return frame

        x1, y1, x2, y2 = mesh.bbox_xyxy
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        radius = max(1, int((x2 - x1) / 2))

        angle = 0.0
        if hasattr(mesh, "pose") and hasattr(mesh.pose, "rotation"):
            try:
                angle = float(mesh.pose.rotation[2])  # Z rotation
            except Exception:
                angle = 0.0

        arrow_len = int(radius * 1.2)
        end_x = int(cx + arrow_len * np.cos(angle - np.pi / 2))
        end_y = int(cy + arrow_len * np.sin(angle - np.pi / 2))

        cv2.arrowedLine(frame, (cx, cy), (end_x, end_y), (0, 0, 255), 3, tipLength=0.2)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
        return frame

    # ---------------------------------------------------------------------
    # Preview rendering + embedding
    # ---------------------------------------------------------------------

    def _render_mesh_preview(self, mesh: MeshObject, size: Tuple[int, int] = (280, 280)) -> Optional[np.ndarray]:
        if mesh.vertices is None or mesh.faces is None or len(mesh.vertices) == 0:
            return None

        verts = mesh.vertices.astype(np.float32)

        # Rotate around Y axis (yaw): affects X and Z
        cos_a, sin_a = np.cos(self.preview_angle), np.sin(self.preview_angle)
        rx = verts[:, 0] * cos_a + verts[:, 2] * sin_a
        rz = (-verts[:, 0] * sin_a) + (verts[:, 2] * cos_a)
        ry = verts[:, 1]

        pts2 = np.stack([rx, ry], axis=1)

        min_xy = pts2.min(axis=0)
        max_xy = pts2.max(axis=0)
        span = np.maximum(max_xy - min_xy, 1e-3)

        padding = 16
        scale = min((size[0] - 2 * padding) / span[0], (size[1] - 2 * padding) / span[1])
        pts_scaled = (pts2 - min_xy) * scale + padding

        z_min, z_max = float(rz.min()), float(rz.max())
        depth_range = (z_max - z_min) if (z_max - z_min) != 0 else 1.0

        canvas = np.ones((size[1], size[0], 3), dtype=np.uint8) * 245
        tris = pts_scaled[mesh.faces]

        for face_idx, face in enumerate(mesh.faces):
            tri_pts = tris[face_idx].astype(np.int32)
            stripe_color = self.base_color if ((face_idx // 2) % 2 == 0) else self.highlight_color

            avg_depth = float(np.mean(rz[face]))
            depth_norm = (avg_depth - z_min) / depth_range
            shade = 0.6 + 0.4 * (1.0 - depth_norm)
            shaded_color = tuple(int(c * shade) for c in stripe_color)

            cv2.fillPoly(canvas, [tri_pts.reshape(-1, 1, 2)], shaded_color)

        cv2.putText(canvas, "drag to rotate", (12, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1, cv2.LINE_AA)

        angle_text = f"view: {np.degrees(self.preview_angle):.1f} deg"
        cv2.putText(canvas, angle_text, (12, size[1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 1, cv2.LINE_AA)

        return canvas

    def _embed_preview(
        self,
        base_frame: np.ndarray,
        preview: np.ndarray,
        margin: int = 16,
        max_ratio: float = 0.35,
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Embed preview in top-right corner.
        Returns: (frame_with_preview, rect_xyxy).
        """
        h, w = base_frame.shape[:2]
        max_w = int(w * max_ratio)
        max_h = int(h * max_ratio)

        scale = min(max_w / preview.shape[1], max_h / preview.shape[0], 1.0)
        new_w = max(1, int(preview.shape[1] * scale))
        new_h = max(1, int(preview.shape[0] * scale))
        resized = cv2.resize(preview, (new_w, new_h), interpolation=cv2.INTER_AREA)

        x1 = w - new_w - margin
        y1 = margin
        x2 = x1 + new_w
        y2 = y1 + new_h

        out = base_frame.copy()
        cv2.rectangle(out, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (20, 20, 20), 2)
        out[y1:y2, x1:x2] = resized

        return out, (x1, y1, x2, y2)

    # ---------------------------------------------------------------------
    # Mouse interaction
    # ---------------------------------------------------------------------

    def _in_preview(self, x: int, y: int) -> bool:
        if self._preview_rect is None:
            return False
        x1, y1, x2, y2 = self._preview_rect
        return (x1 <= x < x2) and (y1 <= y < y2)

    def _on_mouse(self, event, x, y, flags, param) -> None:
        if self._preview_rect is None:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            if self._in_preview(x, y):
                self._mouse_down = True
                self._mouse_last_x = x
                self._mouse_last_y = y

        elif event == cv2.EVENT_LBUTTONUP:
            self._mouse_down = False

        elif event == cv2.EVENT_MOUSEMOVE:
            if not self._mouse_down:
                return
            if not self._in_preview(x, y):
                return

            dx = x - self._mouse_last_x
            self.preview_angle += float(dx) * self._drag_sensitivity

            self._mouse_last_x = x
            self._mouse_last_y = y

