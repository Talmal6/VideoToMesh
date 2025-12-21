import cv2
import numpy as np
from typing import List, Optional, Tuple
from mesh.mesh_shapes.mesh_object import MeshObject

class Renderer:
    def __init__(self,
                 base_color=(0, 255, 255),      # צהוב (ברירת מחדל)
                 highlight_color=(0, 165, 255), # כתום (צבע שני לפסים)
                 alpha=0.6,
                 preview_angle: float = -np.pi / 3):
        """
        :param base_color: הצבע הראשי של הגליל (BGR)
        :param highlight_color: הצבע השני ליצירת אפקט הפסים
        :param alpha: שקיפות (0.0 - 1.0)
        """
        self.base_color = base_color
        self.highlight_color = highlight_color
        self.alpha = alpha
        self.preview_angle = preview_angle

    def render_frame(self, frame: np.ndarray, mesh_objects: List[MeshObject]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if not mesh_objects:
            return frame, None

        overlay = frame.copy()
        draw_count = 0
        mesh_preview = None

        for mesh in mesh_objects:
            if (mesh.vertices is None or len(mesh.vertices) == 0 or
                mesh.faces is None or len(mesh.faces) == 0):
                continue

            draw_count += 1

            # 1. המרת קודקודים ל-2D (זריקת ציר Z)
            points_2d = mesh.vertices[:, :2].astype(np.int32)
            z_values = mesh.vertices[:, 2]
            z_min, z_max = z_values.min(), z_values.max()
            depth_range = (z_max - z_min) if (z_max - z_min) != 0 else 1.0

            # 2. שליפת כל המשולשים
            # shape: (Num_Triangles, 3, 2)
            all_triangles = points_2d[mesh.faces]

            # 3. ציור משולשים עם הצללה עמוקה
            # כל זוג משולשים מייצג "פס" גלילי; נצבע בזוגיות/אי-זוגיות
            # ובנוסף נוסיף הצללה לפי עומק Z כדי לשדר עומק.
            for face_idx, face in enumerate(mesh.faces):
                tri_pts = all_triangles[face_idx]
                stripe_color = self.base_color if ((face_idx // 2) % 2 == 0) else self.highlight_color

                # עומק ממוצע למשולש (נורמליזציה ל-0..1)
                avg_depth = np.mean(z_values[face])
                depth_norm = (avg_depth - z_min) / depth_range

                shade = 0.6 + 0.4 * (1 - depth_norm)
                shaded_color = tuple(int(c * shade) for c in stripe_color)

                cv2.fillPoly(overlay, pts=[tri_pts.reshape(-1, 1, 2)], color=shaded_color)

            if mesh_preview is None:
                mesh_preview = self._render_mesh_preview(mesh)

            # --- Visualization of Rotation ---
            # Draw a line indicating the current rotation angle
            x1, y1, x2, y2 = mesh.bbox_xyxy
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            radius = int((x2 - x1) / 2)
            
            angle = mesh.pose.rotation[1]
            # Project angle to 2D vector (visualizing Y-rotation as a clock hand)
            end_x = int(cx + radius * np.cos(angle))
            end_y = int(cy + radius * np.sin(angle))

            # (אופציונלי) מסגרת דקה לכל משולש לחידוד נוסף
            # cv2.polylines(overlay, pts=all_triangles, isClosed=True, color=(0, 0, 0), thickness=1)

        if draw_count == 0:
            return frame, None

        # 5. מיזוג שקיפות
        blended = cv2.addWeighted(overlay, self.alpha, frame, 1 - self.alpha, 0)

        # 6. ציור הקו האדום למעלה לאחר השקיפות כדי שיהיה חד
        for mesh in mesh_objects:
            if mesh.vertices is None or mesh.faces is None or len(mesh.vertices) == 0:
                continue

            x1, y1, x2, y2 = mesh.bbox_xyxy
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            radius = int((x2 - x1) / 2)

            angle = mesh.pose.rotation[1]
            end_x = int(cx + radius * np.cos(angle))
            end_y = int(cy + radius * np.sin(angle))

            cv2.line(blended, (cx, cy), (end_x, end_y), (0, 0, 255), 3)
            cv2.circle(blended, (cx, cy), 4, (0, 0, 255), -1)

        if mesh_preview is not None:
            blended = self._embed_preview(blended, mesh_preview)

        return blended, mesh_preview

    def set_preview_angle(self, angle: float) -> None:
        self.preview_angle = angle

    def adjust_preview_angle(self, delta_angle: float) -> None:
        self.preview_angle += delta_angle

    def get_preview_angle(self) -> float:
        return self.preview_angle

    def _render_mesh_preview(self, mesh: MeshObject, size: Tuple[int, int] = (280, 280)) -> Optional[np.ndarray]:
        if mesh.vertices is None or mesh.faces is None or len(mesh.vertices) == 0:
            return None

        verts = mesh.vertices.astype(np.float32)
        cos_a, sin_a = np.cos(self.preview_angle), np.sin(self.preview_angle)

        rotated_x = verts[:, 0] * cos_a + verts[:, 2] * sin_a
        rotated_z = (-verts[:, 0] * sin_a) + (verts[:, 2] * cos_a)
        rotated_y = verts[:, 1]

        points_2d = np.stack([rotated_x, rotated_y], axis=1)

        min_xy = points_2d.min(axis=0)
        max_xy = points_2d.max(axis=0)
        span = np.maximum(max_xy - min_xy, 1e-3)
        padding = 16
        scale = min((size[0] - 2 * padding) / span[0], (size[1] - 2 * padding) / span[1])
        points_scaled = (points_2d - min_xy) * scale + padding

        z_min, z_max = rotated_z.min(), rotated_z.max()
        depth_range = (z_max - z_min) if (z_max - z_min) != 0 else 1.0

        canvas = np.ones((size[1], size[0], 3), dtype=np.uint8) * 245

        all_triangles = points_scaled[mesh.faces]

        for face_idx, face in enumerate(mesh.faces):
            tri_pts = all_triangles[face_idx].astype(np.int32)
            stripe_color = self.base_color if ((face_idx // 2) % 2 == 0) else self.highlight_color

            avg_depth = np.mean(rotated_z[face])
            depth_norm = (avg_depth - z_min) / depth_range
            shade = 0.6 + 0.4 * (1 - depth_norm)
            shaded_color = tuple(int(c * shade) for c in stripe_color)

            cv2.fillPoly(canvas, pts=[tri_pts.reshape(-1, 1, 2)], color=shaded_color)

        angle_text = f"view: {np.degrees(self.preview_angle):.1f} deg"
        cv2.putText(
            canvas,
            angle_text,
            (12, size[1] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (50, 50, 50),
            1,
            cv2.LINE_AA,
        )

        return canvas

    def _embed_preview(self, base_frame: np.ndarray, preview: np.ndarray,
                       margin: int = 16, max_ratio: float = 0.35) -> np.ndarray:
        h, w = base_frame.shape[:2]
        max_w = int(w * max_ratio)
        max_h = int(h * max_ratio)

        scale = min(max_w / preview.shape[1], max_h / preview.shape[0], 1.0)
        new_w = max(1, int(preview.shape[1] * scale))
        new_h = max(1, int(preview.shape[0] * scale))
        resized = cv2.resize(preview, (new_w, new_h), interpolation=cv2.INTER_AREA)

        x_start = w - new_w - margin
        y_start = margin

        x_end = x_start + new_w
        y_end = y_start + new_h

        inset = base_frame.copy()
        cv2.rectangle(inset, (x_start - 2, y_start - 2), (x_end + 2, y_end + 2), (20, 20, 20), thickness=2)
        inset[y_start:y_end, x_start:x_end] = resized
        return inset
