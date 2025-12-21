import cv2
import numpy as np
from typing import List, Optional, Tuple
from mesh.mesh_shapes.mesh_object import MeshObject

class Renderer:
    def __init__(self,
                 base_color=(0, 255, 255),      # צהוב (ברירת מחדל)
                 highlight_color=(0, 165, 255), # כתום (צבע שני לפסים)
                 alpha=0.6,
                 mask_color=(0, 180, 0),
                 mask_alpha=0.35,
                 preview_angle: float = -np.pi / 3):
        """
        :param base_color: הצבע הראשי של הגליל (BGR)
        :param highlight_color: הצבע השני ליצירת אפקט הפסים
        :param alpha: שקיפות (0.0 - 1.0)
        """
        self.base_color = base_color
        self.highlight_color = highlight_color
        self.alpha = alpha
        self.mask_color = mask_color
        self.mask_alpha = mask_alpha
        self.preview_angle = preview_angle
        self.max_faces_to_draw = 1500  # תתחיל מ-1500 ותכוון
    def render_frame(self, frame: np.ndarray, mesh_object: Optional[MeshObject]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        mesh_objects = [mesh_object]
        if not mesh_objects:
            return frame, None

        overlay = frame.copy()
        draw_count = 0
        mesh_preview = None

        # מסכת הדגשה (אם יש)
        mask_overlay = np.zeros_like(frame)
        has_mask = False

        # --- Performance knobs (אפשר לשים ב-__init__ אם אתה מעדיף) ---
        max_faces_to_draw = getattr(self, "max_faces_to_draw", 1500)  # cap על מספר המשולשים לציור
        preview_every = getattr(self, "_preview_every", 6)            # preview פעם ב-N פריימים
        if not hasattr(self, "_preview_counter"):
            self._preview_counter = 0
        self._preview_counter += 1

        for mesh in mesh_objects:
            if mesh is None or mesh.vertices is None or mesh.faces is None:
                continue
            if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
                continue

            draw_count += 1

            # -----------------------------
            # 0) מסכה ירוקה (זולה יחסית)
            # -----------------------------
            if mesh.mask is not None:
                mask = mesh.mask
                if mask.dtype != np.uint8:
                    mask = (mask > 0).astype(np.uint8) * 255
                elif mask.max() <= 1:
                    mask = (mask > 0).astype(np.uint8) * 255

                if mask.ndim == 3:
                    mask = mask.squeeze()

                if mask.shape[:2] != frame.shape[:2]:
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

                mb = mask > 0
                if mb.any():
                    has_mask = True
                    mask_overlay[mb] = self.mask_color

            # -----------------------------
            # 1) הקרנה ל-2D + עומק
            # -----------------------------
            verts = mesh.vertices
            points_2d = verts[:, :2].astype(np.int32)  # (V,2)
            z_values = verts[:, 2].astype(np.float32)

            z_min = float(z_values.min())
            z_max = float(z_values.max())
            depth_range = (z_max - z_min) if (z_max - z_min) != 0 else 1.0

            # -----------------------------
            # 2) דגימת faces אם יש יותר מדי
            # -----------------------------
            faces_all = mesh.faces
            F = len(faces_all)
            if F > max_faces_to_draw:
                step = int(np.ceil(F / max_faces_to_draw))
                faces = faces_all[::step]
            else:
                faces = faces_all

            # משולשים ב-2D
            all_triangles = points_2d[faces]  # (F',3,2)

            # -----------------------------
            # 3) ציור משולשים עם הצללה
            # -----------------------------
            # שים לב: face_idx כאן מתייחס ל-faces המדוגמים.
            # הפסים יהיו "בערך" כמו קודם, לא 1:1 לכל mesh, אבל נראה טוב ובמהיר.
            for face_idx, face in enumerate(faces):
                tri_pts = all_triangles[face_idx].reshape(-1, 1, 2)

                stripe_color = self.base_color if ((face_idx // 2) % 2 == 0) else self.highlight_color

                avg_depth = float(np.mean(z_values[face]))
                depth_norm = (avg_depth - z_min) / depth_range
                shade = 0.6 + 0.4 * (1.0 - depth_norm)

                shaded_color = (
                    int(stripe_color[0] * shade),
                    int(stripe_color[1] * shade),
                    int(stripe_color[2] * shade),
                )

                cv2.fillPoly(overlay, [tri_pts], shaded_color)

            # -----------------------------
            # 4) Mesh preview (רק מדי פעם)
            # -----------------------------
            if mesh_preview is None and (self._preview_counter % preview_every == 0):
                mesh_preview = self._render_mesh_preview(mesh)

        if draw_count == 0:
            return frame, None

        # -----------------------------
        # 5) Blend mask overlay אם יש
        # -----------------------------
        if has_mask:
            overlay = cv2.addWeighted(mask_overlay, self.mask_alpha, overlay, 1.0 - self.mask_alpha, 0)

        # -----------------------------
        # 6) Alpha blend עם הפריים המקורי
        # -----------------------------
        blended = cv2.addWeighted(overlay, self.alpha, frame, 1.0 - self.alpha, 0)

        # -----------------------------
        # 7) קו אדום חד להצגת סיבוב (אחרי ה-blend)
        # -----------------------------
        for mesh in mesh_objects:
            if mesh is None or getattr(mesh, "bbox_xyxy", None) is None:
                continue

            x1, y1, x2, y2 = mesh.bbox_xyxy
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            radius = max(1, int((x2 - x1) / 2))

            # הגנה: אם אין pose/rotation
            angle = 0.0
            if hasattr(mesh, "pose") and hasattr(mesh.pose, "rotation"):
                try:
                    angle = float(mesh.pose.rotation[2]) # Use Z rotation (roll)
                except Exception:
                    angle = 0.0

            # Draw a nicer arrow
            arrow_len = int(radius * 1.2)
            end_x = int(cx + arrow_len * np.cos(angle - np.pi/2)) # -pi/2 to point up at 0 angle
            end_y = int(cy + arrow_len * np.sin(angle - np.pi/2))

            # Draw main line
            cv2.arrowedLine(blended, (cx, cy), (end_x, end_y), (0, 0, 255), 3, tipLength=0.2)
            
            # Draw a small arc to indicate rotation direction
            if abs(angle) > 0.1:
                axes = (int(radius * 0.5), int(radius * 0.5))
                start_angle = -90
                end_angle = start_angle + np.degrees(angle)
                # Ensure we draw the arc in the correct direction
                if end_angle < start_angle:
                    start_angle, end_angle = end_angle, start_angle
                
                cv2.ellipse(blended, (cx, cy), axes, 0, start_angle, end_angle, (0, 255, 255), 2)

            cv2.circle(blended, (cx, cy), 5, (0, 255, 0), -1) # Green center dot

        # -----------------------------
        # 8) Embed preview אם נוצר
        # -----------------------------
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
