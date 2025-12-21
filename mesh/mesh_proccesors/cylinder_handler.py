from __future__ import annotations
from typing import Tuple, List, Optional
import cv2
import numpy as np

# הנחת מבנה Imports
from class_proccesors.detection import Detection
from mesh.mesh_proccesors.mesh_handler import Handler
from mesh.mesh_shapes.mesh_object import MeshObject


class CylinderHandler(Handler):
    def __init__(self,
                 labels=("bottle", "cup", "can"),
                 sides=24,
                 y_step=5,
                 rotation_sensitivity=0.005):
        self.labels = labels
        self.sides = sides
        self.y_step = y_step
        self.rotation_sensitivity = rotation_sensitivity

        # --- ניהול זיכרון (State) ---
        self.current_mesh: Optional[MeshObject] = None
        self.prev_gray_frame: Optional[np.ndarray] = None  # שומרים רק את האפור (חוסך זיכרון)
        self.last_rotation_delta = 0.0  # For smoothing

    def can_handle(self, det: Detection) -> bool:
        return det.label in self.labels

    def create_object(self, det: Detection) -> MeshObject:
        """יצירה ראשונית"""
        mesh = MeshObject(
            object_id=det.object_id,
            label=det.label,
            confidence=det.confidence,
            frame_index=det.frame_index,
            bbox_xyxy=det.bbox_xyxy,
            mask=det.mask
        )
        mesh.vertical_base = 0.0
        return mesh

    def process(self, det: Detection, curr_frame: np.ndarray) -> MeshObject:
        """
        הפונקציה הראשית.
        מקבלת: זיהוי נוכחי + פריים נוכחי.
        מנהלת עצמאית את ההשוואה לפריים הקודם.
        """
        if curr_frame is None:
            return self.current_mesh

        # המרה לאפור כבר בהתחלה (נשתמש בזה גם לחישוב וגם לשמירה לפעם הבאה)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        rotation_delta = 0.0

        # 1. בדיקה: האם יש לנו היסטוריה להשוות אליה?
        # צריך גם mesh קודם וגם פריים קודם
        if self.current_mesh is not None and self.prev_gray_frame is not None:

            # שליפת נתונים מההיסטוריה
            prev_bbox = self.current_mesh.bbox_xyxy
            prev_mask = self.current_mesh.mask

            # חישוב הסיבוב (שולחים את האפורים ישירות)
            raw_rotation_delta = self._calculate_rotation_delta(
                curr_gray, self.prev_gray_frame,
                det.bbox_xyxy, prev_bbox,
                prev_mask
            )

            # --- שיפור: החלקה (Smoothing) ---
            alpha = 0.2
            rotation_delta = (alpha * raw_rotation_delta) + ((1 - alpha) * self.last_rotation_delta)

            # --- שיפור: Deadband ---
            if abs(rotation_delta) < 0.002:
                rotation_delta = 0.0

            self.last_rotation_delta = rotation_delta

            # עדכון האובייקט הקיים
            self.current_mesh = self._update_mesh_state(self.current_mesh, det, rotation_delta)

        else:
            # פעם ראשונה שאנחנו רואים את האובייקט (או אחרי איפוס)
            self.current_mesh = self.create_object(det)

        # 2. בניית הגיאומטריה (תמיד מתבצעת לפי המסכה הנוכחית)
        self._generate_geometry(self.current_mesh, det.mask, det.bbox_xyxy)

        # 3. שמירת הפריים הנוכחי לפעם הבאה (עדכון ה-State)
        self.prev_gray_frame = curr_gray.copy()

        return self.current_mesh

    def _calculate_rotation_delta(self, curr_gray, prev_gray,
                                    curr_bbox, last_bbox,
                                    last_mask: np.ndarray = None) -> float:
        """
        גרסה משופרת: כוללת כיווץ מסכה (Erosion) וסינון וקטורים אנכיים
        כדי לתפוס סיבוב בצורה נקייה יותר כשמסתובבים סביב האובייקט.
        בנוסף, מחזקת את הרמז האופקי לפי פיצ'רים שנעלמים ומחליפים כיוון.
        """
        try:
            x1, y1, x2, y2 = map(int, last_bbox)

            # בדיקות גבולות
            h, w = prev_gray.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                return 0.0

            # --- שיפור 1: הכנת מסכה מחמירה ---
            if last_mask is not None:
                flow_mask = last_mask.astype(np.uint8).copy()
                if flow_mask.max() > 1:
                    flow_mask[flow_mask > 0] = 255

                kernel_size = int((x2 - x1) * 0.1)  # 10% מרוחב הבקבוק
                if kernel_size > 0:
                    kernel = np.ones((kernel_size, kernel_size), np.uint8)
                    flow_mask = cv2.erode(flow_mask, kernel, iterations=1)
            else:
                flow_mask = np.zeros_like(prev_gray)
                margin = int((x2 - x1) * 0.3)  # חיתוך אגרסיבי של 30% מכל צד
                flow_mask[y1:y2, x1 + margin:x2 - margin] = 255

            # 1. מציאת נקודות ("פיצ'רים") בפריים הקודם
            p0 = cv2.goodFeaturesToTrack(prev_gray, mask=flow_mask, maxCorners=120,
                                        qualityLevel=0.01, minDistance=4)

            if p0 is None or len(p0) < 4:
                return 0.0

            # 2. מעקב (Optical Flow)
            p1, status, err = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, p0, None,
                winSize=(21, 21),  # חלון גדול יותר תופס תנועה מהירה יותר
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )

            # סינון נקודות שלא נמצאו
            good_new = p1[status == 1]
            good_old = p0[status == 1]

            if len(good_new) < 4:
                return 0.0

            # --- שיפור 2: סינון וקטורים רעשניים ---
            valid_diffs_x = []

            for new, old in zip(good_new, good_old):
                dx = new[0] - old[0]
                dy = new[1] - old[1]

                # אם התנועה האנכית גדולה מהאופקית - זה כנראה רעש מצלמה ולא סיבוב גליל
                if abs(dy) > abs(dx) and abs(dy) > 2.0:
                    continue

                valid_diffs_x.append(dx)

            if not valid_diffs_x:
                return 0.0

            valid_diffs_x = np.array(valid_diffs_x)

            # --- שיפור 3: חישוב חכם (IQR Filtering) ---
            # זריקת ערכי קיצון (Outliers) שנובעים מהשתקפויות שזזות הפוך
            q75, q25 = np.percentile(valid_diffs_x, [75, 25])
            iqr = q75 - q25

            if iqr > 0:
                lower_bound = q25 - (1.5 * iqr)
                upper_bound = q75 + (1.5 * iqr)
                clean_diffs = valid_diffs_x[(valid_diffs_x >= lower_bound) & (valid_diffs_x <= upper_bound)]
            else:
                clean_diffs = valid_diffs_x

            if len(clean_diffs) == 0:
                return 0.0

            # חישוב תזוזת הטקסטורה נטו
            texture_dx = np.mean(clean_diffs)  # ממוצע על הערכים הנקיים

            # תזוזת האובייקט (מרכז ה-bbox)
            curr_center_x = (curr_bbox[0] + curr_bbox[2]) / 2
            last_center_x = (last_bbox[0] + last_bbox[2]) / 2
            object_dx = curr_center_x - last_center_x

            # בדיקת יחס פיצ'רים שנעלמו לטובת פיצ'רים חדשים (חיזוק סיבוב בצדדים)
            lost_ratio = 1.0 - (len(good_new) / len(p0))

            curr_candidates = cv2.goodFeaturesToTrack(curr_gray, mask=flow_mask, maxCorners=80,
                                                     qualityLevel=0.01, minDistance=4)
            centroid_shift = 0.0
            if curr_candidates is not None and len(curr_candidates) >= 4:
                prev_mean_x = np.mean(p0[:, 0, 0])
                curr_mean_x = np.mean(curr_candidates[:, 0, 0])
                bbox_width = max(1.0, (x2 - x1))
                centroid_shift = np.clip((curr_mean_x - prev_mean_x) / bbox_width, -1.0, 1.0)

            # החישוב הסופי
            net_rotation_px = texture_dx - object_dx
            churn_boost = 1.0 + (0.6 * lost_ratio) + (0.4 * centroid_shift)

            # החזרת התוצאה עם הרגישות שהוגדרה
            return net_rotation_px * churn_boost * self.rotation_sensitivity * 1.5
        except Exception as e:
            # print(f"Rotation calc error: {e}")
            return 0.0

    def _update_mesh_state(self, prev_mesh: MeshObject, det: Detection, rotation_delta: float) -> MeshObject:
        """עדכון נתונים יבשים"""
        new_mesh = prev_mesh.copy()

        new_mesh.bbox_xyxy = det.bbox_xyxy
        new_mesh.mask = det.mask
        new_mesh.frame_index = det.frame_index
        new_mesh.confidence = det.confidence

        if not hasattr(new_mesh, 'vertical_base'):
            new_mesh.vertical_base = 0.0

        new_mesh.vertical_base += rotation_delta
        # Keep the pose rotation in sync so renderers can visualize horizontal rotation
        new_mesh.apply_rotation(rotation_delta)
        return new_mesh

    def _generate_geometry(self, mesh: MeshObject, mask: np.ndarray, bbox: Tuple[float, float, float, float]) -> None:
        """בניית גיאומטריית הגליל לפי מסכה וזווית"""
        x1, y1, x2, y2 = map(int, bbox)
        h_img, w_img = mask.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)

        if x2 <= x1 or y2 <= y1:
            return

        obj_mask = mask[y1:y2, x1:x2]
        vertices: List[List[float]] = []
        valid_rows_indices: List[int] = []

        # סריקת המסכה (Contour Scanning)
        for row_y in range(0, obj_mask.shape[0], self.y_step):
            row_data = obj_mask[row_y, :]
            white_pixels = np.where(row_data > 0)[0]

            if len(white_pixels) == 0:
                continue

            min_x, max_x = white_pixels[0], white_pixels[-1]
            width = max_x - min_x
            if width < 2:
                continue

            radius = width / 2.0
            local_center_x = min_x + radius + x1
            global_y = row_y + y1

            current_ring_start_idx = len(vertices)
            valid_rows_indices.append(current_ring_start_idx)

            base_angle = mesh.vertical_base

            for i in range(self.sides):
                theta = base_angle + (2 * np.pi * i / self.sides)
                vx = local_center_x + radius * np.cos(theta)
                vz = radius * np.sin(theta)
                vy = global_y
                vertices.append([vx, vy, vz])

        mesh.vertices = np.array(vertices)

        # בניית המשולשים (Faces)
        faces = []
        num_rings = len(valid_rows_indices)

        for r in range(num_rings - 1):
            ring_start = valid_rows_indices[r]
            next_ring_start = valid_rows_indices[r + 1]

            for i in range(self.sides):
                curr_idx = ring_start + i
                curr_next = ring_start + ((i + 1) % self.sides)
                bottom_idx = next_ring_start + i
                bottom_next = next_ring_start + ((i + 1) % self.sides)

                faces.append([curr_idx, curr_next, bottom_idx])
                faces.append([curr_next, bottom_next, bottom_idx])

        mesh.faces = np.array(faces)