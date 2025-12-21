import cv2
import numpy as np
from typing import List, Tuple
from mesh.mesh_shapes.mesh_object import MeshObject

class Renderer:
    def __init__(self, 
                 base_color=(0, 255, 255),      # צהוב (ברירת מחדל)
                 highlight_color=(0, 165, 255), # כתום (צבע שני לפסים)
                 alpha=0.6):
        """
        :param base_color: הצבע הראשי של הגליל (BGR)
        :param highlight_color: הצבע השני ליצירת אפקט הפסים
        :param alpha: שקיפות (0.0 - 1.0)
        """
        self.base_color = base_color
        self.highlight_color = highlight_color
        self.alpha = alpha

    def render_frame(self, frame: np.ndarray, mesh_objects: List[MeshObject]) -> np.ndarray:
        if not mesh_objects:
            return frame

        overlay = frame.copy()
        draw_count = 0

        for mesh in mesh_objects:
            if (mesh.vertices is None or len(mesh.vertices) == 0 or 
                mesh.faces is None or len(mesh.faces) == 0):
                continue
            
            draw_count += 1

            # 1. המרת קודקודים ל-2D (זריקת ציר Z)
            points_2d = mesh.vertices[:, :2].astype(np.int32)

            # 2. שליפת כל המשולשים
            # shape: (Num_Triangles, 3, 2)
            all_triangles = points_2d[mesh.faces]

            # 3. יצירת אפקט הפסים (Stripes Effect)
            # כל מלבן אנכי בגליל מורכב מ-2 משולשים עוקבים (אינדקסים 0,1 ואז 2,3 וכו').
            # אנחנו רוצים לצבוע מלבנים זוגיים בצבע אחד, ואי-זוגיים בצבע שני.
            
            # קבוצה A (פסים זוגיים): לוקחים את משולשים 0,1, 4,5, 8,9...
            # slice [0::4] לוקח את 0, 4, 8...
            # slice [1::4] לוקח את 1, 5, 9...
            group_a = np.vstack((all_triangles[0::4], all_triangles[1::4]))
            
            # קבוצה B (פסים אי-זוגיים): לוקחים את משולשים 2,3, 6,7, 10,11...
            group_b = np.vstack((all_triangles[2::4], all_triangles[3::4]))

            # 4. ציור מהיר
            if len(group_a) > 0:
                cv2.fillPoly(overlay, pts=group_a, color=self.base_color)
            
            if len(group_b) > 0:
                cv2.fillPoly(overlay, pts=group_b, color=self.highlight_color)

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
            
            # Draw the line on the overlay (or directly on frame if we want it sharp)
            # Drawing on overlay to be affected by alpha, or maybe on top?
            # Let's draw on top (after alpha blending) to make it visible.
            # But the loop does alpha blending at the end.
            # So I'll draw it on 'overlay' with a distinct color.
            
            cv2.line(overlay, (cx, cy), (end_x, end_y), (0, 0, 255), 3) # Red line
            cv2.circle(overlay, (cx, cy), 4, (0, 0, 255), -1)

            # (אופציונלי) מסגרת דקה לכל משולש לחידוד נוסף
            # cv2.polylines(overlay, pts=all_triangles, isClosed=True, color=(0, 0, 0), thickness=1)

        if draw_count == 0:
            return frame

        # 5. מיזוג שקיפות
        return cv2.addWeighted(overlay, self.alpha, frame, 1 - self.alpha, 0)