import cv2
import numpy as np
import os
from typing import Optional, List

# --- Imports (מותאם למבנה התיקיות שלך) ---
from class_proccesors.object_analyzer import ObjectAnalyzer
from class_proccesors.detection import Detection
from class_proccesors.predictor import Predictor
from mesh.mesh_proccesors.cylinder_handler import CylinderHandler
from helpers.renderer import Renderer 

class Processor:
    def __init__(self, predictor: ObjectAnalyzer):
        self.predictor = predictor
        
        # 1. Handler Setup
        # מכיוון שה-Handler שלנו חכם ושומר State, אנחנו יוצרים מופע יחיד שלו כאן.
        self.handler = CylinderHandler(
            labels=("bottle", "cup", "can"), 
            sides=24,           # רזולוציית העיגול
            y_step=5,           # דיוק סריקת הגובה
            rotation_sensitivity=0.03
        )
        
        # 2. Renderer Setup
        self.renderer = Renderer(base_color=(0, 255, 255), alpha=0.4)

    def proccess_path(self, source):
        """
        פונקציית עזר לפתיחת מקור הווידאו (כפי שביקשת)
        """
        if isinstance(source, str):
            if not os.path.exists(source):
                # בדיקה אם המחרוזת היא מספר (למשל "0")
                if source.isdigit(): 
                    source = int(source)
                else: 
                    raise FileNotFoundError(f"Video file not found: {source}")
            else:
                cap = cv2.VideoCapture(source)
                if not cap.isOpened(): 
                    raise RuntimeError(f"Failed to open video file: {source}")
                return cap

        if isinstance(source, int) or source is None:
            if source is None: source = 0
            cap = cv2.VideoCapture(source)
            if cap.isOpened(): 
                # אופטימיזציה למצלמות רשת (Buffer נמוך = פחות דיליי)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return cap
            raise RuntimeError("No working camera found.")
        
        raise TypeError(f"Source error: {source}")

    def _bootstrap(self, frame, conf_threshold, frame_idx) -> List:
        """ 
        מבצע את כל השרשרת: Predict -> Detect -> Mesh Process 
        """
        

        best_result = self.predictor.predict(frame, conf_threshold=conf_threshold)
        
        mesh_objects = []

        if best_result:

            det = Detection(
                object_id=0, 
                label=best_result["class_name"],
                confidence=best_result["confidence"],
                frame_index=frame_idx,
                bbox_xyxy=tuple(best_result["bbox"]),
                mask=best_result["mask"]
            )

            # 3. שליחה ל-Handler
            # ה-Handler זוכר את הפריים הקודם ומחשב את הסיבוב
            if self.handler.can_handle(det):
                mesh = self.handler.process(det, frame)
                if mesh:
                    mesh_objects.append(mesh)

        return mesh_objects

    def run(self, source=0, conf_threshold=0.3):
        # אתחול המצלמה/וידאו
        try:
            cap = self.proccess_path(source)
        except Exception as e:
            print(f"Error initializing source: {e}")
            return

        frame_idx = 0
        print(f"Starting Processor on source: {source}...")
        print("Press 'q' or 'ESC' to exit.")

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("End of stream or error reading frame.")
                break

            # --- הלב של התוכנית: _bootstrap ---
            try:
                mesh_objects = self._bootstrap(frame, conf_threshold, frame_idx)
            except Exception as e:
                print(f"Error in _bootstrap: {e}")
                import traceback
                traceback.print_exc()
                break

            # --- תצוגה ---
            vis_frame, mesh_preview = self.renderer.render_frame(frame, mesh_objects)

            cv2.imshow("Mesh AR Tracker", vis_frame)
            if mesh_preview is not None:
                cv2.imshow("Mesh AR Tracker - Side View", mesh_preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

            frame_idx += 1

        cap.release()
        cv2.destroyAllWindows()
        print("Processing finished.")

# --- Main Entry Point ---
def main():


    predictor = Predictor()
    
  
    p = Processor(predictor)


    source = "./data/bottle_vid.mp4"
    
    p.run(source, conf_threshold=0.4)

if __name__ == "__main__":
    main()