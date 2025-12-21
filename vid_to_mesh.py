import cv2
import numpy as np
import os
from typing import Optional, List

# --- Imports (מותאם למבנה התיקיות שלך) ---
from class_proccesors.object_analyzer import ObjectAnalyzer
from class_proccesors.detection import Detection
from class_proccesors.predictor import Predictor
from mesh.mesh_manager import MeshManager
from mesh.mesh_proccesors.cylinder_handler import CylinderHandler 
from helpers.renderer import Renderer 

class VidToMesh:
    def __init__(self, predictor: ObjectAnalyzer):
        self.predictor = predictor
        
        self.handler = CylinderHandler()
        self.mesh_manager = MeshManager(handlers=[self.handler])
    
        self.renderer = Renderer(base_color=(0, 255, 255), alpha=0.4)

    def process_path(self, source):

        if isinstance(source, str):
            if not os.path.exists(source):
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
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return cap
            raise RuntimeError("No working camera found.")
        
        raise TypeError(f"Source error: {source}")

    def _bootstrap(self, frame, conf_threshold, frame_idx) -> List:
        detection = self.predictor.predict(frame, conf_threshold=conf_threshold)
        
        if detection:
            mesh = self.mesh_manager.get_mesh(detection, frame)
            if mesh and mesh.vertices is not None and len(mesh.vertices) > 0:
                return mesh
        return None

        

    def run(self, source=0, conf_threshold=0.3):
        # אתחול המצלמה/וידאו
        try:
            cap = self.process_path(source)
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
                mesh_object = self._bootstrap(frame, conf_threshold, frame_idx)
            except Exception as e:
                print(f"Error in _bootstrap: {e}")
                import traceback
                traceback.print_exc()
                break

            # --- תצוגה ---
            vis_frame, mesh_preview = self.renderer.render_frame(frame, mesh_object)

            cv2.imshow("Mesh AR Tracker", vis_frame)
            if mesh_preview is not None:
                cv2.imshow("Mesh AR Tracker - Side View", mesh_preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            
            obj_count = 1 if mesh_object else 0
            print(f"Processed frame {frame_idx}, objects: {obj_count}")

            frame_idx += 1

        cap.release()
        cv2.destroyAllWindows()
        print("Processing finished.")

# --- Main Entry Point ---
def main():


    predictor = Predictor()
    
  
    p = VidToMesh(predictor)


    source = "./data/napoleon_vid.mp4"
    
    p.run(source, conf_threshold=0.4)

if __name__ == "__main__":
    main()