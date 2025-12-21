from class_proccesors.detection import Detection
from class_proccesors.object_analyzer import ObjectAnalyzer
from class_proccesors.object_tracker import ObjectTracker
import copy
from dataclasses import asdict


class Predictor:
    def __init__(self):
        
        self.analyzer = ObjectAnalyzer(model_path="yolov8n-seg.pt")
        self.tracker = ObjectTracker()
        self.last_detection : Detection = None

    def predict(self, frame, conf_threshold=0.2):
        if self.last_detection is None:
            detection = self.analyzer.predict(frame, conf_threshold=conf_threshold)
        else:
            last_frame = self.last_detection.frame if hasattr(self.last_detection, "frame") else self.last_detection["frame"]
            
            # Convert Detection object to dict for tracker
            if hasattr(self.last_detection, "bbox_xyxy"):
                det_dict = asdict(self.last_detection)
                det_dict["bbox"] = det_dict["bbox_xyxy"]
            else:
                det_dict = self.last_detection

            tracked_objects = self.tracker.track(last_frame, frame, [det_dict], conf_threshold=conf_threshold)
            
            if tracked_objects:
                res_dict = tracked_objects[0]
                # Convert back to Detection object
                detection = Detection(
                    object_id=res_dict["object_id"],
                    label=res_dict["label"],
                    confidence=res_dict["confidence"],
                    frame_index=res_dict["frame_index"],
                    bbox_xyxy=tuple(res_dict["bbox"]),
                    mask=res_dict["mask"],
                    frame=res_dict.get("frame")
                )
            else:
                detection = None
            
            if detection is None:
                detection = self.analyzer.predict(frame, conf_threshold=conf_threshold)
        
        if detection:
            self.last_detection = detection
            # support Detection objects with .frame attribute or dicts with 'frame' key
            if hasattr(self.last_detection, "frame"):
                self.last_detection.frame = copy.deepcopy(frame)
            elif isinstance(self.last_detection, dict):
                self.last_detection["frame"] = copy.deepcopy(frame)
        else:
            self.last_detection = None
            
        return detection