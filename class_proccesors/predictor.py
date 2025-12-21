from class_proccesors.detection import Detection
from class_proccesors.object_analyzer import ObjectAnalyzer
from class_proccesors.object_tracker import ObjectTracker
import copy


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
            tracked_objects = self.tracker.track(last_frame, frame, [self.last_detection], conf_threshold=conf_threshold)
            detection = tracked_objects[0] if tracked_objects else None
            
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