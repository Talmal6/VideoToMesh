from detection.detection import Detection
from detection.object_analyzer import ObjectAnalyzer
from detection.object_tracker import ObjectTracker
import copy
from dataclasses import asdict
import logging

logger = logging.getLogger(__name__)


class Predictor:
    def __init__(self, use_analyzer_every_n_frames=3, analyzer = ObjectAnalyzer(model_path="yolov8n-seg.pt")):
        self.analyzer = analyzer
        self.tracker = ObjectTracker()
        self.last_detection: Detection = None
        self.use_analyzer_every_n_frames = use_analyzer_every_n_frames
        self.frames_since_analyzer = 0

    def predict(self, frame, conf_threshold=0.2):
        self.frames_since_analyzer += 1
        
        if self.frames_since_analyzer > self.use_analyzer_every_n_frames or self.last_detection is None:
            logger.info("Running analyzer prediction.")
            detection = self.analyzer.predict(frame, conf_threshold=conf_threshold)
            self.frames_since_analyzer = 0
        else:
            logger.info("Running tracker.")
            last_frame = self.last_detection.frame if hasattr(self.last_detection, "frame") else self.last_detection["frame"]
            
            # Convert Detection object to dict for tracker
            if hasattr(self.last_detection, "bbox_xyxy"):
                det_dict = asdict(self.last_detection)
                det_dict["bbox"] = det_dict["bbox_xyxy"]
            else:
                det_dict = self.last_detection

            tracked_objects = self.tracker.track(last_frame, frame, [det_dict], conf_threshold=0)
            
            if tracked_objects:
                res_dict = tracked_objects[0]
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
                logger.info("Tracking failed, falling back to analyzer.")
                detection = self.analyzer.predict(frame, conf_threshold=conf_threshold)
                self.frames_since_analyzer = 0
        
        if detection:
            logger.info(f"Detection found: {detection.label} ({detection.confidence:.2f})")
            self.last_detection = detection
            # support Detection objects with .frame attribute or dicts with 'frame' key
            if hasattr(self.last_detection, "frame"):
                self.last_detection.frame = copy.deepcopy(frame)
            elif isinstance(self.last_detection, dict):
                self.last_detection["frame"] = copy.deepcopy(frame)
        else:
            logger.info("No detection found.")
            self.last_detection = None
            
        return detection

    def close(self) -> None:
        """Release heavy resources to avoid shutdown-time crashes."""
        try:
            model = getattr(self.analyzer, "model", None)
            if model is not None:
                del model
                self.analyzer.model = None
        except Exception:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception:
            pass
