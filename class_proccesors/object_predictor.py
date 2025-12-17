from ultralytics import YOLO


class ObjectPredictor:
    def __init__(self, model_path="yolov8n-seg.pt"):
        self.model = YOLO(model_path)
        self.names = self.model.names

    @staticmethod
    def _area(b):
        x1, y1, x2, y2 = b
        return max(0, x2 - x1) * max(0, y2 - y1)

    @staticmethod
    def _iou(a, b):
        xA = max(a[0], b[0])
        yA = max(a[1], b[1])
        xB = min(a[2], b[2])
        yB = min(a[3], b[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        areaA = (a[2] - a[0]) * (a[3] - a[1])
        areaB = (b[2] - b[0]) * (b[3] - b[1])
        union = areaA + areaB - inter
        return inter / union if union > 0 else 0.0

    def predict(self, frame, conf_threshold=0.3, iou_threshold=0.5):
        results = self.model(frame, conf=conf_threshold, verbose=False)

        dets = []
        for r in results:
            if r.boxes is None:
                continue
            for i, box in enumerate(r.boxes):
                cls = int(box.cls.item())
                dets.append(
                    {
                        "bbox": box.xyxy.tolist()[0],
                        "confidence": float(box.conf.item()),
                        "class": cls,
                        "class_name": self.names[cls],
                        "mask": (
                            r.masks.data[i].cpu().numpy()
                            if r.masks is not None
                            else None
                        ),
                    }
                )

        # --- suppress overlaps by confidence ---
        dets.sort(key=lambda d: d["confidence"], reverse=True)
        kept = []
        for d in dets:
            if any(
                self._iou(d["bbox"], k["bbox"]) >= iou_threshold for k in kept
            ):
                continue
            kept.append(d)

        if not kept:
            return []

        # --- keep only largest ---
        best = max(kept, key=lambda d: self._area(d["bbox"]))
        return [best]
