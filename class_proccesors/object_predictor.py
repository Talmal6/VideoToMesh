import numpy as np
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

    @staticmethod
    def _center_distance(bbox, img_w, img_h):
        """Compute normalized Euclidean distance from bbox center to image center [0,1]."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        img_cx = img_w / 2
        img_cy = img_h / 2

        dist = np.sqrt((cx - img_cx) ** 2 + (cy - img_cy) ** 2)
        max_dist = np.sqrt(img_cx ** 2 + img_cy ** 2)
        return dist / max_dist if max_dist > 0 else 0.0

    @staticmethod
    def _touches_border(bbox, img_w, img_h, margin=5):
        """
        Check if bbox touches or nearly touches image borders.
        Returns a penalty in [0,1]: 0 = far from all borders, 1 = touching all borders.
        """
        x1, y1, x2, y2 = bbox

        touches = 0
        if x1 <= margin:
            touches += 1
        if y1 <= margin:
            touches += 1
        if x2 >= img_w - margin:
            touches += 1
        if y2 >= img_h - margin:
            touches += 1

        return touches / 4.0

    @staticmethod
    def _normalized_area(bbox, img_w, img_h):
        """Return bbox area as fraction of frame area [0,1]."""
        x1, y1, x2, y2 = bbox
        bbox_area = max(0, x2 - x1) * max(0, y2 - y1)
        frame_area = img_w * img_h
        return bbox_area / frame_area if frame_area > 0 else 0.0

    def predict(self, frame, conf_threshold=0.3, iou_threshold=0.5):
        h, w = frame.shape[:2]
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

        # --- select foreground object (centered, small, not touching borders) ---
        def score(d):
            bbox = d["bbox"]
            conf = d["confidence"]

            # Center proximity: 1 = perfectly centered, 0 = corner
            center_dist = self._center_distance(bbox, w, h)
            center_score = 1.0 - center_dist

            # Area penalty: prefer smaller objects (suppress tables/backgrounds)
            norm_area = self._normalized_area(bbox, w, h)
            area_penalty = 1.0 - norm_area  # 1 = tiny, 0 = fills frame

            # Border penalty: objects touching borders are likely background
            border_touch = self._touches_border(bbox, w, h, margin=8)
            border_score = 1.0 - border_touch  # 1 = far from borders, 0 = touching all

            # Weighted combination:
            # - Center proximity is most important (foreground objects are centered)
            # - Area penalty suppresses large background objects
            # - Border penalty suppresses tables that span the frame
            # - Confidence is included but doesn't dominate
            return (
                0.70 * center_score
                + 0.15 * area_penalty
                + 0.10 * border_score
                + 0.05 * conf
            )

        best = max(kept, key=score)
        return best
