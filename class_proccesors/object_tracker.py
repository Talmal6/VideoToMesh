import cv2
import numpy as np


class ObjectTracker:
    def __init__(
        self,
        max_corners=50,
        quality_level=0.01,
        min_distance=5,
        lk_win=(21, 21),
        lk_max_level=3,
    ):
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance

        self.lk_params = dict(
            winSize=lk_win,
            maxLevel=lk_max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

    @staticmethod
    def _to_gray(frame):
        if frame.ndim == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    @staticmethod
    def _clip_bbox(b, w, h):
        x1, y1, x2, y2 = map(float, b)
        x1 = int(np.clip(x1, 0, w - 1))
        y1 = int(np.clip(y1, 0, h - 1))
        x2 = int(np.clip(x2, 0, w - 1))
        y2 = int(np.clip(y2, 0, h - 1))
        if x2 <= x1 or y2 <= y1:
            return None
        return [x1, y1, x2, y2]

    def _track_bbox(self, old_gray, new_gray, bbox):
        h, w = old_gray.shape[:2]
        bbox = self._clip_bbox(bbox, w, h)
        if bbox is None:
            return None

        x1, y1, x2, y2 = bbox
        roi = old_gray[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        pts = cv2.goodFeaturesToTrack(
            roi,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
        )

        if pts is None or len(pts) < 6:
            return None

        pts = pts.reshape(-1, 2) + np.array([x1, y1], dtype=np.float32)
        pts = pts.reshape(-1, 1, 2).astype(np.float32)

        new_pts, st, _ = cv2.calcOpticalFlowPyrLK(
            old_gray, new_gray, pts, None, **self.lk_params
        )
        if new_pts is None or st is None:
            return None

        st = st.reshape(-1).astype(bool)
        if st.sum() < 6:
            return None

        old_good = pts[st].reshape(-1, 2)
        new_good = new_pts[st].reshape(-1, 2)

        flow = new_good - old_good
        dx, dy = np.median(flow, axis=0)

        new_bbox = [x1 + dx, y1 + dy, x2 + dx, y2 + dy]
        new_bbox = self._clip_bbox(new_bbox, new_gray.shape[1], new_gray.shape[0])
        if new_bbox is None:
            return None

        return {
            "bbox": new_bbox,
            "dx": float(dx),
            "dy": float(dy),
            "tracking_confidence": float(st.mean()),
        }

    def track(self, old_frame, new_frame, detections, conf_threshold=0.25):
        old_gray = self._to_gray(old_frame)
        new_gray = self._to_gray(new_frame)

        tracked = []
        for det in detections:
            bbox = det.get("bbox")
            if bbox is None:
                continue

            res = self._track_bbox(old_gray, new_gray, bbox)
            if res is None:
                continue

            if res["tracking_confidence"] * det.get("confidence", 1.0) < conf_threshold:
                continue

            new_det = dict(det)
            new_det.update(res)
            new_det["confidence"] = (
                det.get("confidence", 1.0) * res["tracking_confidence"]
            )
            tracked.append(new_det)

        return tracked
