import cv2
import numpy as np


class ObjectTracker:
    """
    Tracks detections between consecutive frames using KLT optical flow.

    Key improvements vs. translation-only:
    - Fits an affine-partial transform (translation + rotation + uniform scale) using RANSAC.
    - Uses inlier ratio as tracking confidence (instead of st.mean()).
    - Updates bbox by warping its 4 corners with the estimated transform.
    - Warps the full-frame mask using the same affine matrix (keeps mask aligned).
    - Optional: when a mask exists, detect features only inside the mask region (reduces background drift).
    """

    def __init__(
        self,
        max_corners=80,
        quality_level=0.01,
        min_distance=5,
        lk_win=(21, 21),
        lk_max_level=3,
        # Robust motion params
        ransac_thresh=3.0,
        ransac_confidence=0.99,
        ransac_max_iters=2000,
        ransac_refine_iters=10,
        min_points=10,
        min_inliers=8,
        min_inlier_ratio=0.5,
        # If True and detection has mask, use it to constrain GFTT features to object pixels
        use_mask_for_features=True,
    ):
        self.max_corners = int(max_corners)
        self.quality_level = float(quality_level)
        self.min_distance = float(min_distance)

        self.lk_params = dict(
            winSize=tuple(lk_win),
            maxLevel=int(lk_max_level),
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        self.ransac_thresh = float(ransac_thresh)
        self.ransac_confidence = float(ransac_confidence)
        self.ransac_max_iters = int(ransac_max_iters)
        self.ransac_refine_iters = int(ransac_refine_iters)

        self.min_points = int(min_points)
        self.min_inliers = int(min_inliers)
        self.min_inlier_ratio = float(min_inlier_ratio)

        self.use_mask_for_features = bool(use_mask_for_features)

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        if frame is None:
            raise ValueError("frame is None")
        if frame.ndim == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    @staticmethod
    def _clip_bbox(b, w: int, h: int):
        x1, y1, x2, y2 = map(float, b)
        x1 = int(np.clip(x1, 0, w - 1))
        y1 = int(np.clip(y1, 0, h - 1))
        x2 = int(np.clip(x2, 0, w - 1))
        y2 = int(np.clip(y2, 0, h - 1))
        if x2 <= x1 or y2 <= y1:
            return None
        return [x1, y1, x2, y2]

    @staticmethod
    def _ensure_u8_mask(m: np.ndarray) -> np.ndarray:
        """
        Convert mask to uint8 with values 0 or 255, suitable for OpenCV masking.
        Accepts bool/0-1/0-255/float masks.
        """
        if m is None:
            return None
        m = np.asarray(m)
        if m.ndim > 2:
            m = np.squeeze(m)
        if m.dtype == np.bool_:
            return (m.astype(np.uint8) * 255)
        if m.dtype != np.uint8:
            # Normalize conservatively: treat >0 as foreground
            return ((m > 0).astype(np.uint8) * 255)
        # uint8 already: ensure binary-ish
        if m.max() <= 1:
            return (m.astype(np.uint8) * 255)
        return m

    @staticmethod
    def _warp_mask_affine(mask: np.ndarray, M: np.ndarray):
        if mask is None or M is None:
            return mask

        m = np.asarray(mask)
        if m.ndim > 2:
            m = np.squeeze(m)

        h, w = m.shape[:2]
        if h == 0 or w == 0:
            return None

        warped = cv2.warpAffine(
            m,
            M.astype(np.float32),
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return warped.astype(mask.dtype, copy=False)

    @staticmethod
    def _translate_mask(mask: np.ndarray, dx: float, dy: float):
        """Fallback: shift full-frame mask by translation only."""
        if mask is None:
            return None

        m = np.asarray(mask)
        if m.ndim > 2:
            m = np.squeeze(m)

        h, w = m.shape[:2]
        if h == 0 or w == 0:
            return None

        translated = cv2.warpAffine(
            m,
            np.float32([[1, 0, dx], [0, 1, dy]]),
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return translated.astype(mask.dtype, copy=False)

    @staticmethod
    def _bbox_from_warped_corners(bbox, M, w, h):
        x1, y1, x2, y2 = bbox
        corners = np.float32(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        ).reshape(-1, 1, 2)

        warped = cv2.transform(corners, M.astype(np.float32))
        xs = warped[:, 0, 0]
        ys = warped[:, 0, 1]
        new_bbox = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]

        # Clip to frame bounds
        x1 = int(np.clip(new_bbox[0], 0, w - 1))
        y1 = int(np.clip(new_bbox[1], 0, h - 1))
        x2 = int(np.clip(new_bbox[2], 0, w - 1))
        y2 = int(np.clip(new_bbox[3], 0, h - 1))
        if x2 <= x1 or y2 <= y1:
            return None
        return [x1, y1, x2, y2]

    # -------------------------------------------------------------------------
    # Core tracking
    # -------------------------------------------------------------------------

    def _track_bbox(self, old_gray, new_gray, det):
        """
        Track one detection dict:
        det must have "bbox".
        det may have "mask" (full-frame) to constrain feature selection.
        """
        h, w = old_gray.shape[:2]
        bbox = det.get("bbox", None)
        if bbox is None:
            return None

        bbox = self._clip_bbox(bbox, w, h)
        if bbox is None:
            return None

        x1, y1, x2, y2 = bbox
        roi = old_gray[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        # Optional: constrain features to mask region to avoid background drift
        roi_mask = None
        if self.use_mask_for_features and det.get("mask", None) is not None:
            full_m = self._ensure_u8_mask(det["mask"])
            if full_m is not None and full_m.shape[0] == h and full_m.shape[1] == w:
                roi_mask = full_m[y1:y2, x1:x2]
                # If mask too empty, ignore it
                if roi_mask is not None and cv2.countNonZero(roi_mask) < 50:
                    roi_mask = None

        pts = cv2.goodFeaturesToTrack(
            roi,
            mask=roi_mask,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
        )

        if pts is None or len(pts) < self.min_points:
            return None

        # Convert ROI points to full-frame coordinates
        pts = pts.reshape(-1, 2) + np.array([x1, y1], dtype=np.float32)
        pts = pts.reshape(-1, 1, 2).astype(np.float32)

        new_pts, st, _ = cv2.calcOpticalFlowPyrLK(
            old_gray, new_gray, pts, None, **self.lk_params
        )
        if new_pts is None or st is None:
            return None

        st = st.reshape(-1).astype(bool)
        if st.sum() < self.min_points:
            return None

        old_good = pts[st].reshape(-1, 2)
        new_good = new_pts[st].reshape(-1, 2)

        # Robust affine-partial fit (translation + rotation + uniform scale)
        M, inliers = cv2.estimateAffinePartial2D(
            old_good,
            new_good,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_thresh,
            maxIters=self.ransac_max_iters,
            confidence=self.ransac_confidence,
            refineIters=self.ransac_refine_iters,
        )

        if M is not None and inliers is not None:
            inliers = inliers.reshape(-1).astype(bool)
            inlier_count = int(inliers.sum())
            inlier_ratio = float(inlier_count / max(1, inliers.size))

            # Enforce minimum inliers and ratio
            if inlier_count < self.min_inliers or inlier_ratio < self.min_inlier_ratio:
                M = None
        else:
            inlier_ratio = 0.0

        if M is None:
            # Fallback: translation-only via median flow
            flow = new_good - old_good
            dx, dy = np.median(flow, axis=0)
            M = np.float32([[1, 0, float(dx)], [0, 1, float(dy)]])
            # Weak confidence (still better than nothing)
            inlier_ratio = float(st.mean())
        else:
            dx = float(M[0, 2])
            dy = float(M[1, 2])

        # Update bbox using warped corners (works for affine)
        new_bbox = self._bbox_from_warped_corners(bbox, M, w, h)
        if new_bbox is None:
            return None

        return {
            "bbox": new_bbox,
            "dx": float(dx),
            "dy": float(dy),
            "M": M.astype(np.float32),
            "tracking_confidence": float(inlier_ratio),
        }

    def track(self, old_frame, new_frame, detections, conf_threshold=0.25):
        """
        detections: list of dicts, each should include:
          - bbox: [x1,y1,x2,y2]
          - confidence: float (optional)
          - mask: full-frame mask (optional) shape (H,W)
        Returns updated detections with:
          - bbox updated
          - confidence updated
          - dx, dy
          - tracking_confidence
          - mask warped (if existed)
        """
        old_gray = self._to_gray(old_frame)
        new_gray = self._to_gray(new_frame)

        tracked = []
        for det in detections:
            bbox = det.get("bbox")
            if bbox is None:
                continue

            res = self._track_bbox(old_gray, new_gray, det)
            if res is None:
                continue

            det_conf = float(det.get("confidence", 1.0))
            track_conf = float(res.get("tracking_confidence", 0.0))

            if track_conf * det_conf < conf_threshold:
                continue

            new_det = dict(det)
            new_det.update(res)

            # Combine detector confidence with tracker confidence
            new_det["confidence"] = det_conf * track_conf

            # Warp mask with same transform (best), fallback to translation if needed
            if "mask" in new_det and new_det["mask"] is not None:
                M = new_det.get("M", None)
                if M is not None:
                    new_det["mask"] = self._warp_mask_affine(new_det["mask"], M)
                else:
                    new_det["mask"] = self._translate_mask(
                        new_det["mask"], new_det["dx"], new_det["dy"]
                    )

            tracked.append(new_det)

        return tracked
