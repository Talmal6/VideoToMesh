import cv2
import numpy as np
from typing import Optional, Dict, Any, Tuple


class SymmetricShapeFitter:
    """Model-based fitter that estimates cylinder/box proxies from mask + bbox."""

    def __init__(
        self,
        min_contour_area: float = 80.0,
        min_component_area: float = 60.0,
        min_component_ratio: float = 0.01,
        depth_ratio: float = 0.6,
        morph_kernel: int = 3,
    ) -> None:
        self.min_contour_area = float(min_contour_area)
        self.min_component_area = float(min_component_area)
        self.min_component_ratio = float(min_component_ratio)
        self.depth_ratio = float(depth_ratio)
        self.morph_kernel = int(morph_kernel)

    @staticmethod
    def _to_uint8_mask(mask: Any) -> Optional[np.ndarray]:
        if mask is None:
            return None

        mask_arr = np.asarray(mask)
        if mask_arr.ndim > 2:
            mask_arr = np.squeeze(mask_arr)
        if mask_arr.ndim != 2:
            return None

        mask_u8 = (mask_arr > 0).astype(np.uint8) * 255
        return mask_u8

    @staticmethod
    def _clip_bbox(bbox: Any, width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
        if bbox is None or len(bbox) != 4 or width <= 1 or height <= 1:
            return None

        x1, y1, x2, y2 = map(float, bbox)
        x1 = int(np.clip(np.floor(x1), 0, width - 1))
        y1 = int(np.clip(np.floor(y1), 0, height - 1))
        x2 = int(np.clip(np.ceil(x2), 0, width - 1))
        y2 = int(np.clip(np.ceil(y2), 0, height - 1))

        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        angle = ((angle + 180.0) % 360.0) - 180.0
        if angle <= -90.0:
            angle += 180.0
        if angle > 90.0:
            angle -= 180.0
        return float(angle)

    def _clean_mask(self, mask_roi: np.ndarray, keep_area: float) -> Optional[np.ndarray]:
        if mask_roi is None or mask_roi.size == 0:
            return None

        mask_bin = (mask_roi > 0).astype(np.uint8) * 255

        if self.morph_kernel > 0:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (self.morph_kernel, self.morph_kernel)
            )
            mask_bin = cv2.erode(mask_bin, k, iterations=1)
            mask_bin = cv2.dilate(mask_bin, k, iterations=1)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask_bin, connectivity=8
        )
        if num_labels <= 1:
            return None

        filtered = np.zeros_like(mask_bin)
        for lbl in range(1, num_labels):
            area = stats[lbl, cv2.CC_STAT_AREA]
            if area >= keep_area:
                filtered[labels == lbl] = 255

        if not filtered.any():
            return None
        return filtered

    @staticmethod
    def _largest_contour(mask_u8: np.ndarray):
        cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        return max(cnts, key=cv2.contourArea)

    def _rect_from_contour(self, cnt):
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w, h), angle = rect
        if w > h:
            w, h = h, w
            angle += 90.0
        angle = self._wrap_angle(angle)
        w = max(float(w), 1e-3)
        h = max(float(h), 1e-3)
        return float(cx), float(cy), w, h, angle

    def fit(
        self,
        mask: Any,
        bbox: Any,
        shape_hint: str = "box",
        frame_shape: Optional[Tuple[int, int]] = None,
    ) -> Optional[Dict[str, Any]]:
        mask_u8 = self._to_uint8_mask(mask)
        if mask_u8 is None:
            return None

        if frame_shape is not None:
            fh, fw = frame_shape[:2]
            if mask_u8.shape[:2] != (fh, fw):
                mask_u8 = cv2.resize(mask_u8, (fw, fh), interpolation=cv2.INTER_NEAREST)

        h, w = mask_u8.shape[:2]
        clipped_bbox = self._clip_bbox(bbox, w, h)
        if clipped_bbox is None:
            return None

        x1, y1, x2, y2 = clipped_bbox
        mask_roi = mask_u8[y1:y2, x1:x2]
        if mask_roi.size == 0 or not mask_roi.any():
            return None

        bbox_area = max(1.0, float((x2 - x1) * (y2 - y1)))
        keep_area = max(self.min_component_area, bbox_area * self.min_component_ratio)
        cleaned = self._clean_mask(mask_roi, keep_area)
        if cleaned is None:
            return None

        cnt = self._largest_contour(cleaned)
        if cnt is None or len(cnt) < 5:
            return None

        contour_area = float(cv2.contourArea(cnt))
        if contour_area < self.min_contour_area:
            return None

        cnt = cnt + np.array([[[x1, y1]]], dtype=cnt.dtype)
        cx, cy, width, height, angle = self._rect_from_contour(cnt)

        quality = float(np.clip(contour_area / bbox_area, 0.0, 1.0))
        shape_type = "cylinder" if shape_hint == "cylinder" else "box"

        if shape_type == "cylinder":
            dims = {"radius": width * 0.5, "height": height}
        else:
            dims = {
                "width": width,
                "height": height,
                "depth": self.depth_ratio * width,
            }

        return {
            "shape": shape_type,
            "center": (cx, cy),
            "angle_deg": angle,
            "obb": {"width": width, "height": height},
            "dimensions": dims,
            "quality": quality,
            "contour_area": contour_area,
            "bbox": clipped_bbox,
        }
