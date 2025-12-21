
from typing import Optional, Tuple

import cv2
from matplotlib.pylab import Any
import numpy as np


def is_cylindrical(class_name: str) -> bool:
    name = (class_name or "").lower()
    return any(kw in name for kw in ("bottle", "cup", "can"))

def to_u8_mask(mask: Any, frame_shape: Tuple[int, ...]) -> Optional[np.ndarray]:
    if mask is None:
        return None
    m = np.asarray(mask)
    if m.ndim > 2:
        m = np.squeeze(m)
    if m.ndim != 2:
        return None

    if m.dtype == bool:
        m = m.astype(np.uint8) * 255
    elif m.max() <= 1.0:
        m = (m > 0.5).astype(np.uint8) * 255
    else:
        m = (m > 127).astype(np.uint8) * 255

    fh, fw = frame_shape[:2]
    if m.shape[:2] != (fh, fw):
        m = cv2.resize(m, (fw, fh), interpolation=cv2.INTER_NEAREST)
    return m

def clip_bbox(bbox: Any, w: int, h: int) -> Optional[Tuple[int, int, int, int]]:
    if bbox is None or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = map(float, bbox)
    x1 = int(np.clip(np.floor(x1), 0, w - 1))
    y1 = int(np.clip(np.floor(y1), 0, h - 1))
    x2 = int(np.clip(np.ceil(x2), 0, w - 1))
    y2 = int(np.clip(np.ceil(y2), 0, h - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2

def _wrap_angle(angle: float) -> float:
    while angle > 90:
        angle -= 180
    while angle <= -90:
        angle += 180
    return angle

def _largest_contour(mask_u8: np.ndarray) -> Optional[np.ndarray]:
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)



def _roi_from_bbox(mask_u8: np.ndarray, bbox: Tuple[int, int, int, int],
                   frame_shape: Tuple[int, ...], pad_frac: float = 0.08
                   ) -> Tuple[Optional[np.ndarray], Tuple[int, int, int, int]]:
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    pad = int(round(pad_frac * max(bw, bh)))
    x1p = max(0, x1 - pad)
    y1p = max(0, y1 - pad)
    x2p = min(w, x2 + pad)
    y2p = min(h, y2 + pad)
    if x2p <= x1p or y2p <= y1p:
        return None, (0, 0, 0, 0)
    roi = mask_u8[y1p:y2p, x1p:x2p].copy()
    return roi, (x1p, y1p, x2p, y2p)

def _clean_mask(roi_u8: np.ndarray) -> np.ndarray:
    H, W = roi_u8.shape[:2]
    k = max(3, int(round(min(H, W) * 0.03)) | 1)
    k = min(k, 15)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    out = cv2.morphologyEx(roi_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel, iterations=1)
    return out

def _order_corners_stable(pts4: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts4, np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = pts[:, 0] - pts[:, 1]
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmax(d)]
    bl = pts[np.argmin(d)]
    return np.stack([tl, tr, br, bl], axis=0).astype(np.float32)

