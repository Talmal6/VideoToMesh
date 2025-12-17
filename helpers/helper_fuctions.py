import math
import cv2
import numpy as np


def _clip_bbox_to_frame(bbox, w, h):
    if bbox is None or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = bbox
    x1 = int(np.clip(round(x1), 0, w - 1))
    y1 = int(np.clip(round(y1), 0, h - 1))
    x2 = int(np.clip(round(x2), 0, w - 1))
    y2 = int(np.clip(round(y2), 0, h - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def plot_closeups(frame, detections, tile_size=256, max_cols=3, window_name="closeups"):
    """Render a grid of close-up crops for every detection, labeled with class/conf."""
    h, w = frame.shape[:2]
    tiles = []

    for det in detections or []:
        bbox = det.get("bbox") or det.get("bbox_xyxy")
        clipped = _clip_bbox_to_frame(bbox, w, h)
        if clipped is None:
            continue

        x1, y1, x2, y2 = clipped
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        tile = cv2.resize(crop, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)
        label = det.get("class_name", "object")
        conf = float(det.get("confidence", 0.0))
        cv2.putText(
            tile,
            f"{label} {conf:.2f}",
            (8, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        tiles.append(tile)

    if not tiles:
        blank = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
        cv2.putText(
            blank,
            "no detections",
            (10, tile_size // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(window_name, blank)
        cv2.waitKey(1)
        return

    num = len(tiles)
    cols = min(max_cols, num)
    rows = int(math.ceil(num / cols))
    grid = np.zeros((rows * tile_size, cols * tile_size, 3), dtype=np.uint8)

    for idx, tile in enumerate(tiles):
        r = idx // cols
        c = idx % cols
        y0 = r * tile_size
        x0 = c * tile_size
        grid[y0 : y0 + tile_size, x0 : x0 + tile_size] = tile

    cv2.imshow(window_name, grid)
    cv2.waitKey(1)


def draw_detections(frame, detections):
    for d in detections or []:
        if not isinstance(d, dict):
            continue
        if "bbox" not in d:
            continue

        x1, y1, x2, y2 = map(int, d["bbox"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{d['class_name']} {d['confidence']:.2f}"
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

def draw_shape_fit(frame, fit, label_prefix=""):
    if not fit:
        return

    center = fit.get("center")
    obb = fit.get("obb") or {}
    if center is None or not obb:
        return

    cx, cy = [int(round(v)) for v in center]
    angle = float(fit.get("angle_deg", 0.0))
    w = float(obb.get("width", 0.0))
    h = float(obb.get("height", 0.0))
    if w <= 0 or h <= 0:
        return

    # center
    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    # axis (long)
    theta = np.deg2rad(angle)
    dx = int(np.cos(theta) * (h * 0.5))
    dy = int(np.sin(theta) * (h * 0.5))
    cv2.line(frame, (cx - dx, cy - dy), (cx + dx, cy + dy), (0, 0, 255), 2)

    # oriented box
    rect = ((cx, cy), (w, h), angle)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    cv2.polylines(frame, [box], True, (255, 255, 0), 2)

    # text
    shape = fit.get("shape", "shape")
    quality = fit.get("quality", 0.0)
    dims = fit.get("dimensions", {})
    if shape == "cylinder":
        dims_txt = f"r={dims.get('radius', 0.0):.0f} h={dims.get('height', 0.0):.0f}"
    else:
        dims_txt = (
            f"w={dims.get('width', 0.0):.0f} "
            f"h={dims.get('height', 0.0):.0f} "
            f"d={dims.get('depth', 0.0):.0f}"
        )
    txt = (
        f"{label_prefix}{shape} a={angle:.1f} {dims_txt} q={quality:.2f}"
    )
    cv2.putText(frame, txt, (cx + 10, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)




def draw_mask_overlay(frame, mask_u8, alpha: float = 0.35):
    """
    Draw segmentation mask overlay on the frame.
    mask_u8: HxW uint8 (0/255) or bool/0-1.
    """
    if mask_u8 is None:
        return

    if mask_u8.dtype != np.uint8:
        mask_u8 = mask_u8.astype(np.uint8)

    # normalize to 0/255
    if mask_u8.max() <= 1:
        mask_u8 = mask_u8 * 255

    h, w = frame.shape[:2]
    if mask_u8.shape[:2] != (h, w):
        mask_u8 = cv2.resize(mask_u8, (w, h), interpolation=cv2.INTER_NEAREST)

    # green overlay
    overlay = frame.copy()
    overlay[mask_u8 > 0] = (0, 255, 0)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # contour on top
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cv2.drawContours(frame, cnts, -1, (0, 255, 255), 2)
