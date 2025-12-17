"""
MeshFactory: Computes ALL geometry and 2D render primitives.
The renderer uses ONLY the output MeshData - no additional computation.

Dependencies: cv2, numpy, dataclasses, typing only.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any


# =============================================================================
# MeshData: All pre-computed data needed for rendering
# =============================================================================

@dataclass
class MeshData:
    """
    Contains all pre-computed geometry and render primitives.
    Renderer uses ONLY these fields - no additional computation.
    """
    # --- 3D Mesh ---
    vertices: np.ndarray  # shape (V, 3) float32
    faces: np.ndarray     # shape (F, 3) int32

    # --- 2D Render Primitives (pre-computed for cv2 drawing) ---
    poly2d: np.ndarray           # shape (N, 1, 2) int32 - silhouette contour in frame coords
    rings2d: np.ndarray          # shape (K, 5) int32 - [cx, cy, ax, ay, angle_deg]
    centerline: Tuple[int, int, int, int]  # (x1, y1, x2, y2)

    # --- Box-specific 2D primitives ---
    box_front2d: Optional[np.ndarray] = None  # shape (4, 2) int32 - front face corners [TL, TR, BR, BL]
    box_back2d: Optional[np.ndarray] = None   # shape (4, 2) int32 - back face corners
    box_edges2d: Optional[np.ndarray] = None  # shape (12, 2, 2) int32 - all 12 edges as line segments

    # --- Style ---
    color: Tuple[int, int, int] = (0, 255, 0)
    outline_black_thick: int = 6
    outline_thick: int = 3
    ring_thick: int = 2
    centerline_thick: int = 2

    # --- Metadata ---
    shape_type: str = "cylinder"  # "cylinder" or "box"
    class_name: str = ""          # detected class name from YOLO
    angle_deg: float = 0.0
    quality: float = 0.0

    # --- Tracking (updated by MeshTracker) ---
    dx: float = 0.0
    dy: float = 0.0
    scale: float = 1.0
    tracking_confidence: float = 1.0
    confidence: float = 1.0


# =============================================================================
# Helper functions
# =============================================================================

def is_cylindrical(class_name: str) -> bool:
    """Return True if class_name indicates a cylindrical object."""
    name = (class_name or "").lower()
    return any(kw in name for kw in ("bottle", "cup", "can"))


def to_u8_mask(mask: Any, frame_shape: Tuple[int, ...]) -> Optional[np.ndarray]:
    """Convert mask to uint8 0/255, resized to frame dimensions."""
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
    """Clip bbox to frame bounds."""
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
    """Wrap angle to [-90, 90]."""
    while angle > 90:
        angle -= 180
    while angle <= -90:
        angle += 180
    return angle


def _largest_contour(mask_u8: np.ndarray) -> Optional[np.ndarray]:
    """Find largest external contour."""
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)


# =============================================================================
# Box Corner Extraction Helpers (Stage A + Stage B pipeline)
# =============================================================================

def _roi_from_bbox(mask_u8: np.ndarray, bbox: Tuple[int, int, int, int], 
                   frame_shape: Tuple[int, ...], pad_frac: float = 0.08
                   ) -> Tuple[Optional[np.ndarray], Tuple[int, int, int, int]]:
    """Extract ROI from mask with padding around bbox."""
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
    """Clean ROI mask with morphological operations (CLOSE then OPEN)."""
    H, W = roi_u8.shape[:2]
    k = max(3, int(round(min(H, W) * 0.03)) | 1)  # odd kernel size
    k = min(k, 15)  # cap for realtime
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    out = cv2.morphologyEx(roi_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel, iterations=1)
    return out


def _order_corners_stable(pts4: np.ndarray) -> np.ndarray:
    """
    Order 4 corners as TL, TR, BR, BL using sum/diff method (more stable).
    pts4: shape (4, 2)
    Returns: (4, 2) float32
    """
    pts = np.asarray(pts4, np.float32).reshape(4, 2)
    s = pts.sum(axis=1)          # x + y
    d = pts[:, 0] - pts[:, 1]    # x - y
    tl = pts[np.argmin(s)]       # min sum = top-left
    br = pts[np.argmax(s)]       # max sum = bottom-right
    tr = pts[np.argmax(d)]       # max diff = top-right
    bl = pts[np.argmin(d)]       # min diff = bottom-left
    return np.stack([tl, tr, br, bl], axis=0).astype(np.float32)


def _corners_minarearect(roi_mask_u8: np.ndarray, roi_offset_xy: Tuple[int, int]
                         ) -> Optional[np.ndarray]:
    """
    Stage A corners: hull + minAreaRect (always available, stable).
    Returns corners in full-frame coords or None.
    """
    cnt = _largest_contour(roi_mask_u8)
    if cnt is None or cv2.contourArea(cnt) < 200:
        return None
    hull = cv2.convexHull(cnt)
    rect = cv2.minAreaRect(hull)
    pts = cv2.boxPoints(rect)  # (4, 2) float in ROI coords
    pts = _order_corners_stable(pts)
    xoff, yoff = roi_offset_xy
    pts[:, 0] += xoff
    pts[:, 1] += yoff
    return pts  # float32 full-frame coords


# --- Stage B: LSD line detection refinement ---

def _line_angle_deg(line: np.ndarray) -> float:
    """Compute line angle in [0, 180) degrees."""
    x1, y1, x2, y2 = line
    ang = np.degrees(np.arctan2((y2 - y1), (x2 - x1)))
    return ang % 180.0


def _lsd_lines(gray_roi: np.ndarray, obj_mask_roi_u8: np.ndarray) -> np.ndarray:
    """Detect line segments using LSD, constrained to object mask."""
    edges = cv2.Canny(gray_roi, 60, 160, L2gradient=True)
    edges = cv2.bitwise_and(edges, obj_mask_roi_u8)  # keep inside object
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines = lsd.detect(edges)[0]  # (N, 1, 4) or None
    if lines is None:
        return np.zeros((0, 4), np.float32)
    return lines.reshape(-1, 4).astype(np.float32)  # x1, y1, x2, y2 in ROI coords


def _filter_lines(lines: np.ndarray, roi_w: int, roi_h: int) -> np.ndarray:
    """Filter lines by minimum length."""
    if len(lines) == 0:
        return lines
    min_len = 0.25 * min(roi_w, roi_h)
    out = []
    for l in lines:
        x1, y1, x2, y2 = l
        L = np.hypot(x2 - x1, y2 - y1)
        if L >= min_len:
            out.append(l)
    return np.array(out, np.float32) if out else np.zeros((0, 4), np.float32)


def _select_two_directions(lines: np.ndarray) -> Optional[Tuple[int, int]]:
    """Select two dominant angle bins (roughly perpendicular)."""
    if len(lines) < 4:
        return None
    angs = np.array([_line_angle_deg(l) for l in lines], np.float32)
    bins = (angs / 10.0).astype(np.int32)  # 10-degree bins
    uniq, cnt = np.unique(bins, return_counts=True)
    order = np.argsort(cnt)[::-1]
    if len(order) < 2:
        return None
    b1 = int(uniq[order[0]])
    # Pick second bin sufficiently different (~30 degrees)
    b2 = None
    for idx in order[1:]:
        cand = int(uniq[idx])
        diff = min(abs(cand - b1), 18 - abs(cand - b1))  # wrap around at 180
        if diff >= 3:  # 3 bins ~ 30 degrees
            b2 = cand
            break
    if b2 is None:
        return None
    return b1, b2


def _line_to_normal_form(line: np.ndarray) -> Tuple[float, float, float]:
    """Convert line to normal form (nx, ny, d)."""
    x1, y1, x2, y2 = line
    vx, vy = x2 - x1, y2 - y1
    # Normal (perpendicular)
    nx, ny = -vy, vx
    nlen = np.hypot(nx, ny) + 1e-6
    nx, ny = nx / nlen, ny / nlen
    # Signed distance of point to origin along normal
    d = nx * x1 + ny * y1
    return float(nx), float(ny), float(d)


def _pick_two_extremes(lines_bin: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Pick two extreme lines (opposite sides) from a direction bin."""
    if len(lines_bin) < 2:
        return None
    ds = np.array([_line_to_normal_form(l)[2] for l in lines_bin], np.float32)
    i_min = int(np.argmin(ds))
    i_max = int(np.argmax(ds))
    if i_min == i_max:
        return None
    return lines_bin[i_min], lines_bin[i_max]


def _intersect(l1: np.ndarray, l2: np.ndarray) -> Optional[np.ndarray]:
    """Compute intersection of two lines (as infinite lines)."""
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-6:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
    return np.array([px, py], np.float32)


def _validate_corners(corners: np.ndarray, roi_w: int, roi_h: int, margin: float = 0.1
                      ) -> bool:
    """Validate that corners form a reasonable quadrilateral inside ROI."""
    if corners is None or len(corners) != 4:
        return False
    # Check all finite
    if not np.all(np.isfinite(corners)):
        return False
    # Check inside ROI with margin
    m = margin * min(roi_w, roi_h)
    if np.any(corners[:, 0] < -m) or np.any(corners[:, 0] > roi_w + m):
        return False
    if np.any(corners[:, 1] < -m) or np.any(corners[:, 1] > roi_h + m):
        return False
    # Check reasonable area (not too small or too large)
    area = cv2.contourArea(corners.reshape(-1, 1, 2).astype(np.float32))
    roi_area = roi_w * roi_h
    if area < 0.05 * roi_area or area > 0.95 * roi_area:
        return False
    return True


def _refine_corners_lsd(gray_roi: np.ndarray, mask_roi_u8: np.ndarray
                        ) -> Optional[np.ndarray]:
    """
    Stage B refinement: LSD lines -> 2 directions -> 4 intersections.
    Returns 4 corners in ROI coords or None if refinement fails.
    """
    roi_h, roi_w = gray_roi.shape[:2]
    
    # Detect and filter lines
    lines = _lsd_lines(gray_roi, mask_roi_u8)
    lines = _filter_lines(lines, roi_w, roi_h)
    if len(lines) < 4:
        return None
    
    # Select two dominant directions
    dirs = _select_two_directions(lines)
    if dirs is None:
        return None
    b1, b2 = dirs
    
    # Get angles and assign lines to bins
    angs = np.array([_line_angle_deg(l) for l in lines], np.float32)
    bins = (angs / 10.0).astype(np.int32)
    
    lines_b1 = lines[bins == b1]
    lines_b2 = lines[bins == b2]
    
    # Pick two extreme lines per direction
    ext1 = _pick_two_extremes(lines_b1)
    ext2 = _pick_two_extremes(lines_b2)
    if ext1 is None or ext2 is None:
        return None
    
    a1, a2 = ext1  # direction 1 extremes
    b1_line, b2_line = ext2  # direction 2 extremes
    
    # Compute 4 intersections
    c1 = _intersect(a1, b1_line)
    c2 = _intersect(a1, b2_line)
    c3 = _intersect(a2, b2_line)
    c4 = _intersect(a2, b1_line)
    
    if any(c is None for c in [c1, c2, c3, c4]):
        return None
    
    corners = np.array([c1, c2, c3, c4], np.float32)
    
    # Validate
    if not _validate_corners(corners, roi_w, roi_h):
        return None
    
    return _order_corners_stable(corners)


# =============================================================================
# MeshFactory: Computes ALL geometry and render primitives
# =============================================================================

class MeshFactory:
    """
    Factory that builds MeshData from detection.
    ALL computations happen here - renderer does zero computation.
    """

    def __init__(
        self,
        ring_step: int = 12,
        smooth_ksize: int = 9,
        top_bottom_margin: float = 0.05,
        minor_scale: float = 0.25,
        cylinder_segments: int = 32,
        box_ema_alpha: float = 0.5,
    ):
        self.ring_step = ring_step
        self.smooth_ksize = smooth_ksize
        self.top_bottom_margin = top_bottom_margin
        self.minor_scale = minor_scale
        self.cylinder_segments = cylinder_segments
        self.box_ema_alpha = box_ema_alpha
        # Temporal smoothing state for box corners
        self._prev_box_corners: Optional[np.ndarray] = None
        self._prev_bbox: Optional[Tuple[int, int, int, int]] = None

    def build(
        self,
        det: Dict[str, Any],
        frame_shape: Tuple[int, ...],
        color: Tuple[int, int, int] = (0, 255, 0),
        frame_bgr: Optional[np.ndarray] = None,
    ) -> Optional[MeshData]:
        """
        Build MeshData from detection.
        Routes to shape-specific builders based on class_name.
        Returns None if geometry cannot be computed.
        
        Args:
            det: Detection dict with 'mask', 'bbox', 'class_name'
            frame_shape: (H, W, C) shape of the frame
            color: RGB color for rendering
            frame_bgr: Optional BGR frame for Stage B LSD refinement
        """
        class_name = det.get("class_name", "")
        mask = det.get("mask")
        bbox = det.get("bbox")

        if mask is None or bbox is None:
            return None

        # Prepare grayscale for box refinement
        frame_gray = None
        if frame_bgr is not None:
            if frame_bgr.ndim == 3:
                frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            else:
                frame_gray = frame_bgr

        # Dispatch by shape type
        if is_cylindrical(class_name):
            return self._build_cylinder(mask, bbox, frame_shape, color, class_name)
        else:
            # Box/rectangular shapes
            return self._build_box(mask, bbox, frame_shape, color, class_name, frame_gray)

    def _build_cylinder(
        self,
        mask: Any,
        bbox: Any,
        frame_shape: Tuple[int, ...],
        color: Tuple[int, int, int],
        class_name: str = "",
    ) -> Optional[MeshData]:
        """Build cylinder mesh and render primitives from mask."""
        fh, fw = frame_shape[:2]

        # Convert mask to uint8 full-frame
        mask_u8 = to_u8_mask(mask, frame_shape)
        if mask_u8 is None:
            return None

        # Clip bbox
        clipped = clip_bbox(bbox, fw, fh)
        if clipped is None:
            return None

        x1, y1, x2, y2 = clipped

        # Get largest contour in full frame (for silhouette)
        poly2d = _largest_contour(mask_u8)
        if poly2d is None or cv2.contourArea(poly2d) < 100:
            return None

        # Extract ROI
        roi = mask_u8[y1:y2, x1:x2].copy()
        roi_h, roi_w = roi.shape[:2]

        if roi_h < 10 or roi_w < 5:
            return None

        # Find largest contour in ROI for geometry extraction
        roi_cnt = _largest_contour(roi)
        if roi_cnt is None or cv2.contourArea(roi_cnt) < 100:
            return None

        # Compute alignment angle
        rect = cv2.minAreaRect(roi_cnt)
        (_, _), (rw, rh), angle = rect
        if rw > rh:
            rw, rh = rh, rw
            angle += 90.0
        angle = _wrap_angle(angle)

        # Rotate ROI to align vertically
        center = (roi_w / 2.0, roi_h / 2.0)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned = cv2.warpAffine(
            roi, M, (roi_w, roi_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        # Build radius profile r(y)
        r = np.zeros(roi_h, dtype=np.float32)
        for y in range(roi_h):
            xs = np.where(aligned[y] > 0)[0]
            if len(xs) >= 2:
                r[y] = 0.5 * (xs[-1] - xs[0])

        # Zero out margins
        margin = int(self.top_bottom_margin * roi_h)
        if margin > 0:
            r[:margin] = 0
            r[-margin:] = 0

        # Smooth radius profile
        if self.smooth_ksize >= 3 and self.smooth_ksize % 2 == 1:
            r_col = r.reshape(-1, 1)
            r_smooth = cv2.GaussianBlur(r_col, (1, self.smooth_ksize), 0)
            r = r_smooth.flatten()

        # Compute inverse transform for mapping back
        A = np.vstack([M, [0, 0, 1]])
        A_inv = np.linalg.inv(A)

        # Select ring rows and pre-compute ring parameters
        ring_ys = [y for y in range(0, roi_h, self.ring_step) if r[y] > 2]

        rings2d_list = []
        for y_aligned in ring_ys:
            if y_aligned < 0 or y_aligned >= len(r):
                continue

            major = 2 * r[y_aligned]
            minor = self.minor_scale * major
            if major < 6:
                continue

            # Center in aligned coords
            cx_a = roi_w / 2.0
            cy_a = float(y_aligned)

            # Map back to original ROI coords
            p = A_inv @ np.array([cx_a, cy_a, 1.0])
            x_roi, y_roi = p[0], p[1]

            # Map to full-frame
            ex = int(round(x1 + x_roi))
            ey = int(round(y1 + y_roi))

            ax = int(round(major / 2))
            ay = int(round(minor / 2))

            if ax >= 1 and ay >= 1:
                rings2d_list.append([ex, ey, ax, ay, int(round(angle))])

        rings2d = np.array(rings2d_list, dtype=np.int32) if rings2d_list else np.zeros((0, 5), dtype=np.int32)

        # Centerline
        cx = (x1 + x2) // 2
        centerline = (cx, y1, cx, y2)

        # Generate 3D mesh (surface of revolution)
        vertices, faces = self._generate_cylinder_mesh(r, roi_h, self.cylinder_segments)

        # Compute quality metric
        nonzero = np.count_nonzero(r)
        quality = nonzero / max(len(r), 1)

        return MeshData(
            vertices=vertices,
            faces=faces,
            poly2d=poly2d,
            rings2d=rings2d,
            centerline=centerline,
            color=color,
            shape_type="cylinder",
            class_name=class_name,
            angle_deg=angle,
            quality=quality,
        )

    @staticmethod
    def _order_corners_tl_tr_br_bl(pts: np.ndarray) -> np.ndarray:
        """
        Order 4 corners as TL, TR, BR, BL (clockwise from top-left).
        pts: shape (4, 2)
        """
        pts = pts.reshape(4, 2).astype(np.float32)
        # Sort by y first (top 2 vs bottom 2)
        sorted_by_y = pts[np.argsort(pts[:, 1])]
        top_two = sorted_by_y[:2]
        bottom_two = sorted_by_y[2:]
        # Sort top by x: left, right
        top_two = top_two[np.argsort(top_two[:, 0])]
        # Sort bottom by x: left, right (but we want BR, BL order -> right, left)
        bottom_two = bottom_two[np.argsort(bottom_two[:, 0])[::-1]]
        # TL, TR, BR, BL
        return np.array([top_two[0], top_two[1], bottom_two[0], bottom_two[1]], dtype=np.int32)

    def _build_box(
        self,
        mask: Any,
        bbox: Any,
        frame_shape: Tuple[int, ...],
        color: Tuple[int, int, int],
        class_name: str = "",
        frame_gray: Optional[np.ndarray] = None,
    ) -> Optional[MeshData]:
        """
        Build box mesh and render primitives from mask using two-stage pipeline:
        - Stage A (always): ROI cleanup + convex hull + minAreaRect -> 4 corners
        - Stage B (optional): LSD line detection refinement for higher accuracy
        """
        fh, fw = frame_shape[:2]

        mask_u8 = to_u8_mask(mask, frame_shape)
        if mask_u8 is None:
            return None

        clipped = clip_bbox(bbox, fw, fh)
        if clipped is None:
            return None

        x1, y1, x2, y2 = clipped

        # 1) Find largest contour (for silhouette poly2d)
        poly2d = _largest_contour(mask_u8)
        if poly2d is None or cv2.contourArea(poly2d) < 100:
            return None

        # 2) Extract ROI with padding
        roi_mask, roi_box = _roi_from_bbox(mask_u8, clipped, frame_shape, pad_frac=0.08)
        if roi_mask is None:
            return None
        x1p, y1p, x2p, y2p = roi_box
        roi_offset = (x1p, y1p)

        # 3) Clean ROI mask
        clean_roi = _clean_mask(roi_mask)

        # 4) Stage A: Get stable corners from minAreaRect (always available)
        stage_a_corners = _corners_minarearect(clean_roi, roi_offset)
        if stage_a_corners is None:
            return None

        # 5) Stage B: Try LSD refinement if ROI is large enough
        quality = 0.7  # Stage A quality
        front_corners_float = stage_a_corners

        roi_h, roi_w = clean_roi.shape[:2]
        if roi_w >= 50 and roi_h >= 50 and frame_gray is not None:
            # Extract gray ROI for edge detection
            gray_roi = frame_gray[y1p:y2p, x1p:x2p]
            if gray_roi.shape[:2] == clean_roi.shape[:2]:
                refined = _refine_corners_lsd(gray_roi, clean_roi)
                if refined is not None:
                    # Convert from ROI coords to full-frame coords
                    refined[:, 0] += x1p
                    refined[:, 1] += y1p
                    front_corners_float = refined
                    quality = 1.0  # Stage B success

        # 6) Apply temporal smoothing (EMA) if previous corners exist
        if self._prev_box_corners is not None and self._prev_bbox is not None:
            # Check IoU with previous bbox to ensure same object
            prev_x1, prev_y1, prev_x2, prev_y2 = self._prev_bbox
            # Simple overlap check
            xi1 = max(x1, prev_x1)
            yi1 = max(y1, prev_y1)
            xi2 = min(x2, prev_x2)
            yi2 = min(y2, prev_y2)
            inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            area1 = (x2 - x1) * (y2 - y1)
            area2 = (prev_x2 - prev_x1) * (prev_y2 - prev_y1)
            union = area1 + area2 - inter
            iou = inter / union if union > 0 else 0

            if iou > 0.5:
                alpha = self.box_ema_alpha
                front_corners_float = alpha * front_corners_float + (1 - alpha) * self._prev_box_corners

        # Update state for next frame
        self._prev_box_corners = front_corners_float.copy()
        self._prev_bbox = clipped

        # 7) Order corners as TL, TR, BR, BL (stable ordering)
        front_corners = _order_corners_stable(front_corners_float).astype(np.int32)

        # 8) Compute back face offset based on top edge normal
        edge_tl_tr = front_corners[1].astype(float) - front_corners[0].astype(float)
        edge_len = np.linalg.norm(edge_tl_tr) + 1e-6
        # Normal to top edge (pointing "into" the scene)
        nx, ny = -edge_tl_tr[1] / edge_len, edge_tl_tr[0] / edge_len
        # Depth offset (~15% of diagonal)
        diag = np.linalg.norm(front_corners[2] - front_corners[0])
        depth_offset = diag * 0.15
        # Offset direction: use normal, but ensure it goes up-left-ish for 3D look
        if ny > 0:
            nx, ny = -nx, -ny
        offset_vec = np.array([nx * depth_offset, ny * depth_offset - depth_offset * 0.5], dtype=np.float32)
        back_corners = (front_corners.astype(float) + offset_vec).astype(np.int32)

        # Clip back corners to frame bounds
        back_corners[:, 0] = np.clip(back_corners[:, 0], 0, fw - 1)
        back_corners[:, 1] = np.clip(back_corners[:, 1], 0, fh - 1)

        # 9) Build 3D mesh
        w_box = float(np.linalg.norm(front_corners[1] - front_corners[0]))
        h_box = float(np.linalg.norm(front_corners[3] - front_corners[0]))
        d_box = float(depth_offset) * 2

        vertices, faces = self._generate_box_mesh(w_box, h_box, d_box)

        # 10) Pre-compute all 12 edges for rendering
        edges = []
        # Front face (4 edges)
        for i in range(4):
            j = (i + 1) % 4
            edges.append([front_corners[i], front_corners[j]])
        # Back face (4 edges)
        for i in range(4):
            j = (i + 1) % 4
            edges.append([back_corners[i], back_corners[j]])
        # Connecting edges (4 edges)
        for i in range(4):
            edges.append([front_corners[i], back_corners[i]])

        box_edges2d = np.array(edges, dtype=np.int32)

        # Centerline
        center_top = ((front_corners[0] + front_corners[1]) // 2).astype(int)
        center_bottom = ((front_corners[2] + front_corners[3]) // 2).astype(int)
        centerline = (int(center_top[0]), int(center_top[1]), int(center_bottom[0]), int(center_bottom[1]))

        # No rings for box
        rings2d = np.zeros((0, 5), dtype=np.int32)

        # Compute angle from front face orientation
        angle_deg = float(np.degrees(np.arctan2(edge_tl_tr[1], edge_tl_tr[0])))

        return MeshData(
            vertices=vertices,
            faces=faces,
            poly2d=poly2d,
            rings2d=rings2d,
            centerline=centerline,
            box_front2d=front_corners,
            box_back2d=back_corners,
            box_edges2d=box_edges2d,
            color=color,
            shape_type="box",
            class_name=class_name,
            angle_deg=angle_deg,
            quality=quality,
        )

    @staticmethod
    def _generate_cylinder_mesh(
        r: np.ndarray,
        height: int,
        segments: int = 32,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate surface-of-revolution mesh from radius profile.
        Returns (vertices, faces).
        """
        seg = max(3, segments)
        angles = np.linspace(0, 2 * np.pi, seg, endpoint=False)

        # Find valid radii range
        valid_ys = np.where(r > 0)[0]
        if len(valid_ys) == 0:
            # Fallback: simple cylinder with average radius
            avg_r = 10.0
            return MeshFactory._generate_simple_cylinder(avg_r, float(height), seg)

        y_min, y_max = valid_ys[0], valid_ys[-1]
        if y_max <= y_min:
            avg_r = float(r[r > 0].mean()) if np.any(r > 0) else 10.0
            return MeshFactory._generate_simple_cylinder(avg_r, float(height), seg)

        # Sample radii at regular intervals
        sample_ys = np.linspace(y_min, y_max, min(32, y_max - y_min + 1), dtype=int)
        sample_rs = r[sample_ys]

        # Generate vertices
        verts = []
        for i, (y, rad) in enumerate(zip(sample_ys, sample_rs)):
            if rad < 1:
                rad = 1  # minimum radius
            for a in angles:
                x = rad * np.cos(a)
                z = rad * np.sin(a)
                # y is along the cylinder axis
                verts.append([x, float(y), z])

        # Add top and bottom center vertices
        top_center_idx = len(verts)
        verts.append([0.0, float(sample_ys[0]), 0.0])
        bottom_center_idx = len(verts)
        verts.append([0.0, float(sample_ys[-1]), 0.0])

        vertices = np.array(verts, dtype=np.float32)

        # Generate faces
        faces = []
        n_rings = len(sample_ys)

        # Side faces
        for ring in range(n_rings - 1):
            for s in range(seg):
                s_next = (s + 1) % seg
                v0 = ring * seg + s
                v1 = ring * seg + s_next
                v2 = (ring + 1) * seg + s_next
                v3 = (ring + 1) * seg + s
                faces.append([v0, v1, v2])
                faces.append([v0, v2, v3])

        # Top cap
        for s in range(seg):
            s_next = (s + 1) % seg
            faces.append([top_center_idx, s_next, s])

        # Bottom cap
        last_ring_start = (n_rings - 1) * seg
        for s in range(seg):
            s_next = (s + 1) % seg
            faces.append([bottom_center_idx, last_ring_start + s, last_ring_start + s_next])

        faces_arr = np.array(faces, dtype=np.int32)
        return vertices, faces_arr

    @staticmethod
    def _generate_simple_cylinder(
        radius: float,
        height: float,
        segments: int = 32,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate simple cylinder mesh centered at origin."""
        seg = max(3, segments)
        angles = np.linspace(0, 2 * np.pi, seg, endpoint=False)

        x = radius * np.cos(angles)
        z = radius * np.sin(angles)

        bottom = np.stack([x, np.zeros_like(x), z], axis=1)
        top = np.stack([x, np.full_like(x, height), z], axis=1)

        verts = [bottom, top]
        verts.append(np.array([[0, 0, 0], [0, height, 0]], dtype=np.float32))

        vertices = np.concatenate(verts, axis=0).astype(np.float32)

        faces = []
        # Side faces
        for i in range(seg):
            j = (i + 1) % seg
            b0, b1 = i, j
            t0, t1 = i + seg, j + seg
            faces.append([b0, t0, t1])
            faces.append([b0, t1, b1])

        # Caps
        bottom_center = 2 * seg
        top_center = 2 * seg + 1
        for i in range(seg):
            j = (i + 1) % seg
            faces.append([bottom_center, j, i])
            faces.append([top_center, i + seg, j + seg])

        faces_arr = np.array(faces, dtype=np.int32)
        return vertices, faces_arr

    @staticmethod
    def _generate_box_mesh(
        width: float,
        height: float,
        depth: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate axis-aligned box mesh centered at origin."""
        w, h, d = width / 2, height / 2, depth / 2

        vertices = np.array([
            [-w, -h, -d],
            [+w, -h, -d],
            [+w, +h, -d],
            [-w, +h, -d],
            [-w, -h, +d],
            [+w, -h, +d],
            [+w, +h, +d],
            [-w, +h, +d],
        ], dtype=np.float32)

        faces = np.array([
            [0, 1, 2], [0, 2, 3],
            [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1],
            [1, 5, 6], [1, 6, 2],
            [2, 6, 7], [2, 7, 3],
            [3, 7, 4], [3, 4, 0],
        ], dtype=np.int32)

        return vertices, faces
