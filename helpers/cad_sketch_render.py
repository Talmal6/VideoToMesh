"""
CAD-style sketch renderer.
ONLY draws using pre-computed MeshData - zero computations here.

Dependencies: cv2, numpy only.
"""

import cv2
import numpy as np

from mesh.mesh_shapes.mesh_object import MeshObject
from mesh.mesh_shapes.cylinder import Cylinder


def render_mesh_overlay(
    frame_bgr: np.ndarray,
    obj: MeshObject,
) -> np.ndarray:
    """
    Render mesh overlay onto frame.
    This function does ZERO computation - only cv2 draw calls using MeshObject fields/cache.
    Dispatches to cylinder or box rendering based on object type.
    """
    vis = frame_bgr.copy()

    if isinstance(obj, Cylinder):
        return _render_cylinder(vis, obj)
    else:
        # Fallback or box if implemented
        return _render_box(vis, obj)


def _render_cylinder(vis: np.ndarray, obj: Cylinder) -> np.ndarray:
    """Render cylinder: silhouette + (kink) rings + volume guide-lines + centerline."""

    # Extract sketch data from cache (precomputed by handler)
    poly2d = obj.cache.get("sketch_poly2d")
    centerline = obj.cache.get("sketch_centerline")
    rings2d = obj.cache.get("sketch_rings2d")
    guides2d = obj.cache.get("sketch_guides2d")  # List[np.ndarray] of polylines (Nx2)

    color = obj.color_bgr
    black = (0, 0, 0)

    outline_thick = 2
    outline_black_thick = 4

    # Make centerline subtle (it's a guide, not the main feature)
    centerline_thick = 2

    # Rings: draw black under-stroke then color stroke (CAD look)
    ring_thick = 2
    ring_black_thick = 4

    # Guide-lines: thin highlight rails for volume
    guide_thick = 1
    guide_black_thick = 3

    # --- Silhouette ---
    if poly2d is not None:
        # Black outline (thick)
        cv2.drawContours(vis, [poly2d], -1, black, outline_black_thick, cv2.LINE_AA)
        # Colored outline
        cv2.drawContours(vis, [poly2d], -1, color, outline_thick, cv2.LINE_AA)

    # --- Rings (ONLY where handler decided: top/bottom + kinks) ---
    if rings2d is not None:
        for ring in rings2d:
            cx, cy, ax, ay, angle_int = ring
            c = (int(round(cx)), int(round(cy)))
            a = (max(1, int(round(ax))), max(1, int(round(ay))))
            ang = int(round(angle_int))

            # Under-stroke (black)
            cv2.ellipse(vis, c, a, ang, 0, 360, black, ring_black_thick, cv2.LINE_AA)
            # Color stroke
            cv2.ellipse(vis, c, a, ang, 0, 360, color, ring_thick, cv2.LINE_AA)

    # --- Volume guide-lines (add depth) ---
    if guides2d is not None:
        for poly in guides2d:
            if poly is None or len(poly) < 2:
                continue
            # Under-stroke (black)
            cv2.polylines(vis, [poly], False, black, guide_black_thick, cv2.LINE_AA)
            # Color stroke (slightly darker than outline)
            guide_color = tuple(max(0, c - 30) for c in color)
            cv2.polylines(vis, [poly], False, guide_color, guide_thick, cv2.LINE_AA)

    # --- Centerline (last, subtle) ---
    if centerline is not None:
        x1, y1, x2, y2 = centerline
        darker_color = tuple(max(0, c - 70) for c in color)
        cv2.line(
            vis,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            darker_color,
            centerline_thick,
            cv2.LINE_AA,
        )

    return vis



def _render_box(vis: np.ndarray, obj: MeshObject) -> np.ndarray:
    """Render box: 3D wireframe with front face, back face, and connecting edges."""
    # Placeholder implementation or extract from cache if available
    # Assuming box_edges2d is in cache
    
    box_edges2d = obj.cache.get("sketch_box_edges2d")
    box_front2d = obj.cache.get("sketch_box_front2d")
    
    color = obj.color_bgr
    black = (0, 0, 0)
    thick = 2
    black_thick = 4

    # Draw all 12 edges from pre-computed box_edges2d
    if box_edges2d is not None and len(box_edges2d) > 0:
        edges = box_edges2d  # shape (12, 2, 2)

        # Draw black outlines first (all edges)
        for edge in edges:
            p1 = tuple(map(int, edge[0]))
            p2 = tuple(map(int, edge[1]))
            cv2.line(vis, p1, p2, black, black_thick, cv2.LINE_AA)

        # Draw colored edges
        # Front face edges (0-3): solid, full color
        for i in range(min(4, len(edges))):
            p1 = tuple(map(int, edges[i][0]))
            p2 = tuple(map(int, edges[i][1]))
            cv2.line(vis, p1, p2, color, thick, cv2.LINE_AA)

        # Back face edges (4-7): slightly darker (receding)
        darker = tuple(max(0, c - 40) for c in color)
        for i in range(4, min(8, len(edges))):
            p1 = tuple(map(int, edges[i][0]))
            p2 = tuple(map(int, edges[i][1]))
            cv2.line(vis, p1, p2, darker, thick, cv2.LINE_AA)

        # Connecting edges (8-11): dashed appearance via lighter color
        connect_color = tuple(max(0, c - 20) for c in color)
        for i in range(8, min(12, len(edges))):
            p1 = tuple(map(int, edges[i][0]))
            p2 = tuple(map(int, edges[i][1]))
            cv2.line(vis, p1, p2, connect_color, thick, cv2.LINE_AA)

    # Draw corner markers on front face
    if box_front2d is not None:
        for pt in box_front2d:
            cv2.circle(vis, tuple(map(int, pt)), 5, color, -1, cv2.LINE_AA)
            cv2.circle(vis, tuple(map(int, pt)), 5, black, 1, cv2.LINE_AA)

    return vis


def render_debug_overlay(
    vis: np.ndarray,
    obj: MeshObject,
) -> None:
    """Draw debug text: class name, shape type, angle, quality."""
    # Build info text
    class_str = obj.label if obj.label else "unknown"
    shape_str = obj.__class__.__name__.upper()
    
    # Try to get angle from cache or pose
    angle_deg = obj.cache.get("angle_deg", 0.0)
    
    text = f"{class_str} | {shape_str} | a={angle_deg:.1f} q={obj.confidence:.2f}"

    # Draw with background for readability
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(vis, (8, 8), (12 + tw, 12 + th + 8), (0, 0, 0), -1)
    cv2.putText(
        vis, text, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA,
    )
