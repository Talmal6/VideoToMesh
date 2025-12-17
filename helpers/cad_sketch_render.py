"""
CAD-style sketch renderer.
ONLY draws using pre-computed MeshData - zero computations here.

Dependencies: cv2, numpy only.
"""

import cv2
import numpy as np

# Import MeshData from mesh_proccesors
from mesh_proccesors.mesh_factory import MeshData


def render_mesh_overlay(
    frame_bgr: np.ndarray,
    mesh_data: MeshData,
) -> np.ndarray:
    """
    Render mesh overlay onto frame.
    This function does ZERO computation - only cv2 draw calls using MeshData fields.
    Dispatches to cylinder or box rendering based on shape_type.
    """
    vis = frame_bgr.copy()

    if mesh_data.shape_type == "box":
        return _render_box(vis, mesh_data)
    else:
        return _render_cylinder(vis, mesh_data)


def _render_cylinder(vis: np.ndarray, mesh_data: MeshData) -> np.ndarray:
    """Render cylinder: silhouette + rings + centerline."""
    # --- Silhouette ---
    # Black outline (thick)
    cv2.drawContours(vis, [mesh_data.poly2d], -1, (0, 0, 0), mesh_data.outline_black_thick, cv2.LINE_AA)
    # Colored outline
    cv2.drawContours(vis, [mesh_data.poly2d], -1, mesh_data.color, mesh_data.outline_thick, cv2.LINE_AA)

    # --- Centerline ---
    x1, y1, x2, y2 = mesh_data.centerline
    darker_color = tuple(max(0, c - 60) for c in mesh_data.color)
    cv2.line(vis, (x1, y1), (x2, y2), darker_color, mesh_data.centerline_thick, cv2.LINE_AA)

    # --- Rings ---
    for ring in mesh_data.rings2d:
        cx, cy, ax, ay, angle_int = ring
        cv2.ellipse(vis, (cx, cy), (ax, ay), angle_int, 0, 360, mesh_data.color, mesh_data.ring_thick, cv2.LINE_AA)

    return vis


def _render_box(vis: np.ndarray, mesh_data: MeshData) -> np.ndarray:
    """Render box: 3D wireframe with front face, back face, and connecting edges."""
    color = mesh_data.color
    black = (0, 0, 0)
    thick = mesh_data.outline_thick
    black_thick = mesh_data.outline_black_thick

    # Draw all 12 edges from pre-computed box_edges2d
    if mesh_data.box_edges2d is not None and len(mesh_data.box_edges2d) > 0:
        edges = mesh_data.box_edges2d  # shape (12, 2, 2)

        # Draw black outlines first (all edges)
        for edge in edges:
            p1 = tuple(edge[0])
            p2 = tuple(edge[1])
            cv2.line(vis, p1, p2, black, black_thick, cv2.LINE_AA)

        # Draw colored edges
        # Front face edges (0-3): solid, full color
        for i in range(4):
            p1 = tuple(edges[i][0])
            p2 = tuple(edges[i][1])
            cv2.line(vis, p1, p2, color, thick, cv2.LINE_AA)

        # Back face edges (4-7): slightly darker (receding)
        darker = tuple(max(0, c - 40) for c in color)
        for i in range(4, 8):
            p1 = tuple(edges[i][0])
            p2 = tuple(edges[i][1])
            cv2.line(vis, p1, p2, darker, thick, cv2.LINE_AA)

        # Connecting edges (8-11): dashed appearance via lighter color
        connect_color = tuple(max(0, c - 20) for c in color)
        for i in range(8, 12):
            p1 = tuple(edges[i][0])
            p2 = tuple(edges[i][1])
            cv2.line(vis, p1, p2, connect_color, thick, cv2.LINE_AA)

    # Draw corner markers on front face
    if mesh_data.box_front2d is not None:
        for pt in mesh_data.box_front2d:
            cv2.circle(vis, tuple(pt), 5, color, -1, cv2.LINE_AA)
            cv2.circle(vis, tuple(pt), 5, black, 1, cv2.LINE_AA)

    return vis


def render_debug_overlay(
    vis: np.ndarray,
    mesh_data: MeshData,
) -> None:
    """Draw debug text: class name, shape type, angle, quality."""
    # Build info text
    class_str = mesh_data.class_name if mesh_data.class_name else "unknown"
    shape_str = mesh_data.shape_type.upper()
    text = f"{class_str} | {shape_str} | a={mesh_data.angle_deg:.1f} q={mesh_data.quality:.2f}"

    # Draw with background for readability
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(vis, (8, 8), (12 + tw, 12 + th + 8), (0, 0, 0), -1)
    cv2.putText(
        vis, text, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA,
    )
