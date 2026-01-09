"""
DEPRECATED: This file is kept for backward compatibility.
Please use pipelines/yolo_pipeline.py instead.
"""

import warnings

warnings.warn(
    "vid_to_mesh_yolo.py is deprecated. Use 'python pipelines/yolo_pipeline.py'",
    DeprecationWarning,
    stacklevel=2
)

from core.video_processor import VidToMesh


def main():
    """Legacy entry point - redirects to new pipeline."""
    from pipelines.yolo_pipeline import main as new_main
    new_main()


if __name__ == "__main__":
    main()
