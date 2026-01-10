"""SAM3D-based mesh extraction pipeline using depth estimation."""

import sys
import os
from pathlib import Path

# Add parent directory to path for direct execution
if __name__ == "__main__":
    parent_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(parent_dir))

import logging

import argparse

from detection.predictor import Predictor
from mesh.mesh_proccesors.sam_3d_mesh_handler import SAM3DMeshHandler
from core.video_processor import VidToMesh

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app_sam3d.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main(
    source: str = "./data/remote.mp4",
    conf_threshold: float = 0.05,
    loop: bool = True,
    realtime: bool = False,
    output_path: str = None,
    show: bool = True
):
    """
    Run SAM3D-based mesh extraction pipeline.
    
    Requirements:
        - torch (MiDaS depth estimation)
        - open3d (point cloud to mesh conversion)
        - timm (MiDaS model backbone)
    
    Note: First run downloads MiDaS models (~82MB).
    """
    logger.info("Initializing SAM3D Mesh Pipeline")
    logger.info("NOTE: First run downloads MiDaS model (~82MB)")
    
    predictor = Predictor()
    
    sam3d_handler = SAM3DMeshHandler(
        fx=600.0,
        fy=600.0,
        cx=320.0,
        cy=240.0,
        use_mono_depth=True
    )
    
    app = VidToMesh(
        predictor=predictor,
        handlers=[sam3d_handler],
        window_title="SAM3D Mesh AR Tracker",
        renderer_color=(255, 128, 0),
        renderer_alpha=0.5
    )
    
    logger.info(f"Processing source: {source}")
    
    try:
        app.run(
            source,
            conf_threshold=conf_threshold,
            loop=loop,
            realtime=realtime,
            output_path=output_path,
            show=show
        )
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Install with: pip install torch open3d timm")
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM3D Mesh Pipeline")
    parser.add_argument("--source", default="./data/remote.mp4", help="Video source path or camera index")
    parser.add_argument("--conf", type=float, default=0.05, help="Confidence threshold (0.0-1.0)")
    parser.add_argument("--loop", action="store_true", help="Loop the input video")
    parser.add_argument("--no-loop", action="store_true", help="Disable looping the input video")
    parser.add_argument("--realtime", action="store_true", help="Treat file input like realtime stream")
    parser.add_argument("--output", default=None, help="Output video file path for rendered frames")
    parser.add_argument("--headless", action="store_true", help="Disable display window (headless mode)")
    args = parser.parse_args()

    loop = True
    if args.loop:
        loop = True
    if args.no_loop:
        loop = False

    main(
        source=args.source,
        conf_threshold=args.conf,
        loop=loop,
        realtime=args.realtime,
        output_path=args.output,
        show=not args.headless
    )
