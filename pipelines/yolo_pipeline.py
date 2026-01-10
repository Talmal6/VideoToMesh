"""YOLO-based mesh extraction pipeline using geometric primitives."""

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
from mesh.mesh_proccesors.cylinder_handler import CylinderHandler
from mesh.mesh_proccesors.box_handler import BoxHandler
from core.video_processor import VidToMesh
from pipelines.yolo_config import SHAPE_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app_yolo.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main(
    source: str = "./data/remote.mp4",
    conf_threshold: float = 0.05,
    realtime: bool = False,
    output_path: str = None,
    show: bool = True
):
    """Run YOLO-based mesh extraction pipeline."""
    logger.info("Initializing YOLO Mesh Pipeline")
    
    predictor = Predictor()
    # Initialize handlers with specific target classes from config
    handlers = [
        CylinderHandler(labels=SHAPE_CONFIG['cylinder']),
        BoxHandler(labels=SHAPE_CONFIG['box'])
    ]
    
    app = VidToMesh(
        predictor=predictor,
        handlers=handlers,
        window_title="YOLO Mesh AR Tracker",
        renderer_color=(0, 255, 255),
        renderer_alpha=0.4
    )
    
    logger.info(f"Processing source: {source}")
    
    app.run(
        source,
        conf_threshold=conf_threshold,
        realtime=realtime,
        output_path=output_path,
        show=show
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Mesh Pipeline")
    parser.add_argument("--source", default="./data/remote.mp4", help="Video source path or camera index")
    parser.add_argument("--conf", type=float, default=0.05, help="Confidence threshold (0.0-1.0)")
    parser.add_argument("--realtime", action="store_true", help="Treat file input like realtime stream")
    parser.add_argument("--output", default=None, help="Output video file path for rendered frames")
    parser.add_argument("--headless", action="store_true", help="Disable display window (headless mode)")
    args = parser.parse_args()

    main(
        source=args.source,
        conf_threshold=args.conf,
        realtime=args.realtime,
        output_path=args.output,
        show=not args.headless
    )
