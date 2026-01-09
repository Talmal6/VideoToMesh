"""YOLO-based mesh extraction pipeline using geometric primitives."""

import sys
import os
from pathlib import Path

# Add parent directory to path for direct execution
if __name__ == "__main__":
    parent_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(parent_dir))

import logging

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


def main():
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
    
    source = "./data/remote.mp4"
    logger.info(f"Processing source: {source}")
    
    app.run(source, conf_threshold=0.05)


if __name__ == "__main__":
    main()
