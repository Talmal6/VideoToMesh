"""SAM3D-based mesh extraction pipeline using depth estimation."""

import sys
import os
from pathlib import Path

# Add parent directory to path for direct execution
if __name__ == "__main__":
    parent_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(parent_dir))

import logging

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


def main():
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
    
    source = "./data/remote.mp4"
    logger.info(f"Processing source: {source}")
    
    try:
        app.run(source, conf_threshold=0.05, loop=True)
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Install with: pip install torch open3d timm")
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise


if __name__ == "__main__":
    main()
