"""
Simple test script to verify SAM3D mesh handler works and pre-download models.
"""
import logging
import numpy as np
import cv2

from detection.detection import Detection
from mesh.mesh_proccesors.sam_3d_mesh_handler import SAM3DMeshHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_sam3d_handler():
    """Test SAM3D handler with synthetic data."""
    logger.info("=" * 60)
    logger.info("Testing SAM3D Mesh Handler")
    logger.info("=" * 60)
    
    # Create a simple test frame (640x480 color image)
    frame = np.random.randint(0, 256, (640, 480, 3)).astype(np.uint8)
    
    # Create a simple circular mask in the center
    mask = np.zeros((640, 480), dtype=np.float32)
    center = (320, 240)
    radius = 100
    y, x = np.ogrid[:640, :480]
    dist_from_center = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    mask[dist_from_center <= radius] = 1.0
    
    logger.info(f"Test frame shape: {frame.shape}")
    logger.info(f"Test mask shape: {mask.shape}, non-zero pixels: {np.sum(mask > 0)}")
    
    # Create a fake detection
    detection = Detection(
        object_id=1,
        label="test_object",
        confidence=0.9,
        frame_index=0,
        bbox_xyxy=(220, 140, 420, 340),
        mask=mask,
        frame=frame
    )
    
    # Initialize SAM3D handler
    logger.info("\nInitializing SAM3D handler...")
    logger.info("NOTE: First run will download MiDaS model (~82MB)")
    
    handler = SAM3DMeshHandler(
        fx=600.0,
        fy=600.0,
        cx=320.0,
        cy=240.0,
        use_mono_depth=True
    )
    
    # Test can_handle
    logger.info(f"\nTesting can_handle...")
    can_handle = handler.can_handle(detection)
    logger.info(f"✓ can_handle returned: {can_handle}")
    
    if not can_handle:
        logger.error("✗ Handler cannot handle this detection!")
        return False
    
    # Test process
    logger.info(f"\nProcessing mesh (this may take a minute on first run)...")
    try:
        mesh = handler.process(None, detection, frame)
        
        if mesh is None:
            logger.warning("⚠ No mesh was generated (might be due to insufficient points)")
            return False
        
        logger.info(f"✓ Mesh generated successfully!")
        logger.info(f"  Vertices: {mesh.vertices.shape}")
        logger.info(f"  Triangles: {mesh.triangles.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "SAM3D HANDLER TEST" + " " * 24 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    success = test_sam3d_handler()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ SAM3D Handler Test PASSED")
        print("✓ Models are downloaded and cached")
        print("✓ Ready to use vid_to_mesh_sam3d.py")
    else:
        print("✗ SAM3D Handler Test FAILED")
        print("  Check the error messages above")
    print("=" * 60)
