"""Main entry point for DIP2 Mesh AR Tracker."""

import sys
import argparse


def main():
    """Run the appropriate pipeline based on command line arguments."""
    parser = argparse.ArgumentParser(
        description="DIP2 Mesh AR Tracker - Real-time 3D mesh extraction from video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s yolo                    Run YOLO pipeline with default settings
  %(prog)s sam3d                   Run SAM3D pipeline with default settings
  %(prog)s yolo --source 0         Use webcam for YOLO pipeline
  %(prog)s sam3d --conf 0.3        Use custom confidence threshold
        """
    )
    
    parser.add_argument(
        "pipeline",
        choices=["yolo", "sam3d"],
        help="Pipeline to run: 'yolo' for geometric primitives, 'sam3d' for depth-based meshes"
    )
    
    parser.add_argument(
        "--source",
        default=None,
        help="Video source: file path, camera index (0), or None for default"
    )
    
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Confidence threshold for detection (0.0-1.0)"
    )
    
    args = parser.parse_args()
    
    if args.pipeline == "yolo":
        from pipelines.yolo_pipeline import main as yolo_main
        yolo_main()
    elif args.pipeline == "sam3d":
        from pipelines.sam3d_pipeline import main as sam3d_main
        sam3d_main()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
