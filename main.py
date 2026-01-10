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

    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Treat file input like realtime stream (no FPS pacing)"
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Output video file path for rendered frames"
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        help="Disable display window (headless mode)"
    )
    
    args = parser.parse_args()

    source = args.source if args.source is not None else "./data/remote.mp4"
    conf = args.conf if args.conf is not None else 0.05

    if args.pipeline == "yolo":
        from pipelines.yolo_pipeline import main as yolo_main
        yolo_main(
            source=source,
            conf_threshold=conf,
            realtime=args.realtime,
            output_path=args.output,
            show=not args.headless
        )
    elif args.pipeline == "sam3d":
        from pipelines.sam3d_pipeline import main as sam3d_main
        sam3d_main(
            source=source,
            conf_threshold=conf,
            realtime=args.realtime,
            output_path=args.output,
            show=not args.headless
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
