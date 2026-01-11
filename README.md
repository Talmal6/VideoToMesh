# DIP2 - Mesh AR Tracker

Real-time 3D mesh extraction and augmented reality tracking from video streams using a YOLO-based geometric pipeline.

## Quick Start

```bash
python main.py yolo                # default source: ./data/remote.mp4
python main.py yolo --source 0     # webcam
python pipelines/yolo_pipeline.py  # run module directly
```

## Project Structure

```
DIP2/
├── main.py                      # CLI entry point (YOLO only)
├── core/                        # Core infrastructure
│   └── video_processor.py       # Multi-threaded video processing pipeline
├── pipelines/
│   └── yolo_pipeline.py         # YOLO-based geometric mesh extraction
├── detection/                   # Object detection and tracking
├── mesh/                        # Mesh processing and handlers
├── helpers/
│   └── renderer.py              # AR visualization
└── data/                        # Test data and models
```

## Usage (CLI)

```bash
# YOLO pipeline with file source
python main.py yolo --source ./data/remote.mp4 --conf 0.05

# Treat file as realtime stream and save output
python main.py yolo --source ./data/remote.mp4 --realtime --output ./data/output.mp4

# Headless mode (no display window)
python main.py yolo --headless
```

## Python API

```python
from detection.predictor import Predictor
from mesh.mesh_proccesors.cylinder_handler import CylinderHandler
from mesh.mesh_proccesors.box_handler import BoxHandler
from core.video_processor import VidToMesh

predictor = Predictor()
handlers = [CylinderHandler(), BoxHandler()]

processor = VidToMesh(
    predictor=predictor,
    handlers=handlers,
    window_title="Mesh AR Tracker",
    renderer_color=(0, 255, 255),
    renderer_alpha=0.4,
)

processor.run(source="./data/remote.mp4", conf_threshold=0.3)
```

### Rendering Options

```python
processor = VidToMesh(
    predictor=predictor,
    handlers=handlers,
    window_title="Custom Title",
    renderer_color=(255, 0, 0),
    renderer_alpha=0.6,
)
```

## Testing

```bash
python test_pipelines.py
```

## Performance

| Pipeline | FPS | Quality    | Memory | Dependencies |
|----------|-----|------------|--------|--------------|
| YOLO     | ~30 | Geometric  | Low    | Minimal      |

## Troubleshooting

- Low FPS: reduce input resolution, increase confidence threshold, enable GPU if available.
- Import errors: run `pip install -r requirements.txt`.

## Controls

- **Q** or **ESC**: Exit the viewer.
- Mouse interactions depend on `helpers/renderer.py` implementation.

## License

[Your License Here]
