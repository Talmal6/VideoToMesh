# DIP2 - Mesh AR Tracker

Real-time 3D mesh extraction and AR overlay using a YOLO-based pipeline. Current handlers:
- Cylinders (bottles/cups)
- Rectangular boxes (forced rectangle via `RectExtrudeHandler`)

## Quick Start

```bash
pip install -r requirements.txt          # first time
python main.py yolo                      # default source: ./data/remote.mp4
python main.py yolo --source 0           # webcam
python main.py yolo --source 0 --output ./data/output.mp4
```

## Project Structure (key files)

```
main.py                     # CLI entry point
core/video_processor.py     # capture -> mesh -> render pipeline
pipelines/yolo_pipeline.py  # YOLO wiring (cylinder + rect handlers)
detection/                  # predictor + tracker
mesh/                       # handlers, meshes
helpers/renderer.py         # AR visualization
```

## Usage (CLI)

```bash
# File input with confidence override
python main.py yolo --source ./data/remote.mp4 --conf 0.05

# Webcam with recording
python main.py yolo --source 0 --output ./data/output.mp4

# Headless
python main.py yolo --headless
```

## Python API

```python
from detection.predictor import Predictor
from mesh.mesh_proccesors.cylinder_handler import CylinderHandler
from mesh.mesh_proccesors.rect_extrude_handler import RectExtrudeHandler
from core.video_processor import VidToMesh

predictor = Predictor()
handlers = [CylinderHandler(), RectExtrudeHandler()]

app = VidToMesh(
    predictor=predictor,
    handlers=handlers,
    window_title="Mesh AR Tracker",
    renderer_color=(0, 255, 255),
    renderer_alpha=0.4,
)

app.run(source="./data/remote.mp4", conf_threshold=0.3)
```

## Performance Notes

- Default handlers are CPU-friendly; higher input resolution or extra models will reduce FPS.
- Use `--realtime` to skip file pacing; lower input resolution for faster processing.

## Troubleshooting

- Import errors: `pip install -r requirements.txt`
- Low FPS: lower resolution, increase `--conf`, or disable output recording.

## Controls

- **Q** or **ESC**: exit viewer
- Mouse interactions depend on `helpers/renderer.py`

## License

[Your License Here]
