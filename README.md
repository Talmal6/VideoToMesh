# DIP2 - Mesh AR Tracker

Real-time 3D mesh extraction and augmented reality tracking from video streams.

## Quick Start

```bash
# YOLO Pipeline (geometric primitives)
python main.py yolo

# SAM3D Pipeline (depth-based meshes)
python main.py sam3d

# Or use pipeline modules directly
python pipelines/yolo_pipeline.py
python pipelines/sam3d_pipeline.py
```

## Project Structure

```
DIP2/
├── main.py                     # Main entry point
├── core/                       # Core infrastructure
│   └── video_processor.py      # Multi-threaded video processing pipeline
├── pipelines/                  # Pipeline implementations
│   ├── yolo_pipeline.py        # YOLO-based geometric mesh extraction
│   └── sam3d_pipeline.py       # SAM3D depth-based mesh extraction
├── detection/                  # Object detection and tracking
│   ├── predictor.py            # Main prediction interface
│   ├── object_analyzer.py      # YOLO-based object analysis
│   ├── object_tracker.py       # Frame-to-frame tracking
│   └── detection.py            # Detection data structure
├── mesh/                       # Mesh processing
│   ├── mesh_manager.py         # Mesh handler coordinator
│   ├── mesh_proccesors/        # Mesh generation handlers
│   │   ├── mesh_handler.py     # Handler base class
│   │   ├── cylinder_handler.py # Cylinder mesh generation
│   │   ├── box_handler.py      # Box mesh generation
│   │   └── sam_3d_mesh_handler.py  # Depth-based mesh generation
│   ├── mesh_shapes/            # Mesh data structures
│   │   └── mesh_object.py      # Mesh representation
│   └── mesh_trackers/          # Mesh tracking
│       ├── mesh_tracker.py     # Tracker base class
│       └── mesh_cylinder_tracker.py  # Cylinder tracking
├── helpers/                    # Utility modules
│   ├── renderer.py             # AR visualization
│   └── ...                     # Other helper functions
└── data/                       # Test data and models

# Legacy files (for backward compatibility):
├── vid_to_mesh_yolo.py        # Redirects to pipelines/yolo_pipeline.py
└── vid_to_mesh_sam3d.py       # Redirects to pipelines/sam3d_pipeline.py
```

## Pipelines

### YOLO Pipeline
Fast geometric mesh extraction using YOLO segmentation and primitive fitting.

**Features:**
- Real-time performance (~30 FPS)
- Cylinder and box detection
- Deterministic geometry
- Minimal dependencies

**Use Cases:**
- Known object shapes
- Real-time tracking
- Production environments

### SAM3D Pipeline
Detailed 3D mesh reconstruction using monocular depth estimation.

**Features:**
- Arbitrary object shapes
- Point cloud generation
- Poisson surface reconstruction
- High-quality meshes

**Dependencies:**
```bash
pip install torch open3d timm
```

**Use Cases:**
- Unknown object shapes
- Detailed reconstruction
- Research and prototyping

**Note:** First run downloads MiDaS models (~82MB)

## Installation

```bash
# Clone repository
git clone <repository-url>
cd DIP2

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# For SAM3D pipeline
pip install torch open3d timm
```

## Usage

### Command Line

```bash
# YOLO pipeline
python main.py yolo

# SAM3D pipeline
python main.py sam3d

# Use webcam instead of file
python main.py yolo --source 0

# Custom confidence threshold
python main.py sam3d --conf 0.3
```

### Python API

```python
from detection.predictor import Predictor
from mesh.mesh_proccesors.cylinder_handler import CylinderHandler
from core.video_processor import VidToMesh

# Create predictor
predictor = Predictor()

# Create handlers
handlers = [CylinderHandler()]

# Create processor
processor = VidToMesh(
    predictor=predictor,
    handlers=handlers,
    window_title="My Tracker",
    renderer_color=(0, 255, 255),
    renderer_alpha=0.4
)

# Run
processor.run(source="./data/video.mp4", conf_threshold=0.3)
```

### Custom Handlers

Create your own mesh handler:

```python
from mesh.mesh_proccesors.mesh_handler import Handler
from detection.detection import Detection

class MyHandler(Handler):
    def can_handle(self, det: Detection) -> bool:
        return det.label == "my_object"
    
    def process(self, obj, det: Detection, frame_or_shape):
        # Generate mesh from detection
        vertices = ...  # (V, 3) array
        triangles = ...  # (F, 3) array
        return MeshObject(vertices=vertices, triangles=triangles)
```

## Configuration

### Camera Intrinsics (SAM3D)

```python
from mesh.mesh_proccesors.sam_3d_mesh_handler import SAM3DMeshHandler

handler = SAM3DMeshHandler(
    fx=600.0,  # Focal length X
    fy=600.0,  # Focal length Y
    cx=320.0,  # Principal point X
    cy=240.0,  # Principal point Y
    use_mono_depth=True
)
```

### Rendering Options

```python
processor = VidToMesh(
    predictor=predictor,
    handlers=handlers,
    window_title="Custom Title",
    renderer_color=(255, 0, 0),  # RGB
    renderer_alpha=0.6           # Transparency
)
```

## Testing

```bash
# Run all tests
python test_pipelines.py

# Test SAM3D handler specifically
python test_sam3d_handler.py
```

## Architecture

### Multi-threaded Design

The video processor uses three threads for optimal performance:

1. **Capture Thread**: Reads frames from source
2. **Mesh Thread**: Processes detections and generates meshes
3. **Render Thread** (main): Displays results with AR overlay

### Handler Pattern

Mesh generation is modular using the Handler pattern:
- Each handler implements `can_handle()` and `process()`
- MeshManager routes detections to appropriate handlers
- Multiple handlers can coexist for different object types

## Performance

| Pipeline | FPS | Quality | Memory | Dependencies |
|----------|-----|---------|--------|--------------|
| YOLO | ~30 | Geometric | Low | Minimal |
| SAM3D | ~5-10 | Detailed | High | PyTorch, Open3D |

## Troubleshooting

### ImportError: No module named 'timm'
```bash
pip install timm
```

### ImportError: No module named 'open3d'
```bash
pip install open3d
```

### MiDaS download fails
Ensure internet connectivity. Models are cached in `~/.cache/torch/hub/`

### Low FPS
- Reduce input resolution
- Increase confidence threshold  
- Use YOLO pipeline instead of SAM3D
- Enable GPU if available

## Controls

- **Q** or **ESC**: Exit
- Mouse interactions depend on renderer implementation

## License

[Your License Here]

## Contributing

Contributions welcome! Please follow the existing code structure:
- Place pipelines in `pipelines/`
- Place handlers in `mesh/mesh_proccesors/`
- Update tests in `test_pipelines.py`

## Citation

If you use this code, please cite:
```
[Your Citation Here]
```
