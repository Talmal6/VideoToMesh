# Quick Reference Guide

## Running Pipelines

### YOLO (Fast, Geometric)
```bash
python main.py yolo
# or
python pipelines/yolo_pipeline.py
```

### SAM3D (Detailed, Depth-based)
```bash
python main.py sam3d
# or (can run directly now)
python pipelines/sam3d_pipeline.py
```

## Import Reference

### Core Components
```python
from core.video_processor import VidToMesh
from detection.predictor import Predictor
from detection.detection import Detection
```

### Handlers
```python
from mesh.mesh_proccesors.cylinder_handler import CylinderHandler
from mesh.mesh_proccesors.box_handler import BoxHandler
from mesh.mesh_proccesors.sam_3d_mesh_handler import SAM3DMeshHandler
```

### Managers
```python
from mesh.mesh_manager import MeshManager
from helpers.renderer import Renderer
```

## Common Tasks

### Create Custom Pipeline
```python
from detection.predictor import Predictor
from mesh.mesh_proccesors.cylinder_handler import CylinderHandler
from core.video_processor import VidToMesh

predictor = Predictor()
handlers = [CylinderHandler()]
processor = VidToMesh(predictor, handlers=handlers)
processor.run(source="video.mp4", conf_threshold=0.3)
```

### Use Webcam
```python
processor.run(source=0)  # Default camera
processor.run(source=1)  # Second camera
```

### Combine Multiple Handlers
```python
from mesh.mesh_proccesors.cylinder_handler import CylinderHandler
from mesh.mesh_proccesors.box_handler import BoxHandler

handlers = [CylinderHandler(), BoxHandler()]
processor = VidToMesh(predictor, handlers=handlers)
```

## Project Structure

```
core/           → Core infrastructure
pipelines/      → Pipeline entry points  
detection/      → Object detection
mesh/           → Mesh processing
helpers/        → Utilities
data/           → Test data
```

## Key Files

- `main.py` - Unified entry point
- `core/video_processor.py` - VidToMesh class
- `pipelines/yolo_pipeline.py` - YOLO pipeline
- `pipelines/sam3d_pipeline.py` - SAM3D pipeline
- `detection/predictor.py` - Detection interface
- `mesh/mesh_manager.py` - Handler coordinator
- `README.md` - Full documentation

## Testing

```bash
python test_pipelines.py      # Test all components
python test_sam3d_handler.py  # Test SAM3D specifically
```

## Troubleshooting

### Import Errors
- Old: `from class_proccesors...`
- New: `from detection...`

### Module Not Found
```bash
pip install -r requirements.txt
pip install torch open3d timm  # For SAM3D
```

### Backward Compatibility
Old scripts still work:
```bash
python vid_to_mesh_yolo.py    # Shows deprecation warning
python vid_to_mesh_sam3d.py   # Shows deprecation warning
```
