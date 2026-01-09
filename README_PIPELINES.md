# DIP2 - Mesh AR Tracker

## Overview
This project provides two mesh extraction pipelines for augmented reality tracking:

1. **YOLO-based Pipeline** (`vid_to_mesh_yolo.py`) - Geometric shape detection and mesh creation
2. **SAM3D Pipeline** (`vid_to_mesh_sam3d.py`) - Segmentation-based 3D mesh reconstruction

Both pipelines share the same core infrastructure (`VidToMesh` class) but use different mesh handlers.

---

## Architecture

### Shared Components

#### `VidToMesh` Class
The main processing engine that handles:
- Multi-threaded video capture
- Frame processing with configurable handlers
- Real-time rendering and visualization
- Thread-safe communication between capture, processing, and rendering

#### `MeshManager`
Routes detections to appropriate handlers based on detection type and capabilities.

#### `Predictor`
Performs object detection and segmentation on video frames.

#### `Renderer`
Visualizes meshes overlaid on video frames with AR-style rendering.

---

## Pipeline Comparison

### YOLO Pipeline (`vid_to_mesh_yolo.py`)

**Purpose**: Extract meshes for specific geometric shapes (cylinders, boxes)

**Handlers**:
- `CylinderHandler` - Detects and creates cylindrical meshes
- `BoxHandler` - Detects and creates box-shaped meshes

**Use Cases**:
- Tracking specific object types (bottles, boxes, etc.)
- Fast geometric approximations
- Known object categories

**Advantages**:
- Fast processing
- No additional dependencies beyond YOLO
- Deterministic geometric shapes

**Example**:
```python
from class_proccesors.predictor import Predictor
from vid_to_mesh_yolo import VidToMesh

predictor = Predictor()
app = VidToMesh(predictor)  # Uses default cylinder + box handlers
app.run("./data/remote.mp4", conf_threshold=0.05)
```

---

### SAM3D Pipeline (`vid_to_mesh_sam3d.py`)

**Purpose**: Create detailed 3D meshes from segmentation masks using depth estimation

**Handlers**:
- `SAM3DMeshHandler` - Converts segmentation masks to 3D point clouds and meshes

**Process**:
1. Segmentation mask from detection
2. Monocular depth estimation (MiDaS)
3. 3D point cloud backprojection
4. Mesh reconstruction (Poisson surface reconstruction)

**Use Cases**:
- Detailed 3D reconstruction of arbitrary objects
- Objects without clear geometric primitives
- Research and prototyping 3D capture

**Advantages**:
- Works with any segmented object
- Detailed mesh geometry
- No prior shape knowledge required

**Disadvantages**:
- Requires PyTorch and Open3D
- Slower processing
- Depth estimation quality varies

**Dependencies**:
```bash
pip install torch open3d
```

**Example**:
```python
from class_proccesors.predictor import Predictor
from mesh.mesh_proccesors.sam_3d_mesh_handler import SAM3DMeshHandler
from vid_to_mesh_yolo import VidToMesh

predictor = Predictor()
sam3d_handler = SAM3DMeshHandler(
    fx=600.0, fy=600.0,  # Camera intrinsics
    cx=320.0, cy=240.0,
    use_mono_depth=True
)

app = VidToMesh(
    predictor=predictor,
    handlers=[sam3d_handler],
    window_title="SAM3D Mesh AR Tracker"
)

app.run("./data/remote.mp4", conf_threshold=0.05)
```

---

## Creating Custom Handlers

To create your own mesh handler, inherit from `Handler` base class:

```python
from mesh.mesh_proccesors.mesh_handler import Handler
from class_proccesors.detection import Detection
from mesh.mesh_shapes.mesh_object import MeshObject

class MyCustomHandler(Handler):
    def can_handle(self, det: Detection) -> bool:
        # Return True if this handler should process this detection
        return det.label == "my_object_type"
    
    def process(self, obj, det: Detection, frame_or_shape):
        # Create and return a MeshObject
        # obj: previous mesh (for tracking continuity)
        # det: current detection with bbox, mask, etc.
        # frame_or_shape: current frame or (H, W) tuple
        
        # Your mesh creation logic here
        vertices = ...  # shape (V, 3)
        triangles = ...  # shape (F, 3)
        
        return MeshObject(vertices=vertices, triangles=triangles)
```

Then use it with VidToMesh:

```python
from vid_to_mesh_yolo import VidToMesh

predictor = Predictor()
my_handler = MyCustomHandler()
app = VidToMesh(predictor, handlers=[my_handler])
app.run(source=0)  # Use webcam
```

---

## Configuration

### Camera Intrinsics (SAM3D)
If using RGB-D or need accurate scaling:

```python
sam3d_handler = SAM3DMeshHandler(
    fx=focal_length_x,
    fy=focal_length_y,
    cx=principal_point_x,
    cy=principal_point_y,
    use_mono_depth=True  # False if providing depth directly
)
```

### Rendering Options
```python
app = VidToMesh(
    predictor=predictor,
    handlers=[...],
    window_title="Custom Title",
    renderer_color=(255, 128, 0),  # RGB color
    renderer_alpha=0.5  # Transparency
)
```

### Confidence Threshold
```python
app.run(source="video.mp4", conf_threshold=0.3)  # 0.0 to 1.0
```

---

## File Structure

```
.
├── vid_to_mesh_yolo.py          # YOLO pipeline entry + VidToMesh class
├── vid_to_mesh_sam3d.py         # SAM3D pipeline entry
├── class_proccesors/
│   ├── predictor.py             # Detection/segmentation
│   ├── detection.py             # Detection data structure
│   └── ...
├── mesh/
│   ├── mesh_manager.py          # Handler routing and tracking
│   ├── mesh_proccesors/
│   │   ├── mesh_handler.py      # Handler base class
│   │   ├── cylinder_handler.py  # YOLO: Cylinder detection
│   │   ├── box_handler.py       # YOLO: Box detection
│   │   └── sam_3d_mesh_handler.py  # SAM3D: Segmentation to mesh
│   └── mesh_shapes/
│       └── mesh_object.py       # Mesh data structure
└── helpers/
    └── renderer.py              # AR visualization
```

---

## Usage Examples

### Process Video File (YOLO)
```bash
python vid_to_mesh_yolo.py
```

### Process Video File (SAM3D)
```bash
python vid_to_mesh_sam3d.py
```

### Use Webcam
Modify `main()` in either file:
```python
source = 0  # Default webcam
app.run(source, conf_threshold=0.3)
```

### Combine Multiple Handlers
```python
from mesh.mesh_proccesors.cylinder_handler import CylinderHandler
from mesh.mesh_proccesors.sam_3d_mesh_handler import SAM3DMeshHandler

handlers = [
    CylinderHandler(),
    SAM3DMeshHandler()
]

app = VidToMesh(predictor, handlers=handlers)
```

---

## Performance Notes

- **YOLO Pipeline**: ~30 FPS on CPU, faster on GPU
- **SAM3D Pipeline**: ~5-10 FPS depending on depth estimation model
- Frame queue size is limited (maxsize=2) to prevent lag buildup
- Use `preview_every` parameter in Renderer to reduce rendering load

---

## Troubleshooting

### ImportError: torch
```bash
pip install torch torchvision
```

### ImportError: open3d
```bash
pip install open3d
```

### MiDaS download fails
The first run downloads MiDaS weights (~10MB). Ensure internet connectivity or manually download to torch hub cache.

### Low FPS
- Reduce input resolution
- Increase confidence threshold
- Disable depth estimation for YOLO pipeline
- Use GPU if available

---

## Future Enhancements

1. **RGB-D Support**: Pass depth directly to SAM3DMeshHandler
2. **Mesh Optimization**: Simplify mesh triangles for performance
3. **Tracking**: Implement mesh trackers for SAM3D pipeline
4. **Export**: Save meshes to OBJ/PLY files
5. **Multi-object**: Handle multiple objects simultaneously
