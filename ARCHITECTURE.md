# Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         main.py                              │
│                    (Entry Point)                             │
└────────────────┬────────────────────────────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
        ▼                 ▼
┌──────────────┐   ┌──────────────┐
│ yolo_pipeline│   │sam3d_pipeline│
│              │   │              │
└──────┬───────┘   └──────┬───────┘
       │                  │
       └──────┬───────────┘
              │
              ▼
    ┌─────────────────┐
    │ video_processor │
    │  (VidToMesh)    │
    └────────┬────────┘
             │
    ┌────────┼────────┐
    │        │        │
    ▼        ▼        ▼
┌─────┐  ┌──────┐  ┌────────┐
│Pred │  │Mesh  │  │Render  │
│ictor│  │Mgr   │  │er      │
└──┬──┘  └───┬──┘  └────────┘
   │         │
   │         └──────────┐
   │                    │
   ▼                    ▼
┌──────────┐      ┌─────────────┐
│ Analyzer │      │  Handlers   │
│ Tracker  │      │  (Cylinder, │
└──────────┘      │   Box,      │
                  │   SAM3D)    │
                  └─────────────┘
```

## Module Dependencies

```
core/
  └── video_processor.py
      ├── uses: detection.predictor
      ├── uses: mesh.mesh_manager
      └── uses: helpers.renderer

pipelines/
  ├── yolo_pipeline.py
  │   ├── uses: core.video_processor
  │   ├── uses: detection.predictor
  │   └── uses: mesh.mesh_proccesors.[cylinder|box]_handler
  │
  └── sam3d_pipeline.py
      ├── uses: core.video_processor
      ├── uses: detection.predictor
      └── uses: mesh.mesh_proccesors.sam_3d_mesh_handler

detection/
  ├── predictor.py
  │   ├── uses: object_analyzer
  │   ├── uses: object_tracker
  │   └── uses: detection
  │
  ├── object_analyzer.py (YOLO)
  ├── object_tracker.py
  └── detection.py (data class)

mesh/
  ├── mesh_manager.py
  │   ├── uses: mesh_proccesors.mesh_handler
  │   └── uses: mesh_trackers.mesh_tracker
  │
  ├── mesh_proccesors/
  │   ├── mesh_handler.py (base)
  │   ├── cylinder_handler.py
  │   ├── box_handler.py
  │   └── sam_3d_mesh_handler.py
  │
  ├── mesh_shapes/
  │   └── mesh_object.py
  │
  └── mesh_trackers/
      ├── mesh_tracker.py (base)
      └── mesh_cylinder_tracker.py

helpers/
  ├── renderer.py
  └── [other utilities]
```

## Data Flow

```
Video Source
    │
    ▼
[Capture Thread] ──────> Frame Queue
    │                        │
    │                        ▼
    │                  [Mesh Thread]
    │                        │
    │                        ├─> Predictor
    │                        │     └─> Analyzer/Tracker
    │                        │
    │                        ├─> MeshManager
    │                        │     └─> Handler Selection
    │                        │           └─> Mesh Generation
    │                        │
    │                        ▼
    │                   Mesh Queue
    │                        │
    └────────────────────────┤
                             │
                             ▼
                      [Render Thread]
                             │
                             ├─> Renderer
                             │     └─> AR Overlay
                             │
                             ▼
                        Display Window
```

## Threading Model

```
┌─────────────────────────────────────────────────────────┐
│                    Main Thread                          │
│                                                         │
│  1. Initialize components                               │
│  2. Start worker threads                                │
│  3. Run render loop                                     │
│  4. Handle user input                                   │
│  5. Cleanup on exit                                     │
└─────────────────────────────────────────────────────────┘
                 │                    │
        ┌────────┴────────┐  ┌────────┴────────┐
        ▼                 ▼  ▼                 ▼
┌──────────────┐   ┌──────────────┐
│Capture Thread│   │ Mesh Thread  │
│              │   │              │
│ Read frames  │   │ Process det. │
│ Push to queue│   │ Generate mesh│
│              │   │ Push to queue│
└──────────────┘   └──────────────┘

Communication via:
  - Frame Queue (maxsize=2)
  - LatestValue containers
  - Thread events
```

## Handler Pattern

```
Detection
    │
    ▼
MeshManager
    │
    ├─> can_handle(detection)?
    │   ├─> CylinderHandler ─> YES ─> process()
    │   ├─> BoxHandler ─────── NO
    │   └─> SAM3DHandler ────── NO
    │
    └─> Selected Handler
            │
            └─> process(detection, frame)
                    │
                    ├─> Extract features
                    ├─> Generate mesh
                    └─> Return MeshObject
```

## Class Relationships

```
VidToMesh
  ├── HAS: Predictor
  ├── HAS: MeshManager
  │     └── HAS: List[Handler]
  └── HAS: Renderer

Predictor
  ├── HAS: ObjectAnalyzer
  ├── HAS: ObjectTracker
  └── PRODUCES: Detection

Handler (Interface)
  ├── IMPLEMENTS: can_handle()
  └── IMPLEMENTS: process()

CylinderHandler ───┐
BoxHandler ────────┼─── EXTENDS: Handler
SAM3DMeshHandler ──┘

Detection (Data)
  ├── bbox_xyxy
  ├── mask
  ├── label
  ├── confidence
  └── frame

MeshObject (Data)
  ├── vertices
  └── triangles
```

## Pipeline Flow

### YOLO Pipeline
```
Video → Predictor → YOLO Segmentation
                         │
                         ├─> Cylinder Detection?
                         │   └─> CylinderHandler → Geometric Mesh
                         │
                         └─> Box Detection?
                             └─> BoxHandler → Geometric Mesh
```

### SAM3D Pipeline
```
Video → Predictor → YOLO Segmentation
                         │
                         └─> Mask Available?
                             └─> SAM3DHandler
                                   │
                                   ├─> Depth Estimation (MiDaS)
                                   ├─> 3D Backprojection
                                   ├─> Point Cloud
                                   └─> Poisson Reconstruction → Mesh
```

## File Organization Logic

```
core/           → Shared, reusable infrastructure
pipelines/      → Specific application configurations
detection/      → Object detection and tracking logic
mesh/           → Mesh generation and manipulation
  ├── proccesors/  → Mesh creation from detections
  ├── shapes/      → Mesh data structures
  └── trackers/    → Mesh temporal tracking
helpers/        → Utility functions
data/           → Input data and models
```

## Design Principles

1. **Separation of Concerns**
   - Core infrastructure separate from pipelines
   - Detection separate from mesh generation
   - Each module has single responsibility

2. **Modularity**
   - Handlers are pluggable
   - Easy to add new pipelines
   - Components are independently testable

3. **Extensibility**
   - Handler interface for new mesh types
   - Pipeline templates for new applications
   - Clear extension points

4. **Maintainability**
   - Clear folder structure
   - Consistent naming
   - Comprehensive documentation

5. **Performance**
   - Multi-threaded design
   - Queue-based communication
   - Frame dropping to prevent lag
