"""Configuration for YOLO pipeline shapes."""

# Configuration: Map YOLO class labels to specific Shape Handlers
SHAPE_CONFIG = {
    'cylinder': ('cup', 'bottle', 'vase', 'can'),
    'box': ('remote', 'keyboard', 'laptop', 'cell phone', 'book', 'monitor'),
}
