from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class Detection:
    object_id: int
    label: str
    confidence: float
    frame_index: int
    bbox_xyxy: Tuple[float, float, float, float]
    mask: Optional[np.ndarray] = None