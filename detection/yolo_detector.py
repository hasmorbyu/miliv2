
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np


_model = None
_model_name_loaded: Optional[str] = None
_is_custom_model: bool = False

AIRPLANE_LABEL = "airplane"
DRONE_LABEL    = "drone"


@dataclass
class Detection:
    class_label: str
    confidence: float
    bbox: tuple
    centroid: tuple



def _get_model(model_name: Optional[str] = None):
    global _model, _model_name_loaded, _is_custom_model

    requested = model_name or "yolov8n.pt"
    if _model is not None and _model_name_loaded == requested:
        return _model

    from ultralytics import YOLO

    if model_name and model_name != "yolov8n.pt":
        _model         = YOLO(model_name)
        _is_custom_model = True
        print(f"[YOLO] Custom model loaded: {model_name}")
        print(f"[YOLO] Classes: {set(_model.names.values())}")
    else:
        _model           = YOLO("yolov8n.pt")
        _is_custom_model = False
        print(f"[YOLO] COCO model (yolov8n.pt) — mapping \"airplane\" → \"drone\"")

    _model_name_loaded = requested
    return _model



def detect(frame: np.ndarray,
           model_name: Optional[str] = None,
           conf_threshold: float = 0.25) -> List[Detection]:
    
    model = _get_model(model_name)
    results = model(frame, verbose=False, conf=conf_threshold)[0]

    detections: List[Detection] = []
    for box in results.boxes:
        cls_id    = int(box.cls[0])
        raw_label = model.names[cls_id]

        if _is_custom_model:
            label = raw_label
        else:
            if raw_label != AIRPLANE_LABEL:
                continue
            label = DRONE_LABEL        

        conf        = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cx, cy      = (x1 + x2) // 2, (y1 + y2) // 2
        detections.append(Detection(
            class_label=label,
            confidence=conf,
            bbox=(x1, y1, x2, y2),
            centroid=(cx, cy),
        ))
    return detections
