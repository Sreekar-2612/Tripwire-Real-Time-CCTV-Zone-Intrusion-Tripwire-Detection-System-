import os
import logging
from typing import List, Dict, Any, Tuple
import cv2
import numpy as np
from ultralytics import YOLO
import torch

from backend.app.config import settings

logger = logging.getLogger("tripwire.detector")

class PersonDetectorTracker:
    """
    Wraps YOLOv8n with ByteTrack tracking restricted strictly to class 0 (person).
    Maintains track history for velocity and directional vectors.
    """
    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence: float = 0.40,
        iou: float = 0.45,
        imgsz: int = 640
    ):
        self.confidence = confidence
        self.iou = iou
        self.imgsz = imgsz
        
        # Determine device
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
            
        logger.info(f"Initializing YOLO model '{model_name}' on device '{self.device}'")
        
        # Model path
        model_path = settings.MODELS_DIR / model_name
        if not model_path.exists():
            # If local file doesn't exist, YOLO will download to current dir,
            # so we point to model_name and save into models dir
            self.model = YOLO(str(model_path) if model_path.exists() else model_name)
        else:
            self.model = YOLO(str(model_path))

        # Track history for smooth trail visualization: track_id -> list of (x, y)
        self.track_history: Dict[int, List[Tuple[int, int]]] = {}
        self.max_history_len = 30

    def set_confidence(self, conf: float):
        self.confidence = max(0.05, min(conf, 1.0))

    def track(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Runs YOLO + ByteTrack on a single BGR frame.
        Returns list of person detections with track IDs and footpoints.
        """
        h, w = frame.shape[:2]
        results = self.model.track(
            source=frame,
            persist=True,
            classes=[0],  # Class 0 = Person only
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            tracker="bytetrack.yaml",
            verbose=False
        )

        detections = []
        if not results or len(results) == 0:
            return detections

        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            return detections

        track_ids = boxes.id.int().cpu().tolist()
        xyxy_coords = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        for track_id, (x1, y1, x2, y2), conf in zip(track_ids, xyxy_coords, confs):
            x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
            
            # Bottom-center foot point
            foot_px_x = int((x1 + x2) / 2.0)
            foot_px_y = int(y2)

            norm_foot_x = max(0.0, min(foot_px_x / float(w), 1.0))
            norm_foot_y = max(0.0, min(foot_px_y / float(h), 1.0))

            # Update trail history
            if track_id not in self.track_history:
                self.track_history[track_id] = []
            self.track_history[track_id].append((foot_px_x, foot_px_y))
            if len(self.track_history[track_id]) > self.max_history_len:
                self.track_history[track_id].pop(0)

            detections.append({
                "track_id": track_id,
                "bbox": [x1, y1, x2, y2],
                "norm_bbox": [x1 / w, y1 / h, x2 / w, y2 / h],
                "foot_point": (foot_px_x, foot_px_y),
                "norm_foot": (norm_foot_x, norm_foot_y),
                "confidence": float(conf),
                "trail": list(self.track_history[track_id])
            })

        # Evict old history
        active_ids = set(track_ids)
        for tid in list(self.track_history.keys()):
            if tid not in active_ids and len(self.track_history[tid]) > 0:
                self.track_history[tid].pop(0)
                if len(self.track_history[tid]) == 0:
                    del self.track_history[tid]

        return detections
