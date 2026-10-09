"""Ultralytics adapter; heavy imports stay outside the plain data interface."""

from typing import Any

import numpy as np

from sightguard.config import PerceptionConfig
from sightguard.data import Detection


def convert_result(result: Any) -> tuple[Detection, ...]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return ()
    coordinates = boxes.xyxy.cpu().numpy()
    classes = boxes.cls.cpu().numpy()
    confidences = boxes.conf.cpu().numpy()
    identities = boxes.id.cpu().numpy() if boxes.id is not None else [None] * len(boxes)
    return tuple(
        Detection(
            track_id=int(identity) if identity is not None else None,
            class_id=int(class_id),
            class_name=result.names[int(class_id)],
            confidence=float(confidence),
            bbox_xyxy=tuple(float(v) for v in bbox),
        )
        for bbox, class_id, confidence, identity in zip(
            coordinates, classes, confidences, identities, strict=True
        )
    )


class YoloPerceptor:
    """One instance per video. Never share persistent tracker state across videos."""

    def __init__(self, config: PerceptionConfig) -> None:
        from pathlib import Path

        from ultralytics import YOLO

        if not Path(config.model_path).is_file():
            raise FileNotFoundError(
                f"Model not found: {config.model_path}. Download weights explicitly first."
            )
        self.config = config
        self.model = YOLO(config.model_path)
        if config.class_ids and any(item not in self.model.names for item in config.class_ids):
            raise ValueError("class_ids contains a class not supported by the model")

    def process(self, frame: np.ndarray) -> tuple[Detection, ...]:
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.config.tracker,
            conf=self.config.confidence,
            device=self.config.device,
            imgsz=self.config.image_size,
            classes=list(self.config.class_ids) if self.config.class_ids is not None else None,
            verbose=False,
            save=False,
        )
        return convert_result(results[0])
