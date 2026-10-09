import cv2
import numpy as np

from sightguard.data import FrameRecord


def annotate(frame: np.ndarray, record: FrameRecord) -> np.ndarray:
    canvas = frame.copy()
    for detection in record.detections:
        x1, y1, x2, y2 = (round(value) for value in detection.bbox_xyxy)
        identity = "?" if detection.track_id is None else str(detection.track_id)
        label = f"{detection.class_name} #{identity} {detection.confidence:.2f}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 210, 80), 2)
        cv2.putText(
            canvas,
            label,
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 210, 80),
            1,
            cv2.LINE_AA,
        )
    label = f"frame {record.frame_index} | {record.timestamp_ms:.1f} ms | {record.status}"
    cv2.putText(
        canvas,
        label,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255) if record.error else (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas
