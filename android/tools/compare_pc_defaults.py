"""Single-photo comparison; this is not a video accuracy benchmark."""

import json
from pathlib import Path

import ultralytics
from ultralytics import YOLO

root = Path(__file__).resolve().parents[2]
model = YOLO(root / "models/yolo26n.pt")
records = {}
for name, size, score, classes in [
    ("pc_default_detection_settings", 640, 0.25, [0, 1, 3]),
    ("android_detection_settings_in_python", 320, 0.35, None),
]:
    result = model.predict(
        root / "android/validation/validation.jpg",
        imgsz=size,
        conf=score,
        classes=classes,
        rect=name == "pc_default_detection_settings",
        device="cpu",
        verbose=False,
    )[0]
    records[name] = {
        "image_size": size,
        "confidence_threshold": score,
        "class_filter": classes,
        "rect": name == "pc_default_detection_settings",
        "detections": [
            {
                "class_id": int(box.cls.item()),
                "confidence": float(box.conf.item()),
                "xyxy": box.xyxy[0].tolist(),
            }
            for box in result.boxes
        ],
    }
output = {
    "schema_version": 1,
    "ultralytics_version": ultralytics.__version__,
    "limitation": "One photo only. No tracking, video benchmark, recall, or safety claim.",
    "results": records,
}
path = root / "android/validation/pc_default_comparison.json"
path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
print({name: len(record["detections"]) for name, record in records.items()})
