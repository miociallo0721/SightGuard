"""Real scene preflight. Android instrumentation remains authoritative for event/audio acceptance.

Inputs are public ADE20K validation photographs, plus the preserved CC0 wall fixture.
No training, synthetic masks, or confidence threshold search is performed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = ROOT / "android/app/src/main/assets/segformer_mit_b0_ade20k_256.onnx"
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
    reports = []
    for path in sorted(
        p
        for p in args.images.iterdir()
        if p.name.startswith(("ade-val-", "kerb_", "whitney", "doorstep"))
        and p.suffix.lower() in (".jpg", ".png")
    ):
        image = cv2.imread(str(path))
        if image is None:
            continue
        height, width = image.shape[:2]
        scale = min(256 / width, 256 / height)
        rw, rh = int(width * scale + 0.5), int(height * scale + 0.5)
        dx, dy = (256 - rw) // 2, (256 - rh) // 2
        rgb = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (rw, rh))
        values = (rgb.astype(np.float32) / 255 - [0.485, 0.456, 0.406]) / [
            0.229,
            0.224,
            0.225,
        ]
        tensor = np.zeros((1, 3, 256, 256), np.float32)
        tensor[0, :, dy : dy + rh, dx : dx + rw] = values.transpose(2, 0, 1)
        start = time.perf_counter()
        logits = session.run(["class_logits"], {"pixel_values": tensor})[0][0]
        elapsed = (time.perf_counter() - start) * 1000
        best = logits.argmax(axis=0)
        confidence = 1 / np.exp(logits - logits.max(axis=0)).sum(axis=0)
        valid = confidence >= 0.55
        # Letterbox pixels are excluded. This is raw class evidence, not connected event support.
        yy, xx = np.indices((64, 64))
        valid &= (xx * 4 < dx + rw) & ((xx + 1) * 4 > dx)
        valid &= (yy * 4 < dy + rh) & ((yy + 1) * 4 > dy)
        reports.append(
            {
                "file": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "latency_ms": elapsed,
                "class_counts_055": {
                    str(i): int(((best == i) & valid).sum())
                    for i in range(150)
                    if ((best == i) & valid).any()
                },
            }
        )
    args.output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "runtime": ort.__version__,
                "opencv": cv2.__version__,
                "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                "limitations": (
                    "PC resize; illustrative validation-set photos, no accuracy benchmark; "
                    "no Android audio or moving camera claims"
                ),
                "samples": reports,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Actual ORT preflight: {len(reports)} photographs")


if __name__ == "__main__":
    main()
