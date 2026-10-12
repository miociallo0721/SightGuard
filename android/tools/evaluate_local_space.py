"""Real CPU ONNX inference on frozen ADE validation samples; no route ground truth."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    assets = root / "android/app/src/main/assets"
    output = root / "android/app/src/androidTest/assets/local_space"
    output.mkdir(parents=True, exist_ok=True)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    semantic_path = assets / "segformer_mit_b0_ade20k_256.onnx"
    # Match the existing 256 fallback contract; separately verify Android actual inference.
    if not semantic_path.exists():
        semantic_path = assets / "segformer_b0_ade20k.onnx"
    semantic = ort.InferenceSession(str(semantic_path), opts)
    detector = ort.InferenceSession(str(assets / "yoloe26n.onnx"), opts)
    mapping = json.loads((assets / "yoloe_model_manifest.json").read_text())["app_class_ids"]
    size = semantic.get_inputs()[0].shape[2]
    records = []
    for split in ("ade20k-furniture-v06", "ade20k-furniture-v06-holdout"):
        folder = root / "datasets/imported" / split
        manifest = json.loads((folder / "manifest.json").read_text())
        for case in manifest["cases"]:
            image_path, mask_path = folder / case["image"], folder / case["mask"]
            assert sha(image_path) == case["image_sha256"] and sha(mask_path) == case["mask_sha256"]
            rgb = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            start = time.perf_counter()
            ratio = min(size / w, size / h)
            rw, rh = int(w * ratio + 0.5), int(h * ratio + 0.5)
            dx, dy = (size - rw) // 2, (size - rh) // 2
            x = cv2.resize(rgb, (rw, rh)).astype(np.float32) / 255
            x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
                [0.229, 0.224, 0.225], np.float32
            )
            tensor = np.zeros((1, 3, size, size), np.float32)
            tensor[0, :, dy : dy + rh, dx : dx + rw] = x.transpose(2, 0, 1)
            logits = semantic.run(None, {semantic.get_inputs()[0].name: tensor})[0][0]
            classes = logits.argmax(axis=0)
            confidence = 1 / np.exp(logits - logits.max(axis=0)).sum(axis=0)
            cells = []
            for y in range(classes.shape[0]):
                for x in range(classes.shape[1]):
                    left, right = np.clip([(x * 4 - dx) / rw, ((x + 1) * 4 - dx) / rw], 0, 1)
                    top, bottom = np.clip([(y * 4 - dy) / rh, ((y + 1) * 4 - dy) / rh], 0, 1)
                    if confidence[y, x] >= 0.35 and right > left and bottom > top:
                        cells.append(
                            [
                                float(left),
                                float(top),
                                float(right),
                                float(bottom),
                                int(classes[y, x]),
                                float(confidence[y, x]),
                            ]
                        )
            semantic_ms = (time.perf_counter() - start) * 1000
            scale = min(320 / w, 320 / h)
            yw, yh = int(w * scale + 0.5), int(h * scale + 0.5)
            px, py = (320 - yw) // 2, (320 - yh) // 2
            inp = np.full((320, 320, 3), 114, np.float32)
            inp[py : py + yh, px : px + yw] = cv2.resize(rgb, (yw, yh))
            rows = detector.run(
                None, {detector.get_inputs()[0].name: (inp / 255).transpose(2, 0, 1)[None]}
            )[0][0]
            boxes = []
            for row in rows:
                if row[4] < 0.35:
                    continue
                box = np.clip(
                    [
                        (row[0] - px) / yw,
                        (row[1] - py) / yh,
                        (row[2] - px) / yw,
                        (row[3] - py) / yh,
                    ],
                    0,
                    1,
                )
                if box[2] > box[0] and box[3] > box[1]:
                    boxes.append([mapping[int(row[5])], float(row[4]), *map(float, box)])
            with Image.open(mask_path) as im:
                gt = np.asarray(im).astype(np.int32) - 1
            # Independent annotation statistics per lower image third; no physical clearance labels.
            truths = []
            for col in range(3):
                part = gt[int(h * 8 / 15) :, int(w * col / 3) : int(w * (col + 1) / 3)]
                truths.append(
                    {
                        "ground_fraction": float(np.isin(part, [3, 6, 11]).mean()),
                        "hazard_fraction": float(
                            np.isin(part, [21, 38, 53, 59, 95, 96, 121]).mean()
                        ),
                        "unknown_fraction": float((part < 0).mean()),
                    }
                )
            record = {
                "image": case["image"],
                "image_sha256": case["image_sha256"],
                "mask_sha256": case["mask_sha256"],
                "split": split,
                "width": w,
                "height": h,
                "cells": cells,
                "detections": boxes,
                "gt_bands": truths,
                "semantic_ms": semantic_ms,
                "model_sha256": sha(semantic_path),
            }
            (output / (image_path.stem + ".json")).write_text(json.dumps(record), encoding="utf-8")
            records.append({k: v for k, v in record.items() if k not in ("cells", "detections")})
            print(case["image"], round(semantic_ms), flush=True)
    (output / "index.json").write_text(json.dumps(records), encoding="utf-8")
    # Select Android photos using annotation-only direction margins and hazard coverage.
    selected = []
    for direction in range(3):
        selected.append(
            max(
                records,
                key=lambda r: (
                    r["gt_bands"][direction]["ground_fraction"]
                    - max(r["gt_bands"][j]["ground_fraction"] for j in range(3) if j != direction)
                ),
            )
        )
    selected.extend(
        sorted(
            records, key=lambda r: max(b["hazard_fraction"] for b in r["gt_bands"]), reverse=True
        )[:2]
    )
    selected.extend(
        sorted(records, key=lambda r: sum(b["ground_fraction"] for b in r["gt_bands"]))[:1]
    )
    names = list(dict.fromkeys(r["image"] for r in selected))
    for record in selected:
        shutil.copyfile(
            root / "datasets/imported" / record["split"] / record["image"], output / record["image"]
        )
    (output / "actual_cases.json").write_text(json.dumps(names), encoding="utf-8")
    report = {
        "schema_version": 1,
        "cases": len(records),
        "semantic_model_sha256": sha(semantic_path),
        "detector_sha256": sha(assets / "yoloe26n.onnx"),
        "onnxruntime": ort.__version__,
        "threads": 1,
        "source": "https://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip",
        "scope": "Static validation images, no temporal or route/clearance ground truth",
        "image_terms": "https://ade20k.csail.mit.edu/terms/",
        "image_use": "Non-commercial research only; images not redistributed",
        "annotation_terms": "BSD-3-Clause; image rights are distinct",
        "records": records,
    }
    (root / "android/deliveries/local-space/pc_model_results.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
