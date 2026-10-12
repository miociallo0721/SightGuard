"""CPU pixel-accuracy ablation; never infers Android event acceptance from PC speed."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from fetch_furniture_validation import GROUPS
from PIL import Image


def evaluate(model: Path, manifests: list[Path]) -> dict:
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
    shape = session.get_inputs()[0].shape
    assert shape[:2] == [1, 3] and shape[2] == shape[3]
    size = shape[2]
    results = []
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for case in manifest["cases"]:
            path = manifest_path.parent / case["image"]
            mask_path = manifest_path.parent / case["mask"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == case["image_sha256"]
            assert hashlib.sha256(mask_path.read_bytes()).hexdigest() == case["mask_sha256"]
            rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            started = time.perf_counter()
            height, width = rgb.shape[:2]
            ratio = min(size / width, size / height)
            rw, rh = int(width * ratio + 0.5), int(height * ratio + 0.5)
            dx, dy = (size - rw) // 2, (size - rh) // 2
            resized = (
                cv2.resize(rgb, (rw, rh), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
            )
            values = (resized - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
                [0.229, 0.224, 0.225], np.float32
            )
            tensor = np.zeros((1, 3, size, size), np.float32)
            tensor[0, :, dy : dy + rh, dx : dx + rw] = values.transpose(2, 0, 1)
            logits = session.run(["class_logits"], {"pixel_values": tensor})[0][0]
            top = logits.argmax(axis=0)
            confidence = 1 / np.exp(logits - logits.max(axis=0)).sum(axis=0)
            hh, ww = top.shape
            ys = (((np.arange(128) + 0.5) / 128 * rh + dy) / size * hh).astype(int)
            xs = (((np.arange(128) + 0.5) / 128 * rw + dx) / size * ww).astype(int)
            prediction = top[np.ix_(ys, xs)]
            confidence = confidence[np.ix_(ys, xs)]
            prediction = np.where(confidence >= 0.35, prediction, -1)
            elapsed = (time.perf_counter() - started) * 1000
            with Image.open(mask_path) as image:
                gt = np.asarray(image)
            gy = ((np.arange(128) + 0.5) / 128 * gt.shape[0]).astype(int)
            gx = ((np.arange(128) + 0.5) / 128 * gt.shape[1]).astype(int)
            labels = gt[np.ix_(gy, gx)].astype(np.int32) - 1
            known = labels >= 0
            pixels = {}
            for group, ids in GROUPS.items():
                target, actual = np.isin(labels, list(ids)), np.isin(prediction, list(ids))
                pixels[group] = {
                    "tp": int(np.sum(known & target & actual)),
                    "fp": int(np.sum(known & ~target & actual)),
                    "fn": int(np.sum(known & target & ~actual)),
                }
            results.append(
                {
                    "image": case["image"],
                    "split_manifest": str(manifest_path),
                    "image_sha256": case["image_sha256"],
                    "latency_ms": elapsed,
                    "pixels": pixels,
                }
            )
    groups = {}
    for group in GROUPS:
        tp, fp, fn = (sum(c["pixels"][group][k] for c in results) for k in ("tp", "fp", "fn"))
        groups[group] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "iou": tp / (tp + fp + fn) if tp + fp + fn else None,
        }
    return {
        "model": str(model),
        "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "input_size": size,
        "onnxruntime": ort.__version__,
        "threads": 1,
        "groups": groups,
        "cases": results,
        "latency_ms_p50": float(np.median([c["latency_ms"] for c in results])),
        "latency_ms_p95": float(np.percentile([c["latency_ms"] for c in results], 95)),
        "scope": (
            "PC ONNX with OpenCV linear resize; 128x128 center samples; .35 confidence gate. "
            "Not Android performance or event validation."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, action="append", required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = []
    for model in args.model:
        reports.append(evaluate(model, args.manifest))
        args.output.write_text(
            json.dumps({"schema": 1, "experiments": reports}, indent=2), encoding="utf-8"
        )
        print(model, reports[-1]["latency_ms_p50"], flush=True)


if __name__ == "__main__":
    main()
