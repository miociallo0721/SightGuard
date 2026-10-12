"""Small, reproducible box/semantic-ground-truth screen, not an instance AP benchmark."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO, YOLOE

GROUPS = {
    "TABLE": {15, 33, 64},
    "CHAIR": {19, 30, 31, 69, 75, 110},
    "CABINET": {10, 35},
    "SOFA": {23},
    "BED": {7},
    "SHELF": {24},
    "COUNTER": {45, 70},
    "APPLIANCE": {50},
    "MONITOR": {143},
}
NAMES = {
    "dining table": "TABLE",
    "desk": "TABLE",
    "coffee table": "TABLE",
    "chair": "CHAIR",
    "bench": "CHAIR",
    "stool": "CHAIR",
    "armchair": "CHAIR",
    "office chair": "CHAIR",
    "cabinet": "CABINET",
    "wardrobe": "CABINET",
    "couch": "SOFA",
    "bed": "BED",
    "bookshelf": "SHELF",
    "shelf": "SHELF",
    "counter": "COUNTER",
    "refrigerator": "APPLIANCE",
    "computer monitor": "MONITOR",
}
EXTRA = [
    "desk",
    "coffee table",
    "stool",
    "armchair",
    "office chair",
    "cabinet",
    "wardrobe",
    "bookshelf",
    "counter",
    "computer monitor",
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    work = root / "android/deliveries/yoloe26-quick"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    manifests = [
        root / "datasets/imported" / name / "manifest.json"
        for name in ["ade20k-furniture-v06", "ade20k-furniture-v06-holdout"]
    ]
    cases = []
    for mp in manifests:
        for case in json.loads(mp.read_text(encoding="utf-8"))["cases"]:
            image, mask = mp.parent / case["image"], mp.parent / case["mask"]
            assert sha(image) == case["image_sha256"] and sha(mask) == case["mask_sha256"]
            cases.append((image, mask, case, mp.parent.name))
    report = {
        "schema": "sightguard.yoloe-quick.v1",
        "confidence": 0.35,
        "input_size": 320,
        "threads": 1,
        "cases": len(cases),
        "torch": torch.__version__,
        "protocol": (
            "Fixed 28 pilot + 60 holdout ADE validation images. "
            "Positive >=1.5% whole-image semantic GT. A box supports a group if "
            ">=10% of its known pixels have the group label; image hit additionally "
            "covers >=20% of group GT. This is coarse semantic-supported box recall, "
            "NOT instance AP. No tuning or failure removal. Latency PC batch=1, warm, "
            "including preprocessing/postprocessing."
        ),
        "models": {},
    }
    baseline = args.baseline.resolve()
    prompts = None
    for key, weights in [("yolo26n", baseline), ("yoloe26n", work / "yoloe-26n-seg.pt")]:
        started = time.perf_counter()
        if key == "yolo26n":
            model = YOLO(str(weights))
            prompts = list(model.names.values()) + EXTRA
        else:
            model = YOLOE("yoloe-26n.yaml").load(str(weights))
            model.set_classes(prompts)
            model.save_prompt_embeddings(str(work / "prompts.npz"))
            model.save(str(work / "yoloe26n-fixed.pt"))
        print(key, "loaded seconds", time.perf_counter() - started, flush=True)
        rows = []
        model.predict(
            str(cases[0][0]), imgsz=320, conf=0.35, device="cpu", verbose=False, save=False
        )
        for index, (image, mask, case, split) in enumerate(cases):
            rgb = cv2.imread(str(image))
            gt = cv2.imread(str(mask), cv2.IMREAD_UNCHANGED).astype(np.int32) - 1
            assert gt.ndim == 2
            begin = time.perf_counter()
            pred = model.predict(
                rgb, imgsz=320, conf=0.35, device="cpu", verbose=False, save=False, rect=False
            )[0]
            latency = (time.perf_counter() - begin) * 1000
            boxes = []
            covered = {g: np.zeros(gt.shape, bool) for g in GROUPS}
            for box in pred.boxes:
                cid = int(box.cls.item())
                name = model.names[cid]
                xy = box.xyxy[0].cpu().numpy()
                h, w = gt.shape
                x1, y1 = np.maximum(0, np.floor(xy[:2])).astype(int)
                x2, y2 = np.minimum([w, h], np.ceil(xy[2:])).astype(int)
                group = NAMES.get(name)
                support = None
                if group and x2 > x1 and y2 > y1:
                    crop = gt[y1:y2, x1:x2]
                    known = crop >= 0
                    support = float(np.isin(crop, list(GROUPS[group])).sum() / max(1, known.sum()))
                    if support >= 0.1:
                        covered[group][y1:y2, x1:x2] = True
                boxes.append(
                    {
                        "class_id": cid,
                        "name": name,
                        "score": float(box.conf.item()),
                        "xyxy": xy.tolist(),
                        "group": group,
                        "gt_support": support,
                    }
                )
            metrics = {}
            for group, ids in GROUPS.items():
                truth = np.isin(gt, list(ids))
                n = int(truth.sum())
                det = [b for b in boxes if b["group"] == group]
                cov = float((covered[group] & truth).sum() / max(1, n))
                metrics[group] = {
                    "gt_fraction": n / gt.size,
                    "positive": n / gt.size >= 0.015,
                    "hit": n / gt.size >= 0.015 and cov >= 0.2,
                    "gt_coverage": cov,
                    "zero_gt_prediction": n == 0 and bool(det),
                    "box_count": len(det),
                }
            rows.append(
                {
                    "image": image.name,
                    "split": split,
                    "image_sha256": case["image_sha256"],
                    "latency_ms": latency,
                    "boxes": boxes,
                    "groups": metrics,
                }
            )
            if index % 20 == 0:
                print(key, index + 1, "/", len(cases), flush=True)
        summary = {}
        for group in GROUPS:
            summary[group] = {
                "positive_images": sum(r["groups"][group]["positive"] for r in rows),
                "hit_images": sum(r["groups"][group]["hit"] for r in rows),
                "zero_gt_predictions": [
                    r["image"] for r in rows if r["groups"][group]["zero_gt_prediction"]
                ],
            }
        report["models"][key] = {
            "weights_sha256": sha(weights),
            "prompts": list(model.names.values()),
            "variant": "official segmentation checkpoint in official detection architecture"
            if key == "yoloe26n"
            else "official detection checkpoint",
            "latency_ms_p50": float(np.median([r["latency_ms"] for r in rows])),
            "latency_ms_p95": float(np.percentile([r["latency_ms"] for r in rows], 95)),
            "summary": summary,
            "results": rows,
        }
        (work / "results.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(key, json.dumps(summary), flush=True)
        del model
        gc.collect()
    print("DONE", work / "results.json", flush=True)


if __name__ == "__main__":
    main()
