"""Reuse the wall estimator on immutable same-RGB model outputs, without speech."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from sightguard.research.resources import check_resources, lower_own_priority
from sightguard.research.wall_metric import MetricDepthField, WallMetricEstimator
from sightguard.scene.spatial_grid import FrameKey


def rows(path: Path) -> dict[str, dict]:
    return {
        row["sample_id"]: row
        for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery", type=Path, required=True)
    parser.add_argument("--previous-delivery", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-utc", required=True)
    args = parser.parse_args()
    lower_own_priority()
    cv2.setNumThreads(1)
    manifests = ["tum100-manifest.json", "nyu32-manifest.json"]
    samples = [
        sample
        for name in manifests
        for sample in json.loads((args.delivery / name).read_text(encoding="utf-8"))["samples"]
    ]
    metric_records = {
        **rows(args.delivery / "tum100/metric3d-results.jsonl"),
        **rows(args.delivery / "nyu/metric3d-results.jsonl"),
    }
    results = []
    with args.output.open("x", encoding="utf-8") as log:
        for sample in samples:
            check_resources(stop_utc=args.stop_utc, output_dir=args.output.parent)
            sid = sample["sample_id"]
            current = args.delivery / ("nyu" if sample["dataset"] == "NYU" else "tum100")
            semantic_path = current / "semantics" / f"{sid}.npz"
            if sample["dataset"] == "TUM":
                semantic_path = next(
                    p / "semantics" / f"{sid}.npz"
                    for p in (
                        args.previous_delivery / "evaluation",
                        args.previous_delivery / "evaluation-extension",
                    )
                    if (p / "semantics" / f"{sid}.npz").exists()
                )
            sem = dict(np.load(semantic_path))
            depth = np.load(current / "metric3d" / f"{sid}.npz")["depth"]
            height, width = depth.shape
            key = FrameKey(sid, 0, 0.0, width, height)
            profile = metric_records[sid]
            # Offline phases ran separately. This is a hypothetical serial latency
            # profile from measured costs, never a CameraX capture-to-consume age.
            modeled_age = float(sem["processing_ms"]) + profile["processing_ms"]
            result = WallMetricEstimator().estimate(
                key,
                sem["classes"],
                sem["confidence"],
                MetricDepthField(
                    key,
                    depth,
                    "metric3dv2_vit_small_800k_focal_decanonicalized",
                    300.0,
                    source_age_ms=modeled_age,
                    model_latency_ms=profile["inference_ms"],
                ),
                semantic_key=key,
                semantic_age_ms=modeled_age,
                foreground_mask=sem["foreground"],
            )
            record = {
                "sample_id": sid,
                "dataset": sample["dataset"],
                "split": sample.get("split", "existing_tum_diagnostic"),
                "source_age_kind": "offline_hypothetical_serial_measured_cost_not_capture_age",
                "timing_evidence_level": 1,
                "model_and_image_evidence_level": 2,
                "distance_accuracy_interval": None,
                "android_validated": False,
                **result.to_dict(),
            }
            results.append(record)
            log.write(json.dumps(record) + "\n")
    summary = {
        "records": len(results),
        "statuses": dict(Counter(r["status"] for r in results)),
        "reasons": dict(Counter(reason for r in results for reason in r["rejection_reasons"])),
        "automatic_alert_enabled": False,
        "timing_is_offline_profile_not_camera_validation": True,
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
