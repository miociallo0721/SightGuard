"""Evaluate frozen real predictions against public GT after inference completes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from sightguard.research.dataset_metrics import (
    DepthErrorTotals,
    binary_semantic_counts,
    summarize_semantic_counts,
)
from sightguard.research.local_corridor import LocalCorridorCandidateEvaluator
from sightguard.research.resources import check_resources, lower_own_priority
from sightguard.scene.spatial_grid import (
    EvidenceSource,
    FrameKey,
    SpatialGridFusion,
    SpatialGridInput,
)


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def verified_hash(path: Path, expected: str | None = None) -> str:
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if expected is not None and actual != expected:
        raise ValueError(f"frozen evaluation artifact changed: {path}")
    return actual


def candidate_diagnostic(
    sample: dict, sem: dict, relative: dict, gt: dict, detections: list
) -> dict:
    classes, confidence = sem["classes"], sem["confidence"]
    height, width = classes.shape
    # Each unrelated still is a separate session: never fake temporal confirmation.
    key = FrameKey(sample["scene"], 0, 0, width, height)
    source = EvidenceSource(
        key, "valid", "model_observed", sample["rgb_path"], sample["rgb_sha256"]
    )
    grid = SpatialGridFusion().fuse(
        SpatialGridInput(
            key,
            source,
            source,
            source,
            classes,
            confidence,
            sem["components"],
            relative["depth"],
            tuple(detections),
        )
    )
    result = LocalCorridorCandidateEvaluator().evaluate(
        grid,
        classes,
        confidence,
        semantic_key=key,
        source_age_ms=float(sem["processing_ms"]) + float(relative["latency_ms"]),
        motion_quality="UNKNOWN",
    )
    # This GT comparison is appended only after the unchanged evaluator returns.
    for candidate in result["candidates"]:
        mask = np.zeros(classes.shape, bool)
        for x1, y1, x2, y2 in candidate["image_row_bounds_xyxy"]:
            mask[y1:y2, x1:x2] = True
        labeled = mask & (gt["labels"] > 0)
        count = int(labeled.sum())
        candidate["independent_gt_evaluation"] = {
            "labeled_pixels": count,
            "floor_fraction_of_labeled": float(np.mean(gt["labels"][labeled] == 11))
            if count
            else None,
            "wall_fraction_of_labeled": float(np.mean(gt["labels"][labeled] == 21))
            if count
            else None,
            "unknown_label_fraction": float(np.mean(gt["labels"][mask] == 0))
            if mask.any()
            else None,
            "not_passability_ground_truth": True,
        }
    return {"sample_id": sample["sample_id"], "single_still_no_temporal_validation": True, **result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery", type=Path, required=True)
    parser.add_argument("--previous-delivery", type=Path, required=True)
    parser.add_argument("--nyu-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-utc", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    lower_own_priority()
    cv2.setNumThreads(1)
    samples = json.loads((args.delivery / "tum100-manifest.json").read_text())["samples"]
    if args.nyu_manifest:
        nyu_manifest = json.loads(args.nyu_manifest.read_text())
        if nyu_manifest["labels"] != {"wall": 21, "floor": 11}:
            raise ValueError("NYU runtime class mapping changed")
        samples += nyu_manifest["samples"]
    totals: dict[tuple, DepthErrorTotals] = defaultdict(DepthErrorTotals)
    semantic_totals: dict[tuple, Counter] = defaultdict(Counter)
    rows, semantic_rows, candidates = [], [], []
    provenance = []
    cached_logs: dict[Path, dict[str, dict]] = {}
    for sample in samples:
        check_resources(stop_utc=args.stop_utc, output_dir=args.output)
        sid, dataset = sample["sample_id"], sample["dataset"]
        artifact_hashes = {"rgb": verified_hash(Path(sample["rgb_path"]), sample["rgb_sha256"])}
        split = sample.get("split", "existing_tum_diagnostic")
        group = sample.get("scene", sample.get("sequence"))
        if dataset == "TUM":
            artifact_hashes["truth"] = verified_hash(
                Path(sample["depth_path"]), sample["depth_sha256"]
            )
            previous = next(
                p
                for p in (
                    args.previous_delivery / "evaluation",
                    args.previous_delivery / "evaluation-extension",
                )
                if (p / "semantics" / f"{sid}.npz").exists()
            )
            sem = dict(np.load(previous / "semantics" / f"{sid}.npz"))
            artifact_hashes["semantic"] = verified_hash(previous / "semantics" / f"{sid}.npz")
            for filename in (
                "semantic_results.jsonl",
                "metric_hypersim_280.jsonl",
                "metric_hypersim_518.jsonl",
            ):
                log_path = previous / filename
                if log_path not in cached_logs:
                    cached_logs[log_path] = {row["sample_id"]: row for row in read_rows(log_path)}
                source_record = cached_logs[log_path][sid]
                if source_record["rgb_sha256"] != sample["rgb_sha256"]:
                    raise ValueError("historical model cache came from different RGB")
                if filename.startswith("metric_") and (
                    source_record["depth_sha256"] != sample["depth_sha256"]
                ):
                    raise ValueError("historical evaluation used different depth truth")
            predictions = {
                "metric3d": args.delivery / "tum100/metric3d" / f"{sid}.npz",
                "hypersim280": previous / "hypersim_280" / f"{sid}.npz",
                "hypersim518": previous / "hypersim_518" / f"{sid}.npz",
            }
            truth = cv2.imread(sample["depth_path"], cv2.IMREAD_UNCHANGED).astype(np.float32) / 5000
            truths = {"raw_registered": truth}
            gt = None
        else:
            artifact_hashes["truth"] = verified_hash(Path(sample["gt_path"]), sample["gt_sha256"])
            current = args.delivery / "nyu"
            sem = dict(np.load(current / "semantics" / f"{sid}.npz"))
            artifact_hashes["semantic"] = verified_hash(current / "semantics" / f"{sid}.npz")
            predictions = {
                name: current / name / f"{sid}.npz"
                for name in ["metric3d", "hypersim280", "hypersim518"]
            }
            gt = dict(np.load(sample["gt_path"]))
            truths = {"raw_valid": gt["raw_depth_m"], "inpainted_separate": gt["filled_depth_m"]}
        shape = sem["classes"].shape
        regions = {
            "all_pixels": np.ones(shape, bool),
            "predicted_wall": (sem["classes"] == 0) & (sem["confidence"] >= 0.55),
            "predicted_ground": np.isin(sem["classes"], [3, 6, 11]) & (sem["confidence"] >= 0.55),
        }
        if gt is not None:
            regions.update(gt_wall=gt["labels"] == 21, gt_floor=gt["labels"] == 11)
            for name, predicted in [
                ("wall", sem["classes"] == 0),
                ("floor", np.isin(sem["classes"], [3, 6, 11])),
            ]:
                for threshold in [0.35, 0.55]:
                    counts = binary_semantic_counts(
                        predicted & (sem["confidence"] >= threshold),
                        gt["labels"] == (21 if name == "wall" else 11),
                        gt["labels"] > 0,
                    )
                    semantic_totals[(split, name, threshold)].update(counts)
                    semantic_rows.append(
                        {
                            "sample_id": sid,
                            "scene": group,
                            "split": split,
                            "class": name,
                            "threshold": threshold,
                            **summarize_semantic_counts(counts),
                        }
                    )
            relative = dict(np.load(args.delivery / "nyu/relative_280" / f"{sid}.npz"))
            detection_rows = read_rows(args.delivery / "nyu/semantic_results.jsonl")
            detections = next(
                row["detections"] for row in detection_rows if row["sample_id"] == sid
            )
            candidates.append(candidate_diagnostic(sample, sem, relative, gt, detections))
        for model, path in predictions.items():
            artifact_hashes[model] = verified_hash(path)
            prediction = np.load(path)["depth"]
            for truth_kind, truth in truths.items():
                for region_name, region in regions.items():
                    current = DepthErrorTotals()
                    current.add(prediction, truth, region)
                    valid = (
                        region
                        & np.isfinite(truth)
                        & (truth > 0)
                        & np.isfinite(prediction)
                        & (prediction > 0)
                    )
                    errors = np.abs(prediction[valid] - truth[valid])
                    row = {
                        "sample_id": sid,
                        "dataset": dataset,
                        "scene": group,
                        "split": split,
                        "model": model,
                        "truth": truth_kind,
                        "region": region_name,
                        **current.summary(),
                        "absolute_error_p50_m": float(np.quantile(errors, 0.5))
                        if errors.size
                        else None,
                        "absolute_error_p90_m": float(np.quantile(errors, 0.9))
                        if errors.size
                        else None,
                    }
                    rows.append(row)
                    for category in ["all_scenes", group]:
                        totals[
                            (dataset, split, category, model, truth_kind, region_name, "all")
                        ].add(prediction, truth, region)
                    for low, high in [
                        (0, 0.5),
                        (0.5, 1),
                        (1, 2),
                        (2, 3),
                        (3, 5),
                        (5, 10),
                        (10, float("inf")),
                    ]:
                        band = region & (truth > low) & (truth <= high)
                        totals[
                            (
                                dataset,
                                split,
                                "all_scenes",
                                model,
                                truth_kind,
                                region_name,
                                f"({low},{high}]",
                            )
                        ].add(prediction, truth, band)
        provenance.append({"sample_id": sid, "sha256": artifact_hashes})
        print(sid, "evaluated", flush=True)
    with (args.output / "depth_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summaries = []
    for key, total in totals.items():
        dataset, split, group, model, truth_kind, region, band = key
        summaries.append(
            {
                "dataset": dataset,
                "split": split,
                "scene": group,
                "model": model,
                "truth": truth_kind,
                "region": region,
                "gt_distance_bin_m": band,
                **total.summary(),
            }
        )
    semantic_summary = [
        {"split": key[0], "class": key[1], "threshold": key[2], **summarize_semantic_counts(count)}
        for key, count in semantic_totals.items()
    ]
    summary = {
        "schema_version": "sightguard.dataset-evaluation.v1",
        "samples": len(samples),
        "counts_by_dataset": dict(Counter(s["dataset"] for s in samples)),
        "aggregation": "pixel_weighted_raw_valid_and_inpainted_separate",
        "gt_only_used_after_inference": True,
        "android_validation": False,
        "metrics": summaries,
        "semantics": semantic_summary,
        "candidate_statuses": dict(Counter(c["status"] for c in candidates)),
        "strict_d_statuses": dict(Counter(c["strict_d_status"] for c in candidates)),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for name, records in [
        ("semantic_results.jsonl", semantic_rows),
        ("corridor_diagnostics.jsonl", candidates),
        ("evaluated_artifact_hashes.jsonl", provenance),
    ]:
        (args.output / name).write_text(
            "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
