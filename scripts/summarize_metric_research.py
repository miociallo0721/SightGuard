"""Pixel-weighted real-depth errors and complete abstention/negative-case accounting."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from sightguard.research.resources import check_resources, lower_own_priority
from sightguard.research.wall_metric import WallMetricConfig, WallMetricEstimator
from sightguard.scene.spatial_grid.schema import FrameKey

ROOT = Path(__file__).resolve().parents[1]
DELIVERY = ROOT / "android/deliveries/metric-corridor-research"


def stream(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def metric_summary(evaluations: list[Path]) -> dict:
    aggregates = defaultdict(
        lambda: {
            "frames": 0,
            "accepted_frames": 0,
            "valid_pixels": 0,
            "mae_sum": 0.0,
            "squared_sum": 0.0,
            "relative_sum": 0.0,
            "latencies": [],
            "wall_reasons": Counter(),
            "distance_bins": defaultdict(
                lambda: {"valid_pixels": 0, "mae_sum": 0.0, "squared_sum": 0.0, "relative_sum": 0.0}
            ),
        }
    )
    rows = []
    for folder in evaluations:
        for path in folder.glob("metric_*.jsonl"):
            for record in stream(path):
                for region, error in [
                    ("whole_frame", record["whole_frame_errors"]),
                    ("predicted_wall_region", record["predicted_wall_region_errors"]),
                ]:
                    group = (record["model"], record["input_size"], record["sequence"], region)
                    item = aggregates[group]
                    item["frames"] += 1
                    item["accepted_frames"] += record["wall"]["status"] == "PREDICTED_METRIC_DEPTH"
                    item["latencies"].append(record["inference_ms"])
                    item["wall_reasons"].update(record["wall"]["rejection_reasons"])
                    if region == "predicted_wall_region":
                        for name, value in record["wall_distance_bins"].items():
                            if value and value["valid_pixels"]:
                                bin_item = item["distance_bins"][name]
                                pixels = value["valid_pixels"]
                                bin_item["valid_pixels"] += pixels
                                bin_item["mae_sum"] += value["mae_m"] * pixels
                                bin_item["squared_sum"] += value["rmse_m"] ** 2 * pixels
                                bin_item["relative_sum"] += value["abs_rel"] * pixels
                    if error and error["valid_pixels"]:
                        count = error["valid_pixels"]
                        item["valid_pixels"] += count
                        item["mae_sum"] += error["mae_m"] * count
                        item["squared_sum"] += error["rmse_m"] ** 2 * count
                        item["relative_sum"] += error["abs_rel"] * count
                rows.append(
                    {
                        "sample_id": record["sample_id"],
                        "model": record["model"],
                        "input_size": record["input_size"],
                        "sequence": record["sequence"],
                        "inference_ms": record["inference_ms"],
                        "wall_status": record["wall"]["status"],
                        "wall_median_m": record["wall"]["median_m"],
                        "wall_coverage": record["wall"]["wall_coverage"],
                        "wall_valid_fraction": record["wall"]["valid_pixel_fraction"],
                        "source_age_ms": record["wall"]["source_frame_age_ms"],
                        "wall_mae_m": (record["predicted_wall_region_errors"] or {}).get("mae_m"),
                        "wall_abs_rel": (record["predicted_wall_region_errors"] or {}).get(
                            "abs_rel"
                        ),
                        "whole_mae_m": (record["whole_frame_errors"] or {}).get("mae_m"),
                        "whole_rmse_m": (record["whole_frame_errors"] or {}).get("rmse_m"),
                        "whole_abs_rel": (record["whole_frame_errors"] or {}).get("abs_rel"),
                        "rejection_reasons": "|".join(record["wall"]["rejection_reasons"]),
                    }
                )
    results = []
    for group, item in aggregates.items():
        count = item["valid_pixels"]
        results.append(
            {
                "model": group[0],
                "input_size": group[1],
                "sequence": group[2],
                "region": group[3],
                "frames": item["frames"],
                "wall_accepted_frames": item["accepted_frames"],
                "valid_pixels": count,
                "mae_m": item["mae_sum"] / count if count else None,
                "rmse_m": (item["squared_sum"] / count) ** 0.5 if count else None,
                "abs_rel": item["relative_sum"] / count if count else None,
                "inference_p50_ms": float(np.median(item["latencies"])),
                "inference_p95_ms": float(np.quantile(item["latencies"], 0.95)),
                "wall_rejection_counts": dict(item["wall_reasons"]),
                "ground_truth_z_distance_bins": {
                    name: {
                        "valid_pixels": value["valid_pixels"],
                        "mae_m": value["mae_sum"] / value["valid_pixels"],
                        "rmse_m": (value["squared_sum"] / value["valid_pixels"]) ** 0.5,
                        "abs_rel": value["relative_sum"] / value["valid_pixels"],
                    }
                    for name, value in item["distance_bins"].items()
                },
            }
        )
    with (DELIVERY / "metric_depth_results.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {
        "results": results,
        "metric_runs": len(rows),
        "aggregation": "pixel_weighted_MAE_RMSE_AbsRel_no_scale_fitting",
        "semantic_truth": "NO_ground_truth_wall_labels_errors_conditioned_on_predicted_mask",
        "relative_model": "relative_inverse_depth_no_metric_error_or_distance_conversion",
    }


def pooled_error_quantiles(evaluations: list[Path]) -> list[dict]:
    """Bounded 1mm histograms, not averages of per-frame quantiles or scale fitting."""
    histograms: dict[tuple, np.ndarray] = {}
    for folder in evaluations:
        for path in folder.glob("metric_*.jsonl"):
            cache = folder / path.stem.removeprefix("metric_")
            for record in stream(path):
                check_resources(output_dir=DELIVERY)
                if not record["depth_association_valid"]:
                    continue
                with np.load(cache / f"{record['sample_id']}.npz") as data:
                    prediction = data["depth"]
                truth = (
                    cv2.imread(record["depth_path"], cv2.IMREAD_UNCHANGED).astype(np.float32) / 5000
                )
                key = FrameKey(**record["wall"]["key"])
                estimator = WallMetricEstimator(WallMetricConfig(**record["config"]))
                with np.load(folder / "semantics" / f"{record['sample_id']}.npz") as data:
                    wall, _, _, _ = estimator.mask(
                        key, data["classes"], data["confidence"], data["foreground"]
                    )
                valid = (
                    np.isfinite(prediction) & (prediction > 0) & np.isfinite(truth) & (truth > 0)
                )
                for region, mask in [
                    ("whole_frame", valid),
                    ("predicted_wall_region", valid & wall),
                ]:
                    group = (record["model"], record["input_size"], record["sequence"], region)
                    histogram = histograms.setdefault(group, np.zeros(100001, np.int64))
                    error = np.abs(prediction[mask].astype(np.float64) - truth[mask])
                    indices = np.minimum((error * 1000).astype(np.int64), 100000)
                    histogram += np.bincount(indices, minlength=len(histogram))
    results = []
    for group, histogram in histograms.items():
        count = int(histogram.sum())
        cumulative = np.cumsum(histogram)
        results.append(
            {
                "model": group[0],
                "input_size": group[1],
                "sequence": group[2],
                "region": group[3],
                "valid_pixels": count,
                "absolute_error_quantile_bin_upper_edge_m": {
                    name: (int(np.searchsorted(cumulative, max(1, int(np.ceil(count * q))))) + 1)
                    / 1000
                    if count
                    else None
                    for name, q in [("p50", 0.5), ("p90", 0.9), ("p95", 0.95)]
                },
                "histogram_bin_width_m": 0.001,
                "overflow_ge_100m": int(histogram[-1]),
                "method": "pooled_pixel_absolute_errors_1mm_histogram_no_scale_fitting",
            }
        )
    return results


def spatial_summary(evaluations: list[Path]) -> dict:
    states, reasons, truth, confusions = Counter(), Counter(), Counter(), defaultdict(Counter)
    corridors, strict, corridor_reasons = Counter(), Counter(), Counter()
    approach_rows, corridor_rows = [], []
    for folder in evaluations:
        for row in stream(folder / "wall_approach_results.jsonl"):
            states[row["state"]] += 1
            reasons[row.get("abstain_reason") or "accepted_candidate"] += 1
            label = row["ground_truth_pair"]["label"]
            truth[label] += 1
            confusions[label][row["state"]] += 1
            approach_rows.append(row)
        for row in stream(folder / "corridor_candidates.jsonl"):
            corridors[row["status"]] += 1
            strict[row["strict_d_status"]] += 1
            for candidate in row["candidates"]:
                corridor_reasons.update(candidate["rejection_reasons"])
            corridor_rows.append(row)
    for name, rows in [
        ("wall_approach_results.jsonl", approach_rows),
        ("corridor_candidates.jsonl", corridor_rows),
    ]:
        with (DELIVERY / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    return {
        "approach_states": dict(states),
        "approach_reasons": dict(reasons),
        "mocap_pair_labels": dict(truth),
        "approach_mocap_cross_table": {key: dict(value) for key, value in confusions.items()},
        "corridor_states": dict(corridors),
        "strict_d_states": dict(strict),
        "corridor_rejections": dict(corridor_reasons),
        "body_approach_verified": False,
        "safe_or_passable_verified": False,
    }


def interval_research(evaluations: list[Path]) -> dict:
    """Sequence-held-out descriptive error bands; no calibrated/phone validity claim."""
    estimator = WallMetricEstimator()
    measurements = []
    for folder in evaluations:
        for record in stream(folder / "metric_hypersim_280.jsonl"):
            check_resources(output_dir=DELIVERY)
            if (
                record["wall"]["status"] != "PREDICTED_METRIC_DEPTH"
                or not record["depth_association_valid"]
            ):
                continue
            key_data = record["wall"]["key"]
            key = FrameKey(**key_data)
            with np.load(folder / "semantics" / f"{record['sample_id']}.npz") as data:
                mask, _, _, _ = estimator.mask(
                    key, data["classes"], data["confidence"], data["foreground"]
                )
            truth = cv2.imread(record["depth_path"], cv2.IMREAD_UNCHANGED).astype(np.float32) / 5000
            gt = truth[mask & (truth > 0) & np.isfinite(truth)]
            if len(gt) < 64:
                continue
            measurements.append(
                {
                    "sample_id": record["sample_id"],
                    "sequence": record["sequence"],
                    "predicted_median_z_m": record["wall"]["median_m"],
                    "gt_median_same_predicted_wall_pixels_z_m": float(np.median(gt)),
                    "valid_truth_fraction": len(gt) / max(1, mask.sum()),
                }
            )
    calibration = [row for row in measurements if row["sequence"].startswith("freiburg1")]
    held_out = [row for row in measurements if row["sequence"].startswith("freiburg3")]
    errors = [
        abs(row["predicted_median_z_m"] - row["gt_median_same_predicted_wall_pixels_z_m"])
        for row in calibration
    ]
    bound = float(np.quantile(errors, 0.9, method="higher")) if errors else None
    coverage = (
        np.mean(
            [
                abs(row["predicted_median_z_m"] - row["gt_median_same_predicted_wall_pixels_z_m"])
                <= bound
                for row in held_out
            ]
        )
        if bound is not None and held_out
        else None
    )
    return {
        "status": "RESEARCH_ONLY_NOT_CALIBRATED",
        "automatic_output_enabled": False,
        "calibration_sequences": sorted({row["sequence"] for row in calibration}),
        "held_out_sequences": sorted({row["sequence"] for row in held_out}),
        "calibration_frames": len(calibration),
        "held_out_frames": len(held_out),
        "empirical_p90_absolute_median_error_bound_m": bound,
        "held_out_empirical_coverage": float(coverage) if coverage is not None else None,
        "interval_construction": "max(0,predicted_median-bound),predicted_median+bound",
        "rejection_reasons": [
            "few_correlated_sequences_not_exchangeable",
            "no_ground_truth_semantic_wall_annotation",
            "no_measured_physical_phone_wall_distance",
            "no_claim_of_90_percent_coverage_on_new_domains",
        ],
        "measurements": measurements,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluations",
        nargs="+",
        type=Path,
        default=[DELIVERY / "evaluation", DELIVERY / "evaluation-extension"],
    )
    args = parser.parse_args()
    lower_own_priority()
    cv2.setNumThreads(1)
    result = {
        "schema_version": "sightguard.metric-corridor-validation.v1",
        "metric": metric_summary(args.evaluations),
        "pooled_error_quantiles": pooled_error_quantiles(args.evaluations),
        "spatial": spatial_summary(args.evaluations),
        "interval_research": interval_research(args.evaluations),
    }
    (DELIVERY / "pc_validation_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "metric_runs": result["metric"]["metric_runs"],
                **result["spatial"],
                "interval_bound_m": result["interval_research"][
                    "empirical_p90_absolute_median_error_bound_m"
                ],
                "held_out_coverage": result["interval_research"]["held_out_empirical_coverage"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
