"""Summarize retained Android model results without removing misses or false positives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fetch_furniture_validation import GROUPS


def summarize(report: dict) -> dict:
    """The preregistered gate is a small-sample enablement check, not safety accuracy."""
    groups = {}
    cases = report["cases"]
    audio_ids = {e["event"] for e in report["audio"] if e["phase"] == "onStart"}
    for group in GROUPS:
        counts = {
            key: sum(c["pixels"][group][key] for c in cases) for key in ("tp", "fp", "fn", "tn")
        }
        tp, fp, fn = (counts[k] for k in ("tp", "fp", "fn"))
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        iou = tp / (tp + fp + fn) if tp + fp + fn else None
        positives = [c for c in cases if c["case"]["gt_fractions"][group] >= 0.015]

        def has_event(case: dict, name: str = group) -> bool:
            return any(f":{name}:" in event for o in case["observations"] for event in o["events"])

        detected = [c["case"]["image"] for c in positives if has_event(c)]
        false_events = [
            c["case"]["image"]
            for c in cases
            if c["case"]["gt_fractions"][group] == 0 and has_event(c)
        ]
        reasons = []
        if len(positives) < 4:
            reasons.append("fewer_than_four_gt_positive_images")
        if len(detected) < len(positives) / 2:
            reasons.append("fewer_than_half_positive_images_have_fresh_confirmed_event")
        if precision is None or precision < 0.60:
            reasons.append("pixel_precision_below_060")
        if iou is None or iou < 0.30:
            reasons.append("pixel_iou_below_030")
        if false_events:
            reasons.append("event_on_image_without_any_gt_class_pixels")
        if not any(f":{group}:" in event for event in audio_ids):
            reasons.append("no_real_model_event_audio_onStart")
        groups[group] = {
            "pixels": counts,
            "precision": precision,
            "recall": recall,
            "iou": iou,
            "gt_positive_images": len(positives),
            "fresh_event_images": detected,
            "missed_positive_images": [c["case"]["image"] for c in positives if not has_event(c)],
            "false_event_images": false_events,
            "pilot_gate_pass": not reasons,
            "rejection_reasons": reasons,
        }
    timings = [o for c in cases for o in c["observations"]]

    def percentile(field: str, fraction: float):
        values = sorted(o[field] for o in timings)
        return values[round((len(values) - 1) * fraction)] if values else None

    return {
        "schema": "sightguard.furniture-validation-summary.v1",
        "device": report["device"],
        "sdk": report["sdk"],
        "model_sha256": report["model_sha256"],
        "model_input_size": report.get("model_input_size", 256),
        "family_confidence": report.get("family_confidence", False),
        "case_count": len(cases),
        "scope": report["evidence_level"],
        "metric_definition": report["metric"],
        "groups": groups,
        "model_load_ms": report["model_load_ms"],
        "latency_ms_p50": percentile("latency_ms", 0.5),
        "latency_ms_p95": percentile("latency_ms", 0.95),
        "source_age_ms_p95": percentile("source_age_ms", 0.95),
        "component_measure_ms_p95": percentile("measure_ms", 0.95),
        "peak_sampled_pss_kb": max((o["pss_kb"] for o in timings), default=None),
        "fresh_observation_fraction": sum(o["source_age_ms"] <= 500 for o in timings) / len(timings)
        if timings
        else None,
        "audio_onStart_count": sum(e["phase"] == "onStart" for e in report["audio"]),
        "warning": (
            "Repeated validation stills, same ADE20K domain used for model training. "
            "No live field accuracy, metric ranging, motion, or body clearance validation."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = summarize(json.loads(args.report.read_text(encoding="utf-8-sig")))
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    for name, group in result["groups"].items():
        print(name, "PASS" if group["pilot_gate_pass"] else "HOLD", group["rejection_reasons"])


if __name__ == "__main__":
    main()
