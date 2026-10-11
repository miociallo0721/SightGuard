"""Causal image windows and research freshness, without navigation decisions."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from sightguard.scene.motion.schema import (
    FrameMotionEstimate,
    MotionConfig,
    MotionQuality,
    MotionWindowSummary,
)
from sightguard.scene.spatial_grid.schema import FrameKey


def summarize_window(
    key: FrameKey,
    stream_id: str,
    estimates: list[FrameMotionEstimate],
    config: MotionConfig | None = None,
) -> MotionWindowSummary:
    cfg = config or MotionConfig()
    start, end = key.timestamp_ms - cfg.window_ms, key.timestamp_ms
    mismatch = any(x.key.video_id != key.video_id or x.stream_id != stream_id for x in estimates)
    selected = (
        sorted(
            (x for x in estimates if start < x.key.timestamp_ms <= end),
            key=lambda x: x.key.timestamp_ms,
        )
        if not mismatch
        else []
    )
    identities = [x.key.frame_index for x in selected]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate motion frame in temporal window")
    valid = [
        x
        for x in selected
        if x.estimation_status == "VALID"
        and x.previous_key is not None
        and x.previous_key.video_id == key.video_id
        and (x.previous_key.width, x.previous_key.height) == (key.width, key.height)
        and x.previous_key.timestamp_ms >= start
        and x.key.frame_index == x.previous_key.frame_index + 1
        and x.dt_ms is not None
        and cfg.min_dt_ms <= x.dt_ms <= cfg.max_dt_ms
        and math.isclose(x.dt_ms, x.key.timestamp_ms - x.previous_key.timestamp_ms, abs_tol=1e-6)
    ]
    intervals = sorted((x.previous_key.timestamp_ms, x.key.timestamp_ms) for x in valid)
    covered = 0.0
    cursor = start
    gaps = []
    for a, b in intervals:
        if a > cursor:
            gaps.append(a - cursor)
        covered += max(0, b - max(cursor, a))
        cursor = max(cursor, b)
    if cursor < end:
        gaps.append(end - cursor)
    coverage = min(1, covered / cfg.window_ms)
    fraction = len(valid) / len(selected) if selected else None
    rates = [
        x.motion_diagonals_per_second for x in valid if x.motion_diagonals_per_second is not None
    ]
    p50, p95 = (float(np.percentile(rates, p)) for p in (50, 95)) if rates else (None, None)
    labels = dict(Counter(x.quality.label for x in selected))
    reasons = []
    if mismatch or not stream_id:
        reasons.append("window_sequence_identity_mismatch")
    if not selected:
        reasons.append("no_past_motion_samples")
    if fraction is None or fraction < cfg.min_window_valid_fraction:
        reasons.append("insufficient_window_valid_estimates")
    if coverage < cfg.min_window_time_coverage:
        reasons.append("incomplete_valid_time_window")
    if gaps and max(gaps) > cfg.max_window_gap_ms:
        reasons.append("motion_window_gap")
    if not rates:
        reasons.append("motion_rate_unavailable")
    if reasons:
        label = "UNKNOWN"
    else:
        if len(valid) < len(selected) or coverage < 1 - 1e-6 or labels.get("LIMITED", 0):
            reasons.append("partial_or_limited_motion_window")
        label = (
            "LIMITED"
            if reasons
            else (
                "STABLE_IMAGE_MOTION"
                if p95 <= cfg.stable_diagonals_per_second
                else "RAPID_IMAGE_MOTION"
                if p95 >= cfg.rapid_diagonals_per_second
                else "MODERATE_IMAGE_MOTION"
            )
        )
    return MotionWindowSummary(
        key,
        stream_id,
        start,
        end,
        min((x.previous_key.timestamp_ms for x in valid), default=None),
        max((x.key.timestamp_ms for x in valid), default=None),
        len(selected),
        len(valid),
        fraction,
        coverage,
        max(gaps, default=0.0),
        p50,
        p95,
        labels,
        tuple(identities),
        MotionQuality(label, tuple(reasons)),
        tuple(
            str(x.provenance.get("record_reference", f"frame_motion:{x.key.frame_index}"))
            for x in selected
        ),
    )


@dataclass(frozen=True)
class FreshnessAssessment:
    source_key: FrameKey
    stream_id: str
    consumer_timestamp_ms: float
    age_ms: float | None
    ttl_ms: float | None
    status: str
    reuse_allowed_for_research: bool
    reasons: tuple[str, ...]
    consumer_time_kind: str = "SIMULATED_VIDEO_PTS_NOT_ONLINE_LATENCY"
    absolute_camera_pose_status: str = "UNKNOWN"
    schema_version: str = "sightguard.freshness.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_freshness(
    source: dict[str, Any],
    window: MotionWindowSummary,
    consumer_timestamp_ms: float,
    stream_id: str,
    config: MotionConfig | None = None,
) -> FreshnessAssessment:
    cfg = config or MotionConfig()
    key = FrameKey(**source["key"])
    reasons = []
    age = consumer_timestamp_ms - key.timestamp_ms if math.isfinite(consumer_timestamp_ms) else None
    ttl = None
    if key != window.key or stream_id != window.stream_id or not stream_id:
        status = "INVALID"
        reasons.append("freshness_source_window_identity_mismatch")
    elif age is None or age < 0:
        status = "INVALID"
        reasons.append("consumer_time_invalid_or_before_source")
    elif source["status"] == "ABSTAIN":
        status = "INVALID"
        reasons.append("source_corridor_abstained")
    elif window.quality.label in ("UNKNOWN", "LIMITED"):
        status = "UNKNOWN"
        reasons.extend(("motion_quality_does_not_support_reuse", *window.quality.reasons))
    else:
        ttl = {
            "STABLE_IMAGE_MOTION": cfg.fresh_stable_ms,
            "MODERATE_IMAGE_MOTION": cfg.fresh_moderate_ms,
            "RAPID_IMAGE_MOTION": cfg.fresh_rapid_ms,
        }.get(window.quality.label)
        if ttl is None:
            status = "UNKNOWN"
            reasons.append("unrecognized_motion_quality")
        else:
            status = "FRESH" if age <= ttl else "STALE"
            reasons.append(
                "engineering_image_motion_ttl" if status == "FRESH" else "source_evidence_expired"
            )
    return FreshnessAssessment(
        key, stream_id, consumer_timestamp_ms, age, ttl, status, status == "FRESH", tuple(reasons)
    )


def diagnose_change(
    current: dict[str, Any],
    previous: dict[str, Any] | None,
    config: MotionConfig | None = None,
) -> dict[str, Any]:
    """Disentangle cost order, support status and evidence changes, not direction jitter."""
    cfg = config or MotionConfig()
    if previous is None:
        return {
            "types": ["INITIAL_OBSERVATION"],
            "candidate_deltas": [],
            "lowest_cost_changed": False,
            "status_changed": False,
            "best_supported_identity_changed": False,
        }
    if current["key"]["video_id"] != previous["key"]["video_id"]:
        raise ValueError("cross-video temporal comparison")
    if (
        current["ablation"] != previous["ablation"]
        or current["key"]["timestamp_ms"] <= previous["key"]["timestamp_ms"]
    ):
        raise ValueError("ablation or non-monotonic temporal comparison")
    types = []
    a, b = previous["status"], current["status"]
    if a != b:
        if b == "ABSTAIN":
            types.append("RESULT_BECAME_ABSTAIN")
        elif a == "ABSTAIN":
            types.append("RESULT_RECOVERED_FROM_ABSTAIN")
        elif a == "BEST_SUPPORTED" and b == "TIED":
            types.append("BEST_SUPPORTED_TO_TIED")
        elif a == "TIED" and b == "BEST_SUPPORTED":
            types.append("TIED_TO_BEST_SUPPORTED")
    prev_low = previous["ranking"][0] if previous["ranking"] else None
    curr_low = current["ranking"][0] if current["ranking"] else None
    lowest_changed = prev_low is not None and curr_low is not None and prev_low != curr_low
    if lowest_changed:
        types.append("LOWEST_COST_CANDIDATE_CHANGED")
    if previous["best_supported_candidate"] != current["best_supported_candidate"]:
        types.append("BEST_SUPPORTED_IDENTITY_OR_NULL_CHANGED")
    old = {x["candidate"]["candidate_id"]: x for x in previous["candidates"]}
    if set(old) != {x["candidate"]["candidate_id"] for x in current["candidates"]}:
        raise ValueError("candidate geometry identities changed")
    deltas = []
    for candidate in current["candidates"]:
        name = candidate["candidate"]["candidate_id"]
        before = old[name]
        weighted = {
            term: candidate["weighted_contributions"][term] - before["weighted_contributions"][term]
            if candidate["weighted_contributions"][term] is not None
            and before["weighted_contributions"][term] is not None
            else None
            for term in candidate["weighted_contributions"]
        }
        evidence = {
            term: candidate["evidence"][term] - before["evidence"][term]
            for term in (
                "ground_support",
                "ground_continuity",
                "bbox_proxy_coverage",
                "near_bbox_proxy_coverage",
                "unknown_exposure",
                "conflict_coverage",
            )
        }
        if any(abs(x) >= cfg.evidence_change_threshold for x in evidence.values()):
            types.append("IMAGE_REGION_EVIDENCE_CHANGED")
        if before["eligibility_reasons"] != candidate["eligibility_reasons"]:
            types.append("QUALITY_OR_ELIGIBILITY_GATE_CHANGED")
        dominant = max(
            (term for term, value in weighted.items() if value is not None and abs(value) > 1e-12),
            key=lambda term: abs(weighted[term]),
            default=None,
        )
        deltas.append(
            {
                "candidate_id": name,
                "weighted_component_deltas": weighted,
                "evidence_deltas": evidence,
                "largest_absolute_cost_delta_term": dominant,
                "object_relation_count_before": len(before["object_relations"]),
                "object_relation_count_after": len(candidate["object_relations"]),
            }
        )
    if lowest_changed:

        def close(row):
            costs = sorted(
                x["total_cost"] for x in row["candidates"] if x["total_cost"] is not None
            )
            return len(costs) > 1 and costs[1] - costs[0] <= cfg.close_cost_margin

        if close(current) or close(previous):
            types.append("CLOSE_COST_ORDER_SWAP")
    return {
        "types": sorted(set(types)) or ["NO_CLASSIFIED_CHANGE"],
        "candidate_deltas": deltas,
        "lowest_cost_changed": lowest_changed,
        "status_changed": a != b,
        "best_supported_identity_changed": previous["best_supported_candidate"]
        != current["best_supported_candidate"],
        "dt_ms": current["key"]["timestamp_ms"] - previous["key"]["timestamp_ms"],
        "previous_key": previous["key"],
        "scope": "image_candidate_identity_not_direction_jitter_or_world_motion",
    }
