"""Temporal image candidates reusing frozen Corridor/Spatial Grid; strict D is intact."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from sightguard.scene.corridor.engine import CorridorEvaluator
from sightguard.scene.spatial_grid.schema import FrameKey, SpatialGridFrame


@dataclass(frozen=True)
class LocalCorridorConfig:
    maximum_source_age_ms: float = 500
    maximum_pair_ms: float = 500
    minimum_confirmations: int = 3
    maximum_unknown_fraction: float = 0.45
    maximum_conflict_fraction: float = 0.4
    maximum_wall_fraction: float = 0.2
    maximum_near_bbox_fraction: float = 0.4
    minimum_semantic_confidence: float = 0.45
    tie_margin: float = 0.03

    def __post_init__(self) -> None:
        if self.minimum_confirmations < 1:
            raise ValueError("invalid corridor temporal support")
        for value in (self.maximum_source_age_ms, self.maximum_pair_ms):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("invalid corridor time threshold")
        for value in (
            self.maximum_unknown_fraction,
            self.maximum_conflict_fraction,
            self.maximum_wall_fraction,
            self.maximum_near_bbox_fraction,
            self.minimum_semantic_confidence,
            self.tie_margin,
        ):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("invalid corridor fraction")


class LocalCorridorCandidateEvaluator:
    """Image support only. Pose, body direction and body clearance remain UNKNOWN."""

    def __init__(self, config: LocalCorridorConfig | None = None) -> None:
        self.config = config or LocalCorridorConfig()
        self.core = CorridorEvaluator()
        self.previous: FrameKey | None = None
        self.previous_best: str | None = None
        self.confirmations = 0

    def reset(self) -> None:
        self.previous = None
        self.previous_best = None
        self.confirmations = 0

    def evaluate(
        self,
        grid: SpatialGridFrame,
        classes: np.ndarray,
        confidence: np.ndarray,
        *,
        semantic_key: FrameKey,
        source_age_ms: float,
        motion_quality: str,
        metric_z_m: np.ndarray | None = None,
        metric_key: FrameKey | None = None,
    ) -> dict[str, Any]:
        cfg, key = self.config, grid.key
        reasons = []
        if semantic_key != key or classes.shape != (key.height, key.width):
            reasons.append("semantic_frame_or_geometry_mismatch")
        if confidence.shape != classes.shape:
            reasons.append("semantic_confidence_geometry_mismatch")
        if not math.isfinite(source_age_ms) or not 0 <= source_age_ms <= cfg.maximum_source_age_ms:
            reasons.append("source_stale")
        if motion_quality in ("RAPID_IMAGE_MOTION", "DISCONTINUITY"):
            reasons.append("rapid_motion_or_scene_change")
        if grid.source_status.get("semantic") != "valid":
            reasons.append("semantic_source_unavailable")
        if grid.source_status.get("detection") != "valid":
            reasons.append("detection_source_unavailable")
        if self.previous is not None and (
            self.previous.video_id != key.video_id
            or key.frame_index <= self.previous.frame_index
            or not 0 < key.timestamp_ms - self.previous.timestamp_ms <= cfg.maximum_pair_ms
            or (self.previous.width, self.previous.height) != (key.width, key.height)
        ):
            self.reset()
        # Always expose strict D, with its unchanged observed-orientation gate.
        strict = self.core.assess(grid, "D")
        base = {
            "schema_version": "sightguard.local-corridor-candidate.v1",
            "key": asdict(key),
            "scope": "CANDIDATE_ONLY",
            "strict_d_status": strict.status,
            "strict_d_reasons": strict.abstain_reasons,
            "camera_orientation": "UNKNOWN",
            "phone_body_alignment": "UNKNOWN",
            "body_clearance_width_m": None,
            "physical_corridor_width_m": None,
            "automatic_alert_enabled": False,
            "production_decision_changed": False,
            "motion_quality": motion_quality,
            "source_age_ms": source_age_ms,
            "config": asdict(cfg),
        }
        if reasons:
            self.reset()
            return {
                **base,
                "status": "ABSTAIN",
                "reasons": reasons,
                "candidates": [],
                "best_candidate": None,
                "temporal_support": 0,
            }
        ablation = "C" if grid.source_status.get("depth") == "valid" else "B"
        comparison = self.core.assess(grid, ablation)
        candidates = []
        for item in comparison.candidates:
            evidence = item.evidence
            mask = np.zeros(classes.shape, bool)
            for x1, y1, x2, y2 in item.candidate.row_bounds_xyxy:
                mask[y1:y2, x1:x2] = True
            wall_fraction = float(
                ((classes == 0) & (confidence >= 0.55) & mask).sum() / max(1, mask.sum())
            )
            rejected = list(item.eligibility_reasons)
            for value, maximum, reason in [
                (evidence.unknown_exposure, cfg.maximum_unknown_fraction, "unknown_not_free"),
                (evidence.conflict_coverage, cfg.maximum_conflict_fraction, "conflicting_sources"),
                (wall_fraction, cfg.maximum_wall_fraction, "wall_separates_image_band"),
                (
                    evidence.near_bbox_proxy_coverage,
                    cfg.maximum_near_bbox_fraction,
                    "known_obstacle_covers_lower_band",
                ),
            ]:
                if value > maximum:
                    rejected.append(reason)
            if evidence.semantic_confidence < cfg.minimum_semantic_confidence:
                rejected.append("semantic_confidence_limited")
            metric_stats = None
            if metric_z_m is not None:
                if metric_key != key or metric_z_m.shape != classes.shape:
                    rejected.append("metric_depth_frame_or_geometry_mismatch")
                else:
                    ground = mask & np.isin(classes, [3, 6, 11]) & (confidence >= 0.55)
                    valid = ground & np.isfinite(metric_z_m) & (metric_z_m > 0)
                    values = metric_z_m[valid]
                    metric_stats = {
                        "valid_ground_pixels": int(values.size),
                        "median_predicted_z_m": float(np.median(values)) if values.size else None,
                        "purpose": "research_distribution_not_clearance_or_width",
                    }
            candidates.append(
                {
                    "candidate_id": item.candidate.candidate_id,
                    "image_row_bounds_xyxy": item.candidate.row_bounds_xyxy,
                    "image_polygon_xy": item.candidate.polygon_xy,
                    "evidence": asdict(evidence),
                    "wall_fraction": wall_fraction,
                    "visible_width_fraction_by_row": [
                        (x2 - x1) / key.width for x1, _, x2, _ in item.candidate.row_bounds_xyxy
                    ],
                    "relative_cost": item.total_cost,
                    "metric_ground": metric_stats,
                    "rejection_reasons": rejected,
                    "eligible": not rejected and item.total_cost is not None,
                }
            )
        eligible = sorted(
            (item for item in candidates if item["eligible"]),
            key=lambda item: (item["relative_cost"], item["candidate_id"]),
        )
        tied = [
            item["candidate_id"]
            for item in eligible
            if item["relative_cost"] - eligible[0]["relative_cost"] <= cfg.tie_margin
        ]
        best = eligible[0]["candidate_id"] if len(tied) == 1 else None
        self.confirmations = (
            self.confirmations + 1 if best and best == self.previous_best else (1 if best else 0)
        )
        self.previous, self.previous_best = key, best
        status = (
            "INSUFFICIENT_EVIDENCE"
            if not eligible
            else "TIED"
            if len(tied) > 1
            else "SUPPORTED_CANDIDATE"
            if self.confirmations >= cfg.minimum_confirmations
            else "INSUFFICIENT_EVIDENCE"
        )
        return {
            **base,
            "status": status,
            "candidates": candidates,
            "best_candidate": best,
            "tied_candidates": tied if len(tied) > 1 else [],
            "temporal_support": self.confirmations,
            "comparison_ablation": ablation,
            "reasons": ["not_proof_of_body_passage", "pose_and_body_direction_unknown"],
        }
