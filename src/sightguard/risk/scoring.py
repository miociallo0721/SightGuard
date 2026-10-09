"""Explainable weighted scoring, optional causal EMA, and asymmetric hysteresis."""

import math
from dataclasses import dataclass

from sightguard.risk.config import ScoringConfig
from sightguard.risk.features import clamp
from sightguard.risk.schema import RiskAssessment, RiskFeatures, RiskLevel


@dataclass
class _ScoreState:
    score: float
    level: RiskLevel
    last_ms: float
    pending_level: RiskLevel | None = None
    pending_since_ms: float | None = None


class RiskScorer:
    def __init__(self, config: ScoringConfig) -> None:
        self.config = config
        self.states: dict[str, _ScoreState] = {}

    def forget(self, event_id: str) -> None:
        self.states.pop(event_id, None)

    def interrupt(self, event_id: str) -> None:
        state = self.states.get(event_id)
        if state is not None:
            state.pending_level = None
            state.pending_since_ms = None

    def _level(self, score: float) -> RiskLevel:
        if score >= self.config.high_threshold:
            return RiskLevel.HIGH
        if score >= self.config.medium_threshold:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def assess(self, event_id: str, features: RiskFeatures, timestamp_ms: float) -> RiskAssessment:
        c = self.config
        spatial = features.spatial.focus_relevance
        area = clamp(features.spatial.area / c.area_full_score)
        bottom = clamp(
            (features.spatial.bottom_center_xy[1] - c.bottom_start)
            / (c.bottom_full_score - c.bottom_start)
        )
        appearance = c.appearance_area_weight * area + (1 - c.appearance_area_weight) * bottom
        motion = None
        if features.quality.motion_reliable:
            growth = clamp(max(0, features.scale_change_per_s) / c.scale_rate_full_score)
            inward = clamp(max(0, features.inward_change_per_s) / c.inward_rate_full_score)
            motion = c.motion_scale_weight * growth + (1 - c.motion_scale_weight) * inward
        numerator = c.spatial_weight * spatial + c.appearance_weight * appearance
        if motion is None:
            raw = numerator / (c.spatial_weight + c.appearance_weight)
        else:
            raw = numerator + c.motion_weight * motion
        raw = clamp(raw)
        previous = self.states.get(event_id)
        score = raw
        if previous is not None and c.smoothing_tau_ms > 0:
            alpha = 1 - math.exp(-(timestamp_ms - previous.last_ms) / c.smoothing_tau_ms)
            score = previous.score + alpha * (raw - previous.score)
        reasons = []
        if spatial > 0:
            reasons.append("image_focus_relevance")
        if appearance > 0:
            reasons.append("image_scale_and_bottom_position")
        if motion is None:
            reasons.append("motion_unavailable_renormalized_spatial_appearance_weights")
        elif motion > 0:
            reasons.append("positive_scale_or_inward_image_change")
        else:
            reasons.append("no_positive_motion_evidence_static_risk_retained")
        target = self._level(score)
        state = previous or _ScoreState(score, target, timestamp_ms)
        if previous is not None:
            if features.quality.interrupted:
                self.interrupt(event_id)
            if target.rank > state.level.rank:
                upgrade_target = self._level(max(0, score - c.upgrade_margin))
                if upgrade_target.rank > state.level.rank:
                    state.level = upgrade_target
                state.pending_level = None
                state.pending_since_ms = None
            elif target.rank < state.level.rank:
                lower = self._level(min(1, score + c.downgrade_margin))
                if lower.rank < state.level.rank:
                    if state.pending_level != lower:
                        state.pending_level, state.pending_since_ms = lower, timestamp_ms
                    if timestamp_ms - state.pending_since_ms >= c.downgrade_hold_ms:
                        state.level = lower
                        state.pending_level, state.pending_since_ms = None, None
                else:
                    state.pending_level, state.pending_since_ms = None, None
            else:
                state.pending_level, state.pending_since_ms = None, None
            if state.level != target:
                reasons.append("risk_level_held_by_hysteresis")
        state.score, state.last_ms = score, timestamp_ms
        self.states[event_id] = state
        return RiskAssessment(
            spatial,
            appearance,
            motion,
            raw,
            score,
            state.level,
            features.quality,
            motion is None,
            tuple(reasons),
        )
