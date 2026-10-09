from dataclasses import replace

import pytest

from sightguard.data import Detection, FrameRecord
from sightguard.risk.config import FeatureConfig, RiskConfig
from sightguard.risk.features import FeatureExtractor, spatial_features
from sightguard.risk.schema import (
    PerceptionQuality,
    RiskFeatures,
    RiskLevel,
    SpatialFeatures,
)
from sightguard.risk.scoring import RiskScorer


def _detection(bbox: tuple[float, float, float, float], track_id: int | None = 7) -> Detection:
    return Detection(track_id, 0, "person", 0.9, bbox)


def _frame(index: int, timestamp_ms: float, detection: Detection) -> FrameRecord:
    return FrameRecord(index, timestamp_ms, 640, 480, (detection,), timestamp_source="video_pts")


def _step_features(
    extractor: FeatureExtractor,
    index: int,
    timestamp_ms: float,
    detection: Detection,
    event_id: str = "E000001",
):
    frame = _frame(index, timestamp_ms, detection)
    extractor.begin_frame(frame, complete=True)
    spatial = spatial_features(detection, 640, 480, extractor.config)
    return extractor.observe(event_id, detection, frame, spatial)


def _synthetic_features(
    *, focus: float, area: float, bottom_y: float, scale_rate: float, inward_rate: float
) -> RiskFeatures:
    spatial = SpatialFeatures(
        (0.4, 0.4, 0.6, 0.9),
        (0.5, 0.65),
        0.2,
        0.5,
        area,
        (0.5, bottom_y),
        focus,
        False,
    )
    quality = PerceptionQuality(0.9, 1.0, 3, 3, False, False, True, "video_pts", 0.9, ())
    return RiskFeatures(spatial, 500.0, 250.0, scale_rate, (0.0, 0.0), inward_rate, 1.0, quality)


def test_spatial_features_normalize_coordinates_and_rank_focus_region() -> None:
    config = FeatureConfig()
    centered = spatial_features(_detection((280, 300, 360, 450)), 640, 480, config)
    lateral = spatial_features(_detection((10, 300, 90, 450)), 640, 480, config)

    assert centered.center_xy == pytest.approx((0.5, 0.78125))
    assert centered.width == pytest.approx(0.125)
    assert centered.height == pytest.approx(0.3125)
    assert centered.area == pytest.approx(0.0390625)
    assert centered.bottom_center_xy == pytest.approx((0.5, 0.9375))
    assert centered.focus_relevance == 1.0
    assert lateral.focus_relevance < centered.focus_relevance


def test_lateral_motion_uses_video_time_and_focus_relevance_changes_causally() -> None:
    extractor = FeatureExtractor(FeatureConfig(motion_smoothing_tau_ms=0))
    left = _step_features(extractor, 0, 0, _detection((0, 300, 64, 420)))
    middle = _step_features(extractor, 1, 250, _detection((160, 300, 224, 420)))
    center = _step_features(extractor, 2, 700, _detection((288, 300, 352, 420)))

    assert (
        left.spatial.focus_relevance
        < middle.spatial.focus_relevance
        <= center.spatial.focus_relevance
    )
    assert middle.center_change_per_s == pytest.approx((1.0, 0.0))
    assert center.center_change_per_s == pytest.approx((0.4444444, 0.0))
    assert middle.observation_interval_ms == 250
    assert center.observation_interval_ms == 450


def test_scale_growth_rate_uses_irregular_timestamp_intervals() -> None:
    extractor = FeatureExtractor(FeatureConfig(motion_smoothing_tau_ms=0))
    _step_features(extractor, 0, 20, _detection((300, 300, 320, 340)))
    doubled = _step_features(extractor, 1, 220, _detection((290, 290, 330, 370)))
    redoubled = _step_features(extractor, 2, 570, _detection((270, 250, 350, 410)))

    assert doubled.scale_change_per_s == pytest.approx(3.4657359)
    assert redoubled.scale_change_per_s == pytest.approx(1.9804209)
    assert doubled.quality.motion_reliable is True
    assert doubled.observation_interval_ms == 200
    assert redoubled.observation_interval_ms == 350


def test_high_frame_rate_keeps_observation_interval_and_accumulates_motion_baseline() -> None:
    extractor = FeatureExtractor(FeatureConfig(motion_smoothing_tau_ms=0))
    _step_features(extractor, 0, 0, _detection((300, 300, 320, 340)))
    early = _step_features(extractor, 1, 5, _detection((299, 299, 321, 341)))
    accumulated = _step_features(extractor, 2, 10, _detection((298, 298, 322, 342)))

    assert early.observation_interval_ms == 5
    assert early.quality.motion_reliable is False
    assert accumulated.observation_interval_ms == 5
    assert accumulated.motion_interval_ms == 10
    assert accumulated.quality.motion_reliable is True
    assert accumulated.scale_change_per_s == pytest.approx(13.89, rel=0.01)


def test_stationary_observation_has_no_positive_motion_evidence() -> None:
    extractor = FeatureExtractor(FeatureConfig(motion_smoothing_tau_ms=0))
    _step_features(extractor, 0, 0, _detection((280, 300, 360, 450)))
    same = _step_features(extractor, 1, 200, _detection((280, 300, 360, 450)))

    assert same.scale_change_per_s == pytest.approx(0.0)
    assert same.center_change_per_s == pytest.approx((0.0, 0.0))
    assert same.inward_change_per_s == pytest.approx(0.0)
    assert same.quality.motion_reliable is True


def test_missing_track_id_and_unavailable_motion_remain_explicit() -> None:
    extractor = FeatureExtractor(FeatureConfig())
    features = _step_features(extractor, 0, 0, _detection((280, 300, 360, 450), None))

    assert features.quality.track_id_missing is True
    assert "missing_track_id" in features.quality.limitations
    assert features.quality.motion_reliable is False
    assert features.scale_change_per_s is None
    assert "motion_unavailable" in features.quality.limitations


def test_missing_motion_renormalizes_spatial_appearance_and_quality_is_separate() -> None:
    extractor = FeatureExtractor(FeatureConfig())
    features = _step_features(extractor, 0, 0, _detection((280, 300, 360, 450)))
    scorer = RiskScorer(RiskConfig().scoring)
    assessment = scorer.assess("E000001", features, 0)
    low_quality = replace(features, quality=replace(features.quality, score=0.01))
    second = RiskScorer(RiskConfig().scoring).assess("E000002", low_quality, 0)

    expected = (
        RiskConfig().scoring.spatial_weight * assessment.spatial_score
        + RiskConfig().scoring.appearance_weight * assessment.appearance_score
    ) / (RiskConfig().scoring.spatial_weight + RiskConfig().scoring.appearance_weight)
    assert assessment.motion_score is None
    assert assessment.degraded is True
    assert assessment.raw_score == pytest.approx(expected)
    assert second.raw_score == pytest.approx(assessment.raw_score)
    assert second.quality.score == pytest.approx(0.01)


def test_threshold_hysteresis_holds_downgrade_for_configured_video_time() -> None:
    scorer = RiskScorer(RiskConfig().scoring)
    high = _synthetic_features(focus=1, area=0.2, bottom_y=0.95, scale_rate=1, inward_rate=1)
    low = _synthetic_features(focus=0, area=0, bottom_y=0.30, scale_rate=-1, inward_rate=-1)

    assert scorer.assess("E000001", high, 0).level == RiskLevel.HIGH
    assert scorer.assess("E000001", low, 100).level == RiskLevel.HIGH
    assert scorer.assess("E000001", low, 499).level == RiskLevel.HIGH
    assert scorer.assess("E000001", low, 500).level == RiskLevel.LOW
    assert scorer.assess("E000001", high, 600).level == RiskLevel.HIGH
