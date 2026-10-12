from dataclasses import replace

import numpy as np
import pytest

from sightguard.research.wall_metric import (
    MetricDepthField,
    WallMetricConfig,
    WallMetricEstimator,
    depth_errors,
)
from sightguard.scene.spatial_grid.schema import FrameKey


@pytest.fixture
def evidence():
    key = FrameKey("fixture", 1, 100, 80, 60)
    classes = np.zeros((60, 80), np.uint8)
    confidence = np.full((60, 80), 0.9, np.float32)
    depth = MetricDepthField(key, np.full((60, 80), 2, np.float32), "fixture", 20)
    return key, classes, confidence, depth


def estimate(evidence, **kwargs):
    key, classes, confidence, depth = evidence
    return WallMetricEstimator().estimate(
        key, classes, confidence, depth, semantic_key=key, **kwargs
    )


def test_constant_wall_is_not_rejected_for_lacking_depth_variation(evidence):
    result = estimate(evidence)
    assert result.status == "PREDICTED_METRIC_DEPTH"
    assert (result.median_m, result.p10_m, result.p90_m) == (2, 2, 2)
    assert result.wall_present and not result.automatic_alert_enabled
    assert "spatial_not_accuracy" in result.uncertainty_status


@pytest.mark.parametrize("quantity", ["relative_inverse_depth", "euclidean_range_m"])
def test_relative_depth_cannot_be_reinterpreted_as_metric(evidence, quantity):
    *base, depth = evidence
    result = estimate((*base, replace(depth, quantity=quantity)))
    assert result.predicted_metric_depth is None
    assert "depth_is_not_metric_z" in result.rejection_reasons


@pytest.mark.parametrize("age", [-1, 501, float("nan")])
def test_stale_or_future_depth_is_rejected(evidence, age):
    *base, depth = evidence
    assert (
        "depth_source_stale"
        in estimate((*base, replace(depth, source_age_ms=age))).rejection_reasons
    )


def test_wrong_depth_frame_is_rejected(evidence):
    *base, depth = evidence
    result = estimate((*base, replace(depth, key=replace(depth.key, frame_index=2))))
    assert result.wall_present is True
    assert result.status == "ABSTAIN" and "depth_frame_mismatch" in result.rejection_reasons


def test_missing_depth_preserves_wall_presence(evidence):
    *base, _ = evidence
    result = estimate((*base, None))
    assert result.wall_present is True and result.predicted_metric_depth is None


@pytest.mark.parametrize(
    "extra", [{"semantic_from_model": False}, {"scene_current": False}, {"semantic_age_ms": 501}]
)
def test_unverified_or_old_semantics_cannot_create_presence(evidence, extra):
    result = estimate(evidence, **extra)
    assert result.wall_present is None and result.predicted_metric_depth is None


def test_no_wall_and_weak_wall_do_not_create_presence(evidence):
    key, classes, confidence, depth = evidence
    for class_map, conf in [
        (np.full_like(classes, 3), confidence),
        (classes, np.full_like(confidence, 0.2)),
    ]:
        result = estimate((key, class_map, conf, depth))
        assert result.wall_present is False and result.status == "ABSTAIN"


def test_foreground_exclusion_and_component_support(evidence):
    key, classes, confidence, depth = evidence
    foreground = np.ones(classes.shape, bool)
    result = estimate(evidence, foreground_mask=foreground)
    assert result.wall_pixels == 0 and not result.wall_present
    classes[:] = 3
    classes[30:34, 40:44] = 0  # isolated 16 pixels must not pass component gate
    result = estimate((key, classes, confidence, depth))
    assert result.component_count == 0 and result.wall_pixels == 0


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -1, 0, 20])
def test_invalid_depth_values_abstain(evidence, invalid):
    *base, depth = evidence
    result = estimate((*base, replace(depth, values=np.full_like(depth.values, invalid))))
    assert result.valid_pixel_fraction == 0 and result.predicted_metric_depth is None


def test_metric_error_excludes_invalid_truth_and_does_not_fit_scale():
    result = depth_errors(np.array([[2.0, 4.0, 99.0]]), np.array([[1.0, 2.0, 0.0]]))
    assert result["valid_pixels"] == 2
    assert result["mae_m"] == 1.5 and result["abs_rel"] == 1
    assert depth_errors(np.zeros((2, 2)), np.zeros((2, 2)))["mae_m"] is None
    with pytest.raises(ValueError):
        depth_errors(np.ones((2, 2)), np.ones((1, 2)))


def test_invalid_roi_config():
    with pytest.raises(ValueError):
        WallMetricConfig(roi_xyxy=(0.8, 0, 0.2, 1))


@pytest.mark.parametrize(
    "change", [{"model_id": ""}, {"model_latency_ms": float("nan")}, {"model_latency_ms": -1}]
)
def test_untraceable_or_invalid_runtime_depth_abstains(evidence, change):
    *base, depth = evidence
    assert estimate((*base, replace(depth, **change))).predicted_metric_depth is None


def test_nan_expiry_is_not_an_acceptable_configuration():
    with pytest.raises(ValueError):
        WallMetricConfig(maximum_source_age_ms=float("nan"))
