import cv2
import numpy as np
import pytest

from sightguard.research.wall_approach import (
    ApproachConfig,
    WallApproachEstimator,
    rotation_counter_model,
)
from sightguard.scene.spatial_grid.schema import FrameKey

K = np.array([[520.0, 0, 320], [0, 520.0, 240], [0, 0, 1]])
POINTS = np.array([(x, y) for y in range(80, 421, 40) for x in range(100, 561, 40)], np.float32)


def tracks(estimator, after, index=1):
    previous = FrameKey("synthetic", index - 1, (index - 1) * 100, 640, 480)
    current = FrameKey("synthetic", index, index * 100, 640, 480)
    return estimator.evaluate_tracks(
        {"automatic_alert_enabled": False},
        previous,
        current,
        POINTS,
        after,
        K,
        np.zeros(5),
        feature_count=len(POINTS),
        mask_iou=1,
    )


def test_three_windows_of_expansion_are_synthetic_camera_candidate_only():
    estimator = WallApproachEstimator()
    after = (POINTS - [320, 240]) * 1.02 + [320, 240]
    for i in (1, 2):
        result = tracks(estimator, after, i)
        assert result["state"] == "WALL_PRESENT"
        assert not result["approaching_candidate"]
    result = tracks(estimator, after, 3)
    assert result["state"] == "WALL_APPROACHING_CANDIDATE"
    assert not result["automatic_alert_enabled"]


@pytest.mark.parametrize("axis", [(0.0, 0.06, 0.0), (0.06, 0.0, 0.0), (0.0, 0.0, 0.06)])
def test_real_pinhole_rotation_does_not_become_approach(axis):
    rotation, _ = cv2.Rodrigues(np.array(axis))
    rays = np.column_stack([(POINTS - [320, 240]) / 520, np.ones(len(POINTS))])
    transformed = rays @ rotation.T
    after = transformed[:, :2] / transformed[:, 2:] * 520 + [320, 240]
    residual, _ = rotation_counter_model(POINTS, after, K, np.zeros(5))
    assert residual < 0.001
    result = tracks(WallApproachEstimator(), after)
    assert result["state"] == "ABSTAIN"
    assert not result.get("approaching_candidate", False)


@pytest.mark.parametrize(
    "after",
    [
        POINTS,
        POINTS + [10, 0],
        (POINTS - [320, 240]) * 0.98 + [320, 240],
        (POINTS - [-2000, 240]) * 1.02 + [-2000, 240],
    ],
)
def test_stationary_lateral_receding_off_axis_rejected(after):
    assert tracks(WallApproachEstimator(), after)["state"] == "ABSTAIN"


def test_rejection_resets_temporal_confirmation():
    estimator = WallApproachEstimator()
    expansion = (POINTS - [320, 240]) * 1.02 + [320, 240]
    tracks(estimator, expansion)
    tracks(estimator, expansion, 2)
    assert tracks(estimator, POINTS, 3)["temporal_support"] == 0
    assert tracks(estimator, expansion, 4)["temporal_support"] == 1


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"intrinsic": None}, "intrinsics_or_fixed_lens_unverified"),
        ({"fixed_lens_verified": False}, "intrinsics_or_fixed_lens_unverified"),
        ({"source_age_ms": 501}, "source_stale"),
        ({"motion_quality": "RAPID_IMAGE_MOTION"}, "rapid_motion_or_discontinuity"),
    ],
)
def test_gates_precede_flow(change, reason):
    image = np.zeros((480, 640), np.uint8)
    mask = np.ones_like(image, bool)
    kwargs = dict(source_age_ms=0, wall_present=True, intrinsic=K, fixed_lens_verified=True)
    kwargs.update(change)
    result = WallApproachEstimator().evaluate(
        FrameKey("a", 0, 0, 640, 480),
        FrameKey("a", 1, 100, 640, 480),
        image,
        image,
        mask,
        mask,
        **kwargs,
    )
    assert result["abstain_reason"] == reason


def test_low_texture_wall_correctly_abstains():
    image = np.full((480, 640), 240, np.uint8)
    mask = np.ones_like(image, bool)
    result = WallApproachEstimator().evaluate(
        FrameKey("a", 0, 0, 640, 480),
        FrameKey("a", 1, 100, 640, 480),
        image,
        image,
        mask,
        mask,
        source_age_ms=0,
        wall_present=True,
        intrinsic=K,
        fixed_lens_verified=True,
    )
    assert result["abstain_reason"] == "low_texture_or_insufficient_wall_features"


@pytest.mark.parametrize(
    "change",
    [
        {"maximum_rotation_rad": float("nan")},
        {"minimum_features": 0},
        {"minimum_inlier_fraction": 1.1},
    ],
)
def test_invalid_approach_thresholds_are_rejected(change):
    with pytest.raises(ValueError):
        ApproachConfig(**change)


def test_invalid_feature_tracks_do_not_produce_candidate():
    result = tracks(WallApproachEstimator(), np.full_like(POINTS, np.nan))
    assert result["abstain_reason"] == "invalid_feature_tracks"
