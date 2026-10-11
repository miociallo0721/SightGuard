"""Synthetic image OpenCV checks and separately named fault-injection tests."""

import ast
import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from sightguard.data import Detection, FrameRecord
from sightguard.scene.corridor.artifacts import parse_spatial_frame
from sightguard.scene.motion.estimator import FrameMotionEstimator, background_mask
from sightguard.scene.motion.schema import MotionConfig, MotionFrame
from sightguard.scene.motion.temporal import diagnose_change, evaluate_freshness, summarize_window
from sightguard.scene.spatial_grid.schema import FrameKey


def frame(index=0, timestamp=None, video="synthetic", shift=0, boxes=()):
    image = np.random.default_rng(41).integers(0, 256, (160, 240), dtype=np.uint8)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    if shift:
        image = cv2.warpAffine(image, np.array([[1, 0, shift], [0, 1, 0]], np.float32), (240, 160))
    timestamp = index * 50.0 if timestamp is None else timestamp
    key = FrameKey(video, index, timestamp, 240, 160)
    tracks = FrameRecord(index, timestamp, 240, 160, tuple(boxes), timestamp_source="video_pts")
    return MotionFrame(key, video, image, tracks)


def estimates():
    engine = FrameMotionEstimator()
    return [engine.estimate(frame(i - 1), frame(i, shift=2)) for i in range(1, 11)]


def corridor(status="BEST_SUPPORTED", video="synthetic", index=10, ranking=None):
    names = ["IMAGE_LEFT", "IMAGE_CENTER", "IMAGE_RIGHT"]
    costs = [0.1, 0.2, 0.3]
    if status == "TIED":
        costs = [0.1, 0.11, 0.3]
    if status == "ABSTAIN":
        costs = [None, None, None]
    return {
        "key": {
            "video_id": video,
            "frame_index": index,
            "timestamp_ms": index * 50.0,
            "width": 240,
            "height": 160,
        },
        "ablation": "C",
        "status": status,
        "ranking": ranking or (names if status != "ABSTAIN" else []),
        "best_supported_candidate": "IMAGE_LEFT" if status == "BEST_SUPPORTED" else None,
        "candidates": [
            {
                "candidate": {"candidate_id": name},
                "total_cost": cost,
                "weighted_contributions": {"ground": cost, "depth": 0.01},
                "eligibility_reasons": ["missing"] if cost is None else [],
                "evidence": {
                    term: 0.1
                    for term in (
                        "ground_support",
                        "ground_continuity",
                        "bbox_proxy_coverage",
                        "near_bbox_proxy_coverage",
                        "unknown_exposure",
                        "conflict_coverage",
                    )
                },
                "object_relations": [],
            }
            for name, cost in zip(names, costs, strict=True)
        ],
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"resize_width": 20},
        {"lk_window": 20},
        {"feature_quality": 1.1},
        {"min_matches": 2},
        {"max_dt_ms": 1},
        {"min_scale": 1.1},
        {"max_scale": 0.9},
        {"min_inlier_ratio": 0.8},
        {"stable_diagonals_per_second": 0.6},
        {"mask_margin_pixels": float("nan")},
        {"min_features": True},
        {"temporal_shadow_enabled": True},
        {"simulated_consumer_delays_ms": (-1,)},
    ],
)
def test_invalid_motion_config(kwargs):
    with pytest.raises(ValueError):
        MotionConfig(**kwargs)


@pytest.mark.parametrize(
    "shift,label",
    [(0, "STABLE_IMAGE_MOTION"), (3, "MODERATE_IMAGE_MOTION"), (12, "RAPID_IMAGE_MOTION")],
)
def test_actual_opencv_synthetic_small_and_large_translation(shift, label):
    result = FrameMotionEstimator().estimate(frame(), frame(1, shift=shift))
    assert result.estimation_status == "VALID" and result.quality.label == label
    assert abs(result.translation_source_pixels[0] - shift) < 0.3
    assert (
        result.absolute_camera_pose_status
        == result.quality.absolute_camera_pose_status
        == "UNKNOWN"
    )
    assert result.inlier_ratio > 0.9


@pytest.mark.parametrize(
    "reason,change",
    [
        ("frame_decode_failed", {"image": None}),
        ("unverified_timestamp_source", {"timestamp_source": "fps_fallback"}),
        ("sequence_identity_mismatch", {"stream_id": "another"}),
        ("invalid_image_contract", {"image": np.zeros((160, 240), np.float32)}),
    ],
)
def test_unobservable_inputs_never_stable(reason, change):
    result = FrameMotionEstimator().estimate(frame(), replace(frame(1), **change))
    assert result.quality.label == "UNKNOWN" and reason in result.quality.reasons


@pytest.mark.parametrize("index,timestamp", [(2, 100), (0, 0), (1, 0), (1, -0.0), (1, 500)])
def test_missing_duplicate_or_abnormal_time(index, timestamp):
    result = FrameMotionEstimator().estimate(frame(), frame(index, timestamp))
    assert result.quality.label == "UNKNOWN"


def test_backwards_time_and_dimensions():
    assert (
        FrameMotionEstimator().estimate(frame(1), frame(2, timestamp=25)).quality.label == "UNKNOWN"
    )
    current = replace(frame(1), key=FrameKey("synthetic", 1, 50, 241, 160))
    assert (
        "image_dimensions_changed"
        in FrameMotionEstimator().estimate(frame(), current).quality.reasons
    )


def test_no_previous_and_missing_previous():
    engine = FrameMotionEstimator()
    assert engine.estimate(None, frame()).quality.label == "UNKNOWN"
    assert (
        "previous_frame_unavailable"
        in engine.estimate(replace(frame(), image=None), frame(1)).quality.reasons
    )


def test_textureless_image_is_unknown():
    a, b = frame(), frame(1)
    result = FrameMotionEstimator().estimate(
        replace(a, image=np.full_like(a.image, 120)), replace(b, image=np.full_like(b.image, 120))
    )
    assert result.quality.label == "UNKNOWN" and result.motion_diagonals_per_second is None


def test_static_background_fully_masked_and_low_confidence_box_not_removed():
    reliable = Detection(1, 0, "person", 0.9, (0, 0, 240, 160))
    result = FrameMotionEstimator().estimate(frame(boxes=(reliable,)), frame(1, boxes=(reliable,)))
    assert result.background_available_fraction == 0 and result.quality.label == "UNKNOWN"
    weak = replace(reliable, confidence=0.1)
    assert np.all(background_mask(frame(boxes=(weak,)), MotionConfig()) == 255)


def test_actual_opencv_mask_excludes_dynamic_patch_matches():
    a, b = frame(), frame(1, shift=3)
    a.image[40:120, 70:170] = np.random.default_rng(9).integers(0, 256, (80, 100), dtype=np.uint8)
    b.image[40:120, 90:190] = a.image[40:120, 70:170]
    a = replace(
        a,
        tracks=replace(a.tracks, detections=(Detection(1, 0, "person", 0.9, (70, 40, 170, 120)),)),
    )
    b = replace(
        b,
        tracks=replace(b.tracks, detections=(Detection(1, 0, "person", 0.9, (90, 40, 190, 120)),)),
    )
    result = FrameMotionEstimator().estimate(a, b)
    assert (
        result.estimation_status == "VALID" and abs(result.translation_source_pixels[0] - 3) < 0.5
    )
    for sample in result.flow_samples:
        x, y = sample["previous_xy"]
        assert not (58 <= x < 182 and 28 <= y < 132)


@pytest.mark.parametrize(
    "matrix", [None, np.full((2, 3), float("nan")), np.array([[2, 0, 0], [0, 2, 0]], float)]
)
def test_mock_ransac_failure_or_invalid_parameters(monkeypatch, matrix):
    monkeypatch.setattr(
        cv2, "estimateAffinePartial2D", lambda a, b, **kw: (matrix, np.ones((len(a), 1), np.uint8))
    )
    result = FrameMotionEstimator().estimate(frame(), frame(1))
    assert result.quality.label == "UNKNOWN"


def test_mock_feature_failure(monkeypatch):
    monkeypatch.setattr(cv2, "goodFeaturesToTrack", lambda *a, **kw: None)
    result = FrameMotionEstimator().estimate(frame(), frame(1))
    assert "insufficient_features" in result.quality.reasons


def test_same_input_deterministic_serializable_and_video_isolation():
    engine = FrameMotionEstimator()
    a, b = frame(), frame(1, shift=3)
    first = engine.estimate(a, b).to_dict()
    engine.estimate(frame(video="x"), frame(1, video="x"))
    assert first == engine.estimate(a, b).to_dict()
    assert json.loads(json.dumps(first, allow_nan=False))["key"]["video_id"] == "synthetic"
    assert engine.estimate(a, frame(1, video="x")).quality.label == "UNKNOWN"


def test_past_window_no_future_or_missing_filled_zero():
    rows = estimates()
    key = frame(10).key
    window = summarize_window(key, "synthetic", rows)
    assert window.valid_count == 10 and window.time_coverage_fraction == 1
    future = FrameMotionEstimator().estimate(frame(10), frame(11, shift=12))
    assert window == summarize_window(key, "synthetic", rows + [future])
    empty = summarize_window(key, "synthetic", [])
    assert empty.quality.label == "UNKNOWN" and empty.motion_rate_p50 is None
    missing = summarize_window(key, "synthetic", rows[3:])
    assert missing.time_coverage_fraction < 0.8 and missing.quality.label == "UNKNOWN"


def test_window_unknown_partial_duplicate_and_other_stream():
    rows = estimates()
    key = frame(10).key
    assert summarize_window(frame().key, "synthetic", rows).quality.label == "UNKNOWN"
    assert summarize_window(key, "other", rows).quality.label == "UNKNOWN"
    with pytest.raises(ValueError):
        summarize_window(key, "synthetic", rows + rows[:1])
    row = replace(rows[0], estimation_status="UNKNOWN")
    window = summarize_window(key, "synthetic", [row, *rows[1:]])
    assert window.quality.label == "LIMITED"


def test_freshness_uses_age_and_quality_and_preserves_unknown_pose():
    source = corridor()
    window = summarize_window(frame(10).key, "synthetic", estimates())
    assert evaluate_freshness(source, window, 500, "synthetic").status == "FRESH"
    assert evaluate_freshness(source, window, 2000, "synthetic").status == "STALE"
    assert evaluate_freshness(source, window, 400, "synthetic").status == "INVALID"
    assert evaluate_freshness(corridor("ABSTAIN"), window, 500, "synthetic").status == "INVALID"
    unknown = summarize_window(frame(10).key, "synthetic", [])
    result = evaluate_freshness(source, unknown, 99999, "synthetic")
    assert result.status == "UNKNOWN" and not result.reuse_allowed_for_research
    assert result.absolute_camera_pose_status == "UNKNOWN"


@pytest.mark.parametrize(
    "before,after,kind",
    [
        ("BEST_SUPPORTED", "TIED", "BEST_SUPPORTED_TO_TIED"),
        ("TIED", "BEST_SUPPORTED", "TIED_TO_BEST_SUPPORTED"),
        ("BEST_SUPPORTED", "ABSTAIN", "RESULT_BECAME_ABSTAIN"),
        ("ABSTAIN", "TIED", "RESULT_RECOVERED_FROM_ABSTAIN"),
    ],
)
def test_support_transitions_not_all_direction_jitter(before, after, kind):
    result = diagnose_change(corridor(after, index=20), corridor(before))
    assert kind in result["types"]


def test_close_rank_swap_and_obstacle_evidence_change():
    previous = corridor("TIED")
    current = corridor("TIED", index=20, ranking=["IMAGE_CENTER", "IMAGE_LEFT", "IMAGE_RIGHT"])
    current["candidates"][0]["evidence"]["bbox_proxy_coverage"] = 0.6
    result = diagnose_change(current, previous)
    assert (
        "CLOSE_COST_ORDER_SWAP" in result["types"]
        and "IMAGE_REGION_EVIDENCE_CHANGED" in result["types"]
    )
    assert not result["best_supported_identity_changed"]
    with pytest.raises(ValueError):
        diagnose_change(corridor(video="other"), previous)


def test_motion_core_import_isolation_and_saved_grid_adapter_unchanged():
    forbidden = (
        "torch",
        "ultralytics",
        "gradio",
        "sightguard.risk",
        "sightguard.decision",
        "sightguard.spatial",
        "sightguard.ui",
    )
    for path in Path("src/sightguard/scene/motion").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else []
            )
            assert not any(name.startswith(forbidden) for name in names)
    assert callable(parse_spatial_frame)
