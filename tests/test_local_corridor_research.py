import numpy as np
import pytest

from sightguard.research.local_corridor import LocalCorridorCandidateEvaluator
from sightguard.scene.spatial_grid import (
    EvidenceSource,
    FrameKey,
    SpatialGridFusion,
    SpatialGridInput,
)


def evidence(index=0, *, boxes=(), unknown=False, sequence="fixture", depth_available=True):
    key = FrameKey(sequence, index, index * 100, 90, 90)
    source = EvidenceSource(key, "valid", "high", "synthetic", "fixture", "fixture.v1")
    classes = np.full((90, 90), 11, np.uint8)
    confidence = np.full((90, 90), 0.9, np.float32)
    grid = SpatialGridFusion().fuse(
        SpatialGridInput(
            key,
            None if unknown else source,
            source if depth_available else None,
            source,
            classes,
            confidence,
            np.ones((90, 90), np.int32),
            np.linspace(0.1, 2, 8100, dtype=np.float32).reshape(90, 90)
            if depth_available
            else None,
            boxes,
        )
    )
    return grid, classes, confidence


def evaluate(engine, data, **kwargs):
    grid, classes, confidence = data
    defaults = dict(semantic_key=grid.key, source_age_ms=0, motion_quality="UNKNOWN")
    defaults.update(kwargs)
    return engine.evaluate(grid, classes, confidence, **defaults)


def test_uniform_image_support_remains_tied_with_strict_d_abstain():
    engine = LocalCorridorCandidateEvaluator()
    for index in range(3):
        result = evaluate(engine, evidence(index))
        assert result["status"] == "TIED"
        assert result["strict_d_status"] == "ABSTAIN"
        assert result["scope"] == "CANDIDATE_ONLY"
        assert result["physical_corridor_width_m"] is None
        assert not result["automatic_alert_enabled"]


def test_unknown_semantics_are_not_free_space():
    result = evaluate(LocalCorridorCandidateEvaluator(), evidence(unknown=True))
    assert result["status"] == "ABSTAIN" and result["best_candidate"] is None


@pytest.mark.parametrize(
    "change",
    [
        {"source_age_ms": 501},
        {"source_age_ms": -1},
        {"source_age_ms": float("nan")},
        {"motion_quality": "RAPID_IMAGE_MOTION"},
        {"semantic_key": FrameKey("wrong", 0, 0, 90, 90)},
    ],
)
def test_time_motion_and_identity_gates(change):
    assert evaluate(LocalCorridorCandidateEvaluator(), evidence(), **change)["status"] == "ABSTAIN"


def test_dynamic_obstacle_cancels_previous_image_support():
    engine = LocalCorridorCandidateEvaluator()
    evaluate(engine, evidence())
    box = {"bbox_xyxy": [0, 30, 90, 90], "confidence": 0.99, "class_id": 0}
    result = evaluate(engine, evidence(1, boxes=(box,)))
    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert result["temporal_support"] == 0
    assert all(not item["eligible"] for item in result["candidates"])


def test_metric_mismatch_cannot_be_silently_fused():
    result = evaluate(
        LocalCorridorCandidateEvaluator(),
        evidence(),
        metric_z_m=np.ones((90, 90)),
        metric_key=FrameKey("wrong", 0, 0, 90, 90),
    )
    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert all(
        "metric_depth_frame_or_geometry_mismatch" in item["rejection_reasons"]
        for item in result["candidates"]
    )


def test_depth_optional_does_not_invent_metric_clearance():
    result = evaluate(LocalCorridorCandidateEvaluator(), evidence(depth_available=False))
    assert result["comparison_ablation"] == "B"
    assert result["body_clearance_width_m"] is None
    assert result["strict_d_status"] == "ABSTAIN"
