"""Deterministic image-space tests; these do not require model weights."""

import numpy as np
import pytest

from sightguard.path_relevance.engine import (
    CorridorEstimate,
    ObjectPathAssessment,
    associate_detection,
    estimate_corridors,
    scene_is_fresh,
)
from sightguard.path_relevance.model import (
    ModelUnavailableError,
    SegformerB0Perceptor,
    SegmentationOutput,
)
from sightguard.path_relevance.pipeline import _shadow_record, run_path_relevance_batch


def _segmentation(class_map: np.ndarray, confidence: float = 0.9) -> SegmentationOutput:
    return SegmentationOutput(
        class_map=class_map,
        confidence_map=np.full(class_map.shape, confidence, dtype=np.float32),
        inference_latency_ms=10.0,
        input_width=class_map.shape[1],
        input_height=class_map.shape[0],
    )


def _estimate() -> CorridorEstimate:
    classes = np.zeros((100, 120), dtype=np.uint8)
    classes[35:, 20:55] = 11
    return estimate_corridors(_segmentation(classes))


def test_corridors_keep_multiple_semantic_components_connected_to_lower_frame() -> None:
    classes = np.zeros((100, 120), dtype=np.uint8)
    classes[35:, 5:40] = 6
    classes[45:, 75:110] = 11
    classes[0:20, 48:70] = 3

    result = estimate_corridors(_segmentation(classes))

    assert result.status == "valid"
    assert result.quality == "high"
    assert len(result.corridors) == 2
    assert result.mask[80, 15] == 1
    assert result.mask[80, 90] == 1
    assert result.mask[10, 55] == 0
    assert "camera_pose_unobserved" in result.reasons


def test_no_supported_forward_ground_returns_unknown() -> None:
    classes = np.zeros((100, 120), dtype=np.uint8)
    classes[10:25, 10:100] = 11

    result = estimate_corridors(_segmentation(classes))

    assert result.status == "unknown"
    assert not result.corridors
    assert result.confidence == 0
    assert not result.mask.any()


def test_low_confidence_walkable_pixels_return_unknown() -> None:
    classes = np.zeros((100, 100), dtype=np.uint8)
    classes[30:, 30:70] = 11

    result = estimate_corridors(_segmentation(classes, confidence=0.2))

    assert result.status == "unknown"


def test_limited_scene_abstains_instead_of_publishing_low_score() -> None:
    classes = np.zeros((100, 120), dtype=np.uint8)
    classes[35:, 20:55] = 11
    estimate = estimate_corridors(_segmentation(classes, confidence=0.5))
    assessment = associate_detection(
        {
            "track_id": 7,
            "class_id": 2,
            "class_name": "car",
            "confidence": 0.8,
            "bbox_xyxy": [80, 45, 110, 90],
        },
        0,
        estimate,
        frame_width=120,
        frame_height=100,
    )

    assert estimate.quality == "limited"
    assert assessment.relevance_level == "UNKNOWN"
    assert assessment.path_relevance_score is None
    assert assessment.corridor_intersection == 0


def test_object_inside_corridor_has_overlap_and_class_specific_anchor_uncertainty() -> None:
    estimate = _estimate()
    person = associate_detection(
        {"track_id": 1, "class_id": 0, "class_name": "person", "bbox_xyxy": [25, 45, 48, 95]},
        0,
        estimate,
        event_id="event-1",
        frame_width=120,
        frame_height=100,
    )
    chair = associate_detection(
        {"track_id": 2, "class_id": 56, "class_name": "chair", "bbox_xyxy": [25, 45, 48, 95]},
        1,
        estimate,
        frame_width=120,
        frame_height=100,
    )

    assert person.path_relevance_score is not None and person.path_relevance_score > 0.5
    assert person.estimated_spatial_relation == "corridor_overlap"
    assert person.contact_point_confidence > chair.contact_point_confidence
    assert person.contact_point_status != chair.contact_point_status
    assert person.event_id == "event-1"


def test_object_outside_corridor_is_low_and_other_side_stays_outside() -> None:
    result = associate_detection(
        {"track_id": 5, "class_id": 2, "class_name": "car", "bbox_xyxy": [80, 50, 110, 90]},
        0,
        _estimate(),
        frame_width=120,
        frame_height=100,
    )

    assert result.path_relevance_score == 0
    assert result.relevance_level == "LOW"
    assert result.estimated_spatial_relation == "outside_corridor"


@pytest.mark.parametrize(
    "bbox",
    ([20, 50, 20, 60], [200, 50, 220, 80], [float("nan"), 1, 3, 4]),
)
def test_invalid_and_out_of_frame_bbox_abstains(bbox: list[float]) -> None:
    result = associate_detection(
        {"track_id": 5, "class_id": 2, "class_name": "car", "bbox_xyxy": bbox},
        0,
        _estimate(),
        frame_width=120,
        frame_height=100,
    )

    assert result.relevance_level == "UNKNOWN"
    assert result.path_relevance_score is None
    assert result.uncertainty == 1


def test_unknown_scene_and_misaligned_semantic_output_are_not_no_risk() -> None:
    unknown = estimate_corridors(_segmentation(np.zeros((100, 100), dtype=np.uint8)))
    result = associate_detection(
        {"track_id": 3, "class_id": 0, "class_name": "person", "bbox_xyxy": [10, 10, 30, 40]},
        0,
        unknown,
        frame_width=100,
        frame_height=100,
    )
    assert result.relevance_level == "UNKNOWN"
    assert result.path_relevance_score is None
    with pytest.raises(ValueError, match="aligned"):
        estimate_corridors(
            SegmentationOutput(
                class_map=np.zeros((4, 5), dtype=np.uint8),
                confidence_map=np.zeros((3, 5), dtype=np.float32),
                inference_latency_ms=1,
                input_width=5,
                input_height=4,
            )
        )


@pytest.mark.parametrize(
    ("scene_time", "object_time", "max_age", "expected"),
    [(100, 100, 0, True), (100, 200, 100, True), (100, 201, 100, False), (200, 100, 500, False)],
)
def test_scene_result_freshness(
    scene_time: float, object_time: float, max_age: float, expected: bool
) -> None:
    assert scene_is_fresh(scene_time, object_time, max_age_ms=max_age) is expected


def test_high_baseline_alert_is_always_preserved_in_shadow_only() -> None:
    original_alert = {"event_id": "event-1", "alert_type": "first"}
    path = ObjectPathAssessment(
        track_id=1,
        detection_index=0,
        class_id=0,
        class_name="person",
        detection_confidence=0.91,
        event_id="event-1",
        path_relevance_score=0.0,
        relevance_level="LOW",
        corridor_intersection=0.0,
        estimated_spatial_relation="outside_corridor",
        contact_point_status="bbox_bottom_center_heuristic_person",
        contact_point_confidence=0.6,
        scene_quality="high",
        confidence=0.8,
        uncertainty=0.2,
        reasons=(),
        evidence_source=("test",),
        bbox_xyxy=(0, 0, 10, 10),
    )

    shadow = _shadow_record(
        video_id="clip",
        frame_index=0,
        timestamp_ms=0.0,
        path=path,
        risk_observation={"event_id": "event-1", "assessment": {"score": 0.9, "level": "HIGH"}},
        alert_event_ids={"event-1"},
        geometry_score=0.3,
    )

    assert original_alert == {"event_id": "event-1", "alert_type": "first"}
    assert shadow["baseline_alert_event_present"] is True
    assert shadow["shadow_recommendation"] == "preserve_baseline_high"
    assert shadow["production_decision_changed"] is False


def test_unavailable_model_writes_unknown_for_real_video_inputs(tmp_path) -> None:
    import json

    import cv2

    video_path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (64, 64))
    assert writer.isOpened()
    writer.write(np.full((64, 64, 3), 96, dtype=np.uint8))
    writer.write(np.full((64, 64, 3), 128, dtype=np.uint8))
    writer.release()
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps({"videos": [{"video_id": "clip", "filepath": str(video_path)}]}),
        encoding="utf-8",
    )
    baseline = tmp_path / "baseline" / "clip"
    (baseline / "perception").mkdir(parents=True)
    (baseline / "assessment").mkdir()
    detections = [
        {
            "track_id": 1,
            "class_id": 0,
            "class_name": "person",
            "confidence": 0.8,
            "bbox_xyxy": [20, 10, 40, 50],
        }
    ]
    tracks = [
        {
            "frame_index": index,
            "timestamp_ms": index * 500.0,
            "frame_width": 64,
            "frame_height": 64,
            "detections": detections,
        }
        for index in range(2)
    ]
    risks = [
        {"frame_index": index, "timestamp_ms": index * 500.0, "observations": []}
        for index in range(2)
    ]
    (baseline / "perception/tracks.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in tracks), encoding="utf-8"
    )
    for name, rows in (("risk_frames.jsonl", risks), ("events.jsonl", []), ("alerts.jsonl", [])):
        (baseline / f"assessment/{name}").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    result = run_path_relevance_batch(
        dataset,
        tmp_path / "baseline",
        tmp_path / "output",
        checkpoint_path=tmp_path / "missing.pth",
        device="cpu",
    )

    scene = json.loads(
        (tmp_path / "output/clip/scene_frames.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    objects = [
        json.loads(line)
        for line in (tmp_path / "output/clip/objects.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert result["status"] == "completed"
    assert result["model"]["status"] == "unavailable"
    assert scene["status"] == "unknown"
    assert scene["inference_status"] == "unavailable"
    assert all(row["relevance_level"] == "UNKNOWN" for row in objects)


def test_missing_checkpoint_is_reported_as_unavailable_model(tmp_path) -> None:
    with pytest.raises(ModelUnavailableError, match="checkpoint not found"):
        SegformerB0Perceptor(tmp_path / "not-present.pth", device="cpu")
