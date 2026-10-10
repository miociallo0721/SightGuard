"""Synthetic image boxes test demo rules; these do not validate road navigation."""

import pytest

from sightguard.data import Detection, FrameRecord
from sightguard.decision.demo_guidance import DemoGuidance, DemoGuidanceConfig


def frame(time: float, boxes: tuple = (), status: str = "ok") -> FrameRecord:
    return FrameRecord(
        int(time),
        time,
        1000,
        1000,
        tuple(Detection(i, 0, "person", 0.9, box) for i, box in enumerate(boxes)),
        status=status,
        error="test failure" if status != "ok" else None,
    )


def test_empty_and_failed_frames_do_not_imply_forward() -> None:
    engine = DemoGuidance(DemoGuidanceConfig())
    assert engine.step(frame(0)).action == "unknown"
    assert engine.step(frame(400, status="processing_error")).action == "unknown"


@pytest.mark.parametrize(
    ("boxes", "action"),
    [
        (((100, 300, 550, 800),), "right"),
        (((450, 300, 900, 800),), "left"),
        (((0, 300, 180, 800),), "forward"),
    ],
)
def test_direction_requires_repeated_video_time_evidence(boxes: tuple, action: str) -> None:
    engine = DemoGuidance(DemoGuidanceConfig())
    assert engine.step(frame(0, boxes)).action == "unknown"
    assert engine.step(frame(100, boxes)).action == "unknown"
    output = engine.step(frame(400, boxes))
    assert output.action == action
    assert output.mode == "image_region_demo"
    assert output.hardware_connected is False


def test_central_large_object_and_ambiguous_sides_stop() -> None:
    engine = DemoGuidance(DemoGuidanceConfig())
    assert engine.step(frame(0, ((100, 100, 900, 950),))).action == "stop"
    assert engine.step(frame(400, ((400, 400, 600, 800),))).action == "stop"


def test_failure_resets_confirmation_and_sessions_are_independent() -> None:
    boxes = ((0, 300, 180, 800),)
    engine = DemoGuidance(DemoGuidanceConfig())
    engine.step(frame(0, boxes))
    assert engine.step(frame(400, boxes)).action == "forward"
    engine.step(frame(500, status="processing_error"))
    assert engine.step(frame(600, boxes)).action == "unknown"
    fresh = DemoGuidance(DemoGuidanceConfig())
    assert fresh.step(frame(800, boxes)).action == "unknown"
    with pytest.raises(ValueError, match="backwards"):
        fresh.step(frame(700, boxes))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True])
def test_invalid_thresholds_rejected(value: float) -> None:
    with pytest.raises(ValueError):
        DemoGuidanceConfig(confirmation_ms=value)


def test_zero_area_boxes_and_equal_timestamps_do_not_confirm_direction() -> None:
    engine = DemoGuidance(DemoGuidanceConfig())
    assert engine.step(frame(0, ((0, 0, 0, 900),))).action == "unknown"
    boxes = ((0, 300, 180, 800),)
    assert engine.step(frame(400, boxes)).action == "unknown"
    assert engine.step(frame(400, boxes)).action == "unknown"
