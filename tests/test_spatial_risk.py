from __future__ import annotations

from dataclasses import replace

import pytest

from sightguard.data import Detection, FrameRecord
from sightguard.risk.config import AlertConfig, RiskConfig
from sightguard.risk.engine import AssessmentEngine
from sightguard.risk.schema import EventState, StepResult
from sightguard.spatial.config import SpatialConfig
from sightguard.spatial.navigation_adapter import NavigationAdapter, NavigationCue
from sightguard.spatial.risk_adapter import RiskCueAdapter
from sightguard.spatial.schemas import CueSource, DirectionReference


def make_frame(
    frame_index: int,
    timestamp_ms: float,
    detections: tuple[Detection, ...] = (),
    *,
    status: str = "ok",
) -> FrameRecord:
    return FrameRecord(
        frame_index,
        timestamp_ms,
        640,
        480,
        detections,
        status=status,
        error="synthetic processing failure" if status == "processing_error" else None,
        timestamp_source="video_pts",
    )


def make_detection(
    track_id: int | None,
    bbox: tuple[float, float, float, float],
) -> Detection:
    return Detection(track_id, 0, "person", 0.9, bbox)


def actual_first_alert(
    stream_id: str,
    *,
    track_id: int = 7,
    bbox: tuple[float, float, float, float] = (210, 296, 230, 336),
) -> tuple[FrameRecord, StepResult]:
    engine = AssessmentEngine(stream_id=stream_id)
    engine.step(make_frame(0, 0, (make_detection(track_id, bbox),)))
    record = make_frame(1, 250, (make_detection(track_id, bbox),))
    step = engine.step(record)
    assert len(step.alerts) == 1 and step.alerts[0].alert_type == "first"
    return record, step


def assert_centered_unavailable_direction(
    record: FrameRecord,
    step: StepResult,
    now_ms: float,
) -> None:
    cues = RiskCueAdapter(SpatialConfig()).convert(record, step, now_ms=now_ms)
    assert len(cues) == 1
    assert cues[0].timestamp_ms == step.alerts[0].timestamp_ms
    assert cues[0].expires_at_ms == step.alerts[0].timestamp_ms + SpatialConfig().risk_ttl_ms
    assert cues[0].azimuth_deg == cues[0].elevation_deg == 0
    assert cues[0].direction_method == "unavailable"


def test_actual_first_and_escalation_alerts_convert_with_event_time_and_track_switch() -> None:
    engine = AssessmentEngine(stream_id="camera-a")

    def square(width: int, height: int, *, center_x: int = 220, track_id: int = 7) -> Detection:
        return make_detection(
            track_id,
            (center_x - width // 2, 336 - height, center_x + width // 2, 336),
        )

    frames = [
        make_frame(0, 0, (square(20, 40),)),
        make_frame(1, 250, (square(20, 40),)),
        make_frame(2, 500, (square(30, 60),)),
        make_frame(3, 750, (square(45, 90),)),
        make_frame(4, 1_000, (square(68, 136),)),
        make_frame(5, 1_250, (square(102, 204, center_x=240, track_id=99),)),
    ]
    steps = [engine.step(frame) for frame in frames]
    config = SpatialConfig()
    adapter = RiskCueAdapter(config)

    first = adapter.convert(frames[1], steps[1], now_ms=300)
    escalation = adapter.convert(frames[5], steps[5], now_ms=1_300)

    assert len(first) == len(escalation) == 1
    assert first[0].source == CueSource.RISK
    assert first[0].cue_type == "risk_initial"
    assert first[0].event_id == "E000001"
    assert first[0].timestamp_ms == 250
    assert first[0].expires_at_ms == 250 + config.risk_ttl_ms
    assert first[0].priority == config.risk_priority
    assert first[0].azimuth_deg == pytest.approx(-14.0625)
    assert first[0].direction_reference == DirectionReference.CAMERA
    assert first[0].direction_method == "approximate_screen"
    assert first[0].direction_confidence is None

    switched_observation = steps[5].frame.observations[0]
    assert switched_observation.source_track_id == 99
    assert switched_observation.event_id == "E000001"
    assert steps[5].frame.events[0].last_track_id == 99
    assert escalation[0].event_id == "E000001"
    assert escalation[0].cue_type == "risk_escalation"
    assert escalation[0].timestamp_ms == 1_250
    assert escalation[0].priority == config.high_risk_priority
    assert escalation[0].azimuth_deg == pytest.approx(-11.25)


def test_adapter_deduplicates_by_stream_and_alert_id_and_emits_no_continuous_sound() -> None:
    adapter = RiskCueAdapter(SpatialConfig())
    record_a, step_a = actual_first_alert("camera-a")
    cues = adapter.convert(record_a, step_a, now_ms=250)
    assert len(cues) == 1
    assert adapter.convert(record_a, step_a, now_ms=251) == ()

    engine = AssessmentEngine(stream_id="camera-a")
    detection = make_detection(7, (210, 296, 230, 336))
    engine.step(make_frame(0, 0, (detection,)))
    engine.step(make_frame(1, 250, (detection,)))
    quiet_record = make_frame(2, 500, (detection,))
    quiet_step = engine.step(quiet_record)
    assert quiet_step.alerts == ()
    assert adapter.convert(quiet_record, quiet_step, now_ms=500) == ()

    record_b, step_b = actual_first_alert("camera-b")
    assert step_b.alerts[0].alert_id == step_a.alerts[0].alert_id
    cues_b = adapter.convert(record_b, step_b, now_ms=250)
    assert len(cues_b) == 1
    assert cues_b[0].cue_id != cues[0].cue_id

    unsupported = replace(step_b.alerts[0], alert_type="repeat", alert_id="repeat-1")
    assert adapter.convert(record_b, replace(step_b, alerts=(unsupported,)), now_ms=250) == ()


def test_multiple_current_events_map_to_their_own_camera_directions() -> None:
    config = replace(RiskConfig(), alerts=replace(AlertConfig(), max_alerts_per_frame=2))
    engine = AssessmentEngine(config, stream_id="two-targets")
    left = make_detection(1, (190, 296, 210, 336))
    right = make_detection(2, (430, 296, 450, 336))
    engine.step(make_frame(0, 0, (left, right)))
    record = make_frame(1, 250, (left, right))
    step = engine.step(record)
    assert {alert.event_id for alert in step.alerts} == {"E000001", "E000002"}

    cues = RiskCueAdapter(SpatialConfig()).convert(record, step, now_ms=250)
    by_event = {cue.event_id: cue.azimuth_deg for cue in cues}
    assert by_event == pytest.approx({"E000001": -16.875, "E000002": 16.875})
    assert all(cue.direction_method == "approximate_screen" for cue in cues)


def test_perception_health_alert_is_centered_system_priority() -> None:
    engine = AssessmentEngine(stream_id="camera-a")
    engine.step(make_frame(0, 0, status="processing_error"))
    record = make_frame(1, 500, status="processing_error")
    step = engine.step(record)
    assert [alert.alert_type for alert in step.alerts] == ["perception_limited"]

    config = SpatialConfig()
    cues = RiskCueAdapter(config).convert(record, step, now_ms=500)
    assert len(cues) == 1
    cue = cues[0]
    assert cue.source == CueSource.SYSTEM
    assert cue.cue_type == "perception_limited"
    assert cue.event_id == step.alerts[0].event_id
    assert cue.azimuth_deg == cue.elevation_deg == 0
    assert cue.direction_reference == DirectionReference.LISTENER
    assert cue.direction_method == "unavailable"
    assert cue.priority == config.system_priority == 100
    assert cue.direction_confidence is None


@pytest.mark.parametrize(
    "case",
    [
        "missing_observation",
        "ambiguous_observation",
        "bad_detection_index",
        "source_track_mismatch",
        "class_mismatch",
        "stale_direction",
    ],
)
def test_invalid_current_direction_keeps_live_alert_centered(case: str) -> None:
    record, step = actual_first_alert("camera-a")
    frame = step.frame
    observation = frame.observations[0]
    alert = step.alerts[0]
    now_ms = alert.timestamp_ms + 20

    if case == "missing_observation":
        frame = replace(frame, observations=())
    elif case == "ambiguous_observation":
        frame = replace(frame, observations=(observation, observation))
    elif case == "bad_detection_index":
        frame = replace(frame, observations=(replace(observation, detection_index=99),))
    elif case == "source_track_mismatch":
        frame = replace(frame, observations=(replace(observation, source_track_id=999),))
    elif case == "class_mismatch":
        frame = replace(frame, observations=(replace(observation, class_id=42),))
    elif case == "stale_direction":
        now_ms = alert.timestamp_ms + SpatialConfig().max_direction_age_ms + 1
    else:
        raise AssertionError(f"unhandled case: {case}")

    step = replace(step, frame=frame)
    assert_centered_unavailable_direction(record, step, now_ms)


def test_event_must_be_active_observed_and_seen_at_the_alert_time() -> None:
    for case in ("not_observed_now", "last_seen_mismatch", "resolved_event"):
        record, step = actual_first_alert("camera-a")
        event = step.frame.events[0]
        if case == "not_observed_now":
            event = replace(event, observed_now=False)
        elif case == "last_seen_mismatch":
            event = replace(event, last_seen_ms=0)
        else:
            event = replace(event, state=EventState.RESOLVED)
        frame = replace(step.frame, events=(event,))
        assert_centered_unavailable_direction(record, replace(step, frame=frame), now_ms=270)


def test_record_step_and_alert_timestamps_must_align_for_direction() -> None:
    for case in ("record", "step", "alert"):
        record, step = actual_first_alert("camera-a")
        if case == "record":
            record = replace(record, timestamp_ms=record.timestamp_ms + 1)
        elif case == "step":
            frame = replace(step.frame, timestamp_ms=step.frame.timestamp_ms + 1)
            step = replace(step, frame=frame)
        else:
            alert = replace(step.alerts[0], timestamp_ms=step.alerts[0].timestamp_ms + 1)
            step = replace(step, alerts=(alert,))
        assert_centered_unavailable_direction(record, step, now_ms=270)


def test_navigation_adapter_is_deterministic_listener_relative_and_serializable() -> None:
    config = SpatialConfig()
    adapter = NavigationAdapter(config)
    navigation = NavigationCue("goal-northwest", azimuth_deg=210, elevation_deg=12, lifetime_ms=350)
    round_trip = NavigationCue.from_json(navigation.to_json())
    cue = adapter.convert(round_trip, now_ms=123)

    assert adapter.convert(round_trip, now_ms=123) == cue
    assert cue.cue_id == adapter.convert(round_trip, now_ms=123).cue_id
    assert cue.source == CueSource.NAVIGATION
    assert cue.event_id == "goal-northwest"
    assert cue.cue_type == "navigation"
    assert cue.azimuth_deg == -150
    assert cue.elevation_deg == 12
    assert cue.timestamp_ms == 123
    assert cue.expires_at_ms == 473
    assert cue.priority == config.navigation_priority
    assert cue.direction_reference == DirectionReference.LISTENER
    assert cue.direction_method == "simulated"
    assert cue.direction_confidence is None

    default_lifetime = adapter.convert(NavigationCue("default-ttl", azimuth_deg=0), 1_000)
    assert default_lifetime.expires_at_ms == 1_000 + config.navigation_ttl_ms
    with pytest.raises(ValueError):
        NavigationCue("bad-angle", azimuth_deg=float("nan"))
    with pytest.raises(ValueError):
        NavigationCue("bad-elevation", azimuth_deg=0, elevation_deg=91)
    with pytest.raises(ValueError):
        NavigationCue("bad-ttl", azimuth_deg=0, lifetime_ms=0)
