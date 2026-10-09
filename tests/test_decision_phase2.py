import pytest

from sightguard.data import Detection, FrameRecord
from sightguard.decision.alerts import AlertScheduler
from sightguard.decision.events import EventManager
from sightguard.decision.health import PerceptionMonitor
from sightguard.risk.config import AlertConfig, AssociationConfig, FeatureConfig, RiskConfig
from sightguard.risk.engine import AssessmentEngine
from sightguard.risk.features import spatial_features
from sightguard.risk.schema import (
    EventSnapshot,
    EventState,
    PerceptionQuality,
    RiskAssessment,
    RiskLevel,
)


def _detection(track_id: int | None, bbox: tuple[float, float, float, float]) -> Detection:
    return Detection(track_id, 0, "person", 0.9, bbox)


def _frame(
    index: int,
    timestamp_ms: float,
    detections: tuple[Detection, ...] = (),
    *,
    status: str = "ok",
) -> FrameRecord:
    return FrameRecord(
        index,
        timestamp_ms,
        640,
        480,
        detections,
        status=status,
        error="synthetic decode failure" if status == "processing_error" else None,
        timestamp_source="video_pts",
    )


def _assessment(
    level: RiskLevel = RiskLevel.MEDIUM,
    score: float = 0.5,
    spatial_score: float = 0.6,
    quality_score: float = 0.9,
) -> RiskAssessment:
    quality = PerceptionQuality(0.9, 1.0, 3, 3, False, False, True, "video_pts", quality_score, ())
    return RiskAssessment(
        spatial_score,
        0.5,
        0.2,
        score,
        score,
        level,
        quality,
        False,
        ("synthetic test risk",),
    )


def _snapshot(
    event_id: str,
    level: RiskLevel = RiskLevel.MEDIUM,
    score: float = 0.5,
    spatial_score: float = 0.6,
    source_track_ids: tuple[int, ...] = (7,),
) -> EventSnapshot:
    return EventSnapshot(
        event_id,
        EventState.ACTIVE,
        0,
        "person",
        0,
        1000,
        3,
        True,
        source_track_ids,
        source_track_ids[-1] if source_track_ids else None,
        True,
        _assessment(level, score, spatial_score),
    )


def _observe_frame(manager: EventManager, frame: FrameRecord):
    complete = frame.status == "ok"
    changes = list(manager.prepare(frame.timestamp_ms, complete))
    spatial = {
        index: spatial_features(detection, frame.frame_width, frame.frame_height, FeatureConfig())
        for index, detection in enumerate(frame.detections)
    }
    associations = manager.associate(frame, spatial)
    for association in associations:
        detection = frame.detections[association.detection_index]
        changes.extend(
            manager.observe(
                association,
                detection,
                spatial[association.detection_index],
                _assessment(),
                frame.timestamp_ms,
            )
        )
    changes.extend(manager.end_frame(frame.timestamp_ms, complete))
    return associations, tuple(changes)


def test_track_switch_recovers_same_event_and_keeps_source_id_history() -> None:
    manager = EventManager(AssociationConfig(), "camera-a")
    box = (280, 280, 360, 440)
    first, _ = _observe_frame(manager, _frame(0, 0, (_detection(4, box),)))
    second, confirmed_changes = _observe_frame(manager, _frame(1, 250, (_detection(4, box),)))

    assert first[0].event.event_id == second[0].event.event_id == "E000001"
    assert any(
        change.action == "activated" and change.timestamp_ms == 250 for change in confirmed_changes
    )
    assert manager.events["E000001"].state == EventState.ACTIVE

    _, lost_changes = _observe_frame(manager, _frame(2, 500))
    recovered, recovery_changes = _observe_frame(manager, _frame(3, 700, (_detection(99, box),)))
    event = manager.events["E000001"]
    assert any(change.action == "lost" and change.event_id == "E000001" for change in lost_changes)
    assert recovered[0].method == "geometry"
    assert recovered[0].event.event_id == "E000001"
    assert event.state == EventState.ACTIVE
    assert tuple(event.source_track_ids) == (4, 99)
    assert any(
        change.action == "recovered" and change.timestamp_ms == 700 for change in recovery_changes
    )


def test_two_distinct_targets_use_one_to_one_geometric_association() -> None:
    manager = EventManager(AssociationConfig(), "camera-a")
    first_boxes = ((170, 280, 230, 420), (410, 280, 470, 420))
    first, _ = _observe_frame(
        manager,
        _frame(0, 0, (_detection(1, first_boxes[0]), _detection(2, first_boxes[1]))),
    )
    current, _ = _observe_frame(
        manager,
        _frame(
            1,
            100,
            (
                _detection(11, (173, 280, 233, 420)),
                _detection(12, (407, 280, 467, 420)),
            ),
        ),
    )

    first_by_index = {item.detection_index: item.event.event_id for item in first}
    current_by_index = {item.detection_index: item.event.event_id for item in current}
    assert first_by_index == {0: "E000001", 1: "E000002"}
    assert current_by_index == first_by_index
    assert [item.method for item in current] == ["geometry", "geometry"]
    assert manager.events["E000001"].source_track_ids == [1, 11]
    assert manager.events["E000002"].source_track_ids == [2, 12]


def test_ambiguous_duplicate_geometry_does_not_force_merge() -> None:
    manager = EventManager(AssociationConfig(), "camera-a")
    box = (280, 280, 360, 440)
    first, _ = _observe_frame(manager, _frame(0, 0, (_detection(1, box), _detection(2, box))))
    second, _ = _observe_frame(manager, _frame(1, 100, (_detection(3, box), _detection(4, box))))

    assert [item.event.event_id for item in first] == ["E000001", "E000002"]
    assert [item.event.event_id for item in second] == ["E000003", "E000004"]
    assert [item.method for item in second] == ["new", "new"]
    assert manager.events["E000001"].state == EventState.LOST
    assert manager.events["E000002"].state == EventState.LOST


def test_incomplete_frame_freezes_unseen_lifecycle_then_complete_timeout_resolves() -> None:
    manager = EventManager(AssociationConfig(), "camera-a")
    box = (280, 280, 360, 440)
    _observe_frame(manager, _frame(0, 0, (_detection(7, box),)))
    _observe_frame(manager, _frame(1, 250, (_detection(7, box),)))

    _, error_changes = _observe_frame(manager, _frame(2, 900, status="processing_error"))
    event = manager.events["E000001"]
    assert error_changes == ()
    assert event.state == EventState.ACTIVE
    assert event.last_seen_ms == 250
    assert event.confirmation_count == 0

    _, timeout_changes = _observe_frame(manager, _frame(3, 1001))
    assert event.state == EventState.RESOLVED
    assert [(change.action, change.timestamp_ms) for change in timeout_changes] == [
        ("resolved", 1001)
    ]


def test_expired_event_does_not_revive_when_tracker_reuses_id() -> None:
    manager = EventManager(AssociationConfig(), "camera-a")
    box = (280, 280, 360, 440)
    first, _ = _observe_frame(manager, _frame(0, 0, (_detection(23, box),)))
    _observe_frame(manager, _frame(1, 751))
    replacement, _ = _observe_frame(manager, _frame(2, 752, (_detection(23, box),)))

    assert first[0].event.event_id == "E000001"
    assert manager.events["E000001"].state == EventState.RESOLVED
    assert replacement[0].event.event_id == "E000002"
    assert replacement[0].method == "new"


def test_alert_scheduler_suppresses_repeats_and_allows_qualified_escalation() -> None:
    config = AlertConfig()
    scheduler = AlertScheduler(config, "camera-a")
    monitor = PerceptionMonitor(config, "camera-a")

    first = scheduler.schedule((_snapshot("E000001"),), monitor, 250)
    unchanged = scheduler.schedule((_snapshot("E000001"),), monitor, 500)
    upgraded = scheduler.schedule((_snapshot("E000001", RiskLevel.HIGH, 0.58),), monitor, 1250)

    assert [(alert.alert_type, alert.timestamp_ms, alert.risk_level) for alert in first] == [
        ("first", 250, RiskLevel.MEDIUM)
    ]
    assert unchanged == ()
    assert [(alert.alert_type, alert.timestamp_ms, alert.risk_score) for alert in upgraded] == [
        ("escalation", 1250, 0.58)
    ]
    assert upgraded[0].source_track_ids == (7,)


def test_alert_priority_and_tie_break_are_stable_and_respect_per_frame_limit() -> None:
    config = AlertConfig(max_alerts_per_frame=1)
    scheduler = AlertScheduler(config, "camera-a")
    monitor = PerceptionMonitor(config, "camera-a")
    snapshots = (
        _snapshot("E000003", RiskLevel.MEDIUM, 0.6, 1.0),
        _snapshot("E000002", RiskLevel.HIGH, 0.8, 0.8),
        _snapshot("E000001", RiskLevel.HIGH, 0.8, 0.8),
    )

    alerts = scheduler.schedule(snapshots, monitor, 10)

    assert len(alerts) == 1
    assert alerts[0].event_id == "E000001"
    assert alerts[0].risk_level == RiskLevel.HIGH


def test_persistent_perception_failure_emits_separate_limited_alert() -> None:
    config = AlertConfig()
    scheduler = AlertScheduler(config, "camera-a")
    monitor = PerceptionMonitor(config, "camera-a")
    monitor.update(0, "processing_error", "decode failed")
    changes = monitor.update(500, "processing_error", "decode failed")

    limited = scheduler.schedule((), monitor, 500)
    repeated = scheduler.schedule((), monitor, 1000)
    resolved = monitor.update(1100, "available", None)

    assert any(change.action == "activated" and change.timestamp_ms == 500 for change in changes)
    assert len(limited) == 1
    assert limited[0].alert_type == "perception_limited"
    assert limited[0].event_id == "H000001"
    assert limited[0].risk_score is None
    assert "environment unknown" in limited[0].reason
    assert repeated == ()
    assert [(change.action, change.state) for change in resolved] == [
        ("resolved", EventState.RESOLVED)
    ]


def test_engine_stationary_target_emits_one_first_alert_at_confirmation_time() -> None:
    engine = AssessmentEngine(stream_id="camera-a")
    box = (280, 300, 360, 450)

    first = engine.step(_frame(0, 0, (_detection(7, box),)))
    confirmed = engine.step(_frame(1, 250, (_detection(7, box),)))
    stationary = engine.step(_frame(2, 500, (_detection(7, box),)))
    later = engine.step(_frame(3, 1000, (_detection(7, box),)))

    assert first.alerts == ()
    assert [
        (alert.event_id, alert.alert_type, alert.timestamp_ms) for alert in confirmed.alerts
    ] == [("E000001", "first", 250)]
    assert confirmed.frame.events[0].state == EventState.ACTIVE
    assert stationary.alerts == ()
    assert later.alerts == ()


def test_engine_emits_upgrade_after_score_increase_and_escalation_interval() -> None:
    engine = AssessmentEngine(stream_id="camera-a")

    def square(width: int, height: int) -> Detection:
        return _detection(7, (320 - width // 2, 336 - height, 320 + width // 2, 336))

    results = [
        engine.step(_frame(0, 0, (square(20, 40),))),
        engine.step(_frame(1, 250, (square(20, 40),))),
        engine.step(_frame(2, 500, (square(30, 60),))),
        engine.step(_frame(3, 750, (square(45, 90),))),
        engine.step(_frame(4, 1000, (square(68, 136),))),
        engine.step(_frame(5, 1250, (square(102, 204),))),
    ]
    assessments = [result.frame.observations[0].assessment for result in results]
    alerts = [alert for result in results for alert in result.alerts]

    assert assessments[1].level == RiskLevel.MEDIUM
    assert all(assessment.level == RiskLevel.HIGH for assessment in assessments[2:])
    assert [(alert.alert_type, alert.timestamp_ms, alert.risk_level) for alert in alerts] == [
        ("first", 250, RiskLevel.MEDIUM),
        ("escalation", 1250, RiskLevel.HIGH),
    ]
    assert alerts[1].risk_score >= alerts[0].risk_score + 0.05


def test_processing_failures_freeze_active_scene_event_and_emit_health_alert() -> None:
    engine = AssessmentEngine(stream_id="camera-a")
    box = (280, 300, 360, 450)
    engine.step(_frame(0, 0, (_detection(7, box),)))
    confirmed = engine.step(_frame(1, 250, (_detection(7, box),)))

    first_error = engine.step(_frame(2, 500, status="processing_error"))
    persistent_error = engine.step(_frame(3, 1000, status="processing_error"))
    recovered = engine.step(_frame(4, 1100))

    assert confirmed.frame.events[0].state == EventState.ACTIVE
    assert first_error.frame.events[0].state == EventState.ACTIVE
    assert first_error.frame.events[0].last_seen_ms == 250
    assert first_error.frame.perception.status == "unavailable"
    assert first_error.alerts == ()
    assert [
        (alert.alert_type, alert.timestamp_ms, alert.event_id) for alert in persistent_error.alerts
    ] == [("perception_limited", 1000, "H000001")]
    assert persistent_error.alerts[0].risk_score is None
    assert recovered.frame.perception.status == "available"
    assert any(
        change.event_kind == "perception" and change.action == "resolved"
        for change in recovered.changes
    )


def test_invalid_detection_makes_partial_frame_and_freezes_unseen_event() -> None:
    engine = AssessmentEngine(stream_id="camera-a")
    box = (280, 300, 360, 450)
    engine.step(_frame(0, 0, (_detection(7, box),)))
    engine.step(_frame(1, 250, (_detection(7, box),)))

    partial = engine.step(_frame(2, 500, (_detection(99, (320, 320, 320, 400)),)))

    assert partial.frame.warnings == (
        "detection_0: invalid_bbox: zero area or fully outside the image",
    )
    assert partial.frame.observations[0].error == partial.frame.warnings[0].removeprefix(
        "detection_0: "
    )
    assert partial.frame.events[0].event_id == "E000001"
    assert partial.frame.events[0].state == EventState.ACTIVE
    assert partial.frame.events[0].last_seen_ms == 250
    assert partial.frame.events[0].observed_now is False
    assert partial.frame.perception.status == "limited"
    assert partial.alerts == ()


def test_partial_frame_resets_candidate_confirmation_timer() -> None:
    engine = AssessmentEngine(stream_id="camera-a")
    box = (280, 300, 360, 450)
    first = engine.step(_frame(0, 0, (_detection(7, box),)))
    engine.step(_frame(1, 200, (_detection(99, (320, 320, 320, 400)),)))
    resumed = engine.step(_frame(2, 300, (_detection(7, box),)))
    before_confirmation = engine.step(_frame(3, 500, (_detection(7, box),)))
    confirmed = engine.step(_frame(4, 550, (_detection(7, box),)))

    assert first.frame.events[0].state == EventState.CANDIDATE
    assert resumed.frame.events[0].state == EventState.CANDIDATE
    assert resumed.frame.events[0].valid_observations == 2
    assert before_confirmation.frame.events[0].state == EventState.CANDIDATE
    assert confirmed.frame.events[0].state == EventState.ACTIVE
    assert any(
        change.action == "activated" and change.timestamp_ms == 550 for change in confirmed.changes
    )


def test_engine_tracks_null_ids_geometrically_without_inventing_source_ids() -> None:
    engine = AssessmentEngine(stream_id="camera-a")
    box = (280, 300, 360, 450)
    first = engine.step(_frame(0, 0, (_detection(None, box),)))
    confirmed = engine.step(_frame(1, 250, (_detection(None, box),)))

    assert first.frame.observations[0].association_method == "new"
    assert first.frame.observations[0].source_track_id is None
    assert confirmed.frame.observations[0].association_method == "geometry"
    assert confirmed.frame.observations[0].event_id == "E000001"
    assert confirmed.frame.events[0].source_track_ids == ()


def test_engine_rejects_noncontiguous_frames_and_decreasing_video_time() -> None:
    engine = AssessmentEngine(stream_id="camera-a")
    engine.step(_frame(0, 100))

    with pytest.raises(ValueError, match="contiguous"):
        engine.step(_frame(2, 200))
    with pytest.raises(ValueError, match="timestamps must not decrease"):
        engine.step(_frame(1, 99))


def test_duplicate_timestamps_are_accepted_without_negative_motion() -> None:
    engine = AssessmentEngine(stream_id="camera-a")
    box = (280, 300, 360, 450)
    engine.step(_frame(0, 100, (_detection(7, box),)))
    duplicate = engine.step(_frame(1, 100, (_detection(7, box),)))

    feature = duplicate.frame.observations[0].features
    assert duplicate.frame.warnings == ("duplicate_timestamp_motion_unavailable",)
    assert feature.observation_interval_ms == 0
    assert feature.scale_change_per_s is None
    assert feature.center_change_per_s is None
    assert feature.quality.motion_reliable is False


def test_low_frame_rate_observation_and_configured_recovery_window_preserve_event() -> None:
    config = RiskConfig(
        association=AssociationConfig(
            recovery_window_ms=900,
            confirmation_max_gap_ms=800,
        )
    )
    engine = AssessmentEngine(config, "camera-a")
    box = (280, 300, 360, 450)
    first = engine.step(_frame(0, 0, (_detection(7, box),)))
    confirmed = engine.step(_frame(1, 700, (_detection(7, box),)))
    lost = engine.step(_frame(2, 1400))
    recovered = engine.step(_frame(3, 1600, (_detection(99, box),)))

    assert first.frame.events[0].state == EventState.CANDIDATE
    assert any(
        change.action == "activated" and change.timestamp_ms == 700 for change in confirmed.changes
    )
    assert lost.frame.events[0].state == EventState.LOST
    assert recovered.frame.events[0].event_id == "E000001"
    assert recovered.frame.events[0].state == EventState.ACTIVE
    assert recovered.frame.events[0].source_track_ids == (7, 99)
    assert any(
        change.action == "recovered" and change.timestamp_ms == 1600 for change in recovered.changes
    )


def test_assessment_engines_isolate_stream_ids_and_censor_at_eof() -> None:
    box = (280, 300, 360, 450)
    left = AssessmentEngine(stream_id="video-left")
    right = AssessmentEngine(stream_id="video-right")
    left.step(_frame(0, 0, (_detection(7, box),)))
    left.step(_frame(1, 250, (_detection(7, box),)))
    right_first = right.step(_frame(0, 0, (_detection(7, box),)))

    censored = left.finish()
    assert right_first.frame.events[0].event_id == "E000001"
    assert right_first.frame.events[0].last_seen_ms == 0
    assert right_first.frame.stream_id == "video-right"
    assert [(change.event_id, change.action, change.state) for change in censored] == [
        ("E000001", "censored", EventState.ACTIVE)
    ]
    assert left.finish() == ()


def test_prefix_results_do_not_depend_on_future_frames() -> None:
    box = (280, 300, 360, 450)
    prefix = (
        _frame(0, 0, (_detection(7, box),)),
        _frame(1, 250, (_detection(7, box),)),
    )
    short_engine = AssessmentEngine(stream_id="same-video")
    long_engine = AssessmentEngine(stream_id="same-video")
    short_results = tuple(short_engine.step(frame) for frame in prefix)
    long_results = tuple(long_engine.step(frame) for frame in prefix)
    long_engine.step(_frame(2, 500, (_detection(7, (220, 250, 420, 450)),)))

    assert long_results == short_results


def test_engine_replay_is_deterministic_for_same_video_records() -> None:
    frames = (
        _frame(0, 0, (_detection(4, (280, 300, 360, 450)),)),
        _frame(1, 250, (_detection(4, (280, 300, 360, 450)),)),
        _frame(2, 500),
        _frame(3, 700, (_detection(99, (280, 300, 360, 450)),)),
    )
    results = []
    for _ in range(2):
        engine = AssessmentEngine(stream_id="repeatable-video")
        results.append(tuple(engine.step(frame) for frame in frames))

    assert results[0] == results[1]
