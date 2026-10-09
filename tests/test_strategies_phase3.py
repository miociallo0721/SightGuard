from dataclasses import asdict

import pytest

from sightguard.data import Detection, FrameRecord
from sightguard.evaluation.config import BenchmarkConfig
from sightguard.evaluation.schema import StrategyAlert
from sightguard.evaluation.strategies import run_strategies
from sightguard.risk.config import AlertConfig, AssociationConfig, RiskConfig
from sightguard.risk.engine import AssessmentEngine
from sightguard.risk.schema import RiskLevel


def _detection(
    track_id: int | None,
    bbox: tuple[float, float, float, float],
    *,
    class_id: int = 0,
    class_name: str | None = None,
) -> Detection:
    names = {0: "person", 1: "bicycle", 3: "motorcycle"}
    return Detection(track_id, class_id, class_name or names.get(class_id, "object"), 0.9, bbox)


def _frame(
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
        error="synthetic decode failure" if status == "processing_error" else None,
        timestamp_source="video_pts",
    )


def _run(
    records: tuple[FrameRecord, ...],
    *,
    video_id: str = "synthetic-video",
    risk_config: RiskConfig | None = None,
    config: BenchmarkConfig | None = None,
    strategies: tuple[str, ...] = ("A", "B", "C"),
):
    return run_strategies(
        records,
        video_id,
        risk_config or RiskConfig(),
        config or BenchmarkConfig(generate_figures=False),
        strategies,
    )


def _stationary_records() -> tuple[FrameRecord, ...]:
    box = (280, 300, 360, 450)
    return tuple(
        _frame(index, timestamp, (_detection(7, box),))
        for index, timestamp in enumerate(range(0, 3251, 250))
    )


def _expected_engine_frames(
    records: tuple[FrameRecord, ...], risk_config: RiskConfig, video_id: str
):
    engine = AssessmentEngine(risk_config, stream_id=video_id)
    return tuple(engine.step(record).frame for record in records)


def _escalation_records() -> tuple[FrameRecord, ...]:
    def square(width: int, height: int) -> Detection:
        return _detection(7, (320 - width // 2, 336 - height, 320 + width // 2, 336))

    dimensions = ((20, 40), (20, 40), (30, 60), (45, 90), (68, 136), (102, 204))
    timestamps = (0, 250, 500, 750, 1000, 1250)
    return tuple(
        _frame(index, timestamp, (square(*size),))
        for index, (timestamp, size) in enumerate(zip(timestamps, dimensions, strict=True))
    )


def _alert_signature(alert: StrategyAlert) -> tuple:
    return (
        alert.alert_id,
        alert.event_id,
        alert.timestamp_ms,
        alert.risk_level,
        alert.risk_score,
        alert.alert_type,
        alert.reason,
        alert.source_track_ids,
        alert.class_id,
        alert.bbox_normalized,
        alert.frame_index,
    )


def test_all_strategies_share_exact_risk_frames_and_have_distinct_decisions() -> None:
    records = _stationary_records()
    risk_config = RiskConfig()
    run = _run(records, risk_config=risk_config)
    engine_frames = _expected_engine_frames(records, risk_config, "synthetic-video")

    assert run.risk_frames == engine_frames
    assert tuple(run.alerts) == ("A", "B", "C")
    assert len(run.alerts["A"]) == len(records)
    assert [alert.timestamp_ms for alert in run.alerts["B"]] == [0, 3000]
    assert [(alert.alert_type, alert.timestamp_ms) for alert in run.alerts["C"]] == [("first", 250)]
    assert all(alert.alert_type == "trigger" for alert in run.alerts["A"])
    assert all(alert.risk_level == RiskLevel.MEDIUM for alert in run.alerts["A"])
    assert all(alert.risk_score is None for alert in run.alerts["A"])
    assert all(alert.risk_score is not None for alert in run.alerts["B"])
    assert all(alert.timestamp_ms in {0, 3000} for alert in run.alerts["B"])


def test_strategy_b_chooses_highest_priority_target_and_fires_without_confirmation() -> None:
    records = (
        _frame(0, 40, (_detection(1, (220, 260, 420, 450)), _detection(2, (310, 350, 330, 390)))),
    )
    run = _run(records, strategies=("B",))

    assert tuple(run.alerts) == ("B",)
    assert len(run.alerts["B"]) == 1
    assert run.alerts["B"][0].source_track_ids == (1,)
    assert run.alerts["B"][0].timestamp_ms == 40
    assert run.alerts["B"][0].alert_type == "trigger"
    assert run.alerts["B"][0].risk_score == max(
        observation.assessment.score for observation in run.risk_frames[0].observations
    )
    assert run.alerts["B"][0].event_id.startswith("E")
    assert run.risk_frames[0].events[0].confirmed is False


def test_baseline_a_uses_fixed_class_and_focus_gates() -> None:
    config = BenchmarkConfig(
        baseline_classes=(0,), baseline_focus_threshold=0.8, generate_figures=False
    )
    records = (
        _frame(
            0,
            0,
            (
                _detection(1, (280, 300, 360, 450), class_id=0),
                _detection(2, (280, 300, 360, 450), class_id=1),
                _detection(3, (0, 300, 64, 420), class_id=0),
            ),
        ),
    )
    run = _run(records, config=config, strategies=("A",))

    assert len(run.alerts["A"]) == 1
    alert = run.alerts["A"][0]
    assert alert.class_id == 0
    assert alert.source_track_ids == (1,)
    assert alert.bbox_normalized == pytest.approx((280 / 640, 300 / 480, 360 / 640, 450 / 480))
    assert run.risk_frames[0].observations[2].features.spatial.focus_relevance < 0.8


def test_strategy_c_alerts_match_the_existing_assessment_engine_including_escalation() -> None:
    records = _escalation_records()
    risk_config = RiskConfig()
    run = _run(records, risk_config=risk_config, strategies=("C",))
    engine = AssessmentEngine(risk_config, stream_id="synthetic-video")
    core_alerts = []
    core_frames = []
    for record in records:
        step = engine.step(record)
        core_frames.append(step.frame)
        core_alerts.extend(step.alerts)

    strategy_alerts = run.alerts["C"]
    assert run.risk_frames == tuple(core_frames)
    assert [
        (alert.alert_type, alert.timestamp_ms, alert.risk_level) for alert in strategy_alerts
    ] == [
        ("first", 250, RiskLevel.MEDIUM),
        ("escalation", 1250, RiskLevel.HIGH),
    ]
    assert [_alert_signature(alert)[1:8] for alert in strategy_alerts] == [
        (
            alert.event_id,
            alert.timestamp_ms,
            alert.risk_level,
            alert.risk_score,
            alert.alert_type,
            alert.reason,
            alert.source_track_ids,
        )
        for alert in core_alerts
    ]
    assert [alert.alert_id for alert in strategy_alerts] == [
        alert.alert_id for alert in core_alerts
    ]
    assert [alert.frame_index for alert in strategy_alerts] == [1, 5]
    assert strategy_alerts[1].risk_score >= strategy_alerts[0].risk_score + 0.05


def test_baseline_cooldown_uses_irregular_video_timestamps() -> None:
    box = (280, 300, 360, 450)
    times = (0, 700, 1550, 3100)
    records = tuple(
        _frame(index, timestamp, (_detection(7, box),)) for index, timestamp in enumerate(times)
    )
    risk_config = RiskConfig(
        association=AssociationConfig(confirmation_max_gap_ms=1000, recovery_window_ms=1200),
        alerts=AlertConfig(max_frame_gap_ms=2000),
    )
    run = _run(records, risk_config=risk_config)

    assert [alert.timestamp_ms for alert in run.alerts["A"]] == list(times)
    assert [alert.timestamp_ms for alert in run.alerts["B"]] == [0, 3100]
    assert [alert.timestamp_ms for alert in run.alerts["C"]] == [700]


def test_failed_perception_frames_create_no_scene_alerts_for_any_strategy() -> None:
    records = (
        _frame(0, 0, status="processing_error"),
        _frame(1, 500, status="processing_error"),
    )
    run = _run(records)

    assert set(run.alerts) == {"A", "B", "C"}
    for alerts in run.alerts.values():
        assert len(alerts) == 1
        assert alerts[0].alert_type == "perception_limited"
        assert alerts[0].timestamp_ms == 500
        assert alerts[0].risk_level is None
        assert alerts[0].risk_score is None
        assert alerts[0].class_id is None
        assert alerts[0].bbox_normalized is None
    assert [frame.perception.status for frame in run.risk_frames] == [
        "unavailable",
        "unavailable",
    ]


def test_single_strategy_mode_and_separate_videos_keep_local_alert_state() -> None:
    records = _stationary_records()
    first = _run(records, video_id="video-one", strategies=("B",))
    second = _run(records, video_id="video-two", strategies=("B",))
    only_c = _run(records[:2], video_id="video-three", strategies=("C",))

    assert tuple(first.alerts) == ("B",)
    assert tuple(second.alerts) == ("B",)
    assert [alert.video_id for alert in first.alerts["B"]] == ["video-one", "video-one"]
    assert [alert.video_id for alert in second.alerts["B"]] == ["video-two", "video-two"]
    assert [alert.timestamp_ms for alert in first.alerts["B"]] == [
        alert.timestamp_ms for alert in second.alerts["B"]
    ]
    assert first.alerts["B"][0].alert_id == second.alerts["B"][0].alert_id == "A000001"
    assert tuple(only_c.alerts) == ("C",)
    assert [alert.timestamp_ms for alert in only_c.alerts["C"]] == [250]


def test_strategy_outputs_are_prefix_causal_and_deterministic() -> None:
    records = _stationary_records()
    prefix = records[:5]
    short = _run(prefix, video_id="causal-video")
    long = _run(records, video_id="causal-video")
    replay = _run(records, video_id="causal-video")

    assert long.risk_frames[: len(prefix)] == short.risk_frames
    for strategy in ("A", "B", "C"):
        short_prefix = tuple(alert for alert in short.alerts[strategy])
        long_prefix = tuple(
            alert for alert in long.alerts[strategy] if alert.frame_index < len(prefix)
        )
        assert long_prefix == short_prefix
    assert long == replay


def test_each_baseline_alert_score_comes_from_the_shared_risk_frame() -> None:
    records = _stationary_records()[:3]
    run = _run(records)
    b_alert = run.alerts["B"][0]
    observation = next(
        row
        for row in run.risk_frames[b_alert.frame_index].observations
        if row.event_id == b_alert.event_id
    )

    assert b_alert.risk_score == observation.assessment.score
    assert b_alert.risk_level == observation.assessment.level
    assert run.alerts["C"][0].risk_score == run.risk_frames[1].observations[0].assessment.score
    assert asdict(run.risk_frames[0].observations[0].assessment) == asdict(
        _expected_engine_frames(records, RiskConfig(), "synthetic-video")[0]
        .observations[0]
        .assessment
    )
