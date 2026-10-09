from __future__ import annotations

from dataclasses import replace

import pytest

from sightguard.evaluation.config import MatchingConfig
from sightguard.evaluation.matching import evaluate_video, prepare_ground_truth
from sightguard.evaluation.metrics import aggregate_metrics
from sightguard.evaluation.schema import (
    AlertOpportunity,
    ExcludedInterval,
    LabeledEvent,
    StrategyAlert,
    TargetRegion,
    VideoAnnotations,
)
from sightguard.risk.schema import RiskLevel


def _opportunity(
    opportunity_id: str,
    start_ms: float,
    deadline_ms: float,
    *,
    kind: str = "initial",
    severity: RiskLevel = RiskLevel.MEDIUM,
) -> AlertOpportunity:
    return AlertOpportunity(
        opportunity_id=opportunity_id,
        kind=kind,
        window_start_ms=start_ms,
        deadline_ms=deadline_ms,
        minimum_severity=severity,
    )


def _event(
    event_id: str = "gt-1",
    *,
    start_ms: float = 0,
    end_ms: float = 5000,
    severity: RiskLevel = RiskLevel.HIGH,
    opportunities: tuple[AlertOpportunity, ...] | None = None,
    regions: tuple[TargetRegion, ...] = (),
    class_ids: tuple[int, ...] = (2,),
    status: str = "confirmed",
) -> LabeledEvent:
    if opportunities is None:
        opportunities = (_opportunity(f"{event_id}-initial", 1000, 2000),)
    return LabeledEvent(
        video_id="video-1",
        event_id=event_id,
        event_type="crossing_obstacle",
        start_ms=start_ms,
        end_ms=end_ms,
        severity=severity,
        annotation_status=status,
        alert_opportunities=opportunities,
        annotator_id="independent-annotator",
        notes="synthetic fixture",
        class_ids=class_ids,
        target_regions=regions,
    )


def _labels(
    events: tuple[LabeledEvent, ...] = (),
    *,
    duration_ms: float = 10_000,
    excluded: tuple[ExcludedInterval, ...] = (),
) -> VideoAnnotations:
    return VideoAnnotations(
        video_id="video-1",
        duration_ms=duration_ms,
        data_kind="synthetic",
        annotation_status="reviewed",
        annotator_id="independent-annotator",
        label_source="synthetic_scenario",
        source_video_sha256="synthetic-fixture",
        notes="unit-test labels",
        events=events,
        excluded_intervals=excluded,
    )


def _alert(
    alert_id: str,
    timestamp_ms: float,
    *,
    alert_type: str = "first",
    class_id: int | None = 2,
    severity: RiskLevel | None = RiskLevel.MEDIUM,
    bbox: tuple[float, float, float, float] | None = None,
    video_id: str = "video-1",
    strategy: str = "C",
) -> StrategyAlert:
    return StrategyAlert(
        video_id=video_id,
        strategy=strategy,
        alert_id=alert_id,
        event_id=f"model-event-{alert_id}",
        timestamp_ms=timestamp_ms,
        risk_level=severity,
        risk_score=0.7 if severity is not None else None,
        alert_type=alert_type,
        reason="synthetic strategy output",
        source_track_ids=(100 + len(alert_id),),
        class_id=class_id,
        bbox_normalized=bbox,
    )


def test_single_initial_alert_is_a_true_positive() -> None:
    result = evaluate_video(
        _labels((_event(),)),
        (_alert("a1", 1250),),
        MatchingConfig(),
        strategy="C",
    )

    assert (result.metrics["tp"], result.metrics["fp"], result.metrics["fn"]) == (1, 0, 0)
    assert result.metrics["event_recall"] == 1
    assert result.metrics["opportunity_recall"] == 1
    assert result.metrics["response_delays_ms"] == [250]
    assert result.metrics["first_response_delay_mean_ms"] == 250
    assert result.event_rows[0]["detected"] is True
    assert result.matches[0].gt_event_id == "gt-1"


def test_draft_annotations_cannot_be_evaluated() -> None:
    labels = replace(_labels((_event(),)), annotation_status="draft")

    with pytest.raises(ValueError, match="Draft annotations"):
        evaluate_video(labels, (), MatchingConfig(), strategy="C")


def test_alert_from_another_video_is_rejected() -> None:
    with pytest.raises(ValueError, match="video_id"):
        evaluate_video(
            _labels(),
            (_alert("wrong-video", 1000, video_id="video-2"),),
            MatchingConfig(),
            strategy="C",
        )


def test_alert_from_another_strategy_is_rejected() -> None:
    with pytest.raises(ValueError, match="strategy"):
        evaluate_video(
            _labels(),
            (_alert("wrong-strategy", 1000, strategy="A"),),
            MatchingConfig(),
            strategy="C",
        )


def test_duplicate_alert_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="Duplicate alert_id"):
        evaluate_video(
            _labels(),
            (_alert("same-id", 1000), _alert("same-id", 1200)),
            MatchingConfig(),
            strategy="C",
        )


def test_complete_miss_is_an_explicit_false_negative() -> None:
    result = evaluate_video(_labels((_event(),)), (), MatchingConfig(), strategy="A")

    assert (result.metrics["tp"], result.metrics["fp"], result.metrics["fn"]) == (0, 0, 1)
    assert result.metrics["missed_events"] == 1
    assert result.metrics["event_recall"] == 0
    assert result.metrics["first_response_delay_mean_ms"] is None
    assert result.metrics["response_delays_ms"] == []
    assert [match.status for match in result.matches] == ["missed_opportunity"]
    assert result.event_rows[0]["initial_latency_ms"] is None


def test_unimportant_initial_match_does_not_inflate_important_event_rate() -> None:
    low_event = _event(
        "low",
        severity=RiskLevel.LOW,
        opportunities=(_opportunity("low-initial", 1000, 2000, severity=RiskLevel.LOW),),
    )
    important_miss = _event("important-miss")
    result = evaluate_video(
        _labels((low_event, important_miss)),
        (_alert("low-alert", 1200, severity=RiskLevel.LOW, strategy="A"),),
        MatchingConfig(),
        strategy="A",
    )

    assert result.metrics["tp"] == 1
    assert result.metrics["important_events"] == 1
    assert result.metrics["detected_events"] == 0
    assert result.metrics["on_time_event_rate"] == 0
    assert result.metrics["event_recall"] == 0
    assert result.metrics["response_delays_ms"] == []


def test_repeated_same_event_alerts_are_fp_and_duplicate_subset() -> None:
    result = evaluate_video(
        _labels((_event(),)),
        (
            _alert("first", 1100, strategy="B"),
            _alert("repeat-1", 2500, alert_type="repeat", strategy="B"),
            _alert("repeat-2", 3000, alert_type="trigger", strategy="B"),
        ),
        MatchingConfig(),
        strategy="B",
    )

    assert (result.metrics["tp"], result.metrics["fp"], result.metrics["fn"]) == (1, 2, 0)
    assert result.metrics["duplicates"] == 2
    assert result.metrics["unsupported"] == 0
    assert result.metrics["duplicate_rate"] == 2 / 3
    assert [match.status for match in result.matches[:3]] == ["matched", "duplicate", "duplicate"]


def test_supported_escalation_matches_and_unjustified_escalation_is_duplicate() -> None:
    event = _event(
        end_ms=5000,
        opportunities=(
            _opportunity("initial", 1000, 1500),
            _opportunity(
                "escalation",
                3000,
                3500,
                kind="escalation",
                severity=RiskLevel.HIGH,
            ),
        ),
    )
    result = evaluate_video(
        _labels((event,)),
        (
            _alert("initial-alert", 1100),
            _alert("supported-upgrade", 3200, alert_type="escalation", severity=RiskLevel.HIGH),
            _alert("unjustified-upgrade", 4000, alert_type="escalation", severity=RiskLevel.HIGH),
        ),
        MatchingConfig(),
        strategy="C",
    )

    assert (result.metrics["tp"], result.metrics["fp"], result.metrics["fn"]) == (2, 1, 0)
    assert result.metrics["duplicates"] == 1
    assert result.matches[1].opportunity_kind == "escalation"
    assert result.matches[2].reason == "unjustified_or_repeated_escalation"


def test_opportunity_window_boundaries_are_inclusive() -> None:
    event = _event(start_ms=900, end_ms=3000, opportunities=(_opportunity("initial", 1000, 2000),))
    result = evaluate_video(
        _labels((event,)),
        (
            _alert("early", 999, strategy="A"),
            _alert("at-deadline", 2000, strategy="A"),
            _alert("late", 2001, strategy="A"),
        ),
        MatchingConfig(),
        strategy="A",
    )

    assert (result.metrics["tp"], result.metrics["fp"], result.metrics["fn"]) == (1, 2, 0)
    assert result.metrics["duplicates"] == 1
    statuses = {match.alert_id: match.status for match in result.matches if match.alert_id}
    assert statuses == {"early": "false_positive", "at-deadline": "matched", "late": "duplicate"}


def test_overlapping_same_class_events_are_separated_by_target_regions() -> None:
    first = _event(
        "left",
        start_ms=0,
        end_ms=5000,
        regions=(TargetRegion(0, 5000, (0.1, 0.1, 0.4, 0.5)),),
    )
    second = _event(
        "right",
        start_ms=0,
        end_ms=5000,
        regions=(TargetRegion(0, 5000, (0.6, 0.1, 0.9, 0.5)),),
    )
    result = evaluate_video(
        _labels((first, second)),
        (
            _alert("left-alert", 1200, bbox=(0.15, 0.2, 0.3, 0.4)),
            _alert("right-alert", 1300, bbox=(0.7, 0.2, 0.8, 0.4)),
        ),
        MatchingConfig(minimum_iou=0.1),
        strategy="C",
    )

    assert (result.metrics["tp"], result.metrics["fp"], result.metrics["fn"]) == (2, 0, 0)
    matches = {match.alert_id: match.gt_event_id for match in result.matches if match.alert_id}
    assert matches == {"left-alert": "left", "right-alert": "right"}


def test_augmenting_path_finds_maximum_cardinality_matching() -> None:
    # The first alert can match either event, while the second can match only
    # the first. A greedy first-fit assignment would leave one opportunity free.
    first = _event(
        "event-a",
        start_ms=0,
        end_ms=5000,
        regions=(TargetRegion(0, 5000, (0.1, 0.1, 0.5, 0.5)),),
    )
    second = _event(
        "event-b",
        start_ms=0,
        end_ms=5000,
        regions=(TargetRegion(0, 5000, (0.2, 0.1, 0.6, 0.5)),),
    )
    result = evaluate_video(
        _labels((first, second)),
        (
            _alert("flexible", 1100, bbox=(0.25, 0.1, 0.35, 0.2)),
            _alert("only-a", 1200, bbox=(0.1, 0.1, 0.2, 0.2)),
        ),
        MatchingConfig(minimum_iou=0.05),
        strategy="C",
    )

    assert (result.metrics["tp"], result.metrics["fp"], result.metrics["fn"]) == (2, 0, 0)
    matches = {match.alert_id: match.gt_event_id for match in result.matches if match.alert_id}
    assert matches == {"flexible": "event-b", "only-a": "event-a"}


def test_overlapping_same_class_events_without_regions_report_ambiguity() -> None:
    first = _event("event-a", start_ms=0, end_ms=5000)
    second = _event("event-b", start_ms=0, end_ms=5000)
    result = evaluate_video(
        _labels((first, second)),
        (_alert("ambiguous", 1200),),
        MatchingConfig(),
        strategy="C",
    )

    assert (result.metrics["tp"], result.metrics["fp"], result.metrics["fn"]) == (0, 1, 2)
    assert result.metrics["duplicates"] == 0
    assert result.matches[0].reason == "ambiguous_ground_truth_identity_without_spatial_regions"


def test_regions_inactive_at_alert_time_do_not_match_or_establish_duplicates() -> None:
    event = _event(
        "region-limited",
        regions=(TargetRegion(0, 2000, (0.1, 0.1, 0.5, 0.5)),),
    )
    result = evaluate_video(
        _labels((event,)),
        (
            _alert("initial", 1200, bbox=(0.2, 0.2, 0.4, 0.4)),
            _alert("outside-region", 2500, alert_type="repeat", bbox=(0.2, 0.2, 0.4, 0.4)),
        ),
        MatchingConfig(),
        strategy="C",
    )

    assert (result.metrics["tp"], result.metrics["fp"], result.metrics["fn"]) == (1, 1, 0)
    assert result.metrics["duplicates"] == 0
    assert result.metrics["unsupported"] == 1
    assert result.matches[1].status == "false_positive"


def test_exclusion_union_expands_over_overlapping_events_but_not_boundary_touches() -> None:
    uncertain = _event(
        "uncertain",
        start_ms=1500,
        end_ms=2500,
        opportunities=(),
        status="uncertain",
    )
    first_confirmed = _event(
        "censored-1",
        start_ms=2400,
        end_ms=3000,
        opportunities=(_opportunity("c1", 2550, 2600),),
    )
    second_confirmed = _event(
        "censored-2",
        start_ms=2900,
        end_ms=4000,
        opportunities=(_opportunity("c2", 3100, 3200),),
    )
    event_ending_at_mask_start = _event(
        "touching-before",
        start_ms=500,
        end_ms=1000,
        opportunities=(_opportunity("before-opportunity", 600, 700),),
    )
    event_starting_at_mask_end = _event(
        "touching-after",
        start_ms=4000,
        end_ms=4500,
        opportunities=(_opportunity("after-opportunity", 4100, 4200),),
    )
    labels = _labels(
        (
            uncertain,
            first_confirmed,
            second_confirmed,
            event_ending_at_mask_start,
            event_starting_at_mask_end,
        ),
        excluded=(ExcludedInterval(1000, 2000, "camera obscured"),),
    )
    eligible, union, excluded_ids = prepare_ground_truth(labels)
    result = evaluate_video(
        labels,
        (
            _alert("inside-unknown", 1800, strategy="B"),
            _alert("inside-censored-1", 2700, strategy="B"),
            _alert("before-boundary", 650, strategy="B"),
            _alert("after-boundary", 4150, strategy="B"),
        ),
        MatchingConfig(),
        strategy="B",
    )

    assert {event.event_id for event in eligible} == {"touching-before", "touching-after"}
    assert union == ((1000, 4000),)
    assert excluded_ids == frozenset({"uncertain", "censored-1", "censored-2"})
    assert result.metrics["effective_duration_ms"] == 7000
    assert result.metrics["excluded_duration_ms"] == 3000
    assert result.metrics["opportunities"] == 2
    assert result.metrics["ignored_alerts"] == 2
    assert (result.metrics["tp"], result.metrics["fp"], result.metrics["fn"]) == (2, 0, 0)


def test_empty_scene_with_perception_diagnostic_has_no_scene_fp() -> None:
    labels = _labels(duration_ms=60_000)
    diagnostic = _alert(
        "perception",
        1000,
        alert_type="perception_limited",
        class_id=None,
        severity=None,
        strategy="A",
    )
    result = evaluate_video(labels, (diagnostic,), MatchingConfig(), strategy="A")
    pooled = aggregate_metrics((result,))

    assert result.metrics["tp"] == result.metrics["fp"] == result.metrics["fn"] == 0
    assert result.metrics["scene_alerts"] == 0
    assert result.metrics["perception_alerts"] == 1
    assert result.metrics["total_alerts"] == 1
    assert result.metrics["total_alerts_per_minute"] == 1
    assert result.metrics["event_recall"] is None
    assert result.metrics["invalid_alerts_per_hour"] == 0
    assert pooled["total_alerts"] == 1
    assert pooled["first_response_delay_mean_ms"] is None
