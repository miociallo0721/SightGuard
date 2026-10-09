"""Independent matching of strategy alerts to human-labelled opportunities."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from sightguard.evaluation.config import MatchingConfig
from sightguard.evaluation.schema import (
    AlertOpportunity,
    EvaluationResult,
    LabeledEvent,
    MatchRecord,
    StrategyAlert,
    VideoAnnotations,
)


@dataclass(frozen=True)
class _Opportunity:
    event: LabeledEvent
    opportunity: AlertOpportunity


def _merge_intervals(intervals: Iterable[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    ordered = sorted(intervals)
    merged: list[list[float]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return tuple((start, end) for start, end in merged)


def _intersects(left: tuple[float, float], right: tuple[float, float]) -> bool:
    """Return whether half-open intervals share positive duration."""
    return max(left[0], right[0]) < min(left[1], right[1])


def prepare_ground_truth(
    labels: VideoAnnotations,
) -> tuple[tuple[LabeledEvent, ...], tuple[tuple[float, float], ...], frozenset[str]]:
    """Apply the predeclared exclusion policy and return usable events and intervals.

    Explicitly excluded intervals and uncertain/excluded events seed the exclusion
    union. A confirmed event is censored only when its half-open interval overlaps
    that union for positive duration; sharing a boundary alone does not censor it.
    Censoring is repeated because one event can extend the union into another.
    """
    excluded_event_ids = {
        event.event_id for event in labels.events if event.annotation_status != "confirmed"
    }
    intervals: list[tuple[float, float]] = [
        (item.start_ms, item.end_ms) for item in labels.excluded_intervals
    ]
    intervals.extend(
        (event.start_ms, event.end_ms)
        for event in labels.events
        if event.annotation_status != "confirmed"
    )
    merged = _merge_intervals(intervals)

    censored_confirmed: set[str] = set()
    while True:
        newly_censored = [
            event
            for event in labels.events
            if event.annotation_status == "confirmed"
            and event.event_id not in censored_confirmed
            and any(_intersects((event.start_ms, event.end_ms), interval) for interval in merged)
        ]
        if not newly_censored:
            break
        for event in newly_censored:
            censored_confirmed.add(event.event_id)
            intervals.append((event.start_ms, event.end_ms))
        merged = _merge_intervals(intervals)

    excluded_event_ids.update(censored_confirmed)
    eligible = tuple(
        event
        for event in labels.events
        if event.annotation_status == "confirmed" and event.event_id not in censored_confirmed
    )
    return eligible, merged, frozenset(excluded_event_ids)


def _interval_contains(intervals: tuple[tuple[float, float], ...], timestamp_ms: float) -> bool:
    return any(start <= timestamp_ms < end for start, end in intervals)


def _iou(
    first: tuple[float, float, float, float], second: tuple[float, float, float, float]
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_first = (first[2] - first[0]) * (first[3] - first[1])
    area_second = (second[2] - second[0]) * (second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union > 0 else 0.0


def _active_regions(event: LabeledEvent, timestamp_ms: float):
    return tuple(
        region
        for region in event.target_regions
        if region.start_ms <= timestamp_ms <= region.end_ms
    )


def _spatially_matches(
    alert: StrategyAlert, event: LabeledEvent, timestamp_ms: float, config: MatchingConfig
) -> bool:
    if not event.target_regions:
        return True
    regions = _active_regions(event, timestamp_ms)
    if not regions:
        return False
    if alert.bbox_normalized is None:
        return False
    return any(
        _iou(alert.bbox_normalized, region.bbox_normalized) >= config.minimum_iou
        for region in regions
    )


def _kind_matches(alert_type: str, opportunity_kind: str) -> bool:
    if alert_type == "first":
        return opportunity_kind == "initial"
    if alert_type == "escalation":
        return opportunity_kind == "escalation"
    # A trigger/repeat is eligible for either kind when the independent label
    # marks a new opportunity. The algorithm's event ID is intentionally unused.
    return alert_type in ("trigger", "repeat")


def _eligible_edge(alert: StrategyAlert, candidate: _Opportunity, config: MatchingConfig) -> bool:
    event = candidate.event
    opportunity = candidate.opportunity
    if alert.video_id != event.video_id:
        return False
    if alert.class_id is None or alert.class_id not in event.class_ids:
        return False
    if alert.risk_level is None or alert.risk_level.rank < opportunity.minimum_severity.rank:
        return False
    if not _kind_matches(alert.alert_type, opportunity.kind):
        return False
    if not opportunity.window_start_ms <= alert.timestamp_ms <= opportunity.deadline_ms:
        return False
    return _spatially_matches(alert, event, alert.timestamp_ms, config)


def _ambiguous_event_identity(
    alert: StrategyAlert,
    edge_indices: tuple[int, ...],
    opportunities: tuple[_Opportunity, ...],
) -> bool:
    event_by_id = {item.event.event_id: item.event for item in opportunities}
    candidate_events = {opportunities[index].event.event_id for index in edge_indices}
    if len(candidate_events) < 2:
        return False
    events = [event_by_id[event_id] for event_id in sorted(candidate_events)]
    for index, first in enumerate(events):
        for second in events[index + 1 :]:
            same_class = bool(set(first.class_ids).intersection(second.class_ids))
            both_active = (
                first.start_ms <= alert.timestamp_ms <= first.end_ms
                and second.start_ms <= alert.timestamp_ms <= second.end_ms
            )
            if (
                same_class
                and both_active
                and (
                    not _active_regions(first, alert.timestamp_ms)
                    or not _active_regions(second, alert.timestamp_ms)
                )
            ):
                return True
    return False


def _identity_candidates(
    alert: StrategyAlert, events: tuple[LabeledEvent, ...], config: MatchingConfig
) -> tuple[LabeledEvent, ...]:
    if alert.class_id is None:
        return ()
    return tuple(
        event
        for event in events
        if event.video_id == alert.video_id
        and alert.class_id in event.class_ids
        and event.start_ms <= alert.timestamp_ms <= event.end_ms
        and _spatially_matches(alert, event, alert.timestamp_ms, config)
    )


def _stable_alert_key(item: tuple[int, StrategyAlert]) -> tuple[float, int, str, int]:
    index, alert = item
    return (alert.timestamp_ms, alert.frame_index, alert.alert_id, index)


def evaluate_video(
    labels: VideoAnnotations,
    alerts: tuple[StrategyAlert, ...],
    config: MatchingConfig,
    *,
    strategy: str,
) -> EvaluationResult:
    """Evaluate one video using deterministic maximum-cardinality matching."""
    if labels.annotation_status != "reviewed":
        raise ValueError("Draft annotations cannot be evaluated")
    seen_alert_ids: set[str] = set()
    for alert in alerts:
        if alert.video_id != labels.video_id:
            raise ValueError("Alert video_id does not match annotation video_id")
        if alert.strategy != strategy:
            raise ValueError("Alert strategy does not match evaluation strategy")
        if alert.alert_id in seen_alert_ids:
            raise ValueError(f"Duplicate alert_id: {alert.alert_id}")
        seen_alert_ids.add(alert.alert_id)

    eligible_events, excluded_intervals, _excluded_event_ids = prepare_ground_truth(labels)
    eligible_event_ids = {event.event_id for event in eligible_events}
    opportunities = tuple(
        sorted(
            (
                _Opportunity(event, opportunity)
                for event in eligible_events
                for opportunity in event.alert_opportunities
            ),
            key=lambda item: (
                item.opportunity.window_start_ms,
                item.opportunity.deadline_ms,
                item.event.event_id,
                item.opportunity.opportunity_id,
            ),
        )
    )
    ordered_alerts = tuple(sorted(enumerate(alerts), key=_stable_alert_key))
    valid_alerts: list[tuple[int, StrategyAlert]] = []
    ignored_alerts: list[tuple[int, StrategyAlert, str]] = []
    for index, alert in ordered_alerts:
        if alert.timestamp_ms >= labels.duration_ms:
            ignored_alerts.append((index, alert, "outside_video_duration"))
        elif _interval_contains(excluded_intervals, alert.timestamp_ms):
            ignored_alerts.append((index, alert, "predeclared_excluded_interval"))
        else:
            valid_alerts.append((index, alert))

    scene_alerts = [item for item in valid_alerts if item[1].alert_type != "perception_limited"]
    perception_alerts = [
        item for item in valid_alerts if item[1].alert_type == "perception_limited"
    ]

    adjacency: list[tuple[int, ...]] = []
    ambiguous_indices: set[int] = set()
    for local_index, (_, alert) in enumerate(scene_alerts):
        edges = tuple(
            index
            for index, candidate in enumerate(opportunities)
            if _eligible_edge(alert, candidate, config)
        )
        if _ambiguous_event_identity(alert, edges, opportunities):
            ambiguous_indices.add(local_index)
            adjacency.append(())
        else:
            adjacency.append(edges)

    # Kuhn's augmenting-path algorithm gives maximum cardinality. Stable alert
    # and opportunity order make the selected maximum matching reproducible.
    opportunity_to_alert: dict[int, int] = {}

    def augment(alert_index: int, seen_opportunities: set[int]) -> bool:
        for opportunity_index in adjacency[alert_index]:
            if opportunity_index in seen_opportunities:
                continue
            seen_opportunities.add(opportunity_index)
            assigned_alert = opportunity_to_alert.get(opportunity_index)
            if assigned_alert is None or augment(assigned_alert, seen_opportunities):
                opportunity_to_alert[opportunity_index] = alert_index
                return True
        return False

    for alert_index in range(len(scene_alerts)):
        augment(alert_index, set())
    alert_to_opportunity = {
        alert_index: opportunity_index
        for opportunity_index, alert_index in opportunity_to_alert.items()
    }

    matched_by_event: dict[str, list[tuple[float, int, int]]] = {}
    for alert_index, opportunity_index in alert_to_opportunity.items():
        _, alert = scene_alerts[alert_index]
        candidate = opportunities[opportunity_index]
        matched_by_event.setdefault(candidate.event.event_id, []).append(
            (alert.timestamp_ms, opportunity_index, alert_index)
        )

    records: list[MatchRecord] = []
    matched_initial_by_event: dict[str, tuple[float, float]] = {}
    duplicate_count = 0
    unsupported_count = 0

    for local_index, (_, alert) in enumerate(scene_alerts):
        opportunity_index = alert_to_opportunity.get(local_index)
        if opportunity_index is not None:
            candidate = opportunities[opportunity_index]
            opportunity = candidate.opportunity
            latency = alert.timestamp_ms - opportunity.window_start_ms
            records.append(
                MatchRecord(
                    video_id=labels.video_id,
                    strategy=strategy,
                    alert_id=alert.alert_id,
                    timestamp_ms=alert.timestamp_ms,
                    status="matched",
                    gt_event_id=candidate.event.event_id,
                    opportunity_id=opportunity.opportunity_id,
                    opportunity_kind=opportunity.kind,
                    latency_ms=latency,
                    reason="matched_independent_opportunity",
                )
            )
            if opportunity.kind == "initial":
                matched_initial_by_event[candidate.event.event_id] = (
                    alert.timestamp_ms,
                    opportunity.window_start_ms,
                )
            continue

        if local_index in ambiguous_indices:
            unsupported_count += 1
            records.append(
                MatchRecord(
                    video_id=labels.video_id,
                    strategy=strategy,
                    alert_id=alert.alert_id,
                    timestamp_ms=alert.timestamp_ms,
                    status="false_positive",
                    gt_event_id=None,
                    opportunity_id=None,
                    opportunity_kind=None,
                    latency_ms=None,
                    reason="ambiguous_ground_truth_identity_without_spatial_regions",
                )
            )
            continue

        identities = _identity_candidates(alert, eligible_events, config)
        ambiguous_identity = len(identities) > 1 and any(
            first.start_ms <= alert.timestamp_ms <= first.end_ms
            and second.start_ms <= alert.timestamp_ms <= second.end_ms
            and bool(set(first.class_ids).intersection(second.class_ids))
            and (
                not _active_regions(first, alert.timestamp_ms)
                or not _active_regions(second, alert.timestamp_ms)
            )
            for index, first in enumerate(identities)
            for second in identities[index + 1 :]
        )
        duplicate_event: LabeledEvent | None = None
        if not ambiguous_identity and len(identities) == 1:
            event = identities[0]
            prior_matches = [
                item
                for item in matched_by_event.get(event.event_id, ())
                if item[0] <= alert.timestamp_ms
            ]
            if prior_matches:
                # A maximum matching cannot leave this alert adjacent to a free
                # opportunity. A consumed opportunity therefore cannot justify
                # another true positive for the same event.
                duplicate_event = event

        if duplicate_event is not None:
            duplicate_count += 1
            reason = (
                "unmatched_same_event_alert_without_new_opportunity"
                if alert.alert_type != "escalation"
                else "unjustified_or_repeated_escalation"
            )
            records.append(
                MatchRecord(
                    video_id=labels.video_id,
                    strategy=strategy,
                    alert_id=alert.alert_id,
                    timestamp_ms=alert.timestamp_ms,
                    status="duplicate",
                    gt_event_id=duplicate_event.event_id,
                    opportunity_id=None,
                    opportunity_kind=None,
                    latency_ms=None,
                    reason=reason,
                )
            )
        else:
            unsupported_count += 1
            reason = (
                "ambiguous_ground_truth_identity_without_spatial_regions"
                if ambiguous_identity
                else "no_eligible_ground_truth_opportunity"
            )
            records.append(
                MatchRecord(
                    video_id=labels.video_id,
                    strategy=strategy,
                    alert_id=alert.alert_id,
                    timestamp_ms=alert.timestamp_ms,
                    status="false_positive",
                    gt_event_id=identities[0].event_id if len(identities) == 1 else None,
                    opportunity_id=None,
                    opportunity_kind=None,
                    latency_ms=None,
                    reason=reason,
                )
            )

    for _, alert in perception_alerts:
        records.append(
            MatchRecord(
                video_id=labels.video_id,
                strategy=strategy,
                alert_id=alert.alert_id,
                timestamp_ms=alert.timestamp_ms,
                status="perception_limited",
                gt_event_id=None,
                opportunity_id=None,
                opportunity_kind=None,
                latency_ms=None,
                reason="diagnostic_alert_not_scored_as_scene_false_positive",
            )
        )
    for _, alert, reason in ignored_alerts:
        records.append(
            MatchRecord(
                video_id=labels.video_id,
                strategy=strategy,
                alert_id=alert.alert_id,
                timestamp_ms=alert.timestamp_ms,
                status="ignored",
                gt_event_id=None,
                opportunity_id=None,
                opportunity_kind=None,
                latency_ms=None,
                reason=reason,
            )
        )

    matched_opportunities = set(opportunity_to_alert)
    for opportunity_index, candidate in enumerate(opportunities):
        if opportunity_index in matched_opportunities:
            continue
        records.append(
            MatchRecord(
                video_id=labels.video_id,
                strategy=strategy,
                alert_id=None,
                timestamp_ms=None,
                status="missed_opportunity",
                gt_event_id=candidate.event.event_id,
                opportunity_id=candidate.opportunity.opportunity_id,
                opportunity_kind=candidate.opportunity.kind,
                latency_ms=None,
                reason="no_alert_matched_within_labelled_window",
            )
        )

    important_events = [
        event for event in eligible_events if event.severity.value in config.important_severities
    ]
    event_rows: list[dict] = []
    detected_events = 0
    response_delays: list[float] = []
    matched_initial_count = len(matched_initial_by_event)
    on_time_count = 0
    for event in sorted(labels.events, key=lambda item: (item.start_ms, item.event_id)):
        important = event.severity.value in config.important_severities
        eligible = event.event_id in eligible_event_ids
        initial = next(
            (
                opportunity
                for opportunity in event.alert_opportunities
                if opportunity.kind == "initial"
            ),
            None,
        )
        matched_initial = matched_initial_by_event.get(event.event_id) if eligible else None
        detected = matched_initial is not None
        if important and eligible and detected:
            detected_events += 1
        initial_latency = None
        if matched_initial is not None:
            alert_time, window_start = matched_initial
            initial_latency = alert_time - window_start
            if important:
                response_delays.append(initial_latency)
            if important and initial is not None and alert_time <= initial.deadline_ms:
                on_time_count += 1
        event_rows.append(
            {
                "event_id": event.event_id,
                "event_type": event.event_type,
                "annotation_status": event.annotation_status,
                "exclusion_reason": None if eligible else "uncertain_excluded_or_overlap_censored",
                "severity": event.severity.value,
                "important": important,
                "eligible": eligible,
                "detected": detected,
                "initial_latency_ms": initial_latency,
            }
        )

    excluded_duration_ms = sum(end - start for start, end in excluded_intervals)
    effective_duration_ms = max(0.0, labels.duration_ms - excluded_duration_ms)
    opportunity_count = len(opportunities)
    tp = len(matched_opportunities)
    fp = duplicate_count + unsupported_count
    metrics = {
        "tp": tp,
        "fp": fp,
        "fn": opportunity_count - tp,
        "duplicates": duplicate_count,
        "unsupported": unsupported_count,
        "ignored_alerts": len(ignored_alerts),
        "perception_alerts": len(perception_alerts),
        "scene_alerts": len(scene_alerts),
        "total_alerts": len(valid_alerts),
        "important_events": len(important_events),
        "labeled_events": len(labels.events),
        "eligible_events": len(eligible_events),
        "excluded_events": len(labels.events) - len(eligible_events),
        "detected_events": detected_events,
        "missed_events": len(important_events) - detected_events,
        "opportunities": opportunity_count,
        "matched_initial_count": matched_initial_count,
        "response_delays_ms": response_delays,
        "effective_duration_ms": effective_duration_ms,
        "excluded_duration_ms": excluded_duration_ms,
        "event_recall": detected_events / len(important_events) if important_events else None,
        "opportunity_recall": tp / opportunity_count if opportunity_count else None,
        "invalid_alerts_per_hour": (
            fp / (effective_duration_ms / 3_600_000) if effective_duration_ms > 0 else None
        ),
        "duplicate_rate": duplicate_count / len(scene_alerts) if scene_alerts else None,
        "first_response_delay_mean_ms": (
            sum(response_delays) / len(response_delays) if response_delays else None
        ),
        "on_time_event_rate": on_time_count / len(important_events) if important_events else None,
        "total_alerts_per_minute": (
            len(valid_alerts) / (effective_duration_ms / 60_000)
            if effective_duration_ms > 0
            else None
        ),
    }
    records.sort(
        key=lambda record: (
            record.timestamp_ms is None,
            record.timestamp_ms if record.timestamp_ms is not None else 0.0,
            record.alert_id or "",
            record.opportunity_id or "",
        )
    )
    return EvaluationResult(
        video_id=labels.video_id,
        strategy=strategy,
        metrics=metrics,
        matches=tuple(records),
        event_rows=tuple(event_rows),
    )
