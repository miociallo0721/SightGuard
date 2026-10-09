"""Aggregation of per-video evaluation counts into corpus-level metrics."""

from __future__ import annotations

from collections.abc import Iterable

from sightguard.evaluation.schema import EvaluationResult

_COUNT_FIELDS = (
    "tp",
    "fp",
    "fn",
    "duplicates",
    "unsupported",
    "ignored_alerts",
    "perception_alerts",
    "scene_alerts",
    "total_alerts",
    "important_events",
    "labeled_events",
    "eligible_events",
    "excluded_events",
    "detected_events",
    "missed_events",
    "opportunities",
    "matched_initial_count",
    "effective_duration_ms",
    "excluded_duration_ms",
)


def aggregate_metrics(results: Iterable[EvaluationResult]) -> dict:
    """Combine per-video counts, then compute rates from pooled denominators.

    The returned rates are corpus-level ratios rather than an unweighted mean of
    video-level rates. Zero denominators are reported as ``None``.
    """
    items = tuple(results)
    totals = {
        field: sum(result.metrics.get(field, 0) for result in items) for field in _COUNT_FIELDS
    }
    delays = [delay for result in items for delay in result.metrics.get("response_delays_ms", ())]
    effective_duration_ms = totals["effective_duration_ms"]
    important_events = totals["important_events"]
    opportunities = totals["opportunities"]
    scene_alerts = totals["scene_alerts"]
    total_alerts = totals["total_alerts"]
    detected_events = totals["detected_events"]
    fp = totals["fp"]
    duplicates = totals["duplicates"]
    tp = totals["tp"]

    return {
        **totals,
        "response_delays_ms": delays,
        "event_recall": detected_events / important_events if important_events else None,
        "opportunity_recall": tp / opportunities if opportunities else None,
        "invalid_alerts_per_hour": (
            fp / (effective_duration_ms / 3_600_000) if effective_duration_ms > 0 else None
        ),
        "duplicate_rate": duplicates / scene_alerts if scene_alerts else None,
        "first_response_delay_mean_ms": (sum(delays) / len(delays) if delays else None),
        "on_time_event_rate": detected_events / important_events if important_events else None,
        "total_alerts_per_minute": (
            total_alerts / (effective_duration_ms / 60_000) if effective_duration_ms > 0 else None
        ),
    }
