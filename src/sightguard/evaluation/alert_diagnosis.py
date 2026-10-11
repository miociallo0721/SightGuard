"""Read-only behavioral diagnostics; no Ground Truth or policy changes."""

from collections import Counter, defaultdict
from typing import Any


def diagnose_video(
    video_id: str, duration_ms: float, frames: list[dict], events: list[dict], alerts: list[dict]
) -> dict[str, Any]:
    """Join alerts to raw observations and lifecycle changes on video time and ID."""
    observations: dict[str, list[dict]] = defaultdict(list)
    changes: dict[str, list[dict]] = defaultdict(list)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for frame in frames:
        for observation in frame.get("observations", []):
            if observation.get("event_id"):
                observations[observation["event_id"]].append(
                    {
                        "timestamp_ms": frame["timestamp_ms"],
                        "frame_index": frame["frame_index"],
                        **observation,
                    }
                )
    for row in events:
        changes[row["event_id"]].append(row)
    for alert in alerts:
        grouped[alert["event_id"]].append(alert)
    details = []
    for event_id in sorted(set(observations) | set(changes) | set(grouped)):
        samples = sorted(observations[event_id], key=lambda r: r["timestamp_ms"])
        lifecycle = sorted(changes[event_id], key=lambda r: r["timestamp_ms"])
        history = sorted(grouped[event_id], key=lambda r: r["timestamp_ms"])
        transitions = []
        previous = None
        for row in lifecycle:
            if row.get("state") != previous:
                transitions.append(
                    {
                        k: row.get(k)
                        for k in ("timestamp_ms", "state", "action", "reason", "source_track_id")
                    }
                )
                previous = row.get("state")
        score_changes = []
        for left, right in zip(samples, samples[1:], strict=False):
            a, b = left.get("assessment") or {}, right.get("assessment") or {}
            if a.get("score") is None or b.get("score") is None:
                continue
            delta = b["score"] - a["score"]
            if a.get("level") != b.get("level") or abs(delta) >= 0.15:
                score_changes.append(
                    {
                        "from_ms": left["timestamp_ms"],
                        "to_ms": right["timestamp_ms"],
                        "from_score": a["score"],
                        "to_score": b["score"],
                        "from_level": a.get("level"),
                        "to_level": b.get("level"),
                        "delta": delta,
                    }
                )
        joined = []
        for alert in history:
            exact = next(
                (r for r in samples if abs(r["timestamp_ms"] - alert["timestamp_ms"]) < 0.001), None
            )
            state = next(
                (
                    r["state"]
                    for r in reversed(lifecycle)
                    if r["timestamp_ms"] <= alert["timestamp_ms"] + 0.001
                ),
                None,
            )
            joined.append(
                {
                    **alert,
                    "state_at_alert": state,
                    "observation": exact,
                    "observation_missing": exact is None,
                }
            )
        first = (
            lifecycle[0]["timestamp_ms"]
            if lifecycle
            else samples[0]["timestamp_ms"]
            if samples
            else history[0]["timestamp_ms"]
        )
        resolved = next((r["timestamp_ms"] for r in lifecycle if r["state"] == "RESOLVED"), None)
        last_seen = max((r["timestamp_ms"] for r in samples), default=first)
        tracks = sorted(
            {r["source_track_id"] for r in samples if r.get("source_track_id") is not None}
        )
        details.append(
            {
                "video_id": video_id,
                "event_id": event_id,
                "event_kind": lifecycle[0].get("event_kind", "risk") if lifecycle else "risk",
                "class_name": samples[0].get("class_name") if samples else None,
                "start_ms": first,
                "end_ms": resolved if resolved is not None else duration_ms,
                "right_censored": resolved is None,
                "last_seen_ms": last_seen,
                "observed_span_ms": last_seen - first,
                "source_track_ids": tracks,
                "alert_count": len(history),
                "alert_types": dict(Counter(a["alert_type"] for a in history)),
                "transitions": transitions,
                "score_changes": score_changes,
                "alerts": joined,
            }
        )
    track_events: dict[int, list[str]] = defaultdict(list)
    for row in details:
        for track in row["source_track_ids"]:
            track_events[track].append(row["event_id"])
    return {
        "video_id": video_id,
        "duration_ms": duration_ms,
        "alert_count": len(alerts),
        "alerts_per_minute": len(alerts) * 60000 / duration_ms,
        "alert_types": dict(Counter(a["alert_type"] for a in alerts)),
        "alert_levels": dict(Counter(str(a.get("risk_level")) for a in alerts)),
        "event_count": len(details),
        "risk_event_count": sum(e["event_kind"] == "risk" for e in details),
        "perception_event_count": sum(e["event_kind"] == "perception" for e in details),
        "alerted_event_count": sum(bool(v) for v in grouped.values()),
        "same_track_multiple_events": {str(k): v for k, v in track_events.items() if len(v) > 1},
        "events": sorted(details, key=lambda r: (-r["alert_count"], r["event_id"])),
    }
