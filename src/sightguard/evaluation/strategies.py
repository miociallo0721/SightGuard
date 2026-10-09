"""Causal A/B/C policies over exactly one shared Phase 2 assessment stream."""

from dataclasses import dataclass, replace

from sightguard.data import FrameRecord
from sightguard.decision.alerts import AlertScheduler
from sightguard.decision.health import PerceptionMonitor
from sightguard.evaluation.config import BenchmarkConfig
from sightguard.evaluation.schema import StrategyAlert
from sightguard.risk.config import RiskConfig
from sightguard.risk.engine import AssessmentEngine
from sightguard.risk.schema import AlertEvent, EventChange, RiskFrame, RiskLevel, RiskObservation


@dataclass(frozen=True)
class StrategyRun:
    alerts: dict[str, tuple[StrategyAlert, ...]]
    risk_frames: tuple[RiskFrame, ...]
    changes: tuple[EventChange, ...]


def _wrap(video_id: str, strategy: str, alert: AlertEvent, frame: RiskFrame) -> StrategyAlert:
    observation = next((row for row in frame.observations if row.event_id == alert.event_id), None)
    return StrategyAlert(
        video_id,
        strategy,
        alert.alert_id,
        alert.event_id,
        alert.timestamp_ms,
        alert.risk_level,
        alert.risk_score,
        alert.alert_type,
        alert.reason,
        alert.source_track_ids,
        observation.class_id if observation else None,
        observation.features.spatial.bbox_normalized if observation else None,
        frame.frame_index,
    )


def _baseline_alert(
    video_id: str,
    strategy: str,
    ordinal: int,
    row: RiskObservation,
    frame: RiskFrame,
    level: RiskLevel,
    score: float | None,
) -> StrategyAlert:
    identity = (
        row.event_id if strategy == "B" else f"D{frame.frame_index:06d}-{row.detection_index}"
    )
    return StrategyAlert(
        video_id,
        strategy,
        f"A{ordinal:06d}",
        identity,
        frame.timestamp_ms,
        level,
        score,
        "trigger",
        "fixed class and image focus"
        if strategy == "A"
        else "fixed risk threshold and global cooldown",
        () if row.source_track_id is None else (row.source_track_id,),
        row.class_id,
        row.features.spatial.bbox_normalized,
        frame.frame_index,
    )


def run_strategies(
    records: tuple[FrameRecord, ...],
    video_id: str,
    risk_config: RiskConfig,
    config: BenchmarkConfig,
    strategies: tuple[str, ...] = ("A", "B", "C"),
) -> StrategyRun:
    if (
        not strategies
        or len(set(strategies)) != len(strategies)
        or set(strategies) - {"A", "B", "C"}
    ):
        raise ValueError("Choose unique strategy names from A, B, C")
    strategies = tuple(sorted(strategies))
    engine = AssessmentEngine(risk_config, stream_id=video_id)
    output: dict[str, list[StrategyAlert]] = {name: [] for name in strategies}
    monitors = {
        name: PerceptionMonitor(risk_config.alerts, video_id) for name in strategies if name != "C"
    }
    health_schedulers = {name: AlertScheduler(risk_config.alerts, video_id) for name in monitors}
    last_b_alert_ms = None
    risk_frames, changes = [], []
    for record in records:
        step = engine.step(record)
        frame = step.frame
        risk_frames.append(frame)
        changes.extend(step.changes)
        if "C" in output:
            output["C"].extend(_wrap(video_id, "C", alert, frame) for alert in step.alerts)
        for name in monitors:
            monitor = monitors[name]
            monitor.update(frame.timestamp_ms, frame.perception.status, frame.perception.reason)
            health = health_schedulers[name].schedule((), monitor, frame.timestamp_ms)
            for alert in health:
                wrapped = _wrap(video_id, name, alert, frame)
                # Baseline IDs cover health and scene alerts in one ordered namespace.
                output[name].append(replace(wrapped, alert_id=f"A{len(output[name]) + 1:06d}"))
        if record.status != "ok":
            continue
        valid = [
            row
            for row in frame.observations
            if row.features is not None and row.assessment is not None
        ]
        if "A" in output:
            for row in sorted(valid, key=lambda observation: observation.detection_index):
                if (
                    row.class_id in config.baseline_classes
                    and row.features.spatial.focus_relevance >= config.baseline_focus_threshold
                ):
                    output["A"].append(
                        _baseline_alert(
                            video_id,
                            "A",
                            len(output["A"]) + 1,
                            row,
                            frame,
                            RiskLevel(config.baseline_severity),
                            None,
                        )
                    )
        if "B" in output and (
            last_b_alert_ms is None
            or frame.timestamp_ms - last_b_alert_ms >= config.fixed_cooldown_ms
        ):
            eligible = sorted(
                (row for row in valid if row.assessment.score >= config.fixed_risk_threshold),
                key=lambda row: (
                    -row.assessment.level.rank,
                    -row.assessment.spatial_score,
                    -row.assessment.score,
                    row.event_id,
                ),
            )
            if eligible:
                row = eligible[0]
                output["B"].append(
                    _baseline_alert(
                        video_id,
                        "B",
                        len(output["B"]) + 1,
                        row,
                        frame,
                        row.assessment.level,
                        row.assessment.score,
                    )
                )
                last_b_alert_ms = frame.timestamp_ms
    changes.extend(engine.finish())
    return StrategyRun(
        {name: tuple(alerts) for name, alerts in output.items()}, tuple(risk_frames), tuple(changes)
    )
