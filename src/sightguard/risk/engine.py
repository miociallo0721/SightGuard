"""One causal assessment session per video; core requires no video or ML library."""

from sightguard.data import FrameRecord
from sightguard.decision.alerts import AlertScheduler
from sightguard.decision.events import EventManager
from sightguard.decision.health import PerceptionMonitor
from sightguard.risk.config import RiskConfig
from sightguard.risk.features import FeatureExtractor, spatial_features
from sightguard.risk.schema import EventChange, RiskFrame, RiskObservation, StepResult
from sightguard.risk.scoring import RiskScorer


class AssessmentEngine:
    """All mutable state belongs to this stream. No future frames are read."""

    def __init__(self, config: RiskConfig | None = None, stream_id: str = "video") -> None:
        self.config = config or RiskConfig()
        self.stream_id = stream_id
        self.features = FeatureExtractor(self.config.features)
        self.scorer = RiskScorer(self.config.scoring)
        self.manager = EventManager(self.config.association, stream_id)
        self.monitor = PerceptionMonitor(self.config.alerts, stream_id)
        self.scheduler = AlertScheduler(self.config.alerts, stream_id)
        self.last_index = -1
        self.last_timestamp: float | None = None
        self.finished = False

    def step(self, frame: FrameRecord) -> StepResult:
        if self.finished:
            raise ValueError("Assessment stream is finished; create a new engine for another video")
        if frame.frame_index != self.last_index + 1:
            raise ValueError("Assessment frames must start at zero and be contiguous")
        if self.last_timestamp is not None and frame.timestamp_ms < self.last_timestamp:
            raise ValueError("Assessment timestamps must not decrease")
        warnings = []
        if self.last_timestamp is not None:
            interval = frame.timestamp_ms - self.last_timestamp
            if interval == 0:
                warnings.append("duplicate_timestamp_motion_unavailable")
            elif interval > self.config.alerts.max_frame_gap_ms:
                warnings.append("frame_gap_exceeds_limit")
        geometry, invalid = {}, {}
        for index, detection in enumerate(frame.detections):
            try:
                geometry[index] = spatial_features(
                    detection, frame.frame_width, frame.frame_height, self.config.features
                )
            except ValueError as exc:
                invalid[index] = str(exc)
                warnings.append(f"detection_{index}: {exc}")
        complete = frame.status == "ok" and not invalid
        changes = list(self.manager.prepare(frame.timestamp_ms, complete))
        for change in changes:
            self.features.forget(change.event_id)
            self.scorer.forget(change.event_id)
            self.scheduler.history.pop(change.event_id, None)
        self.features.begin_frame(frame, complete)
        observations = [
            RiskObservation(
                index,
                frame.detections[index].track_id,
                frame.detections[index].class_id,
                frame.detections[index].class_name,
                None,
                None,
                None,
                None,
                error,
            )
            for index, error in sorted(invalid.items())
        ]
        for association in self.manager.associate(frame, geometry):
            index, event = association.detection_index, association.event
            detection = frame.detections[index]
            features = self.features.observe(event.event_id, detection, frame, geometry[index])
            assessment = self.scorer.assess(event.event_id, features, frame.timestamp_ms)
            changes.extend(
                self.manager.observe(
                    association, detection, geometry[index], assessment, frame.timestamp_ms
                )
            )
            observations.append(
                RiskObservation(
                    index,
                    detection.track_id,
                    detection.class_id,
                    detection.class_name,
                    event.event_id,
                    association.method,
                    features,
                    assessment,
                )
            )
        changes.extend(self.manager.end_frame(frame.timestamp_ms, complete))
        for event in self.manager.live_events():
            if not event.observed_now:
                self.scorer.interrupt(event.event_id)
        status, reason = "available", None
        if frame.status == "processing_error":
            status, reason = "unavailable", frame.error
        elif warnings:
            status, reason = "limited", "; ".join(warnings)
        elif any(
            obs.assessment is not None
            and obs.assessment.quality.score < self.config.alerts.limited_quality_threshold
            for obs in observations
        ):
            status, reason = "limited", "one_or_more_targets_have_limited_perception_quality"
        changes.extend(self.monitor.update(frame.timestamp_ms, status, reason))
        snapshots = tuple(event.snapshot() for event in self.manager.live_events())
        alerts = self.scheduler.schedule(snapshots, self.monitor, frame.timestamp_ms)
        self.last_index, self.last_timestamp = frame.frame_index, frame.timestamp_ms
        result = RiskFrame(
            self.stream_id,
            frame.frame_index,
            frame.timestamp_ms,
            frame.timestamp_source,
            frame.status,
            self.monitor.current,
            tuple(sorted(observations, key=lambda row: row.detection_index)),
            snapshots,
            tuple(warnings),
        )
        return StepResult(result, tuple(changes), alerts)

    def finish(self) -> tuple[EventChange, ...]:
        if self.finished:
            return ()
        self.finished = True
        if self.last_timestamp is None:
            return ()
        return self.manager.finish(self.last_timestamp) + self.monitor.finish(self.last_timestamp)
