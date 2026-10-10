"""Latest-frame capture and single-owner inference/assessment threads."""

from __future__ import annotations

import logging
import sys
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from threading import Condition, Event, Lock, Thread
from time import monotonic
from typing import TYPE_CHECKING, Protocol
from uuid import uuid4

import cv2
import numpy as np

from sightguard.config import PerceptionConfig
from sightguard.data import Detection, FrameRecord
from sightguard.live.config import LiveConfig
from sightguard.perception.yolo import YoloPerceptor
from sightguard.risk.config import RiskConfig
from sightguard.risk.engine import AssessmentEngine
from sightguard.risk.schema import AlertEvent, StepResult

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sightguard.spatial.config import SpatialConfig


def open_camera(index: int) -> Camera:
    """Windows uses DirectShow indices, matching the device-list ordering."""
    return cv2.VideoCapture(index, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)


class Camera(Protocol):
    def isOpened(self) -> bool: ...  # noqa: N802
    def set(self, prop: int, value: float) -> bool: ...
    def read(self) -> tuple[bool, np.ndarray | None]: ...
    def release(self) -> None: ...


class Perceptor(Protocol):
    def process(self, frame: np.ndarray) -> tuple[Detection, ...]: ...


@dataclass(frozen=True)
class CapturedFrame:
    capture_index: int
    timestamp_ms: float
    captured_at: float
    image: np.ndarray


class LatestFrame:
    """Capacity one. Publishing replaces unconsumed frames, never queues them."""

    def __init__(self) -> None:
        self.condition = Condition()
        self.frame: CapturedFrame | None = None

    def publish(self, frame: CapturedFrame) -> None:
        with self.condition:
            self.frame = frame
            self.condition.notify_all()

    def take_after(self, index: int, timeout: float = 0.05) -> CapturedFrame | None:
        with self.condition:
            self.condition.wait_for(
                lambda: self.frame is not None and self.frame.capture_index > index, timeout
            )
            if self.frame is not None and self.frame.capture_index > index:
                return self.frame
            return None

    def peek(self) -> CapturedFrame | None:
        with self.condition:
            return self.frame


@dataclass(frozen=True)
class LiveResult:
    captured: CapturedFrame | None
    record: FrameRecord
    decision: StepResult
    processing_ms: float
    completed_at: float
    update_fps: float


class LiveSession:
    """Fresh model/tracker and AssessmentEngine per session; engine owned by worker only."""

    def __init__(
        self,
        perception: PerceptionConfig,
        risk: RiskConfig,
        config: LiveConfig,
        *,
        camera_factory: Callable[[int], Camera] = open_camera,
        perceptor_factory: Callable[[PerceptionConfig], Perceptor] = YoloPerceptor,
        clock: Callable[[], float] = monotonic,
        spatial_config: SpatialConfig | None = None,
        audio_factory: Callable | None = None,
    ) -> None:
        self.perception, self.risk, self.config = perception, risk, config
        self.camera_factory, self.perceptor_factory, self.clock = (
            camera_factory,
            perceptor_factory,
            clock,
        )
        self.engine = AssessmentEngine(risk, stream_id=f"live-{uuid4().hex}")
        self.frames = LatestFrame()
        self.stop_event = Event()
        self.lock = Lock()
        self.result: LiveResult | None = None
        self.recent_alerts: deque[tuple[AlertEvent, float]] = deque(maxlen=16)
        self.camera: Camera | None = None
        self.error: str | None = None
        self.failed_at: float | None = None
        self.origin = 0.0
        self.started = False
        self.threads: tuple[Thread, ...] = ()
        self.spatial_config = spatial_config
        self.audio_factory = audio_factory
        self.audio = None
        self.audio_adapter = None
        self.audio_error: str | None = None

    def start(self) -> None:
        if self.started:
            raise RuntimeError("LiveSession cannot be reused; create a new session")
        self.started = True
        self.origin = self.clock()
        if self.spatial_config is not None:
            try:
                from sightguard.spatial.risk_adapter import RiskCueAdapter
                from sightguard.spatial.worker import AudioWorker

                factory = self.audio_factory or AudioWorker
                self.audio = factory(
                    self.spatial_config, clock_ms=lambda: (self.clock() - self.origin) * 1000
                )
                self.audio_adapter = RiskCueAdapter(self.spatial_config)
                self.audio.start()  # Native device opening stays off the inference thread.
            except Exception as exc:
                self.audio_error = f"Spatial audio unavailable: {exc}"
                LOGGER.warning("%s; vision continues", self.audio_error)
        self.threads = (
            Thread(target=self._capture, name="sightguard-camera", daemon=True),
            Thread(target=self._infer, name="sightguard-inference", daemon=True),
        )
        for thread in self.threads:
            thread.start()

    def _fail(self, message: str) -> None:
        with self.lock:
            if self.error is None:
                self.error, self.failed_at = message, self.clock()
                LOGGER.error("%s", message)

    def failure(self) -> tuple[str | None, float | None]:
        with self.lock:
            return self.error, self.failed_at

    def snapshot(self) -> tuple[LiveResult | None, tuple[tuple[AlertEvent, float], ...]]:
        with self.lock:
            return self.result, tuple(self.recent_alerts)

    def _capture(self) -> None:
        camera = None
        try:
            camera = self.camera_factory(self.config.camera)
            self.camera = camera
            if not camera.isOpened():
                raise RuntimeError(f"Cannot open camera {self.config.camera}")
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            # Negotiate size and FPS first, codec last. DirectShow may reset the codec
            # when changing dimensions; Xiaomi's default YUY2 mode produces corrupt frames.
            fps_set = camera.set(cv2.CAP_PROP_FPS, self.config.camera_fps)
            codec_set = camera.set(
                cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.config.camera_fourcc)
            )
            LOGGER.info(
                "Camera requested fps=%s codec=%s; accepted fps=%s codec=%s",
                self.config.camera_fps,
                self.config.camera_fourcc,
                fps_set,
                codec_set,
            )
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Best effort; not supported by every driver.
            index = 0
            while not self.stop_event.is_set():
                ok, image = camera.read()
                now = self.clock()
                if not ok or image is None or image.size == 0:
                    raise RuntimeError("Camera disconnected or frame read failed")
                if index == 0:
                    LOGGER.info("Camera first frame: %dx%d", image.shape[1], image.shape[0])
                self.frames.publish(
                    CapturedFrame(index, (now - self.origin) * 1000, now, image.copy())
                )
                index += 1
        except Exception as exc:
            if not self.stop_event.is_set():
                self._fail(f"{type(exc).__name__}: {exc}")
        finally:
            if camera is not None:
                camera.release()

    def _infer(self) -> None:
        index, last_capture = 0, -1
        completion_times: deque[float] = deque(maxlen=20)
        last_failure_update = -1.0
        try:
            try:
                perceptor = self.perceptor_factory(self.perception)
            except Exception as exc:
                perceptor = None
                self._fail(f"Model initialization failed: {exc}")
            while not self.stop_event.is_set():
                packet = self.frames.take_after(last_capture)
                now = self.clock()
                failure, _ = self.failure()
                if packet is None:
                    packet = self.frames.peek()
                    age_ms = (now - (packet.captured_at if packet else self.origin)) * 1000
                    if not failure and age_ms <= self.config.stale_after_ms:
                        continue
                    failure = failure or "No fresh camera frame: perception stale"
                elif (now - packet.captured_at) * 1000 > self.config.stale_after_ms:
                    failure = failure or "Camera frame expired before inference"
                if failure and now - last_failure_update < 0.1:
                    self.stop_event.wait(0.02)
                    continue
                start = self.clock()
                detections: tuple[Detection, ...] = ()
                if not failure and packet is not None and perceptor is not None:
                    try:
                        detections = perceptor.process(packet.image)
                        if (self.clock() - packet.captured_at) * 1000 > self.config.stale_after_ms:
                            failure = "Inference result expired: perception stale"
                            detections = ()
                    except Exception as exc:
                        failure = f"Inference failed: {exc}"
                        self._fail(failure)  # Do not reuse potentially corrupted tracker state.
                if packet is not None:
                    last_capture = packet.capture_index
                    height, width = packet.image.shape[:2]
                else:
                    width, height = self.config.width, self.config.height
                timestamp = (self.clock() - self.origin) * 1000 if failure else packet.timestamp_ms
                # Failure heartbeats use current time. Reject a later-arriving older sample.
                if (
                    self.engine.last_timestamp is not None
                    and timestamp < self.engine.last_timestamp
                ):
                    failure, detections = "Camera sample older than last health update", ()
                    timestamp = (self.clock() - self.origin) * 1000
                record = FrameRecord(
                    index,
                    timestamp,
                    width,
                    height,
                    detections,
                    status="processing_error" if failure else "ok",
                    error=failure,
                    timestamp_source="monotonic",
                )
                decision = self.engine.step(record)
                completed = self.clock()
                if failure:
                    last_failure_update = completed
                else:
                    completion_times.append(completed)
                fps = (
                    (len(completion_times) - 1) / (completion_times[-1] - completion_times[0])
                    if len(completion_times) > 1 and completion_times[-1] > completion_times[0]
                    else 0.0
                )
                result = LiveResult(
                    packet, record, decision, (completed - start) * 1000, completed, fps
                )
                with self.lock:
                    self.result = result
                    self.recent_alerts.extend((alert, completed) for alert in decision.alerts)
                if self.audio is not None and self.audio_adapter is not None:
                    try:
                        for cue in self.audio_adapter.convert(
                            record, decision, (completed - self.origin) * 1000
                        ):
                            self.audio.submit(cue)
                    except Exception as exc:
                        self.audio_error = f"Spatial cue submission failed: {exc}"
                        LOGGER.warning("%s; vision continues", self.audio_error)
                for alert in decision.alerts:
                    LOGGER.info("Alert %s %s: %s", alert.alert_type, alert.event_id, alert.reason)
                index += 1
        except Exception as exc:
            self._fail(f"Live processing stopped: {exc}")
        finally:
            self.engine.finish()

    def close(self) -> None:
        self.stop_event.set()
        if self.audio is not None:
            try:
                self.audio.close()
            except Exception as exc:
                self.audio_error = f"Audio cleanup failed: {exc}"
                LOGGER.warning("%s", self.audio_error)
        with self.frames.condition:
            self.frames.condition.notify_all()
        for thread in self.threads:
            thread.join(self.config.join_timeout_s)
        # Most drivers return from read normally; release helps unblock stalled backends.
        if any(thread.is_alive() for thread in self.threads) and self.camera is not None:
            self.camera.release()
            for thread in self.threads:
                thread.join(self.config.join_timeout_s)
        if any(thread.is_alive() for thread in self.threads):
            raise RuntimeError("Camera/inference backend did not stop within shutdown timeout")
