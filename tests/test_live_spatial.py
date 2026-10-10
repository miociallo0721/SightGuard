from __future__ import annotations

from collections.abc import Callable
from queue import Empty, Queue
from threading import Event, Lock
from time import monotonic

import cv2
import numpy as np

from sightguard.config import PerceptionConfig
from sightguard.data import Detection, FrameRecord
from sightguard.live.config import LiveConfig
from sightguard.live.runtime import LiveSession
from sightguard.risk.config import RiskConfig
from sightguard.spatial.config import SpatialConfig
from sightguard.spatial.worker import AudioWorker


class MockCamera:
    def __init__(self) -> None:
        self.frames: Queue[np.ndarray] = Queue()
        self.released = Event()
        self.configured = Event()
        self.set_calls: list[tuple[int, float]] = []

    def isOpened(self) -> bool:  # noqa: N802
        return True

    def set(self, _prop: int, _value: float) -> bool:
        self.set_calls.append((_prop, _value))
        if len(self.set_calls) == 5:
            self.configured.set()
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        while not self.released.is_set():
            try:
                return True, self.frames.get(timeout=0.01)
            except Empty:
                continue
        return False, None

    def release(self) -> None:
        self.released.set()

    def push(self, track_id: int) -> None:
        self.frames.put(np.full((480, 640, 3), track_id, dtype=np.uint8))


class MarkerPerceptor:
    def process(self, image: np.ndarray) -> tuple[Detection, ...]:
        track_id = int(image[0, 0, 0])
        return (Detection(track_id, 0, "person", 0.9, (280, 300, 360, 450)),)


class ControlledClock:
    def __init__(self) -> None:
        self._lock = Lock()
        self._seconds = 0.0

    def __call__(self) -> float:
        with self._lock:
            return self._seconds

    def advance_ms(self, amount: float) -> None:
        with self._lock:
            self._seconds += amount / 1000


class CollectingAudioWorker:
    def __init__(self, _config: SpatialConfig, *, clock_ms: Callable[[], float]) -> None:
        self.clock_ms = clock_ms
        self.started = Event()
        self.submitted = Event()
        self.closed = Event()
        self._lock = Lock()
        self.cues = []

    def start(self) -> None:
        self.started.set()

    def submit(self, cue) -> bool:
        with self._lock:
            self.cues.append(cue)
        self.submitted.set()
        return True

    def close(self) -> None:
        self.closed.set()


def wait_for(predicate: Callable[[], object], timeout: float = 2.0):
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        result = predicate()
        if result:
            return result
        Event().wait(0.005)
    raise AssertionError("condition was not met before the deadline")


def result_at(session: LiveSession, frame_index: int):
    return wait_for(
        lambda: (
            result
            if (result := session.snapshot()[0]) is not None
            and result.record.frame_index >= frame_index
            else None
        )
    )


def make_session(
    camera: MockCamera,
    clock: ControlledClock,
    audio_factory: Callable | None,
) -> LiveSession:
    return LiveSession(
        PerceptionConfig(),
        RiskConfig(),
        LiveConfig(stale_after_ms=5_000, join_timeout_s=0.05),
        camera_factory=lambda _index: camera,
        perceptor_factory=lambda _config: MarkerPerceptor(),
        clock=clock,
        spatial_config=SpatialConfig(),
        audio_factory=audio_factory,
    )


def produce_confirmed_live_alert(
    session: LiveSession,
    camera: MockCamera,
    clock: ControlledClock,
    track_id: int = 7,
) -> tuple[FrameRecord, FrameRecord]:
    camera.push(track_id)
    first = result_at(session, 0)
    assert first.record.timestamp_ms == 0
    clock.advance_ms(300)
    camera.push(track_id)
    second = result_at(session, 1)
    assert second.record.timestamp_ms == 300
    assert [alert.alert_type for alert in second.decision.alerts] == ["first"]
    return first.record, second.record


def test_live_camera_negotiates_size_fps_codec_then_buffer_size() -> None:
    camera = MockCamera()
    session = LiveSession(
        PerceptionConfig(),
        RiskConfig(),
        LiveConfig(
            width=800,
            height=600,
            camera_fps=30,
            camera_fourcc="MJPG",
            stale_after_ms=5_000,
            join_timeout_s=0.05,
        ),
        camera_factory=lambda _index: camera,
        perceptor_factory=lambda _config: MarkerPerceptor(),
    )
    session.start()
    try:
        assert camera.configured.wait(1)
        assert [prop for prop, _value in camera.set_calls] == [
            cv2.CAP_PROP_FRAME_WIDTH,
            cv2.CAP_PROP_FRAME_HEIGHT,
            cv2.CAP_PROP_FPS,
            cv2.CAP_PROP_FOURCC,
            cv2.CAP_PROP_BUFFERSIZE,
        ]
        assert camera.set_calls == [
            (cv2.CAP_PROP_FRAME_WIDTH, 800),
            (cv2.CAP_PROP_FRAME_HEIGHT, 600),
            (cv2.CAP_PROP_FPS, 30),
            (cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")),
            (cv2.CAP_PROP_BUFFERSIZE, 1),
        ]
    finally:
        session.close()

    assert all(not thread.is_alive() for thread in session.threads)


def test_live_session_sends_real_engine_alert_to_spatial_worker_and_closes_threads() -> None:
    camera, clock = MockCamera(), ControlledClock()
    workers: list[CollectingAudioWorker] = []

    def audio_factory(config: SpatialConfig, *, clock_ms: Callable[[], float]):
        worker = CollectingAudioWorker(config, clock_ms=clock_ms)
        workers.append(worker)
        return worker

    session = make_session(camera, clock, audio_factory)
    session.start()
    try:
        assert len(workers) == 1
        assert workers[0].started.wait(1)
        _first, confirmed_record = produce_confirmed_live_alert(session, camera, clock)
        assert workers[0].submitted.wait(1)
        result, recent_alerts = session.snapshot()
        assert result is not None and result.record == confirmed_record
        assert len(recent_alerts) == 1
        assert recent_alerts[0][0] == result.decision.alerts[0]
        assert len(workers[0].cues) == 1
        cue = workers[0].cues[0]
        assert cue.cue_id.startswith(f"risk-alert:{len(result.decision.frame.stream_id)}:")
        assert cue.event_id == result.decision.alerts[0].event_id == "E000001"
        assert cue.timestamp_ms == confirmed_record.timestamp_ms == 300
        assert cue.cue_type == "risk_initial"
        assert session.failure() == (None, None)
    finally:
        session.close()

    assert workers[0].closed.is_set()
    assert all(not thread.is_alive() for thread in session.threads)


def test_live_sessions_keep_assessment_adapters_and_spatial_cues_isolated() -> None:
    left_camera, right_camera = MockCamera(), MockCamera()
    left_clock, right_clock = ControlledClock(), ControlledClock()
    workers: list[CollectingAudioWorker] = []

    def audio_factory(config: SpatialConfig, *, clock_ms: Callable[[], float]):
        worker = CollectingAudioWorker(config, clock_ms=clock_ms)
        workers.append(worker)
        return worker

    left = make_session(left_camera, left_clock, audio_factory)
    right = make_session(right_camera, right_clock, audio_factory)
    left.start()
    right.start()
    try:
        assert len(workers) == 2
        produce_confirmed_live_alert(left, left_camera, left_clock, track_id=7)
        produce_confirmed_live_alert(right, right_camera, right_clock, track_id=7)
        assert workers[0].submitted.wait(1)
        assert workers[1].submitted.wait(1)

        left_result = result_at(left, 1)
        right_result = result_at(right, 1)
        assert left.engine is not right.engine
        assert left.audio_adapter is not right.audio_adapter
        assert left.audio is not right.audio
        assert left_result.decision.frame.stream_id != right_result.decision.frame.stream_id
        assert left_result.decision.alerts[0].alert_id == right_result.decision.alerts[0].alert_id
        assert left_result.decision.alerts[0].event_id == right_result.decision.alerts[0].event_id
        assert len(workers[0].cues) == len(workers[1].cues) == 1
        assert workers[0].cues[0].cue_id != workers[1].cues[0].cue_id

        left.close()
        assert workers[0].closed.is_set()
        assert not workers[1].closed.is_set()
        assert all(thread.is_alive() for thread in right.threads)
    finally:
        left.close()
        right.close()

    assert all(not thread.is_alive() for thread in left.threads + right.threads)
    assert workers[1].closed.is_set()


def test_audio_backend_failure_does_not_stop_live_assessment_and_threads_exit() -> None:
    class FailingBackend:
        def __init__(self) -> None:
            self.opened = Event()
            self.closed = Event()

        def open(self):
            self.opened.set()
            raise RuntimeError("mock audio device unavailable")

        def close(self) -> None:
            self.closed.set()

    camera, clock = MockCamera(), ControlledClock()
    backend = FailingBackend()
    audio_workers: list[AudioWorker] = []

    def audio_factory(config: SpatialConfig, *, clock_ms: Callable[[], float]) -> AudioWorker:
        worker = AudioWorker(
            config,
            backend_factory=lambda: backend,
            clock_ms=clock_ms,
        )
        audio_workers.append(worker)
        return worker

    session = make_session(camera, clock, audio_factory)
    session.start()
    try:
        assert len(audio_workers) == 1
        assert backend.opened.wait(1)
        _first, confirmed_record = produce_confirmed_live_alert(session, camera, clock)
        assert backend.closed.wait(1)
        result, recent_alerts = session.snapshot()
        assert result is not None and result.record == confirmed_record
        assert [alert.alert_type for alert, _created_at in recent_alerts] == ["first"]
        assert session.failure() == (None, None)
        assert "mock audio device unavailable" in audio_workers[0].snapshot()["error"]
        assert audio_workers[0].thread is not None
        audio_workers[0].thread.join(1)
        assert not audio_workers[0].thread.is_alive()
    finally:
        session.close()

    assert all(not thread.is_alive() for thread in session.threads)


def test_camera_negotiates_size_and_fps_before_mjpg_codec() -> None:
    camera, clock = MockCamera(), ControlledClock()
    session = make_session(camera, clock, None)
    session.start()
    try:
        assert camera.configured.wait(2)
        assert camera.set_calls == [
            (cv2.CAP_PROP_FRAME_WIDTH, 640),
            (cv2.CAP_PROP_FRAME_HEIGHT, 480),
            (cv2.CAP_PROP_FPS, 30),
            (cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")),
            (cv2.CAP_PROP_BUFFERSIZE, 1),
        ]
    finally:
        camera.release()
        session.close()
    assert all(not thread.is_alive() for thread in session.threads)
