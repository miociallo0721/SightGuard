from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from queue import Empty, Queue
from threading import Event
from time import monotonic

import numpy as np
import pytest

from sightguard.config import PerceptionConfig
from sightguard.data import Detection, FrameRecord
from sightguard.live.config import LiveConfig
from sightguard.live.runtime import CapturedFrame, LatestFrame, LiveSession
from sightguard.risk.config import RiskConfig
from sightguard.risk.engine import AssessmentEngine


class MockCamera:
    def __init__(self, *, opened: bool = True, fail_read: bool = False) -> None:
        self.opened = opened
        self.fail_read = fail_read
        self.frames: Queue[np.ndarray] = Queue()
        self.released = Event()
        self.read_count = 0

    def isOpened(self) -> bool:  # noqa: N802
        return self.opened

    def set(self, _prop: int, _value: float) -> bool:
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        self.read_count += 1
        if self.fail_read:
            return False, None
        while not self.released.is_set():
            try:
                return True, self.frames.get(timeout=0.01)
            except Empty:
                continue
        return False, None

    def release(self) -> None:
        self.released.set()

    def push(self, value: int) -> None:
        self.frames.put(np.full((480, 640, 3), value, dtype=np.uint8))


class ImageDetectionPerceptor:
    def process(self, image: np.ndarray) -> tuple[Detection, ...]:
        track_id = int(image[0, 0, 0])
        return (Detection(track_id, 0, "person", 0.9, (280, 300, 360, 450)),)


def _wait_for(predicate: Callable[[], object], timeout: float = 1.5):
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        result = predicate()
        if result:
            return result
        Event().wait(0.005)
    raise AssertionError("condition was not met before the deadline")


def _live_config(**values: object) -> LiveConfig:
    return LiveConfig(**(values | {"join_timeout_s": values.get("join_timeout_s", 0.05)}))


def _session(camera: MockCamera, perceptor: object, **config_values: object) -> LiveSession:
    return LiveSession(
        PerceptionConfig(),
        RiskConfig(),
        _live_config(**config_values),
        camera_factory=lambda _index: camera,
        perceptor_factory=lambda _config: perceptor,
    )


def _result_at(session: LiveSession, frame_index: int):
    return _wait_for(
        lambda: (
            result
            if (result := session.snapshot()[0]) is not None
            and result.record.frame_index >= frame_index
            else None
        )
    )


def test_latest_frame_replaces_unconsumed_frame() -> None:
    latest = LatestFrame()
    first = CapturedFrame(0, 0, 1.0, np.zeros((1, 1, 3), dtype=np.uint8))
    newest = CapturedFrame(2, 20, 1.02, np.ones((1, 1, 3), dtype=np.uint8))

    latest.publish(first)
    latest.publish(newest)

    assert latest.peek() is newest
    assert latest.take_after(-1, timeout=0) is newest
    assert latest.take_after(2, timeout=0) is None


def test_live_flow_keeps_image_detection_binding_and_contiguous_serializable_records() -> None:
    camera = MockCamera()
    perceptor_entered = Event()
    perceptor_release = Event()
    third_entered = Event()
    third_release = Event()

    class BlockingPerceptor:
        calls = 0

        def process(self, image: np.ndarray) -> tuple[Detection, ...]:
            self.calls += 1
            if self.calls == 2:
                perceptor_entered.set()
                assert perceptor_release.wait(1)
            elif self.calls == 3:
                third_entered.set()
                assert third_release.wait(1)
            track_id = int(image[0, 0, 0])
            return (Detection(track_id, 0, "person", 0.9, (280, 300, 360, 450)),)

    session = _session(camera, BlockingPerceptor())
    session.start()
    try:
        camera.push(11)
        first = _result_at(session, 0)
        assert first.captured is not None
        assert first.captured.capture_index == 0
        assert first.record.frame_index == 0
        assert first.record.detections[0].track_id == int(first.captured.image[0, 0, 0])

        # Hold the second inference while capture overwrites its single-frame slot.
        camera.push(12)
        _wait_for(perceptor_entered.is_set)
        camera.push(13)
        camera.push(14)
        _wait_for(lambda: (packet := session.frames.peek()) and packet.capture_index == 3)
        perceptor_release.set()
        _wait_for(third_entered.is_set)

        second = _result_at(session, 1)
        assert second.captured is not None
        assert second.captured.capture_index == 1
        assert second.record.frame_index == 1
        assert second.record.detections[0].track_id == int(second.captured.image[0, 0, 0]) == 12
        assert second.record.timestamp_source == "monotonic"
        assert second.record.timestamp_ms >= first.record.timestamp_ms
        assert FrameRecord.from_json(second.record.to_json()) == second.record
        third_release.set()
        skipped = _result_at(session, 2)
        assert skipped.captured is not None
        assert (skipped.captured.capture_index, skipped.record.frame_index) == (3, 2)
        assert skipped.record.detections[0].track_id == int(skipped.captured.image[0, 0, 0]) == 14
    finally:
        perceptor_release.set()
        third_release.set()
        session.close()
    assert all(not thread.is_alive() for thread in session.threads)


def test_live_assessment_alerts_once_for_persistent_stationary_target() -> None:
    camera = MockCamera()
    session = _session(camera, ImageDetectionPerceptor())
    session.start()
    try:
        camera.push(7)
        _result_at(session, 0)
        Event().wait(0.28)
        camera.push(7)
        second = _result_at(session, 1)
        assert second.decision.frame.events[0].confirmed
        assert [alert.alert_type for alert in second.decision.alerts] == ["first"]

        camera.push(7)
        third = _result_at(session, 2)
        assert third.decision.alerts == ()
        alerts = tuple(alert for alert, _created_at in session.snapshot()[1])
        assert [alert.alert_type for alert in alerts] == ["first"]
    finally:
        session.close()


def test_empty_detections_and_missing_track_id_remain_explicit() -> None:
    class EmptyThenUntracked:
        def process(self, image: np.ndarray) -> tuple[Detection, ...]:
            if int(image[0, 0, 0]) == 1:
                return ()
            return (Detection(None, 0, "person", 0.9, (280, 300, 360, 450)),)

    camera = MockCamera()
    session = _session(camera, EmptyThenUntracked())
    session.start()
    try:
        camera.push(1)
        empty = _result_at(session, 0)
        assert empty.record.detections == ()
        assert empty.record.status == "ok"
        camera.push(2)
        untracked = _result_at(session, 1)
        observation = untracked.decision.frame.observations[0]
        assert observation.source_track_id is None
        assert observation.features is not None
        assert observation.features.quality.track_id_missing
    finally:
        session.close()


@pytest.mark.parametrize(
    ("failure", "message"),
    (("open", "Cannot open camera"), ("read", "frame read failed")),
)
def test_camera_failure_is_reported_and_threads_close(failure: str, message: str) -> None:
    camera = MockCamera(opened=failure != "open", fail_read=failure == "read")
    session = _session(camera, ImageDetectionPerceptor())
    session.start()
    try:
        error, _failed_at = _wait_for(lambda: session.failure() if session.failure()[0] else None)
        assert error is not None and message in error
        _wait_for(camera.released.is_set)
    finally:
        session.close()
    assert all(not thread.is_alive() for thread in session.threads)


def test_stale_inference_marks_processing_error_and_eventually_alerts_on_health() -> None:
    entered = Event()
    release = Event()

    class DelayedPerceptor:
        def process(self, _image: np.ndarray) -> tuple[Detection, ...]:
            entered.set()
            assert release.wait(1)
            return (Detection(1, 0, "person", 0.9, (280, 300, 360, 450)),)

    camera = MockCamera()
    session = _session(
        camera,
        DelayedPerceptor(),
        stale_after_ms=25,
        join_timeout_s=0.05,
    )
    session.start()
    try:
        camera.push(1)
        _wait_for(entered.is_set)
        Event().wait(0.06)
        release.set()
        stale = _wait_for(
            lambda: (
                result
                if (result := session.snapshot()[0]) is not None
                and result.record.status == "processing_error"
                else None
            )
        )
        assert "perception stale" in (stale.record.error or "")
        _wait_for(
            lambda: any(
                alert.alert_type == "perception_limited"
                for alert, _created_at in session.snapshot()[1]
            ),
            timeout=1.5,
        )
    finally:
        release.set()
        session.close()
    assert all(not thread.is_alive() for thread in session.threads)


def test_live_sessions_have_independent_assessment_streams_and_event_ids() -> None:
    left_camera, right_camera = MockCamera(), MockCamera()
    left, right = (
        _session(left_camera, ImageDetectionPerceptor()),
        _session(right_camera, ImageDetectionPerceptor()),
    )
    left.start()
    right.start()
    try:
        left_camera.push(7)
        right_camera.push(7)
        left_result, right_result = _result_at(left, 0), _result_at(right, 0)
        assert left_result.decision.frame.stream_id != right_result.decision.frame.stream_id
        assert left_result.decision.frame.events[0].event_id == "E000001"
        assert right_result.decision.frame.events[0].event_id == "E000001"
        assert left.engine is not right.engine
    finally:
        left.close()
        right.close()
    assert all(not thread.is_alive() for thread in left.threads + right.threads)


def test_live_config_and_cli_preserve_existing_interfaces(tmp_path: Path, monkeypatch) -> None:
    from sightguard.cli import main
    from sightguard.live import app
    from sightguard.live.config import load_live_config

    config = tmp_path / "live.yaml"
    config.write_text("camera: 1\nstale_after_ms: 1500\n", encoding="utf-8")
    assert load_live_config(config).stale_after_ms == 1500
    config.write_text("stale_after_ms: .nan\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_live_config(config)
    calls = []
    monkeypatch.setattr(app, "run_live", lambda *args, **kwargs: calls.append((args, kwargs)))
    assert (
        main(
            [
                "live",
                "--camera",
                "1",
                "--device",
                "cuda:0",
                "--imgsz",
                "320",
                "--width",
                "800",
                "--height",
                "600",
                "--duration",
                "2",
            ]
        )
        == 0
    )
    args, kwargs = calls[0]
    assert args[0].device == "0" and args[0].image_size == 320
    assert (args[2].camera, args[2].width, args[2].height) == (1, 800, 600)
    assert kwargs == {"duration_s": 2.0}
    perception_config = tmp_path / "perception.yaml"
    perception_config.write_text("device: cpu\n", encoding="utf-8")
    assert main(["live", "--config", str(perception_config)]) == 0
    assert calls[1][0][0].device == "cpu"


def test_display_uses_real_engine_alerts_and_hides_expired_boxes(monkeypatch) -> None:
    import cv2

    from sightguard.live.display import COLORS, render_live
    from sightguard.live.runtime import LiveResult

    engine = AssessmentEngine(stream_id="synthetic-live-visual")
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    texts = []
    original_put = cv2.putText

    def record_text(image, text, *args, **kwargs):
        texts.append(text)
        return original_put(image, text, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", record_text)
    alerts = []
    for index, (width, height) in enumerate(
        ((20, 40), (20, 40), (30, 60), (45, 90), (68, 136), (102, 204))
    ):
        record = FrameRecord(
            index,
            index * 250,
            640,
            480,
            (
                Detection(
                    7, 0, "person", 0.9, (320 - width // 2, 336 - height, 320 + width // 2, 336)
                ),
            ),
            timestamp_source="monotonic",
        )
        decision = engine.step(record)
        now = index * 0.25
        packet = CapturedFrame(index, record.timestamp_ms, now, image)
        result = LiveResult(packet, record, decision, 25, now, 4)
        alerts.extend((alert, now) for alert in decision.alerts)
        render_live(result, tuple(alerts), RiskConfig(), LiveConfig(), now)
    assert [alert.alert_type for alert, _ in alerts] == ["first", "escalation"]
    assert "FIRST ALERT | E000001" in texts and "RISK ESCALATION | E000001" in texts
    score = result.decision.frame.observations[0].assessment.score
    assert any(f"R:{score:.3f}" in text and "HIGH" in text for text in texts)
    assert np.array_equal(image, np.zeros_like(image))  # Renderer preserves captured source.
    texts.clear()
    expired = render_live(result, tuple(alerts), RiskConfig(), LiveConfig(), 5.0)
    assert any("risk UNKNOWN" in text for text in texts)
    assert not any(text.startswith("person track:") for text in texts)
    assert not any(f"R:{score:.3f}" in text for text in texts)
    assert (expired[132, 269] != COLORS["HIGH"]).any()
    for index, timestamp in ((6, 1500), (7, 2100)):
        record = FrameRecord(
            index,
            timestamp,
            640,
            480,
            status="processing_error",
            error="mock camera disconnected",
            timestamp_source="monotonic",
        )
        decision = engine.step(record)
    assert [alert.alert_type for alert in decision.alerts] == ["perception_limited"]
    health_result = LiveResult(packet, record, decision, 1, 2.1, 4)
    render_live(
        health_result,
        tuple((alert, 2.1) for alert in decision.alerts),
        RiskConfig(),
        LiveConfig(),
        2.1,
    )
    assert "PERCEPTION LIMITED | H000001" in texts
    engine.finish()


def test_window_h_toggle_and_q_exit_release_session(monkeypatch) -> None:
    from sightguard.live import app

    state = {"started": False, "closed": False, "windows_closed": False}
    details = []

    class MockSession:
        origin = 0
        frames = LatestFrame()

        def __init__(self, *_args):
            pass

        def start(self):
            state["started"] = True

        def snapshot(self):
            return None, ()

        def failure(self):
            return None, None

        def close(self):
            state["closed"] = True

    def render(*_args, **kwargs):
        details.append(kwargs["show_details"])
        return np.zeros((1, 1, 3), dtype=np.uint8)

    keys = iter((ord("h"), ord("Q")))
    monkeypatch.setattr(app, "LiveSession", MockSession)
    monkeypatch.setattr(app, "render_live", render)
    monkeypatch.setattr(app.cv2, "namedWindow", lambda *_args: None)
    monkeypatch.setattr(app.cv2, "imshow", lambda *_args: None)
    monkeypatch.setattr(app.cv2, "waitKey", lambda *_args: next(keys))
    monkeypatch.setattr(app.cv2, "getWindowProperty", lambda *_args: 1)
    monkeypatch.setattr(app.cv2, "destroyAllWindows", lambda: state.update(windows_closed=True))
    app.run_live(PerceptionConfig(), RiskConfig(), LiveConfig())
    assert details == [True, False]
    assert all(state.values())
