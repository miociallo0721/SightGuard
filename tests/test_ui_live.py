from __future__ import annotations

import base64
import re
from pathlib import Path
from threading import Event

import cv2
import numpy as np
import pandas as pd
import pytest

from sightguard.data import Detection, FrameRecord
from sightguard.live.config import LiveConfig
from sightguard.live.runtime import CapturedFrame, LatestFrame, LiveResult
from sightguard.risk.config import RiskConfig
from sightguard.risk.engine import AssessmentEngine
from sightguard.ui.live import EMPTY_CAMERA, LiveController
from sightguard.ui.services import ConsoleDataError, ConsoleStore


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


class MockSession:
    def __init__(self, risk: RiskConfig, config: LiveConfig, clock: FakeClock) -> None:
        self.risk = risk
        self.config = config
        self.clock = clock
        self.stop_event = Event()
        self.frames = LatestFrame()
        self.result: LiveResult | None = None
        self.alerts = ()
        self.error: str | None = None
        self.failed_at: float | None = None
        self.started = 0
        self.closed = 0
        self.snapshot_calls = 0

    def start(self) -> None:
        self.started += 1
        self.frames.publish(
            CapturedFrame(
                0,
                0,
                self.clock(),
                np.full((480, 640, 3), 32, dtype=np.uint8),
            )
        )

    def snapshot(self):
        self.snapshot_calls += 1
        return self.result, self.alerts

    def failure(self):
        return self.error, self.failed_at

    def close(self) -> None:
        self.closed += 1
        self.stop_event.set()


def _workspace(
    root: Path,
    *,
    model_path: str = "models/local.pt",
    include_model: bool = True,
    failure_display_ms: int = 100,
    stale_after_ms: int = 100,
) -> ConsoleStore:
    for name in ("configs", "models", "outputs", "datasets"):
        (root / name).mkdir(parents=True, exist_ok=True)
    (root / "configs" / "default.yaml").write_text(
        f"schema_version: '1.0'\nmodel_path: {model_path}\n", encoding="utf-8"
    )
    (root / "configs" / "risk.yaml").write_text('schema_version: "2.0"\n', encoding="utf-8")
    (root / "configs" / "live.yaml").write_text(
        f"stale_after_ms: {stale_after_ms}\nfailure_display_ms: {failure_display_ms}\n",
        encoding="utf-8",
    )
    if include_model and model_path == "models/local.pt":
        (root / "models" / "local.pt").write_bytes(b"local test placeholder")
    return ConsoleStore(root)


def _controller(
    store: ConsoleStore,
    clock: FakeClock,
    *,
    lease_timeout_s: float = 10,
) -> tuple[LiveController, list[MockSession]]:
    sessions: list[MockSession] = []

    def create_session(_perception, risk, config):
        session = MockSession(risk, config, clock)
        sessions.append(session)
        return session

    return (
        LiveController(
            store,
            session_factory=create_session,
            clock=clock,
            lease_timeout_s=lease_timeout_s,
        ),
        sessions,
    )


def _start(controller: LiveController, owner: str = "browser-a"):
    return controller.start(owner, 0, "cpu", 640, 480, 640, "default.yaml", "risk.yaml")


def test_only_owner_can_poll_frames_and_repeated_start_does_not_reinitialize(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    controller, sessions = _controller(_workspace(tmp_path), clock)

    owner_view = _start(controller)
    assert owner_view.running
    assert "data:image/jpeg;base64," in owner_view.image
    assert len(sessions) == 1
    assert sessions[0].started == 1

    calls_before_intruder = sessions[0].snapshot_calls
    intruder_view = controller.poll("browser-b")
    assert intruder_view.running is False
    assert intruder_view.image == EMPTY_CAMERA
    assert "被占用" in intruder_view.status
    assert sessions[0].snapshot_calls == calls_before_intruder

    assert _start(controller).running
    assert len(sessions) == 1
    with pytest.raises(ConsoleDataError, match="其他页面占用"):
        _start(controller, "browser-b")
    assert len(sessions) == 1
    controller.shutdown()
    assert sessions[0].closed == 1


def test_stop_closes_session_and_restart_creates_a_fresh_one(tmp_path: Path) -> None:
    clock = FakeClock()
    controller, sessions = _controller(_workspace(tmp_path), clock)

    _start(controller)
    stopped = controller.stop("browser-a")
    assert not stopped.running
    assert sessions[0].closed == 1

    restarted = _start(controller)
    assert restarted.running
    assert len(sessions) == 2
    assert sessions[1] is not sessions[0]
    assert sessions[1].started == 1
    controller.shutdown()
    assert sessions[1].closed == 1


def test_intelligence_mode_and_model_lifecycle_keep_the_live_session(tmp_path: Path) -> None:
    clock = FakeClock()
    controller, sessions = _controller(_workspace(tmp_path), clock)
    _start(controller)
    session = sessions[0]

    class MockIntelligence:
        def __init__(self) -> None:
            self.calls = []
            self.status = "Not Installed"

        def select(self, mode, *, auxiliary, load):
            self.calls.append((mode, auxiliary, load))
            if load:
                self.status = "Loading"

        def unload(self):
            self.status = "Not Installed"

        def snapshot(self):
            return {"mode": "hybrid", "auxiliary": "vision", "model_status": self.status}

    manager = MockIntelligence()

    def enable(_root):
        session.intelligence = manager
        return manager

    session.enable_intelligence = enable
    selected = controller.intelligence("browser-a", "hybrid", "vision", "load")
    assert selected.intelligence["model_status"] == "Loading"
    assert manager.calls == [("hybrid", "vision", True)]
    assert controller.session is session
    assert session.started == 1

    controller.intelligence("browser-a", "vision", "vision", "select")
    assert manager.calls[-1] == ("vision", "vision", False)
    with pytest.raises(ValueError, match="启动实时会话"):
        controller.intelligence("browser-b", "hybrid", "laya", "load")
    controller.intelligence("browser-a", "hybrid", "vision", "unload")
    assert manager.status == "Not Installed"
    assert controller.session is session
    assert session.started == 1
    controller.shutdown()


def test_lease_reaper_expires_without_intruder_poll_extending_owner_lease(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    controller, sessions = _controller(_workspace(tmp_path), clock, lease_timeout_s=10)
    _start(controller)

    clock.value = 9
    controller.poll("browser-b")
    assert controller.last_poll == 0
    clock.value = 10.1
    controller.reap()

    assert sessions[0].closed == 1
    assert controller.session is None
    assert "自动停止" in controller.poll("browser-a").status


def test_terminal_camera_failure_is_released_after_display_window(tmp_path: Path) -> None:
    clock = FakeClock()
    controller, sessions = _controller(_workspace(tmp_path), clock)
    _start(controller)
    sessions[0].error = "camera disconnected"
    sessions[0].failed_at = clock()

    clock.value = 0.11
    controller.reap()

    assert sessions[0].closed == 1
    assert controller.session is None
    assert "异常停止" in controller.poll("browser-a").status


@pytest.mark.parametrize(
    ("model_path", "include_model", "width", "error"),
    [
        ("models/missing.pt", False, 640, "本地权重"),
        ("../outside.pt", False, 640, "本地权重"),
        ("models/local.pt", True, 640.5, "width 必须是整数"),
    ],
)
def test_invalid_model_path_or_live_input_fails_before_session_start(
    tmp_path: Path, model_path: str, include_model: bool, width: float, error: str
) -> None:
    clock = FakeClock()
    controller, sessions = _controller(
        _workspace(tmp_path, model_path=model_path, include_model=include_model), clock
    )

    with pytest.raises(ConsoleDataError, match=error):
        controller.start("browser-a", 0, "cpu", width, 480, 640, "default.yaml", "risk.yaml")

    assert sessions == []
    assert controller.session is None


def _decode_data_url(value: str) -> np.ndarray:
    match = re.search(r'src="data:image/jpeg;base64,([^"]+)"', value)
    assert match is not None
    payload = base64.b64decode(match.group(1))
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image is not None
    return image


def test_webview_uses_engine_alerts_and_hides_stale_risk_without_writing_cache(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    store = _workspace(tmp_path, stale_after_ms=100)
    controller, sessions = _controller(store, clock)
    _start(controller)
    session = sessions[0]

    risk = RiskConfig()
    engine = AssessmentEngine(risk, stream_id="live-view-test")
    detection = Detection(7, 0, "person", 0.9, (280, 300, 360, 450))
    engine.step(FrameRecord(0, 0, 640, 480, (detection,), timestamp_source="monotonic"))
    record = FrameRecord(1, 250, 640, 480, (detection,), timestamp_source="monotonic")
    decision = engine.step(record)
    assert decision.alerts and decision.alerts[0].alert_type == "first"
    packet = CapturedFrame(
        1,
        250,
        clock(),
        np.full((480, 640, 3), 32, dtype=np.uint8),
    )
    session.frames.publish(packet)
    session.result = LiveResult(packet, record, decision, 12.5, clock(), 4.0)
    session.alerts = tuple((alert, clock()) for alert in decision.alerts)

    fresh = controller.poll("browser-a")
    assert fresh.running
    assert fresh.events is not None
    assert fresh.events.iloc[0]["等级"] == decision.frame.events[0].assessment.level.value
    assert fresh.events.iloc[0]["风险评分"] == pytest.approx(
        round(decision.frame.events[0].assessment.score, 3)
    )
    assert fresh.alerts is not None
    assert fresh.alerts.iloc[0]["类型"] == "首次提醒"
    assert "data:image/jpeg;base64," in fresh.image
    fresh_image = _decode_data_url(fresh.image)

    clock.value = 0.2
    stale = controller.poll("browser-a")
    assert "UNKNOWN" in stale.status
    assert stale.events is not None
    assert stale.events.iloc[0]["等级"] == "UNKNOWN"
    assert pd.isna(stale.events.iloc[0]["风险评分"])
    stale_image = _decode_data_url(stale.image)
    active_box_pixel = fresh_image[348:353, 279:282].mean()
    stale_box_pixel = stale_image[348:353, 279:282].mean()
    assert active_box_pixel > stale_box_pixel + 40
    assert not any((store.outputs_root).rglob("*"))
    controller.shutdown()
