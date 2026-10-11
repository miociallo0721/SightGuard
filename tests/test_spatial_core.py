from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pytest

from sightguard.spatial.config import SpatialConfig, load_spatial_config
from sightguard.spatial.coordinates import camera_azimuth, normalize_azimuth, to_openal
from sightguard.spatial.renderer import AudioStatus
from sightguard.spatial.scheduler import CueScheduler
from sightguard.spatial.schemas import CueSource, DirectionReference, SpatialCue
from sightguard.spatial.worker import AudioWorker


def make_cue(
    cue_id: str,
    *,
    source: CueSource = CueSource.RISK,
    priority: int = 50,
    timestamp_ms: float = 0,
    expires_at_ms: float = 1_000,
    cue_type: str | None = None,
) -> SpatialCue:
    return SpatialCue(
        cue_id=cue_id,
        source=source,
        event_id="event-1" if source == CueSource.RISK else None,
        cue_type=cue_type or ("navigation" if source == CueSource.NAVIGATION else "risk_initial"),
        timestamp_ms=timestamp_ms,
        azimuth_deg=0,
        elevation_deg=0,
        direction_reference=DirectionReference.CAMERA,
        direction_confidence=0.8,
        priority=priority,
        expires_at_ms=expires_at_ms,
        direction_method="camera_geometry",
    )


def test_coordinate_mapping_covers_left_front_right_and_elevation() -> None:
    left, _ = camera_azimuth(0)
    front, _ = camera_azimuth(0.5)
    right, _ = camera_azimuth(1)
    left_position = to_openal(left)
    front_position = to_openal(front)
    right_position = to_openal(right)
    above = to_openal(0, 25)
    below = to_openal(0, -25)

    assert left < front < right
    assert left_position[0] < 0 and left_position[2] < 0
    assert front_position[0] == pytest.approx(0, abs=1e-12)
    assert front_position[2] == pytest.approx(-1)
    assert right_position[0] > 0 and right_position[2] < 0
    assert above[1] > 0 and below[1] < 0


def test_angle_normalization_and_calibrated_camera_geometry() -> None:
    assert normalize_azimuth(181) == -179
    assert normalize_azimuth(-181) == 179
    assert normalize_azimuth(540) == -180
    edge_angle, method = camera_azimuth(1, horizontal_fov_deg=90)
    assert edge_angle == pytest.approx(45)
    assert method == "camera_geometry"


def test_coordinate_helpers_reject_non_finite_and_out_of_range_values() -> None:
    for angle in (float("nan"), float("inf"), True, "0"):
        with pytest.raises(ValueError):
            normalize_azimuth(angle)  # type: ignore[arg-type]
    for center_x in (float("nan"), -0.01, 1.01):
        with pytest.raises(ValueError):
            camera_azimuth(center_x)
    with pytest.raises(ValueError):
        camera_azimuth(0.5, horizontal_fov_deg=float("nan"))
    with pytest.raises(ValueError):
        to_openal(0, elevation_deg=float("nan"))
    with pytest.raises(ValueError):
        to_openal(0, radius=float("inf"))


def test_spatial_cue_json_round_trip_is_lossless() -> None:
    cue = make_cue("risk-1")
    assert SpatialCue.from_json(cue.to_json()) == cue
    assert json.loads(cue.to_json())["schema_version"] == "1.0"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", "9.0"),
        ("cue_type", "unrecognized"),
        ("timestamp_ms", float("nan")),
        ("priority", True),
        ("direction_confidence", 1.1),
    ],
)
def test_spatial_cue_json_rejects_invalid_contract_values(field: str, value: object) -> None:
    raw = json.loads(make_cue("risk-1").to_json())
    raw[field] = value
    with pytest.raises(ValueError):
        SpatialCue.from_json(json.dumps(raw, allow_nan=True))


def test_spatial_cue_json_rejects_missing_schema_and_unknown_fields() -> None:
    with pytest.raises(ValueError, match="schema"):
        SpatialCue.from_json("{}")
    raw = json.loads(make_cue("risk-1").to_json())
    raw["unrecognized"] = 1
    with pytest.raises(TypeError):
        SpatialCue.from_json(json.dumps(raw))
    with pytest.raises(ValueError):
        SpatialCue.from_json("not-json")


def test_spatial_config_loads_repository_defaults() -> None:
    assert load_spatial_config(Path("configs/spatial.yaml")) == SpatialConfig()


@pytest.mark.parametrize(
    "text",
    [
        "[]",
        "unknown_setting: true",
        "queue_capacity: 257",
        "gain: .nan",
    ],
)
def test_spatial_config_rejects_invalid_yaml_contract(tmp_path: Path, text: str) -> None:
    path = tmp_path / "spatial.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises((TypeError, ValueError)):
        load_spatial_config(path)


def test_scheduler_selects_deterministically_and_keeps_one_active_cue() -> None:
    scheduler = CueScheduler(capacity=4)
    assert scheduler.submit(make_cue("z", priority=50), 0)
    assert scheduler.submit(make_cue("b", priority=50), 0)
    assert scheduler.submit(make_cue("a", priority=50), 0)

    first = scheduler.update(0, playing=False)
    assert first.cue is not None and first.cue.cue_id == "a"
    assert scheduler.update(0, playing=True).cue is None
    second = scheduler.update(0, playing=False)
    assert second.cue is not None and second.cue.cue_id == "b"


def test_scheduler_capacity_retains_higher_priority_pending_cues() -> None:
    scheduler = CueScheduler(capacity=2)
    assert scheduler.submit(make_cue("low", priority=20), 0)
    assert scheduler.submit(make_cue("middle", priority=40), 0)
    assert not scheduler.submit(make_cue("lowest", priority=10), 0)
    assert scheduler.submit(make_cue("urgent", priority=80), 0)
    assert {cue.cue_id for cue in scheduler.pending} == {"middle", "urgent"}
    assert scheduler.update(0, playing=False).cue == make_cue("urgent", priority=80)


def test_scheduler_rejects_duplicate_and_expired_cues_and_stops_expired_active() -> None:
    scheduler = CueScheduler()
    expired = make_cue("old", expires_at_ms=5)
    assert not scheduler.submit(expired, 5)
    cue = make_cue("active", expires_at_ms=10)
    assert scheduler.submit(cue, 0)
    assert not scheduler.submit(cue, 0)
    assert scheduler.update(0, playing=False).cue == cue
    stopped = scheduler.update(10, playing=True)
    assert stopped.stop
    assert stopped.cue is None
    assert scheduler.active is None


def test_scheduler_preempts_navigation_for_risk_then_resumes_navigation() -> None:
    scheduler = CueScheduler()
    navigation = make_cue("nav", source=CueSource.NAVIGATION, priority=20, expires_at_ms=500)
    risk = make_cue("risk", priority=80, expires_at_ms=500)
    assert scheduler.submit(navigation, 0)
    assert scheduler.update(0, playing=False).cue == navigation
    assert scheduler.submit(risk, 1)

    preemption = scheduler.update(1, playing=True)
    assert preemption.stop and preemption.cue == risk
    assert scheduler.update(2, playing=False).cue == navigation


def test_scheduler_pauses_and_resumes_navigation_but_cancel_removes_it() -> None:
    scheduler = CueScheduler()
    navigation = make_cue("nav", source=CueSource.NAVIGATION, priority=20, expires_at_ms=500)
    assert scheduler.submit(navigation, 0)
    assert scheduler.update(0, playing=False).cue == navigation
    scheduler.navigation_paused = True
    paused = scheduler.update(1, playing=True)
    assert paused.stop and paused.cue is None
    assert navigation in scheduler.pending

    scheduler.navigation_paused = False
    assert scheduler.update(2, playing=False).cue == navigation
    scheduler.cancel_navigation()
    cancelled = scheduler.update(3, playing=True)
    assert cancelled.stop and cancelled.cue is None


def test_scheduler_does_not_restore_navigation_after_its_deadline() -> None:
    scheduler = CueScheduler()
    navigation = make_cue("nav", source=CueSource.NAVIGATION, priority=20, expires_at_ms=10)
    assert scheduler.submit(navigation, 0)
    assert scheduler.update(0, playing=False).cue == navigation
    scheduler.navigation_paused = True
    assert scheduler.update(1, playing=True).stop
    assert scheduler.update(10, playing=False).cue is None
    scheduler.navigation_paused = False
    assert scheduler.update(11, playing=False).cue is None


class MockAudioBackend:
    def __init__(
        self,
        *,
        play_gate: threading.Event | None = None,
        fail_open: bool = False,
    ) -> None:
        self.play_gate = play_gate
        self.fail_open = fail_open
        self.opened = threading.Event()
        self.play_started = threading.Event()
        self.closed = threading.Event()
        self.stopped = threading.Event()
        self.second_poll = threading.Event()
        self.lock = threading.Lock()
        self.playing = False
        self.play_polls_remaining = 0
        self.play_attempts = 0
        self.poll_count = 0
        self.thread_ids: list[int] = []
        self.status = AudioStatus("mock", "mock-device", False, False, "unsupported", None, None)

    def _record_thread(self) -> None:
        self.thread_ids.append(threading.get_ident())

    def open(self) -> AudioStatus:
        self._record_thread()
        self.opened.set()
        if self.fail_open:
            raise RuntimeError("mock open failure")
        return self.status

    def play(
        self,
        samples: np.ndarray,
        sample_rate: int,
        position: tuple[float, ...],
        gain: float,
    ) -> None:
        self._record_thread()
        self.play_attempts += 1
        self.play_started.set()
        if self.play_gate is not None:
            self.play_gate.wait()
        with self.lock:
            self.playing = True
            self.play_polls_remaining = 2

    def is_playing(self) -> bool:
        self._record_thread()
        with self.lock:
            self.poll_count += 1
            if self.poll_count >= 2:
                self.second_poll.set()
            playing = self.playing
            # Worker checks once immediately after play, then once on its next loop.
            if playing:
                self.play_polls_remaining -= 1
                if self.play_polls_remaining == 0:
                    self.playing = False
            return playing

    def stop(self) -> None:
        self._record_thread()
        with self.lock:
            self.playing = False
        self.stopped.set()

    def close(self) -> None:
        self._record_thread()
        self.closed.set()


def worker_config() -> SpatialConfig:
    return SpatialConfig(poll_interval_ms=1, join_timeout_s=2)


def test_audio_worker_owns_backend_lifecycle_on_one_thread() -> None:
    backend = MockAudioBackend()
    worker = AudioWorker(worker_config(), backend_factory=lambda: backend, clock_ms=lambda: 0)
    worker.start()
    try:
        assert backend.opened.wait(2)
        assert worker.submit(make_cue("cue", expires_at_ms=5_000))
        assert backend.play_started.wait(2)
    finally:
        worker.close()

    assert backend.closed.is_set()
    assert worker.snapshot()["audio_status"]["backend"] == "mock"
    assert len(set(backend.thread_ids)) == 1


def test_audio_worker_reports_backend_open_failure_and_cleans_up() -> None:
    backend = MockAudioBackend(fail_open=True)
    worker = AudioWorker(worker_config(), backend_factory=lambda: backend, clock_ms=lambda: 0)
    worker.start()
    assert worker.ready.wait(2)
    try:
        assert worker.snapshot()["error"] == "RuntimeError: mock open failure"
        assert not worker.submit(make_cue("cue"))
    finally:
        worker.close()
    assert backend.closed.is_set()
    assert len(set(backend.thread_ids)) == 1


def test_audio_worker_submit_stays_available_during_play_and_mute_stops_queue() -> None:
    gate = threading.Event()
    backend = MockAudioBackend(play_gate=gate)
    worker = AudioWorker(worker_config(), backend_factory=lambda: backend, clock_ms=lambda: 0)
    worker.start()
    returned = threading.Event()
    result: list[bool] = []

    try:
        assert backend.opened.wait(2)
        assert worker.submit(make_cue("active", priority=60, expires_at_ms=5_000))
        assert backend.play_started.wait(2)

        def submit_while_playing() -> None:
            result.append(worker.submit(make_cue("queued", priority=80, expires_at_ms=5_000)))
            returned.set()

        producer = threading.Thread(target=submit_while_playing)
        producer.start()
        try:
            assert returned.wait(2)
            assert result == [True]
            worker.set_enabled(False)
            assert not worker.submit(make_cue("muted", expires_at_ms=5_000))
        finally:
            gate.set()
            producer.join(2)
        assert backend.stopped.wait(2)
    finally:
        gate.set()
        worker.close()


def test_audio_worker_cancels_active_playback_after_backend_call_returns() -> None:
    gate = threading.Event()
    backend = MockAudioBackend(play_gate=gate)
    worker = AudioWorker(worker_config(), backend_factory=lambda: backend, clock_ms=lambda: 0)
    worker.start()
    try:
        assert backend.opened.wait(2)
        assert worker.submit(make_cue("active", expires_at_ms=5_000))
        assert backend.play_started.wait(2)
        worker.cancel("active")
        gate.set()
        assert backend.stopped.wait(2)
    finally:
        gate.set()
        worker.close()


def test_audio_worker_rechecks_expiry_after_synthesis_before_native_play(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sightguard.spatial.sounds as sounds

    synthesis_started = threading.Event()
    synthesis_release = threading.Event()
    expiry_read = threading.Event()
    now = [0.0]

    def slow_synthesize(cue_type: str) -> np.ndarray:
        synthesis_started.set()
        synthesis_release.wait()
        return np.zeros(1, dtype=np.int16)

    def clock_ms() -> float:
        value = now[0]
        if value >= 1_000:
            expiry_read.set()
        return value

    monkeypatch.setattr(sounds, "synthesize", slow_synthesize)
    backend = MockAudioBackend()
    worker = AudioWorker(worker_config(), backend_factory=lambda: backend, clock_ms=clock_ms)
    worker.start()
    try:
        assert backend.opened.wait(2)
        assert worker.submit(make_cue("expires", timestamp_ms=0, expires_at_ms=1_000))
        assert synthesis_started.wait(2)
        now[0] = 1_000
        synthesis_release.set()
        assert expiry_read.wait(2)
        assert backend.second_poll.wait(2)
        assert backend.play_attempts == 0
    finally:
        synthesis_release.set()
        worker.close()


def test_audio_device_lease_is_exclusive_and_released_after_close() -> None:
    first_backend = MockAudioBackend()
    first = AudioWorker(worker_config(), backend_factory=lambda: first_backend)
    second_backend = MockAudioBackend()
    second = AudioWorker(worker_config(), backend_factory=lambda: second_backend)
    first.start()
    try:
        assert first.ready.wait(2) and first.error is None
        second.start()
        assert second.ready.wait(2)
        assert "already owned" in second.error
        assert not second_backend.opened.is_set()
    finally:
        first.close()
        second.close()
    third = AudioWorker(worker_config(), backend_factory=MockAudioBackend)
    third.start()
    try:
        assert third.ready.wait(2) and third.error is None
    finally:
        third.close()
