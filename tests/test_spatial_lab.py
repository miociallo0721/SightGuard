from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

from sightguard.spatial.config import SpatialConfig
from sightguard.spatial.schemas import CUE_TYPES, CueSource, DirectionReference, SpatialCue
from sightguard.ui import spatial_lab
from sightguard.ui.spatial_lab import TONE_CHOICES, SpatialLabController, direction_svg


class MockAudioWorker:
    def __init__(self, config: SpatialConfig) -> None:
        self.config = config
        self.started = False
        self.closed = False
        self.enabled = True
        self.gain = config.gain
        self.cues: list[SpatialCue] = []
        self.records: list[dict] = []
        self.error: str | None = None
        self.navigation_paused = False
        self.cancel_navigation_calls = 0
        self.pause_navigation_calls: list[bool] = []

    def start(self) -> None:
        self.started = True

    def now_ms(self) -> float:
        return 125.0

    def submit(self, cue: SpatialCue) -> bool:
        self.cues.append(cue)
        self.records.append(
            {
                "cue": asdict(cue),
                "submitted_at_ms": 128.0,
                "enqueue_to_submit_ms": 3.0,
                "cue_age_ms": 3.0,
                "action": "play",
            }
        )
        return self.enabled and not self.closed

    def snapshot(self) -> dict:
        return {
            "audio_status": {
                "backend": "mock OpenAL",
                "device": "mock device",
                "hrtf_supported": True,
                "hrtf_enabled": True,
                "hrtf_status": "enabled",
                "hrtf_dataset": "mock dataset",
                "fallback_reason": None,
            },
            "ready": self.started,
            "running": self.started and not self.closed,
            "error": self.error,
            "enabled": self.enabled,
            "gain": self.gain,
            "queued": 0,
            "active_cue": None,
            "navigation_paused": self.navigation_paused,
            "records": list(self.records),
        }

    def set_gain(self, gain: float) -> None:
        self.gain = gain

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

    def stop(self) -> None:
        self.records.clear()

    def pause_navigation(self, paused: bool) -> None:
        self.pause_navigation_calls.append(paused)
        self.navigation_paused = paused

    def cancel_navigation(self) -> None:
        self.cancel_navigation_calls += 1

    def close(self) -> None:
        self.closed = True


def _factory_for(workers: list[MockAudioWorker]):
    def factory(config: SpatialConfig) -> MockAudioWorker:
        worker = MockAudioWorker(config)
        workers.append(worker)
        return worker

    return factory


def test_play_builds_schema_cue_and_uses_worker_monotonic_time(tmp_path: Path) -> None:
    workers: list[MockAudioWorker] = []
    controller = SpatialLabController(tmp_path, worker_factory=_factory_for(workers))

    view = controller.play("browser-a", -45, 12, 0.2, "navigation", None, True)

    worker = workers[0]
    cue = worker.cues[0]
    assert worker.started
    assert cue.source is CueSource.NAVIGATION
    assert cue.direction_reference is DirectionReference.LISTENER
    assert cue.direction_method == "manual_test"
    assert (cue.timestamp_ms, cue.expires_at_ms) == (125.0, 1625.0)
    assert (cue.azimuth_deg, cue.elevation_deg) == (-45, 12)
    assert worker.config.gain == 0.2
    assert view.backend_status["hrtf_dataset"] == "mock dataset"
    assert view.records[0][7] == 3.0


def test_perception_limited_tone_uses_system_source_and_priority(tmp_path: Path) -> None:
    workers: list[MockAudioWorker] = []
    controller = SpatialLabController(tmp_path, worker_factory=_factory_for(workers))

    controller.play("browser-a", 0, 0, 0.12, "perception_limited", None, True)

    cue = workers[0].cues[0]
    assert cue.cue_type == "perception_limited"
    assert cue.source is CueSource.SYSTEM
    assert cue.priority == controller.config.system_priority
    assert {value for _label, value in TONE_CHOICES} == CUE_TYPES


def test_other_browser_cannot_stop_mute_or_release_owner_worker(tmp_path: Path) -> None:
    workers: list[MockAudioWorker] = []
    controller = SpatialLabController(tmp_path, worker_factory=_factory_for(workers))
    controller.play("browser-a", 0, 0, 0.12, "system_test", None, True)
    worker = workers[0]

    assert "占用" in controller.stop("browser-b").status
    assert "占用" in controller.toggle_mute("browser-b").status
    assert "占用" in controller.release("browser-b").status
    assert not worker.closed
    assert worker.enabled

    assert "静音" in controller.toggle_mute("browser-a").status
    assert not worker.enabled
    assert controller.release("browser-a").status == "音频设备已释放"
    assert worker.closed


def test_device_or_hrtf_change_releases_and_recreates_worker(tmp_path: Path) -> None:
    workers: list[MockAudioWorker] = []
    controller = SpatialLabController(tmp_path, worker_factory=_factory_for(workers))
    controller.play("browser-a", 0, 0, 0.12, "system_test", None, True)
    controller.play("browser-a", 45, 0, 0.12, "risk_initial", "headphones", False)

    assert len(workers) == 2
    assert workers[0].closed
    assert workers[1].config.device_name == "headphones"
    assert workers[1].config.request_hrtf is False
    assert workers[1].cues[0].source is CueSource.RISK


def test_reap_releases_browser_session_after_ten_second_lease(tmp_path: Path) -> None:
    workers: list[MockAudioWorker] = []
    now = [0.0]
    controller = SpatialLabController(
        tmp_path,
        worker_factory=_factory_for(workers),
        clock=lambda: now[0],
        lease_timeout_s=10,
    )
    controller.play("browser-a", 0, 0, 0.12, "system_test", None, True)
    now[0] = 10.1

    controller.reap()

    assert workers[0].closed
    assert controller.worker is None
    assert "心跳" in controller.poll("browser-a").status


def test_direction_svg_uses_numeric_input_and_clamps_invalid_values() -> None:
    svg = direction_svg(-90, 30)

    assert 'aria-label="手动测试方向，方位 -90.0 度，俯仰 30.0 度"' in svg
    assert 'x2="53.00"' in svg
    assert "NaN" not in direction_svg(float("nan"), float("inf"))
    assert "onload" not in direction_svg(0, 0)


def test_direction_svg_keeps_rear_arrow_and_label_inside_viewbox() -> None:
    svg = direction_svg(180, 0)

    assert 'viewBox="0 0 240 230"' in svg
    assert 'y2="173.00"' in svg
    assert 'y="190"' in svg and ">后</text>" in svg
    assert 'y="215"' in svg


def test_tab_build_does_not_create_worker_before_explicit_play(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import gradio as gr

    workers: list[MockAudioWorker] = []
    monkeypatch.setattr(
        spatial_lab,
        "_device_choices",
        lambda *_args: ([("系统默认设备", None)], None, None),
    )
    with gr.Blocks() as app:
        controller = spatial_lab.add_spatial_tab(
            app, tmp_path, worker_factory=_factory_for(workers)
        )

    assert controller.worker is None
    assert workers == []
    assert app.sightguard_spatial is controller
    assert controller.tab.label == "空间音频实验"


def test_configured_library_path_takes_precedence_over_bundled_library(
    tmp_path: Path,
) -> None:
    from sightguard.ui.spatial_lab import _lab_config

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "spatial.yaml").write_text(
        "library_path: custom/soft_oal.dll\n", encoding="utf-8"
    )
    bundled = tmp_path / "models" / "openal-soft"
    bundled.mkdir(parents=True)
    (bundled / "soft_oal.dll").write_bytes(b"test placeholder")

    config = _lab_config(tmp_path)

    assert config.library_path == str((tmp_path / "custom" / "soft_oal.dll").resolve())


def test_navigation_simulation_uses_adapter_and_replaces_previous_cue(
    tmp_path: Path,
) -> None:
    workers: list[MockAudioWorker] = []
    controller = SpatialLabController(tmp_path, worker_factory=_factory_for(workers))

    not_ready = controller.submit_navigation("browser-a", "mock-goal", 90)
    assert "先播放一次" in not_ready.status
    assert workers == []

    controller.play("browser-a", 0, 0, 0.12, "system_test", None, True)
    first = controller.submit_navigation("browser-a", "mock-goal", -45)
    second = controller.submit_navigation("browser-a", "mock-goal", 90, 5)

    worker = workers[0]
    first_nav, second_nav = worker.cues[1:]
    assert first.status.startswith("导航模拟已提交一次")
    assert second.status.startswith("导航模拟已提交一次")
    assert first_nav.source is CueSource.NAVIGATION
    assert first_nav.direction_method == "simulated"
    assert first_nav.event_id == "mock-goal"
    assert first_nav.azimuth_deg == -45
    assert second_nav.azimuth_deg == 90
    assert second_nav.elevation_deg == 5
    assert second_nav.expires_at_ms - second_nav.timestamp_ms == controller.config.navigation_ttl_ms
    assert worker.cancel_navigation_calls == 2


def test_navigation_controls_enforce_browser_owner_and_call_worker_api(tmp_path: Path) -> None:
    workers: list[MockAudioWorker] = []
    controller = SpatialLabController(tmp_path, worker_factory=_factory_for(workers))
    controller.play("browser-a", 0, 0, 0.12, "system_test", None, True)
    worker = workers[0]

    assert "占用" in controller.set_navigation_paused("browser-b", True).status
    assert "占用" in controller.cancel_navigation("browser-b").status
    assert worker.pause_navigation_calls == []
    assert worker.cancel_navigation_calls == 0

    assert controller.set_navigation_paused("browser-a", True).status == "导航模拟已暂停"
    assert controller.set_navigation_paused("browser-a", False).status == "导航模拟已恢复"
    assert controller.cancel_navigation("browser-a").status == "导航模拟已取消"
    assert worker.pause_navigation_calls == [True, False]
    assert worker.cancel_navigation_calls == 1


def test_device_enumeration_marks_realtek_without_opening_a_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sightguard.spatial import openal_backend

    monkeypatch.setattr(
        openal_backend,
        "enumerate_devices",
        lambda _library: ["Speakers (Realtek(R) Audio)", "Voicemeeter Input"],
    )

    choices, selected, notice = spatial_lab._device_choices(None)

    assert choices[0][0].startswith("OpenAL 系统默认")
    assert choices[1] == (
        "Realtek 物理输出 · Speakers (Realtek(R) Audio)",
        "Speakers (Realtek(R) Audio)",
    )
    assert choices[2] == ("Voicemeeter Input", "Voicemeeter Input")
    assert selected is None
    assert notice is None


@pytest.mark.parametrize(
    "azimuth,elevation,gain",
    [(181, 0, 0.12), (0, 46, 0.12), (0, 0, float("nan"))],
)
def test_play_rejects_out_of_contract_numbers(
    tmp_path: Path, azimuth: float, elevation: float, gain: float
) -> None:
    controller = SpatialLabController(tmp_path, worker_factory=lambda _config: pytest.fail())

    with pytest.raises(ValueError):
        controller.play("browser-a", azimuth, elevation, gain, "system_test", None, True)
