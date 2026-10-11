"""Manual spatial-audio lab controls for the research console."""

from __future__ import annotations

import atexit
import html
import logging
import math
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from time import monotonic, sleep
from uuid import uuid4

# The flags must be present before Gradio is imported or a Blocks object starts.
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import gradio as gr

from sightguard.spatial.config import SpatialConfig, load_spatial_config
from sightguard.spatial.navigation_adapter import NavigationAdapter, NavigationCue
from sightguard.spatial.schemas import CueSource, DirectionReference, SpatialCue
from sightguard.spatial.worker import AudioWorker

LOGGER = logging.getLogger(__name__)

TONE_CHOICES = [
    ("系统测试 · 三短音", "system_test"),
    ("首次提醒 · 双音", "risk_initial"),
    ("风险升级 · 快速三音", "risk_escalation"),
    ("感知受限 · 长音", "perception_limited"),
    ("导航提示 · 双节奏", "navigation"),
]
RECORD_COLUMNS = [
    "cue_id",
    "source",
    "cue_type",
    "timestamp_ms",
    "azimuth_deg",
    "elevation_deg",
    "submitted_at_ms",
    "enqueue_to_submit_ms",
    "cue_age_ms",
    "action",
]


@dataclass(frozen=True)
class SpatialLabView:
    status: str = "未播放 · 点击播放后才会初始化本机音频设备"
    backend_status: dict | None = None
    records: list[list[object]] | None = None
    enabled: bool = True


def _lab_config(root: Path) -> SpatialConfig:
    config_file = root / "configs" / "spatial.yaml"
    config = load_spatial_config(config_file) if config_file.is_file() else SpatialConfig()
    # An explicit config wins; the bundled soft_oal.dll is only a convenience default.
    if config.library_path:
        library_path = Path(config.library_path).expanduser()
        if not library_path.is_absolute():
            library_path = root / library_path
        return replace(config, library_path=str(library_path.resolve()))
    bundled_library = root / "models" / "openal-soft" / "soft_oal.dll"
    if bundled_library.is_file():
        config = replace(config, library_path=str(bundled_library.resolve()))
    return config


def _device_choices(
    library_path: str | None,
    configured_device: str | None = None,
) -> tuple[list[tuple[str, str | None]], str | None, str | None]:
    """Enumerate OpenAL names without opening a playback device."""
    choices: list[tuple[str, str | None]] = [("OpenAL 系统默认（实际设备以初始化状态为准）", None)]
    try:
        from sightguard.spatial.openal_backend import enumerate_devices

        names = enumerate_devices(library_path)
    except Exception as exc:  # Device discovery must not prevent the lab from opening.
        if configured_device:
            choices.append((f"配置设备 · {configured_device}", configured_device))
        return (
            choices,
            configured_device,
            f"设备枚举不可用：{type(exc).__name__}: {exc}；将尝试系统默认设备。",
        )
    for name in names:
        if not name:
            continue
        label = f"Realtek 物理输出 · {name}" if "realtek" in name.casefold() else name
        choices.append((label, name))
    if configured_device and not any(value == configured_device for _label, value in choices):
        choices.append((f"配置设备 · {configured_device}", configured_device))
    selected = configured_device if configured_device else None
    return choices, selected, None


def direction_svg(azimuth_deg: float, elevation_deg: float) -> str:
    """Render a small direction indicator from bounded numeric input only."""
    azimuth = float(azimuth_deg)
    elevation = float(elevation_deg)
    if not math.isfinite(azimuth) or not -180 <= azimuth <= 180:
        azimuth = 0.0
    if not math.isfinite(elevation) or not -45 <= elevation <= 45:
        elevation = 0.0
    cx, cy, radius = 120.0, 106.0, 67.0
    radians = math.radians(azimuth)
    x = cx + math.sin(radians) * radius
    y = cy - math.cos(radians) * radius
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" role="img" '
        f'aria-label="手动测试方向，方位 {azimuth:.1f} 度，俯仰 {elevation:.1f} 度" '
        'viewBox="0 0 240 230" width="240" height="230">'
        '<rect x="1" y="1" width="238" height="228" rx="14" fill="#111a1e" '
        'stroke="#293b3d"/><circle cx="120" cy="106" r="67" fill="none" '
        'stroke="#52686a" stroke-dasharray="3 5"/><text x="120" y="28" '
        'text-anchor="middle" fill="#c6d6d4" font-size="12">前</text>'
        '<text x="120" y="190" text-anchor="middle" fill="#c6d6d4" '
        'font-size="12">后</text><text x="35" y="110" text-anchor="middle" fill="#c6d6d4" '
        'font-size="12">左</text><text x="205" y="110" text-anchor="middle" fill="#c6d6d4" '
        'font-size="12">右</text><line x1="120" y1="106" '
        f'x2="{x:.2f}" y2="{y:.2f}" stroke="#49e0cc" stroke-width="3" '
        'stroke-linecap="round"/><circle '
        f'cx="{x:.2f}" cy="{y:.2f}" r="7" fill="#49e0cc"/>'
        '<text x="120" y="215" text-anchor="middle" fill="#9db2b0" font-size="11">'
        f"方位 {azimuth:.1f}° · 俯仰 {elevation:.1f}°</text></svg>"
    )


def _finite_range(value: object, name: str, low: float, high: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number in {low:g}..{high:g}")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number in {low:g}..{high:g}") from exc
    if not math.isfinite(result) or not low <= result <= high:
        raise ValueError(f"{name} must be a number in {low:g}..{high:g}")
    return result


def _source_for(cue_type: str) -> CueSource:
    if cue_type == "navigation":
        return CueSource.NAVIGATION
    if cue_type in {"risk_initial", "risk_escalation"}:
        return CueSource.RISK
    if cue_type in {"system_test", "perception_limited"}:
        return CueSource.SYSTEM
    raise ValueError(f"Unsupported test cue type: {cue_type}")


def _priority_for(cue_type: str, config: SpatialConfig) -> int:
    return {
        "system_test": config.system_priority,
        "perception_limited": config.system_priority,
        "risk_initial": config.risk_priority,
        "risk_escalation": config.high_risk_priority,
        "navigation": config.navigation_priority,
    }[cue_type]


def _records(snapshot: dict) -> list[list[object]]:
    rows: list[list[object]] = []
    for record in snapshot.get("records", []):
        cue = record.get("cue", {})
        rows.append(
            [
                cue.get("cue_id"),
                cue.get("source"),
                cue.get("cue_type"),
                cue.get("timestamp_ms"),
                cue.get("azimuth_deg"),
                cue.get("elevation_deg"),
                record.get("submitted_at_ms"),
                record.get("enqueue_to_submit_ms"),
                record.get("cue_age_ms"),
                record.get("action"),
            ]
        )
    return rows


class SpatialLabController:
    """Single browser owner for the native audio worker, with a short lease."""

    def __init__(
        self,
        root: Path,
        *,
        worker_factory: Callable[[SpatialConfig], AudioWorker] = AudioWorker,
        clock: Callable[[], float] = monotonic,
        lease_timeout_s: float = 10.0,
    ) -> None:
        self.root = Path(root)
        self.config = _lab_config(self.root)
        self.worker_factory = worker_factory
        self.clock = clock
        self.lease_timeout_s = lease_timeout_s
        self.lock = threading.RLock()
        self.worker: AudioWorker | None = None
        self.owner: str | None = None
        self.last_poll = 0.0
        self.device_identity: tuple[str | None, str | None, bool] | None = None
        self.notices: dict[str, str] = {}
        self.tab: gr.Tab | None = None

    def _view(self, status: str, snapshot: dict | None = None) -> SpatialLabView:
        value = snapshot or {}
        return SpatialLabView(
            status=status,
            backend_status=value.get("audio_status"),
            records=_records(value),
            enabled=bool(value.get("enabled", True)),
        )

    def poll(self, owner: str) -> SpatialLabView:
        with self.lock:
            if self.worker is None:
                return self._view(
                    self.notices.get(owner, "未播放 · 点击播放后才会初始化本机音频设备")
                )
            if self.owner != owner:
                return self._view("设备已被另一个浏览器页面占用；该页面不能查看或控制当前会话。")
            self.last_poll = self.clock()
            snapshot = self.worker.snapshot()
            error = snapshot.get("error")
            if error:
                status = f"音频初始化／运行失败 · {error}"
            elif not snapshot.get("ready"):
                status = "正在初始化音频设备 · 原生音频线程异步启动中"
            elif not snapshot.get("running"):
                status = "音频线程已结束；可释放设备后重新播放"
            elif not snapshot.get("enabled", True):
                status = "已静音 · 本机音频播放已禁用"
            elif snapshot.get("queued"):
                status = f"已提交测试提示 · 队列 {snapshot.get('queued')}，等待原生音频线程"
            elif snapshot.get("active_cue"):
                status = "正在播放本机空间音频提示"
            else:
                status = "音频设备已就绪 · 等待播放"
            return self._view(status, snapshot)

    def play(
        self,
        owner: str,
        azimuth_deg: object,
        elevation_deg: object,
        gain: object,
        cue_type: str,
        device_name: str | None,
        request_hrtf: bool,
    ) -> SpatialLabView:
        if not owner:
            return self._view("缺少浏览器会话标识，请刷新页面后再试")
        azimuth = _finite_range(azimuth_deg, "azimuth_deg", -180, 180)
        elevation = _finite_range(elevation_deg, "elevation_deg", -45, 45)
        gain_value = _finite_range(gain, "gain", 0, 1)
        if cue_type not in {value for _label, value in TONE_CHOICES}:
            raise ValueError("Unsupported test cue type")
        if type(request_hrtf) is not bool:
            raise ValueError("request_hrtf must be bool")

        with self.lock:
            if self.worker is not None and self.owner != owner:
                return self._view("设备已被另一个浏览器页面占用；不能抢占当前音频会话。")
            desired_identity = (self.config.library_path, device_name, request_hrtf)
            if self.worker is not None and self.device_identity != desired_identity:
                self._release_locked("音频设备设置已更新 · 旧设备已释放")
            if self.worker is None:
                config = replace(
                    self.config,
                    device_name=device_name,
                    request_hrtf=request_hrtf,
                    gain=gain_value,
                )
                worker = self.worker_factory(config)
                try:
                    worker.start()
                except Exception:
                    try:
                        worker.close()
                    except Exception:
                        LOGGER.exception("Spatial audio worker cleanup failed after start error")
                    raise
                self.worker = worker
                self.owner = owner
                self.device_identity = desired_identity
                self.last_poll = self.clock()
                self.notices.pop(owner, None)
                threading.Thread(
                    target=self._watch, args=(worker,), name="sightguard-spatial-lease", daemon=True
                ).start()
            else:
                self.worker.set_gain(gain_value)
                snapshot = self.worker.snapshot()
                if not snapshot.get("enabled", True):
                    return self._view("已静音 · 请先解除静音，再播放测试提示", snapshot)
            worker = self.worker
            if worker is None:
                raise RuntimeError("Audio worker was not created")
            timestamp_ms = max(0.0, float(worker.now_ms()))
            cue = SpatialCue(
                cue_id=uuid4().hex,
                source=_source_for(cue_type),
                event_id=None,
                cue_type=cue_type,
                timestamp_ms=timestamp_ms,
                azimuth_deg=azimuth,
                elevation_deg=elevation,
                direction_reference=DirectionReference.LISTENER,
                direction_confidence=None,
                priority=_priority_for(cue_type, self.config),
                expires_at_ms=timestamp_ms + 1500,
                direction_method="manual_test",
            )
            accepted = worker.submit(cue)
            self.last_poll = self.clock()
            snapshot = worker.snapshot()
            if not accepted:
                return self._view("测试提示未进入队列；请检查音频状态或释放后重试", snapshot)
            return self._view("已提交测试提示 · 声音从运行 SightGuard 的电脑播放", snapshot)

    def stop(self, owner: str) -> SpatialLabView:
        with self.lock:
            if self.worker is None:
                return self.poll(owner)
            if self.owner != owner:
                return self._view("设备已被另一个浏览器页面占用；该页面不能停止当前播放。")
            self.worker.stop()
            self.last_poll = self.clock()
            return self._view("已停止当前提示 · 音频设备仍由此页面占用", self.worker.snapshot())

    def toggle_mute(self, owner: str) -> SpatialLabView:
        with self.lock:
            if self.worker is None:
                return self._view("尚未初始化音频设备；播放一次后可静音或解除静音")
            if self.owner != owner:
                return self._view("设备已被另一个浏览器页面占用；该页面不能更改播放状态。")
            snapshot = self.worker.snapshot()
            enabled = not bool(snapshot.get("enabled", True))
            self.worker.set_enabled(enabled)
            self.last_poll = self.clock()
            return self._view("已解除静音" if enabled else "已静音", self.worker.snapshot())

    def submit_navigation(
        self,
        owner: str,
        goal_id: str,
        azimuth_deg: object,
        elevation_deg: object = 0.0,
    ) -> SpatialLabView:
        """Submit one user-triggered mock-goal cue; never runs a route or repeat loop."""
        azimuth = _finite_range(azimuth_deg, "azimuth_deg", -180, 180)
        elevation = _finite_range(elevation_deg, "elevation_deg", -90, 90)
        navigation = NavigationCue(
            goal_id=goal_id,
            azimuth_deg=azimuth,
            elevation_deg=elevation,
        )
        with self.lock:
            if self.worker is None:
                return self._view("导航模拟未提交 · 请先播放一次并初始化音频设备")
            if self.owner != owner:
                return self._view("设备已被另一个浏览器页面占用；不能提交导航模拟。")
            worker = self.worker
            snapshot = worker.snapshot()
            if snapshot.get("error"):
                return self._view(f"音频初始化／运行失败 · {snapshot['error']}", snapshot)
            if not snapshot.get("enabled", True):
                return self._view("当前已静音 · 请先解除静音，再提交导航模拟", snapshot)

            # A newer manual direction replaces the previous mock goal cue.
            worker.cancel_navigation()
            now_ms = max(0.0, float(worker.now_ms()))
            spatial_cue = NavigationAdapter(self.config).convert(navigation, now_ms)
            accepted = worker.submit(spatial_cue)
            self.last_poll = self.clock()
            snapshot = worker.snapshot()
            if not accepted:
                return self._view(f"导航模拟未进入队列 · goal_id={goal_id}", snapshot)
            if snapshot.get("navigation_paused"):
                status = f"导航模拟已提交并等待恢复 · mock goal_id={goal_id}"
            else:
                status = f"导航模拟已提交一次 · mock goal_id={goal_id}"
            return self._view(status, snapshot)

    def set_navigation_paused(self, owner: str, paused: bool) -> SpatialLabView:
        if type(paused) is not bool:
            raise ValueError("paused must be bool")
        with self.lock:
            if self.worker is None:
                return self._view("导航控制未启用 · 请先播放一次并初始化音频设备")
            if self.owner != owner:
                return self._view("设备已被另一个浏览器页面占用；不能更改导航模拟状态。")
            self.worker.pause_navigation(paused)
            self.last_poll = self.clock()
            status = "导航模拟已暂停" if paused else "导航模拟已恢复"
            return self._view(status, self.worker.snapshot())

    def cancel_navigation(self, owner: str) -> SpatialLabView:
        with self.lock:
            if self.worker is None:
                return self._view("导航控制未启用 · 请先播放一次并初始化音频设备")
            if self.owner != owner:
                return self._view("设备已被另一个浏览器页面占用；不能取消导航模拟。")
            self.worker.cancel_navigation()
            self.last_poll = self.clock()
            return self._view("导航模拟已取消", self.worker.snapshot())

    def release(self, owner: str) -> SpatialLabView:
        with self.lock:
            if self.worker is None:
                return self._view(self.notices.get(owner, "音频设备尚未初始化"))
            if self.owner != owner:
                return self._view("设备已被另一个浏览器页面占用；该页面不能释放当前设备。")
            self._release_locked("音频设备已释放")
            return self._view(self.notices.get(owner, "音频设备已释放"))

    def shutdown(self) -> None:
        with self.lock:
            if self.worker is not None:
                self._release_locked("服务已关闭，音频设备已释放")

    def reap(self) -> None:
        """Release a failed worker or an owner whose browser heartbeat expired."""
        with self.lock:
            worker = self.worker
            if worker is None:
                return
            snapshot = worker.snapshot()
            error = snapshot.get("error")
            if error:
                self._release_locked(f"音频初始化／运行失败 · {error}")
            elif self.clock() - self.last_poll > self.lease_timeout_s:
                self._release_locked("页面心跳已断开 · 音频设备自动释放")

    def _release_locked(self, notice: str) -> None:
        worker = self.worker
        if worker is None:
            return
        worker.close()
        if self.owner:
            self.notices[self.owner] = notice
            if len(self.notices) > 32:
                self.notices.pop(next(iter(self.notices)))
        self.worker, self.owner, self.device_identity = None, None, None

    def _watch(self, worker: AudioWorker) -> None:
        while True:
            sleep(0.5)
            with self.lock:
                if self.worker is not worker:
                    return
                try:
                    self.reap()
                except Exception:
                    LOGGER.exception("Spatial audio worker cleanup failed")
                    return
                if self.worker is not worker:
                    return


def add_spatial_tab(
    app: gr.Blocks,
    root: Path,
    *,
    worker_factory: Callable[[SpatialConfig], AudioWorker] = AudioWorker,
) -> SpatialLabController:
    """Add an interactive, locally rendered spatial-audio research tab."""
    controller = SpatialLabController(root, worker_factory=worker_factory)
    atexit.register(controller.shutdown)
    devices, preferred_device, discovery_notice = _device_choices(
        controller.config.library_path, controller.config.device_name
    )

    def view_outputs(view: SpatialLabView) -> tuple:
        status = html.escape(view.status)
        return (
            f'<div class="sg-status">{status}</div>',
            view.backend_status,
            view.records or [],
            gr.update(value="解除静音" if not view.enabled else "静音"),
        )

    def update_direction(azimuth: float, elevation: float) -> str:
        return direction_svg(azimuth, elevation)

    def play(azimuth, elevation, gain, cue_type, device_name, request_hrtf, request: gr.Request):
        try:
            view = controller.play(
                request.session_hash or "",
                azimuth,
                elevation,
                gain,
                cue_type,
                device_name,
                request_hrtf,
            )
        except Exception as exc:
            view = controller.poll(request.session_hash or "")
            view = replace(view, status=f"播放失败 · {type(exc).__name__}: {exc}")
        return view_outputs(view)

    def stop(request: gr.Request):
        return view_outputs(controller.stop(request.session_hash or ""))

    def mute(request: gr.Request):
        return view_outputs(controller.toggle_mute(request.session_hash or ""))

    def release(request: gr.Request):
        return view_outputs(controller.release(request.session_hash or ""))

    def refresh(request: gr.Request):
        return view_outputs(controller.poll(request.session_hash or ""))

    def navigation_submit(goal_id, azimuth, elevation, request: gr.Request):
        try:
            view = controller.submit_navigation(
                request.session_hash or "", goal_id, azimuth, elevation
            )
        except Exception as exc:
            view = replace(
                controller.poll(request.session_hash or ""),
                status=f"导航模拟失败 · {type(exc).__name__}: {exc}",
            )
        return view_outputs(view)

    def navigation_pause(request: gr.Request):
        return view_outputs(controller.set_navigation_paused(request.session_hash or "", True))

    def navigation_resume(request: gr.Request):
        return view_outputs(controller.set_navigation_paused(request.session_hash or "", False))

    def navigation_cancel(request: gr.Request):
        return view_outputs(controller.cancel_navigation(request.session_hash or ""))

    with gr.Tab("空间音频实验", id="spatial-lab") as tab:
        gr.Markdown(
            "## Spatial Audio Lab\n"
            "声音由运行 SightGuard 的**服务器电脑**播放，浏览器不会输出提示音。"
            "Gradio analytics 与可选启动版本查询已关闭；OpenAL 在本机输出音频，不上传声音。"
            "从低增益开始；页面方向仅是手动声学实验输入，不代表碰撞预测或安全保证。"
            "HRTF 与数据集字段只显示音频后端实际返回的状态；尚未验证听者辨向效果。"
            "系统／风险测试音可与导航模拟 cue 竞争队列，但不代表实时风险提醒。"
        )
        with gr.Row():
            with gr.Column(scale=1):
                azimuth = gr.Slider(-180, 180, value=0, step=1, label="方位角 · 左负右正 / °")
                elevation = gr.Slider(-45, 45, value=0, step=1, label="俯仰角 / °")
                shortcuts = (
                    ("左", -90),
                    ("左前", -45),
                    ("前", 0),
                    ("右前", 45),
                    ("右", 90),
                    ("后", 180),
                )
                preset_buttons: list[gr.Button] = []
                with gr.Row():
                    for label, _value in shortcuts:
                        preset_buttons.append(gr.Button(label, size="sm"))
                gain = gr.Slider(0, 1, value=0.12, step=0.01, label="播放增益 · 默认 0.12")
                cue_type = gr.Dropdown(
                    TONE_CHOICES, value="system_test", label="提示节奏 · 原生 cue type"
                )
                with gr.Row():
                    device = gr.Dropdown(devices, value=preferred_device, label="OpenAL 播放设备")
                    hrtf = gr.Checkbox(value=controller.config.request_hrtf, label="请求 HRTF")
                if discovery_notice:
                    gr.Markdown(f"设备枚举提示：{html.escape(discovery_notice)}")
                with gr.Row():
                    play_button = gr.Button("播放一次", variant="primary")
                    stop_button = gr.Button("停止")
                    mute_button = gr.Button("静音")
                    release_button = gr.Button("释放音频设备")
                with gr.Accordion("导航模拟 · 手动提交一次，不使用地图或自动循环", open=False):
                    gr.Markdown(
                        "Goal ID 是模拟标识。只有点击提交才会发布这次手动方向；"
                        "提交新方向会取消队列中的旧导航 cue。"
                    )
                    nav_goal_id = gr.Textbox(value="mock-goal-1", label="模拟 goal_id")
                    nav_azimuth = gr.Slider(-180, 180, value=45, step=1, label="导航模拟方位角 / °")
                    nav_elevation = gr.Slider(-90, 90, value=0, step=1, label="导航模拟俯仰角 / °")
                    with gr.Row():
                        nav_submit_button = gr.Button("播放一次导航模拟", variant="secondary")
                        nav_pause_button = gr.Button("暂停导航模拟")
                        nav_resume_button = gr.Button("恢复导航模拟")
                        nav_cancel_button = gr.Button("取消导航模拟")
            with gr.Column(scale=1):
                direction = gr.HTML(direction_svg(0, 0))
                status = gr.HTML(
                    '<div class="sg-status">未播放 · 点击播放后才会初始化本机音频设备</div>'
                )
                backend_status = gr.JSON(label="实际后端状态 · 包含 HRTF 与数据集")
        records = gr.Dataframe(
            headers=RECORD_COLUMNS,
            datatype=[
                "str",
                "str",
                "str",
                "number",
                "number",
                "number",
                "number",
                "number",
                "number",
                "str",
            ],
            value=[],
            label="后端提交诊断记录",
            interactive=False,
            wrap=True,
        )
        timer = gr.Timer(1.0, active=True)
        view_targets = [status, backend_status, records, mute_button]
        for preset_button, (_label, value) in zip(preset_buttons, shortcuts, strict=True):
            preset_button.click(
                lambda value=value: value,
                outputs=azimuth,
                queue=False,
                api_name=False,
            ).then(
                play,
                [azimuth, elevation, gain, cue_type, device, hrtf],
                view_targets,
                queue=False,
                api_name=False,
            )
        azimuth.change(
            update_direction, [azimuth, elevation], direction, queue=False, api_name=False
        )
        elevation.change(
            update_direction, [azimuth, elevation], direction, queue=False, api_name=False
        )
        play_button.click(
            play,
            [azimuth, elevation, gain, cue_type, device, hrtf],
            view_targets,
            queue=False,
            api_name="spatial_play",
        )
        stop_button.click(stop, None, view_targets, queue=False, api_name="spatial_stop")
        mute_button.click(mute, None, view_targets, queue=False, api_name="spatial_mute")
        release_button.click(release, None, view_targets, queue=False, api_name="spatial_release")
        nav_submit_button.click(
            navigation_submit,
            [nav_goal_id, nav_azimuth, nav_elevation],
            view_targets,
            queue=False,
            api_name="spatial_navigation_submit",
        )
        nav_pause_button.click(
            navigation_pause, None, view_targets, queue=False, api_name="spatial_navigation_pause"
        )
        nav_resume_button.click(
            navigation_resume, None, view_targets, queue=False, api_name="spatial_navigation_resume"
        )
        nav_cancel_button.click(
            navigation_cancel, None, view_targets, queue=False, api_name="spatial_navigation_cancel"
        )
        timer.tick(refresh, None, view_targets, queue=False, show_progress="hidden")
    app.unload(release)
    app.sightguard_spatial = controller
    controller.tab = tab
    return controller


def build_spatial_lab(
    root: Path,
    *,
    worker_factory: Callable[[SpatialConfig], AudioWorker] = AudioWorker,
) -> gr.Blocks:
    """Build a standalone Blocks app for the audio lab."""
    with gr.Blocks(analytics_enabled=False) as app:
        add_spatial_tab(app, root, worker_factory=worker_factory)
    return app
