"""Single-owner camera adapter; JPEG frames stay in memory, never in Gradio file cache."""

import base64
import html
import logging
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from threading import RLock, Thread
from time import monotonic

import cv2
import numpy as np
import pandas as pd

from sightguard.config import load_config
from sightguard.live.config import LiveConfig, load_live_config, resolve_device
from sightguard.live.display import render_live
from sightguard.live.runtime import LiveSession
from sightguard.risk.config import load_risk_config
from sightguard.ui.presentation import kpi_html
from sightguard.ui.services import ConsoleDataError, ConsoleStore

LOGGER = logging.getLogger(__name__)
EVENT_COLUMNS = ["event_id", "类别", "track_id", "状态", "等级", "风险评分", "当前观察"]
ALERT_COLUMNS = ["alert_id", "event_id", "时间 ms", "类型", "触发原因"]
ALERT_NAMES = {
    "first": "首次提醒",
    "escalation": "风险升级",
    "perception_limited": "感知受限",
    "repeat": "重复提醒",
}
EMPTY_CAMERA = '<div class="sg-live-empty">摄像头未启动 · 点击“开始实时 Demo”</div>'


def camera_choices() -> tuple[list[tuple[str, int]], int]:
    """List DirectShow video devices without opening them; fall back to explicit indices."""
    choices = []
    if sys.platform == "win32":
        import imageio_ffmpeg

        try:
            result = subprocess.run(
                [
                    imageio_ffmpeg.get_ffmpeg_exe(),
                    "-hide_banner",
                    "-list_devices",
                    "true",
                    "-f",
                    "dshow",
                    "-i",
                    "dummy",
                ],
                capture_output=True,
                timeout=5,
                encoding="utf-8",
                errors="replace",
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            names = re.findall(r'"([^"\n]+)" \(video\)', result.stderr)
            choices = [(f"{index} · {name}", index) for index, name in enumerate(names)]
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
            LOGGER.warning("Camera device listing unavailable: %s", exc)
    choices = choices or [("编号 0 · 请核对设备", 0), ("编号 1 · 请核对设备", 1)]
    default = next(
        (index for label, index in choices if "xiaomi" in label.casefold()), choices[0][1]
    )
    return choices, default


def _integer(value: float | int, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not float(value).is_integer()
    ):
        raise ConsoleDataError(f"{name} 必须是整数")
    return int(value)


def _jpeg_html(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise ConsoleDataError("实时画面编码失败")
    data = base64.b64encode(encoded).decode("ascii")
    return (
        '<div class="sg-live-video"><img alt="实时摄像头风险画面" '
        f'src="data:image/jpeg;base64,{data}" /></div>'
    )


@dataclass(frozen=True)
class LiveView:
    image: str = EMPTY_CAMERA
    status: str = "**未运行** · 摄像头只在点击开始后打开"
    metrics: str = ""
    alert: str = '<div class="sg-live-alert">尚无 AlertEvent</div>'
    events: pd.DataFrame | None = None
    alerts: pd.DataFrame | None = None
    running: bool = False


class LiveController:
    """One active session per console, owned by a browser session hash."""

    def __init__(
        self,
        store: ConsoleStore,
        *,
        session_factory: Callable[..., LiveSession] = LiveSession,
        clock: Callable[[], float] = monotonic,
        lease_timeout_s: float = 10,
    ) -> None:
        self.store, self.session_factory, self.clock = store, session_factory, clock
        self.lease_timeout_s = lease_timeout_s
        self.lock = RLock()
        self.session: LiveSession | None = None
        self.owner: str | None = None
        self.last_poll = 0.0
        self.notices: dict[str, str] = {}

    def start(
        self,
        owner: str,
        camera: int,
        device: str,
        width: int,
        height: int,
        imgsz: int,
        perception_id: str,
        risk_id: str,
    ) -> LiveView:
        if not owner:
            raise ConsoleDataError("缺少浏览器会话，请刷新页面")
        with self.lock:
            if self.session is not None:
                if self.owner != owner:
                    raise ConsoleDataError("摄像头已被其他页面占用；请先在该页面停止")
                return self.poll(owner)
            perception = load_config(self.store.config_path(perception_id))
            model = (self.store.root / perception.model_path).resolve()
            if not model.is_relative_to(self.store.root / "models") or not model.is_file():
                raise ConsoleDataError("请先准备 models 目录中的本地权重；不会自动下载")
            risk = load_risk_config(self.store.config_path(risk_id))
            live_path = self.store.root / "configs" / "live.yaml"
            live = (
                load_live_config(self.store.config_path("live.yaml"))
                if live_path.is_file()
                else LiveConfig()
            )
            live = replace(
                live,
                camera=_integer(camera, "camera"),
                width=_integer(width, "width"),
                height=_integer(height, "height"),
            )
            perception = replace(
                perception,
                model_path=str(model),
                device=resolve_device(device),
                image_size=_integer(imgsz, "imgsz"),
            )
            session = self.session_factory(perception, risk, live)
            try:
                session.start()
            except Exception:
                session.close()
                raise
            self.session, self.owner, self.last_poll = session, owner, self.clock()
            self.notices.pop(owner, None)
            Thread(
                target=self._watch, args=(session,), name="sightguard-live-lease", daemon=True
            ).start()
            return self.poll(owner)

    def _stop_locked(self, message: str) -> None:
        if self.session is None:
            return
        self.session.close()  # Keep ownership if release fails; do not start another session.
        if self.owner:
            self.notices[self.owner] = message
            # Bound notices for long-running consoles with many disconnected browser sessions.
            if len(self.notices) > 32:
                self.notices.pop(next(iter(self.notices)))
        self.session, self.owner = None, None

    def stop(self, owner: str) -> LiveView:
        with self.lock:
            if self.session is not None and self.owner == owner:
                self._stop_locked("**已停止** · 摄像头与推理线程已释放")
            return self.poll(owner)

    def shutdown(self) -> None:
        with self.lock:
            self._stop_locked("**服务已关闭**")

    def reap(self) -> None:
        """Release on lost browser heartbeat or terminal camera/model failure."""
        with self.lock:
            if self.session is None:
                return
            error, failed_at = self.session.failure()
            if self.clock() - self.last_poll > self.lease_timeout_s:
                self._stop_locked("**已自动停止** · 页面心跳断开，摄像头已释放")
            elif (
                error
                and failed_at is not None
                and ((self.clock() - failed_at) * 1000 >= self.session.config.failure_display_ms)
            ):
                self._stop_locked(f"**异常停止** · {html.escape(error)} · 摄像头已释放")

    def _watch(self, session: LiveSession) -> None:
        while not session.stop_event.wait(0.5):
            with self.lock:
                if self.session is not session:
                    return
            try:
                self.reap()
            except Exception:
                LOGGER.exception("Live camera release failed")
                return

    def poll(self, owner: str) -> LiveView:
        with self.lock:
            if self.session is None or self.owner != owner:
                status = self.notices.get(owner, LiveView().status)
                if self.session is not None:
                    status = "**设备被占用** · 实时画面仅提供给启动它的页面"
                return LiveView(status=status)
            now = self.clock()
            self.last_poll = now
            session = self.session
            result, alerts = session.snapshot()
            error, _ = session.failure()
            packet = result.captured if result is not None else session.frames.peek()
            age_ms = (now - packet.captured_at) * 1000 if packet else None
            valid = (
                result is not None
                and result.record.status == "ok"
                and age_ms is not None
                and age_ms <= session.config.stale_after_ms
                and not error
            )
            if error:
                status = f"**感知异常 · UNKNOWN** · {html.escape(error)}"
            elif packet is None:
                status = "**启动中** · 正在打开摄像头／加载本地模型"
            elif age_ms > session.config.stale_after_ms:
                status = "**数据过期 · UNKNOWN** · 已隐藏过期目标框"
            elif result is None:
                status = "**等待首次风险更新** · 摄像头已读取画面"
            elif result.record.status != "ok":
                status = f"**感知受限 · UNKNOWN** · {html.escape(result.record.error or '')}"
            else:
                status = f"**实时运行** · 感知状态：{result.decision.frame.perception.status}"
            image = render_live(
                result,
                alerts,
                session.risk,
                session.config,
                now,
                error=error,
                preview=session.frames.peek(),
                include_panel=False,
            )
            events = []
            if result is not None:
                for event in result.decision.frame.events:
                    current = valid and event.observed_now and event.assessment is not None
                    events.append(
                        [
                            event.event_id,
                            event.class_name,
                            event.last_track_id,
                            event.state.value,
                            event.assessment.level.value if current else "UNKNOWN",
                            round(event.assessment.score, 3) if current else None,
                            "是" if event.observed_now else "否 · 历史事件",
                        ]
                    )
            alert_html = LiveView().alert
            if alerts:
                alert, arrived_at = alerts[-1]
                active = (now - arrived_at) * 1000 <= session.config.alert_banner_ms
                alert_html = (
                    f'<div class="sg-live-alert {"active" if active else ""}"><strong>'
                    f"{'新提醒' if active else '最近提醒 · 历史'} · "
                    f"{ALERT_NAMES.get(alert.alert_type, html.escape(alert.alert_type))}</strong>"
                    f"<p>{html.escape(alert.event_id)} · {alert.timestamp_ms:.0f} ms</p>"
                    f"<p>{html.escape(alert.reason)}</p></div>"
                )
            metrics = kpi_html(
                {
                    "风险更新 FPS": f"{result.update_fps if valid else 0:.2f}" if result else "N/A",
                    "处理耗时": f"{result.processing_ms:.1f} ms" if result else "N/A",
                    "帧龄": f"{age_ms:.0f} ms" if age_ms is not None else "N/A",
                    "采集 / 处理序号": f"{packet.capture_index} / {result.record.frame_index}"
                    if packet and result
                    else "N/A",
                }
            )
            return LiveView(
                _jpeg_html(image),
                status,
                metrics,
                alert_html,
                pd.DataFrame(events, columns=EVENT_COLUMNS),
                pd.DataFrame(
                    [
                        [
                            a.alert_id,
                            a.event_id,
                            round(a.timestamp_ms, 1),
                            ALERT_NAMES.get(a.alert_type, a.alert_type),
                            a.reason,
                        ]
                        for a, _ in alerts
                    ],
                    columns=ALERT_COLUMNS,
                ),
                True,
            )
