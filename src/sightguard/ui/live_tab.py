"""Gradio control panel around the existing LiveSession adapter."""

import atexit
import html

import gradio as gr
import pandas as pd

from sightguard.ui.live import (
    ALERT_COLUMNS,
    EVENT_COLUMNS,
    LiveController,
    LiveView,
    camera_choices,
)
from sightguard.ui.services import ConsoleStore
from sightguard.ui.showcase import section_heading


def add_live_tab(
    app: gr.Blocks,
    store: ConsoleStore,
    configs: dict[str, list[str]],
    other_tabs: tuple[gr.Tab, ...] = (),
) -> LiveController:
    controller = LiveController(store)
    atexit.register(controller.shutdown)
    choices, preferred_camera = camera_choices()

    def outputs(view: LiveView) -> tuple:
        def table(frame: pd.DataFrame | None, columns: list[str]) -> pd.DataFrame:
            value = frame if frame is not None else pd.DataFrame(columns=columns)
            return value.astype(object).where(pd.notna(value), "N/A")

        return (
            view.image,
            view.status,
            view.metrics,
            view.alert,
            table(view.events, EVENT_COLUMNS),
            table(view.alerts, ALERT_COLUMNS),
            gr.update(active=view.running),
            gr.update(interactive=not view.running),
            gr.update(interactive=view.running),
        )

    def start(camera, device, width, height, imgsz, perception, risk, request: gr.Request) -> tuple:
        try:
            return outputs(
                controller.start(
                    request.session_hash or "",
                    camera,
                    device,
                    width,
                    height,
                    imgsz,
                    perception,
                    risk,
                )
            )
        except Exception as exc:
            view = controller.poll(request.session_hash or "")
            return outputs(
                LiveView(
                    view.image,
                    f"**启动失败** · {html.escape(str(exc))}",
                    view.metrics,
                    view.alert,
                    view.events,
                    view.alerts,
                    view.running,
                )
            )

    def refresh(request: gr.Request) -> tuple:
        return outputs(controller.poll(request.session_hash or ""))

    def stop(request: gr.Request) -> tuple:
        return outputs(controller.stop(request.session_hash or ""))

    def release(request: gr.Request) -> None:
        controller.stop(request.session_hash or "")

    def leave(event: gr.SelectData, request: gr.Request) -> tuple:
        if not event.selected:
            return outputs(controller.stop(request.session_hash or ""))
        return outputs(controller.poll(request.session_hash or ""))

    with gr.Tab("实时体验", id="live", elem_id="sg-live-tab") as tab:
        gr.HTML(
            section_heading(
                "LIVE EXPERIENCE",
                "此刻的画面，持续的感知。",
                "选择本机摄像头，开启视觉感知演示。画面仅在内存中处理。",
            )
        )
        with gr.Row(elem_classes="sg-control-bar"):
            camera = gr.Dropdown(choices, value=preferred_camera, label="选择摄像头", scale=3)
            start_button = gr.Button("开始实时体验  ↗", variant="primary", scale=1)
            stop_button = gr.Button("结束体验", interactive=False, scale=1)
        with gr.Accordion("高级设置 · 设备与画面", open=False):
            with gr.Row():
                device = gr.Dropdown(["auto", "cpu", "cuda"], value="auto", label="推理设备")
                width = gr.Number(value=640, precision=0, label="请求宽度", minimum=1)
                height = gr.Number(value=480, precision=0, label="请求高度", minimum=1)
                imgsz = gr.Number(value=640, precision=0, label="模型输入尺寸", minimum=1)
                perception = gr.Dropdown(
                    configs["perception"],
                    value=next(iter(configs["perception"]), None),
                    label="感知配置 · 本地模型与类别",
                )
                risk = gr.Dropdown(
                    configs["risk"], value=next(iter(configs["risk"]), None), label="风险配置"
                )
            gr.Markdown("设备编号变化后重启控制台刷新列表；其他页面运行时不能抢占设备。")
        status = gr.Markdown(LiveView().status, elem_classes="sg-status")
        metrics = gr.HTML()
        with gr.Row():
            with gr.Column(scale=3, min_width=400):
                image = gr.HTML(LiveView().image)
            with gr.Column(scale=2, min_width=300):
                alert = gr.HTML(LiveView().alert)
                gr.HTML(
                    '<div class="sg-live-guide"><span class="sg-eyebrow">READ THE SIGNAL</span>'
                    "<h3>理解画面中的变化</h3>"
                    '<div class="sg-levels"><span><i class="low"></i>低关注</span>'
                    '<span><i class="medium"></i>中关注</span>'
                    '<span><i class="high"></i>高关注</span>'
                    '<span><i class="unknown"></i>未知</span></div>'
                    "<p>等级对应图像风险评分。感知过期或受限时显示未知状态。</p>"
                    "<p>同等级事件默认抑制重复提醒；评分不是碰撞概率。</p></div>"
                )
        with gr.Accordion("查看事件与提醒详情", open=False):
            events = gr.Dataframe(
                pd.DataFrame(columns=EVENT_COLUMNS), interactive=False, label="事件状态"
            )
            alerts = gr.Dataframe(
                pd.DataFrame(columns=ALERT_COLUMNS),
                interactive=False,
                label="最近实际 AlertEvent",
                wrap=True,
            )
        timer = gr.Timer(0.2, active=False)
        targets = [image, status, metrics, alert, events, alerts, timer, start_button, stop_button]
        start_button.click(
            start,
            [camera, device, width, height, imgsz, perception, risk],
            targets,
            api_name="live_start",
            queue=False,
        )
        stop_button.click(stop, None, targets, api_name="live_stop", queue=False)
        timer.tick(
            refresh,
            None,
            targets,
            api_name="live_poll",
            queue=False,
            show_progress="hidden",
            trigger_mode="once",
        )
        tab.select(leave, None, targets, api_name=False, queue=False)
    app.unload(release)
    for other_tab in other_tabs:
        other_tab.select(stop, None, targets, api_name=False, queue=False)
    return controller
