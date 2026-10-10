"""Local Gradio Research Console, with session-scoped IDs and read-only result views."""

import html
import json
import os
from pathlib import Path
from typing import Any

# Opt out before importing Gradio; no remote themes or public tunnels are used.
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import gradio as gr
import pandas as pd

from sightguard.ui.annotation_tab import add_annotation_tab
from sightguard.ui.jobs import AnalysisJobs
from sightguard.ui.live_tab import add_live_tab
from sightguard.ui.plots import metrics_plot, risk_plot, timeline_plot
from sightguard.ui.presentation import alert_frame, experiment_summary, kpi_html, metrics_frame
from sightguard.ui.services import BenchmarkView, ConsoleDataError, ConsoleStore
from sightguard.ui.showcase import BRAND_HEADER, FOOTER, add_overview, section_heading
from sightguard.ui.spatial_lab import add_spatial_tab


def _table(value: pd.DataFrame) -> pd.DataFrame:
    """Display undefined values explicitly, without changing persisted numeric metrics."""
    frame = value.astype(object).where(pd.notna(value), "N/A")
    return frame.map(
        lambda item: (
            json.dumps(item, ensure_ascii=False) if isinstance(item, (dict, list, tuple)) else item
        )
    )


def _message(error: Exception) -> str:
    return f"**操作失败 · {type(error).__name__}**\n\n{html.escape(str(error))}"


def _metrics_table(view: BenchmarkView) -> pd.DataFrame:
    if not view.metrics:
        return pd.DataFrame(columns=["strategy", "evaluation"])
    return _table(
        metrics_frame(view.metrics)
        .drop(columns=["response_delays_ms"], errors="ignore")
        .reset_index()
    )


def _video_ids(view: BenchmarkView) -> list[str]:
    return sorted(
        {
            row["video_id"]
            for row in view.metadata.get("inputs", [])
            if isinstance(row, dict) and isinstance(row.get("video_id"), str)
        }
        | {row["video_id"] for row in view.alerts if "video_id" in row}
        | set(view.labels)
    )


def _scored_metrics(view: BenchmarkView) -> dict[str, Any]:
    """Aggregate accuracy needs verified labels for every registered input video."""
    videos = _video_ids(view)
    return view.metrics if videos and all(video in view.labels for video in videos) else {}


def _comparison(
    view: BenchmarkView, video: str | None, strategies: list[str], event: str | None = None
) -> tuple[Any, pd.DataFrame, pd.DataFrame]:
    if video not in _video_ids(view):
        raise ConsoleDataError("请选择本次实验登记的视频")
    selected = tuple(name for name in ("A", "B", "C") if name in strategies)
    if not selected:
        raise ConsoleDataError("至少选择一个策略")
    event = None if event in (None, "__all__") else event
    scope, model_strategy = "gt", None
    if event is not None and "|" in event:
        scope, event = event.split("|", 1)
        if scope == "model":
            model_strategy, event = event.split("|", 1)
    alerts = tuple(
        row
        for row in view.alerts
        if row.get("video_id", row.get("stream_id")) == video and row.get("strategy") in selected
    )
    matches = (
        tuple(
            row
            for row in view.matches
            if row.get("video_id") == video and row.get("strategy") in selected
        )
        if video in view.labels
        else ()
    )
    mapped = {
        (row.get("strategy"), row.get("alert_id")): row
        for row in matches
        if row.get("alert_id") is not None
    }
    related_model_events = (
        {
            (row["strategy"], row["event_id"])
            for row in alerts
            if mapped.get((row["strategy"], row["alert_id"]), {}).get("gt_event_id") == event
        }
        if event is not None and scope == "gt"
        else set()
    )
    visible_alerts = tuple(
        row
        for row in alerts
        if event is None
        or (
            scope == "model"
            and row.get("event_id") == event
            and row.get("strategy") == model_strategy
        )
        or (
            scope == "gt"
            and (
                mapped.get((row["strategy"], row["alert_id"]), {}).get("gt_event_id") == event
                or (row["strategy"], row["event_id"]) in related_model_events
            )
        )
    )
    frame = alert_frame(visible_alerts)
    if not frame.empty:
        frame["match_status"] = [
            mapped.get((row["strategy"], row["alert_id"]), {}).get("status", "unscored")
            for row in visible_alerts
        ]
    detail = pd.DataFrame(
        [
            row
            for row in matches
            if event is None
            or row.get("gt_event_id") == event
            or any(
                alert.get("event_id") == event
                and alert.get("alert_id") == row.get("alert_id")
                and alert.get("strategy") == row.get("strategy")
                for alert in visible_alerts
            )
        ]
    )
    return (
        timeline_plot(
            visible_alerts if scope == "model" else alerts,
            matches,
            None if scope == "model" else view.labels.get(video),
            selected,
            video,
            None if scope == "model" else event,
        ),
        _table(frame),
        _table(detail),
    )


def _event_choices(view: BenchmarkView, video: str | None) -> list[tuple[str, str]]:
    labels = view.labels.get(video)
    choices = [("所有事件", "__all__")]
    if labels:
        choices += [
            (f"GT · {event.event_id} · {event.event_type}", f"gt|{event.event_id}")
            for event in labels.events
        ]
    choices += [
        (f"算法 · {strategy} · {value}", f"model|{strategy}|{value}")
        for strategy, value in sorted(
            {
                (row["strategy"], row["event_id"])
                for row in view.alerts
                if row.get("video_id") == video
            }
        )
    ]
    return choices


def build_app(root: Path | None = None) -> gr.Blocks:
    store = ConsoleStore(root or Path.cwd())
    jobs = AnalysisJobs(store)
    os.environ["GRADIO_TEMP_DIR"] = str(jobs.uploads)
    analysis_ids, benchmark_ids = store.analysis_runs(), store.benchmark_runs()
    manifests = store.dataset_manifests()
    configs = {kind: store.configs(kind) for kind in ("perception", "risk", "benchmark")}
    first_analysis = (
        "tracking_verified"
        if "tracking_verified" in analysis_ids
        else next(iter(analysis_ids), None)
    )
    first_benchmark = (
        "phase3_final" if "phase3_final" in benchmark_ids else next(iter(benchmark_ids), None)
    )
    first_manifest = (
        "synthetic/manifest.csv"
        if "synthetic/manifest.csv" in manifests
        else next(iter(manifests), None)
    )

    def default_config(kind: str, preferred: str) -> str | None:
        return preferred if preferred in configs[kind] else next(iter(configs[kind]), None)

    def load_vision(run_id: str | None) -> tuple:
        try:
            view = store.load_analysis(run_id or "")
            preview, observations, current_alerts, frame_info = store.select_frame(view, 0)
            milestones = [
                (
                    f"{row['event_id']} · {row['timestamp_ms']:g} ms",
                    f"{row['event_id']}|{row['timestamp_ms']}",
                )
                for row in view.events
            ]
            status = f"**已加载 · {html.escape(view.run_id)}** · 来源：{view.source_kind}\n\n"
            status += "\n\n".join(html.escape(warning) for warning in view.warnings)
            return (
                status,
                view.run_id,
                kpi_html(
                    {
                        label: view.stats.get(key)
                        for key, label in (
                            ("frame_count", "处理帧数"),
                            ("detection_count", "检测总数"),
                            ("tracked_object_count", "视频内追踪 ID 数"),
                            ("risk_event_count", "风险事件数"),
                            ("first_alert_count", "首次提醒"),
                            ("escalation_alert_count", "升级提醒"),
                            ("perception_limited_alert_count", "感知受限提示"),
                        )
                    },
                    "运行统计 · 非准确率",
                ),
                jobs.playable_video(view.video_path),
                risk_plot(view.risk_frames),
                _table(alert_frame(view.alerts)),
                _table(pd.DataFrame(view.events)),
                gr.Slider(minimum=0, maximum=max(1, view.duration_ms), value=0, step=1),
                preview,
                _table(observations.drop(columns=["risk_features"], errors="ignore")),
                _table(current_alerts),
                frame_info,
                gr.Dropdown(choices=milestones, value=None),
                None,
            )
        except Exception as exc:
            return (
                _message(exc),
                None,
                kpi_html({}),
                None,
                risk_plot(()),
                pd.DataFrame(),
                pd.DataFrame(),
                gr.Slider(minimum=0, maximum=1, value=0),
                None,
                pd.DataFrame(),
                pd.DataFrame(),
                "未加载有效分析数据",
                gr.Dropdown(choices=[], value=None),
                None,
            )

    def frame_at(run_id: str | None, timestamp: float) -> tuple:
        try:
            if not run_id:
                raise ConsoleDataError("请先加载分析任务")
            image, rows, alerts, info = store.select_frame(store.load_analysis(run_id), timestamp)
            return (
                image,
                _table(rows.drop(columns=["risk_features"], errors="ignore")),
                _table(alerts),
                info,
            )
        except Exception as exc:
            return None, pd.DataFrame(), pd.DataFrame(), _message(exc)

    def event_at(run_id: str | None, token: str | None) -> tuple:
        try:
            if not run_id or not token:
                raise ConsoleDataError("请选择生命周期记录")
            event_id, timestamp = token.rsplit("|", 1)
            view = store.load_analysis(run_id)
            if not any(
                row["event_id"] == event_id and row["timestamp_ms"] == float(timestamp)
                for row in view.events
            ):
                raise ConsoleDataError("定位记录不属于当前分析")
            return (float(timestamp), *frame_at(run_id, float(timestamp)))
        except Exception as exc:
            return (0, None, pd.DataFrame(), pd.DataFrame(), _message(exc))

    # Gradio injects request-specific progress into this recognized default object.
    progress_tracker = gr.Progress()

    def analyze(upload: str, perception: str, risk: str, progress=progress_tracker) -> tuple:
        try:
            run_id = jobs.run(
                upload, perception, risk, lambda value, text: progress(value * 0.95, desc=text)
            )
            progress(0.96, desc="准备浏览器视频副本与结果视图")
            result = load_vision(run_id)
            progress(1, desc="结果已加载")
            return (*result, gr.Dropdown(choices=store.analysis_runs(), value=run_id))
        except Exception as exc:
            result = list(load_vision(None))
            result[0] = _message(exc)
            return (*result, gr.Dropdown(choices=store.analysis_runs(), value=None))

    def export(run_id: str | None) -> str:
        try:
            return str(store.export_analysis(run_id or ""))
        except Exception as exc:
            raise gr.Error(str(exc)) from exc

    def load_comparison(run_id: str | None) -> tuple:
        try:
            view = store.load_benchmark(run_id or "")
            videos = _video_ids(view)
            video = next(iter(videos), None)
            names = list(view.metadata.get("strategies", view.metrics))
            plot, alerts, matches = _comparison(view, video, names)
            accuracy_available = bool(_scored_metrics(view))
            return (
                experiment_summary(view.metadata, view.metrics if accuracy_available else {}),
                view.run_id,
                gr.Dropdown(choices=videos, value=video),
                gr.CheckboxGroup(choices=names, value=names),
                gr.Dropdown(choices=_event_choices(view, video), value="__all__"),
                plot,
                alerts,
                matches,
                _metrics_table(view) if accuracy_available else pd.DataFrame(),
                "\n\n".join(view.warnings) or "已读取完成的实验结果；未重新计算指标。",
            )
        except Exception as exc:
            return (
                _message(exc),
                None,
                gr.Dropdown(choices=[], value=None),
                gr.CheckboxGroup(choices=["A", "B", "C"], value=[]),
                gr.Dropdown(choices=[], value=None),
                timeline_plot(()),
                pd.DataFrame(),
                pd.DataFrame(),
                pd.DataFrame(),
                "未加载有效实验",
            )

    def filter_comparison(run_id: str, video: str, names: list[str], event: str) -> tuple:
        try:
            return (*_comparison(store.load_benchmark(run_id), video, names, event), "筛选完成")
        except Exception as exc:
            return timeline_plot(()), pd.DataFrame(), pd.DataFrame(), _message(exc)

    def comparison_video(run_id: str, video: str, names: list[str]) -> tuple:
        try:
            view = store.load_benchmark(run_id)
            return (
                gr.Dropdown(choices=_event_choices(view, video), value="__all__"),
                *_comparison(view, video, names),
                "已切换视频；事件筛选已重置",
            )
        except Exception as exc:
            return (
                gr.Dropdown(choices=[], value=None),
                timeline_plot(()),
                pd.DataFrame(),
                pd.DataFrame(),
                _message(exc),
            )

    def preview_videos(manifest: str | None) -> gr.Dropdown:
        try:
            frame, _ = store.browse_dataset(manifest or "")
            choices = frame["video_id"].tolist() if "video_id" in frame else []
            return gr.Dropdown(choices=choices, value=next(iter(choices), None))
        except ConsoleDataError:
            return gr.Dropdown(choices=[], value=None)

    def preview(manifest: str, video: str, names: list[str], config: str, risk: str) -> tuple:
        try:
            view = store.strategy_preview(manifest, video, tuple(names), config, risk)
            plot, rows, _ = _comparison(view, video, names)
            return (
                "**策略行为重放 · 未评价准确性**\n\n未使用 Ground Truth；不提供 TP/FP/FN 或召回。",
                plot,
                rows,
            )
        except Exception as exc:
            return _message(exc), timeline_plot(()), pd.DataFrame()

    def load_analytics(run_id: str | None, field: str) -> tuple:
        try:
            view = store.load_benchmark(run_id or "")
            metrics = _scored_metrics(view)
            tables = view.tables if metrics else {}
            return (
                experiment_summary(view.metadata, metrics),
                view.run_id,
                _metrics_table(view) if metrics else pd.DataFrame(),
                metrics_plot(metrics, field),
                _table(tables.get("per_scene", pd.DataFrame())),
                _table(tables.get("per_video", pd.DataFrame())),
                _table(tables.get("per_severity", pd.DataFrame())),
                _table(tables.get("error_cases", pd.DataFrame())),
                "\n\n".join(view.warnings)
                or (
                    "只读加载：指标来自 metrics.json。"
                    if metrics
                    else "缺少完整、可验证的 Ground Truth；隐藏准确性指标及错误分类。"
                ),
            )
        except Exception as exc:
            return (
                _message(exc),
                None,
                pd.DataFrame(),
                metrics_plot({}, field),
                pd.DataFrame(),
                pd.DataFrame(),
                pd.DataFrame(),
                pd.DataFrame(),
                "加载失败",
            )

    def analytics_metric(run_id: str | None, field: str) -> Any:
        try:
            return metrics_plot(_scored_metrics(store.load_benchmark(run_id or "")), field)
        except Exception:
            return metrics_plot({}, field)

    def error_filter(run_id: str | None, strategy: str, video: str) -> pd.DataFrame:
        try:
            view = store.load_benchmark(run_id or "")
            if not _scored_metrics(view):
                return pd.DataFrame()
            frame = view.tables.get("error_cases", pd.DataFrame())
            if strategy != "全部" and "strategy" in frame:
                frame = frame[frame["strategy"] == strategy]
            if video.strip() and "video_id" in frame:
                frame = frame[frame["video_id"] == video.strip()]
            return _table(frame)
        except Exception as exc:
            raise gr.Error(str(exc)) from exc

    def browse(manifest: str | None) -> tuple:
        try:
            frame, message = store.browse_dataset(manifest or "")
            return _table(frame), message
        except Exception as exc:
            return pd.DataFrame(), _message(exc)

    def validate(manifest: str | None) -> str:
        try:
            return store.validate_dataset(manifest or "")
        except Exception as exc:
            return _message(exc)

    metric_choices = [
        ("事件召回率", "event_recall"),
        ("提醒机会召回率", "opportunity_recall"),
        ("无效提醒 / 视频小时", "invalid_alerts_per_hour"),
        ("重复提醒率", "duplicate_rate"),
        ("按时提醒率", "on_time_event_rate"),
        ("命中事件首次延迟 / ms", "first_response_delay_mean_ms"),
        ("总提醒 / 视频分钟", "total_alerts_per_minute"),
        ("TP（机会）", "tp"),
        ("FP（含重复）", "fp"),
        ("FN（机会）", "fn"),
    ]
    with gr.Blocks(
        title="SightGuard · 视界 | 视觉辅助感知",
        analytics_enabled=False,
        fill_width=True,
        delete_cache=(86400, 604800),
    ) as app:
        gr.HTML(BRAND_HEADER)
        with gr.Tabs(selected="overview", elem_id="sg-navigation") as navigation:
            with gr.Tab("产品概览", id="overview") as overview_tab:
                open_live, open_replay = add_overview()
            with gr.Tab("视频回放", id="vision") as vision_tab:
                gr.HTML(
                    section_heading(
                        "VIDEO EXPLORER",
                        "让每一帧，都有迹可循。",
                        "选择已有记录，或上传一段视频，探索目标、风险变化与提醒事件。",
                    )
                )
                vision_state = gr.State(None)
                with gr.Row():
                    vision_run = gr.Dropdown(
                        analysis_ids, value=first_analysis, label="选择回放记录", scale=5
                    )
                    load_button = gr.Button("打开回放", scale=1, variant="primary")
                    refresh = gr.Button("刷新记录", scale=1)
                with gr.Accordion("＋ 上传视频，开始新的分析", open=False):
                    upload = gr.File(
                        file_types=[".mp4"], type="filepath", label="上传 MP4 · ≤500 MiB"
                    )
                    with gr.Row():
                        perception_config = gr.Dropdown(
                            configs["perception"],
                            value=default_config("perception", "default.yaml"),
                            label="感知配置",
                        )
                        risk_config = gr.Dropdown(
                            configs["risk"],
                            value=default_config("risk", "risk.yaml"),
                            label="风险/决策配置",
                        )
                    analyze_button = gr.Button("开始视频分析  ↗", variant="primary")
                    gr.Markdown("每次新建 run_id；使用 models 中的本地权重，原有实验不被覆盖。")
                vision_status = gr.Markdown("请选择分析任务", elem_classes="sg-status")
                vision_kpis = gr.HTML(kpi_html({}))
                with gr.Row():
                    video_player = gr.Video(
                        label="感知画面 · 标注视频",
                        interactive=False,
                        height=360,
                        scale=1,
                    )
                    risk_chart = gr.Plot(label="风险评分曲线", value=risk_plot(()), scale=1)
                with gr.Accordion("逐帧探索 · 目标、时间与事件", open=False):
                    gr.Markdown(
                        "**独立时间定位**：滑块选择最近的先前帧，并显示实际帧号/时间。"
                        "关键帧与数据对应；播放器播放位置不会与滑块双向同步。"
                    )
                    time_slider = gr.Slider(0, 1, value=0, step=1, label="查询视频时间 / ms")
                    frame_info = gr.Markdown("未加载帧")
                    with gr.Row():
                        keyframe = gr.Image(
                            label="所选实际帧 · 视觉标注", interactive=False, height=290, scale=1
                        )
                        observations = gr.Dataframe(
                            label="当前目标 · 风险特征 / 生命周期",
                            interactive=False,
                            wrap=True,
                            max_height=290,
                            scale=2,
                        )
                    current_alerts = gr.Dataframe(label="当前帧提醒与原因", interactive=False)
                    with gr.Row():
                        event_locator = gr.Dropdown([], label="事件生命周期定位", scale=4)
                        locate_button = gr.Button("定位该记录", scale=1)
                with gr.Accordion("完整提醒与生命周期记录", open=False):
                    vision_alerts = gr.Dataframe(
                        label="结构化 AlertEvent", interactive=False, wrap=True
                    )
                    vision_events = gr.Dataframe(label="事件变化记录", interactive=False, wrap=True)
                export_button = gr.Button("导出当前分析结果 ZIP")
                export_file = gr.File(label="下载导出副本", interactive=False)
                vision_outputs = [
                    vision_status,
                    vision_state,
                    vision_kpis,
                    video_player,
                    risk_chart,
                    vision_alerts,
                    vision_events,
                    time_slider,
                    keyframe,
                    observations,
                    current_alerts,
                    frame_info,
                    event_locator,
                    export_file,
                ]
                load_button.click(load_vision, vision_run, vision_outputs, api_name="load_analysis")
                refresh.click(
                    lambda: gr.Dropdown(choices=store.analysis_runs(), value=None),
                    outputs=vision_run,
                    api_name="refresh_analysis",
                )
                analyze_button.click(
                    analyze,
                    [upload, perception_config, risk_config],
                    [*vision_outputs, vision_run],
                    api_name="analyze_video",
                    concurrency_limit=1,
                    concurrency_id="vision-model",
                )
                time_slider.change(
                    frame_at,
                    [vision_state, time_slider],
                    [keyframe, observations, current_alerts, frame_info],
                    api_name="select_frame",
                    trigger_mode="always_last",
                )
                locate_button.click(
                    event_at,
                    [vision_state, event_locator],
                    [time_slider, keyframe, observations, current_alerts, frame_info],
                    api_name="locate_event",
                )
                export_button.click(export, vision_state, export_file, api_name="export_analysis")
            with gr.Tab("策略对比", id="comparison") as comparison_tab:
                gr.HTML(
                    section_heading(
                        "STRATEGY COMPARISON",
                        "同一画面，不同提醒方式。",
                        "在共同的视频时间轴上，观察三种策略如何筛选和触发提醒。",
                    )
                )
                comparison_state = gr.State(None)
                with gr.Row():
                    comparison_run = gr.Dropdown(
                        benchmark_ids,
                        value=first_benchmark,
                        label="选择对照实验",
                        scale=5,
                    )
                    comparison_load = gr.Button("加载策略对比", variant="primary", scale=1)
                comparison_summary = gr.HTML("请选择完成的实验")
                with gr.Row():
                    comparison_video_id = gr.Dropdown([], label="实验视频")
                    strategy_names = gr.CheckboxGroup(
                        ["A", "B", "C"], value=["A", "B", "C"], label="对照策略"
                    )
                    comparison_event = gr.Dropdown([], label="独立 GT / 算法事件")
                comparison_plot = gr.Plot(value=timeline_plot(()), label="共享视频时间轴")
                comparison_status = gr.Markdown("未加载实验")
                with gr.Accordion("提醒行为与匹配详情", open=False):
                    comparison_alerts = gr.Dataframe(
                        label="所选事件的策略行为", interactive=False, wrap=True, max_height=300
                    )
                    comparison_matches = gr.Dataframe(
                        label="机会匹配与遗漏记录", interactive=False, wrap=True, max_height=250
                    )
                with gr.Accordion("整次实验的汇总指标 · 视频筛选不改变此表", open=False):
                    comparison_metrics = gr.Dataframe(
                        interactive=False, label="指标原值 · N/A 未定义"
                    )
                with gr.Accordion("仅重放策略行为 · 不使用 Ground Truth", open=False):
                    gr.Markdown("用于查看同一感知输入下的策略差异；本区域不生成 TP/FP/FN 或召回。")
                    with gr.Row():
                        preview_manifest = gr.Dropdown(
                            manifests, value=first_manifest, label="感知数据登记"
                        )
                        preview_video = gr.Dropdown([], label="视频 ID")
                        preview_names = gr.CheckboxGroup(
                            ["A", "B", "C"], value=["A", "B", "C"], label="策略"
                        )
                    with gr.Row():
                        preview_benchmark_config = gr.Dropdown(
                            configs["benchmark"],
                            value=default_config("benchmark", "benchmark.yaml"),
                            label="基线配置",
                        )
                        preview_risk_config = gr.Dropdown(
                            configs["risk"],
                            value=default_config("risk", "risk.yaml"),
                            label="风险配置",
                        )
                    preview_button = gr.Button("执行行为重放", variant="primary")
                    preview_status = gr.Markdown("未执行重放")
                    preview_plot = gr.Plot(value=timeline_plot(()), label="未评价时间线")
                    preview_alerts = gr.Dataframe(interactive=False, label="行为记录", wrap=True)
                comparison_outputs = [
                    comparison_summary,
                    comparison_state,
                    comparison_video_id,
                    strategy_names,
                    comparison_event,
                    comparison_plot,
                    comparison_alerts,
                    comparison_matches,
                    comparison_metrics,
                    comparison_status,
                ]
                comparison_load.click(
                    load_comparison, comparison_run, comparison_outputs, api_name="load_comparison"
                )
                comparison_video_id.input(
                    comparison_video,
                    [comparison_state, comparison_video_id, strategy_names],
                    [
                        comparison_event,
                        comparison_plot,
                        comparison_alerts,
                        comparison_matches,
                        comparison_status,
                    ],
                    api_name="comparison_video",
                )
                for control in (strategy_names, comparison_event):
                    control.input(
                        filter_comparison,
                        [comparison_state, comparison_video_id, strategy_names, comparison_event],
                        [comparison_plot, comparison_alerts, comparison_matches, comparison_status],
                        api_name=False,
                    )
                preview_manifest.change(
                    preview_videos, preview_manifest, preview_video, api_name="preview_videos"
                )
                preview_button.click(
                    preview,
                    [
                        preview_manifest,
                        preview_video,
                        preview_names,
                        preview_benchmark_config,
                        preview_risk_config,
                    ],
                    [preview_status, preview_plot, preview_alerts],
                    api_name="preview_strategies",
                )
            with gr.Tab("实验洞察", id="analytics") as analytics_tab:
                gr.HTML(
                    section_heading(
                        "RESEARCH INSIGHTS",
                        "用数据，理解每一次提醒。",
                        "浏览已保存的实验指标；数据来源与验证状态始终随结果呈现。",
                    )
                )
                analytics_state = gr.State(None)
                with gr.Row():
                    analytics_run = gr.Dropdown(
                        benchmark_ids, value=first_benchmark, label="Benchmark 运行结果", scale=5
                    )
                    analytics_load = gr.Button("加载实验分析", variant="primary", scale=1)
                analytics_summary = gr.HTML("请选择完成的实验")
                with gr.Accordion("查看原始指标", open=False):
                    analytics_metrics = gr.Dataframe(
                        label="策略指标 · 原始数值 / N/A", interactive=False
                    )
                metric = gr.Dropdown(metric_choices, value="event_recall", label="图表指标与单位")
                analytics_plot = gr.Plot(
                    value=metrics_plot({}, "event_recall"), label="策略指标对比"
                )
                with gr.Accordion("分场景、逐视频与严重程度", open=False):
                    scene_table = gr.Dataframe(label="按场景", interactive=False)
                    per_video_table = gr.Dataframe(label="逐视频", interactive=False)
                    severity_table = gr.Dataframe(label="按重要程度的事件召回", interactive=False)
                with gr.Row():
                    error_strategy = gr.Dropdown(
                        ["全部", "A", "B", "C"], value="全部", label="错误策略筛选"
                    )
                    error_video = gr.Textbox(label="错误视频 ID 精确筛选 · 留空查看全部")
                    errors_button = gr.Button("筛选错误案例")
                errors_table = gr.Dataframe(
                    label="错误案例 · 包含遗漏，不隐藏失败事件", interactive=False, wrap=True
                )
                analytics_status = gr.Markdown("未加载实验")
                analytics_outputs = [
                    analytics_summary,
                    analytics_state,
                    analytics_metrics,
                    analytics_plot,
                    scene_table,
                    per_video_table,
                    severity_table,
                    errors_table,
                    analytics_status,
                ]
                analytics_load.click(
                    load_analytics,
                    [analytics_run, metric],
                    analytics_outputs,
                    api_name="load_analytics",
                )
                metric.input(
                    analytics_metric,
                    [analytics_state, metric],
                    analytics_plot,
                    api_name="analytics_metric",
                )
                errors_button.click(
                    error_filter,
                    [analytics_state, error_strategy, error_video],
                    errors_table,
                    api_name="filter_errors",
                )
            with gr.Tab("数据资料", id="datasets") as dataset_tab:
                gr.HTML(
                    section_heading(
                        "DATA LIBRARY",
                        "每一份观察，都有来源。",
                        "查看视频来源、标注与数据划分，保留研究记录的可追溯性。",
                    )
                )
                with gr.Row():
                    dataset = gr.Dropdown(
                        manifests, value=first_manifest, label="项目数据集 manifest", scale=5
                    )
                    browse_button = gr.Button("浏览登记信息", scale=1)
                    validate_button = gr.Button(
                        "执行 Schema / 来源校验", variant="primary", scale=2
                    )
                gr.Markdown(
                    "只读查看。draft 标注不能视为无风险样本；"
                    "开发集/测试集划分和标注哈希不会被修改。"
                )
                dataset_table = gr.Dataframe(
                    label="视频、划分、标注与感知文件状态", interactive=False, wrap=True
                )
                dataset_status = gr.Markdown("未读取数据集", elem_classes="sg-status")
                browse_button.click(
                    browse, dataset, [dataset_table, dataset_status], api_name="browse_dataset"
                )
                validate_button.click(
                    validate, dataset, dataset_status, api_name="validate_dataset"
                )
            spatial_controller = add_spatial_tab(app, root)
            annotation_tab = add_annotation_tab(root)
            live_controller = add_live_tab(
                app,
                store,
                configs,
                (
                    overview_tab,
                    vision_tab,
                    comparison_tab,
                    analytics_tab,
                    dataset_tab,
                    spatial_controller.tab,
                    annotation_tab,
                ),
            )

        def release_spatial(request: gr.Request) -> None:
            spatial_controller.release(request.session_hash or "")

        for other in (
            overview_tab,
            vision_tab,
            comparison_tab,
            analytics_tab,
            dataset_tab,
            live_controller.tab,
            annotation_tab,
        ):
            other.select(release_spatial, api_name=False, queue=False)
        open_live.click(lambda: gr.update(selected="live"), outputs=navigation, api_name=False)
        open_replay.click(lambda: gr.update(selected="vision"), outputs=navigation, api_name=False)
        gr.HTML(FOOTER)
        if first_analysis:
            app.load(load_vision, vision_run, vision_outputs, api_name=False)
        if first_benchmark:
            app.load(load_comparison, comparison_run, comparison_outputs, api_name=False)
            app.load(load_analytics, [analytics_run, metric], analytics_outputs, api_name=False)
        if first_manifest:
            app.load(browse, dataset, [dataset_table, dataset_status], api_name=False)
            app.load(preview_videos, preview_manifest, preview_video, api_name=False)
    # Narrow media allowlist: metadata/configs/models are never served as file endpoints.
    app.sightguard_store = store
    app.sightguard_jobs = jobs
    app.sightguard_live = live_controller
    app.sightguard_spatial = spatial_controller
    return app.queue(default_concurrency_limit=2, max_size=16)


def launch_ui(
    root: Path, port: int = 7860, *, inbrowser: bool = False, prevent_thread_lock: bool = False
) -> tuple:
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("UI port must be between 1 and 65535")
    app = build_app(root)
    jobs = app.sightguard_jobs
    theme = gr.themes.Base(primary_hue="cyan", neutral_hue="slate").set(
        body_background_fill="#080d10",
        body_background_fill_dark="#080d10",
        block_background_fill="#111a1e",
        block_background_fill_dark="#111a1e",
        body_text_color="#e6edf5",
        body_text_color_dark="#e6edf5",
        input_background_fill="#0c1418",
        input_background_fill_dark="#0c1418",
        input_border_color="#293b3d",
        input_border_color_dark="#293b3d",
        block_border_color="#253436",
        block_border_color_dark="#253436",
        button_secondary_background_fill="#152326",
        button_secondary_background_fill_dark="#152326",
        button_secondary_text_color="#e6edf5",
        button_secondary_text_color_dark="#e6edf5",
    )
    return app.launch(
        server_name="127.0.0.1",
        server_port=port,
        share=False,
        inbrowser=inbrowser,
        prevent_thread_lock=prevent_thread_lock,
        show_error=False,
        allowed_paths=[str(jobs.cache), str(jobs.workspace / "exports")],
        blocked_paths=[str(root / area) for area in ("models", "configs", "datasets", "examples")],
        max_file_size="500mb",
        enable_monitoring=False,
        run_history=False,
        footer_links=[],
        theme=theme,
        css_paths=Path(__file__).parent / "styles" / "console.css",
        js="() => { document.documentElement.classList.add('dark'); }",
    )
