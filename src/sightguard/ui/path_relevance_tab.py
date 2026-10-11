"""Read-only Research Console view for saved Path Relevance Shadow artifacts."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import gradio as gr


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _safe_child(root: Path, *parts: str) -> Path:
    path = root.joinpath(*parts).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("Path Relevance artifact path is outside the output directory") from exc
    return path


def add_path_relevance_tab(app: gr.Blocks, root: Path) -> gr.Tab:
    """Add the offline-only analysis tab; never exposes model suggestions in Annotation Studio."""
    output_root = (root / "outputs" / "path_relevance").resolve()
    runs = (
        sorted(
            path.name
            for path in output_root.iterdir()
            if not path.name.startswith(".")
            and path.is_dir()
            and (path / "run_metadata.json").is_file()
        )
        if output_root.is_dir()
        else []
    )
    first_run = runs[-1] if runs else None
    with gr.Tab("通行相关性 · Shadow", id="path-relevance") as tab:
        gr.Markdown(
            "## PATH RELEVANCE · SHADOW ANALYSIS\n"
            "仅浏览离线研究输出。绿色叠加代表模型识别到的语义地面连通区域，"
            "不是经验证的安全路径，也不改变正式风险分数、AlertEvent 或音频。"
        )
        state = gr.State(None)
        with gr.Row():
            run_selector = gr.Dropdown(runs, value=first_run, label="Path Relevance 运行")
            video_selector = gr.Dropdown([], label="UrbanEgo 视频")
            refresh = gr.Button("刷新输出")
        status = gr.Markdown("尚无 Path Relevance 输出。运行 `sightguard path-relevance` 后刷新。")
        with gr.Row():
            sample_selector = gr.Dropdown([], label="采样帧 · 帧号 / PTS")
            track_selector = gr.Dropdown([("全部", "__all__")], value="__all__", label="目标 Track")
            event_selector = gr.Dropdown([("全部", "__all__")], value="__all__", label="Risk Event")
        with gr.Row():
            preview = gr.Image(
                label="语义区域 / 检测框 / Path Relevance 估计", interactive=False, height=430
            )
            scene_info = gr.Markdown("选择视频和采样帧")
        objects = gr.Dataframe(
            label="当前帧目标 · Baseline Risk / Path Relevance / 不确定性",
            interactive=False,
            wrap=True,
            max_height=340,
        )

        def available_videos(run_id: str | None) -> gr.Dropdown:
            if not run_id:
                return gr.Dropdown(choices=[], value=None)
            run_dir = _safe_child(output_root, run_id)
            choices = sorted(
                path.name
                for path in run_dir.iterdir()
                if path.is_dir() and (path / "scene_frames.jsonl").is_file()
            )
            return gr.Dropdown(choices=choices, value=choices[0] if choices else None)

        def refresh_runs() -> tuple[gr.Dropdown, gr.Dropdown, str]:
            current = (
                sorted(
                    path.name
                    for path in output_root.iterdir()
                    if not path.name.startswith(".")
                    and path.is_dir()
                    and (path / "run_metadata.json").is_file()
                )
                if output_root.is_dir()
                else []
            )
            value = current[-1] if current else None
            return (
                gr.Dropdown(choices=current, value=value),
                available_videos(value),
                "已刷新 Path Relevance 输出" if current else "尚无 Path Relevance 输出。",
            )

        def load_video(run_id: str | None, video_id: str | None) -> tuple:
            if not run_id or not video_id:
                return (
                    None,
                    [],
                    gr.Dropdown(choices=[]),
                    gr.Dropdown(choices=[]),
                    None,
                    "没有可加载的视频结果",
                    [],
                )
            run_dir = _safe_child(output_root, run_id, video_id)
            scene_rows = _read_jsonl(run_dir / "scene_frames.jsonl")
            object_rows = _read_jsonl(run_dir / "objects.jsonl")
            shadow_rows = _read_jsonl(run_dir / "shadow.jsonl")
            overlay_dir = _safe_child(run_dir, "overlays")
            if not scene_rows:
                raise ValueError("The selected run contains no sampled scene results")
            state_value = {
                "run_dir": str(run_dir),
                "scene_rows": scene_rows,
                "object_rows": object_rows,
                "shadow_rows": shadow_rows,
            }
            scene = scene_rows[0]
            frame = int(scene["frame_index"])
            sample_choices = [
                (
                    f"{row['frame_index']} · {float(row['timestamp_ms']):.0f} ms",
                    str(row["frame_index"]),
                )
                for row in scene_rows
            ]
            tracks = sorted(
                {str(row["track_id"]) for row in object_rows if row.get("track_id") is not None}
            )
            events = sorted({str(row["event_id"]) for row in object_rows if row.get("event_id")})
            image_path = _safe_child(overlay_dir, f"{frame:06d}.jpg")
            info, rows = render_frame(state_value, str(frame), "__all__", "__all__")
            return (
                state_value,
                gr.Dropdown(choices=sample_choices, value=str(frame)),
                gr.Dropdown(
                    choices=[("全部", "__all__"), *[(f"Track {value}", value) for value in tracks]],
                    value="__all__",
                ),
                gr.Dropdown(
                    choices=[("全部", "__all__"), *[(value, value) for value in events]],
                    value="__all__",
                ),
                str(image_path),
                info,
                rows,
            )

        def render_frame(
            value: dict[str, Any] | None,
            frame_value: str | None,
            track_id: str | None,
            event_id: str | None,
        ) -> tuple[str, list[dict[str, Any]]]:
            if not value or not frame_value:
                return "选择一个采样帧", []
            frame_index = int(frame_value)
            scene = next(row for row in value["scene_rows"] if row["frame_index"] == frame_index)
            selected = [row for row in value["object_rows"] if row["frame_index"] == frame_index]
            if track_id not in (None, "__all__"):
                selected = [row for row in selected if str(row.get("track_id")) == track_id]
            if event_id not in (None, "__all__"):
                selected = [row for row in selected if row.get("event_id") == event_id]
            fields = (
                "track_id",
                "event_id",
                "class_name",
                "baseline_risk_score",
                "baseline_risk_level",
                "path_relevance_score",
                "relevance_level",
                "corridor_intersection",
                "estimated_spatial_relation",
                "scene_quality",
                "confidence",
                "uncertainty",
                "reasons",
            )
            rows = [{key: row.get(key) for key in fields} for row in selected]
            info = (
                f"**{html.escape(str(scene['video_id']))} · 帧 {frame_index} · "
                f"{float(scene['timestamp_ms']):.0f} ms**\n\n"
                f"场景状态：`{scene['status']}` · 质量：`{scene['quality']}` · "
                f"置信度：`{float(scene['confidence']):.2f}` · "
                f"SegFormer 推理：`{scene.get('inference_latency_ms')}` ms · "
                f"场景处理：`{scene.get('full_scene_processing_ms')}` ms\n\n"
                f"原因：{html.escape(', '.join(scene.get('reasons', [])))}\n\n"
                "相机姿态：未观测；米制深度：不可用。此帧没有模型输出时显示 UNKNOWN。"
            )
            return info, rows

        def select_frame(
            value: dict[str, Any] | None,
            frame_value: str | None,
            track_id: str | None,
            event_id: str | None,
        ) -> tuple:
            info, rows = render_frame(value, frame_value, track_id, event_id)
            if not value or not frame_value:
                return None, info, rows
            frame_index = int(frame_value)
            image_path = _safe_child(Path(value["run_dir"]), "overlays", f"{frame_index:06d}.jpg")
            return str(image_path), info, rows

        refresh.click(
            refresh_runs,
            outputs=[run_selector, video_selector, status],
            api_name="refresh_path_runs",
        )
        run_selector.change(available_videos, run_selector, video_selector, api_name="path_videos")
        video_selector.change(
            load_video,
            [run_selector, video_selector],
            [state, sample_selector, track_selector, event_selector, preview, scene_info, objects],
            api_name="load_path_video",
        )
        for control in (sample_selector, track_selector, event_selector):
            control.change(
                select_frame,
                [state, sample_selector, track_selector, event_selector],
                [preview, scene_info, objects],
                api_name=False,
            )
        if first_run:
            app.load(available_videos, run_selector, video_selector, api_name=False)
    return tab
