"""Raw-frame player and directly editable human-only interval timeline."""

from __future__ import annotations

import copy
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from sightguard.ui.annotation_store import AnnotationStore


def parse_opportunities(rows: list[list]) -> list[dict]:
    """Gradio numeric cells may return strings after keyboard editing."""
    return [
        {
            "opportunity_id": row[0],
            "kind": row[1],
            "window_start_ms": float(row[2]),
            "deadline_ms": float(row[3]),
            "minimum_severity": row[4],
        }
        for row in rows
        if any(value not in ("", None) for value in row)
    ]


def timeline(state: dict) -> tuple[np.ndarray, list[tuple[str, int]]]:
    label = state["label"]
    items = [
        (kind, i) for kind in ("events", "excluded_intervals") for i in range(len(label[kind]))
    ]
    image = np.full((max(120, 60 + 40 * len(items)), 1000, 3), 24, np.uint8)
    duration = label["duration_ms"]

    def x(t: float) -> int:
        return 180 + round(t / duration * 800)

    for row, (kind, i) in enumerate(items):
        item = label[kind][i]
        y = 55 + row * 40
        title = item.get("event_id", f"excluded-{i}")
        color = (82, 154, 240) if kind == "events" else (145, 145, 145)
        if item.get("annotation_status") == "uncertain" or item.get("reason", "").startswith(
            "uncertain:"
        ):
            color = (190, 120, 225)
        cv2.putText(image, title[:22], (5, y + 5), 0, 0.45, (240, 240, 240), 1)
        cv2.rectangle(image, (x(item["start_ms"]), y - 10), (x(item["end_ms"]), y + 10), color, -1)
        for op in item.get("alert_opportunities", []):
            c = (90, 220, 130) if op["kind"] == "initial" else (255, 170, 65)
            cv2.rectangle(
                image, (x(op["window_start_ms"]), y - 7), (x(op["deadline_ms"]), y + 7), c, -1
            )
    t = state["pts"][state["frame_index"]]
    cv2.line(image, (x(t), 20), (x(t), image.shape[0] - 1), (255, 255, 255), 2)
    cv2.putText(image, f"{t:.3f} / {duration:.3f} ms", (180, 15), 0, 0.45, (255, 255, 255), 1)
    return image, items


def add_annotation_tab(root: Path) -> gr.Tab:
    store = AnnotationStore(root)
    manifests = store.manifests()
    default = next(
        (m for m in manifests if m == "real_world/annotations/phase3/manifest.csv"),
        manifests[0] if manifests else None,
    )
    ids = [e.video_id for e in store.entries(default)] if default else []
    with gr.Tab("Annotation Studio", id="annotation") as tab:
        gr.Markdown(
            "## Blind Annotation · 人工独立标注\n只显示原始帧及人工区间。"
            "所有时间为视频 PTS 毫秒；播放可能慢于原帧率。"
        )
        state = gr.State(None)
        selection = gr.State(None)
        with gr.Row():
            dataset = gr.Dropdown(manifests, value=default, label="标注数据集")
            video = gr.Dropdown(ids, value=ids[0] if ids else None, label="标注 video_id")
            load = gr.Button("加载原始视频", variant="primary")
            previous_video = gr.Button("上一段")
            next_video = gr.Button("下一段")
        with gr.Row():
            discard = gr.Checkbox(label="明确丢弃未保存编辑后加载", value=False)
            refresh = gr.Button("刷新进度")
        info = gr.Markdown()
        message = gr.Markdown("未保存的编辑会阻止切换。Reviewed 和 test 标签只读。")
        raw = gr.Image(
            label="原始视频帧 · 无模型叠加",
            type="numpy",
            interactive=False,
            height=390,
            elem_id="studio-raw",
        )
        with gr.Row():
            first = gr.Button("开头")
            back = gr.Button("上一帧")
            play = gr.Button("播放原始帧")
            pause = gr.Button("暂停")
            forward = gr.Button("下一帧")
            last = gr.Button("结尾")
        timer = gr.Timer(0.05, active=False)
        frame_slider = gr.Slider(0, 299, value=0, step=1, label="精确帧编号（0 起始）")
        with gr.Row():
            current_time = gr.Number(label="当前原始 PTS / ms", interactive=False)
            jump_time = gr.Number(label="跳转时间 / ms", value=0)
            jump = gr.Button("定位至不晚于输入时间的原始帧")
        chart = gr.Image(
            elem_id="studio-timeline",
            label="人工时间轴 · 点击条带编辑，空白位置定位",
            type="numpy",
            interactive=False,
        )
        gr.Markdown(
            "蓝：事件；绿：首次机会；橙：升级机会；紫：无法判断；灰：排除。白线是精确选中帧，与浏览器播放器进度无关。"
        )
        with gr.Row():
            new_event = gr.Button("新建事件")
            new_interval = gr.Button("新建排除/无法判断区间")
            event_start = gr.Button("跳到选中区间起点")
            delete = gr.Button("删除选中区间")
        with gr.Row():
            kind = gr.Dropdown(["event", "excluded", "uncertain"], value="event", label="区间类型")
            event_id = gr.Textbox(label="事件 ID", value="event-1")
            start = gr.Number(label="开始 / ms", value=0)
            end = gr.Number(label="结束 / ms", value=1000)
            set_start = gr.Button("当前帧设为开始")
            set_end = gr.Button("当前帧设为结束")
        with gr.Row():
            event_type = gr.Textbox(label="人工事件类型", value="visible_obstruction")
            severity = gr.Dropdown(
                ["LOW", "MEDIUM", "HIGH"], value="MEDIUM", label="人工事件最高等级"
            )
            event_status = gr.Dropdown(
                ["confirmed", "uncertain", "excluded"], value="confirmed", label="事件证据状态"
            )
            classes = gr.Textbox(label="人工 COCO class_ids（逗号分隔）", value="0")
        gr.Markdown(
            "证据不足可直接选“uncertain 区间”，只记录原因，不强迫判断等级。"
            "uncertain/excluded 事件的等级只是 Schema 必填占位，不参与风险匹配。"
        )
        event_notes = gr.Textbox(label="事件备注 / 排除或无法判断原因")
        opportunities = gr.Dataframe(
            headers=[
                "opportunity_id",
                "kind",
                "window_start_ms",
                "deadline_ms",
                "minimum_severity",
            ],
            datatype=["str", "str", "number", "number", "str"],
            type="array",
            value=[["initial-1", "initial", 0, 500, "MEDIUM"]],
            label="首次与升级提醒机会",
            interactive=True,
        )
        regions = gr.Dataframe(
            headers=["start_ms", "end_ms", "x1", "y1", "x2", "y2"],
            datatype=["number"] * 6,
            type="array",
            value=[],
            label="人工目标区域 · 原帧归一化 xyxy，不使用跟踪 ID",
            interactive=True,
        )
        with gr.Row():
            add_initial = gr.Button("添加首次机会行")
            add_upgrade = gr.Button("添加升级机会行")
            add_region = gr.Button("添加人工目标区域行")
        apply = gr.Button("应用区间编辑到草稿", variant="primary")
        with gr.Row():
            annotator = gr.Textbox(label="annotator_id")
            reviewer = gr.Textbox(label="第二审核者 ID")
        notes = gr.Textbox(label="整段标注备注")
        disagreements = gr.Textbox(label="独立复核标识 / 分歧记录（历史证据保留）")
        with gr.Row():
            watched = gr.Checkbox(label="已完整查看原始视频并完成独立人工标注")
            no_risk = gr.Checkbox(label="人工明确确认可评价区间没有值得提醒的事件")
        with gr.Row():
            save = gr.Button("保存 Draft")
            ready = gr.Button("提交 Ready for Review")
            reviewed = gr.Button("第二审核者确认 Reviewed")
            validate = gr.Button("验证当前草稿 Schema 3.0")
            gate = gr.Button("检查正式 Benchmark 闸门")
        progress = gr.Dataframe(
            headers=["video_id", "status", "annotator_id", "reviewer_id", "benchmark_ready"],
            interactive=False,
        )
        totals = gr.Markdown()
        with gr.Accordion("Visual Attributes · 独立视觉属性，不转换风险 GT", open=False):
            visual_image = gr.Image(label="原始关键帧", interactive=False)
            visual_load = gr.Button("加载已有视觉任务")
            object_class = gr.Textbox(label="可见物体类别（人工文本，无法判断留空）")
            visible = gr.Dropdown(["unknown", "true", "false"], value="unknown", label="目标可见")
            position = gr.Dropdown(
                ["unknown", "left", "center", "right"], value="unknown", label="画面位置"
            )
            occupancy = gr.Dropdown(
                ["unknown", "true", "false"], value="unknown", label="可见候选通行区重叠"
            )
            occlusion = gr.Dropdown(["unknown", "true", "false"], value="unknown", label="遮挡")
            motion = gr.Dropdown(
                ["unknown", "true", "false"], value="unknown", label="画面运动（查看相邻帧）"
            )
            sufficient = gr.Dropdown(
                ["unknown", "true", "false"], value="unknown", label="证据充分"
            )
            visual_save = gr.Button("保存独立人工视觉属性")
            visual_message = gr.Markdown()

        nav_outputs = [state, raw, frame_slider, current_time, chart]
        form_outputs = [
            selection,
            kind,
            event_id,
            start,
            end,
            event_type,
            severity,
            event_status,
            classes,
            event_notes,
            opportunities,
            regions,
        ]

        def render(s: dict, index: int) -> tuple:
            if not s:
                raise gr.Error("请先加载视频")
            s = copy.deepcopy(s)
            image, index, pts = store.frame(s, index)
            s["frame_index"] = index
            return s, image, gr.update(value=index, maximum=len(s["pts"]) - 1), pts, timeline(s)[0]

        def opened(manifest: str, vid: str, old: dict, discard_edits: bool) -> tuple:
            try:
                s = store.open(manifest, vid, None if discard_edits else old)
                label = s["label"]
                review = s["review"]
                text = (
                    f"**{vid}** · {s['width']}×{s['height']} · FPS {s['fps']} · "
                    f"{label['duration_ms']:.3f} ms · 来源 {s['source_group']} · "
                    f"状态 {review.get('status', label['annotation_status'])} · "
                    f"作者 {label['annotator_id']} · 审核 {review.get('reviewer_id', '未审核')}"
                )
                return (
                    *render(s, 0),
                    text,
                    "原始视频已加载；未显示模型输出。",
                    None,
                    "" if label["annotator_id"] == "UNASSIGNED" else label["annotator_id"],
                    label["notes"],
                    gr.update(active=False),
                )
            except Exception as exc:
                raise gr.Error(str(exc)) from exc

        load_outputs = nav_outputs + [info, message, selection, annotator, notes, timer]
        load.click(opened, [dataset, video, state, discard], load_outputs, api_name=False)
        dataset.change(
            lambda m: gr.update(
                choices=[e.video_id for e in store.entries(m)], value=store.entries(m)[0].video_id
            ),
            dataset,
            video,
            api_name=False,
        )

        def adjacent(s: dict, step: int, discard_edits: bool) -> tuple:
            if not s:
                raise gr.Error("请先加载视频")
            values = [e.video_id for e in store.entries(s["manifest"])]
            vid = values[max(0, min(len(values) - 1, values.index(s["video_id"]) + step))]
            return (*opened(s["manifest"], vid, s, discard_edits), gr.update(value=vid))

        previous_video.click(
            lambda s, d: adjacent(s, -1, d),
            [state, discard],
            load_outputs + [video],
            api_name=False,
        )
        next_video.click(
            lambda s, d: adjacent(s, 1, d), [state, discard], load_outputs + [video], api_name=False
        )
        first.click(lambda s: render(s, 0), state, nav_outputs, api_name=False)
        last.click(lambda s: render(s, len(s["pts"]) - 1), state, nav_outputs, api_name=False)
        back.click(lambda s: render(s, s["frame_index"] - 1), state, nav_outputs, api_name=False)
        forward.click(lambda s: render(s, s["frame_index"] + 1), state, nav_outputs, api_name=False)
        frame_slider.input(render, [state, frame_slider], nav_outputs, api_name=False)
        jump.click(
            lambda s, t: render(s, store.seek(s, t)),
            [state, jump_time],
            nav_outputs,
            api_name=False,
        )
        play.click(lambda: gr.update(active=True), outputs=timer, api_name=False)
        pause.click(lambda: gr.update(active=False), outputs=timer, api_name=False)

        def tick(s: dict) -> tuple:
            if not s:
                return (*[gr.skip()] * 5, gr.update(active=False))
            index = min(s["frame_index"] + 1, len(s["pts"]) - 1)
            delay = (
                (s["pts"][index + 1] - s["pts"][index]) / 1000
                if index + 1 < len(s["pts"])
                else 0.05
            )
            return (
                *render(s, index),
                (
                    gr.update(value=max(0.01, delay))
                    if index < len(s["pts"]) - 1
                    else gr.update(active=False)
                ),
            )

        timer.tick(tick, state, nav_outputs + [timer], api_name=False)

        def marked(s):
            result = copy.deepcopy(s)
            result.update(dirty=True, form_dirty=True)
            return result, s["pts"][s["frame_index"]]

        set_start.click(marked, state, [state, start], api_name=False)
        set_end.click(marked, state, [state, end], api_name=False)

        def chosen(s: dict, item: tuple[str, int] | None) -> tuple:
            if s and s.get("form_dirty"):
                raise gr.Error("先应用表单编辑，或明确丢弃后重新加载，再切换区间")
            if item is None:
                return (
                    None,
                    "event",
                    "event-new",
                    0,
                    1000,
                    "visible_obstruction",
                    "MEDIUM",
                    "confirmed",
                    "0",
                    "",
                    [["initial-new", "initial", 0, 500, "MEDIUM"]],
                    [],
                )
            collection, index = item
            record = s["label"][collection][index]
            if collection == "excluded_intervals":
                return (
                    item,
                    "uncertain" if record["reason"].startswith("uncertain:") else "excluded",
                    "",
                    record["start_ms"],
                    record["end_ms"],
                    "",
                    "LOW",
                    "excluded",
                    "",
                    record["reason"],
                    [],
                    [],
                )
            ops = [
                [
                    o[k]
                    for k in (
                        "opportunity_id",
                        "kind",
                        "window_start_ms",
                        "deadline_ms",
                        "minimum_severity",
                    )
                ]
                for o in record["alert_opportunities"]
            ]
            boxes = [
                [r["start_ms"], r["end_ms"], *r["bbox_normalized"]]
                for r in record["target_regions"]
            ]
            return (
                item,
                "event",
                record["event_id"],
                record["start_ms"],
                record["end_ms"],
                record["event_type"],
                record["severity"],
                record["annotation_status"],
                ",".join(map(str, record["class_ids"])),
                record["notes"],
                ops,
                boxes,
            )

        new_event.click(lambda s: chosen(s, None), state, form_outputs, api_name=False)

        def fresh_interval(s):
            if s and s.get("form_dirty"):
                raise gr.Error("先应用或明确丢弃表单更改")
            return None, "uncertain", "", "", 0, 1000

        new_interval.click(
            fresh_interval,
            inputs=state,
            outputs=[selection, kind, event_id, event_notes, start, end],
            api_name=False,
        )

        def click_chart(s: dict, evt: gr.SelectData) -> tuple:
            x, y = evt.index
            items = timeline(s)[1]
            row = int((y - 35) // 40)
            item = items[row] if 35 <= y and 0 <= row < len(items) else None
            index = store.seek(
                s,
                max(0, min(s["label"]["duration_ms"], (x - 180) / 800 * s["label"]["duration_ms"])),
            )
            return (*render(s, index), *chosen(s, item))

        chart.select(click_chart, state, nav_outputs + form_outputs, api_name=False)
        event_start.click(
            lambda s, selected: render(
                s, store.seek(s, s["label"][selected[0]][selected[1]]["start_ms"])
            ),
            [state, selection],
            nav_outputs,
            api_name=False,
        )

        def edited(
            s, k, eid, a, b, etype, level, status, cids, reason, ops, boxes, selected, author
        ):
            try:
                if not author.strip():
                    raise ValueError("填写人工 annotator_id")
                if k == "event":
                    item = {
                        "video_id": s["video_id"],
                        "event_id": eid,
                        "event_type": etype,
                        "start_ms": a,
                        "end_ms": b,
                        "severity": level,
                        "annotation_status": status,
                        "annotator_id": author,
                        "notes": reason,
                        "class_ids": [int(c.strip()) for c in cids.split(",")],
                        "alert_opportunities": parse_opportunities(ops),
                        "target_regions": [
                            {
                                "start_ms": float(r[0]),
                                "end_ms": float(r[1]),
                                "bbox_normalized": [float(v) for v in r[2:]],
                            }
                            for r in boxes
                            if any(v not in ("", None) for v in r)
                        ],
                    }
                    collection = "events"
                else:
                    item = {
                        "start_ms": a,
                        "end_ms": b,
                        "reason": ("uncertain: " if k == "uncertain" else "") + reason,
                    }
                    collection = "excluded_intervals"
                index = selected[1] if selected and selected[0] == collection else None
                result = store.edit(s, collection, item, index)
                store.validate(result)
                result["form_dirty"] = False
                return (
                    result,
                    timeline(result)[0],
                    (
                        collection,
                        index if index is not None else len(result["label"][collection]) - 1,
                    ),
                    "草稿已修改，尚未保存。",
                )
            except Exception as exc:
                raise gr.Error(str(exc)) from exc

        apply.click(
            edited,
            [
                state,
                kind,
                event_id,
                start,
                end,
                event_type,
                severity,
                event_status,
                classes,
                event_notes,
                opportunities,
                regions,
                selection,
                annotator,
            ],
            [state, chart, selection, message],
            api_name=False,
        )

        def removed(s, selected):
            if not selected:
                raise gr.Error("点击时间轴选中区间")
            result = store.edit(s, selected[0], None, selected[1])
            result["form_dirty"] = False
            return result, timeline(result)[0], None, "区间已从草稿删除，尚未保存。"

        delete.click(
            removed, [state, selection], [state, chart, selection, message], api_name=False
        )

        def dirty(s):
            if not s:
                return s
            result = copy.deepcopy(s)
            result.update(dirty=True, form_dirty=True)
            return result

        for field in [
            kind,
            event_id,
            start,
            end,
            event_type,
            severity,
            event_status,
            classes,
            event_notes,
            opportunities,
            regions,
        ]:
            field.input(dirty, state, state, api_name=False)

        def opportunity_row(s, rows, a, b, kind):
            result = dirty(s)
            return result, list(rows) + [
                [
                    f"{kind}-{len(rows) + 1}",
                    kind,
                    a,
                    min(a + 500, b),
                    "HIGH" if kind == "escalation" else "MEDIUM",
                ]
            ]

        add_initial.click(
            lambda s, r, a, b: opportunity_row(s, r, a, b, "initial"),
            [state, opportunities, start, end],
            [state, opportunities],
            api_name=False,
        )
        add_upgrade.click(
            lambda s, r, a, b: opportunity_row(s, r, a, b, "escalation"),
            [state, opportunities, start, end],
            [state, opportunities],
            api_name=False,
        )
        add_region.click(
            lambda s, r, a, b: (dirty(s), list(r) + [[a, b, None, None, None, None]]),
            [state, regions, start, end],
            [state, regions],
            api_name=False,
        )

        def metadata_dirty(s):
            if not s:
                return s
            result = copy.deepcopy(s)
            result["dirty"] = True
            return result

        for field in [annotator, notes, reviewer, disagreements, watched, no_risk]:
            field.input(metadata_dirty, state, state, api_name=False)

        def visual_dirty(s):
            if not s:
                return s
            result = copy.deepcopy(s)
            result["visual_dirty"] = True
            return result

        for field in [object_class, visible, position, occupancy, occlusion, motion, sufficient]:
            field.input(visual_dirty, state, state, api_name=False)

        def persist(s, author, n, rev, dis, w, nr, target):
            try:
                if s.get("form_dirty"):
                    raise ValueError("先点击应用区间编辑，或重新加载丢弃表单更改")
                result = store.save(
                    s,
                    author,
                    n,
                    status=target,
                    reviewer=rev,
                    disagreements=dis,
                    watched=w,
                    no_risk=nr,
                )
                result["frame_index"] = s["frame_index"]
                result["visual_dirty"] = s.get("visual_dirty", False)
                return (
                    result,
                    f"已安全保存：{target}。历史版本与审核证据已保留。",
                    timeline(result)[0],
                )
            except Exception as exc:
                raise gr.Error(str(exc)) from exc

        save_inputs = [state, annotator, notes, reviewer, disagreements, watched, no_risk]
        save.click(
            lambda *a: persist(*a, "draft"), save_inputs, [state, message, chart], api_name=False
        )
        ready.click(
            lambda *a: persist(*a, "ready"), save_inputs, [state, message, chart], api_name=False
        )
        reviewed.click(
            lambda *a: persist(*a, "reviewed"), save_inputs, [state, message, chart], api_name=False
        )

        def check(s):
            try:
                store.validate(s)
                return "Schema 3.0 有效；保存或验证不等于完成人工审核。"
            except Exception as exc:
                return f"Invalid：{exc}"

        validate.click(check, state, message, api_name=False)

        def benchmark(m):
            try:
                return store.benchmark_gate(m)
            except Exception as exc:
                return f"Benchmark 拒绝：{exc}"

        gate.click(benchmark, dataset, message, api_name=False)

        def summary(m):
            rows = store.progress(m)
            counts = {
                k: sum(r[1] == k for r in rows)
                for k in ("pending", "draft", "ready", "reviewed", "invalid")
            }
            return rows, f"总数 {len(rows)} · " + " · ".join(f"{k}: {v}" for k, v in counts.items())

        refresh.click(summary, dataset, [progress, totals], api_name=False)

        def visual_open(s):
            try:
                image, task = store.visual(s)
                attrs = task.get("human_attributes") or {}

                def boolean(value):
                    return "unknown" if value is None else str(value).lower()

                return (
                    image,
                    f"原始帧 {task['image_timestamp_ms']} ms；与风险 GT 独立。",
                    attrs.get("object_class") or "",
                    boolean(attrs.get("visible")),
                    attrs.get("position_left_center_right") or "unknown",
                    *[
                        boolean(attrs.get(k))
                        for k in (
                            "overlap_visible_passage",
                            "occlusion",
                            "frame_motion",
                            "evidence_sufficient",
                        )
                    ],
                )
            except Exception as exc:
                raise gr.Error(str(exc)) from exc

        visual_load.click(
            visual_open,
            state,
            [
                visual_image,
                visual_message,
                object_class,
                visible,
                position,
                occupancy,
                occlusion,
                motion,
                sufficient,
            ],
            api_name=False,
        )

        def visual_written(s, author, obj, vis, pos, occ, occl, mot, suff):
            def boolean(value):
                return None if value == "unknown" else value == "true"

            attributes = {
                "object_class": obj or None,
                "visible": boolean(vis),
                "position_left_center_right": None if pos == "unknown" else pos,
                "overlap_visible_passage": boolean(occ),
                "occlusion": boolean(occl),
                "frame_motion": boolean(mot),
                "evidence_sufficient": boolean(suff),
            }
            try:
                store.visual(s, attributes, author)
                result = copy.deepcopy(s)
                result["visual_dirty"] = False
                return result, "视觉属性已独立保存；风险标签未修改。"
            except Exception as exc:
                raise gr.Error(str(exc)) from exc

        visual_save.click(
            visual_written,
            [
                state,
                annotator,
                object_class,
                visible,
                position,
                occupancy,
                occlusion,
                motion,
                sufficient,
            ],
            [state, visual_message],
            api_name=False,
        )
        tab.select(lambda: gr.update(active=False), outputs=timer, api_name=False)
    return tab
