"""Plotly figures for Phase 2 risk streams and Phase 3 evaluation artifacts."""

from __future__ import annotations

import html
import math
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import plotly.graph_objects as go

from sightguard.evaluation.matching import prepare_ground_truth
from sightguard.evaluation.schema import VideoAnnotations

_DARK = "plotly_dark"
_STATUS_COLORS = {
    "matched": "#22c55e",
    "duplicate": "#f59e0b",
    "false_positive": "#ef4444",
    "ignored": "#94a3b8",
    "unscored": "#a78bfa",
}
_STATUS_LABELS = {
    "matched": "已匹配",
    "duplicate": "重复提醒",
    "false_positive": "无支持提醒",
    "ignored": "已排除",
    "unscored": "未评分/诊断",
}
_TYPE_SYMBOLS = {
    "first": "circle",
    "escalation": "triangle-up",
    "trigger": "square",
    "repeat": "diamond",
    "perception_limited": "x",
}
_TYPE_LABELS = {
    "first": "首次",
    "escalation": "升级",
    "trigger": "基线触发",
    "repeat": "重复",
    "perception_limited": "感知健康提示",
}


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _text(value: Any, default: str = "") -> str:
    return html.escape(str(value if value is not None else default))


def _dict(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _empty_figure(title: str, message: str, *, x_title: str, y_title: str) -> go.Figure:
    figure = go.Figure()
    figure.update_layout(
        template=_DARK,
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        annotations=[
            {
                "text": message,
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
                "font": {"size": 15, "color": "#cbd5e1"},
            }
        ],
    )
    return figure


def risk_plot(risk_frames: tuple[dict, ...]) -> go.Figure:
    """Plot observed risk scores over video time without interpolating missing frames."""
    points: dict[str, list[tuple[float, float, list[Any]]]] = defaultdict(list)
    for frame in risk_frames:
        frame = _dict(frame)
        timestamp_ms = _number(frame.get("timestamp_ms"))
        if timestamp_ms is None:
            continue
        timestamp_s = timestamp_ms / 1000
        observations = frame.get("observations", ())
        if not isinstance(observations, (list, tuple)):
            continue
        for raw_observation in observations:
            observation = _dict(raw_observation)
            assessment = _dict(observation.get("assessment"))
            score = _number(assessment.get("score"))
            if score is None:
                continue
            event_id = str(observation.get("event_id") or "未关联事件")
            severity = str(assessment.get("level") or "未知")
            quality = _dict(assessment.get("quality"))
            if quality.get("score") is None:
                features = _dict(observation.get("features"))
                quality = _dict(features.get("quality"))
            quality_score = _number(quality.get("score"))
            quality_text = "N/A" if quality_score is None else f"{quality_score:.3f}"
            class_name = str(observation.get("class_name") or "未知类别")
            customdata = [
                html.escape(event_id),
                html.escape(severity),
                quality_text,
                html.escape(class_name),
                frame.get("frame_index"),
            ]
            points[event_id].append((timestamp_s, score, customdata))

    if not points:
        return _empty_figure(
            "风险分数随视频时间变化",
            "没有可绘制的有效风险观测",
            x_title="视频时间（秒）",
            y_title="风险分数（0–1）",
        )

    figure = go.Figure()
    for event_id in sorted(points):
        rows = sorted(points[event_id], key=lambda item: item[0])
        figure.add_trace(
            go.Scatter(
                x=[row[0] for row in rows],
                y=[row[1] for row in rows],
                customdata=[row[2] for row in rows],
                mode="markers",
                name=html.escape(event_id),
                marker={"size": 8},
                hovertemplate=(
                    "时间：%{x:.3f} 秒<br>风险分数：%{y:.3f}<br>"
                    "事件：%{customdata[0]}<br>等级：%{customdata[1]}<br>"
                    "感知质量：%{customdata[2]}<br>类别：%{customdata[3]}<br>"
                    "帧：%{customdata[4]}<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        template=_DARK,
        title="风险分数随视频时间变化",
        xaxis_title="视频时间（秒）",
        yaxis_title="风险分数（0–1）",
        yaxis={"range": [0, 1]},
        legend_title="风险事件",
        hovermode="closest",
    )
    return figure


def _video_id(record: Mapping[str, Any]) -> str | None:
    value = record.get("video_id", record.get("stream_id"))
    return None if value is None else str(value)


def _record_status(rows: list[Mapping[str, Any]]) -> tuple[str, str, str]:
    if not rows:
        return "unscored", "", "没有匹配记录"
    row = rows[0]
    status = str(row.get("status") or "unscored")
    if status == "perception_limited":
        return "unscored", str(row.get("gt_event_id") or ""), str(row.get("reason") or status)
    normalized = status if status in _STATUS_COLORS else "unscored"
    return (
        normalized,
        str(row.get("gt_event_id") or ""),
        str(row.get("reason") or status),
    )


def timeline_plot(
    alerts: tuple[dict, ...],
    matches: tuple[dict, ...] = (),
    labels: VideoAnnotations | None = None,
    strategies: tuple[str, ...] = ("A", "B", "C"),
    video_id: str | None = None,
    event_id: str | None = None,
) -> go.Figure:
    """Show strategy alerts and independent opportunity windows on aligned lanes.

    ``event_id`` is a Ground Truth ID. Model event IDs are used only after a Ground
    Truth match identifies a strategy event, to reveal the rest of that strategy
    event's alert history.
    """
    lane_names = tuple(dict.fromkeys(str(name) for name in strategies))
    label_video = labels.video_id if labels is not None else None
    selected_video = video_id or label_video
    filtered_alerts = [
        _dict(alert)
        for alert in alerts
        if (selected_video is None or _video_id(_dict(alert)) == selected_video)
        and str(_dict(alert).get("strategy") or "C") in lane_names
    ]
    filtered_matches = [
        _dict(match)
        for match in matches
        if (selected_video is None or _video_id(_dict(match)) == selected_video)
        and str(_dict(match).get("strategy") or "") in lane_names
    ]

    match_rows_by_alert: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for match in filtered_matches:
        alert_key = match.get("alert_id")
        strategy = str(match.get("strategy") or "")
        if alert_key is not None:
            match_rows_by_alert[(strategy, str(alert_key))].append(match)

    if event_id is not None:
        direct_alerts: set[tuple[str, str]] = set()
        model_events: set[tuple[str, str]] = set()
        for match in filtered_matches:
            if str(match.get("gt_event_id") or "") != event_id:
                continue
            strategy = str(match.get("strategy") or "")
            alert_key = match.get("alert_id")
            if alert_key is not None:
                direct_alerts.add((strategy, str(alert_key)))
        for alert in filtered_alerts:
            strategy = str(alert.get("strategy") or "C")
            if (strategy, str(alert.get("alert_id") or "")) in direct_alerts:
                model_id = alert.get("event_id")
                if model_id is not None:
                    model_events.add((strategy, str(model_id)))
        filtered_alerts = [
            alert
            for alert in filtered_alerts
            if (str(alert.get("strategy") or "C"), str(alert.get("alert_id") or ""))
            in direct_alerts
            or (
                alert.get("event_id") is not None
                and (str(alert.get("strategy") or "C"), str(alert.get("event_id"))) in model_events
            )
        ]

    figure = go.Figure()
    eligible_events: tuple[Any, ...] = ()
    if labels is not None and (selected_video is None or labels.video_id == selected_video):
        eligible_events, _, _ = prepare_ground_truth(labels)
        if event_id is not None:
            eligible_events = tuple(
                event for event in eligible_events if event.event_id == event_id
            )

    opportunity_lookup: dict[tuple[str, str], tuple[Any, Any]] = {}
    if event_id is not None and labels is not None and not eligible_events:
        figure.add_annotation(
            text="找不到可评价的该 Ground Truth 事件",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
    for event in eligible_events:
        for opportunity in event.alert_opportunities:
            opportunity_lookup[(event.event_id, opportunity.opportunity_id)] = (
                event,
                opportunity,
            )
            color = (
                "rgba(56, 189, 248, 0.28)"
                if opportunity.kind == "initial"
                else "rgba(192, 132, 252, 0.30)"
            )
            label = "首次机会窗口" if opportunity.kind == "initial" else "升级机会窗口"
            for lane_index, lane in enumerate(lane_names):
                figure.add_trace(
                    go.Scatter(
                        x=[opportunity.window_start_ms / 1000, opportunity.deadline_ms / 1000],
                        y=[lane, lane],
                        mode="lines",
                        line={"color": color, "width": 24},
                        name=label,
                        legendgroup=f"opportunity-{opportunity.kind}",
                        showlegend=lane_index == 0,
                        customdata=[
                            [
                                html.escape(event.event_id),
                                html.escape(opportunity.opportunity_id),
                                html.escape(opportunity.kind),
                                html.escape(opportunity.minimum_severity.value),
                                opportunity.window_start_ms / 1000,
                                opportunity.deadline_ms / 1000,
                            ]
                        ]
                        * 2,
                        hovertemplate=(
                            "真值事件：%{customdata[0]}<br>机会：%{customdata[1]}"
                            "（%{customdata[2]}）<br>最低等级：%{customdata[3]}<br>"
                            "有效窗口：%{customdata[4]:.3f}–%{customdata[5]:.3f} 秒"
                            "<extra>%{fullData.name}</extra>"
                        ),
                    )
                )

    grouped: dict[tuple[str, str, str], list[tuple[Mapping[str, Any], str, str, str]]] = (
        defaultdict(list)
    )
    for alert in filtered_alerts:
        strategy = str(alert.get("strategy") or "C")
        alert_key = str(alert.get("alert_id") or "")
        status, gt_event, match_reason = _record_status(
            match_rows_by_alert.get((strategy, alert_key), [])
        )
        alert_type = str(alert.get("alert_type") or "unknown")
        grouped[(strategy, alert_type, status)].append((alert, gt_event, match_reason, alert_key))

    for (strategy, alert_type, status), rows in sorted(grouped.items()):
        x_values, y_values, customdata = [], [], []
        for alert, gt_event, match_reason, alert_key in rows:
            timestamp_ms = _number(alert.get("timestamp_ms"))
            if timestamp_ms is None:
                continue
            x_values.append(timestamp_ms / 1000)
            y_values.append(strategy)
            risk_level = alert.get("risk_level", alert.get("severity"))
            score = _number(alert.get("risk_score"))
            reason = str(alert.get("reason") or "")
            if match_reason:
                reason = f"{reason}; {match_reason}" if reason else match_reason
            customdata.append(
                [
                    html.escape(str(alert.get("video_id", alert.get("stream_id", "")))),
                    html.escape(str(alert.get("event_id") or "")),
                    html.escape(alert_type),
                    html.escape(_STATUS_LABELS.get(status, status)),
                    html.escape(gt_event),
                    html.escape(str(risk_level or "N/A")),
                    "N/A" if score is None else f"{score:.3f}",
                    html.escape(reason),
                    html.escape(alert_key),
                ]
            )
        if not x_values:
            continue
        trace_name = (
            f"{strategy} · {_TYPE_LABELS.get(alert_type, alert_type)} · "
            f"{_STATUS_LABELS.get(status, status)}"
        )
        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="markers",
                name=trace_name,
                marker={
                    "symbol": _TYPE_SYMBOLS.get(alert_type, "circle-open"),
                    "color": _STATUS_COLORS.get(status, _STATUS_COLORS["unscored"]),
                    "size": 11,
                    "line": {"width": 1, "color": "#0f172a"},
                },
                customdata=customdata,
                hovertemplate=(
                    "视频：%{customdata[0]}<br>策略：%{y}<br>时间：%{x:.3f} 秒<br>"
                    "提醒：%{customdata[8]}<br>模型事件：%{customdata[1]}<br>"
                    "提醒类型：%{customdata[2]}<br>匹配状态：%{customdata[3]}<br>"
                    "真值事件：%{customdata[4]}<br>严重程度：%{customdata[5]}<br>"
                    "风险分数：%{customdata[6]}<br>原因：%{customdata[7]}"
                    "<extra></extra>"
                ),
            )
        )

    missed_rows = [
        match
        for match in filtered_matches
        if match.get("status") == "missed_opportunity"
        and (event_id is None or str(match.get("gt_event_id") or "") == event_id)
        and str(match.get("strategy") or "") in lane_names
    ]
    if missed_rows:
        x_values, y_values, customdata = [], [], []
        for match in missed_rows:
            gt_id = str(match.get("gt_event_id") or "")
            opportunity_id = str(match.get("opportunity_id") or "")
            pair = opportunity_lookup.get((gt_id, opportunity_id))
            if pair is None:
                continue
            event, opportunity = pair
            strategy = str(match.get("strategy") or "")
            x_values.append(opportunity.deadline_ms / 1000)
            y_values.append(strategy)
            customdata.append(
                [
                    html.escape(gt_id),
                    html.escape(opportunity_id),
                    html.escape(opportunity.kind),
                    html.escape(opportunity.minimum_severity.value),
                    opportunity.deadline_ms / 1000,
                    html.escape(str(match.get("reason") or "未在有效窗口内提醒")),
                    html.escape(event.event_type),
                ]
            )
        if x_values:
            figure.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode="markers",
                    name="漏提醒截止点",
                    marker={"symbol": "x", "color": "#f87171", "size": 13, "line": {"width": 2}},
                    customdata=customdata,
                    hovertemplate=(
                        "漏提醒截止：%{x:.3f} 秒<br>策略：%{y}<br>"
                        "真值事件：%{customdata[0]}（%{customdata[6]}）<br>"
                        "机会：%{customdata[1]}（%{customdata[2]}）<br>"
                        "最低等级：%{customdata[3]}<br>原因：%{customdata[5]}"
                        "<extra></extra>"
                    ),
                )
            )

    if not lane_names:
        message = "没有选择策略"
    elif not filtered_alerts and not eligible_events:
        message = "所选视频没有提醒或可显示的 Ground Truth 机会"
    elif not filtered_alerts:
        message = "所选视频没有策略提醒；真值窗口仍显示在对应策略行"
    else:
        message = None
    if message:
        figure.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 14, "color": "#cbd5e1"},
        )

    suffix = f" · 视频 {html.escape(selected_video)}" if selected_video else " · 全部视频"
    if event_id is not None:
        suffix += f" · Ground Truth 事件 {html.escape(event_id)}"
    figure.update_layout(
        template=_DARK,
        title=f"提醒时间线{suffix}",
        xaxis_title="视频时间（秒）",
        yaxis_title="策略",
        yaxis={"rangemode": "tozero"},
        yaxis_categoryorder="array",
        yaxis_categoryarray=list(lane_names),
        hovermode="closest",
        legend_title="窗口、提醒类型与匹配状态",
    )
    return figure


def metrics_plot(metrics: dict[str, dict], field: str) -> go.Figure:
    """Compare one metric while keeping missing values visibly distinct from zero."""
    strategy_names = [str(name) for name in metrics]
    if not strategy_names:
        return _empty_figure(
            f"策略指标比较：{html.escape(field)}",
            "没有可用的策略指标",
            x_title="策略",
            y_title=html.escape(field),
        )
    values: list[float | None] = []
    for strategy in strategy_names:
        raw = _dict(metrics.get(strategy)).get(field)
        values.append(_number(raw))
    figure = go.Figure(
        data=[
            go.Bar(
                x=strategy_names,
                y=values,
                text=["" if value is None else f"{value:.4g}" for value in values],
                textposition="outside",
                marker_color="#38bdf8",
                customdata=["N/A" if value is None else value for value in values],
                hovertemplate=(
                    "策略：%{x}<br>" + html.escape(field) + "：%{customdata}<extra></extra>"
                ),
            )
        ]
    )
    annotations = []
    for strategy, value in zip(strategy_names, values, strict=True):
        if value is None:
            annotations.append(
                {
                    "x": strategy,
                    "y": 0.035,
                    "xref": "x",
                    "yref": "paper",
                    "text": "N/A",
                    "showarrow": False,
                    "font": {"color": "#cbd5e1", "size": 12},
                    "bgcolor": "rgba(15, 23, 42, 0.8)",
                }
            )
    if any(value is None for value in values):
        all_missing = all(value is None for value in values)
        annotations.append(
            {
                "text": ("所选指标没有有效数值（N/A 不代表 0）" if all_missing else "N/A 不代表 0"),
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.75,
                "showarrow": False,
                "font": {"color": "#cbd5e1", "size": 14 if all_missing else 12},
            }
        )
    figure.update_layout(
        template=_DARK,
        title=f"策略指标比较：{html.escape(field)}",
        xaxis_title="策略",
        yaxis_title=html.escape(field),
        annotations=annotations,
        hovermode="closest",
    )
    return figure
