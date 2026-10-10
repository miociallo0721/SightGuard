"""Pandas tables and escaped HTML presentation helpers for SightGuard UI."""

from __future__ import annotations

import html
import json
import math
from collections.abc import Mapping
from numbers import Real
from typing import Any

import pandas as pd

ALERT_COLUMNS = (
    "video_id",
    "strategy",
    "alert_id",
    "event_id",
    "timestamp_ms",
    "risk_level",
    "risk_score",
    "alert_type",
    "class_id",
    "source_track_ids",
    "frame_index",
    "reason",
)


def metrics_frame(metrics: dict[str, dict]) -> pd.DataFrame:
    """Return strategy rows while preserving source values and missing values."""
    columns = list(dict.fromkeys(key for row in metrics.values() for key in row))
    frame = pd.DataFrame(index=list(metrics), columns=columns, dtype=object)
    frame.index.name = "strategy"
    for strategy, row in metrics.items():
        for key, value in row.items():
            frame.at[strategy, key] = value
    return frame


def format_metric(value: Any, percent: bool = False) -> str:
    """Format a finite metric; reject bools and malformed values rather than coercing them."""
    if value is None:
        return "N/A"
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("Metric must be a real number or None")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("Metric must be finite")
    return f"{numeric:.1%}" if percent else f"{numeric:,.4g}"


def _display_value(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, Real):
        try:
            return format_metric(value)
        except ValueError:
            return "N/A"
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(value, ensure_ascii=False, allow_nan=False, default=str)
    return str(value)


def kpi_html(stats: dict, title: str = "运行统计") -> str:
    """Render statistic cards with all caller-provided text escaped."""
    cards = []
    for label, value in stats.items():
        cards.append(
            '<div class="sg-kpi">'
            f'<div class="sg-kpi-label">{html.escape(str(label))}</div>'
            '<div class="sg-kpi-value">'
            f"{html.escape(_display_value(value))}</div></div>"
        )
    if not cards:
        cards.append('<div class="sg-kpi-empty">选择一份记录，查看实际运行统计</div>')
    return (
        '<section class="sg-kpi-section">'
        f"<h3>{html.escape(title)}</h3>"
        '<div class="sg-kpi-grid">' + "".join(cards) + "</div></section>"
    )


def experiment_summary(metadata: dict, metrics: dict[str, dict]) -> str:
    """Create an escaped experiment overview without inferring zero from missing data."""
    raw_kind = str(metadata.get("data_kind") or "unknown").lower()
    kind_labels = {
        "synthetic": "Synthetic Data / 合成数据",
        "real": "Real Data / 真实视频数据",
    }
    kind_label = kind_labels.get(raw_kind, "数据类型未知")
    badge_color = "#7c3aed" if raw_kind == "synthetic" else "#0369a1"

    inputs = metadata.get("inputs")
    input_rows = inputs if isinstance(inputs, list) else []
    derived_video_count = len(input_rows)
    declared_count = metadata.get("video_count")
    video_count = (
        derived_video_count
        if isinstance(inputs, list)
        else (
            declared_count
            if isinstance(declared_count, int)
            and not isinstance(declared_count, bool)
            and declared_count >= 0
            else None
        )
    )
    durations = [
        number
        for item in input_rows
        if (number := _finite_positive(_mapping(item).get("duration_ms"))) is not None
    ]
    total_duration_ms = sum(durations) if durations else None
    duration_text = "N/A" if total_duration_ms is None else f"{total_duration_ms / 1000:,.4g} 秒"
    source_groups = metadata.get("source_group_count")
    if not isinstance(source_groups, int) or isinstance(source_groups, bool) or source_groups < 0:
        source_groups_text = "N/A"
    else:
        source_groups_text = str(source_groups)

    rows = []
    for strategy, values in metrics.items():
        values = _mapping(values)
        event_recall = _format_or_na(values.get("event_recall"), percent=True)
        opportunity_recall = _format_or_na(values.get("opportunity_recall"), percent=True)
        tp = _format_or_na(values.get("tp"))
        fp = _format_or_na(values.get("fp"))
        fn = _format_or_na(values.get("fn"))
        first_delay = _format_or_na(values.get("first_response_delay_mean_ms"))
        invalid_rate = _format_or_na(values.get("invalid_alerts_per_hour"))
        duplicate_rate = _format_or_na(values.get("duplicate_rate"), percent=True)
        rows.append(
            "<tr>"
            f"<th>{html.escape(str(strategy))}</th>"
            f"<td>{html.escape(event_recall)}</td>"
            f"<td>{html.escape(opportunity_recall)}</td>"
            f"<td>{html.escape(tp)}</td>"
            f"<td>{html.escape(fp)}</td>"
            f"<td>{html.escape(fn)}</td>"
            f"<td>{html.escape(first_delay)}</td>"
            f"<td>{html.escape(invalid_rate)}</td>"
            f"<td>{html.escape(duplicate_rate)}</td>"
            "</tr>"
        )
    if rows:
        table = (
            '<div style="width:100%;overflow-x:auto;margin-top:12px">'
            '<table style="border-collapse:collapse;width:100%;min-width:820px;'
            'color:#e2e8f0;text-align:right">'
            '<thead><tr style="color:#94a3b8;border-bottom:1px solid #475569">'
            '<th style="padding:8px;text-align:left">策略</th>'
            '<th style="padding:8px">事件召回</th>'
            '<th style="padding:8px">机会召回</th>'
            '<th style="padding:8px">TP</th><th style="padding:8px">FP</th>'
            '<th style="padding:8px">FN</th>'
            '<th style="padding:8px">首次响应均值（ms）</th>'
            '<th style="padding:8px">无效提醒/小时</th>'
            '<th style="padding:8px">重复率</th>'
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
        )
    else:
        table = '<p style="color:#cbd5e1">暂无策略指标；缺失指标不会按 0 处理。</p>'

    if raw_kind == "synthetic":
        warning = "合成数据仅用于验证流程和评价器，不代表真实环境性能。"
    else:
        warning = "研究指标不构成道路安全保证，也不代表碰撞概率。"
    warning += " 离线重放墙钟耗时不是移动端响应延迟。"
    facts = {
        "划分": metadata.get("split"),
        "运行状态": metadata.get("status"),
        "视频数": video_count,
        "独立来源组": source_groups_text,
        "输入总时长": duration_text,
    }
    fact_html = "".join(
        '<div style="min-width:120px"><span style="color:#94a3b8">'
        f"{html.escape(label)}</span><br><strong>{html.escape(_display_value(value))}</strong></div>"
        for label, value in facts.items()
    )
    return (
        '<section class="sg-experiment-summary" '
        'style="font-family:system-ui,sans-serif;color:#f8fafc;'
        'background:#0f172a;border:1px solid #334155;border-radius:12px;padding:16px">'
        f'<span style="display:inline-block;background:{badge_color};color:white;'
        f'padding:4px 10px;border-radius:999px">{html.escape(kind_label)}</span>'
        f'<div style="display:flex;flex-wrap:wrap;gap:20px;margin-top:14px">{fact_html}</div>'
        f"{table}"
        '<p style="background:#1e293b;color:#fde68a;padding:10px;border-radius:8px;'
        f'margin-bottom:0">{html.escape(warning)}</p>'
        "</section>"
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _finite_positive(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    number = float(value)
    return number if math.isfinite(number) and number > 0 else None


def _format_or_na(value: Any, *, percent: bool = False) -> str:
    try:
        return format_metric(value, percent=percent)
    except (TypeError, ValueError):
        return "N/A"


def alert_frame(alerts: tuple[dict, ...]) -> pd.DataFrame:
    """Convert strategy or Phase 2 alert records into a fixed-column table."""
    rows = []
    for alert in alerts:
        alert = _mapping(alert)
        rows.append(
            {
                "video_id": alert.get("video_id", alert.get("stream_id")),
                "strategy": alert.get("strategy"),
                "alert_id": alert.get("alert_id"),
                "event_id": alert.get("event_id"),
                "timestamp_ms": alert.get("timestamp_ms"),
                "risk_level": alert.get("risk_level", alert.get("severity")),
                "risk_score": alert.get("risk_score"),
                "alert_type": alert.get("alert_type", alert.get("type")),
                "class_id": alert.get("class_id"),
                "source_track_ids": alert.get("source_track_ids"),
                "frame_index": alert.get("frame_index"),
                "reason": alert.get("reason"),
            }
        )
    return pd.DataFrame(rows, columns=ALERT_COLUMNS)
