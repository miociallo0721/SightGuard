import math

import pytest

from sightguard.evaluation.schema import AlertOpportunity, LabeledEvent, VideoAnnotations
from sightguard.ui.plots import metrics_plot, risk_plot, timeline_plot
from sightguard.ui.presentation import (
    ALERT_COLUMNS,
    alert_frame,
    experiment_summary,
    format_metric,
    kpi_html,
    metrics_frame,
)


def _labels() -> VideoAnnotations:
    event = LabeledEvent(
        video_id="clip-1",
        event_id="gt-person-1",
        event_type="forward_person",
        start_ms=0,
        end_ms=1800,
        severity="HIGH",
        annotation_status="confirmed",
        alert_opportunities=(
            AlertOpportunity("initial", "initial", 0, 500, "MEDIUM"),
            AlertOpportunity("upgrade", "escalation", 1000, 1500, "HIGH"),
        ),
        annotator_id="annotator-1",
        notes="synthetic test labels",
        class_ids=(0,),
    )
    return VideoAnnotations(
        video_id="clip-1",
        duration_ms=2000,
        data_kind="synthetic",
        annotation_status="reviewed",
        annotator_id="annotator-1",
        label_source="synthetic_scenario",
        source_video_sha256="",
        notes="unit test",
        events=(event,),
    )


def _alert(
    alert_id: str,
    strategy: str,
    event_id: str,
    timestamp_ms: float,
    alert_type: str = "trigger",
    video_id: str = "clip-1",
) -> dict:
    return {
        "video_id": video_id,
        "strategy": strategy,
        "alert_id": alert_id,
        "event_id": event_id,
        "timestamp_ms": timestamp_ms,
        "risk_level": "HIGH",
        "risk_score": 0.7,
        "alert_type": alert_type,
        "reason": "test <reason>",
        "source_track_ids": [1],
        "class_id": 0,
        "bbox_normalized": [0.4, 0.4, 0.6, 0.8],
        "frame_index": int(timestamp_ms // 250),
    }


def _match(
    strategy: str,
    alert_id: str | None,
    status: str,
    *,
    gt_event_id: str | None = "gt-person-1",
    opportunity_id: str | None = "initial",
    timestamp_ms: float | None = None,
) -> dict:
    return {
        "video_id": "clip-1",
        "strategy": strategy,
        "alert_id": alert_id,
        "timestamp_ms": timestamp_ms,
        "status": status,
        "gt_event_id": gt_event_id,
        "opportunity_id": opportunity_id,
        "opportunity_kind": "initial",
        "latency_ms": None,
        "reason": f"reason <{status}>",
    }


def test_timeline_uses_gt_matches_then_shows_each_strategys_full_event_behavior() -> None:
    alerts = (
        _alert("a-first", "A", "D-frame-1", 0),
        _alert("a-repeat", "A", "D-frame-2", 250),
        _alert("b-first", "B", "E000001", 500),
        _alert("c-first", "C", "E000009", 250, "first"),
        _alert("c-upgrade", "C", "E000009", 1250, "escalation"),
        # No direct GT row; this is included only after a C-to-GT match anchors E000009.
        _alert("c-later", "C", "E000009", 1400, "repeat"),
        _alert("other-video", "A", "other", 250, video_id="clip-2"),
    )
    matches = (
        _match("A", "a-first", "matched", timestamp_ms=0),
        _match("A", "a-repeat", "duplicate", opportunity_id=None, timestamp_ms=250),
        _match("B", "b-first", "matched", timestamp_ms=500),
        _match("B", None, "missed_opportunity", opportunity_id="upgrade"),
        _match("C", "c-first", "matched", timestamp_ms=250),
        _match("C", "c-upgrade", "matched", opportunity_id="upgrade", timestamp_ms=1250),
    )

    figure = timeline_plot(
        alerts,
        matches,
        _labels(),
        strategies=("A", "B", "C"),
        event_id="gt-person-1",
    )

    assert figure.layout.template.layout.paper_bgcolor is not None
    assert "Ground Truth 事件 gt-person-1" in figure.layout.title.text
    opportunity_traces = [trace for trace in figure.data if "机会窗口" in trace.name]
    assert len(opportunity_traces) == 6
    assert {trace.y[0] for trace in opportunity_traces} == {"A", "B", "C"}
    marker_traces = [trace for trace in figure.data if trace.mode == "markers"]
    assert any("A · 基线触发 · 已匹配" == trace.name for trace in marker_traces)
    assert any("A · 基线触发 · 重复提醒" == trace.name for trace in marker_traces)
    c_trace = next(trace for trace in marker_traces if "C · 重复 · 未评分/诊断" == trace.name)
    assert list(c_trace.x) == [1.4]
    missed = next(trace for trace in marker_traces if trace.name == "漏提醒截止点")
    assert list(missed.x) == [1.5]
    assert list(missed.y) == ["B"]
    all_alert_times = [x for trace in marker_traces for x in (trace.x or ())]
    assert 0.25 in all_alert_times
    assert all(row[0] != "clip-2" for trace in marker_traces for row in (trace.customdata or ()))


def test_timeline_accepts_phase2_stream_id_and_empty_inputs() -> None:
    figure = timeline_plot(
        ({"stream_id": "camera-1", "event_id": "E1", "timestamp_ms": 1000},),
        strategies=("C",),
        video_id="camera-1",
    )
    assert any(trace.x == (1.0,) for trace in figure.data if trace.mode == "markers")

    empty = timeline_plot((), strategies=("A", "B", "C"))
    assert any("没有提醒" in annotation.text for annotation in empty.layout.annotations)


def test_risk_plot_uses_observations_and_displays_score_level_and_quality() -> None:
    frames = (
        {
            "frame_index": 1,
            "timestamp_ms": 250,
            "observations": [
                {
                    "event_id": "E1",
                    "class_name": "person",
                    "features": {"quality": {"score": 0.8}},
                    "assessment": {"score": 0.42, "level": "MEDIUM", "quality": {"score": 0.8}},
                }
            ],
        },
        {"timestamp_ms": 500, "observations": [{"event_id": "E2", "assessment": None}]},
    )
    figure = risk_plot(frames)

    assert figure.layout.yaxis.range == (0, 1)
    assert figure.layout.xaxis.title.text == "视频时间（秒）"
    assert len(figure.data) == 1
    assert figure.data[0].name == "E1"
    assert list(figure.data[0].x) == [0.25]
    assert list(figure.data[0].y) == [0.42]
    assert "感知质量" in figure.data[0].hovertemplate

    empty = risk_plot(())
    assert any("没有可绘制" in annotation.text for annotation in empty.layout.annotations)


def test_metrics_plot_keeps_null_distinct_from_zero_and_does_not_invent_values() -> None:
    figure = metrics_plot(
        {"A": {"event_recall": None}, "B": {"event_recall": 0}, "C": {"event_recall": 0.5}},
        "event_recall",
    )

    assert list(figure.data[0].y) == [None, 0.0, 0.5]
    assert any(
        annotation.text == "N/A" and annotation.x == "A" for annotation in figure.layout.annotations
    )
    assert any("N/A 不代表 0" in annotation.text for annotation in figure.layout.annotations)
    assert figure.layout.template.layout.paper_bgcolor is not None

    empty = metrics_plot({}, "event_recall")
    assert any("没有可用" in annotation.text for annotation in empty.layout.annotations)


def test_presentation_preserves_values_formats_missing_and_escapes_html() -> None:
    precise = 0.123456789
    frame = metrics_frame({"A": {"recall": precise, "missing": None}, "B": {"recall": 0}})
    assert frame.at["A", "recall"] == precise
    assert frame.at["A", "missing"] is None
    assert frame.at["B", "recall"] == 0
    assert format_metric(None) == "N/A"
    assert format_metric(0.125, percent=True) == "12.5%"
    with pytest.raises(TypeError):
        format_metric(True)
    with pytest.raises(TypeError):
        format_metric("0.5")
    with pytest.raises(ValueError):
        format_metric(math.inf)

    card = kpi_html({"<label>": "<script>alert(1)</script>"}, title="<img src=x>")
    assert "&lt;script&gt;" in card
    assert "&lt;img" in card
    assert "<script>" not in card


def test_experiment_summary_distinguishes_synthetic_and_does_not_infer_zero_fp() -> None:
    summary = experiment_summary(
        {
            "data_kind": "synthetic",
            "split": "test",
            "video_count": 99,
            "source_group_count": 1,
            "inputs": [{"duration_ms": 2000}],
        },
        {},
    )
    assert "Synthetic Data / 合成数据" in summary
    assert "合成数据" in summary
    assert "合成数据仅用于验证流程" in summary
    assert "暂无策略指标" in summary
    assert "FP" not in summary
    assert "0%" not in summary
    assert "2 秒" in summary
    assert ">99</strong>" not in summary

    real = experiment_summary(
        {"data_kind": "real", "split": "development", "inputs": []},
        {"A": {"event_recall": None, "invalid_alerts_per_hour": None}},
    )
    assert "Real Data / 真实视频数据" in real
    assert "真实视频数据" in real
    assert "N/A" in real
    assert "道路安全保证" in real


def test_experiment_summary_shows_two_recalls_and_uses_source_counts() -> None:
    summary = experiment_summary(
        {"data_kind": "real", "split": "development", "inputs": [{"duration_ms": 1000}]},
        {
            "A": {
                "event_recall": 0.5,
                "opportunity_recall": None,
                "tp": 7,
                "fp": 2,
                "fn": 3,
                "first_response_delay_mean_ms": None,
                "invalid_alerts_per_hour": 2.75,
                "duplicate_rate": 0.25,
            }
        },
    )

    assert "事件召回" in summary and "机会召回" in summary
    assert "首次响应均值（ms）" in summary
    assert "<td>50.0%</td><td>N/A</td><td>7</td><td>2</td><td>3</td><td>N/A</td>" in summary
    assert "2.75" in summary
    assert "25.0%" in summary


def test_alert_frame_has_stable_columns_for_empty_and_phase2_records() -> None:
    assert tuple(alert_frame(()).columns) == ALERT_COLUMNS
    frame = alert_frame(
        (
            {
                "stream_id": "camera-1",
                "alert_id": "A1",
                "event_id": "E1",
                "timestamp_ms": 100,
                "risk_level": "HIGH",
                "alert_type": "first",
                "reason": "example",
            },
        )
    )
    assert tuple(frame.columns) == ALERT_COLUMNS
    assert frame.iloc[0]["video_id"] == "camera-1"
    assert frame.iloc[0]["event_id"] == "E1"
    assert frame.iloc[0]["timestamp_ms"] == 100
