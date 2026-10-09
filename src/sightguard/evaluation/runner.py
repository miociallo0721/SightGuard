"""Fail-visible, reproducible offline benchmark orchestration."""

import logging
import platform
import time
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

from sightguard import __version__
from sightguard.evaluation.config import BenchmarkConfig
from sightguard.evaluation.dataset import load_dataset, sha256_file
from sightguard.evaluation.matching import evaluate_video
from sightguard.evaluation.metrics import aggregate_metrics
from sightguard.evaluation.protocol import check_protocol, code_fingerprint
from sightguard.evaluation.reporting import flat_metrics, render_figures, write_csv, write_json
from sightguard.evaluation.strategies import run_strategies
from sightguard.risk.config import RiskConfig

logger = logging.getLogger(__name__)


def benchmark(
    dataset: Path,
    output: Path,
    config: BenchmarkConfig,
    risk_config: RiskConfig,
    *,
    split: str = "test",
    protocol: Path | None = None,
    data_kind: str | None = None,
    strategies: tuple[str, ...] = ("A", "B", "C"),
) -> dict[str, Any]:
    if split not in ("development", "test") or data_kind not in (None, "real", "synthetic"):
        raise ValueError("Invalid split/data kind")
    if (
        not strategies
        or len(set(strategies)) != len(strategies)
        or set(strategies) - {"A", "B", "C"}
    ):
        raise ValueError("Choose unique strategies A, B, C")
    strategies = tuple(sorted(strategies))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "schema_version": "3.0",
        "strategies": strategies,
        "status": "running",
        "split": split,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "dataset": str(dataset.resolve()),
        "benchmark_config": config.to_dict(),
        "risk_config": risk_config.to_dict(),
        "versions": {
            "sightguard": __version__,
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "matplotlib": version("matplotlib"),
        },
        "source_sha256": code_fingerprint(),
        "timing_kind": "offline_replay_wall_clock_not_inference_or_mobile_latency",
        "mobile_latency_ms": None,
        "confidence_intervals": {
            "status": "not_estimated",
            "reason": "No independent corpus bootstrap implemented",
        },
    }
    started = time.perf_counter()
    stage = "dataset_validation"
    current_video = current_strategy = None
    try:
        videos = load_dataset(dataset, check_videos=True)
        metadata["manifest_sha256"] = sha256_file(dataset)
        lock = Path(__file__).resolve().parents[3] / "uv.lock"
        metadata["uv_lock_sha256"] = sha256_file(lock) if lock.is_file() else None
        videos = tuple(
            video
            for video in videos
            if video.entry.split == split
            and (data_kind is None or video.entry.data_kind == data_kind)
        )
        if not videos:
            raise ValueError("No eligible videos in selected split/data kind")
        kinds = {video.entry.data_kind for video in videos}
        if len(kinds) > 1:
            raise ValueError("Report real and synthetic data separately; select --data-kind")
        metadata["data_kind"] = next(iter(kinds))
        stage = "protocol_validation"
        if split == "test" and protocol is None:
            raise ValueError(
                "Test evaluation requires a frozen protocol; run freeze-protocol first"
            )
        if protocol is not None:
            check_protocol(protocol, dataset, config, risk_config)
            metadata["protocol_sha256"] = sha256_file(protocol)
        metadata["inputs"] = [
            {
                **asdict(video.entry),
                "video_path": str(video.entry.video_path) if video.entry.video_path else None,
                "tracks_path": str(video.entry.tracks_path),
                "annotations_path": str(video.entry.annotations_path),
                "strategy_input_sha256": dict.fromkeys(strategies, video.entry.tracks_sha256),
                "perception_metadata_sha256": sha256_file(
                    video.entry.tracks_path.parent / "run_metadata.json"
                )
                if video.entry.data_kind == "real"
                else None,
            }
            for video in videos
        ]
        metadata["video_count"] = len(videos)
        metadata["source_group_count"] = len({v.entry.source_group_id for v in videos})
        results, all_alerts = [], []
        with (
            (output / "alerts.jsonl.partial").open("w", encoding="utf-8") as alert_stream,
            (output / "risk_frames.jsonl.partial").open("w", encoding="utf-8") as risk_stream,
            (output / "events.jsonl.partial").open("w", encoding="utf-8") as event_stream,
        ):
            for video in videos:
                current_video, current_strategy = video.entry.video_id, "shared_risk_and_policies"
                logger.info(
                    "Benchmark %s: %d perception frames", video.entry.video_id, len(video.records)
                )
                stage = "strategy_execution"
                run = run_strategies(
                    video.records, video.entry.video_id, risk_config, config, strategies
                )
                for frame in run.risk_frames:
                    risk_stream.write(frame.to_json() + "\n")
                for event in run.changes:
                    event_stream.write(event.to_json() + "\n")
                for name, alerts in run.alerts.items():
                    current_strategy = name
                    for alert in alerts:
                        alert_stream.write(alert.to_json() + "\n")
                    all_alerts.extend(alerts)
                    stage = "evaluation"
                    results.append(
                        evaluate_video(video.labels, alerts, config.matching, strategy=name)
                    )
        stage = "reporting"
        comparison = {
            name: aggregate_metrics(row for row in results if row.strategy == name)
            for name in strategies
        }
        write_json(output / "metrics.json", comparison)
        write_csv(
            output / "comparison.csv",
            [{"strategy": name, **flat_metrics(values)} for name, values in comparison.items()],
        )
        write_csv(
            output / "per_video.csv",
            [
                {"video_id": row.video_id, "strategy": row.strategy, **flat_metrics(row.metrics)}
                for row in results
            ],
        )
        scenes = {video.entry.video_id: video.entry.scene for video in videos}
        scene_rows = [
            {
                "scene": scene,
                "strategy": name,
                **flat_metrics(
                    aggregate_metrics(
                        row
                        for row in results
                        if row.strategy == name and scenes[row.video_id] == scene
                    )
                ),
            }
            for scene in sorted(set(scenes.values()))
            for name in comparison
        ]
        write_csv(output / "per_scene.csv", scene_rows)
        severity_rows = []
        for name in comparison:
            for severity in ("LOW", "MEDIUM", "HIGH"):
                events = [
                    event
                    for row in results
                    if row.strategy == name
                    for event in row.event_rows
                    if event["severity"] == severity and event["eligible"]
                ]
                important = [event for event in events if event["important"]]
                hits = sum(event["detected"] for event in important)
                severity_rows.append(
                    {
                        "strategy": name,
                        "severity": severity,
                        "eligible_events": len(events),
                        "important_events": len(important),
                        "detected_events": hits,
                        "missed_events": len(important) - hits,
                        "event_recall": hits / len(important) if important else None,
                    }
                )
        write_csv(output / "per_severity.csv", severity_rows)
        write_csv(
            output / "event_results.csv",
            [
                {"video_id": row.video_id, "strategy": row.strategy, **event}
                for row in results
                for event in row.event_rows
            ],
        )
        event_types = sorted(
            {event.event_type for video in videos for event in video.labels.events}
        )
        type_rows = []
        for name in comparison:
            for event_type in event_types:
                events = [
                    event
                    for row in results
                    if row.strategy == name
                    for event in row.event_rows
                    if event["event_type"] == event_type
                    and event["eligible"]
                    and event["important"]
                ]
                hits = sum(event["detected"] for event in events)
                type_rows.append(
                    {
                        "strategy": name,
                        "event_type": event_type,
                        "important_events": len(events),
                        "detected_events": hits,
                        "missed_events": len(events) - hits,
                        "event_recall": hits / len(events) if events else None,
                    }
                )
        write_csv(
            output / "per_event_type.csv",
            type_rows,
            [
                "strategy",
                "event_type",
                "important_events",
                "detected_events",
                "missed_events",
                "event_recall",
            ],
        )
        with (output / "matches.jsonl").open("w", encoding="utf-8") as stream:
            for row in results:
                for match in row.matches:
                    stream.write(match.to_json() + "\n")
        errors = [
            asdict(match)
            for row in results
            for match in row.matches
            if match.status in ("false_positive", "duplicate", "missed_opportunity")
        ]
        opportunity_details = {
            (video.entry.video_id, event.event_id, opportunity.opportunity_id): {
                "window_start_ms": opportunity.window_start_ms,
                "deadline_ms": opportunity.deadline_ms,
                "minimum_severity": opportunity.minimum_severity.value,
                "event_type": event.event_type,
            }
            for video in videos
            for event in video.labels.events
            for opportunity in event.alert_opportunities
        }
        for error in errors:
            error.update(
                opportunity_details.get(
                    (error["video_id"], error["gt_event_id"], error["opportunity_id"]), {}
                )
            )
        write_csv(
            output / "error_cases.csv",
            errors,
            [
                "video_id",
                "strategy",
                "alert_id",
                "timestamp_ms",
                "status",
                "gt_event_id",
                "opportunity_id",
                "opportunity_kind",
                "latency_ms",
                "reason",
                "schema_version",
                "window_start_ms",
                "deadline_ms",
                "minimum_severity",
                "event_type",
            ],
        )
        if config.generate_figures:
            render_figures(output, videos, tuple(all_alerts), tuple(results), comparison)
        lines = [
            "# SightGuard Phase 3 离线实验",
            "",
            f"数据类型：{metadata['data_kind']}；划分：{split}；视频数：{len(videos)}。",
            "",
            "合成数据仅验证算法与评价流程，不能代表真实道路效果或安全性。",
            "",
            "A：固定类别/关注区逐帧提醒；B：相同风险评分加全局固定冷却；"
            "C：Phase 2 原生自适应决策。",
            "",
            "重复提醒包含在 FP 中；无效/小时按有效视频时长归一化，"
            "短视频的该比率不能外推为实际一小时体验。",
            "",
        ]
        for name, values in comparison.items():
            lines.append(
                f"- {name}：TP={values['tp']}，FP={values['fp']}，FN={values['fn']}；"
                f"事件召回={values['event_recall']}，机会召回={values['opportunity_recall']}；"
                f"重复={values['duplicates']}，未匹配原因见 error_cases.csv。"
            )
        lines += [
            "",
            "首提醒延迟只统计命中事件；遗漏事件另列，不以零延迟填补。",
            "本报告未估计统计置信区间；帧不是独立实验样本。",
            "元数据中的运行耗时是离线回放墙钟时间，不是端侧推理/提醒延迟。",
            "完整计数、分场景/严重程度、逐提醒归因、时间线与输入哈希见同目录文件。",
            "",
        ]
        (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
        for name in ("alerts", "risk_frames", "events"):
            (output / f"{name}.jsonl.partial").replace(output / f"{name}.jsonl")
        metadata.update(
            status="completed",
            artifacts_complete=True,
            raw_generated_alerts=len(all_alerts),
            offline_replay_wall_seconds=time.perf_counter() - started,
        )
        write_json(output / "run_metadata.json", metadata)
        return comparison
    except Exception as exc:
        metadata.update(
            status="failed",
            artifacts_complete=False,
            failure_stage=stage,
            failure_video_id=current_video,
            failure_strategy=current_strategy,
            error={"type": type(exc).__name__, "message": str(exc)},
            offline_replay_wall_seconds=time.perf_counter() - started,
        )
        write_json(output / "run_metadata.json", metadata)
        raise
