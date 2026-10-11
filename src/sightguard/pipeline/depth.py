"""Independent offline depth study aligned to existing replay and scene records."""

from __future__ import annotations

import importlib.metadata
import json
import math
import platform
import re
import subprocess
import time
from collections import Counter
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from sightguard.perception.depth.artifacts import save_prediction, sha256_file, write_json
from sightguard.perception.depth.model import DepthEstimator, DepthModelUnavailableError
from sightguard.perception.depth.schema import SCHEMA_VERSION, DepthModelConfig


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def percentiles(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "p50": float(np.percentile(values, 50)) if values else None,
        "p95": float(np.percentile(values, 95)) if values else None,
    }


def sample_indices(count: int, fps: float, interval_seconds: float) -> list[int]:
    """Same index-stride rule as PR v04, rejecting non-representable intervals."""
    if count < 1 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("invalid video count/FPS")
    if not math.isfinite(interval_seconds) or interval_seconds <= 0:
        raise ValueError("sample interval must be finite and positive")
    step = fps * interval_seconds
    if step < 1 or abs(step - round(step)) > 1e-6:
        raise ValueError("sample interval must correspond to an integer source frame stride")
    return list(range(0, count, round(step)))


def _versions() -> dict[str, str]:
    values = {"python": platform.python_version()}
    for name in ("torch", "transformers", "numpy", "opencv-python", "safetensors", "Pillow"):
        try:
            values[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            values[name] = "not installed"
    return values


def _gpu_snapshot() -> dict[str, str]:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader"],
            text=True,
            timeout=5,
        ).strip()
        return {"device_snapshot": output, "scope": "whole device, not process allocation"}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"unavailable_reason": str(exc)}


def _process_video(
    row: dict[str, Any],
    baseline_dir: Path,
    scene_dir: Path,
    output_dir: Path,
    estimator: DepthEstimator,
    interval_seconds: float,
    warmup_frames: int,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    video_id = row["video_id"]
    directory = output_dir / video_id
    directory.mkdir()
    summary: dict[str, Any] = {
        "video_id": video_id,
        "status": "running",
        "sampled_frames": 0,
        "valid_frames": 0,
        "unknown_frames": 0,
        "error": None,
    }
    capture = None
    results = []
    try:
        path = Path(row["filepath"])
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256_file(path)
        if digest != row["processed_sha256"]:
            raise ValueError("video SHA-256 differs from dataset manifest")
        track_path = baseline_dir / video_id / "perception/tracks.jsonl"
        risk_path = baseline_dir / video_id / "assessment/risk_frames.jsonl"
        alert_path = baseline_dir / video_id / "assessment/alerts.jsonl"
        risk_metadata_path = baseline_dir / video_id / "assessment/run_metadata.json"
        scene_path = scene_dir / video_id / "scene_frames.jsonl"
        tracks = read_jsonl(track_path)
        risks = read_jsonl(risk_path)
        scenes = read_jsonl(scene_path)
        risk_metadata = json.loads(risk_metadata_path.read_text(encoding="utf-8"))
        stream_id = risk_metadata["stream_id"]
        summary["provenance"] = {
            "video": {"path": str(path), "sha256": digest},
            "parent_source_id": row.get("parent_source_id"),
            "source_url": row.get("source_url"),
            "license": row.get("license"),
            "split": row.get("split"),
            "selected_start_time": row.get("selected_start_time"),
            "sensor_capture_time_mapping": row.get("sensor_capture_time_mapping"),
            "references": {
                name: {"path": str(p), "sha256": sha256_file(p)}
                for name, p in (
                    ("tracks", track_path),
                    ("risk_frames", risk_path),
                    ("alerts", alert_path),
                    ("path_scene_frames", scene_path),
                    ("risk_metadata", risk_metadata_path),
                )
            },
        }
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise OSError("video could not be opened")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = (
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        if abs(fps - row["fps"]) > 1e-6 or (width, height) != (row["width"], row["height"]):
            raise ValueError("decoded container metadata differs from manifest")
        if len(tracks) != count or len(risks) != count:
            raise ValueError("video/track/risk frame counts differ")
        indices = sample_indices(count, fps, interval_seconds)
        if [s["frame_index"] for s in scenes] != indices:
            raise ValueError("sample indices differ from Path Relevance v04")
        for i, (track, risk) in enumerate(zip(tracks, risks, strict=True)):
            t = track["timestamp_ms"]
            if (
                track["frame_index"] != i
                or risk["frame_index"] != i
                or type(t) not in (int, float)
                or not math.isfinite(t)
                or t < 0
                or risk["timestamp_ms"] != t
                or (i > 0 and t <= tracks[i - 1]["timestamp_ms"])
                or (track["frame_width"], track["frame_height"]) != (width, height)
            ):
                raise ValueError(f"invalid replay identity/time/size at frame {i}")
        for scene in scenes:
            if (
                scene["video_id"] != video_id
                or scene["timestamp_ms"] != tracks[scene["frame_index"]]["timestamp_ms"]
            ):
                raise ValueError("Path Relevance time/video identity differs")
        selected = set(indices)
        summary["container"] = {"fps": fps, "frame_count": count, "width": width, "height": height}
        summary["sampling_alignment"] = "exact indices and timestamps match tracks/risk/PR v04"
        with (
            (directory / "depth_frames.jsonl").open("x", encoding="utf-8") as stream,
            (directory / "alignment.jsonl").open("x", encoding="utf-8") as alignment,
        ):
            for index in range(count):
                frame_started = time.perf_counter()
                ok, frame = capture.read()
                decode_ms = (time.perf_counter() - frame_started) * 1000
                if not ok or frame is None:
                    if index in selected:
                        track = tracks[index]
                        prediction = estimator.infer(
                            None,
                            video_id=video_id,
                            frame_index=index,
                            timestamp_ms=track["timestamp_ms"],
                            source_width=width,
                            source_height=height,
                            timestamp_source=track["timestamp_source"],
                            stream_id=stream_id,
                        )
                        stream.write(
                            json.dumps(prediction.result.to_dict(), allow_nan=False) + "\n"
                        )
                        results.append(prediction.result.to_dict())
                    raise OSError(f"decode failure at frame {index}/{count}")
                if frame.shape[:2] != (height, width):
                    raise ValueError(f"decoded frame dimensions changed at {index}")
                if index not in selected:
                    continue
                track = tracks[index]
                pts_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
                if track["timestamp_source"] == "video_pts" and (
                    not math.isfinite(pts_ms) or abs(pts_ms - track["timestamp_ms"]) > 1
                ):
                    raise ValueError(f"decoded PTS differs from existing replay at {index}")
                context = dict(
                    video_id=video_id,
                    frame_index=index,
                    timestamp_ms=track["timestamp_ms"],
                    timestamp_source=track["timestamp_source"],
                    source_width=width,
                    source_height=height,
                    stream_id=stream_id,
                )
                if not runtime.get("warmup_done") and estimator.model is not None:
                    for _ in range(warmup_frames):
                        warmup = estimator.infer(frame, **context)
                        runtime["warmup"].append(warmup.result.to_dict())
                    runtime["warmup_done"] = True
                processing_started = time.perf_counter()
                prediction = estimator.infer(frame, **context)
                result = save_prediction(directory, prediction, frame)
                result = replace(
                    result,
                    timings_ms={
                        **result.timings_ms,
                        "decode_selected_frame": decode_ms,
                        "full_sample": (time.perf_counter() - processing_started) * 1000
                        + decode_ms,
                    },
                )
                value = result.to_dict()
                value["decoded_pts_ms"] = pts_ms if math.isfinite(pts_ms) else None
                stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                alignment.write(
                    json.dumps(
                        {
                            "video_id": video_id,
                            "stream_id": stream_id,
                            "frame_index": index,
                            "timestamp_ms": track["timestamp_ms"],
                            "depth_record_index": len(results),
                            "track_record_index": index,
                            "risk_record_index": index,
                            "scene_record_index": indices.index(index),
                            "event_ids": [
                                o.get("event_id") for o in risks[index].get("observations", [])
                            ],
                            "note": "join keys only; no depth-risk fusion",
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
                results.append(value)
        summary["decoded_frames"] = count
        summary["status"] = "completed"
        # Whole-device snapshots are explicitly distinct from PyTorch allocator counters.
        if "gpu_during_run" not in runtime:
            runtime["gpu_during_run"] = _gpu_snapshot()
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if capture is not None:
            capture.release()
        summary["sampled_frames"] = len(results)
        summary["valid_frames"] = sum(r["status"] == "valid" for r in results)
        summary["unknown_frames"] = len(results) - summary["valid_frames"]
        summary["quality_reasons"] = dict(
            Counter(reason for r in results for reason in r["quality_reasons"])
        )
        summary["timings_ms"] = {
            name: percentiles([r["timings_ms"][name] for r in results if name in r["timings_ms"]])
            for name in (
                "preprocess",
                "forward",
                "postprocess",
                "inference",
                "save_depth",
                "save_preview",
                "full_sample",
            )
        }
        write_json(directory / "summary.json", summary)
        write_json(directory / "anomalies.json", [r for r in results if r["status"] == "unknown"])
    return summary


def run_depth_batch(
    manifest_path: Path,
    baseline_dir: Path,
    scene_dir: Path,
    output_dir: Path,
    config: DepthModelConfig,
    *,
    interval_seconds: float = 0.5,
    warmup_frames: int = 2,
    video_ids: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Preserve failures, process independent videos, never write to baseline sources."""
    if not math.isfinite(interval_seconds) or interval_seconds <= 0:
        raise ValueError("invalid sample interval")
    if type(warmup_frames) is not int or not 0 <= warmup_frames <= 10:
        raise ValueError("warmup_frames must be an integer in [0, 10]")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "sightguard.real-world.v0.1":
        raise ValueError("unsupported dataset manifest schema")
    rows = manifest.get("videos", [])
    requested = set(video_ids) if video_ids else None
    rows = [r for r in rows if requested is None or r["video_id"] in requested]
    ids = [r["video_id"] for r in rows]
    if not rows or len(set(ids)) != len(ids) or (requested is not None and set(ids) != requested):
        raise ValueError("manifest video selection is empty, duplicate or incomplete")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", v) for v in ids):
        raise ValueError("unsafe video_id")
    # Claim a fresh directory atomically, including rejecting an existing empty folder.
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    estimator = DepthEstimator(config)
    runtime: dict[str, Any] = {"warmup": [], "gpu_before": _gpu_snapshot()}
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": sha256_file(manifest_path),
            "schema_version": manifest.get("schema_version"),
        },
        "config": config.to_dict(),
        "versions": _versions(),
        "platform": platform.platform(),
        "sampling": {"interval_seconds": interval_seconds, "warmup_frames": warmup_frames},
        "runtime": runtime,
        "videos": [],
        "error": None,
        "production_decision_changed": False,
        "timing_scope": "full_sample includes selected read, checks, inference, NPZ/preview; "
        "excludes unsampled decode, warmup, JSONL writes; batch wall includes these",
        "source_sha256": {
            str(p.relative_to(Path(__file__).resolve().parents[1])): sha256_file(p)
            for p in [
                Path(__file__),
                *Path(__file__).resolve().parents[1].glob("perception/depth/*.py"),
            ]
        },
    }
    try:
        try:
            estimator.initialize()
        except DepthModelUnavailableError as exc:
            metadata["error"] = str(exc)
        metadata["model"] = {
            **config.to_dict(),
            **estimator.metadata,
            "license": "Apache-2.0",
            "backend": "HF Transformers FP32",
        }
        for row in rows:
            summary = _process_video(
                row,
                baseline_dir,
                scene_dir,
                output_dir,
                estimator,
                interval_seconds,
                warmup_frames,
                runtime,
            )
            metadata["videos"].append(summary)
            print(
                f"{row['video_id']}: {summary['status']}, "
                f"{summary['valid_frames']}/{summary['sampled_frames']} valid",
                flush=True,
            )
        metadata["status"] = (
            "completed"
            if metadata["error"] is None
            and all(
                v["status"] == "completed" and v["unknown_frames"] == 0 for v in metadata["videos"]
            )
            else "completed_with_unknown_or_failures"
        )
        all_results = [
            r
            for vid in ids
            if (output_dir / vid / "depth_frames.jsonl").is_file()
            for r in read_jsonl(output_dir / vid / "depth_frames.jsonl")
        ]
        metadata["timings_ms"] = {
            name: percentiles(
                [r["timings_ms"][name] for r in all_results if name in r["timings_ms"]]
            )
            for name in (
                "preprocess",
                "forward",
                "postprocess",
                "inference",
                "save_depth",
                "save_preview",
                "full_sample",
            )
        }
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        runtime["before_unload"] = estimator.resource_stats()
        estimator.close()
        runtime["after_unload"] = estimator.resource_stats()
        runtime["gpu_after"] = _gpu_snapshot()
        metadata["finished_at_utc"] = datetime.now(UTC).isoformat()
        metadata["wall_seconds"] = time.perf_counter() - started
        write_json(output_dir / "run_metadata.json", metadata)
    return metadata
