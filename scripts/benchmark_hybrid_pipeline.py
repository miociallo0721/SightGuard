"""Compare Shadow Mode pipeline performance over paced local video replay.

The script uses one fresh LiveSession/YOLO tracker/AssessmentEngine per mode and
starts every replay at frame zero. Auxiliary decisions stay in Shadow Mode; the
existing Rule Engine remains the only alert controller.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from collections import defaultdict
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(root: Path, value: Path) -> Path:
    return (value if value.is_absolute() else root / value).resolve()


def configure_local_artifacts(root: Path) -> Path:
    """Keep model/runtime scratch files on the project drive before heavy imports."""
    temporary = (root / "models" / "hybrid" / "tmp").resolve()
    temporary.mkdir(parents=True, exist_ok=True)
    for name in ("TEMP", "TMP", "TMPDIR", "HF_HOME", "TORCH_HOME", "UV_CACHE_DIR"):
        os.environ[name] = str(temporary)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    return temporary


class PacedVideoCapture:
    """Camera-shaped local video source that replays at the file's nominal FPS."""

    def __init__(self, cv2: Any, path: Path, *, loop: bool = True) -> None:
        self.cv2 = cv2
        self.path = path
        self.capture = cv2.VideoCapture(str(path))
        self.loop = loop
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(self.fps) or self.fps <= 0:
            self.capture.release()
            raise ValueError(f"Video has invalid FPS metadata: {path}")
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.source_frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not self.capture.isOpened() or self.source_frame_count < 1:
            self.capture.release()
            raise OSError(f"Cannot open video or video contains no frames: {path}")
        self.started_at = time.monotonic()
        self.frames_read = 0
        self.loop_count = 0
        self.closed = False

    def isOpened(self) -> bool:  # noqa: N802 - matches OpenCV's camera interface.
        return not self.closed and self.capture.isOpened()

    def set(self, prop: int, value: float) -> bool:
        # LiveSession asks cameras to set capture dimensions/FPS/codec; file metadata wins.
        if prop == self.cv2.CAP_PROP_POS_FRAMES:
            return bool(self.capture.set(prop, value))
        return True

    def read(self) -> tuple[bool, Any | None]:
        if self.closed:
            return False, None
        due_at = self.started_at + self.frames_read / self.fps
        delay = due_at - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        ok, image = self.capture.read()
        if not ok and self.loop:
            self.loop_count += 1
            self.capture.set(self.cv2.CAP_PROP_POS_FRAMES, 0)
            ok, image = self.capture.read()
        if ok and image is not None:
            self.frames_read += 1
            return True, image
        return False, None

    def release(self) -> None:
        if not self.closed:
            self.closed = True
            self.capture.release()


class NvidiaSmiSampler:
    """Persist device-level nvidia-smi samples without conflating them with torch peaks."""

    def __init__(self, output_path: Path, interval_s: float) -> None:
        self.output_path = output_path
        self.interval_s = interval_s
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="gpu-device-sampler", daemon=True)
        self.writer = output_path.open("x", encoding="utf-8", newline="\n")
        self.sample_count = 0
        self.errors: list[str] = []

    def start(self) -> None:
        self.thread.start()

    def _sample(self) -> dict[str, Any]:
        command = [
            "nvidia-smi",
            "--query-gpu=timestamp,index,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
        if completed.returncode:
            raise RuntimeError(
                f"nvidia-smi exited {completed.returncode}: {completed.stderr.strip()}"
            )
        devices = []
        for row in csv.reader(completed.stdout.splitlines(), skipinitialspace=True):
            if len(row) < 6:
                continue
            devices.append(
                {
                    "nvidia_smi_timestamp": row[0].strip(),
                    "index": row[1].strip(),
                    "name": row[2].strip(),
                    "memory_total_mib": parse_number(row[3]),
                    "memory_used_mib": parse_number(row[4]),
                    "gpu_utilization_percent": parse_number(row[5]),
                }
            )
        return {"devices": devices, "raw_stdout": completed.stdout.strip()}

    def _run(self) -> None:
        while not self.stop_event.is_set():
            sample: dict[str, Any] = {
                "sampled_at_utc": utc_now(),
                "sampled_at_monotonic": time.monotonic(),
            }
            try:
                sample.update(self._sample())
            except Exception as exc:  # Hardware tooling is optional; retain the failure.
                message = f"{type(exc).__name__}: {exc}"
                sample["error"] = message
                self.errors.append(message)
            self.writer.write(json.dumps(sample, ensure_ascii=False, allow_nan=False) + "\n")
            self.writer.flush()
            self.sample_count += 1
            if self.errors and "FileNotFoundError" in self.errors[-1]:
                break
            self.stop_event.wait(self.interval_s)

    def close(self) -> None:
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=6)
        self.writer.close()


def parse_number(value: str) -> float | None:
    try:
        parsed = float(value.strip())
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def summarize_values(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "p50": percentile(values, 0.5),
        "p95": percentile(values, 0.95),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def write_json_line(stream: Any, payload: dict[str, Any]) -> None:
    stream.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()


def runtime_metadata() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {},
    }
    for package in ("torch", "ultralytics", "onnxruntime", "opencv-python", "numpy"):
        try:
            result["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][package] = None
    try:
        import torch

        result["torch_runtime"] = {
            "version": str(torch.__version__),
            "cuda_build": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        }
    except Exception as exc:
        result["torch_runtime_error"] = f"{type(exc).__name__}: {exc}"
    try:
        import onnxruntime

        result["onnxruntime_providers"] = onnxruntime.get_available_providers()
    except Exception as exc:
        result["onnxruntime_error"] = f"{type(exc).__name__}: {exc}"
    return result


def wait_for_live_result(session: Any, timeout_s: float, collect: Any) -> tuple[Any, str | None]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        collect("yolo-startup")
        result, _ = session.snapshot()
        if result is not None and result.record.status == "ok":
            return result, None
        error, _ = session.failure()
        if error:
            return result, error
        time.sleep(0.02)
    return session.snapshot()[0], f"YOLO did not produce a successful result within {timeout_s:g}s"


def wait_for_auxiliary_warmup(manager: Any, timeout_s: float, observe: Any) -> str | None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        observe("auxiliary-warmup")
        snapshot = manager.snapshot()
        if snapshot["latest_result"] is not None and snapshot["performance"]["calls"] > 0:
            return None
        if snapshot["model_status"] in {"Failed", "Not Installed"}:
            return snapshot.get("error") or f"Auxiliary model status: {snapshot['model_status']}"
        time.sleep(0.05)
    return f"Auxiliary warmup exceeded {timeout_s:g}s"


def collect_live_result(
    session: Any,
    phase: str,
    risk_stream: Any,
    observed: set[int],
    rows: list[dict[str, Any]],
    capture: PacedVideoCapture | None,
    assessment_timings: dict[int, dict[str, Any]],
    assessment_lock: threading.Lock,
) -> None:
    result, _ = session.snapshot()
    if result is None or result.record.frame_index in observed:
        return
    observed.add(result.record.frame_index)
    with assessment_lock:
        assessment = assessment_timings.get(result.record.frame_index)
    row = {
        "phase": phase,
        "observed_at_utc": utc_now(),
        "frame_index": result.record.frame_index,
        "source_capture_index": result.captured.capture_index if result.captured else None,
        "video_timestamp_ms": result.record.timestamp_ms,
        "perception_status": result.record.status,
        "perception_error": result.record.error,
        "risk_fps": result.update_fps,
        "processing_ms": result.processing_ms,
        "assessment_engine_step_ms": assessment["elapsed_ms"] if assessment else None,
        "alert_count": len(result.decision.alerts),
        "captured_video_frames": capture.frames_read if capture else None,
        "source_video_loops": capture.loop_count if capture else None,
    }
    rows.append(row)
    write_json_line(risk_stream, row)


def instrument_assessment_step(
    session: Any,
    phase: dict[str, str],
    rows: list[dict[str, Any]],
    by_frame: dict[int, dict[str, Any]],
    lock: threading.Lock,
) -> None:
    """Time only AssessmentEngine.step; retain original return and exception behavior."""
    original_step = session.engine.step

    def measured_step(record: Any) -> Any:
        started = time.perf_counter()
        try:
            result = original_step(record)
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000
            row = {
                "phase": phase["value"],
                "frame_index": record.frame_index,
                "timestamp_ms": record.timestamp_ms,
                "elapsed_ms": elapsed_ms,
                "error": f"{type(exc).__name__}: {exc}",
            }
            with lock:
                rows.append(row)
                by_frame[record.frame_index] = row
            raise
        elapsed_ms = (time.perf_counter() - started) * 1000
        row = {
            "phase": phase["value"],
            "frame_index": record.frame_index,
            "timestamp_ms": record.timestamp_ms,
            "elapsed_ms": elapsed_ms,
            "error": None,
        }
        with lock:
            rows.append(row)
            by_frame[record.frame_index] = row
        return result

    session.engine.step = measured_step


def observe_auxiliary(
    manager: Any,
    session: Any,
    phase: str,
    stream: Any,
    seen_ids: set[str],
    rows: list[dict[str, Any]],
    counts: dict[str, dict[str, int]],
    stale_after_ms: float,
) -> None:
    with manager.condition:
        latest = manager.latest
        if latest is None or latest.result_id in seen_ids:
            return
        seen_ids.add(latest.result_id)
        result = latest.to_dict()
        completed_at = getattr(manager, "latest_completed", None)
        model_metadata = dict(manager.model)
        model_source = manager.auxiliary
        model_latencies = list(manager.model_latencies[model_source])
        fps_before = list(manager.fps_before)
        fps_after = list(manager.fps_after)
        performance = {
            "calls": manager.calls,
            "timeouts": manager.timeouts,
            "stale": manager.stale_count,
            "p50_ms": percentile(model_latencies, 0.5),
            "p95_ms": percentile(model_latencies, 0.95),
            "peak_vram_mb": manager.peak_vram_mb,
            "fast_fps_before": statistics.fmean(fps_before) if fps_before else None,
            "fast_fps_after": statistics.fmean(fps_after) if fps_after else None,
        }
        model_status = manager.status
        manager_error = manager.error
    captured_at = session.origin + float(result["timestamp_ms"]) / 1000
    completion_age_ms = (
        max(0.0, (completed_at - captured_at) * 1000) if completed_at is not None else None
    )
    expired = bool(result.get("stale")) or (
        completion_age_ms is not None and completion_age_ms > stale_after_ms
    )
    unknown_abstention = result.get("error") == "Insufficient evidence (UNKNOWN)"
    error_type = str(result.get("error") or "").partition(":")[0]
    runtime_exception = bool(
        result.get("error")
        and not unknown_abstention
        and error_type.endswith(("Error", "Exception"))
    )
    output_error = bool(result.get("error") and not unknown_abstention and not runtime_exception)
    fresh = not expired
    valid = bool(result.get("valid")) and fresh
    row = {
        "phase": phase,
        "observed_at_utc": utc_now(),
        "completion_age_ms": completion_age_ms,
        "fresh_at_completion": fresh,
        "valid_at_completion": valid,
        "expired_or_stale": expired,
        "candidate": valid and result.get("candidate_risk_level") is not None,
        "unknown_abstention": unknown_abstention,
        "runtime_exception": runtime_exception,
        "output_error": output_error,
        "inference_latency_ms": result.get("inference_latency_ms"),
        "result": result,
        "model_metadata": model_metadata,
        "model_status": model_status,
        "manager_error": manager_error,
        "performance": performance,
    }
    rows.append(row)
    write_json_line(stream, row)
    bucket = counts[phase]
    bucket["completed"] += 1
    bucket["fresh"] += int(fresh)
    bucket["valid"] += int(valid)
    bucket["candidate"] += int(row["candidate"])
    bucket["expired_or_stale"] += int(expired)
    bucket["unknown_abstentions"] += int(unknown_abstention)
    bucket["output_errors"] += int(output_error)
    bucket["errors"] += int(runtime_exception)


def summarize_phase(
    risk_rows: list[dict[str, Any]],
    aux_rows: list[dict[str, Any]],
    assessment_rows: list[dict[str, Any]],
    phase: str,
) -> dict[str, Any]:
    risk = [row for row in risk_rows if row["phase"] == phase]
    auxiliary = [row for row in aux_rows if row["phase"] == phase]
    assessment = [row for row in assessment_rows if row["phase"] == phase]
    return {
        "risk_updates": len(risk),
        "risk_fps": summarize_values([float(row["risk_fps"]) for row in risk]),
        "processing_ms": summarize_values([float(row["processing_ms"]) for row in risk]),
        "assessment_engine_step_ms": summarize_values(
            [float(row["elapsed_ms"]) for row in assessment]
        ),
        "perception_error_frames": sum(row["perception_status"] != "ok" for row in risk),
        "auxiliary_decisions": len(auxiliary),
        "auxiliary_completion_age_ms": summarize_values(
            [
                float(row["completion_age_ms"])
                for row in auxiliary
                if row["completion_age_ms"] is not None
            ]
        ),
        "auxiliary_latency_ms": summarize_values(
            [
                float(row["inference_latency_ms"])
                for row in auxiliary
                if row["inference_latency_ms"] is not None
            ]
        ),
    }


def run_stage(
    *,
    root: Path,
    name: str,
    mode: str,
    hybrid_config: Any,
    perception_config: Any,
    risk_config: Any,
    live_config: Any,
    video_path: Path,
    seconds: float,
    startup_timeout_s: float,
    gpu_sample_interval_s: float,
    video_loop: bool,
    spatial_config: Any | None,
    output_dir: Path,
    cv2: Any,
    LiveSession: Any,
    IntelligenceManager: Any,
) -> dict[str, Any]:
    stage_dir = output_dir / name
    stage_dir.mkdir(parents=True, exist_ok=False)
    risk_rows: list[dict[str, Any]] = []
    aux_rows: list[dict[str, Any]] = []
    assessment_rows: list[dict[str, Any]] = []
    assessment_timings: dict[int, dict[str, Any]] = {}
    assessment_lock = threading.Lock()
    phase_state = {"value": "yolo-startup"}
    errors: list[str] = []
    aux_counts: defaultdict[str, dict[str, int]] = defaultdict(
        lambda: {
            "completed": 0,
            "fresh": 0,
            "valid": 0,
            "candidate": 0,
            "expired_or_stale": 0,
            "unknown_abstentions": 0,
            "output_errors": 0,
            "errors": 0,
        }
    )
    observed_frames: set[int] = set()
    observed_decisions: set[str] = set()
    stage_start = time.monotonic()
    stage: dict[str, Any] = {
        "name": name,
        "mode": mode,
        "status": "running",
        "started_at_utc": utc_now(),
        "configured_measurement_seconds": seconds,
        "video_path": str(video_path),
        "audio": {
            "requested": spatial_config is not None,
            "attempted": False,
            "config": asdict(spatial_config) if spatial_config is not None else None,
        },
        "actual_alert_controller": "Rule Engine",
        "shadow_only": True,
        "files": {},
        "errors": errors,
    }
    risk_path = stage_dir / "risk_samples.jsonl"
    auxiliary_path = stage_dir / "auxiliary_observations.jsonl"
    risk_stream = risk_path.open("x", encoding="utf-8", newline="\n")
    auxiliary_stream = auxiliary_path.open("x", encoding="utf-8", newline="\n")
    sampler = NvidiaSmiSampler(stage_dir / "gpu_device_samples.jsonl", gpu_sample_interval_s)
    sampler.start()
    capture: PacedVideoCapture | None = None
    session = None
    manager = None
    warmup_error = None

    def collect_live(phase: str) -> None:
        collect_live_result(
            session,
            phase,
            risk_stream,
            observed_frames,
            risk_rows,
            capture,
            assessment_timings,
            assessment_lock,
        )

    def collect_aux(phase: str) -> None:
        if manager is not None:
            observe_auxiliary(
                manager,
                session,
                phase,
                auxiliary_stream,
                observed_decisions,
                aux_rows,
                aux_counts,
                hybrid_config.stale_after_ms,
            )

    def make_capture(_index: int) -> PacedVideoCapture:
        nonlocal capture
        capture = PacedVideoCapture(cv2, video_path, loop=video_loop)
        return capture

    try:
        stage_hybrid_config = replace(
            hybrid_config, output_dir=str((stage_dir / "shadow").resolve())
        )
        resolved_perception = replace(
            perception_config,
            model_path=str(resolve_path(root, Path(perception_config.model_path))),
        )
        session = LiveSession(
            resolved_perception,
            risk_config,
            live_config,
            camera_factory=make_capture,
            spatial_config=spatial_config,
        )
        instrument_assessment_step(
            session, phase_state, assessment_rows, assessment_timings, assessment_lock
        )
        session.start()
        stage["audio"]["attempted"] = spatial_config is not None
        initial_result, startup_error = wait_for_live_result(
            session, startup_timeout_s, collect_live
        )
        if startup_error:
            errors.append(startup_error)
        live_ready = (
            initial_result is not None
            and initial_result.record.status == "ok"
            and startup_error is None
        )
        if live_ready:
            phase_state["value"] = "auxiliary-warmup" if mode != "rule" else "measurement"
            manager = IntelligenceManager(stage_hybrid_config, root, session.engine.stream_id)
            session.intelligence = manager
            manager.record_source(
                {
                    "perception_config": asdict(resolved_perception),
                    "risk_config": asdict(risk_config),
                    "live_config": asdict(live_config),
                    "timestamp_source": "monotonic relative to live session origin",
                    "video_path": str(video_path),
                    "video_sha256": sha256_file(video_path),
                }
            )
            manager.select(mode, load=mode != "rule")
            manager.submit(initial_result)
            if mode != "rule":
                warmup_timeout = hybrid_config.load_timeout_s + hybrid_config.timeout_s + 5
                warmup_error = wait_for_auxiliary_warmup(manager, warmup_timeout, collect_aux)
                collect_aux("auxiliary-warmup")
                if warmup_error:
                    errors.append(warmup_error)
            else:
                manager.snapshot()
        else:
            errors.append(
                "No successful live result available; Shadow Mode manager was not started"
            )

        phase_state["value"] = "measurement"
        measurement_started = time.monotonic()
        measurement_deadline = measurement_started + seconds
        while time.monotonic() < measurement_deadline:
            collect_live("measurement")
            collect_aux("measurement")
            error, _ = session.failure()
            if error and error not in errors:
                errors.append(error)
            time.sleep(0.02)
        collect_live("measurement")
        collect_aux("measurement")
        stage["measurement_seconds"] = time.monotonic() - measurement_started
        stage["status"] = "completed" if live_ready else "failed"
        if warmup_error:
            stage["auxiliary_warmup_error"] = warmup_error
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        stage["status"] = "failed"
    finally:
        phase_state["value"] = "shutdown-tail"
        if session is not None:
            audio_before_shutdown = (
                session.audio.snapshot() if spatial_config is not None and session.audio else None
            )
            try:
                session.close()
            except Exception as exc:
                errors.append(f"Session shutdown: {type(exc).__name__}: {exc}")
            audio_after_shutdown = (
                session.audio.snapshot() if spatial_config is not None and session.audio else None
            )
            stage["audio"]["before_shutdown"] = audio_before_shutdown
            stage["audio"]["after_shutdown"] = audio_after_shutdown
            audio_errors = [
                session.audio_error,
                (audio_before_shutdown or {}).get("error"),
                (audio_after_shutdown or {}).get("error"),
            ]
            stage["audio"]["errors"] = list(dict.fromkeys(error for error in audio_errors if error))
            if stage["audio"]["errors"]:
                errors.extend(f"Spatial audio: {error}" for error in stage["audio"]["errors"])
            collect_live("shutdown-tail")
            collect_aux("shutdown-tail")
        sampler.close()
        risk_stream.close()
        auxiliary_stream.close()
        assessment_path = stage_dir / "assessment_step_samples.jsonl"
        with assessment_path.open("x", encoding="utf-8", newline="\n") as assessment_stream:
            for row in assessment_rows:
                write_json_line(assessment_stream, row)

    finished = time.monotonic()
    stage["finished_at_utc"] = utc_now()
    stage["elapsed_seconds_including_startup_and_shutdown"] = finished - stage_start
    stage["risk_updates"] = len(risk_rows)
    stage["risk_metrics_by_phase"] = {
        phase: summarize_phase(risk_rows, aux_rows, assessment_rows, phase)
        for phase in ("yolo-startup", "auxiliary-warmup", "measurement", "shutdown-tail")
    }
    stage["assessment_step_calls"] = len(assessment_rows)
    stage["assessment_step_errors"] = [row["error"] for row in assessment_rows if row.get("error")]
    stage["auxiliary_freshness_counts"] = dict(aux_counts)
    stage["auxiliary_model"] = dict(manager.model) if manager is not None else {}
    auxiliary_performance = manager.snapshot() if manager is not None else None
    if (
        auxiliary_performance
        and auxiliary_performance.get("error") == "Insufficient evidence (UNKNOWN)"
    ):
        auxiliary_performance["abstention"] = auxiliary_performance["error"]
        auxiliary_performance["error"] = None
    stage["auxiliary_performance"] = auxiliary_performance
    stage["auxiliary_logs_directory"] = str(manager.output) if manager is not None else None
    stage["gpu_device_sample_count"] = sampler.sample_count
    stage["gpu_sampler_errors"] = list(sampler.errors)
    auxiliary_errors = [
        str(row.get("result", {}).get("error")) for row in aux_rows if row.get("runtime_exception")
    ]
    stage["auxiliary_errors"] = auxiliary_errors
    stage["auxiliary_output_errors"] = [
        str(row.get("result", {}).get("error")) for row in aux_rows if row.get("output_error")
    ]
    stage["auxiliary_unknown_abstentions"] = sum(
        bool(row.get("unknown_abstention")) for row in aux_rows
    )
    errors.extend(error for error in auxiliary_errors if error not in errors)
    errors.extend(
        error if error.startswith("nvidia-smi:") else f"nvidia-smi: {error}"
        for error in sampler.errors
        if error not in errors
    )
    gpu_samples = read_jsonl(stage_dir / "gpu_device_samples.jsonl")
    memory_values = [
        device["memory_used_mib"]
        for row in gpu_samples
        for device in row.get("devices", [])
        if device.get("memory_used_mib") is not None
    ]
    stage["nvidia_smi_device_memory_used_mib"] = summarize_values(memory_values)
    stage["cuda_oom_or_runtime_error_count"] = sum(
        "out of memory" in error.lower() or "cuda" in error.lower() for error in errors
    ) + sum(
        row.get("runtime_exception")
        and (
            "out of memory" in str(row.get("result", {}).get("error", "")).lower()
            or "cuda" in str(row.get("result", {}).get("error", "")).lower()
        )
        for row in aux_rows
    )
    stage["files"] = {
        "risk_samples": str(risk_path),
        "assessment_step_samples": str(stage_dir / "assessment_step_samples.jsonl"),
        "auxiliary_observations": str(auxiliary_path),
        "gpu_device_samples": str(stage_dir / "gpu_device_samples.jsonl"),
        "shadow_logs": str(manager.output) if manager is not None else None,
    }
    stage["errors"] = errors
    (stage_dir / "stage_summary.json").write_text(
        json.dumps(stage, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    return stage


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# SightGuard Hybrid Pipeline Benchmark",
        "",
        f"Status: **{summary['status']}**",
        "",
        f"Video: `{summary['video']['path']}` ({summary['video']['sha256']})",
        f"Hybrid config: `{summary['hybrid_config']['path']}`",
        f"Configured measurement: {summary['seconds_per_stage']} seconds after YOLO/model warmup.",
        "",
        "All auxiliary output is Shadow Mode. Actual alerts remain controlled by the Rule Engine.",
        "The fixture is a repeated still image and does not measure tracking accuracy.",
        "",
        (
            "| Stage | Status | Updates | Mean risk FPS | Mean pipeline ms | Mean rule step ms | "
            "Fresh aux | Candidates | Stale/expired | UNKNOWN |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in summary["stages"]:
        metrics = stage.get("risk_metrics_by_phase", {}).get("measurement", {})
        fps = metrics.get("risk_fps", {}).get("mean")
        processing = metrics.get("processing_ms", {}).get("mean")
        rule_step = metrics.get("assessment_engine_step_ms", {}).get("mean")
        freshness = stage.get("auxiliary_freshness_counts", {}).get(
            "measurement",
            {"fresh": 0, "candidate": 0, "expired_or_stale": 0, "unknown_abstentions": 0},
        )
        lines.append(
            (
                "| {name} | {status} | {updates} | {fps} | {processing} | {rule_step} | "
                "{fresh} | {candidate} | {stale} | {unknown} |"
            ).format(
                name=stage["name"],
                status=stage["status"],
                updates=metrics.get("risk_updates", 0),
                fps=f"{fps:.3f}" if fps is not None else "n/a",
                processing=f"{processing:.3f}" if processing is not None else "n/a",
                rule_step=f"{rule_step:.3f}" if rule_step is not None else "n/a",
                fresh=freshness.get("fresh", 0),
                candidate=freshness.get("candidate", 0),
                stale=freshness.get("expired_or_stale", 0),
                unknown=freshness.get("unknown_abstentions", 0),
            )
        )
    lines.extend(
        [
            "",
            (
                "Raw per-frame samples, auxiliary decisions, device-memory samples, and "
                "manager logs are stored beside this report."
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hybrid-config", type=Path, default=Path("configs/hybrid.yaml"))
    parser.add_argument("--video", type=Path, default=Path("examples/tracking_fixture.mp4"))
    parser.add_argument("--perception-config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--risk-config", type=Path, default=Path("configs/risk.yaml"))
    parser.add_argument("--live-config", type=Path, default=Path("configs/live.yaml"))
    parser.add_argument("--spatial-config", type=Path)
    parser.add_argument("--seconds-per-stage", type=float, default=30.0)
    parser.add_argument("--startup-timeout-s", type=float, default=120.0)
    parser.add_argument("--gpu-sample-interval-s", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/hybrid/gpu-benchmark"))
    parser.add_argument("--no-video-loop", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if (
        args.seconds_per_stage <= 0
        or args.startup_timeout_s <= 0
        or args.gpu_sample_interval_s <= 0
    ):
        raise SystemExit("Duration and sample interval arguments must be positive")
    root = Path(__file__).resolve().parents[1]
    configure_local_artifacts(root)

    # Import project/dependency modules only after D-drive temp/cache paths are set.
    import cv2

    from sightguard.config import load_config
    from sightguard.hybrid.config import load_hybrid_config
    from sightguard.hybrid.manager import IntelligenceManager
    from sightguard.live.config import load_live_config
    from sightguard.live.runtime import LiveSession
    from sightguard.risk.config import load_risk_config

    hybrid_path = resolve_path(root, args.hybrid_config)
    video_path = resolve_path(root, args.video)
    output_base = resolve_path(root, args.output_dir)
    for path in (hybrid_path, video_path):
        if not path.is_file():
            raise SystemExit(f"Required file does not exist: {path}")
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    output_dir = output_base / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    perception_path = resolve_path(root, args.perception_config)
    risk_path = resolve_path(root, args.risk_config)
    live_path = resolve_path(root, args.live_config)
    spatial_path = resolve_path(root, args.spatial_config) if args.spatial_config else None
    hybrid_config = load_hybrid_config(hybrid_path, root)
    perception_config = load_config(perception_path)
    risk_config = load_risk_config(risk_path)
    live_config = load_live_config(live_path)
    spatial_config = None
    if spatial_path is not None:
        from sightguard.spatial.config import load_spatial_config

        spatial_config = load_spatial_config(spatial_path)
    video_probe = cv2.VideoCapture(str(video_path))
    video_info = {
        "path": str(video_path),
        "sha256": sha256_file(video_path),
        "fps": float(video_probe.get(cv2.CAP_PROP_FPS)),
        "width": int(video_probe.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(video_probe.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frame_count": int(video_probe.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    video_probe.release()
    if (
        not math.isfinite(video_info["fps"])
        or video_info["fps"] <= 0
        or video_info["frame_count"] < 1
    ):
        raise SystemExit(f"Video metadata is invalid: {video_path}")

    summary: dict[str, Any] = {
        "schema_version": "1.0",
        "status": "running",
        "created_at_utc": utc_now(),
        "run_id": run_id,
        "project_root": str(root),
        "git_commit": git_commit(root),
        "video": video_info,
        "hybrid_config": {
            "path": str(hybrid_path),
            "sha256": sha256_file(hybrid_path),
            "schema_version": hybrid_config.schema_version,
            "stale_after_ms": hybrid_config.stale_after_ms,
            "frequency_hz": hybrid_config.frequency_hz,
        },
        "config_files": {
            "perception": {"path": str(perception_path), "sha256": sha256_file(perception_path)},
            "risk": {"path": str(risk_path), "sha256": sha256_file(risk_path)},
            "live": {"path": str(live_path), "sha256": sha256_file(live_path)},
            "spatial": (
                {"path": str(spatial_path), "sha256": sha256_file(spatial_path)}
                if spatial_path is not None
                else None
            ),
        },
        "perception_config": asdict(perception_config),
        "risk_config": asdict(risk_config),
        "live_config": asdict(live_config),
        "runtime": runtime_metadata(),
        "seconds_per_stage": args.seconds_per_stage,
        "startup_timeout_s": args.startup_timeout_s,
        "video_loop": not args.no_video_loop,
        "outputs": str(output_dir),
        "stages": [],
        "limitations": [
            (
                "The built-in tracking fixture repeats one still image; it is an integration "
                "smoke input, not a tracking-accuracy benchmark."
            ),
            (
                "Each stage uses a fresh LiveSession, YOLO tracker, and AssessmentEngine, "
                "then replays the same video from frame zero at nominal video FPS."
            ),
            (
                "The measurement window begins after YOLO startup and, for auxiliary modes, "
                "one completed warmup decision; startup samples are retained separately."
            ),
            (
                "nvidia-smi samples are whole-device readings. The isolated host reports "
                "torch.cuda.max_memory_allocated as a per-process peak; that value does not "
                "include all device allocations."
            ),
            (
                "LiveSession uses a capacity-one latest-frame slot. Under load, identical "
                "paced input can yield different processed frame subsets across stages."
            ),
        ],
    }
    if spatial_config is None:
        summary["limitations"].append(
            "No spatial audio device is opened without --spatial-config; "
            "audio compatibility is not exercised."
        )
    else:
        summary["limitations"].append(
            "Spatial audio uses the existing worker/config path; software status is recorded, "
            "but audible output is not measured."
        )
    summary_path = output_dir / "benchmark_hybrid_pipeline.json"
    report_path = output_dir / "benchmark_hybrid_pipeline.md"
    stages = (
        ("rule-only", "rule"),
        ("rule-structured", "laya"),
        ("rule-vision", "vision"),
    )
    fatal_errors = False
    for name, mode in stages:
        print(f"Starting {name}: replay={video_path}", flush=True)
        stage = run_stage(
            root=root,
            name=name,
            mode=mode,
            hybrid_config=hybrid_config,
            perception_config=perception_config,
            risk_config=risk_config,
            live_config=live_config,
            video_path=video_path,
            seconds=args.seconds_per_stage,
            startup_timeout_s=args.startup_timeout_s,
            gpu_sample_interval_s=args.gpu_sample_interval_s,
            video_loop=not args.no_video_loop,
            spatial_config=spatial_config,
            output_dir=output_dir,
            cv2=cv2,
            LiveSession=LiveSession,
            IntelligenceManager=IntelligenceManager,
        )
        summary["stages"].append(stage)
        fatal_errors |= stage["status"] != "completed"
        summary["status"] = "running"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )
        write_markdown(report_path, summary)
        print(
            f"Finished {name}: status={stage['status']} samples={stage.get('risk_updates', 0)}",
            flush=True,
        )
    has_errors = any(stage.get("errors") for stage in summary["stages"])
    summary["status"] = (
        "failed" if fatal_errors else "completed_with_errors" if has_errors else "completed"
    )
    summary["finished_at_utc"] = utc_now()
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    write_markdown(report_path, summary)
    print(f"{summary['status']}: {report_path}", flush=True)
    return 1 if fatal_errors else 0


def git_commit(root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


if __name__ == "__main__":
    raise SystemExit(main())
