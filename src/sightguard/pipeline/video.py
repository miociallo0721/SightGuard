"""Reproducible video I/O, timestamps, provenance, and failure recording."""

import hashlib
import importlib.metadata
import json
import logging
import math
import platform
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from sightguard import __version__
from sightguard.config import PerceptionConfig
from sightguard.data import Detection, FrameRecord
from sightguard.perception.yolo import YoloPerceptor
from sightguard.visualization.annotate import annotate

LOGGER = logging.getLogger(__name__)


class Perceptor(Protocol):
    def process(self, frame: np.ndarray) -> tuple[Detection, ...]: ...


class VideoError(RuntimeError):
    """Input, decoding, or output encoding failed."""


class VideoClock:
    """Prefer strictly increasing OpenCV PTS; deterministic FPS fallback otherwise."""

    def __init__(self, fps: float) -> None:
        if not math.isfinite(fps) or fps <= 0:
            raise VideoError("Video FPS unavailable; timestamps cannot be reconstructed")
        self.interval = 1000.0 / fps
        self.last = -1.0

    def timestamp(self, index: int, pts_ms: float) -> tuple[float, str]:
        if math.isfinite(pts_ms) and pts_ms >= 0 and pts_ms > self.last:
            value, source = pts_ms, "video_pts"
        else:
            value = max(index * self.interval, self.last + self.interval)
            source = "fps_fallback"
        self.last = value
        return value, source


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _versions() -> dict[str, str]:
    versions = {"python": platform.python_version(), "sightguard": __version__}
    for package in ("ultralytics", "torch", "torchvision", "numpy", "opencv-python", "lap"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def _verify_annotation(path: Path, frames: int, size: tuple[int, int]) -> None:
    """Check container frame count and decode the first frame after encoder finalization."""
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise VideoError("Encoded MP4 cannot be opened")
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        ok, frame = capture.read()
        if not math.isfinite(count) or round(count) != frames or not ok or frame is None:
            raise VideoError("Encoded MP4 frame count or decoding check failed")
        if frame.shape[:2] != (size[1], size[0]):
            raise VideoError("Encoder changed output dimensions")
    finally:
        capture.release()


def analyze_video(
    input_path: Path,
    output_dir: Path,
    config: PerceptionConfig,
    *,
    perceptor_factory: Callable[[PerceptionConfig], Perceptor] = YoloPerceptor,
    on_progress: Callable[[int, int | None, float], None] | None = None,
) -> dict[str, Any]:
    """Create a new perceptor per call; preserve partial outputs with failed metadata.

    A processing error records the decoded frame and stops the run, since continuing
    after an uncertain tracker update could corrupt the temporal state.
    """
    input_path, output_dir = input_path.resolve(), output_dir.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    if input_path.suffix.lower() != ".mp4":
        raise VideoError("Phase 1 input must be a local MP4 file")
    output_dir.mkdir(parents=True, exist_ok=True)
    names = ("tracks.jsonl", "annotated.mp4", "run_metadata.json")
    if any((output_dir / name).exists() for name in names):
        raise FileExistsError("Output artifacts already exist; choose a new output directory")
    started = time.perf_counter()
    metadata: dict[str, Any] = {
        "schema_version": "1.0",
        "status": "running",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "input": {"path": str(input_path), "sha256": sha256_file(input_path)},
        "config": config.to_dict(),
        "versions": _versions(),
        "platform": platform.platform(),
        "perceptor": f"{perceptor_factory.__module__}.{perceptor_factory.__qualname__}",
        "tracker_scope": "new perceptor per video; persist within video only",
        "frames_processed": 0,
        "detections": 0,
        "missing_track_ids": 0,
        "timestamp_sources": {"video_pts": 0, "fps_fallback": 0},
        "annotation": {"codec": "mp4v", "timing": "constant source FPS; no audio"},
        "error": None,
    }
    capture = None
    writer = None
    try:
        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise VideoError(f"Cannot open video: {input_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        clock = VideoClock(fps)
        raw_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        expected = round(raw_count) if math.isfinite(raw_count) and raw_count > 0 else None
        metadata["input"].update(fps=fps, reported_frame_count=expected)
        perceptor = perceptor_factory(config)
        if isinstance(perceptor, YoloPerceptor):
            import torch
            from ultralytics.utils import ROOT

            model_path = Path(config.model_path).resolve()
            tracker_path = ROOT / "cfg" / "trackers" / config.tracker
            metadata["model"] = {"path": str(model_path), "sha256": sha256_file(model_path)}
            metadata["tracker"] = {
                "sha256": sha256_file(tracker_path),
                "yaml": tracker_path.read_text(encoding="utf-8"),
            }
            metadata["hardware"] = {
                "cuda_available": torch.cuda.is_available(),
                "torch_cuda": torch.version.cuda,
                "gpu_names": [
                    torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
                ],
            }
        index = 0
        with (output_dir / "tracks.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
            while True:
                ok, frame = capture.read()
                if not ok:
                    if index == 0 or (expected is not None and index < expected):
                        raise VideoError(f"Video read failed at frame {index}; expected {expected}")
                    metadata["end_reason"] = (
                        "reported_frame_count_reached"
                        if expected
                        else "eof_or_decode_failure_unknown"
                    )
                    break
                if frame is None or frame.size == 0:
                    raise VideoError(f"Decoder returned empty frame at {index}")
                height, width = frame.shape[:2]
                if writer is None:
                    metadata["input"].update(width=width, height=height)
                    writer = cv2.VideoWriter(
                        str(output_dir / "annotated.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise VideoError("MP4 encoder could not be opened")
                elif (width, height) != (metadata["input"]["width"], metadata["input"]["height"]):
                    raise VideoError("Video dimensions changed mid-stream")
                timestamp, source = clock.timestamp(index, capture.get(cv2.CAP_PROP_POS_MSEC))
                try:
                    detections = perceptor.process(frame)
                    record = FrameRecord(
                        index, timestamp, width, height, detections, timestamp_source=source
                    )
                except Exception as exc:
                    record = FrameRecord(
                        index,
                        timestamp,
                        width,
                        height,
                        status="processing_error",
                        error=f"{type(exc).__name__}: {exc}",
                        timestamp_source=source,
                    )
                    stream.write(record.to_json() + "\n")
                    writer.write(annotate(frame, record))
                    metadata["frames_processed"] += 1
                    metadata["timestamp_sources"][source] += 1
                    raise
                stream.write(record.to_json() + "\n")
                writer.write(annotate(frame, record))
                index += 1
                metadata["frames_processed"] = index
                metadata["detections"] += len(detections)
                metadata["missing_track_ids"] += sum(d.track_id is None for d in detections)
                metadata["timestamp_sources"][source] += 1
                if on_progress is not None:
                    on_progress(index, expected, timestamp)
                if index % config.log_every_frames == 0:
                    LOGGER.info(
                        "Processed %d/%s frames, video time %.1f ms",
                        index,
                        expected or "?",
                        timestamp,
                    )
        writer.release()
        writer = None
        _verify_annotation(
            output_dir / "annotated.mp4",
            index,
            (metadata["input"]["width"], metadata["input"]["height"]),
        )
        metadata["annotation"]["verified_container_frames"] = index
        metadata["annotation"]["first_frame_decoded"] = True
        if isinstance(perceptor, YoloPerceptor):
            inference_args = vars(perceptor.model.predictor.args)
            metadata["effective_inference"] = {
                key: inference_args.get(key)
                for key in (
                    "conf",
                    "iou",
                    "imgsz",
                    "classes",
                    "device",
                    "half",
                    "rect",
                    "max_det",
                    "agnostic_nms",
                    "nms",
                    "batch",
                    "vid_stride",
                )
            }
            metadata["inference_backend"] = {
                "device": str(perceptor.model.predictor.device),
                "fp16": bool(perceptor.model.predictor.model.fp16),
                "backend": "PyTorch",
            }
        metadata["status"] = "completed"
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        LOGGER.error("Analysis failed: %s", metadata["error"])
        raise
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
        metadata["finished_at_utc"] = datetime.now(UTC).isoformat()
        metadata["processing_seconds"] = time.perf_counter() - started
        (output_dir / "run_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )
    LOGGER.info("Completed %d frames; outputs: %s", metadata["frames_processed"], output_dir)
    return metadata
