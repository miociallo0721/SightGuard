"""Offline Shadow-mode path relevance over existing tracks and risk artifacts."""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import logging
import math
import platform
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from sightguard import __version__
from sightguard.path_relevance.engine import (
    PATH_RELEVANCE_HIGH_MIN,
    PATH_RELEVANCE_MEDIUM_MIN,
    CorridorEstimate,
    ObjectPathAssessment,
    associate_detection,
    estimate_corridors,
    scene_is_fresh,
)
from sightguard.path_relevance.model import (
    InvalidSegmentationOutputError,
    ModelUnavailableError,
    SegformerB0Perceptor,
)

LOGGER = logging.getLogger(__name__)
SCHEMA_VERSION = "sightguard.path-relevance.v1"
MODEL_CHECKPOINT_URL = (
    "https://download.openmmlab.com/mmsegmentation/v0.5/segformer/"
    "segformer_mit-b0_512x512_160k_ade20k/"
    "segformer_mit-b0_512x512_160k_ade20k_20210726_101530-8ffa8fda.pth"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row is not an object at {path}:{line_number}")
            rows.append(row)
    return rows


def _finite(value: Any) -> float | None:
    if type(value) not in (int, float) or not math.isfinite(value):
        return None
    return float(value)


def _installed_version(*distribution_names: str) -> str:
    for name in distribution_names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "unknown"


def _event_associations(risk_row: dict[str, Any] | None) -> dict[int, dict[str, Any]]:
    if risk_row is None:
        return {}
    return {
        int(item["detection_index"]): item
        for item in risk_row.get("observations", [])
        if isinstance(item, dict) and type(item.get("detection_index")) is int
    }


def _geometry_baseline_score(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
    *,
    x_start_fraction: float = 1 / 3,
    x_end_fraction: float = 2 / 3,
    y_start_fraction: float = 0.55,
) -> float:
    """Central lower-third image ROI diagnostic; this is not the model algorithm."""
    x1, y1, x2, y2 = bbox
    left, right = width * x_start_fraction, width * x_end_fraction
    top, bottom = height * y_start_fraction, float(height)
    overlap_w = max(0.0, min(x2, right) - max(x1, left))
    overlap_h = max(0.0, min(y2, bottom) - max(y1, top))
    area = max(0.0, (x2 - x1) * (y2 - y1))
    return float(np.clip(overlap_w * overlap_h / area, 0, 1)) if area > 0 else 0.0


def _shadow_record(
    *,
    video_id: str,
    frame_index: int,
    timestamp_ms: float,
    path: ObjectPathAssessment,
    risk_observation: dict[str, Any] | None,
    alert_event_ids: set[str],
    geometry_score: float,
    medium_threshold: float = PATH_RELEVANCE_MEDIUM_MIN,
    high_threshold: float = PATH_RELEVANCE_HIGH_MIN,
) -> dict[str, Any]:
    assessment = risk_observation.get("assessment") if risk_observation else None
    risk_score = _finite(assessment.get("score")) if isinstance(assessment, dict) else None
    risk_level = assessment.get("level") if isinstance(assessment, dict) else None
    event_id = path.event_id
    baseline_alert = bool(event_id and event_id in alert_event_ids)
    path_score = path.path_relevance_score
    combined = (
        risk_score * path_score if risk_score is not None and path_score is not None else None
    )
    if risk_level == "HIGH":
        recommendation = "preserve_baseline_high"
    elif path_score is None or path.scene_quality == "unknown":
        recommendation = "retain_baseline_path_unknown"
    elif path_score >= high_threshold:
        recommendation = "retain_path_related_candidate"
    elif path_score < medium_threshold and path.confidence >= 0.55:
        recommendation = "lower_shadow_priority_research_only"
    else:
        recommendation = "manual_review"
    return {
        "schema_version": SCHEMA_VERSION,
        "video_id": video_id,
        "frame_index": frame_index,
        "timestamp_ms": timestamp_ms,
        "track_id": path.track_id,
        "event_id": event_id,
        "class_name": path.class_name,
        "baseline_risk_score": risk_score,
        "baseline_risk_level": risk_level,
        "baseline_alert_event_present": baseline_alert,
        "geometry_baseline_score": geometry_score,
        "path_relevance_score": path_score,
        "shadow_combined_priority": combined,
        "shadow_recommendation": recommendation,
        "production_decision_changed": False,
    }


def _draw_overlay(
    frame: np.ndarray,
    estimate: CorridorEstimate,
    objects: list[ObjectPathAssessment],
    inference_ms: float,
) -> np.ndarray:
    import cv2

    output = frame.copy()
    if estimate.status == "valid":
        source_mask = cv2.resize(
            estimate.mask,
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        tint = np.zeros_like(output)
        tint[:] = (50, 180, 60)
        output[source_mask] = cv2.addWeighted(output, 0.58, tint, 0.42, 0)[source_mask]
    colors = {
        "HIGH": (30, 30, 235),
        "MEDIUM": (0, 190, 255),
        "LOW": (220, 180, 30),
        "UNKNOWN": (170, 170, 170),
    }
    for item in objects:
        x1, y1, x2, y2 = (round(value) for value in item.bbox_xyxy)
        color = colors[item.relevance_level]
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        score = "?" if item.path_relevance_score is None else f"{item.path_relevance_score:.2f}"
        label = (
            f"{item.class_name} #{item.track_id if item.track_id is not None else '-'} PR={score}"
        )
        cv2.putText(
            output,
            label[:64],
            (max(0, x1), max(20, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.rectangle(output, (0, 0), (output.shape[1], 50), (18, 25, 28), -1)
    cv2.putText(
        output,
        (
            f"MODEL-ESTIMATED GROUND REGION  |  {estimate.status.upper()} / "
            f"{estimate.quality.upper()}  |  {inference_ms:.1f} ms"
        ),
        (12, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        "Not a validated safe path; image-space study only",
        (12, 41),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    return output


def _update_summary(
    frame_summaries: list[dict[str, Any]],
    object_rows: list[dict[str, Any]],
    shadow_rows: list[dict[str, Any]],
    *,
    video_id: str,
    sample_interval_s: float,
    total_frames: int,
    model_load_ms: float,
    inference_durations: list[float],
    full_durations: list[float],
    device_names: list[str],
    peak_gpu_memory_mb: float | None,
    input_hashes: dict[str, str],
    baseline_alerts: list[dict[str, Any]],
) -> dict[str, Any]:
    class_scores: dict[str, list[float]] = defaultdict(list)
    relevance_counts: Counter[str] = Counter()
    for row in object_rows:
        relevance_counts[str(row["relevance_level"])] += 1
        value = row.get("path_relevance_score")
        if value is not None:
            class_scores[row["class_name"]].append(float(value))
    recommendations = Counter(row["shadow_recommendation"] for row in shadow_rows)
    quality_counts = Counter(row["quality"] for row in frame_summaries)
    scored_objects = [row for row in object_rows if row.get("baseline_risk_score") is not None]
    geometry_scores = [float(row["geometry_baseline_score"]) for row in shadow_rows]
    path_scores = [
        float(row["path_relevance_score"])
        for row in object_rows
        if row.get("path_relevance_score") is not None
    ]
    alert_levels = Counter(str(row.get("risk_level") or "UNKNOWN") for row in baseline_alerts)
    valid_scenes = sum(row["status"] == "valid" for row in frame_summaries)
    return {
        "schema_version": SCHEMA_VERSION,
        "video_id": video_id,
        "status": "completed",
        "inputs": input_hashes,
        "frames_in_video": total_frames,
        "sampled_scene_frames": len(frame_summaries),
        "sample_interval_seconds": sample_interval_s,
        "scene_coverage_fraction": valid_scenes / len(frame_summaries) if frame_summaries else 0.0,
        "scene_quality_counts": dict(quality_counts),
        "unknown_scene_frames": len(frame_summaries) - valid_scenes,
        "tracked_objects_assessed": len(object_rows),
        "path_relevance_levels": dict(relevance_counts),
        "mean_path_relevance_by_class": {
            name: float(np.mean(scores)) for name, scores in sorted(class_scores.items())
        },
        "shadow_recommendation_counts": dict(recommendations),
        "baseline_alert_event_count": len(
            {
                row.get("event_id")
                for row in shadow_rows
                if row.get("baseline_alert_event_present") and row.get("event_id")
            }
        ),
        "comparison": {
            "A_baseline": {
                "unchanged_alert_count": len(baseline_alerts),
                "alert_count_by_level": dict(alert_levels),
                "sampled_risk_observations": len(scored_objects),
                "mean_risk_score_on_sampled_observations": float(
                    np.mean([row["baseline_risk_score"] for row in scored_objects])
                )
                if scored_objects
                else None,
            },
            "B_geometry_baseline": {
                "sampled_associated_observations": len(geometry_scores),
                "mean_geometry_roi_overlap": float(np.mean(geometry_scores))
                if geometry_scores
                else None,
                "observations_over_0_20": sum(value >= 0.20 for value in geometry_scores),
            },
            "C_path_relevance": {
                "objects_with_defined_score": len(path_scores),
                "mean_path_relevance": float(np.mean(path_scores)) if path_scores else None,
                "objects_at_or_above_0_20": sum(value >= 0.20 for value in path_scores),
            },
            "D_shadow_fusion": {
                "suggestion_counts_only": dict(recommendations),
                "production_decisions_changed": 0,
            },
        },
        "model_load_ms": model_load_ms,
        "model_inference_latency_ms_mean": float(np.mean(inference_durations))
        if inference_durations
        else None,
        "model_inference_latency_ms_p95": float(np.percentile(inference_durations, 95))
        if inference_durations
        else None,
        "model_inference_latency_ms_median": float(np.median(inference_durations))
        if inference_durations
        else None,
        "model_inference_latency_ms_first_sample": inference_durations[0]
        if inference_durations
        else None,
        "model_inference_latency_ms_steady_mean": float(np.mean(inference_durations[1:]))
        if len(inference_durations) > 1
        else None,
        "full_sample_processing_ms_mean": float(np.mean(full_durations))
        if full_durations
        else None,
        "runtime_devices": sorted(set(device_names)),
        "peak_cuda_memory_allocated_mb": peak_gpu_memory_mb,
        "human_visual_attribute_labels": "pending_or_null; no accuracy metrics computed",
        "safety_boundary": (
            "semantic ground relevance is not risk probability, collision prediction, "
            "or a validated safe path"
        ),
    }


def _process_one_video(
    *,
    video_id: str,
    video_path: Path,
    tracks_path: Path,
    risk_dir: Path,
    output_dir: Path,
    model: SegformerB0Perceptor | None,
    model_path: Path,
    sample_interval_s: float,
    min_pixel_confidence: float,
    result_ttl_ms: float,
    allow_cpu_fallback: bool,
    min_area_fraction: float,
    bottom_start_fraction: float,
    min_vertical_extent_fraction: float,
    geometry_roi: tuple[float, float, float],
    medium_threshold: float,
    high_threshold: float,
    metadata: dict[str, Any],
) -> tuple[dict[str, Any], SegformerB0Perceptor | None]:
    import cv2

    for path in (
        video_path,
        tracks_path,
        risk_dir / "risk_frames.jsonl",
        risk_dir / "events.jsonl",
        risk_dir / "alerts.jsonl",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Path Relevance output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir()
    tracks = _read_jsonl(tracks_path)
    risks = _read_jsonl(risk_dir / "risk_frames.jsonl")
    alerts = _read_jsonl(risk_dir / "alerts.jsonl")
    if len(tracks) != len(risks):
        raise ValueError(f"Tracks/risk frame count mismatch for {video_id}")
    track_by_index = {int(row["frame_index"]): row for row in tracks}
    risk_by_index = {int(row["frame_index"]): row for row in risks}
    alert_event_ids = {str(row["event_id"]) for row in alerts if row.get("event_id")}
    frames_out: list[dict[str, Any]] = []
    object_out: list[dict[str, Any]] = []
    shadow_out: list[dict[str, Any]] = []
    infer_ms: list[float] = []
    total_ms: list[float] = []
    devices = [model.device_name] if model is not None else ["unavailable"]
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError(f"Could not open source video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError(f"Invalid source FPS for {video_id}: {fps}")
    frame_step = max(1, round(fps * sample_interval_s))
    reported_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    frame_index = 0
    try:
        with (
            (output_dir / "scene_frames.jsonl").open(
                "x", encoding="utf-8", newline="\n"
            ) as scene_stream,
            (output_dir / "objects.jsonl").open(
                "x", encoding="utf-8", newline="\n"
            ) as object_stream,
            (output_dir / "shadow.jsonl").open(
                "x", encoding="utf-8", newline="\n"
            ) as shadow_stream,
        ):
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index % frame_step == 0:
                    full_started = time.perf_counter()
                    track_row = track_by_index.get(frame_index)
                    risk_row = risk_by_index.get(frame_index)
                    if track_row is None or risk_row is None:
                        raise ValueError(f"Missing existing replay record at frame {frame_index}")
                    timestamp_ms = _finite(track_row.get("timestamp_ms"))
                    risk_timestamp_ms = _finite(risk_row.get("timestamp_ms"))
                    if (
                        timestamp_ms is None
                        or risk_timestamp_ms is None
                        or abs(timestamp_ms - risk_timestamp_ms) > 1.0
                    ):
                        raise ValueError(f"Risk timestamp does not align at frame {frame_index}")
                    inference_started = time.perf_counter()
                    inference_error = None
                    inference_status = "completed"
                    current_infer_ms = None
                    mask_width = (
                        model.input_width
                        if model is not None
                        else int(metadata["model"]["input_width"])
                    )
                    mask_height = max(
                        32,
                        round(frame.shape[0] * mask_width / frame.shape[1] / 32) * 32,
                    )
                    if model is None:
                        inference_status = "unavailable"
                        inference_error = metadata["model"].get("unavailable_reason")
                        estimate = CorridorEstimate(
                            status="unknown",
                            quality="unknown",
                            confidence=0.0,
                            reasons=("scene_model_unavailable",),
                            corridors=(),
                            mask=np.zeros((mask_height, mask_width), dtype=np.uint8),
                            width=mask_width,
                            height=mask_height,
                        )
                    else:
                        try:
                            segmentation = model.predict(frame)
                            current_infer_ms = segmentation.inference_latency_ms
                            infer_ms.append(current_infer_ms)
                            estimate = estimate_corridors(
                                segmentation,
                                min_pixel_confidence=min_pixel_confidence,
                                min_area_fraction=min_area_fraction,
                                bottom_start_fraction=bottom_start_fraction,
                                min_vertical_extent_fraction=min_vertical_extent_fraction,
                            )
                        except Exception as exc:
                            # A failed or unusable model result is explicitly UNKNOWN.
                            inference_error = f"{type(exc).__name__}: {exc}"
                            inference_status = (
                                "invalid_output"
                                if isinstance(exc, InvalidSegmentationOutputError)
                                else "failed"
                            )
                            estimate = CorridorEstimate(
                                status="unknown",
                                quality="unknown",
                                confidence=0.0,
                                reasons=(
                                    "scene_model_output_invalid"
                                    if inference_status == "invalid_output"
                                    else "scene_model_failed",
                                ),
                                corridors=(),
                                mask=np.zeros((mask_height, mask_width), dtype=np.uint8),
                                width=mask_width,
                                height=mask_height,
                            )
                            torch = model.torch
                            is_oom = isinstance(exc, torch.cuda.OutOfMemoryError)
                            if is_oom and allow_cpu_fallback and model.device.type == "cuda":
                                LOGGER.warning(
                                    "CUDA OOM at %s frame %d; falling back to CPU",
                                    video_id,
                                    frame_index,
                                )
                                peak = model.torch.cuda.max_memory_allocated(model.device) / (
                                    1024**2
                                )
                                metadata["peak_cuda_memory_allocated_mb"] = max(
                                    float(metadata.get("peak_cuda_memory_allocated_mb", 0.0)),
                                    peak,
                                )
                                model.torch.cuda.empty_cache()
                                del model
                                gc.collect()
                                model = SegformerB0Perceptor(model_path, device="cpu")
                                devices.append(model.device_name)
                                metadata.setdefault("cpu_fallbacks", []).append(
                                    {
                                        "video_id": video_id,
                                        "frame_index": frame_index,
                                        "reason": inference_error,
                                    }
                                )
                    if not scene_is_fresh(
                        timestamp_ms, risk_timestamp_ms, max_age_ms=result_ttl_ms
                    ):
                        estimate = CorridorEstimate(
                            status="unknown",
                            quality="unknown",
                            confidence=0.0,
                            reasons=("scene_result_expired_or_timestamp_misaligned",),
                            corridors=(),
                            mask=np.zeros_like(estimate.mask),
                            width=estimate.width,
                            height=estimate.height,
                        )
                    risk_associations = _event_associations(risk_row)
                    object_assessments = []
                    for detection_index, detection in enumerate(track_row.get("detections", [])):
                        risk_observation = risk_associations.get(detection_index)
                        risk_event_id = (
                            risk_observation.get("event_id") if risk_observation else None
                        )
                        assessment = associate_detection(
                            detection,
                            detection_index,
                            estimate,
                            event_id=risk_event_id,
                            frame_width=int(track_row["frame_width"]),
                            frame_height=int(track_row["frame_height"]),
                            medium_threshold=medium_threshold,
                            high_threshold=high_threshold,
                        )
                        object_assessments.append(assessment)
                        path_row = {
                            "schema_version": SCHEMA_VERSION,
                            "video_id": video_id,
                            "timestamp_ms": timestamp_ms,
                            "frame_index": frame_index,
                            "frame_width": int(track_row["frame_width"]),
                            "frame_height": int(track_row["frame_height"]),
                            "inference_latency_ms": current_infer_ms,
                            **assessment.to_dict(),
                            "baseline_risk_score": (
                                _finite(risk_observation.get("assessment", {}).get("score"))
                                if risk_observation
                                and isinstance(risk_observation.get("assessment"), dict)
                                else None
                            ),
                            "baseline_risk_level": (
                                risk_observation.get("assessment", {}).get("level")
                                if risk_observation
                                and isinstance(risk_observation.get("assessment"), dict)
                                else None
                            ),
                        }
                        object_stream.write(
                            json.dumps(path_row, ensure_ascii=False, allow_nan=False) + "\n"
                        )
                        object_out.append(path_row)
                        if risk_observation:
                            geometry_score = _geometry_baseline_score(
                                assessment.bbox_xyxy,
                                int(track_row["frame_width"]),
                                int(track_row["frame_height"]),
                                x_start_fraction=geometry_roi[0],
                                x_end_fraction=geometry_roi[1],
                                y_start_fraction=geometry_roi[2],
                            )
                            shadow = _shadow_record(
                                video_id=video_id,
                                frame_index=frame_index,
                                timestamp_ms=timestamp_ms,
                                path=assessment,
                                risk_observation=risk_observation,
                                alert_event_ids=alert_event_ids,
                                geometry_score=geometry_score,
                                medium_threshold=medium_threshold,
                                high_threshold=high_threshold,
                            )
                            shadow_stream.write(
                                json.dumps(shadow, ensure_ascii=False, allow_nan=False) + "\n"
                            )
                            shadow_out.append(shadow)
                    complete_ms = (time.perf_counter() - inference_started) * 1000.0
                    scene_row = {
                        "schema_version": SCHEMA_VERSION,
                        "video_id": video_id,
                        "timestamp_ms": timestamp_ms,
                        "timestamp_source": track_row.get("timestamp_source", "unknown"),
                        "frame_index": frame_index,
                        "status": estimate.status,
                        "quality": estimate.quality,
                        "confidence": estimate.confidence,
                        "reasons": list(estimate.reasons),
                        "corridors": estimate.to_dict()["corridors"],
                        "scene_map_size": [estimate.width, estimate.height],
                        "inference_error": inference_error,
                        "inference_status": inference_status,
                        "inference_latency_ms": current_infer_ms,
                        "full_scene_processing_ms": complete_ms,
                        "result_ttl_ms": result_ttl_ms,
                        "camera_pose_status": "unobserved",
                        "metric_depth_status": "unavailable",
                    }
                    scene_stream.write(
                        json.dumps(scene_row, ensure_ascii=False, allow_nan=False) + "\n"
                    )
                    frames_out.append(scene_row)
                    overlay = _draw_overlay(frame, estimate, object_assessments, complete_ms)
                    overlay_path = overlay_dir / f"{frame_index:06d}.jpg"
                    if not cv2.imwrite(str(overlay_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 88]):
                        raise OSError(f"Could not write overlay: {overlay_path}")
                    total_ms.append((time.perf_counter() - full_started) * 1000.0)
                    if frame_index and frame_index % (frame_step * 20) == 0:
                        LOGGER.info(
                            "%s: frame %d/%d; sampled %d",
                            video_id,
                            frame_index,
                            reported_frames,
                            len(frames_out),
                        )
                frame_index += 1
    finally:
        capture.release()

    if not frames_out:
        raise ValueError(f"No source frames sampled for {video_id}")
    if frame_index != len(tracks) or (reported_frames > 0 and frame_index != reported_frames):
        raise ValueError(
            f"Source decode ended at frame {frame_index}; tracks={len(tracks)}, "
            f"container_reported={reported_frames}"
        )
    video_peak_memory = metadata.get("peak_cuda_memory_allocated_mb")
    if model is not None and model.device.type == "cuda":
        video_peak_memory = model.torch.cuda.max_memory_allocated(model.device) / (1024**2)
        metadata["peak_cuda_memory_allocated_mb"] = max(
            float(metadata.get("peak_cuda_memory_allocated_mb", 0.0)), video_peak_memory
        )
    hashes = {
        "video_sha256": _sha256(video_path),
        "tracks_sha256": _sha256(tracks_path),
        "risk_frames_sha256": _sha256(risk_dir / "risk_frames.jsonl"),
        "events_sha256": _sha256(risk_dir / "events.jsonl"),
        "alerts_sha256": _sha256(risk_dir / "alerts.jsonl"),
        "model_sha256": _sha256(model_path) if model_path.is_file() else None,
    }
    summary = _update_summary(
        frames_out,
        object_out,
        shadow_out,
        video_id=video_id,
        sample_interval_s=sample_interval_s,
        total_frames=frame_index,
        model_load_ms=metadata["model_load_ms"],
        inference_durations=infer_ms,
        full_durations=total_ms,
        device_names=devices,
        peak_gpu_memory_mb=video_peak_memory,
        input_hashes=hashes,
        baseline_alerts=alerts,
    )
    summary["source_fps"] = fps
    summary["sampled_fps"] = fps / frame_step
    summary["result_ttl_ms"] = result_ttl_ms
    summary["scene_results_are_same_frame"] = True
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    return summary, model


def run_path_relevance_batch(
    manifest_path: Path,
    baseline_dir: Path,
    output_dir: Path,
    *,
    checkpoint_path: Path,
    device: str = "auto",
    sample_interval_s: float = 0.5,
    input_width: int = 512,
    min_pixel_confidence: float = 0.35,
    result_ttl_ms: float = 500.0,
    allow_cpu_fallback: bool = True,
    min_area_fraction: float = 0.005,
    bottom_start_fraction: float = 0.82,
    min_vertical_extent_fraction: float = 0.35,
    geometry_roi: tuple[float, float, float] = (1 / 3, 2 / 3, 0.55),
    medium_threshold: float = PATH_RELEVANCE_MEDIUM_MIN,
    high_threshold: float = PATH_RELEVANCE_HIGH_MIN,
    video_ids: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Run the perceptual Shadow experiment against previously saved artifacts."""
    manifest_path = manifest_path.resolve()
    baseline_dir = baseline_dir.resolve()
    output_dir = output_dir.resolve()
    checkpoint_path = checkpoint_path.resolve()
    if sample_interval_s <= 0 or not math.isfinite(sample_interval_s):
        raise ValueError("sample_interval_s must be finite and positive")
    if result_ttl_ms < 0 or not math.isfinite(result_ttl_ms):
        raise ValueError("result_ttl_ms must be finite and nonnegative")
    if not 0 <= medium_threshold < high_threshold <= 1:
        raise ValueError("path relevance thresholds must satisfy 0 <= medium < high <= 1")
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output already exists and is non-empty: {output_dir}")
    manifest = _read_json(manifest_path)
    rows = manifest.get("videos")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Real-world manifest must contain a non-empty videos array")
    requested = set(video_ids) if video_ids is not None else None
    selected = [row for row in rows if requested is None or row.get("video_id") in requested]
    if not selected or (
        requested is not None and {row.get("video_id") for row in selected} != requested
    ):
        raise ValueError("Requested video IDs do not match the manifest")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch as torch_module
    except ImportError:
        torch_module = None
    model_started = time.perf_counter()
    initial_fallback: dict[str, Any] | None = None
    initial_model_error: str | None = None
    try:
        model = SegformerB0Perceptor(checkpoint_path, device=device, input_width=input_width)
    except ModelUnavailableError as exc:
        model = None
        initial_model_error = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        requested_cuda = device == "auto" or device.startswith(("cuda", "0", "1", "2", "3"))
        is_oom = bool(torch_module and isinstance(exc, torch_module.cuda.OutOfMemoryError))
        if not (allow_cpu_fallback and requested_cuda and is_oom):
            raise
        torch_module.cuda.empty_cache()
        gc.collect()
        initial_fallback = {
            "video_id": None,
            "frame_index": None,
            "reason": f"{type(exc).__name__}: {exc}",
        }
        model = SegformerB0Perceptor(checkpoint_path, device="cpu", input_width=input_width)
    model_load_ms = (time.perf_counter() - model_started) * 1000.0
    torch = model.torch if model is not None else torch_module
    if model is not None and model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model.device)
    try:
        transformers_version = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        transformers_version = "unknown"
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "project_version": __version__,
        "manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
        "baseline_directory": str(baseline_dir),
        "model": {
            "name": "SegFormer MIT-B0 ADE20K 160k",
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path) if checkpoint_path.is_file() else None,
            "source_url": MODEL_CHECKPOINT_URL,
            "upstream_repository": "https://github.com/open-mmlab/mmsegmentation",
            "upstream_repository_license": "Apache-2.0",
            "checkpoint_license_status": (
                "not separately stated in the model zoo entry; not independently verified"
            ),
            "license_note": (
                "The repository license does not by itself establish a separate license "
                "for the downloaded checkpoint or its training data."
            ),
            "runtime": (
                "Transformers SegformerModel encoder + checkpoint-compatible "
                "OpenMMLab decode head adapter"
            ),
            "transformers_version": transformers_version,
            "input_width": input_width,
            "load_ms": model_load_ms,
            "status": "available" if model is not None else "unavailable",
            "unavailable_reason": initial_model_error,
        },
        "config": {
            "sample_interval_seconds": sample_interval_s,
            "min_pixel_confidence": min_pixel_confidence,
            "result_ttl_ms": result_ttl_ms,
            "allow_cpu_fallback_on_cuda_oom": allow_cpu_fallback,
            "min_area_fraction": min_area_fraction,
            "bottom_start_fraction": bottom_start_fraction,
            "min_vertical_extent_fraction": min_vertical_extent_fraction,
            "geometry_baseline_roi": {
                "x_start_fraction": geometry_roi[0],
                "x_end_fraction": geometry_roi[1],
                "y_start_fraction": geometry_roi[2],
            },
            "score_levels": {
                "medium_min": medium_threshold,
                "high_min": high_threshold,
            },
            "device_requested": device,
        },
        "model_load_ms": model_load_ms,
        "runtime": {
            "initial_device": model.device_name if model is not None else "unavailable",
            "cuda_available": bool(torch and torch.cuda.is_available()),
            "torch_version": torch.__version__ if torch else "not installed",
            "torch_cuda_version": torch.version.cuda if torch else None,
            "numpy_version": np.__version__,
            "opencv_version": _installed_version(
                "opencv-python", "opencv-python-headless", "opencv-contrib-python"
            ),
            "gpu_name": (
                torch.cuda.get_device_name(0) if torch and torch.cuda.is_available() else None
            ),
            "python_version": platform.python_version(),
        },
        "videos": [],
        "cpu_fallbacks": [initial_fallback] if initial_fallback else [],
        "limitations": [
            "No calibrated camera pose or metric depth is available.",
            "Road, sidewalk, and floor labels indicate semantics, not traversability or safety.",
            (
                "Video timestamps align sampled scene frames to saved track frame indices; "
                "no detector rerun occurs."
            ),
            "All outputs are research-only Shadow records and do not modify baseline alerts.",
        ],
    }
    metadata_path = output_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        video_summaries = []
        for manifest_row in selected:
            video_id = str(manifest_row["video_id"])
            video_path = Path(manifest_row["filepath"])
            video_dir = baseline_dir / video_id
            tracks_path = video_dir / "perception" / "tracks.jsonl"
            risk_dir = video_dir / "assessment"
            summary, model = _process_one_video(
                video_id=video_id,
                video_path=video_path,
                tracks_path=tracks_path,
                risk_dir=risk_dir,
                output_dir=output_dir / video_id,
                model=model,
                model_path=checkpoint_path,
                sample_interval_s=sample_interval_s,
                min_pixel_confidence=min_pixel_confidence,
                result_ttl_ms=result_ttl_ms,
                allow_cpu_fallback=allow_cpu_fallback,
                min_area_fraction=min_area_fraction,
                bottom_start_fraction=bottom_start_fraction,
                min_vertical_extent_fraction=min_vertical_extent_fraction,
                geometry_roi=geometry_roi,
                medium_threshold=medium_threshold,
                high_threshold=high_threshold,
                metadata=metadata,
            )
            video_summaries.append(summary)
            metadata["videos"] = video_summaries
            metadata["status"] = "running"
            metadata_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            LOGGER.info(
                "%s: %d/%d scenes have supported ground candidates; "
                "%.1f ms mean full scene processing",
                video_id,
                summary["sampled_scene_frames"] - summary["unknown_scene_frames"],
                summary["sampled_scene_frames"],
                summary["full_sample_processing_ms_mean"] or 0,
            )
        if model is not None and model.device.type == "cuda":
            live_memory_peak = torch.cuda.max_memory_allocated(model.device) / (1024**2)
            metadata["peak_cuda_memory_allocated_mb"] = max(
                float(metadata.get("peak_cuda_memory_allocated_mb", 0.0)),
                live_memory_peak,
            )
        memory_peak = metadata.get("peak_cuda_memory_allocated_mb")
        metadata["runtime"]["final_device"] = (
            model.device_name if model is not None else "unavailable"
        )
        metadata["runtime"]["peak_cuda_memory_allocated_mb"] = memory_peak
        metadata["summary"] = {
            "videos_completed": len(video_summaries),
            "sampled_scene_frames": sum(row["sampled_scene_frames"] for row in video_summaries),
            "valid_scene_frames": sum(
                row["sampled_scene_frames"] - row["unknown_scene_frames"] for row in video_summaries
            ),
            "objects_assessed": sum(row["tracked_objects_assessed"] for row in video_summaries),
            "baseline_alerts_unchanged": sum(
                row["comparison"]["A_baseline"]["unchanged_alert_count"] for row in video_summaries
            ),
            "comparison_by_video": {row["video_id"]: row["comparison"] for row in video_summaries},
            "shadow_recommendations": dict(
                sum(
                    (Counter(row["shadow_recommendation_counts"]) for row in video_summaries),
                    Counter(),
                )
            ),
            "human_visual_attributes": "pending/null; not scored",
        }
        metadata["status"] = "completed"
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        metadata["finished_at_utc"] = datetime.now(UTC).isoformat()
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(metadata["summary"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


def load_path_relevance_rows(
    path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Read completed records for the Research Console (read-only)."""
    if not path.is_file():
        raise FileNotFoundError(path)
    return (
        _read_jsonl(path / "scene_frames.jsonl"),
        _read_jsonl(path / "objects.jsonl"),
        _read_jsonl(path / "shadow.jsonl"),
    )
