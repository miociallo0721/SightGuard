"""Verified offline adapters; source arrays are never modified."""

from __future__ import annotations

import json
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from sightguard.perception.depth.artifacts import load_depth_map, sha256_file
from sightguard.perception.depth.schema import DepthFrameResult
from sightguard.scene.spatial_grid.schema import EvidenceSource, FrameKey, SpatialGridInput

SEMANTIC_SCHEMA = "sightguard.spatial-semantic.v1"


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def fresh_directory(path: Path) -> None:
    """Reject even empty existing directories; never overwrite a previous run."""
    path.mkdir(parents=True, exist_ok=False)


def row_key(row: dict[str, Any], video_id: str | None = None) -> FrameKey:
    return FrameKey(
        row.get("video_id", video_id),
        row["frame_index"],
        row["timestamp_ms"],
        row.get("source_frame_width", row.get("frame_width")),
        row.get("source_frame_height", row.get("frame_height")),
    )


def load_semantic_maps(
    directory: Path,
    row: dict[str, Any],
    key: FrameKey,
    *,
    timings: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Check identity, schema, checksum and native shapes, then nearest-align maps."""
    if row.get("schema_version") != SEMANTIC_SCHEMA or row_key(row) != key:
        raise ValueError("semantic schema or identity mismatch")
    if row.get("status") != "valid":
        raise ValueError("semantic scene UNKNOWN")
    reference = row["data_reference"]
    if not isinstance(reference, str) or Path(reference).is_absolute():
        raise ValueError("semantic reference must be relative")
    path = (directory / reference).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError("semantic reference escapes directory")
    if sha256_file(path) != row["data_sha256"]:
        raise ValueError("semantic SHA-256 mismatch")
    with np.load(path, allow_pickle=False) as saved:
        classes = saved["class_map"]
        confidence = saved["confidence_map"]
        components = saved["candidate_components"]
    expected = tuple(row["native_shape"])
    if (
        classes.shape != expected
        or confidence.shape != expected
        or components.shape != expected
        or classes.dtype != np.uint8
        or confidence.dtype != np.float32
        or components.dtype != np.int32
        or not np.isfinite(confidence).all()
        or np.any(confidence < 0)
        or np.any(confidence > 1)
        or np.any(classes > 149)
        or np.any(components < 0)
    ):
        raise ValueError("invalid native semantic maps")
    import time

    alignment_started = time.perf_counter()
    aligned = tuple(
        cv2.resize(x, (key.width, key.height), interpolation=cv2.INTER_NEAREST)
        for x in (classes, confidence, components)
    )
    if timings is not None:
        timings["mask_alignment"] = (time.perf_counter() - alignment_started) * 1000
    return aligned


def load_inputs(
    key: FrameKey,
    tracks_row: dict[str, Any],
    tracks_path: Path,
    semantic_row: dict[str, Any] | None,
    semantic_dir: Path,
    depth_row: dict[str, Any] | None,
    depth_dir: Path,
    *,
    tracks_sha256: str | None = None,
    timings: dict[str, float] | None = None,
) -> SpatialGridInput:
    """Source failures become explicit partial evidence, never stale substitution."""
    import time

    started = time.perf_counter()
    detection_key = row_key(tracks_row, key.video_id)
    detection_source = EvidenceSource(
        detection_key,
        "valid" if tracks_row.get("status") == "ok" else "unknown",
        "limited",
        str(tracks_path),
        tracks_sha256 or sha256_file(tracks_path),
        tracks_row.get("schema_version"),
    )
    semantic_source = None
    class_map = confidence = components = None
    align_started = time.perf_counter()
    if semantic_row is not None:
        try:
            semantic_source = EvidenceSource(
                row_key(semantic_row),
                semantic_row["status"],
                semantic_row["quality"],
                str(semantic_dir / (semantic_row.get("data_reference") or "semantic_frames.jsonl")),
                semantic_row.get("data_sha256"),
                semantic_row["schema_version"],
                tuple(semantic_row.get("reasons", ())),
            )
            class_map, confidence, components = load_semantic_maps(
                semantic_dir, semantic_row, key, timings=timings
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            semantic_source = replace(
                semantic_source or EvidenceSource(key, "unknown"),
                status="unknown",
                reasons=(f"semantic_read_failed:{type(exc).__name__}:{exc}",),
            )
    if timings is not None:
        timings["semantic_load_hash_decode_align"] = (time.perf_counter() - align_started) * 1000
    depth_source = None
    depth = None
    depth_started = time.perf_counter()
    if depth_row is not None:
        try:
            depth_source = EvidenceSource(
                row_key(depth_row),
                depth_row.get("status", "unknown"),
                "limited",
                str(depth_dir / (depth_row.get("depth_data_reference") or "depth_frames.jsonl")),
                depth_row.get("depth_data_sha256"),
                depth_row.get("schema_version"),
            )
            record = DepthFrameResult(
                **{
                    x.name: depth_row[x.name]
                    for x in fields(DepthFrameResult)
                    if x.name in depth_row
                }
            )
            if not record.matches_frame(key.video_id, key.frame_index, key.timestamp_ms):
                raise ValueError("depth exact frame mismatch or UNKNOWN")
            if row_key(depth_row) != key:
                raise ValueError("depth source dimensions mismatch")
            depth = load_depth_map(depth_dir, depth_row)
            depth_source = EvidenceSource(
                key,
                "valid",
                "limited",
                str(depth_dir / depth_row["depth_data_reference"]),
                depth_row["depth_data_sha256"],
                depth_row["schema_version"],
                tuple(depth_row.get("quality_reasons", ())),
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            depth_source = replace(
                depth_source or EvidenceSource(key, "unknown"),
                status="unknown",
                reasons=(f"depth_read_failed:{type(exc).__name__}:{exc}",),
            )
    if timings is not None:
        timings["depth_load_hash_decode"] = (time.perf_counter() - depth_started) * 1000
        timings["input_load_total"] = (time.perf_counter() - started) * 1000
    return SpatialGridInput(
        key,
        semantic_source,
        depth_source,
        detection_source,
        class_map,
        confidence,
        components,
        depth,
        tuple(tracks_row.get("detections", ())),
    )
