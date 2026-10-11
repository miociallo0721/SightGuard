"""Saved-input Shadow replay and four exact-source ablations."""

from __future__ import annotations

import json
import platform
import subprocess
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from sightguard.perception.depth.artifacts import sha256_file, write_json
from sightguard.scene.spatial_grid.artifacts import fresh_directory, load_inputs, read_rows, row_key
from sightguard.scene.spatial_grid.engine import SpatialGridFusion
from sightguard.scene.spatial_grid.schema import (
    EvidenceSource,
    SpatialGridConfig,
    SpatialGridFrame,
    SpatialGridInput,
)
from sightguard.visualization.spatial_grid import (
    save_spatial_preview,
    write_evidence_script,
    write_spatial_viewer,
)

ABLATIONS = {"A": "SegFormer Only", "B": "Depth Only", "C": "YOLO + SegFormer", "D": "All three"}


def ablation_input(data: SpatialGridInput, name: str) -> SpatialGridInput:
    """Exclude entire sources, not just their display or final labels."""
    if name not in ABLATIONS:
        raise ValueError("unknown ablation")
    result = data
    if name == "B":
        result = replace(
            result,
            semantic_source=EvidenceSource(data.key, "excluded"),
            class_map=None,
            confidence_map=None,
            candidate_components=None,
        )
    if name in ("A", "B"):
        result = replace(
            result, detection_source=EvidenceSource(data.key, "excluded"), detections=()
        )
    if name in ("A", "C"):
        result = replace(result, depth_source=EvidenceSource(data.key, "excluded"), depth_map=None)
    return result


def evidence_metrics(frame: SpatialGridFrame) -> dict[str, Any]:
    cells = frame.cells
    return {
        "key": asdict(frame.key),
        "cells": len(cells),
        "states": dict(Counter(x.state for x in cells)),
        "ground_supported": sum(x.ground.status == "supported" for x in cells),
        "depth_supported": sum(x.depth.valid_count > 0 for x in cells),
        "bbox_covered": sum(bool(x.detection_coverage) for x in cells),
        "conflict_cells": sum(bool(x.conflict_reasons) for x in cells),
        "cells_with_unknown_reasons": sum(bool(x.unknown_reasons) for x in cells),
        "invalid_depth_pixels": sum(
            round(
                x.depth.invalid_fraction
                * ((x.bounds_xyxy[2] - x.bounds_xyxy[0]) * (x.bounds_xyxy[3] - x.bounds_xyxy[1]))
            )
            for x in cells
            if x.source_status["depth"] == "valid"
        ),
        "object_relations": len(frame.object_relations),
        "objects_with_geometry": sum(bool(x.cell_coverage) for x in frame.object_relations),
        "objects_with_depth": sum(x.depth.valid_count > 0 for x in frame.object_relations),
        "objects_with_semantics": sum(
            x.ground_fraction is not None for x in frame.object_relations
        ),
        "failures": list(frame.failures),
        "source_status": frame.source_status,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    states: Counter = Counter()
    fields = (
        "cells",
        "ground_supported",
        "depth_supported",
        "bbox_covered",
        "conflict_cells",
        "cells_with_unknown_reasons",
        "invalid_depth_pixels",
        "object_relations",
        "objects_with_geometry",
        "objects_with_depth",
        "objects_with_semantics",
    )
    summary = {name: sum(x[name] for x in rows) for name in fields}
    for row in rows:
        states.update(row["states"])
    summary.update(
        {
            "frames": len(rows),
            "states": dict(states),
            "unknown_state_fraction": states["UNKNOWN"] / summary["cells"] if rows else None,
            "failure_frames": sum(bool(x["failures"]) for x in rows),
        }
    )
    return summary


def percentiles(rows: list[dict[str, float]]) -> dict[str, Any]:
    keys = sorted({name for row in rows for name in row})
    return {
        name: {
            "p50_ms": float(np.percentile([x[name] for x in rows if name in x], 50)),
            "p95_ms": float(np.percentile([x[name] for x in rows if name in x], 95)),
            "samples": sum(name in x for x in rows),
        }
        for name in keys
    }


def memory_snapshot() -> dict[str, float]:
    """Windows process peak working set; Linux fallback reports ru_maxrss in KiB."""
    if platform.system() == "Windows":
        import ctypes
        from ctypes import wintypes

        class MemoryCounters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t)
                for name in (
                    "PeakWorkingSetSize",
                    "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage",
                    "PagefileUsage",
                    "PeakPagefileUsage",
                )
            ]

        counter = MemoryCounters()
        counter.cb = ctypes.sizeof(counter)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        if not psapi.GetProcessMemoryInfo(
            kernel.GetCurrentProcess(), ctypes.byref(counter), counter.cb
        ):
            raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
        return {
            "working_set_mib": counter.WorkingSetSize / 2**20,
            "process_peak_working_set_mib": counter.PeakWorkingSetSize / 2**20,
        }
    import resource

    return {
        "process_peak_working_set_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    }


def _index(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    result = {x["frame_index"]: x for x in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate frame identities")
    return result


def run_spatial_batch(
    manifest_path: Path,
    tracks_root: Path,
    semantic_root: Path,
    depth_root: Path,
    output: Path,
    config_path: Path,
    *,
    video_ids: tuple[str, ...] = (),
    limit: int | None = None,
    visualize: bool = True,
) -> dict[str, Any]:
    """No model loading or production decision calls; reject existing outputs."""
    config = SpatialGridConfig(**yaml.safe_load(config_path.read_text(encoding="utf-8")))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fresh_directory(output)
    engine = SpatialGridFusion(config)
    metadata: dict[str, Any] = {
        "schema_version": "sightguard.spatial-grid-batch.v1",
        "status": "running",
        "config": asdict(config),
        "config_sha256": sha256_file(config_path),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_schema": manifest["schema_version"],
        "semantic_metadata_sha256": sha256_file(semantic_root / "run_metadata.json"),
        "depth_metadata_sha256": sha256_file(depth_root / "run_metadata.json"),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {
            str(p): sha256_file(p)
            for p in (
                Path(__file__),
                *Path("src/sightguard/scene/spatial_grid").glob("*.py"),
                Path("src/sightguard/visualization/spatial_grid.py"),
                Path("src/sightguard/perception/depth/artifacts.py"),
            )
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "platform": platform.platform(),
        },
        "production_decision_changed": False,
        "videos": [],
        "error": None,
        "timing_scope": (
            "D full_sample includes selected RGB decode, input hash/NPZ decode/align, "
            "fusion, JSONL and preview/inspector writes; ABC separate; model rerun excluded"
        ),
    }
    write_json(output / "run_metadata.json", metadata)
    batch_metrics: dict[str, list] = {name: [] for name in ABLATIONS}
    all_timings: list[dict[str, float]] = []
    samples = []
    started = time.perf_counter()
    try:
        for video in manifest["videos"]:
            video_id = video["video_id"]
            if video_ids and video_id not in video_ids:
                continue
            directory = output / video_id
            fresh_directory(directory)
            fresh_directory(directory / "frames")
            depth_dir, semantic_dir = depth_root / video_id, semantic_root / video_id
            tracks_path = tracks_root / video_id / "perception/tracks.jsonl"
            tracks = _index(read_rows(tracks_path))
            semantic_path = semantic_dir / "semantic_frames.jsonl"
            semantics = _index(read_rows(semantic_path)) if semantic_path.exists() else {}
            depth_rows = read_rows(depth_dir / "depth_frames.jsonl")
            _index(depth_rows)
            if limit is not None:
                depth_rows = depth_rows[:limit]
            hashes = {
                str(p): sha256_file(p)
                for p in (tracks_path, depth_dir / "depth_frames.jsonl", Path(video["filepath"]))
            }
            if semantic_path.exists():
                hashes[str(semantic_path)] = sha256_file(semantic_path)
            if hashes[video["filepath"]] != video["processed_sha256"]:
                raise ValueError(f"video SHA-256 mismatch: {video_id}")
            capture = cv2.VideoCapture(str(video["filepath"]))
            video_metrics: dict[str, list] = {name: [] for name in ABLATIONS}
            timings_list = []
            streams = {
                name: (directory / name).open("w", encoding="utf-8")
                for name in (
                    "spatial_frames.jsonl",
                    "object_relations.jsonl",
                    "ablation_metrics.jsonl",
                )
            }
            try:
                for depth_row in depth_rows:
                    tick = time.perf_counter()
                    timings: dict[str, float] = {}
                    key = row_key(depth_row)
                    if key.video_id != video_id:
                        raise ValueError("cross-video source")
                    track_row = tracks.get(key.frame_index)
                    if track_row is None:
                        track_row = {
                            "frame_index": key.frame_index,
                            "timestamp_ms": key.timestamp_ms,
                            "frame_width": key.width,
                            "frame_height": key.height,
                            "status": "unknown",
                            "detections": [],
                        }
                    decode_started = time.perf_counter()
                    capture.set(cv2.CAP_PROP_POS_FRAMES, key.frame_index)
                    ok, rgb = capture.read()
                    if (
                        not ok
                        or rgb.shape[:2] != (key.height, key.width)
                        or abs(capture.get(cv2.CAP_PROP_POS_MSEC) - key.timestamp_ms) > 0.01
                    ):
                        raise ValueError(f"RGB exact-frame decode failed: {key}")
                    timings["rgb_decode"] = (time.perf_counter() - decode_started) * 1000
                    data = load_inputs(
                        key,
                        track_row,
                        tracks_path,
                        semantics.get(key.frame_index),
                        semantic_dir,
                        depth_row,
                        depth_dir,
                        tracks_sha256=hashes[str(tracks_path)],
                        timings=timings,
                    )
                    fusion_started = time.perf_counter()
                    result = engine.fuse(data, timings=timings)
                    timings["fusion_total"] = (time.perf_counter() - fusion_started) * 1000
                    serialization = time.perf_counter()
                    serialized = json.dumps(
                        result.to_dict(), allow_nan=False, separators=(",", ":")
                    )
                    streams["spatial_frames.jsonl"].write(serialized + "\n")
                    for relation in result.object_relations:
                        streams["object_relations.jsonl"].write(
                            json.dumps({"key": asdict(key), **asdict(relation)}, allow_nan=False)
                            + "\n"
                        )
                    streams["spatial_frames.jsonl"].flush()
                    streams["object_relations.jsonl"].flush()
                    timings["jsonl_serialize_write"] = (time.perf_counter() - serialization) * 1000
                    visual_started = time.perf_counter()
                    if visualize:
                        references = save_spatial_preview(
                            rgb,
                            data,
                            result,
                            directory / "previews",
                            min_pixel_confidence=config.min_pixel_confidence,
                        )
                        write_evidence_script(
                            directory / "frames" / f"{key.frame_index:06d}.js", result, references
                        )
                        samples.append({"video_id": video_id, "frame_index": key.frame_index})
                    timings["visualization_and_inspector"] = (
                        time.perf_counter() - visual_started
                    ) * 1000
                    timings["full_sample"] = (time.perf_counter() - tick) * 1000
                    for name in ABLATIONS:
                        ablation_started = time.perf_counter()
                        fused = result if name == "D" else engine.fuse(ablation_input(data, name))
                        timings[f"ablation_{name}"] = (
                            time.perf_counter() - ablation_started
                        ) * 1000
                        metric = evidence_metrics(fused)
                        video_metrics[name].append(metric)
                        batch_metrics[name].append(metric)
                        streams["ablation_metrics.jsonl"].write(
                            json.dumps({"ablation": name, **metric}, allow_nan=False) + "\n"
                        )
                    timings_list.append(timings)
                    all_timings.append(timings)
            finally:
                capture.release()
                for stream in streams.values():
                    stream.close()
            summary = {
                "video_id": video_id,
                "ablations": {name: summarize(rows) for name, rows in video_metrics.items()},
                "timings_ms": percentiles(timings_list),
            }
            write_json(directory / "summary.json", summary)
            write_json(directory / "timings.json", timings_list)
            write_json(
                directory / "run_metadata.json",
                {
                    "video": video,
                    "input_sha256": hashes,
                    "config": asdict(config),
                    "parent_metadata": "../run_metadata.json",
                    "frames": len(depth_rows),
                    "source_schemas": {
                        "tracks": "1.0",
                        "depth": "sightguard.depth.v1",
                        "semantic": "sightguard.spatial-semantic.v1",
                    },
                },
            )
            metadata["videos"].append(summary)
        if visualize:
            write_spatial_viewer(output, samples)
        metadata["ablations"] = {
            name: {"name": ABLATIONS[name], **summarize(rows)}
            for name, rows in batch_metrics.items()
        }
        metadata["timings_ms"] = percentiles(all_timings)
        metadata["memory_mib"] = memory_snapshot()
        metadata["status"] = "completed"
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        metadata["wall_seconds"] = time.perf_counter() - started
        write_json(output / "run_metadata.json", metadata)
        write_json(output / "summary.json", metadata)
    return metadata
