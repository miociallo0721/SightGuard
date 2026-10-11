"""Minimal real SegFormer rerun when earlier Path Relevance saved no raw maps."""

from __future__ import annotations

import gc
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from sightguard.path_relevance.engine import estimate_corridors
from sightguard.path_relevance.model import SegformerB0Perceptor
from sightguard.perception.depth.artifacts import sha256_file, write_json
from sightguard.scene.spatial_grid.artifacts import SEMANTIC_SCHEMA, fresh_directory, read_rows


def prepare_semantic_batch(
    manifest_path: Path,
    depth_root: Path,
    output: Path,
    path_config: Path,
    *,
    device: str = "auto",
    video_ids: tuple[str, ...] = (),
    limit: int | None = None,
) -> dict[str, Any]:
    """Persist unmodified native maps and independent same-adapter ground components."""
    import json

    config = yaml.safe_load(path_config.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint = Path(config["model"]["checkpoint"])
    fresh_directory(output)
    model = None
    metadata: dict[str, Any] = {
        "schema_version": SEMANTIC_SCHEMA,
        "status": "running",
        "config": config,
        "reason": "Path Relevance v04 contains no raw masks; independent same-adapter rerun",
        "manifest_sha256": sha256_file(manifest_path),
        "config_sha256": sha256_file(path_config),
        "checkpoint_sha256": sha256_file(checkpoint),
        "videos": [],
        "failures": [],
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {
            str(p): sha256_file(p)
            for p in (
                Path(__file__),
                Path("src/sightguard/path_relevance/model.py"),
                Path("src/sightguard/path_relevance/engine.py"),
            )
        },
        "python": platform.python_version(),
        "alignment": "native 288x512 maps retained; source alignment uses nearest for all maps",
    }
    write_json(output / "run_metadata.json", metadata)
    started = time.perf_counter()
    try:
        model = SegformerB0Perceptor(
            checkpoint, device=device, input_width=config["model"]["input_width"]
        )
        torch = model.torch
        metadata["device"] = model.device_name
        metadata["torch"] = torch.__version__
        import transformers

        metadata["transformers"] = transformers.__version__
        metadata["model_load_ms"] = (time.perf_counter() - started) * 1000
        if model.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for video in manifest["videos"]:
            video_id = video["video_id"]
            if video_ids and video_id not in video_ids:
                continue
            directory = output / video_id
            fresh_directory(directory)
            fresh_directory(directory / "maps")
            capture = cv2.VideoCapture(str(video["filepath"]))
            if sha256_file(Path(video["filepath"])) != video["processed_sha256"]:
                capture.release()
                raise ValueError(f"video hash mismatch: {video_id}")
            rows = read_rows(depth_root / video_id / "depth_frames.jsonl")
            if limit is not None:
                rows = rows[:limit]
            saved_rows = []
            try:
                for row in rows:
                    tick = time.perf_counter()
                    capture.set(cv2.CAP_PROP_POS_FRAMES, row["frame_index"])
                    ok, frame = capture.read()
                    record = {
                        "schema_version": SEMANTIC_SCHEMA,
                        "video_id": video_id,
                        "frame_index": row["frame_index"],
                        "timestamp_ms": row["timestamp_ms"],
                        "timestamp_source": row["timestamp_source"],
                        "source_frame_width": row["source_frame_width"],
                        "source_frame_height": row["source_frame_height"],
                        "status": "unknown",
                        "quality": "unknown",
                        "reasons": [],
                        "data_reference": None,
                        "data_sha256": None,
                        "checkpoint_sha256": metadata["checkpoint_sha256"],
                    }
                    try:
                        if not ok or frame.shape[:2] != (
                            row["source_frame_height"],
                            row["source_frame_width"],
                        ):
                            raise ValueError("selected frame unavailable or dimensions mismatch")
                        pts = capture.get(cv2.CAP_PROP_POS_MSEC)
                        if abs(pts - row["timestamp_ms"]) > 0.01:
                            raise ValueError("decoded PTS differs from saved source timestamp")
                        # Warmup remains outside the measured 240 source predictions.
                        if not metadata.get("warmup_completed"):
                            for _ in range(2):
                                model.predict(frame)
                            metadata["warmup_completed"] = 2
                        inference_started = time.perf_counter()
                        prediction = model.predict(frame)
                        record["adapter_full_prediction_ms"] = (
                            time.perf_counter() - inference_started
                        ) * 1000
                        estimate = estimate_corridors(
                            prediction,
                            min_pixel_confidence=config["inference"]["min_pixel_confidence"],
                            min_area_fraction=config["corridor"]["min_area_fraction"],
                            bottom_start_fraction=config["corridor"]["bottom_start_fraction"],
                            min_vertical_extent_fraction=config["corridor"][
                                "min_vertical_extent_fraction"
                            ],
                        )
                        _, labels = cv2.connectedComponents(estimate.mask, connectivity=8)
                        reference = f"maps/{row['frame_index']:06d}.npz"
                        np.savez_compressed(
                            directory / reference,
                            class_map=prediction.class_map,
                            confidence_map=prediction.confidence_map,
                            candidate_components=labels.astype(np.int32),
                        )
                        record.update(
                            {
                                "status": estimate.status,
                                "quality": estimate.quality,
                                "reasons": list(estimate.reasons),
                                "native_shape": list(prediction.class_map.shape),
                                "data_reference": reference,
                                "data_sha256": sha256_file(directory / reference),
                                "decoded_pts_ms": pts,
                                "inference_latency_ms": prediction.inference_latency_ms,
                                "scene_estimate": estimate.to_dict(),
                            }
                        )
                    except Exception as exc:
                        record["error"] = f"{type(exc).__name__}: {exc}"
                        metadata["failures"].append(
                            {
                                "video_id": video_id,
                                "frame_index": row["frame_index"],
                                "error": record["error"],
                            }
                        )
                    record["full_sample_ms"] = (time.perf_counter() - tick) * 1000
                    saved_rows.append(record)
                    with (directory / "semantic_frames.jsonl").open(
                        "a", encoding="utf-8"
                    ) as stream:
                        stream.write(json.dumps(record, allow_nan=False) + "\n")
            finally:
                capture.release()
            metadata["videos"].append(
                {
                    "video_id": video_id,
                    "frames": len(saved_rows),
                    "valid": sum(x["status"] == "valid" for x in saved_rows),
                }
            )
            write_json(directory / "summary.json", metadata["videos"][-1])
        metadata["gpu_memory_mib"] = (
            {
                "peak_allocated": torch.cuda.max_memory_allocated() / 2**20,
                "peak_reserved": torch.cuda.max_memory_reserved() / 2**20,
            }
            if model.device.type == "cuda"
            else {}
        )
        metadata["status"] = "completed" if not metadata["failures"] else "partial"
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["failures"].append({"error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        if model is not None:
            torch = model.torch
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        metadata["wall_seconds"] = time.perf_counter() - started
        write_json(output / "run_metadata.json", metadata)
    return metadata
