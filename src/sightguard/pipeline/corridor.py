"""CPU saved-grid replay, strict preflight and same-source corridor ablations."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from sightguard.perception.depth.artifacts import sha256_file, write_json
from sightguard.pipeline.spatial_grid import memory_snapshot, percentiles
from sightguard.scene.corridor.artifacts import parse_spatial_frame
from sightguard.scene.corridor.engine import ABLATIONS, CorridorEvaluator
from sightguard.scene.corridor.schema import CandidateConfig, CorridorConfig, EvaluationConfig
from sightguard.scene.spatial_grid.artifacts import fresh_directory, read_rows
from sightguard.scene.spatial_grid.schema import FrameKey
from sightguard.visualization.corridor import save_preview, write_viewer


def load_config(path: Path) -> CorridorConfig:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    candidates = dict(value["candidates"])
    for name in ("names", "top_centers", "bottom_centers"):
        if name in candidates:
            candidates[name] = tuple(candidates[name])
    return CorridorConfig(
        CandidateConfig(**candidates),
        EvaluationConfig(**value["evaluation"]),
        value["schema_version"],
    )


def preflight(root: Path, manifest_path: Path) -> list[dict[str, Any]]:
    """Validate every saved frame and immutable parent source before evaluation."""
    parent = json.loads((root / "run_metadata.json").read_text(encoding="utf-8"))
    if parent["status"] != "completed" or parent["production_decision_changed"]:
        raise ValueError("incomplete or production-modifying parent run")
    if sha256_file(manifest_path) != parent["manifest_sha256"]:
        raise ValueError("manifest hash mismatch")
    for path, digest in parent["source_sha256"].items():
        if sha256_file(Path(path)) != digest:
            raise ValueError(f"spatial producer code hash mismatch: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    videos = {x["video_id"]: x for x in manifest["videos"]}
    result = []
    checked_hashes = set()
    for item in parent["videos"]:
        video_id = item["video_id"]
        directory = root / video_id
        metadata_path = directory / "run_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if video_id not in videos or metadata["video"] != videos[video_id]:
            raise ValueError("video manifest mismatch")
        for path, digest in metadata["input_sha256"].items():
            if sha256_file(Path(path)) != digest:
                raise ValueError(f"parent input hash mismatch: {path}")
        source_rows = {}
        for name in ("depth_frames.jsonl", "semantic_frames.jsonl", "tracks.jsonl"):
            source_path = next(Path(x) for x in metadata["input_sha256"] if Path(x).name == name)
            if name != "tracks.jsonl":
                parent_path = source_path.parent.parent / "run_metadata.json"
                expected = parent[
                    "depth_metadata_sha256"
                    if name == "depth_frames.jsonl"
                    else "semantic_metadata_sha256"
                ]
                if sha256_file(parent_path) != expected:
                    raise ValueError("parent model-run metadata hash mismatch")
            rows = read_rows(source_path)
            indexed = {x["frame_index"]: x for x in rows}
            if len(indexed) != len(rows):
                raise ValueError("duplicate parent source frames")
            source_rows[name] = indexed
        spatial = directory / "spatial_frames.jsonl"
        objects_path = directory / "object_relations.jsonl"
        saved_objects = {}
        for obj in read_rows(objects_path):
            identity = (obj["key"]["frame_index"], obj["detection_index"])
            if identity in saved_objects or obj["key"]["video_id"] != video_id:
                raise ValueError("duplicate or wrong-video saved object relation")
            saved_objects[identity] = obj
        count = 0
        previous = None
        objects = 0
        with spatial.open(encoding="utf-8") as handle:
            for line in handle:
                raw = json.loads(line)
                key = FrameKey(**raw["key"])
                video = videos[video_id]
                if key.video_id != video_id or (key.width, key.height) != (
                    video["width"],
                    video["height"],
                ):
                    raise ValueError("grid video or image dimensions mismatch")
                if previous is not None and (
                    key.frame_index <= previous.frame_index
                    or key.timestamp_ms <= previous.timestamp_ms
                ):
                    raise ValueError("duplicate or non-monotonic grid frame")
                previous = key
                for name, indexed in source_rows.items():
                    source = indexed.get(key.frame_index)
                    if (
                        source is None
                        or (name != "tracks.jsonl" and source["video_id"] != video_id)
                        or source["timestamp_ms"] != key.timestamp_ms
                    ):
                        raise ValueError(f"grid/source exact frame mismatch: {name}")
                    dimension_fields = (
                        ("frame_width", "frame_height")
                        if name == "tracks.jsonl"
                        else ("source_frame_width", "source_frame_height")
                    )
                    if tuple(source[x] for x in dimension_fields) != (key.width, key.height):
                        raise ValueError(f"grid/source dimensions mismatch: {name}")
                frame = parse_spatial_frame(raw, key)
                if (frame.rows, frame.cols) != (parent["config"]["rows"], parent["config"]["cols"]):
                    raise ValueError("parent/grid dimensions mismatch")
                for obj in raw["object_relations"]:
                    if saved_objects.get((key.frame_index, obj["detection_index"])) != {
                        "key": raw["key"],
                        **obj,
                    }:
                        raise ValueError("saved object relation/frame disagreement")
                for source in frame.sources.values():
                    if source is not None and source.status == "valid":
                        if not source.reference or not source.sha256:
                            raise ValueError("valid source missing file/hash provenance")
                        identity = (source.reference, source.sha256)
                        if identity not in checked_hashes:
                            if sha256_file(Path(source.reference)) != source.sha256:
                                raise ValueError("frame source file hash mismatch")
                            checked_hashes.add(identity)
                count += 1
                objects += len(frame.object_relations)
        if count != metadata["frames"] or count == 0 or objects != len(saved_objects):
            raise ValueError("parent frame count mismatch")
        result.append(
            {
                "video_id": video_id,
                "video": videos[video_id],
                "frames": count,
                "source_object_relations": objects,
                "spatial_file": str(spatial),
                "spatial_sha256": sha256_file(spatial),
                "metadata_sha256": sha256_file(metadata_path),
                "object_relations_sha256": sha256_file(objects_path),
                "parent_input_sha256": metadata["input_sha256"],
            }
        )
    if not result or len({x["video_id"] for x in result}) != len(result):
        raise ValueError("empty or duplicate parent videos")
    return result


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for name in ABLATIONS:
        subset = [x for x in records if x["ablation"] == name]
        candidates = [c for x in subset for c in x["candidates"]]
        costs = [x["total_cost"] for x in candidates if x["total_cost"] is not None]
        unknown = [x["evidence"]["unknown_exposure"] for x in candidates]
        reason_counts = Counter(reason for x in candidates for reason in x["eligibility_reasons"])
        switches = pairs = 0
        previous = None
        for row in subset:
            if previous and row["key"]["video_id"] == previous["key"]["video_id"]:
                if row["key"]["timestamp_ms"] - previous["key"]["timestamp_ms"] == 500:
                    pairs += 1
                    switches += (
                        row["best_supported_candidate"] != previous["best_supported_candidate"]
                    )
            previous = row
        result[name] = {
            "name": ABLATIONS[name],
            "frames": len(subset),
            "candidates": len(candidates),
            "eligible_candidates": len(costs),
            "eligible_fraction": len(costs) / len(candidates) if candidates else None,
            "frame_statuses": dict(Counter(x["status"] for x in subset)),
            "eligibility_reasons": dict(reason_counts),
            "mean_unknown_exposure": float(np.mean(unknown)) if unknown else None,
            "cost_mean_eligible": float(np.mean(costs)) if costs else None,
            "depth_contribution_mean": float(
                np.mean(
                    [
                        c["weighted_contributions"]["depth"]
                        for c in candidates
                        if c["weighted_contributions"]["depth"] is not None
                    ]
                )
            )
            if name in ("C", "D") and candidates
            else None,
            "adjacent_500ms_pairs": pairs,
            "best_supported_identity_changes": switches,
            "stability_scope": "sampled_image_identity_only_not_motion_or_route_stability",
        }
    paired = {}
    for row in records:
        key = (row["key"]["video_id"], row["key"]["frame_index"])
        paired.setdefault(key, {})[row["ablation"]] = row
    comparisons = []
    for key, pair in paired.items():
        b, c = pair["B"], pair["C"]
        differences = [
            {
                "candidate_id": cb["candidate"]["candidate_id"],
                "cost_B": cb["total_cost"],
                "cost_C": cc["total_cost"],
                "depth_contribution_C": cc["weighted_contributions"]["depth"],
                "bbox_coverage": cc["evidence"]["bbox_proxy_coverage"],
                "near_relative_depth_proxy": cc["evidence"]["near_relative_depth_proxy"],
                "weighted_component_deltas_C_minus_B": {
                    name: (cc["weighted_contributions"][name] or 0)
                    - (cb["weighted_contributions"][name] or 0)
                    for name in cc["weighted_contributions"]
                },
            }
            for cb, cc in zip(b["candidates"], c["candidates"], strict=True)
            if cb["total_cost"] is not None
            and cc["total_cost"] is not None
            and abs(cb["total_cost"] - cc["total_cost"]) > 1e-12
        ]
        if differences or b["ranking"] != c["ranking"] or b["status"] != c["status"]:
            comparisons.append(
                {
                    "video_id": key[0],
                    "frame_index": key[1],
                    "timestamp_ms": b["key"]["timestamp_ms"],
                    "ranking_B": b["ranking"],
                    "ranking_C": c["ranking"],
                    "status_B": b["status"],
                    "status_C": c["status"],
                    "candidate_differences": differences,
                }
            )
    return {
        "ablations": result,
        "depth_comparisons": comparisons,
        "depth_changed_cost_frames": sum(bool(x["candidate_differences"]) for x in comparisons),
        "depth_changed_ranking_frames": sum(x["ranking_B"] != x["ranking_C"] for x in comparisons),
        "depth_changed_status_frames": sum(x["status_B"] != x["status_C"] for x in comparisons),
    }


def run_corridor_batch(
    spatial_root: Path,
    output: Path,
    config_path: Path,
    manifest_path: Path,
    *,
    visualize: bool = True,
) -> dict[str, Any]:
    fresh_directory(output)
    started = time.perf_counter()
    cpu_started = time.process_time()
    metadata: dict[str, Any] = {
        "schema_version": "sightguard.corridor-batch.v1",
        "status": "running",
        "production_decision_changed": False,
        "production_audio_called": False,
        "spatial_root": str(spatial_root),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "platform": platform.platform(),
        },
        "error": None,
    }
    try:
        metadata.update(
            {
                "code_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                "source_sha256": {
                    str(path): sha256_file(path)
                    for path in [
                        Path(__file__),
                        *Path("src/sightguard/scene/corridor").glob("*.py"),
                        Path("src/sightguard/visualization/corridor.py"),
                        Path("src/sightguard/visualization/corridor_viewer.html"),
                    ]
                },
                "config_sha256": sha256_file(config_path),
                "manifest_sha256": sha256_file(manifest_path),
                "parent_metadata_sha256": sha256_file(spatial_root / "run_metadata.json"),
            }
        )
        config = load_config(config_path)
        metadata["config"] = asdict(config)
        preflight_started = time.perf_counter()
        videos = preflight(spatial_root, manifest_path)
        metadata["preflight_wall_seconds"] = time.perf_counter() - preflight_started
        metadata["videos"] = videos
        engine = CorridorEvaluator(config)
        records = []
        timings = []
        failures = []
        for video in videos:
            video_id = video["video_id"]
            directory = output / video_id
            directory.mkdir()
            if visualize:
                (directory / "previews").mkdir()
            files = {
                name: (directory / f"{name}.jsonl").open("w", encoding="utf-8")
                for name in ("corridor_frames", "candidate_corridors", "object_relations")
            }
            viewer_rows = []
            video_records = []
            capture = cv2.VideoCapture(video["video"]["filepath"]) if visualize else None
            try:
                with Path(video["spatial_file"]).open(encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, 1):
                        sample_start = time.perf_counter()
                        stage: dict[str, float] = {}
                        t = time.perf_counter()
                        frame = parse_spatial_frame(json.loads(line))
                        stage["load_validate"] = (time.perf_counter() - t) * 1000
                        t = time.perf_counter()
                        candidates = engine.generator.generate(frame)
                        stage["candidate_generation"] = (time.perf_counter() - t) * 1000
                        frame_records = {}
                        for ablation in ABLATIONS:
                            assessments = []
                            for candidate in candidates:
                                t = time.perf_counter()
                                evidence = engine.aggregate(frame, candidate, ablation)
                                stage["evidence_aggregation"] = (
                                    stage.get("evidence_aggregation", 0)
                                    + (time.perf_counter() - t) * 1000
                                )
                                t = time.perf_counter()
                                assessment = engine.evaluate_candidate(
                                    frame, candidate, ablation, evidence, include_relations=False
                                )
                                stage["candidate_evaluation"] = (
                                    stage.get("candidate_evaluation", 0)
                                    + (time.perf_counter() - t) * 1000
                                )
                                t = time.perf_counter()
                                relations = engine.object_relations(frame, candidate, ablation)
                                stage["object_relation"] = (
                                    stage.get("object_relation", 0)
                                    + (time.perf_counter() - t) * 1000
                                )
                                assessments.append(replace(assessment, object_relations=relations))
                            t = time.perf_counter()
                            result = engine.assemble(
                                frame,
                                ablation,
                                tuple(assessments),
                                {
                                    "spatial_file": video["spatial_file"],
                                    "spatial_sha256": video["spatial_sha256"],
                                    "spatial_line": line_number,
                                    "config_sha256": metadata["config_sha256"],
                                    "parent_metadata_sha256": metadata["parent_metadata_sha256"],
                                },
                            )
                            stage["ranking_and_provenance"] = (
                                stage.get("ranking_and_provenance", 0)
                                + (time.perf_counter() - t) * 1000
                            )
                            t = time.perf_counter()
                            row = result.to_dict()
                            files["corridor_frames"].write(
                                json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                            )
                            for c in row["candidates"]:
                                files["candidate_corridors"].write(
                                    json.dumps(
                                        {"ablation": ablation, **c},
                                        ensure_ascii=False,
                                        allow_nan=False,
                                    )
                                    + "\n"
                                )
                                for relation in c["object_relations"]:
                                    files["object_relations"].write(
                                        json.dumps(
                                            {"ablation": ablation, **relation},
                                            ensure_ascii=False,
                                            allow_nan=False,
                                        )
                                        + "\n"
                                    )
                            stage["serialization"] = (
                                stage.get("serialization", 0) + (time.perf_counter() - t) * 1000
                            )
                            frame_records[ablation] = row
                            records.append(row)
                            video_records.append(row)
                        t = time.perf_counter()
                        if capture is not None:
                            capture.set(cv2.CAP_PROP_POS_FRAMES, frame.key.frame_index)
                            ok, image = capture.read()
                            if not ok:
                                raise ValueError("source video frame decode failed")
                            save_preview(
                                image,
                                frame,
                                frame_records["D"],
                                directory / "previews" / f"{frame.key.frame_index:06d}.jpg",
                            )
                        stage["visualization"] = (time.perf_counter() - t) * 1000
                        stage["full_sample"] = (time.perf_counter() - sample_start) * 1000
                        stage["algorithm_total"] = sum(
                            stage[name]
                            for name in (
                                "candidate_generation",
                                "evidence_aggregation",
                                "candidate_evaluation",
                                "object_relation",
                                "ranking_and_provenance",
                            )
                        )
                        timings.append(stage)
                        viewer_rows.append(
                            {
                                "key": asdict(frame.key),
                                "ablations": frame_records,
                                "cells": [asdict(x) for x in frame.cells],
                            }
                        )
                        d = frame_records["D"]
                        failures.append(
                            {
                                "key": d["key"],
                                "status": d["status"],
                                "reasons": d["abstain_reasons"],
                                "candidate_reasons": {
                                    c["candidate"]["candidate_id"]: c["eligibility_reasons"]
                                    for c in d["candidates"]
                                },
                                "source_failures": list(frame.failures),
                                "review_status": "program_evidence_only_not_ground_truth",
                            }
                        )
                write_json(directory / "summary.json", summarize(video_records))
                write_json(
                    directory / "run_metadata.json",
                    {
                        **video,
                        "config_sha256": metadata["config_sha256"],
                        "frames_written": len(viewer_rows),
                    },
                )
                write_viewer(directory, viewer_rows, visualize)
            finally:
                for file in files.values():
                    file.close()
                if capture is not None:
                    capture.release()
        summary = summarize(records)
        write_json(output / "ablation_summary.json", summary)
        write_json(output / "failure_cases.json", failures)
        with (output / "timings.jsonl").open("w", encoding="utf-8") as handle:
            for index, stage in enumerate(timings):
                handle.write(json.dumps({"sample_ordinal": index, **stage}, allow_nan=False) + "\n")
        links = "".join(
            f'<li><a href="{v["video_id"]}/index.html">{v["video_id"]}</a></li>' for v in videos
        )
        (output / "index.html").write_text(
            '<!doctype html><meta charset="utf-8"><title>Corridor Intelligence v1</title>'
            "<h1>Image evidence research</h1><p>No navigation or safety claim. "
            f"Default D abstains on unobserved camera pose.</p><ul>{links}</ul>",
            encoding="utf-8",
        )
        metadata.update(
            {
                "status": "completed",
                "timings_ms": percentiles(timings),
                "timing_scope": (
                    "per source frame, all four ablations; "
                    "excludes preflight from per-frame timings"
                ),
                "memory_mib": memory_snapshot(),
                "wall_seconds": time.perf_counter() - started,
                "process_cpu_seconds": time.process_time() - cpu_started,
                "torch_loaded": "torch" in sys.modules,
                "samples": len(timings),
                "output_files_sha256": {
                    str(p.relative_to(output)): sha256_file(p)
                    for p in output.rglob("*")
                    if p.is_file()
                },
            }
        )
        metadata["wall_seconds"] = time.perf_counter() - started
        metadata["process_cpu_seconds"] = time.process_time() - cpu_started
        metadata["wall_scope"] = (
            "includes preflight, all frames, viewers and output hashes; "
            "excludes final metadata write"
        )
    except Exception as exc:
        metadata.update(
            status="failed",
            error={"type": type(exc).__name__, "message": str(exc)},
            wall_seconds=time.perf_counter() - started,
        )
        write_json(output / "run_metadata.json", metadata)
        raise
    write_json(output / "run_metadata.json", metadata)
    return metadata
