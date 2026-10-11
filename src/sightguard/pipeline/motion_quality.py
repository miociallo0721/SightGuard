"""Read-only CPU video motion replay and saved corridor temporal diagnostics."""

from __future__ import annotations

import json
import math
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from sightguard.data import FrameRecord
from sightguard.perception.depth.artifacts import sha256_file, write_json
from sightguard.pipeline.corridor import preflight
from sightguard.pipeline.spatial_grid import memory_snapshot, percentiles
from sightguard.scene.motion.estimator import FrameMotionEstimator
from sightguard.scene.motion.schema import MotionConfig, MotionFrame
from sightguard.scene.motion.temporal import diagnose_change, evaluate_freshness, summarize_window
from sightguard.scene.spatial_grid.artifacts import fresh_directory, read_rows
from sightguard.scene.spatial_grid.schema import FrameKey
from sightguard.visualization.motion_quality import save_sample, write_viewer


def load_config(path: Path) -> MotionConfig:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    value["simulated_consumer_delays_ms"] = tuple(value["simulated_consumer_delays_ms"])
    return MotionConfig(**value)


def validate_parents(spatial: Path, corridor: Path, manifest: Path) -> list[dict[str, Any]]:
    sources = preflight(spatial, manifest)
    parent = json.loads((corridor / "run_metadata.json").read_text(encoding="utf-8"))
    if parent["status"] != "completed" or parent["production_decision_changed"]:
        raise ValueError("invalid saved corridor parent")
    if parent["manifest_sha256"] != sha256_file(manifest):
        raise ValueError("corridor manifest mismatch")
    if parent["parent_metadata_sha256"] != sha256_file(spatial / "run_metadata.json"):
        raise ValueError("corridor spatial parent mismatch")
    for reference, digest in parent["source_sha256"].items():
        if sha256_file(Path(reference)) != digest:
            raise ValueError(f"corridor producer hash mismatch: {reference}")
    for reference, digest in parent["output_files_sha256"].items():
        if sha256_file(corridor / reference) != digest:
            raise ValueError(f"corridor output hash mismatch: {reference}")
    return sources


def decoded_frame(
    image: np.ndarray | None,
    decoded_pts: float,
    record: FrameRecord,
    video_id: str,
    stream_id: str,
    provenance: dict[str, Any],
) -> MotionFrame:
    key = FrameKey(
        video_id, record.frame_index, record.timestamp_ms, record.frame_width, record.frame_height
    )
    reasons = []
    if image is None:
        reasons.append("frame_decode_failed")
    if not math.isfinite(decoded_pts) or not math.isclose(
        decoded_pts, record.timestamp_ms, abs_tol=0.01
    ):
        reasons.append("decoded_video_pts_mismatch")
    return MotionFrame(
        key,
        stream_id,
        image,
        record,
        record.timestamp_source,
        tuple(reasons),
        {
            **provenance,
            "decoded_pts_ms": decoded_pts if math.isfinite(decoded_pts) else None,
        },
    )


def join_corridor(
    source: dict[str, Any],
    previous: dict[str, Any] | None,
    window: Any,
    stream: str,
    cfg: MotionConfig,
) -> dict[str, Any]:
    freshness = []
    for delay in cfg.simulated_consumer_delays_ms:
        assessment = evaluate_freshness(
            source, window, source["key"]["timestamp_ms"] + delay, stream, cfg
        )
        freshness.append(
            {
                **assessment.to_dict(),
                "simulated_delay_ms": delay,
                "gated_research_status": source["status"]
                if assessment.reuse_allowed_for_research
                else "ABSTAIN",
                "gated_research_best": source["best_supported_candidate"]
                if assessment.reuse_allowed_for_research
                else None,
            }
        )
    return {
        "schema_version": "sightguard.temporal-corridor.v1",
        "key": source["key"],
        "stream_id": stream,
        "ablation": source["ablation"],
        "source_corridor": source,
        "motion_window": window.to_dict(),
        "change": diagnose_change(source, previous, cfg),
        "freshness": freshness,
        "absolute_camera_pose_status": "UNKNOWN",
        "production_decision_changed": False,
        "production_audio_called": False,
        "temporal_shadow_status": "NOT_RUN",
    }


def statistics(values: list[float]) -> dict[str, Any]:
    return {
        "samples": len(values),
        "p50": float(np.percentile(values, 50)) if values else None,
        "p95": float(np.percentile(values, 95)) if values else None,
    }


def summarize_motion(rows: list[dict[str, Any]], controls: list[dict[str, Any]]) -> dict[str, Any]:
    def describe(data):
        return {
            "quality": dict(Counter(x["quality"]["label"] for x in data)),
            "valid_estimates": sum(x["estimation_status"] == "VALID" for x in data),
            "unknown_reasons": dict(
                Counter(
                    reason
                    for x in data
                    if x["quality"]["label"] == "UNKNOWN"
                    for reason in x["quality"]["reasons"]
                )
            ),
            **{
                name: statistics([x[name] for x in data if x[name] is not None])
                for name in (
                    "feature_count",
                    "match_count",
                    "inlier_ratio",
                    "background_feature_coverage",
                    "motion_diagonals_per_second",
                    "residual_inlier_p95_pixels",
                    "background_available_fraction",
                )
            },
        }

    pairs = list(zip(rows, controls, strict=True))
    rate_differences = [
        abs(a["motion_diagonals_per_second"] - b["motion_diagonals_per_second"])
        for a, b in pairs
        if a["estimation_status"] == b["estimation_status"] == "VALID"
    ]
    return {
        "frames": len(rows),
        "masked": describe(rows),
        "unmasked": describe(controls),
        "quality_label_changed": sum(
            a["quality"]["label"] != b["quality"]["label"] for a, b in pairs
        ),
        "masked_unknown_unmasked_valid": sum(
            a["estimation_status"] == "UNKNOWN" and b["estimation_status"] == "VALID"
            for a, b in pairs
        ),
        "masked_valid_unmasked_unknown": sum(
            a["estimation_status"] == "VALID" and b["estimation_status"] == "UNKNOWN"
            for a, b in pairs
        ),
        "absolute_rate_difference_both_valid": statistics(rate_differences),
        "background_warning": (
            "remaining pixels are unverified background; masking is not ground truth"
        ),
    }


def summarize_temporal(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = {}
    for name in "BCD":
        data = [x for x in rows if x["ablation"] == name]
        transitions = [x for x in data if x["change"]["types"] != ["INITIAL_OBSERVATION"]]
        association = {}
        for quality in sorted({x["motion_window"]["quality"]["label"] for x in transitions}):
            subset = [x for x in transitions if x["motion_window"]["quality"]["label"] == quality]
            association[quality] = {
                "transitions": len(subset),
                **{
                    field: sum(x["change"][field] for x in subset)
                    for field in (
                        "lowest_cost_changed",
                        "status_changed",
                        "best_supported_identity_changed",
                    )
                },
                "abstain": sum(x["source_corridor"]["status"] == "ABSTAIN" for x in subset),
            }
        groups[name] = {
            "samples": len(data),
            "transitions": len(transitions),
            "source_status": dict(Counter(x["source_corridor"]["status"] for x in data)),
            "change_types": dict(
                Counter(kind for x in transitions for kind in x["change"]["types"])
            ),
            "motion_quality_association": association,
            "dominant_candidate_cost_delta": dict(
                Counter(
                    c["largest_absolute_cost_delta_term"]
                    for x in transitions
                    for c in x["change"]["candidate_deltas"]
                    if c["largest_absolute_cost_delta_term"] is not None
                )
            ),
            "freshness_by_simulated_delay_ms": {
                str(delay): dict(
                    Counter(
                        f["status"]
                        for x in data
                        for f in x["freshness"]
                        if f["simulated_delay_ms"] == delay
                    )
                )
                for delay in sorted({f["simulated_delay_ms"] for x in data for f in x["freshness"]})
            },
            "research_gated_status_by_delay_ms": {
                str(delay): dict(
                    Counter(
                        f["gated_research_status"]
                        for x in data
                        for f in x["freshness"]
                        if f["simulated_delay_ms"] == delay
                    )
                )
                for delay in sorted({f["simulated_delay_ms"] for x in data for f in x["freshness"]})
            },
        }
    indexed = {(x["key"]["video_id"], x["key"]["frame_index"], x["ablation"]): x for x in rows}
    differences = []
    for identity, b in indexed.items():
        if identity[2] != "B":
            continue
        c = indexed[(*identity[:2], "C")]
        if (
            b["source_corridor"]["ranking"] != c["source_corridor"]["ranking"]
            or b["source_corridor"]["status"] != c["source_corridor"]["status"]
        ):
            differences.append(
                {
                    "key": b["key"],
                    "quality": b["motion_window"]["quality"]["label"],
                    "ranking_changed": b["source_corridor"]["ranking"]
                    != c["source_corridor"]["ranking"],
                    "status_changed": b["source_corridor"]["status"]
                    != c["source_corridor"]["status"],
                }
            )
    return {
        "groups_original_corridor": groups,
        "b_c_differences": differences,
        "b_c_rank_difference_quality": dict(
            Counter(x["quality"] for x in differences if x["ranking_changed"])
        ),
        "temporal_experiments": {
            "T0_BASELINE": {"status": "COMPLETED", "costs_and_statuses_unchanged": True},
            "T1_MOTION_DIAGNOSTIC": {"status": "COMPLETED", "costs_and_statuses_unchanged": True},
            "T2_FRESHNESS_GATE": {
                "status": "COMPLETED",
                "scope": "simulated consumption; research fields only",
            },
            "T3_STABILIZATION": {
                "status": "NOT_RUN",
                "reason": (
                    "no validated cross-frame corridor correspondence "
                    "or independent temporal passability truth"
                ),
                "switch_reduction": None,
                "hold_count": None,
                "interruption_count": None,
                "extra_delay_ms": None,
            },
        },
        "association_warning": (
            "observational associations from one recording; not causal or navigation correctness"
        ),
    }


def run_motion_batch(
    spatial: Path,
    corridor: Path,
    manifest: Path,
    config_path: Path,
    motion_output: Path,
    temporal_output: Path,
    *,
    visualize: bool = True,
) -> dict[str, Any]:
    if (
        motion_output.resolve() == temporal_output.resolve()
        or motion_output.exists()
        or temporal_output.exists()
    ):
        raise FileExistsError("both distinct output roots must be new")
    cfg = load_config(config_path)
    sources = validate_parents(spatial, corridor, manifest)
    fresh_directory(motion_output)
    fresh_directory(temporal_output)
    start = time.perf_counter()
    metadata: dict[str, Any] = {
        "schema_version": "sightguard.motion-batch.v1",
        "status": "running",
        "config": asdict(cfg),
        "config_sha256": sha256_file(config_path),
        "manifest_sha256": sha256_file(manifest),
        "uv_lock_sha256": sha256_file(Path("uv.lock")),
        "spatial_metadata_sha256": sha256_file(spatial / "run_metadata.json"),
        "corridor_metadata_sha256": sha256_file(corridor / "run_metadata.json"),
        "source_sha256": {
            str(path): sha256_file(path)
            for path in [
                Path(__file__),
                *Path("src/sightguard/scene/motion").glob("*.py"),
                *Path("src/sightguard/visualization").glob("motion_quality*"),
                Path("scripts/run_motion_quality.py"),
            ]
            if path.is_file()
        },
        "code_commit_before_run": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "runtime": {
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "timestamp_source": "verified decoded MP4 PTS; sensor capture time unknown",
        "production_audio_called": False,
        "production_decision_changed": False,
        "absolute_camera_pose_status": "UNKNOWN",
        "videos": [],
        "error": None,
        "cpu_threads": cv2.getNumThreads(),
        "torch_loaded": "torch" in sys.modules,
    }
    all_timing, all_motion, all_control, all_temporal, failures = [], [], [], [], []
    try:
        for source in sources:
            video_id = source["video_id"]
            video = source["video"]
            video_path = Path(video["filepath"])
            video_hash = sha256_file(video_path)
            if video_hash != video["processed_sha256"]:
                raise ValueError("video hash mismatch")
            tracks_path = next(
                Path(x) for x in source["parent_input_sha256"] if Path(x).name == "tracks.jsonl"
            )
            records = [FrameRecord.from_json(json.dumps(x)) for x in read_rows(tracks_path)]
            if not records or any(
                r.frame_index != i
                or r.timestamp_source != "video_pts"
                or (i > 0 and r.timestamp_ms <= records[i - 1].timestamp_ms)
                for i, r in enumerate(records)
            ):
                raise ValueError("invalid complete tracks sequence")
            spatial_rows = {
                x["key"]["frame_index"]: x for x in read_rows(Path(source["spatial_file"]))
            }
            corridor_path = corridor / video_id / "corridor_frames.jsonl"
            saved = read_rows(corridor_path)
            if len(saved) != len(spatial_rows) * 4 or len(
                {(x["key"]["frame_index"], x["ablation"]) for x in saved}
            ) != len(saved):
                raise ValueError("incomplete or duplicate corridor rows")
            grouped = defaultdict(list)
            for row in saved:
                if (
                    row["key"] != spatial_rows[row["key"]["frame_index"]]["key"]
                    or row["ablation"] not in "ABCD"
                ):
                    raise ValueError("corridor exact frame key mismatch")
                grouped[row["key"]["frame_index"]].append(row)
            directory, temporal_dir = motion_output / video_id, temporal_output / video_id
            fresh_directory(directory)
            fresh_directory(temporal_dir)
            stream = f"{video_id}:{video_hash}"
            provenance = {
                "video_reference": str(video_path),
                "video_sha256": video_hash,
                "tracks_reference": str(tracks_path),
                "tracks_sha256": sha256_file(tracks_path),
                "config_sha256": metadata["config_sha256"],
            }
            cap = cv2.VideoCapture(str(video_path))
            previous, last_control, last_primary = None, None, None
            history, video_rows, controls, temporal_rows, windows, timing_rows, viewer_rows = (
                [],
                [],
                [],
                [],
                [],
                [],
                [],
            )
            previous_corridor = {}
            engine = FrameMotionEstimator(cfg)
            video_start = time.perf_counter()
            try:
                with (
                    (directory / "frame_motion.jsonl").open("w", encoding="utf-8") as primary_file,
                    (directory / "unmasked_frame_motion.jsonl").open(
                        "w", encoding="utf-8"
                    ) as control_file,
                ):
                    for record in records:
                        timing = {}
                        t = time.perf_counter()
                        success, image = cap.read()
                        pts = cap.get(cv2.CAP_PROP_POS_MSEC)
                        timing["decode"] = (time.perf_counter() - t) * 1000
                        current = decoded_frame(
                            image if success else None,
                            pts,
                            record,
                            video_id,
                            stream,
                            {
                                **provenance,
                                "record_reference": (
                                    f"{directory / 'frame_motion.jsonl'}#frame={record.frame_index}"
                                ),
                            },
                        )
                        t = time.perf_counter()
                        primary = engine.estimate(
                            previous, current, previous_estimate=last_primary, timings=timing
                        )
                        timing["pure_primary_motion"] = (time.perf_counter() - t) * 1000
                        timing["decode_plus_primary_motion"] = (
                            timing["decode"] + timing["pure_primary_motion"]
                        )
                        t = time.perf_counter()
                        control = engine.estimate(
                            previous,
                            replace(
                                current,
                                provenance={
                                    **current.provenance,
                                    "record_reference": (
                                        f"{directory / 'unmasked_frame_motion.jsonl'}"
                                        f"#frame={record.frame_index}"
                                    ),
                                },
                            ),
                            masked=False,
                            previous_estimate=last_control,
                        )
                        timing["pure_unmasked_motion"] = (time.perf_counter() - t) * 1000
                        primary_row, control_row = primary.to_dict(), control.to_dict()
                        history.append(primary)
                        video_rows.append(primary_row)
                        controls.append(control_row)
                        t = time.perf_counter()
                        primary_file.write(json.dumps(primary_row, allow_nan=False) + "\n")
                        control_file.write(json.dumps(control_row, allow_nan=False) + "\n")
                        timing["serialization"] = (time.perf_counter() - t) * 1000
                        if primary.quality.label in ("UNKNOWN", "LIMITED"):
                            failures.append(
                                {
                                    "key": asdict(current.key),
                                    "quality": primary.quality.label,
                                    "reasons": primary.quality.reasons,
                                    "source_reference": current.provenance["record_reference"],
                                }
                            )
                        if record.frame_index in spatial_rows:
                            if asdict(current.key) != spatial_rows[record.frame_index]["key"]:
                                raise ValueError("tracks/spatial exact frame mismatch")
                            t = time.perf_counter()
                            window = summarize_window(current.key, stream, history, cfg)
                            windows.append(window.to_dict())
                            timing["window_aggregation"] = (time.perf_counter() - t) * 1000
                            t = time.perf_counter()
                            sample_rows = []
                            for row in grouped[record.frame_index]:
                                if row["ablation"] == "A":
                                    continue
                                joined = join_corridor(
                                    row, previous_corridor.get(row["ablation"]), window, stream, cfg
                                )
                                joined["source_reference"] = str(corridor_path)
                                joined["source_sha256"] = (
                                    sha256_file(corridor_path)
                                    if not temporal_rows
                                    else temporal_rows[0]["source_sha256"]
                                )
                                sample_rows.append(joined)
                                previous_corridor[row["ablation"]] = row
                            temporal_rows.extend(sample_rows)
                            timing["corridor_join_and_diagnosis"] = (time.perf_counter() - t) * 1000
                            if visualize and image is not None:
                                t = time.perf_counter()
                                sample = save_sample(
                                    directory,
                                    current,
                                    primary_row,
                                    control_row,
                                    spatial_rows[record.frame_index],
                                    sample_rows,
                                )
                                viewer_rows.append(sample)
                                timing["visualization"] = (time.perf_counter() - t) * 1000
                        timing_rows.append(timing)
                        previous, last_primary, last_control = current, primary, control
            finally:
                cap.release()
            for path, rows in (
                (directory / "motion_windows.jsonl", windows),
                (directory / "timings.jsonl", timing_rows),
                (temporal_dir / "temporal_corridor.jsonl", temporal_rows),
            ):
                path.write_text(
                    "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows),
                    encoding="utf-8",
                )
            if visualize:
                write_viewer(directory, viewer_rows)
            motion_summary = summarize_motion(video_rows, controls)
            temporal_summary = summarize_temporal(temporal_rows)
            write_json(directory / "summary.json", motion_summary)
            write_json(temporal_dir / "summary.json", temporal_summary)
            item = {
                "video_id": video_id,
                "frames": len(records),
                "corridor_samples": len(windows),
                "temporal_rows": len(temporal_rows),
                "wall_seconds": time.perf_counter() - video_start,
                "video_sha256": video_hash,
                "source_sha256": {
                    str(tracks_path): sha256_file(tracks_path),
                    str(corridor_path): sha256_file(corridor_path),
                    source["spatial_file"]: source["spatial_sha256"],
                },
                "stream_id": stream,
                "motion_summary": motion_summary,
                "timings_ms": percentiles(timing_rows),
            }
            write_json(directory / "run_metadata.json", item)
            write_json(temporal_dir / "run_metadata.json", item)
            metadata["videos"].append(item)
            all_motion.extend(video_rows)
            all_control.extend(controls)
            all_temporal.extend(temporal_rows)
            all_timing.extend(timing_rows)
            print(f"{video_id}: {len(records)} RGB frames, {len(windows)} past windows", flush=True)
        write_json(motion_output / "summary.json", summarize_motion(all_motion, all_control))
        write_json(temporal_output / "summary.json", summarize_temporal(all_temporal))
        write_json(motion_output / "failure_cases.json", failures)
        write_json(
            temporal_output / "failure_cases.json",
            [
                {
                    "key": x["key"],
                    "ablation": x["ablation"],
                    "source_status": x["source_corridor"]["status"],
                    "window_quality": x["motion_window"]["quality"],
                    "change": x["change"],
                }
                for x in all_temporal
                if x["source_corridor"]["status"] == "ABSTAIN"
                or x["motion_window"]["quality"]["label"] in ("UNKNOWN", "LIMITED")
            ],
        )
        metadata["status"] = "completed"
    except Exception as exc:
        metadata["status"], metadata["error"] = "failed", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        metadata["frames"] = len(all_motion)
        metadata["samples"] = len(all_temporal)
        metadata["timings_ms"] = percentiles(all_timing)
        metadata["memory_mib"] = memory_snapshot()
        metadata["throughput_fps"] = {
            name: len(all_timing) * 1000 / sum(x[name] for x in all_timing) if all_timing else None
            for name in ("pure_primary_motion", "decode_plus_primary_motion")
        }
        for root in (motion_output, temporal_output):
            metadata.setdefault("output_sha256_by_root", {})[str(root)] = {
                str(path.relative_to(root)): sha256_file(path)
                for path in root.rglob("*")
                if path.is_file() and path != root / "run_metadata.json"
            }
        metadata["wall_seconds"] = time.perf_counter() - start
        metadata["full_research_throughput_fps"] = len(all_motion) / metadata["wall_seconds"]
        metadata["wall_scope"] = (
            "all video processing, both motion variants, diagnostics, previews, artifact writes "
            "and hashes; excludes final metadata writes and preflight"
        )
        for root in (motion_output, temporal_output):
            write_json(root / "run_metadata.json", metadata)
    return metadata
