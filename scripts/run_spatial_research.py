"""Replay cached real model outputs into isolated wall-approach and image-candidate research."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from sightguard.research.local_corridor import LocalCorridorCandidateEvaluator, LocalCorridorConfig
from sightguard.research.resources import check_resources, lower_own_priority
from sightguard.research.settings import load_research_settings, research_cache_root
from sightguard.research.wall_approach import ApproachConfig, WallApproachEstimator
from sightguard.research.wall_metric import WallMetricConfig, WallMetricEstimator
from sightguard.scene.motion.estimator import FrameMotionEstimator
from sightguard.scene.motion.schema import MotionFrame
from sightguard.scene.spatial_grid import (
    EvidenceSource,
    FrameKey,
    SpatialGridFusion,
    SpatialGridInput,
)

ROOT = Path(__file__).resolve().parents[1]
MODELS = research_cache_root()
DELIVERY = ROOT / "android/deliveries/metric-corridor-research"


def rotation_matrix(quaternion: list[float]) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion) / np.linalg.norm(quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def pose_label(previous: dict | None, current: dict) -> dict:
    """Independent mocap labels. NEVER supplied to the approach estimator."""
    if previous is None or previous["sequence"] != current["sequence"]:
        return {"label": "UNAVAILABLE"}
    if not previous["pose_association_valid"] or not current["pose_association_valid"]:
        return {"label": "UNAVAILABLE", "reason": "pose_association_over_20ms"}
    before = rotation_matrix(previous["quaternion_xyzw"])
    after = rotation_matrix(current["quaternion_xyzw"])
    translation = before.T @ (
        np.array(current["translation_world_m"]) - np.array(previous["translation_world_m"])
    )
    angle = math.acos(float(np.clip((np.trace(before.T @ after) - 1) / 2, -1, 1)))
    magnitude = float(np.linalg.norm(translation))
    x, y, z = map(float, translation)
    lateral = math.hypot(x, y)
    label = "MIXED_OR_AMBIGUOUS"
    if magnitude < 0.005 and angle < 0.01:
        label = "STATIC"
    elif magnitude < 0.01 and angle >= 0.02:
        label = "ROTATION_DOMINANT"
    elif z > 0.015 and z / max(magnitude, 1e-9) >= 0.8 and angle <= 0.025:
        label = "FORWARD_CAMERA_TRANSLATION"
    elif z < -0.015 and -z / max(magnitude, 1e-9) >= 0.8 and angle <= 0.025:
        label = "RECEDING_CAMERA_TRANSLATION"
    elif abs(z) < 0.01 and lateral >= 0.015:
        label = "LATERAL_CAMERA_TRANSLATION"
    return {
        "label": label,
        "camera_translation_previous_coordinates_m": [x, y, z],
        "rotation_angle_rad": angle,
        "source": "TUM_optical_center_mocap",
        "not_body_motion_ground_truth": True,
    }


def camera_calibration(sequence: str) -> tuple[np.ndarray, np.ndarray]:
    if sequence.startswith("freiburg3"):
        return np.array([[535.4, 0, 320.1], [0, 539.2, 247.6], [0, 0, 1]]), np.zeros(5)
    return (
        np.array([[517.3, 0, 318.6], [0, 516.5, 255.3], [0, 0, 1]]),
        np.array([0.2624, -0.9531, -0.0054, 0.0026, 1.1633]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MODELS / "sample-manifest.json")
    parser.add_argument("--evaluation", type=Path, default=DELIVERY / "evaluation")
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/metric_corridor_research.yaml"
    )
    args = parser.parse_args()
    settings = load_research_settings(args.config)
    lower_own_priority()
    cv2.setNumThreads(1)
    samples = json.loads(args.manifest.read_text(encoding="utf-8"))["samples"]
    semantic_records = {
        row["sample_id"]: row
        for row in (
            json.loads(line)
            for line in (args.evaluation / "semantic_results.jsonl").read_text().splitlines()
        )
    }
    metric_records = {
        row["sample_id"]: row
        for row in (
            json.loads(line)
            for line in (args.evaluation / "metric_hypersim_518.jsonl").read_text().splitlines()
        )
    }
    wall = WallMetricEstimator(WallMetricConfig(**settings["wall"]))
    approach = WallApproachEstimator(ApproachConfig(**settings["approach"]))
    corridor = LocalCorridorCandidateEvaluator(LocalCorridorConfig(**settings["local_corridor"]))
    grid_engine = SpatialGridFusion()
    original_motion = FrameMotionEstimator()
    previous_key = previous_image = previous_mask = previous_sample = previous_motion = None
    with (
        (args.evaluation / "wall_approach_results.jsonl").open("w", encoding="utf-8") as a_out,
        (args.evaluation / "corridor_candidates.jsonl").open("w", encoding="utf-8") as c_out,
    ):
        for sample in samples:
            check_resources(output_dir=args.evaluation)
            start = time.perf_counter()
            image = cv2.imread(sample["rgb_path"])
            h, w = image.shape[:2]
            key = FrameKey(sample["sequence"], sample["rgb_index"], sample["timestamp_ms"], w, h)
            if previous_key and previous_key.video_id != key.video_id:
                previous_key = previous_image = previous_mask = previous_sample = (
                    previous_motion
                ) = None
                approach.reset()
                corridor.reset()
            with (
                np.load(args.evaluation / "semantics" / f"{sample['sample_id']}.npz") as sem,
                (
                    np.load(args.evaluation / "relative_280" / f"{sample['sample_id']}.npz")
                ) as relative,
                np.load(args.evaluation / "hypersim_518" / f"{sample['sample_id']}.npz") as metric,
            ):
                classes, confidence = sem["classes"], sem["confidence"]
                mask, _, coverage, _ = wall.mask(key, classes, confidence, sem["foreground"])
                present = coverage >= wall.config.minimum_roi_coverage
                motion_frame = MotionFrame(
                    key, key.video_id, image, None, timestamp_source="dataset_capture_timestamp"
                )
                # Do not invent video PTS or adjacent frames to bypass the frozen observer.
                motion = original_motion.estimate(previous_motion, motion_frame, masked=False)
                source = EvidenceSource(
                    key,
                    "valid",
                    "model_observed",
                    sample["rgb_path"],
                    sample["rgb_sha256"],
                    "real_onnx_model",
                )
                detections = tuple(semantic_records[sample["sample_id"]]["detections"])
                grid = grid_engine.fuse(
                    SpatialGridInput(
                        key,
                        source,
                        source,
                        source,
                        classes,
                        confidence,
                        sem["components"],
                        relative["depth"],
                        detections,
                    )
                )
                age = float(sem["processing_ms"]) + float(relative["latency_ms"])
                age += float(metric["latency_ms"])
                intrinsic, distortion = camera_calibration(key.video_id)
                metric_record = metric_records[sample["sample_id"]]
                result = approach.evaluate(
                    previous_key,
                    key,
                    previous_image,
                    image,
                    previous_mask,
                    mask,
                    source_age_ms=age,
                    wall_present=present,
                    intrinsic=intrinsic,
                    distortion=distortion,
                    fixed_lens_verified=True,
                    depth_support=metric_record["wall"]["status"],
                    motion_quality=motion.quality.label,
                )
                candidate = corridor.evaluate(
                    grid,
                    classes,
                    confidence,
                    semantic_key=key,
                    source_age_ms=age,
                    motion_quality=motion.quality.label,
                    metric_z_m=metric["depth"],
                    metric_key=key,
                )
                context = {
                    "sample_id": sample["sample_id"],
                    "rgb_sha256": sample["rgb_sha256"],
                    "context": "offline_model_replay_Level_2_not_CameraX",
                    "age_basis": "simulated_serial_age_from_measured_model_processing",
                    "original_motion": {
                        "quality": asdict(motion.quality),
                        "estimation_status": motion.estimation_status,
                    },
                    "research_latency_ms": (time.perf_counter() - start) * 1000,
                    "generated_utc": datetime.now(UTC).isoformat(),
                }
                a_out.write(
                    json.dumps(
                        {
                            **context,
                            **result,
                            "ground_truth_pair": pose_label(previous_sample, sample),
                        }
                    )
                    + "\n"
                )
                c_out.write(json.dumps({**context, **candidate}) + "\n")
                a_out.flush()
                c_out.flush()
                previous_key, previous_image, previous_mask = key, image, mask.copy()
                previous_sample, previous_motion = sample, motion_frame
                print(
                    sample["sample_id"],
                    result["state"],
                    result["abstain_reason"],
                    candidate["status"],
                    flush=True,
                )


if __name__ == "__main__":
    main()
