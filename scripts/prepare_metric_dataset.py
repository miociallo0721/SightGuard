"""Freeze uniform RGB selection and nearest registered TUM depth/pose associations."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np

from sightguard.research.resources import check_resources, lower_own_priority
from sightguard.research.settings import research_cache_root

ROOT = research_cache_root()


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def rows(path: Path) -> list[list[str]]:
    return [
        line.split()
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


def nearest(items: list[list[str]], stamp: float) -> tuple[list[str], float]:
    selected = min(items, key=lambda row: abs(float(row[0]) - stamp))
    return selected, abs(float(selected[0]) - stamp) * 1000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=["freiburg1_xyz", "freiburg1_rpy", "freiburg3_nostructure_notexture_far"],
    )
    parser.add_argument("--manifest", type=Path, default=ROOT / "sample-manifest.json")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()
    lower_own_priority()
    selected = (
        json.loads(args.manifest.read_text(encoding="utf-8"))["samples"]
        if args.append and args.manifest.exists()
        else []
    )
    for name in args.sequences:
        check_resources(output_dir=ROOT)
        archive = ROOT / f"rgbd_dataset_{name}.tgz"
        target = ROOT / f"rgbd_dataset_{name}"
        complete = target / ".sightguard-extraction-complete.json"
        if not complete.exists():
            available = all(
                (target / filename).exists()
                for filename in ("rgb.txt", "depth.txt", "groundtruth.txt")
            )
            if available:
                available = all(
                    (target / row[1]).is_file()
                    for filename in ("rgb.txt", "depth.txt")
                    for row in rows(target / filename)
                )
            if not available:
                with tarfile.open(archive) as handle:
                    # Python's data filter rejects path traversal, unsafe links and devices.
                    handle.extractall(ROOT, filter="data")
            for filename in ("rgb.txt", "depth.txt"):
                if not all((target / row[1]).is_file() for row in rows(target / filename)):
                    raise ValueError("incomplete_dataset_extraction")
            complete.write_text(
                json.dumps({"archive_sha256": sha256(archive), "all_indexed_images_present": True}),
                encoding="utf-8",
            )
        rgb, depth, poses = (
            rows(target / filename) for filename in ("rgb.txt", "depth.txt", "groundtruth.txt")
        )
        indices = np.unique(np.linspace(0, len(rgb) - 1, args.count).round().astype(int))
        first_stamp = float(rgb[0][0])
        for index in indices:
            stamp = float(rgb[index][0])
            matched_depth, depth_delta = nearest(depth, stamp)
            pose, pose_delta = nearest(poses, stamp)
            image_path = target / rgb[index][1]
            depth_path = target / matched_depth[1]
            selected.append(
                {
                    "sample_id": f"{name}_{index:05d}",
                    "sequence": name,
                    "rgb_index": int(index),
                    "timestamp_seconds": stamp,
                    "timestamp_ms": (stamp - first_stamp) * 1000,
                    "rgb_path": str(image_path),
                    "rgb_sha256": sha256(image_path),
                    "depth_path": str(depth_path),
                    "depth_sha256": sha256(depth_path),
                    "depth_delta_ms": depth_delta,
                    "depth_association_valid": depth_delta <= 20,
                    "pose_timestamp_seconds": float(pose[0]),
                    "pose_delta_ms": pose_delta,
                    "pose_association_valid": pose_delta <= 20,
                    "translation_world_m": list(map(float, pose[1:4])),
                    "quaternion_xyzw": list(map(float, pose[4:8])),
                    "ground_truth_quantity": "registered_optical_axis_z_m_depth_uint16_div_5000",
                    "selection": f"{args.count}_uniform_rgb_indices_frozen_before_model_inference",
                    "license": "CC-BY-4.0",
                }
            )
        print(name, "RGB", len(rgb), "selected", len(indices), flush=True)
    manifest = {
        "schema_version": "sightguard.tum-metric-samples.v1",
        "samples": selected,
        "semantic_truth_available": False,
        "measured_phone_wall_distance_available": False,
        "pose_intrinsics_note": "RGB-D registered; optical Z metrics need no intrinsics. "
        "fr1 distorted RGB geometry is not exact; no Euclidean ranging claim.",
    }
    path = args.manifest
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("manifest", sha256(path), "samples", len(selected), flush=True)


if __name__ == "__main__":
    main()
