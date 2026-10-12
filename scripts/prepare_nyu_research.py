"""Freeze scene-disjoint NYUv2 samples and lazily fetch official HDF5 ranges."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import cv2
import h5py
import numpy as np

from sightguard.research.http_ranges import HttpRangeFile
from sightguard.research.resources import check_resources, lower_own_priority

URL = "https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
SIZE = 2972037809
ETAG = '"b125b2b1-5aa5b95787a80"'


def text_reference(file: h5py.File, reference: object) -> str:
    return "".join(chr(int(value)) for value in file[reference][:].flatten())


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--stop-utc", required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.count <= 100:
        raise ValueError("bounded scene sample count must be 1..100")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    lower_own_priority()
    cv2.setNumThreads(1)
    source = HttpRangeFile(
        URL,
        SIZE,
        ETAG,
        args.cache,
        stop_utc=datetime.fromisoformat(args.stop_utc),
        maximum_bytes=192 * 1024**2,
    )
    with h5py.File(source, "r") as file:
        if file["images"].shape != (1449, 3, 640, 480):
            raise ValueError("unexpected official NYU images contract")
        names = [text_reference(file, ref) for ref in file["names"][0]]
        mapping = {name: index + 1 for index, name in enumerate(names)}
        if "wall" not in mapping or "floor" not in mapping:
            raise ValueError("wall/floor label names absent")
        scenes = [text_reference(file, ref) for ref in file["scenes"][0]]
        first_by_scene: dict[str, int] = {}
        for index, scene in enumerate(scenes):
            first_by_scene.setdefault(scene, index)
        unique = sorted(first_by_scene)
        chosen = [unique[int(i)] for i in np.linspace(0, len(unique) - 1, args.count)]
        samples = [
            {
                "sample_id": f"nyu_{first_by_scene[scene]:04d}",
                "index": first_by_scene[scene],
                "scene": scene,
                "split": "calibration" if order % 4 == 0 else "evaluation",
            }
            for order, scene in enumerate(chosen)
        ]
        manifest = {
            "schema_version": "sightguard.nyu-public-evaluation.v1",
            "source": URL,
            "source_size": SIZE,
            "source_etag": ETAG,
            "integrity": "exact_ranges_etag_local_block_sha_not_full_publisher_checksum",
            "license": (
                "official_public_research_download_citation_requested_no_explicit_license_found"
            ),
            "citation": (
                "Silberman et al., Indoor Segmentation and Support Inference "
                "from RGBD Images, ECCV 2012"
            ),
            "redistribution": "not_redistributed",
            "selection": (
                "32_or_requested_uniform_sorted_unique_scene_names_first_frame_before_inference"
            ),
            "ground_truth_usage": "evaluation_only_not_algorithm_input",
            "labels": {"wall": mapping["wall"], "floor": mapping["floor"]},
            "samples": samples,
        }
        frozen_path = args.output / "frozen-selection.json"
        completed = {}
        if args.resume:
            if json.loads(frozen_path.read_text(encoding="utf-8")) != manifest:
                raise ValueError("resume cannot change frozen selection or source")
            for line in (args.output / "extraction.jsonl").read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                if record["status"] == "complete":
                    completed[record["sample_id"]] = record
        else:
            frozen_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print("selection frozen", len(samples), "labels", manifest["labels"], flush=True)
        with (args.output / "extraction.jsonl").open(
            "a" if args.resume else "x", encoding="utf-8"
        ) as log:
            for sample in samples:
                check_resources(stop_utc=args.stop_utc, output_dir=args.output)
                if sample["sample_id"] in completed:
                    record = completed[sample["sample_id"]]
                    if any(
                        digest(Path(record[field])) != record[field.replace("path", "sha256")]
                        for field in ["rgb_path", "gt_path"]
                    ):
                        raise ValueError("completed sample changed before resume")
                    continue
                index = sample["index"]
                record = dict(sample)
                try:
                    rgb = np.transpose(file["images"][index], (2, 1, 0))
                    labels = file["labels"][index].T
                    raw = file["rawDepths"][index].T
                    filled = file["depths"][index].T
                    rgb_path = args.output / f"{sample['sample_id']}.png"
                    gt_path = args.output / f"{sample['sample_id']}.npz"
                    if not cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                        raise OSError("cannot write RGB image")
                    np.savez_compressed(
                        gt_path, labels=labels, raw_depth_m=raw, filled_depth_m=filled
                    )
                    record.update(
                        status="complete",
                        rgb_path=str(rgb_path.resolve()),
                        gt_path=str(gt_path.resolve()),
                        rgb_sha256=digest(rgb_path),
                        gt_sha256=digest(gt_path),
                        raw_valid_fraction=float(np.mean(np.isfinite(raw) & (raw > 0))),
                        range_bytes=source.downloaded_bytes,
                    )
                except Exception as exc:
                    record.update(status="failed", error=repr(exc))
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                    raise
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(record["sample_id"], record["scene"], source.downloaded_bytes, flush=True)
    (args.output / "range-source.json").write_text(
        json.dumps(
            {"range_manifest": str(source.manifest), "downloaded_bytes": source.downloaded_bytes}
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
