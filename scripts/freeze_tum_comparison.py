"""Freeze uniform subsamples of existing TUM manifests before new model inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--per-sequence", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.per_sequence < 1:
        raise ValueError("positive sample count required")
    grouped: dict[str, list[dict]] = defaultdict(list)
    hashes = {}
    for path in args.manifest:
        if path.name in hashes:
            raise ValueError("source manifest names must be unique")
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        for sample in json.loads(path.read_text(encoding="utf-8"))["samples"]:
            grouped[sample["sequence"]].append(sample)
    selected = []
    for sequence, samples in grouped.items():
        if len(samples) < args.per_sequence:
            raise ValueError("too few source samples for uniform selection")
        samples.sort(key=lambda sample: sample["rgb_index"])
        if sequence.startswith("freiburg1_"):
            intrinsic = [517.3, 516.5, 318.6, 255.3]
        elif sequence.startswith("freiburg3_"):
            intrinsic = [535.4, 539.2, 320.1, 247.6]
        else:
            raise ValueError("camera intrinsic contract not verified for this sequence")
        indices = np.linspace(0, len(samples) - 1, args.per_sequence, dtype=int)
        selected.extend(
            {**samples[int(index)], "dataset": "TUM", "intrinsic": intrinsic} for index in indices
        )
    if len({sample["sample_id"] for sample in selected}) != len(selected):
        raise ValueError("duplicate source sample ID")
    result = {
        "schema_version": "sightguard.dataset-comparison.v1",
        "selection": (
            f"{args.per_sequence}_uniform_indices_per_existing_frozen_sequence_"
            "before_Metric3D_inference"
        ),
        "source_manifests": hashes,
        "samples": selected,
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({"samples": len(selected), "sequences": len(grouped)}))


if __name__ == "__main__":
    main()
