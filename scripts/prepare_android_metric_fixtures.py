"""Public TUM RGB-D diagnostic subset only, never physical CameraX/tape acceptance."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from sightguard.research.settings import research_cache_root

ROOT = Path(__file__).resolve().parents[1]
MODELS = research_cache_root()


def main() -> None:
    samples = json.loads((MODELS / "sample-manifest.json").read_text())["samples"]
    wall_sequence = [
        item for item in samples if item["sequence"] == "freiburg3_nostructure_notexture_far"
    ]
    destination = ROOT / "android/app/src/androidTest/assets/metric_research"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "TUM_NOTICE.txt").write_text(
        "TUM RGB-D Benchmark, CC BY 4.0. https://cvg.cit.tum.de/data/datasets/rgbd-dataset\n"
        "Attribution: J. Sturm, N. Engelhard, F. Endres, W. Burgard, D. Cremers.\n"
        "A Benchmark for the Evaluation of RGB-D SLAM Systems. IROS 2012.\n"
        "Diagnostic samples and converted depth arrays; no measured phone scene.\n",
        encoding="utf-8",
    )
    records = []
    for index, position in enumerate([0, 25, 50, 75, 99]):
        sample = wall_sequence[position]
        shutil.copyfile(sample["rgb_path"], destination / f"scene_{index}.png")
        truth = cv2.imread(sample["depth_path"], cv2.IMREAD_UNCHANGED).astype(np.float32) / 5000
        truth.astype("<f4").tofile(destination / f"truth_{index}.bin")
        records.append(
            {
                **sample,
                "image": f"scene_{index}.png",
                "truth": f"truth_{index}.bin",
                "scope": "public_RGBD_diagnostic_Level_3_not_measured_phone_scene",
            }
        )
    (destination / "scenes.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print("public Android RGBD diagnostics", len(records))


if __name__ == "__main__":
    main()
