"""Run adjacent RGB motion and readonly corridor diagnostics without models."""

import argparse
from pathlib import Path

import cv2

from sightguard.pipeline.motion_quality import run_motion_batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spatial", type=Path, default=Path("outputs/spatial_grid/v03"))
    parser.add_argument("--corridor", type=Path, default=Path("outputs/corridor_intelligence/v01"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("datasets/real_world/manifests/dataset.json")
    )
    parser.add_argument("--config", type=Path, default=Path("configs/motion_quality.yaml"))
    parser.add_argument("--motion-output", type=Path, required=True)
    parser.add_argument("--temporal-output", type=Path, required=True)
    parser.add_argument("--no-previews", action="store_true")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    result = run_motion_batch(
        args.spatial,
        args.corridor,
        args.manifest,
        args.config,
        args.motion_output,
        args.temporal_output,
        visualize=not args.no_previews,
    )
    print(
        {
            name: result[name]
            for name in (
                "status",
                "frames",
                "samples",
                "wall_seconds",
                "throughput_fps",
                "full_research_throughput_fps",
                "memory_mib",
                "torch_loaded",
            )
        }
    )


if __name__ == "__main__":
    main()
