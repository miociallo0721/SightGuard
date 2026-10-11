"""Evaluate existing Spatial Grid v03 without model inference or alerts."""

import argparse
from pathlib import Path

from sightguard.pipeline.corridor import run_corridor_batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spatial-root", type=Path, default=Path("outputs/spatial_grid/v03"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("datasets/real_world/manifests/dataset.json")
    )
    parser.add_argument("--config", type=Path, default=Path("configs/corridor.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-previews", action="store_true")
    args = parser.parse_args()
    result = run_corridor_batch(
        args.spatial_root, args.output, args.config, args.manifest, visualize=not args.no_previews
    )
    print(
        {
            name: result[name]
            for name in (
                "status",
                "samples",
                "wall_seconds",
                "timings_ms",
                "memory_mib",
                "torch_loaded",
            )
        }
    )


if __name__ == "__main__":
    main()
