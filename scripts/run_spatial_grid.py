"""Fuse saved UrbanEgo sources without loading a model or modifying alerts."""

import argparse
from pathlib import Path

from sightguard.pipeline.spatial_grid import run_spatial_batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("datasets/real_world/manifests/dataset.json")
    )
    parser.add_argument("--tracks", type=Path, default=Path("outputs/real_world/v01"))
    parser.add_argument("--semantics", type=Path, default=Path("outputs/spatial_semantics/v01"))
    parser.add_argument("--depth", type=Path, default=Path("outputs/depth_integration/v03"))
    parser.add_argument("--config", type=Path, default=Path("configs/spatial_grid.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-previews", action="store_true")
    args = parser.parse_args()
    result = run_spatial_batch(
        args.manifest,
        args.tracks,
        args.semantics,
        args.depth,
        args.output,
        args.config,
        video_ids=tuple(args.video),
        limit=args.limit,
        visualize=not args.no_previews,
    )
    print(result["status"], result["ablations"])


if __name__ == "__main__":
    main()
