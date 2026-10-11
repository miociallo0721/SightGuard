"""Local-only model rerun, separate from CPU evidence fusion."""

import argparse
from pathlib import Path

from sightguard.pipeline.spatial_semantics import prepare_semantic_batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--video", action="append", default=[])
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    result = prepare_semantic_batch(
        Path("datasets/real_world/manifests/dataset.json"),
        Path("outputs/depth_integration/v03"),
        args.output,
        Path("configs/path_relevance.yaml"),
        device=args.device,
        video_ids=tuple(args.video),
        limit=args.limit,
    )
    print(result["status"], result["videos"])


if __name__ == "__main__":
    main()
