"""Independent Depth Integration CLI; run with the chosen CPU/CUDA interpreter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline relative depth; never emits alerts/audio")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/depth.yaml")
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "datasets/real_world/manifests/dataset.json"
    )
    parser.add_argument("--baseline-dir", type=Path, default=ROOT / "outputs/real_world/v01")
    parser.add_argument("--scene-dir", type=Path, default=ROOT / "outputs/path_relevance/v04")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda:0"))
    parser.add_argument("--video-id", action="append")
    parser.add_argument(
        "--workspace-dependencies",
        action="store_true",
        help="Import chosen Torch/Transformers first, then append workspace .venv libraries",
    )
    args = parser.parse_args()
    if args.workspace_dependencies:
        # Preserve the isolated GPU runtime before adding OpenCV/YAML/pytest dependencies.
        import torch  # noqa: F401
        import transformers  # noqa: F401

        sys.path.append(str(ROOT / ".venv/Lib/site-packages"))
    import yaml

    from sightguard.perception.depth.schema import DepthModelConfig
    from sightguard.pipeline.depth import run_depth_batch

    values = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if values["schema_version"] != "sightguard.depth-config.v1":
        raise ValueError("unsupported depth config version")
    model = values["model"]
    if args.device:
        model["device"] = args.device
    config = DepthModelConfig(**model)
    result = run_depth_batch(
        args.manifest,
        args.baseline_dir,
        args.scene_dir,
        args.output_dir,
        config,
        interval_seconds=values["sampling"]["interval_seconds"],
        warmup_frames=values["sampling"]["warmup_frames"],
        video_ids=tuple(args.video_id) if args.video_id else None,
    )
    print(json.dumps({"status": result["status"], "wall_seconds": result["wall_seconds"]}))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
