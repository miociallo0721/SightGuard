"""Explicitly download and verify the pinned 99 MB official Small checkpoint."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    from huggingface_hub import hf_hub_download

    from sightguard.perception.depth.artifacts import sha256_file
    from sightguard.perception.depth.schema import DepthModelConfig

    config = DepthModelConfig()
    artifacts = (
        ("config.json", config.config_sha256, 950),
        ("preprocessor_config.json", config.processor_sha256, 775),
        ("model.safetensors", config.checkpoint_sha256, 99173660),
        ("README.md", "319566441daaac0c5ed83e75a8fd19bae0863f3275e23f147c06fe0906edf6fb", 4400),
    )
    records = []
    for name, expected, size in artifacts:
        assert size <= 300_000_000
        path = config.checkpoint_dir / name
        if not path.exists():
            path = Path(
                hf_hub_download(
                    config.model_id,
                    name,
                    revision=config.model_revision,
                    local_dir=config.checkpoint_dir,
                )
            )
        digest = sha256_file(path)
        if digest != expected or path.stat().st_size != size:
            raise ValueError(f"Unexpected model artifact; preserve and inspect it: {path}")
        records.append(
            {
                "path": str(path),
                "sha256": digest,
                "bytes": size,
                "url": f"https://huggingface.co/{config.model_id}/resolve/"
                f"{config.model_revision}/{name}",
            }
        )
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
