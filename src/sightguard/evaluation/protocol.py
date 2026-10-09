"""Frozen evaluation evidence: test runs refuse changed labels, splits, code, or settings."""

import hashlib
import json
import platform
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

from sightguard import __version__
from sightguard.evaluation.config import BenchmarkConfig
from sightguard.evaluation.dataset import load_dataset, sha256_file
from sightguard.risk.config import RiskConfig


def code_fingerprint() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    return {
        path.relative_to(root).as_posix(): sha256_file(path) for path in sorted(root.rglob("*.py"))
    }


def json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def protocol_evidence(
    dataset: Path, config: BenchmarkConfig, risk_config: RiskConfig
) -> dict[str, Any]:
    videos = load_dataset(dataset)
    lock = Path(__file__).resolve().parents[3] / "uv.lock"
    project = lock.parent / "pyproject.toml"
    return {
        "schema_version": "3.0",
        "methodology_version": "opportunity-matching-3.0",
        "manifest_sha256": sha256_file(dataset),
        "inputs": {
            video.entry.video_id: {
                "split": video.entry.split,
                "source_group_id": video.entry.source_group_id,
                "video_sha256": video.entry.video_sha256,
                "tracks_sha256": video.entry.tracks_sha256,
                "annotations_sha256": video.entry.annotations_sha256,
                "perception_metadata_sha256": sha256_file(
                    video.entry.tracks_path.parent / "run_metadata.json"
                )
                if video.entry.data_kind == "real"
                else None,
            }
            for video in videos
        },
        "benchmark_config": config.to_dict(),
        "risk_config": risk_config.to_dict(),
        "source_sha256": code_fingerprint(),
        "uv_lock_sha256": sha256_file(lock) if lock.is_file() else None,
        "pyproject_sha256": sha256_file(project) if project.is_file() else None,
        "python_version": platform.python_version(),
        "sightguard_version": __version__,
        "dependency_versions": {
            name: version(name)
            for name in ("numpy", "matplotlib", "PyYAML", "opencv-python", "torch", "ultralytics")
        },
    }


def freeze_protocol(
    dataset: Path, config: BenchmarkConfig, risk_config: RiskConfig, output: Path
) -> dict[str, Any]:
    evidence = protocol_evidence(dataset, config, risk_config)
    evidence["frozen_at_utc"] = datetime.now(UTC).isoformat()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(evidence, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return evidence


def check_protocol(
    path: Path, dataset: Path, config: BenchmarkConfig, risk_config: RiskConfig
) -> None:
    frozen = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(frozen, dict):
        raise ValueError("Invalid frozen protocol")
    expected = protocol_evidence(dataset, config, risk_config)
    for key, value in expected.items():
        if json_hash(frozen.get(key)) != json_hash(value):
            raise ValueError(f"Frozen test protocol mismatch: {key}; do not tune on test outcomes")
