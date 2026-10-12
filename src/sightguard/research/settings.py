"""Versioned experiment configuration, kept separate from production settings."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def research_cache_root() -> Path:
    """External cache; override explicitly when reproducing on a different machine."""
    return Path(
        os.getenv("SIGHTGUARD_METRIC_CACHE", "D:/program/SightGuard/models/metric-corridor")
    )


def load_research_settings(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema_version") != (
        "sightguard.metric-corridor-research.v1"
    ):
        raise ValueError("unsupported research configuration")
    if document.get("shadow_only") is not True or any(
        document.get(name) is not False
        for name in ("automatic_metric_speech", "automatic_approach_speech")
    ):
        raise ValueError("research configuration cannot enable automatic speech")
    return document
