"""Local-only actual auxiliary integration, explicitly enabled independently of mocks."""

import os
from pathlib import Path
from threading import Event

import numpy as np
import pytest

from sightguard.hybrid.config import load_hybrid_config
from sightguard.hybrid.provider import ProcessDecisionProvider
from sightguard.hybrid.schema import parse_output


@pytest.mark.integration
@pytest.mark.parametrize("source", ["laya", "vision"])
def test_real_auxiliary(source: str, tmp_path: Path) -> None:
    if os.environ.get("SIGHTGUARD_TEST_HYBRID") != "1":
        pytest.skip("Set SIGHTGUARD_TEST_HYBRID=1 for explicitly prepared local auxiliary models")
    root = Path(__file__).resolve().parents[1]
    config = load_hybrid_config(root / "configs/hybrid.yaml", root)
    provider = ProcessDecisionProvider(
        source,
        config.models[source],
        timeout_s=config.timeout_s,
        load_timeout_s=config.load_timeout_s,
        log_path=tmp_path / "host.log",
    )
    cancel = Event()
    try:
        metadata = provider.load(cancel)
        assert len(metadata["weight_sha256"]) == 64
        reply = provider.infer(
            {"objects": [], "perception_status": "ok"},
            np.zeros((480, 640, 3), dtype=np.uint8) if source == "vision" else None,
            cancel,
        )
        raw = reply["raw_output"]
        assert "answers" in raw and "risk" in raw["answers"]
        level, score, error = parse_output(raw, source)
        assert level in {None, "LOW", "MEDIUM", "HIGH"}
        assert score is None or isinstance(score, float)
        assert error is None or "UNKNOWN" in error
    finally:
        provider.unload()
    assert provider.process is None
