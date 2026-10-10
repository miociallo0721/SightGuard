"""CPU-safe adaptation checks and an explicit opt-in CUDA host lifecycle test."""

import os
import subprocess
import sys
import textwrap
from dataclasses import replace
from pathlib import Path
from threading import Event

import numpy as np
import pytest

from sightguard.hybrid.config import load_hybrid_config
from sightguard.hybrid.provider import ProcessDecisionProvider
from sightguard.hybrid.schema import parse_output


def test_cpu_precision_and_unavailable_cuda() -> None:
    # Match the product's process isolation: retaining torch in pytest would compete
    # with the real model integration hosts for Windows commit memory.
    code = textwrap.dedent("""
        from types import SimpleNamespace
        import pytest
        import torch
        from sightguard.hybrid.runtime_support import configure_vision
        model = torch.nn.Module()
        model.encoder = torch.nn.Linear(2, 2)
        model.scorer = torch.nn.Linear(2, 1)
        agent = SimpleNamespace(model=model)
        profile = configure_vision(agent, torch, 'cpu', 'fp32')
        assert profile['parameter_device_counts'] == {'cpu': 9}
        assert agent.model.scorer(agent.model.encoder(torch.ones(1, 2))).isfinite().all()
        with pytest.raises(ValueError, match='CPU requires FP32'):
            configure_vision(agent, torch, 'cpu', 'fp16')
        unavailable = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with pytest.raises(RuntimeError, match='CUDA runtime unavailable'):
            configure_vision(agent, unavailable, 'cuda', 'fp16')
    """)
    completed = subprocess.run(
        [sys.executable, "-"], input=code, capture_output=True, text=True, timeout=30
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.integration
def test_real_cuda_vision_host_reload(tmp_path: Path) -> None:
    if os.environ.get("SIGHTGUARD_TEST_HYBRID_GPU") != "1":
        pytest.skip("Set SIGHTGUARD_TEST_HYBRID_GPU=1 for explicitly prepared CUDA Vision")
    root = Path(__file__).resolve().parents[1]
    config = load_hybrid_config(root / "configs/hybrid.yaml", root)
    model = replace(
        config.models["vision"],
        python=str(root / "models/hybrid/env-vision-gpu/Scripts/python.exe"),
        device="cuda",
        precision="fp16",
    )
    provider = ProcessDecisionProvider(
        "vision", model, timeout_s=8, load_timeout_s=90, log_path=tmp_path / "cuda-host.log"
    )
    for _ in range(2):
        try:
            metadata = provider.load(Event())
            assert metadata["device"] == "cuda"
            assert metadata["precision"] == "fp16_encoder_fp32_readout"
            assert set(metadata["parameter_dtype_counts"]) == {"torch.float16", "torch.float32"}
            assert all(key.startswith("cuda") for key in metadata["parameter_device_counts"])
            reply = provider.infer(
                {"objects": [], "perception_status": "ok"},
                np.zeros((480, 640, 3), dtype=np.uint8),
                Event(),
            )
            level, _, error = parse_output(reply["raw_output"], "vision")
            assert level in {None, "LOW", "MEDIUM", "HIGH"}
            assert error is None or "UNKNOWN" in error
            assert reply["peak_vram_mb"] > 0
        finally:
            provider.unload()
        assert provider.process is None
