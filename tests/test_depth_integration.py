"""Opt-in actual checkpoint, CPU/CUDA and UrbanEgo tests; never download models."""

import json
import os
import weakref
from pathlib import Path

import cv2
import numpy as np
import pytest

from sightguard.perception.depth.artifacts import sha256_file
from sightguard.perception.depth.model import DepthEstimator
from sightguard.perception.depth.schema import CHECKPOINT_SHA256, DepthModelConfig
from sightguard.pipeline.depth import run_depth_batch

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[1]


def enabled():
    if os.environ.get("SIGHTGUARD_RUN_DEPTH_INTEGRATION") != "1":
        pytest.skip("Set SIGHTGUARD_RUN_DEPTH_INTEGRATION=1 for real checkpoint/video tests")


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_real_checkpoint_backends_alignment_and_direction(device):
    enabled()
    estimator = DepthEstimator(
        DepthModelConfig(checkpoint_dir=ROOT / "models/depth_anything_v2_small", device=device)
    )
    estimator.initialize()
    model_reference = weakref.ref(estimator.model)
    model_bytes = sum(p.numel() * p.element_size() for p in estimator.model.parameters())
    assert estimator.device_name == device
    assert estimator.metadata["parameter_count"] == 24785089
    assert not estimator.metadata["loading_info"]["missing_keys"]
    capture = cv2.VideoCapture(str(ROOT / "datasets/real_world/processed/urbanego_run8_05.mp4"))
    ok, frame = capture.read()
    capture.release()
    assert ok
    try:
        prediction = estimator.infer(
            frame,
            video_id="urbanego_run8_05",
            frame_index=0,
            timestamp_ms=0,
            source_width=1280,
            source_height=720,
        )
        assert prediction.result.status == "valid"
        assert prediction.depth_map.shape == (720, 1280)
        assert prediction.model_output.shape == (280, 504)
        assert prediction.depth_map.dtype == np.float32
        assert np.isfinite(prediction.depth_map).all()
        # Human-visible near wall vs distant parked car, chosen on original RGB,
        # never on heatmap color. This checks direction, not depth accuracy.
        near = prediction.depth_map[200:500, 1100:1200]
        far = prediction.depth_map[50:100, 520:660]
        assert float(np.median(near)) > float(np.median(far))
        smaller = cv2.resize(frame, (640, 360))
        other = estimator.infer(
            smaller,
            video_id="urbanego_run8_05",
            frame_index=0,
            timestamp_ms=0,
            source_width=640,
            source_height=360,
        )
        assert other.result.status == "valid"
        assert other.depth_map.shape == (360, 640)
        assert (
            sha256_file(estimator.config.checkpoint_dir / "model.safetensors") == CHECKPOINT_SHA256
        )
    finally:
        before_unload = estimator.resource_stats()
        estimator.close()
    assert estimator.model is None
    assert model_reference() is None
    if device.startswith("cuda"):
        # cuBLAS retains process-owned workspaces independently of the model.
        # Assert the entire checkpoint allocation is released, including after
        # predictions escape to the caller as independent CPU arrays.
        freed = before_unload["allocated_mib"] - estimator.resource_stats()["allocated_mib"]
        assert freed >= model_bytes / 2**20


def test_real_urbanego_end_to_end_preserves_baseline(tmp_path):
    enabled()
    alerts = ROOT / "outputs/real_world/v01/urbanego_run8_05/assessment/alerts.jsonl"
    original = sha256_file(alerts)
    output = tmp_path / "depth"
    result = run_depth_batch(
        ROOT / "datasets/real_world/manifests/dataset.json",
        ROOT / "outputs/real_world/v01",
        ROOT / "outputs/path_relevance/v04",
        output,
        DepthModelConfig(checkpoint_dir=ROOT / "models/depth_anything_v2_small", device="cuda:0"),
        video_ids=("urbanego_run8_05",),
    )
    assert result["status"] == "completed"
    assert result["videos"][0]["valid_frames"] == 30
    assert sha256_file(alerts) == original
    rows = [
        json.loads(line)
        for line in (output / "urbanego_run8_05/depth_frames.jsonl").read_text().splitlines()
    ]
    assert [row["frame_index"] for row in rows] == list(range(0, 300, 10))
    assert [row["timestamp_ms"] for row in rows] == [i * 500 for i in range(30)]
    for row in rows:
        path = output / "urbanego_run8_05" / row["depth_data_reference"]
        assert sha256_file(path) == row["depth_data_sha256"]
        with np.load(path, allow_pickle=False) as data:
            assert data["depth"].shape == (720, 1280)
            assert data["model_output"].shape == (280, 504)
