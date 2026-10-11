"""Opt-in integration against real local weights and the V05 UrbanEgo clip."""

import hashlib
import json
import os
from pathlib import Path

import pytest

from sightguard.path_relevance.pipeline import run_path_relevance_batch

pytestmark = pytest.mark.integration


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_real_segformer_and_urbanego_v05_pipeline(tmp_path: Path) -> None:
    if os.environ.get("SIGHTGUARD_RUN_PATH_RELEVANCE_INTEGRATION") != "1":
        pytest.skip("Set SIGHTGUARD_RUN_PATH_RELEVANCE_INTEGRATION=1 for real model/video testing")

    root = Path(__file__).resolve().parents[1]
    baseline_alerts = root / "outputs/real_world/v01/urbanego_run8_05/assessment/alerts.jsonl"
    original_alert_hash = _sha256(baseline_alerts)
    output_dir = tmp_path / "path_relevance_v05"
    result = run_path_relevance_batch(
        root / "datasets/real_world/manifests/dataset.json",
        root / "outputs/real_world/v01",
        output_dir,
        checkpoint_path=root / "models/path_relevance/segformer_mit_b0_ade20k_160k.pth",
        device="auto",
        video_ids=("urbanego_run8_05",),
    )

    assert result["status"] == "completed"
    assert result["summary"]["videos_completed"] == 1
    assert result["summary"]["sampled_scene_frames"] == 30
    assert result["runtime"]["initial_device"].startswith(("cuda", "cpu"))
    assert _sha256(baseline_alerts) == original_alert_hash
    run_metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert run_metadata["model"]["upstream_repository_license"] == "Apache-2.0"
    assert "not independently verified" in run_metadata["model"]["checkpoint_license_status"]
    scene_rows = [
        json.loads(line)
        for line in (output_dir / "urbanego_run8_05/scene_frames.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    object_rows = [
        json.loads(line)
        for line in (output_dir / "urbanego_run8_05/objects.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert scene_rows[0]["inference_status"] == "completed"
    assert scene_rows[0]["corridors"]
    assert any(row["path_relevance_score"] is not None for row in object_rows)
    assert all(row["video_id"] == "urbanego_run8_05" for row in object_rows)
