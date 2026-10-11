"""Opt-in real saved UrbanEgo grid tests, with no model downloads or inference."""

import json
import os
from pathlib import Path

import pytest

from sightguard.pipeline.corridor import preflight, run_corridor_batch
from sightguard.scene.corridor.artifacts import parse_spatial_frame
from sightguard.scene.corridor.engine import CorridorEvaluator

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[1]


def enabled():
    if os.environ.get("SIGHTGUARD_RUN_CORRIDOR_SAVED_INTEGRATION") != "1":
        pytest.skip("Set SIGHTGUARD_RUN_CORRIDOR_SAVED_INTEGRATION=1 for saved UrbanEgo tests")


def test_all_real_saved_frames_and_hashes():
    enabled()
    videos = preflight(
        ROOT / "outputs/spatial_grid/v03", ROOT / "datasets/real_world/manifests/dataset.json"
    )
    assert len(videos) == 8 and sum(x["frames"] for x in videos) == 240
    assert sum(x["source_object_relations"] for x in videos) == 1123
    engine = CorridorEvaluator()
    for video in videos:
        with Path(video["spatial_file"]).open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                frame = parse_spatial_frame(json.loads(line))
                assert frame.key.frame_index == index * 10 and frame.key.timestamp_ms == index * 500
                assert engine.assess(frame, "D").status == "ABSTAIN"


def test_real_batch_replay_is_deterministic(tmp_path):
    enabled()
    result = run_corridor_batch(
        ROOT / "outputs/spatial_grid/v03",
        tmp_path / "replay",
        ROOT / "configs/corridor.yaml",
        ROOT / "datasets/real_world/manifests/dataset.json",
        visualize=False,
    )
    assert result["samples"] == 240 and result["status"] == "completed"
    assert not result["torch_loaded"] and not result["production_audio_called"]
    for directory in (ROOT / "outputs/corridor_intelligence/v01").glob("urbanego_run8_*"):
        replay = tmp_path / "replay" / directory.name
        for name in (
            "corridor_frames.jsonl",
            "candidate_corridors.jsonl",
            "object_relations.jsonl",
            "summary.json",
        ):
            expected = directory / name
            actual = replay / name
            # Provenance paths intentionally reflect the same source and config.
            expected_rows = expected.read_text(encoding="utf-8").replace(str(ROOT) + "\\", "")
            actual_rows = actual.read_text(encoding="utf-8").replace(str(ROOT) + "\\", "")
            if name.endswith("jsonl"):
                left = [json.loads(x) for x in expected_rows.splitlines()]
                right = [json.loads(x) for x in actual_rows.splitlines()]
                # Absolute vs relative spatial provenance is intentionally allowed.
                for a, b in zip(left, right, strict=True):
                    if "provenance" in a:
                        a["provenance"].pop("spatial_file")
                        b["provenance"].pop("spatial_file")
                assert left == right
            else:
                assert json.loads(expected_rows) == json.loads(actual_rows)
