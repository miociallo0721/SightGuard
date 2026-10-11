"""Runner contracts without datasets, models, network or production callbacks."""

import json
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from test_motion_quality import corridor, estimates, frame

from sightguard.pipeline.motion_quality import (
    decoded_frame,
    join_corridor,
    load_config,
    run_motion_batch,
)
from sightguard.scene.motion.schema import MotionConfig
from sightguard.scene.motion.temporal import diagnose_change, summarize_window


def test_yaml_configuration_matches_versioned_defaults():
    assert asdict(load_config(Path("configs/motion_quality.yaml"))) == asdict(MotionConfig())


def test_decoder_pts_mismatch_and_unavailable_image_are_explicit():
    f = frame(1)
    current = decoded_frame(f.image, 100.0, f.tracks, "synthetic", "synthetic", {})
    assert current.reasons == ("decoded_video_pts_mismatch",)
    current = decoded_frame(None, float("nan"), f.tracks, "synthetic", "synthetic", {})
    assert "frame_decode_failed" in current.reasons
    assert current.provenance["decoded_pts_ms"] is None
    json.dumps(current.provenance, allow_nan=False)


def test_preexisting_output_never_written(tmp_path):
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    sentinel = occupied / "sentinel"
    sentinel.write_text("preserve")
    with pytest.raises(FileExistsError):
        run_motion_batch(
            Path("missing"),
            Path("missing"),
            Path("missing"),
            Path("missing"),
            occupied,
            tmp_path / "new",
        )
    assert sentinel.read_text() == "preserve" and not (tmp_path / "new").exists()


def test_same_output_roots_rejected_before_input_or_mutation(tmp_path):
    with pytest.raises(FileExistsError):
        run_motion_batch(
            Path("missing"),
            Path("missing"),
            Path("missing"),
            Path("missing"),
            tmp_path / "same",
            tmp_path / "same",
        )
    assert not (tmp_path / "same").exists()


def test_join_does_not_mutate_original_and_unknown_never_keeps_old_candidate():
    source = corridor()
    before = deepcopy(source)
    empty = summarize_window(frame(10).key, "synthetic", [])
    joined = join_corridor(source, None, empty, "synthetic", MotionConfig())
    assert source == before
    assert all(
        x["status"] == "UNKNOWN" and x["gated_research_best"] is None for x in joined["freshness"]
    )
    assert all(x["gated_research_status"] == "ABSTAIN" for x in joined["freshness"])
    assert joined["absolute_camera_pose_status"] == "UNKNOWN"
    assert not joined["production_audio_called"] and not joined["production_decision_changed"]


def test_window_rejects_forged_previous_identity_and_inconsistent_dt():
    data = estimates()
    for changed in (
        [replace(x, previous_key=replace(x.previous_key, video_id="different")) for x in data],
        [replace(x, dt_ms=75) for x in data],
    ):
        w = summarize_window(frame(10).key, "synthetic", changed)
        assert w.valid_count == 0 and w.quality.label == "UNKNOWN"


def test_unchanged_cost_has_no_dominant_term():
    change = diagnose_change(corridor(index=20), corridor(index=10))
    assert all(x["largest_absolute_cost_delta_term"] is None for x in change["candidate_deltas"])
