"""Explicit opt-in real RGB and independently serialized motion artifacts."""

import json
import os
from pathlib import Path

import cv2
import pytest

from sightguard.data import FrameRecord
from sightguard.perception.depth.artifacts import sha256_file
from sightguard.pipeline.motion_quality import decoded_frame, load_config
from sightguard.scene.motion.estimator import FrameMotionEstimator
from sightguard.scene.spatial_grid.artifacts import read_rows

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("SIGHTGUARD_MOTION_INTEGRATION") != "1",
        reason="opt-in local UrbanEgo videos required",
    ),
]


@pytest.mark.parametrize("number", range(1, 9))
def test_real_video_estimates_match_saved_cpu_run(number):
    directory = Path("outputs/motion_quality/v01")
    video_id = f"urbanego_run8_{number:02d}"
    saved = read_rows(directory / video_id / "frame_motion.jsonl")
    meta = json.loads((directory / "run_metadata.json").read_text(encoding="utf-8"))
    assert meta["status"] == "completed" and not meta["torch_loaded"]
    video = next(
        x
        for x in json.loads(
            Path("datasets/real_world/manifests/dataset.json").read_text(encoding="utf-8")
        )["videos"]
        if x["video_id"] == video_id
    )
    assert sha256_file(Path(video["filepath"])) == video["processed_sha256"]
    references = saved[0]["provenance"]
    records = [
        FrameRecord.from_json(json.dumps(x))
        for x in read_rows(Path(references["tracks_reference"]))
    ]
    engine = FrameMotionEstimator(load_config(Path("configs/motion_quality.yaml")))
    cap = cv2.VideoCapture(video["filepath"])
    previous, previous_estimate = None, None
    try:
        for record in records[:12]:
            ok, image = cap.read()
            assert ok
            current = decoded_frame(
                image, cap.get(cv2.CAP_PROP_POS_MSEC), record, video_id, saved[0]["stream_id"], {}
            )
            assert not current.reasons
            estimate = engine.estimate(previous, current, previous_estimate=previous_estimate)
            row = estimate.to_dict()
            expected = saved[record.frame_index]
            assert row["quality"]["label"] == expected["quality"]["label"]
            assert row["match_count"] == expected["match_count"]
            assert (
                row["inlier_ratio"] == pytest.approx(expected["inlier_ratio"])
                if expected["inlier_ratio"] is not None
                else row["inlier_ratio"] is None
            )
            assert (
                row["motion_diagonals_per_second"]
                == pytest.approx(expected["motion_diagonals_per_second"])
                if expected["motion_diagonals_per_second"] is not None
                else row["motion_diagonals_per_second"] is None
            )
            previous, previous_estimate = current, estimate
    finally:
        cap.release()


def test_real_artifact_baseline_validation_evidence():
    result = json.loads(
        Path("outputs/motion_quality_validation/v01/validation.json").read_text(encoding="utf-8")
    )
    assert result["status"] == "passed" and result["baseline_alerts"] == 257
    assert result["baseline_frames"] == result["frames"] == 2400
    assert result["frozen_files_checked"] == 2631 and not result["frozen_mismatches"]
    assert result["original_corridor_rows_equal"] and result["risk_frames_events_alerts_equal"]
