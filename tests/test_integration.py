"""Opt-in local model test, never downloads weights in the test suite."""

import os
from pathlib import Path

import pytest

from sightguard.config import PerceptionConfig
from sightguard.data import replay_records
from sightguard.pipeline.video import analyze_video


@pytest.mark.integration
def test_real_yolo_cpu(video: Path, tmp_path: Path) -> None:
    model = os.environ.get("SIGHTGUARD_TEST_MODEL")
    if model is None:
        pytest.skip(
            "Set SIGHTGUARD_TEST_MODEL to an existing local YOLO26n weight for real inference"
        )
    assert Path(model).is_file(), "The explicitly requested local model must exist"
    metadata = analyze_video(video, tmp_path / "real", PerceptionConfig(model_path=model))
    assert metadata["status"] == "completed"
    assert len(list(replay_records(tmp_path / "real/tracks.jsonl"))) == 5
    assert metadata["model"]["sha256"]


@pytest.mark.integration
def test_real_tracking_ids(tmp_path: Path) -> None:
    model = os.environ.get("SIGHTGUARD_TEST_MODEL")
    if model is None:
        pytest.skip("Set SIGHTGUARD_TEST_MODEL for the real tracking fixture")
    assert Path(model).is_file()
    import cv2
    from ultralytics.utils import ASSETS

    image = cv2.imread(str(ASSETS / "bus.jpg"))
    assert image is not None
    image = cv2.resize(image, (480, 640))
    video = tmp_path / "tracking.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (480, 640))
    assert writer.isOpened()
    try:
        for _ in range(4):
            writer.write(image)
    finally:
        writer.release()
    output = tmp_path / "tracked"
    analyze_video(video, output, PerceptionConfig(model_path=model))
    records = list(replay_records(output / "tracks.jsonl"))
    assert len(records) == 4
    id_sets = [
        {d.track_id for d in record.detections if d.track_id is not None} for record in records
    ]
    assert all(id_sets), "Real targets must receive tracking IDs"
    assert set.intersection(*id_sets), "Repeated static frames must retain at least one ID"
