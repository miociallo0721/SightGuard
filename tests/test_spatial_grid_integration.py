"""Explicit saved-real-input replay and separate actual SegFormer inference."""

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from sightguard.perception.depth.artifacts import sha256_file
from sightguard.pipeline.spatial_grid import run_spatial_batch
from sightguard.scene.spatial_grid import SpatialGridFusion
from sightguard.scene.spatial_grid.artifacts import load_inputs, read_rows, row_key

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[1]


def enabled(name="SIGHTGUARD_RUN_SPATIAL_SAVED_INTEGRATION"):
    if os.environ.get(name) != "1":
        pytest.skip(f"Set {name}=1; requires local real inputs, never downloads")


@pytest.mark.parametrize("video", ["03", "04", "05", "07", "08"])
def test_real_saved_three_sources_and_exact_depth_statistics(video):
    enabled()
    video_id = f"urbanego_run8_{video}"
    depth_dir = ROOT / "outputs/depth_integration/v03" / video_id
    semantic_dir = ROOT / "outputs/spatial_semantics/v01" / video_id
    tracks = ROOT / "outputs/real_world/v01" / video_id / "perception/tracks.jsonl"
    depth_row = read_rows(depth_dir / "depth_frames.jsonl")[0]
    semantic = read_rows(semantic_dir / "semantic_frames.jsonl")[0]
    path = depth_dir / depth_row["depth_data_reference"]
    original_hash = sha256_file(path)
    data = load_inputs(
        row_key(depth_row),
        read_rows(tracks)[0],
        tracks,
        semantic,
        semantic_dir,
        depth_row,
        depth_dir,
    )
    frame = SpatialGridFusion().fuse(data)
    assert frame.source_status["depth"] == "valid"
    assert len(frame.cells) == 225
    assert frame.key.width == 1280 and frame.key.height == 720
    for cell in frame.cells:
        x1, y1, x2, y2 = cell.bounds_xyxy
        region = data.depth_map[y1:y2, x1:x2]
        valid = region[np.isfinite(region) & (region >= 0)]
        assert cell.depth.valid_count == valid.size
        if valid.size:
            assert cell.depth.median == pytest.approx(float(np.median(valid)), abs=1e-6)
    assert sha256_file(path) == original_hash
    assert not frame.production_decision_changed


def test_real_saved_single_frame_pipeline_and_baseline_unchanged(tmp_path):
    enabled()
    original = ROOT / "outputs/real_world/v01/urbanego_run8_05/assessment/alerts.jsonl"
    path_relevance = ROOT / "outputs/path_relevance/v04/urbanego_run8_05/scene_frames.jsonl"
    hashes = [sha256_file(x) for x in (original, path_relevance)]
    output = tmp_path / "fusion"
    result = run_spatial_batch(
        ROOT / "datasets/real_world/manifests/dataset.json",
        ROOT / "outputs/real_world/v01",
        ROOT / "outputs/spatial_semantics/v01",
        ROOT / "outputs/depth_integration/v03",
        output,
        ROOT / "configs/spatial_grid.yaml",
        video_ids=("urbanego_run8_05",),
        limit=1,
    )
    assert result["status"] == "completed"
    assert result["ablations"]["D"]["cells"] == 225
    assert result["ablations"]["B"]["states"] == {"UNKNOWN": 225}
    assert (output / "index.html").exists()
    assert (output / "urbanego_run8_05/frames/000000.js").is_file()
    assert hashes == [sha256_file(x) for x in (original, path_relevance)]
    with pytest.raises(FileExistsError):
        run_spatial_batch(
            ROOT / "datasets/real_world/manifests/dataset.json",
            ROOT / "outputs/real_world/v01",
            ROOT / "outputs/spatial_semantics/v01",
            ROOT / "outputs/depth_integration/v03",
            output,
            ROOT / "configs/spatial_grid.yaml",
        )


def test_actual_segformer_matches_native_saved_prediction():
    enabled("SIGHTGUARD_RUN_SPATIAL_MODEL_INTEGRATION")
    from sightguard.path_relevance.model import SegformerB0Perceptor

    manifest = json.loads(
        (ROOT / "datasets/real_world/manifests/dataset.json").read_text(encoding="utf-8")
    )
    video = next(x for x in manifest["videos"] if x["video_id"] == "urbanego_run8_05")
    capture = cv2.VideoCapture(video["filepath"])
    ok, frame = capture.read()
    capture.release()
    assert ok
    model = SegformerB0Perceptor(
        ROOT / "models/path_relevance/segformer_mit_b0_ade20k_160k.pth",
        device="cuda:0",
    )
    prediction = model.predict(frame)
    row = read_rows(ROOT / "outputs/spatial_semantics/v01/urbanego_run8_05/semantic_frames.jsonl")[
        0
    ]
    with np.load(
        ROOT / "outputs/spatial_semantics/v01/urbanego_run8_05" / row["data_reference"]
    ) as data:
        np.testing.assert_array_equal(prediction.class_map, data["class_map"])
        np.testing.assert_allclose(prediction.confidence_map, data["confidence_map"], atol=1e-5)
    assert prediction.class_map.shape == (288, 512)
