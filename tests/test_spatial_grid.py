"""Numerical, isolation and failure tests without models, GPU or downloads."""

import ast
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sightguard.perception.depth.artifacts import sha256_file
from sightguard.pipeline.spatial_grid import ablation_input, evidence_metrics, run_spatial_batch
from sightguard.scene.spatial_grid import (
    EvidenceSource,
    FrameKey,
    SpatialGridConfig,
    SpatialGridFusion,
    SpatialGridInput,
)
from sightguard.scene.spatial_grid.artifacts import (
    SEMANTIC_SCHEMA,
    fresh_directory,
    load_inputs,
    load_semantic_maps,
)
from sightguard.scene.spatial_grid.engine import grid_bounds


def inputs(width=30, height=30):
    key = FrameKey("v", 0, 0.0, width, height)
    source = EvidenceSource(key, "valid", "high", "source.npz", "abc", "fixture.v1")
    shape = (height, width)
    return SpatialGridInput(
        key,
        source,
        source,
        source,
        np.full(shape, 6, np.uint8),
        np.full(shape, 0.8, np.float32),
        np.ones(shape, np.int32),
        np.arange(width * height, dtype=np.float32).reshape(shape),
        (),
    )


def detection(bbox=(0, 0, 30, 30), confidence=0.9, track_id=7):
    return {
        "bbox_xyxy": list(bbox),
        "confidence": confidence,
        "class_id": 0,
        "class_name": "person",
        "track_id": track_id,
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rows": 0},
        {"cols": 129},
        {"rows": 1.5},
        {"cols": True},
        {"schema_version": "unknown"},
        {"min_pixel_confidence": float("nan")},
        {"ground_fraction_threshold": 0},
        {"min_detection_confidence": 1.1},
        {"occupancy_coverage_threshold": -0.1},
        {"depth_spread_fraction": float("inf")},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        SpatialGridConfig(**kwargs)


@pytest.mark.parametrize("width,height", [(30, 30), (31, 37), (1280, 720)])
def test_grid_covers_each_pixel_exactly_once(width, height):
    bounds = grid_bounds(width, height, SpatialGridConfig())
    counts = np.zeros((height, width), np.uint8)
    for x1, y1, x2, y2 in bounds:
        counts[y1:y2, x1:x2] += 1
    assert len(bounds) == 225
    assert np.all(counts == 1)
    assert bounds[-1][2:] == (width, height)


def test_too_small_image_rejected():
    with pytest.raises(ValueError):
        SpatialGridFusion().fuse(inputs(14, 30))


@pytest.mark.parametrize("field", ["semantic_source", "depth_source", "detection_source"])
@pytest.mark.parametrize(
    "key_change",
    [
        {"video_id": "other"},
        {"frame_index": 1},
        {"timestamp_ms": 0.001},
        {"width": 31},
    ],
)
def test_sources_never_reuse_other_frame_or_video(field, key_change):
    data = inputs()
    invalid = replace(getattr(data, field), key=replace(data.key, **key_change))
    frame = SpatialGridFusion().fuse(replace(data, **{field: invalid}))
    name = field.removesuffix("_source")
    assert frame.source_status[name] == "unknown"
    assert any("mismatch" in x for x in frame.failures)


@pytest.mark.parametrize(
    "array_field", ["class_map", "confidence_map", "candidate_components", "depth_map"]
)
def test_missing_or_misaligned_arrays_are_unknown(array_field):
    data = inputs()
    for value in (None, getattr(data, array_field)[:20]):
        frame = SpatialGridFusion().fuse(replace(data, **{array_field: value}))
        assert (
            frame.source_status["depth" if array_field == "depth_map" else "semantic"] == "unknown"
        )


def test_ground_and_empty_detections_do_not_mean_free():
    frame = SpatialGridFusion().fuse(inputs())
    assert all(c.state == "GROUND_EVIDENCE" for c in frame.cells)
    assert not frame.object_relations
    assert "safe" not in json.dumps(frame.to_dict()).lower().replace(
        "not_metric_occupancy_or_safe_path", ""
    )


def test_all_unknown_and_no_ground():
    data = inputs()
    absent = SpatialGridFusion().fuse(SpatialGridInput(data.key))
    assert absent.status == "unknown"
    assert all(c.state == "UNKNOWN" for c in absent.cells)
    no_ground = SpatialGridFusion().fuse(replace(data, class_map=np.zeros((30, 30), np.uint8)))
    assert all(c.state == "UNKNOWN" and c.depth.valid_count == 4 for c in no_ground.cells)


def test_semantic_unknown_retains_depth_and_reverse():
    data = inputs()
    depth_only = SpatialGridFusion().fuse(
        replace(data, semantic_source=replace(data.semantic_source, status="unknown"))
    )
    assert all(c.ground.status == "unknown" and c.depth.valid_count == 4 for c in depth_only.cells)
    semantic_only = SpatialGridFusion().fuse(
        replace(data, depth_source=replace(data.depth_source, status="unknown"))
    )
    assert all(c.state == "GROUND_EVIDENCE" and c.depth.median is None for c in semantic_only.cells)
    assert all("depth_unknown" in c.unknown_reasons for c in semantic_only.cells)


def test_cross_cell_bbox_coverage_and_object_depth():
    data = replace(inputs(), detections=(detection((1, 1, 5, 5)),))
    frame = SpatialGridFusion().fuse(data)
    relation = frame.object_relations[0]
    assert len(relation.cell_coverage) == 9
    assert sum(x["bbox_fraction"] for x in relation.cell_coverage) == pytest.approx(1)
    assert relation.depth.median == pytest.approx(np.median(data.depth_map[1:5, 1:5]))
    assert relation.ground_fraction == 1
    assert relation.image_bearing == "image_left"
    assert frame.cells[0].state == "MIXED"
    assert frame.cells[0].detection_coverage[0].cell_fraction == 0.25
    assert "ground_and_bbox_candidate_overlap" in frame.cells[0].conflict_reasons


def test_multiple_objects_missing_track_and_clip():
    data = replace(
        inputs(),
        detections=(detection((-5, -5, 3, 3), track_id=None), detection((0, 0, 2, 2), track_id=8)),
    )
    frame = SpatialGridFusion().fuse(data)
    assert len(frame.cells[0].detection_coverage) == 2
    assert frame.object_relations[0].track_id is None
    assert frame.object_relations[0].clipped_bbox_xyxy == (0, 0, 3, 3)
    assert sum(
        x["bbox_fraction"] for x in frame.object_relations[0].cell_coverage
    ) == pytest.approx(1)


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"confidence": float("nan")}, "invalid_detection_confidence"),
        ({"confidence": -1}, "invalid_detection_confidence"),
        ({"confidence": 1.1}, "invalid_detection_confidence"),
        ({"confidence": None}, "invalid_detection_confidence"),
        ({"bbox_xyxy": [0, 0, float("inf"), 30]}, "invalid_bbox"),
        ({"bbox_xyxy": [3, 0, 2, 10]}, "invalid_bbox_extent"),
        ({"bbox_xyxy": [-3, -3, -1, -1]}, "bbox_outside_image"),
        ({"class_id": None}, "invalid_class_id"),
        ({"track_id": "x"}, "invalid_track_id"),
    ],
)
def test_malformed_detection_preserves_other_sources(changes, reason):
    frame = SpatialGridFusion().fuse(replace(inputs(), detections=({**detection(), **changes},)))
    assert frame.object_relations[0].status == "unknown"
    assert reason in frame.object_relations[0].reasons
    assert frame.cells[0].state == "GROUND_EVIDENCE"
    assert "invalid_detection_records_in_frame" in frame.cells[0].unknown_reasons


def test_low_confidence_not_occupancy_evidence():
    frame = SpatialGridFusion().fuse(replace(inputs(), detections=(detection(confidence=0.1),)))
    assert frame.cells[0].state == "GROUND_EVIDENCE"
    assert frame.cells[0].detection_coverage[0].candidate is False
    assert frame.object_relations[0].status == "limited"


def test_negative_nan_inf_excluded_without_mutation():
    data = inputs()
    data.depth_map[:2, :2] = [[-1, float("nan")], [float("inf"), 10]]
    original = data.depth_map.copy()
    frame = SpatialGridFusion().fuse(data)
    cell = frame.cells[0]
    assert cell.depth.valid_count == 1
    assert cell.depth.median == 10
    assert cell.depth.negative_fraction == 0.25
    assert cell.depth.nonfinite_fraction == 0.5
    assert cell.depth.invalid_fraction == 0.75
    assert cell.quality == "limited"
    np.testing.assert_array_equal(data.depth_map, original)
    json.dumps(frame.to_dict(), allow_nan=False)


def test_constant_depth_has_no_relative_order():
    frame = SpatialGridFusion().fuse(replace(inputs(), depth_map=np.ones((30, 30), np.float32)))
    assert all(c.depth.frame_percentile is None for c in frame.cells)


def test_depth_percentile_uses_exact_median_rank():
    data = inputs()
    frame = SpatialGridFusion().fuse(data)
    # First 2x2 cell contains 0,1,30,31; median 15.5 has rank 16/900.
    assert frame.cells[0].depth.frame_percentile == pytest.approx(16 / 900)


def test_conflicting_depth_variation_is_explained():
    data = inputs()
    data.depth_map[:, ::2] = 0
    data.depth_map[:, 1::2] = 10
    frame = SpatialGridFusion().fuse(data)
    assert all(
        "relative_depth_variation_not_object_boundary" in c.conflict_reasons for c in frame.cells
    )
    assert all(c.state == "GROUND_EVIDENCE" for c in frame.cells)


def test_determinism_provenance_and_session_isolation():
    engine = SpatialGridFusion()
    data = inputs()
    first = engine.fuse(data).to_dict()
    engine.fuse(SpatialGridInput(replace(data.key, video_id="other")))
    assert first == engine.fuse(data).to_dict()
    assert first["cells"][0]["provenance"]["depth_sha256"] == "abc"
    assert first["cells"][0]["ground"]["bottom_connected_component_ids"] == (1,)


def test_ablation_excludes_sources_without_read_failures():
    data = inputs()
    frames = {name: SpatialGridFusion().fuse(ablation_input(data, name)) for name in "ABCD"}
    assert not frames["A"].failures
    assert not frames["B"].failures
    assert all(c.state == "UNKNOWN" and c.depth.valid_count == 4 for c in frames["B"].cells)
    assert evidence_metrics(frames["A"])["invalid_depth_pixels"] == 0
    assert frames["C"].cells[0].depth.median is None


def test_output_rejects_existing_directory_and_batch_writes_failure(tmp_path):
    directory = tmp_path / "output"
    fresh_directory(directory)
    with pytest.raises(FileExistsError):
        fresh_directory(directory)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "fixture",
                "videos": [{"video_id": "v", "filepath": "missing.mp4", "processed_sha256": "bad"}],
            }
        )
    )
    for root in (tmp_path / "semantic", tmp_path / "depth"):
        root.mkdir()
        (root / "run_metadata.json").write_text("{}")
    with pytest.raises(FileNotFoundError):
        run_spatial_batch(
            manifest,
            tmp_path / "tracks",
            tmp_path / "semantic",
            tmp_path / "depth",
            tmp_path / "failed",
            Path("configs/spatial_grid.yaml"),
        )
    assert json.loads((tmp_path / "failed/run_metadata.json").read_text())["status"] == "failed"


def test_saved_semantic_hash_native_alignment_and_corruption(tmp_path):
    data = inputs()
    path = tmp_path / "maps.npz"
    np.savez_compressed(
        path,
        class_map=np.array([[3, 6], [11, 0]], np.uint8),
        confidence_map=np.ones((2, 2), np.float32),
        candidate_components=np.array([[1, 1], [2, 0]], np.int32),
    )
    row = {
        "schema_version": SEMANTIC_SCHEMA,
        "video_id": "v",
        "frame_index": 0,
        "timestamp_ms": 0.0,
        "source_frame_width": 30,
        "source_frame_height": 30,
        "native_shape": [2, 2],
        "status": "valid",
        "quality": "limited",
        "data_reference": "maps.npz",
        "data_sha256": sha256_file(path),
    }
    classes, _, components = load_semantic_maps(tmp_path, row, data.key)
    assert classes[0, 0] == 3 and classes[-1, -1] == 0
    assert components[-1, 0] == 2
    with pytest.raises(ValueError, match="SHA-256"):
        load_semantic_maps(tmp_path, {**row, "data_sha256": "wrong"}, data.key)
    track = {
        "frame_index": 0,
        "timestamp_ms": 0,
        "frame_width": 30,
        "frame_height": 30,
        "status": "ok",
        "detections": [],
    }
    broken = load_inputs(
        data.key, track, path, {**row, "data_reference": "missing.npz"}, tmp_path, None, tmp_path
    )
    assert broken.semantic_source.status == "unknown"
    assert "semantic_read_failed" in broken.semantic_source.reasons[0]


def test_no_model_decision_or_audio_dependencies():
    root = Path("src/sightguard/scene/spatial_grid")
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        names += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        assert not any(
            name.startswith(
                (
                    "torch",
                    "ultralytics",
                    "gradio",
                    "sightguard.risk",
                    "sightguard.decision",
                    "sightguard.audio",
                )
            )
            for name in names
        )
