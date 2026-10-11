"""Image observation semantics, exclusions, failure handling and isolation."""

import ast
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sightguard.scene.corridor.artifacts import parse_spatial_frame, reference_inside
from sightguard.scene.corridor.candidates import CandidateGenerator
from sightguard.scene.corridor.engine import CorridorEvaluator
from sightguard.scene.corridor.schema import CandidateConfig, CorridorConfig, EvaluationConfig
from sightguard.scene.spatial_grid import (
    EvidenceSource,
    FrameKey,
    SpatialGridFusion,
    SpatialGridInput,
)


def fixture(width=31, height=37, classes=11, detections=()):
    key = FrameKey("fixture", 0, 0.0, width, height)
    source = EvidenceSource(key, "valid", "high", "synthetic", "fixture", "fixture.v1")
    data = SpatialGridInput(
        key,
        source,
        source,
        source,
        np.full((height, width), classes, np.uint8),
        np.full((height, width), 0.9, np.float32),
        np.ones((height, width), np.int32),
        np.linspace(0.1, 2, width * height, dtype=np.float32).reshape(height, width),
        detections,
    )
    return SpatialGridFusion().fuse(data)


def box(track_id=7, confidence=0.9):
    return {
        "bbox_xyxy": [8, 22, 23, 36],
        "confidence": confidence,
        "class_id": 0,
        "class_name": "person",
        "track_id": track_id,
    }


def evaluator(**kwargs):
    return CorridorEvaluator(CorridorConfig(evaluation=EvaluationConfig(**kwargs)))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"top_centers": (0.2,)},
        {"names": ("LEFT", "MID", "RIGHT")},
        {"names": ("IMAGE_A",) * 3},
        {"start_row_fraction": 0},
        {"near_row_fraction": 0.3},
        {"bottom_width_fraction": float("nan")},
        {"row_weight_gain": -1},
        {"top_width_fraction": True},
    ],
)
def test_invalid_candidate_config(kwargs):
    with pytest.raises(ValueError):
        CandidateConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ground_weight": 0.9},
        {"min_depth_coverage": 0},
        {"min_ground_support": 0},
        {"tie_margin": float("nan")},
        {"max_unknown": -1},
        {"require_observed_orientation": 1},
        {"depth_near_fraction": True},
        {"road_support_factor": 1.1},
    ],
)
def test_invalid_evaluation_config(kwargs):
    with pytest.raises(ValueError):
        EvaluationConfig(**kwargs)


@pytest.mark.parametrize("size", [(30, 30), (31, 37), (1280, 720)])
def test_geometry_valid_even_on_nondivisible_images(size):
    frame = fixture(*size)
    candidates = CandidateGenerator(CandidateConfig()).generate(frame)
    for candidate in candidates:
        assert candidate.key == frame.key
        assert len(set(candidate.cell_ids)) == len(candidate.cell_ids)
        assert set(candidate.near_cell_ids) <= set(candidate.cell_ids)
        for x1, y1, x2, y2 in candidate.row_bounds_xyxy:
            assert 0 <= x1 < x2 <= size[0] and 0 <= y1 < y2 <= size[1]
    assert set(candidates[0].near_cell_ids) & set(candidates[1].near_cell_ids)
    for candidate in candidates:
        points = candidate.polygon_xy
        polygon_area = (
            abs(
                sum(
                    x1 * y2 - x2 * y1
                    for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1], strict=True)
                )
            )
            / 2
        )
        cell_area = sum(
            (frame.cells[x].bounds_xyxy[2] - frame.cells[x].bounds_xyxy[0])
            * (frame.cells[x].bounds_xyxy[3] - frame.cells[x].bounds_xyxy[1])
            for x in candidate.cell_ids
        )
        assert polygon_area == cell_area


def test_same_geometry_in_every_ablation():
    engine = CorridorEvaluator()
    frame = fixture()
    results = [engine.assess(frame, name) for name in "ABCD"]
    assert all(
        [x.candidate for x in r.candidates] == [x.candidate for x in results[0].candidates]
        for r in results
    )
    assert results[0].status == "ABSTAIN"
    assert all(x.total_cost is None for x in results[0].candidates)


def test_empty_ground_does_not_force_valid_corridors():
    result = CorridorEvaluator().assess(fixture(classes=2), "C")
    assert result.status == "ABSTAIN" and result.ranking == ()
    assert all(x.total_cost is None for x in result.candidates)


def test_bottom_continuity_requires_connected_component_and_adjacency():
    frame = fixture()
    broken = replace(
        frame,
        cells=tuple(
            replace(x, ground=replace(x.ground, bottom_connected_component_ids=()))
            if x.row == 12
            else x
            for x in frame.cells
        ),
    )
    result = CorridorEvaluator().assess(broken, "B")
    assert all(x.evidence.ground_continuity == 2 / 9 for x in result.candidates)
    assert all(12 in x.evidence.interrupted_rows for x in result.candidates)


@pytest.mark.parametrize("classes,name", [(3, "floor"), (6, "road"), (11, "sidewalk")])
def test_ground_classes_are_separate_and_not_safety(classes, name):
    result = CorridorEvaluator().assess(fixture(classes=classes), "B")
    evidence = result.candidates[1].evidence
    assert evidence.class_fractions[name] == 1
    assert evidence.class_adjusted_ground_support <= evidence.ground_support
    assert "safe" not in json.dumps(result.to_dict()).lower()


def test_tie_has_no_forced_unique_selection():
    result = CorridorEvaluator().assess(fixture(), "B")
    assert result.status == "TIED" and result.best_supported_candidate is None
    assert len(result.tied_candidates) == 3


def test_default_orientation_unknown_abstains_even_without_limitation_string():
    result = CorridorEvaluator().assess(replace(fixture(), limitations=()), "D")
    assert result.status == "ABSTAIN"
    assert "camera_orientation_unobserved" in result.abstain_reasons


def test_explicit_gate_toggle_is_research_configuration():
    result = evaluator(require_observed_orientation=False).assess(fixture(), "D")
    assert result.status == "TIED"
    assert all(x.evidence.camera_orientation == "unknown" for x in result.candidates)


def test_multiple_tracks_same_box_retained_without_duplicate_occupancy_sum():
    one = CorridorEvaluator().assess(fixture(detections=(box(),)), "C")
    two = CorridorEvaluator().assess(fixture(detections=(box(), box(8))), "C")
    assert (
        one.candidates[1].evidence.bbox_proxy_coverage
        == two.candidates[1].evidence.bbox_proxy_coverage
    )
    assert {x.track_id for x in two.candidates[1].object_relations} == {7, 8}
    assert all(
        x.event_reference is None and x.quality == "limited"
        for x in two.candidates[1].object_relations
    )


def test_nullable_track_and_background_uncertainty():
    result = CorridorEvaluator().assess(fixture(detections=(box(None),)), "C")
    relation = result.candidates[1].object_relations[0]
    assert relation.track_id is None
    assert "bbox_contains_background_not_instance_mask" in relation.reasons
    assert relation.background_depth_percentile is not None


def test_relative_near_vs_far_changes_cost_without_metric_claim():
    frame = fixture(detections=(box(),))
    engine = CorridorEvaluator()
    far = replace(
        frame,
        cells=tuple(replace(x, depth=replace(x.depth, frame_percentile=0.1)) for x in frame.cells),
    )
    near = replace(
        frame,
        cells=tuple(replace(x, depth=replace(x.depth, frame_percentile=0.9)) for x in frame.cells),
    )
    a = engine.assess(far, "C").candidates[1]
    b = engine.assess(near, "C").candidates[1]
    assert a.weighted_contributions["depth"] < b.weighted_contributions["depth"]
    assert a.total_cost < b.total_cost
    assert (
        engine.assess(far, "B").candidates[1].total_cost
        == engine.assess(near, "B").candidates[1].total_cost
    )


def test_b_excludes_depth_conflicts_and_relations():
    frame = fixture(detections=(box(),))
    altered = replace(
        frame,
        cells=tuple(
            replace(x, conflict_reasons=("relative_depth_variation_not_object_boundary",))
            for x in frame.cells
        ),
    )
    result = CorridorEvaluator().assess(altered, "B")
    assert all(
        x.evidence.conflict_coverage == 0 and x.cost_components["depth"] is None
        for x in result.candidates
    )
    assert all(
        r.relative_depth_percentile is None for c in result.candidates for r in c.object_relations
    )


@pytest.mark.parametrize("source", ["semantic", "detection", "depth"])
def test_missing_sources_remain_unknown_or_ineligible(source):
    frame = fixture()
    statuses = {**frame.source_status, source: "missing"}
    ground = tuple(
        replace(
            x,
            source_status=statuses,
            ground=replace(x.ground, status="unknown") if source == "semantic" else x.ground,
        )
        for x in frame.cells
    )
    frame = replace(
        frame,
        source_status=statuses,
        cells=ground,
        sources={
            name: replace(value, status="missing") if name == source else value
            for name, value in frame.sources.items()
        },
    )
    result = CorridorEvaluator().assess(frame, "D")
    assert result.status == "ABSTAIN"
    assert all(x.evidence.unknown_exposure == 1 for x in result.candidates)


def test_conflict_gate_separate_from_numeric_cost():
    frame = fixture()
    frame = replace(
        frame,
        cells=tuple(
            replace(x, conflict_reasons=("ground_and_bbox_candidate_overlap",)) for x in frame.cells
        ),
    )
    result = evaluator(require_observed_orientation=False).assess(frame, "D")
    assert all(
        x.total_cost is None and "conflicting_evidence" in x.eligibility_reasons
        for x in result.candidates
    )
    assert all(x.cost_components["conflict"] == 1 for x in result.candidates)


@pytest.mark.parametrize(
    "mutation", ["schema", "cells", "timestamp", "quantile", "nan", "coverage"]
)
def test_malformed_saved_grid_rejected(mutation):
    row = fixture(detections=(box(),)).to_dict()
    if mutation == "schema":
        row["schema_version"] = "wrong"
    elif mutation == "cells":
        row["cells"] = row["cells"][:-1]
    elif mutation == "timestamp":
        row["sources"]["depth"]["key"]["timestamp_ms"] = 1
    elif mutation == "quantile":
        row["cells"][0]["depth"]["q10"] = 100
    elif mutation == "nan":
        row["cells"][0]["depth"]["frame_percentile"] = float("nan")
    else:
        row["object_relations"][0]["cell_coverage"] = (
            {"cell_id": 999, "cell_fraction": 1, "bbox_fraction": 1},
        )
    with pytest.raises(ValueError):
        parse_spatial_frame(row)


def test_serialization_determinism_and_provenance():
    frame = fixture()
    engine = CorridorEvaluator()
    first = engine.assess(frame, "C", provenance={"spatial_sha256": "fixture"}).to_dict()
    engine.assess(fixture(classes=2), "D")
    assert first == engine.assess(frame, "C", provenance={"spatial_sha256": "fixture"}).to_dict()
    assert json.loads(json.dumps(first))["provenance"]["spatial_sha256"] == "fixture"
    assert not first["production_decision_changed"]


def test_multiple_videos_never_share_frame_identity():
    engine = CorridorEvaluator()
    frame = fixture()
    other_key = replace(frame.key, video_id="second", timestamp_ms=500, frame_index=10)
    other = replace(
        frame,
        key=other_key,
        sources={name: replace(source, key=other_key) for name, source in frame.sources.items()},
    )
    assert engine.assess(other, "B").key != engine.assess(frame, "B").key
    with pytest.raises(ValueError):
        engine.aggregate(other, engine.generator.generate(frame)[0], "B")


def test_adapter_roundtrip_and_reference_escape(tmp_path):
    frame = fixture(detections=(box(),))
    assert parse_spatial_frame(json.loads(json.dumps(frame.to_dict()))) == frame
    with pytest.raises(ValueError):
        reference_inside(tmp_path, "../escape.json")


def test_core_has_no_model_audio_risk_or_ui_imports():
    forbidden = (
        "torch",
        "ultralytics",
        "gradio",
        "sightguard.risk",
        "sightguard.decision",
        "sightguard.ui",
        "sightguard.audio",
    )
    for path in Path("src/sightguard/scene/corridor").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = (
                [x.name for x in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else []
            )
            assert not any(name.startswith(forbidden) for name in names)


def test_degenerate_depth_remains_unknown_not_midrank_near_evidence():
    frame = replace(fixture(detections=(box(),)), frame_depth_quantiles=(1.0, 1.0, 1.0))
    result = evaluator(require_observed_orientation=False).assess(frame, "D")
    assert result.status == "ABSTAIN"
    assert all(
        x.cost_components["depth"] is None and x.evidence.depth_coverage == 0
        for x in result.candidates
    )


def test_few_valid_depth_pixels_fail_coverage_gate():
    frame = fixture()
    frame = replace(
        frame, cells=tuple(replace(x, depth=replace(x.depth, valid_count=1)) for x in frame.cells)
    )
    result = evaluator(require_observed_orientation=False).assess(frame, "D")
    assert all(x.evidence.depth_coverage < 0.3 for x in result.candidates)
    assert all("relative_depth_coverage_gate" in x.eligibility_reasons for x in result.candidates)


def test_runner_records_preflight_failure_without_overwriting_outputs(tmp_path):
    from sightguard.pipeline.corridor import run_corridor_batch

    root = tmp_path / "source"
    root.mkdir()
    (root / "run_metadata.json").write_text(
        '{"status":"failed","production_decision_changed":false}', encoding="utf-8"
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"videos":[]}', encoding="utf-8")
    output = tmp_path / "result"
    with pytest.raises(ValueError, match="incomplete"):
        run_corridor_batch(root, output, Path("configs/corridor.yaml"), manifest, visualize=False)
    metadata = json.loads((output / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "failed" and metadata["error"]["type"] == "ValueError"
    before = (output / "run_metadata.json").read_bytes()
    with pytest.raises((ValueError, FileExistsError)):
        run_corridor_batch(root, output, Path("configs/corridor.yaml"), manifest, visualize=False)
    assert (output / "run_metadata.json").read_bytes() == before


def test_yaml_config_matches_validated_defaults():
    from sightguard.pipeline.corridor import load_config

    assert load_config(Path("configs/corridor.yaml")) == CorridorConfig()


def test_missing_source_initialization_failure_is_saved(tmp_path):
    from sightguard.pipeline.corridor import run_corridor_batch

    output = tmp_path / "failed"
    with pytest.raises(FileNotFoundError):
        run_corridor_batch(
            tmp_path / "missing",
            output,
            Path("configs/corridor.yaml"),
            tmp_path / "missing_manifest",
            visualize=False,
        )
    result = json.loads((output / "run_metadata.json").read_text(encoding="utf-8"))
    assert result["status"] == "failed" and result["error"]["type"] == "FileNotFoundError"
