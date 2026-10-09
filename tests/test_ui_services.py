from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

import sightguard.ui.services as services_module
from sightguard.evaluation.config import BenchmarkConfig
from sightguard.evaluation.dataset import MANIFEST_FIELDS, sha256_file
from sightguard.evaluation.protocol import freeze_protocol
from sightguard.evaluation.runner import benchmark
from sightguard.pipeline.assess import assess_tracks
from sightguard.risk.config import RiskConfig
from sightguard.ui.presentation import format_metric, metrics_frame
from sightguard.ui.services import ConsoleDataError, ConsoleStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_DATASET = PROJECT_ROOT / "datasets" / "synthetic"
CONFIGS = PROJECT_ROOT / "configs"


def _sha_tree(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _build_workspace(root: Path) -> Path:
    shutil.copytree(SYNTHETIC_DATASET, root / "datasets" / "synthetic")
    shutil.copytree(CONFIGS, root / "configs")

    source_tracks = root / "datasets" / "synthetic" / "tracks" / "test_stationary.jsonl"
    analysis_dir = root / "outputs" / "analysis" / "phase2"
    analysis_dir.mkdir(parents=True)
    shutil.copyfile(source_tracks, analysis_dir.parent / "tracks.jsonl")
    track_records = source_tracks.read_text(encoding="utf-8").splitlines()
    duration_ms = 4000
    _json_write(
        analysis_dir.parent / "run_metadata.json",
        {
            "schema_version": "1.0",
            "status": "completed",
            "source_kind": "synthetic",
            "duration_ms": duration_ms,
            "frames_processed": len(track_records),
            "input": {"path": "", "data_kind": "synthetic"},
        },
    )
    assess_tracks(
        analysis_dir.parent / "tracks.jsonl",
        analysis_dir,
        RiskConfig(),
        stream_id="phase2-fixture",
    )

    manifest = root / "datasets" / "synthetic" / "manifest.csv"
    benchmark_config = BenchmarkConfig(generate_figures=False)
    risk_config = RiskConfig()
    protocol = root / "outputs" / "protocols" / "synthetic-test.json"
    freeze_protocol(manifest, benchmark_config, risk_config, protocol)
    benchmark(
        manifest,
        root / "outputs" / "benchmarks" / "synthetic-test",
        benchmark_config,
        risk_config,
        split="test",
        protocol=protocol,
        data_kind="synthetic",
    )
    return root


@pytest.fixture(scope="session")
def workspace_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _build_workspace(tmp_path_factory.mktemp("ui-services-template") / "workspace")


@pytest.fixture
def workspace(tmp_path: Path, workspace_template: Path) -> Path:
    root = tmp_path / "workspace"
    shutil.copytree(workspace_template, root)

    # Persisted benchmark references are absolute so each copied test workspace must
    # point to its own managed dataset files when the UI verifies Ground Truth.
    benchmark_metadata = root / "outputs" / "benchmarks" / "synthetic-test" / "run_metadata.json"
    metadata = json.loads(benchmark_metadata.read_text(encoding="utf-8"))
    metadata["dataset"] = str(root / "datasets" / "synthetic" / "manifest.csv")
    for item in metadata["inputs"]:
        for name in ("tracks_path", "annotations_path"):
            item[name] = str(root / "datasets" / "synthetic" / Path(item[name]).name)
            # The manifest assets are in tracks/ and annotations/ subdirectories.
            subdirectory = "tracks" if name == "tracks_path" else "annotations"
            item[name] = str(root / "datasets" / "synthetic" / subdirectory / Path(item[name]).name)
    _json_write(benchmark_metadata, metadata)

    analysis_metadata = root / "outputs" / "analysis" / "run_metadata.json"
    analysis = json.loads(analysis_metadata.read_text(encoding="utf-8"))
    analysis["input"]["path"] = ""
    _json_write(analysis_metadata, analysis)
    return root


def test_lists_persisted_analyses_benchmarks_and_manifests(workspace: Path) -> None:
    store = ConsoleStore(workspace)

    assert store.analysis_runs() == ["analysis"]
    assert store.benchmark_runs() == ["benchmarks/synthetic-test"]
    assert store.dataset_manifests() == ["synthetic/manifest.csv"]


def test_load_analysis_reads_empty_frames_without_inventing_scores(workspace: Path) -> None:
    run_dir = workspace / "outputs" / "analysis" / "empty"
    run_dir.mkdir(parents=True)
    (run_dir / "tracks.jsonl").write_text("", encoding="utf-8")
    _json_write(
        run_dir / "run_metadata.json",
        {
            "schema_version": "1.0",
            "status": "completed",
            "source_kind": "synthetic",
            "frames_processed": 0,
        },
    )

    view = ConsoleStore(workspace).load_analysis("analysis/empty")
    image, observations, alerts, info = ConsoleStore(workspace).select_frame(view, 0)

    assert view.records == view.risk_frames == view.events == view.alerts == ()
    assert view.stats["frame_count"] == 0
    assert image is None
    assert observations.empty and alerts.empty
    assert "No perception frames" in info
    assert any("No saved Phase 2" in warning for warning in view.warnings)


def _create_sibling_assessment(workspace: Path, job_id: str) -> tuple[Path, Path]:
    perception_dir = workspace / "outputs" / "console" / job_id / "perception"
    perception_dir.mkdir(parents=True)
    source_tracks = workspace / "datasets" / "synthetic" / "tracks" / "test_stationary.jsonl"
    tracks_path = perception_dir / "tracks.jsonl"
    shutil.copyfile(source_tracks, tracks_path)
    _json_write(
        perception_dir / "run_metadata.json",
        {
            "schema_version": "1.0",
            "status": "completed",
            "source_kind": "synthetic",
            "duration_ms": 4000,
            "frames_processed": 16,
        },
    )
    assess_tracks(
        tracks_path,
        perception_dir.parent / "assessment",
        RiskConfig(),
        stream_id=job_id,
    )
    return perception_dir, tracks_path


def test_load_analysis_reads_matching_phase2_assessment_sibling(workspace: Path) -> None:
    perception_dir, tracks_path = _create_sibling_assessment(workspace, "job-001")

    view = ConsoleStore(workspace).load_analysis("console/job-001/perception")

    assert len(view.records) == 16
    assert len(view.risk_frames) == 16
    assert view.events
    assert view.source_kind == "synthetic"
    assert (
        view.metadata["assessment_metadata"]["input"]["sha256"]
        == hashlib.sha256(tracks_path.read_bytes()).hexdigest()
    )
    runs = ConsoleStore(workspace).analysis_runs()
    assert "console/job-001/perception" in runs
    assert "console/job-001/assessment" not in runs


def test_mismatched_sibling_assessment_hash_is_not_presented(workspace: Path) -> None:
    perception_dir, tracks_path = _create_sibling_assessment(workspace, "job-mismatch")
    rows = [json.loads(line) for line in tracks_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["frame_width"] += 1
    tracks_path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )

    view = ConsoleStore(workspace).load_analysis("console/job-mismatch/perception")

    assert len(view.records) == 16
    assert view.risk_frames == view.events == view.alerts == ()
    assert any("input hash does not match" in warning for warning in view.warnings)
    assert perception_dir.is_dir()


def test_dataset_tracks_under_outputs_are_browsable_validatable_and_previewable(
    workspace: Path,
) -> None:
    manifest = workspace / "datasets" / "synthetic" / "manifest.csv"
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    selected = next(row for row in rows if row["video_id"] == "test_stationary")
    output_tracks = workspace / "outputs" / "shared-perception" / "test_stationary.jsonl"
    output_tracks.parent.mkdir(parents=True)
    shutil.copyfile(manifest.parent / selected["tracks_path"], output_tracks)
    selected["tracks_path"] = os.path.relpath(output_tracks, manifest.parent).replace("\\", "/")
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    benchmark_metadata_path = (
        workspace / "outputs" / "benchmarks" / "synthetic-test" / "run_metadata.json"
    )
    metadata = json.loads(benchmark_metadata_path.read_text(encoding="utf-8"))
    metadata["manifest_sha256"] = sha256_file(manifest)
    for item in metadata["inputs"]:
        if item["video_id"] == "test_stationary":
            item["tracks_path"] = str(output_tracks)
    _json_write(benchmark_metadata_path, metadata)

    store = ConsoleStore(workspace)
    table, _ = store.browse_dataset("synthetic/manifest.csv")
    validation = store.validate_dataset("synthetic/manifest.csv")
    preview = store.strategy_preview(
        "synthetic/manifest.csv", "test_stationary", ("C",), "benchmark.yaml", "risk.yaml"
    )
    benchmark_view = store.load_benchmark("benchmarks/synthetic-test")

    row = table.loc[table["video_id"] == "test_stationary"].iloc[0]
    assert bool(row["tracks_exists"])
    assert "Validation passed" in validation
    assert preview.alerts
    assert "test_stationary" in benchmark_view.labels


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("json", "Invalid risk frames line 1"),
        ("missing_field", "missing observations/events arrays"),
        ("invalid_type", "must be a finite number"),
    ],
)
def test_load_analysis_rejects_corrupted_or_invalid_risk_records(
    workspace: Path, corruption: str, message: str
) -> None:
    risk_path = workspace / "outputs" / "analysis" / "phase2" / "risk_frames.jsonl"
    if corruption == "json":
        risk_path.write_text("{broken json}\n", encoding="utf-8")
    else:
        row = json.loads(risk_path.read_text(encoding="utf-8").splitlines()[0])
        if corruption == "missing_field":
            row.pop("observations")
        else:
            row["timestamp_ms"] = "zero"
        risk_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ConsoleDataError, match=message):
        ConsoleStore(workspace).load_analysis("analysis")


def test_load_analysis_rejects_malformed_perception_jsonl(workspace: Path) -> None:
    (workspace / "outputs" / "analysis" / "tracks.jsonl").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ConsoleDataError, match="Cannot load perception records"):
        ConsoleStore(workspace).load_analysis("analysis")


@pytest.mark.parametrize(
    ("declared", "expected"),
    [("synthetic", "synthetic"), ("real", "real"), (None, "unknown")],
)
def test_source_kind_is_declared_or_unknown_never_inferred_from_video(
    workspace: Path, declared: str | None, expected: str
) -> None:
    metadata_path = workspace / "outputs" / "analysis" / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if declared is None:
        tracks = workspace / "outputs" / "analysis" / "tracks.jsonl"
        rows = [json.loads(line) for line in tracks.read_text(encoding="utf-8").splitlines()]
        rows[0]["frame_width"] += 1
        opaque_dir = workspace / "outputs" / "opaque"
        opaque_dir.mkdir()
        (opaque_dir / "tracks.jsonl").write_text(
            "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
        )
        unknown_metadata = {
            "schema_version": "1.0",
            "status": "completed",
            "duration_ms": metadata["duration_ms"],
            "frames_processed": len(rows),
        }
        _json_write(opaque_dir / "run_metadata.json", unknown_metadata)
        view = ConsoleStore(workspace).load_analysis("opaque")
        assert view.source_kind == "unknown"
        assert any("Source kind is unknown" in warning for warning in view.warnings)
        return
    else:
        metadata["source_kind"] = declared
    _json_write(metadata_path, metadata)

    view = ConsoleStore(workspace).load_analysis("analysis")

    assert view.source_kind == expected
    assert not any("Source kind is unknown" in warning for warning in view.warnings)


def test_select_frame_uses_nearest_timestamp_and_returns_rgb_and_tables(
    workspace: Path, video: Path
) -> None:
    source = workspace / "outputs" / "analysis" / "source.mp4"
    shutil.copyfile(video, source)
    metadata_path = workspace / "outputs" / "analysis" / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["input"]["path"] = str(source)
    _json_write(metadata_path, metadata)
    store = ConsoleStore(workspace)
    view = store.load_analysis("analysis")

    image, observations, alerts, info = store.select_frame(view, 300)

    assert image is not None and image.shape == (48, 64, 3)
    assert isinstance(observations, pd.DataFrame) and not observations.empty
    assert "class_id" in observations and "risk_score" in observations
    assert {
        "bbox_area",
        "bottom_y",
        "focus_relevance",
        "spatial_score",
        "appearance_score",
        "motion_score",
        "quality_score",
        "observation_duration_ms",
        "observation_interval_ms",
        "motion_interval_ms",
        "scale_change_per_s",
        "inward_change_per_s",
        "decision_reason",
    } <= set(observations.columns)
    assert isinstance(alerts, pd.DataFrame)
    assert "selected frame 1 at 250.000 ms" in info
    assert "preview frame decoded" in info
    if view.alerts:
        _, _, alert_rows, _ = store.select_frame(view, view.alerts[0]["timestamp_ms"])
        assert not alert_rows.empty
        assert "reason" in alert_rows


def test_benchmark_view_preserves_exact_stored_counts_and_validated_labels(workspace: Path) -> None:
    run_id = "benchmarks/synthetic-test"
    path = workspace / "outputs" / run_id / "metrics.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    view = ConsoleStore(workspace).load_benchmark(run_id)

    assert view.metrics == raw
    assert {"test_stationary", "test_missed"} <= set(view.labels)
    assert (view.metrics["C"]["tp"], view.metrics["C"]["fp"], view.metrics["C"]["fn"]) == (
        3,
        1,
        1,
    )


def test_changed_labels_are_omitted_without_recomputing_stored_metrics(workspace: Path) -> None:
    run_id = "benchmarks/synthetic-test"
    metrics_path = workspace / "outputs" / run_id / "metrics.json"
    expected_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    annotation_path = workspace / "datasets" / "synthetic" / "annotations" / "test_stationary.json"
    annotation_path.write_text(annotation_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    view = ConsoleStore(workspace).load_benchmark(run_id)

    assert view.metrics == expected_metrics
    assert "test_stationary" not in view.labels
    assert any("changed since the benchmark" in warning for warning in view.warnings)


def test_strategy_preview_is_behavior_only_and_does_not_load_ground_truth(workspace: Path) -> None:
    store = ConsoleStore(workspace)
    before = _sha_tree(workspace / "datasets")

    preview = store.strategy_preview(
        "synthetic/manifest.csv",
        "test_stationary",
        ("C",),
        "benchmark.yaml",
        "risk.yaml",
    )

    assert preview.metrics == {}
    assert preview.matches == ()
    assert preview.labels == {}
    assert preview.alerts
    assert preview.metadata["ground_truth_evaluated"] is False
    assert any("no TP/FP/FN" in warning for warning in preview.warnings)
    assert _sha_tree(workspace / "datasets") == before


def test_browse_and_validate_dataset_report_reviewed_fixture_data(workspace: Path) -> None:
    store = ConsoleStore(workspace)

    table, summary = store.browse_dataset("synthetic/manifest.csv")
    validation = store.validate_dataset("synthetic/manifest.csv")

    assert len(table) == 5
    assert set(table["annotation_status"]) == {"reviewed"}
    assert "Loaded 5 manifest entries" in summary
    assert "Validation passed: 5 reviewed videos" in validation


def test_config_listing_and_resolution_are_scoped_to_yaml_under_configs(workspace: Path) -> None:
    store = ConsoleStore(workspace)

    assert "risk.yaml" in store.configs("risk")
    assert store.config_path("benchmark.yaml").is_file()
    with pytest.raises(ConsoleDataError, match="Invalid Configuration ID"):
        store.config_path("../outputs/benchmarks/synthetic-test/metrics.json")
    with pytest.raises(ConsoleDataError, match="Configuration kind"):
        store.configs("unknown")


def test_nonexistent_run_and_path_traversal_are_rejected(workspace: Path) -> None:
    store = ConsoleStore(workspace)

    with pytest.raises(ConsoleDataError, match="does not exist"):
        store.load_analysis("missing-run")
    with pytest.raises(ConsoleDataError, match="Invalid Run ID"):
        store.load_analysis("../outside")
    with pytest.raises(ConsoleDataError, match="Invalid Dataset manifest ID"):
        store._manifest_path("../outside/manifest.csv")


def test_symlinked_run_outside_managed_outputs_is_rejected(workspace: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "run_metadata.json").write_text("{}\n", encoding="utf-8")
    (outside / "tracks.jsonl").write_text("", encoding="utf-8")
    link = workspace / "outputs" / "linked-run"
    os.symlink(outside, link, target_is_directory=True)

    with pytest.raises(ConsoleDataError, match="escapes the allowed directory"):
        ConsoleStore(workspace).load_analysis("linked-run")


def test_repeated_reads_are_isolated_and_do_not_change_inputs(workspace: Path) -> None:
    store = ConsoleStore(workspace)
    before = _sha_tree(workspace / "datasets") | {
        f"outputs/{key}": value for key, value in _sha_tree(workspace / "outputs").items()
    }

    analysis_first = store.load_analysis("analysis")
    benchmark_first = store.load_benchmark("benchmarks/synthetic-test")
    analysis_first.metadata["mutated"] = True
    benchmark_first.metrics["C"]["tp"] = 999
    analysis_second = store.load_analysis("analysis")
    benchmark_second = store.load_benchmark("benchmarks/synthetic-test")
    after = _sha_tree(workspace / "datasets") | {
        f"outputs/{key}": value for key, value in _sha_tree(workspace / "outputs").items()
    }

    assert "mutated" not in analysis_second.metadata
    assert benchmark_second.metrics["C"]["tp"] == 3
    assert before == after


def test_export_creates_distinct_archives_without_overwriting_prior_exports(
    workspace: Path,
) -> None:
    store = ConsoleStore(workspace)

    first = store.export_analysis("analysis")
    first_bytes = first.read_bytes()
    second = store.export_analysis("analysis")

    assert first != second
    assert first.read_bytes() == first_bytes
    assert zipfile.is_zipfile(first) and zipfile.is_zipfile(second)
    with zipfile.ZipFile(first) as archive:
        assert {"tracks.jsonl", "run_metadata.json", "assessment/risk_frames.jsonl"} <= set(
            archive.namelist()
        )


def test_export_uuid_collision_never_deletes_preexisting_archive(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ConsoleStore(workspace)
    export_dir = workspace / "outputs" / "console" / "exports"
    export_dir.mkdir(parents=True)
    collision = export_dir / "collision.zip"
    collision.write_bytes(b"preexisting archive")
    monkeypatch.setattr(services_module.uuid, "uuid4", lambda: SimpleNamespace(hex="collision"))

    with pytest.raises(FileExistsError):
        store.export_analysis("analysis")

    assert collision.read_bytes() == b"preexisting archive"


def test_null_metrics_remain_missing_in_presentation(workspace: Path) -> None:
    frame = metrics_frame({"A": {"event_recall": None}, "B": {"event_recall": 0.0}})

    assert frame.loc["A", "event_recall"] is None
    assert frame.loc["B", "event_recall"] == 0.0
    assert format_metric(None, percent=True) == "N/A"


def test_app_builds_and_launches_on_local_cpu_without_loading_weights(workspace: Path) -> None:
    import gradio as gr

    from sightguard.ui.app import build_app

    app = build_app(workspace)
    assert isinstance(app, gr.Blocks)
    assert app.sightguard_store.root == workspace.resolve()

    try:
        launched = app.launch(
            server_name="127.0.0.1",
            server_port=None,
            share=False,
            prevent_thread_lock=True,
            quiet=True,
            show_error=False,
        )
        assert launched[1] is not None
    finally:
        app.close()
