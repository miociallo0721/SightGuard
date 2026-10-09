from __future__ import annotations

import csv
import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

import sightguard.evaluation.dataset as dataset_module
import sightguard.evaluation.importing as importing_module
from sightguard.evaluation.config import BenchmarkConfig, MatchingConfig, load_benchmark_config
from sightguard.evaluation.dataset import (
    MANIFEST_FIELDS,
    DatasetError,
    load_dataset,
    read_manifest,
    sha256_file,
)
from sightguard.evaluation.importing import import_video, seal_labels
from sightguard.evaluation.protocol import check_protocol, freeze_protocol
from sightguard.evaluation.runner import benchmark
from sightguard.evaluation.schema import VideoAnnotations, load_annotations
from sightguard.risk.config import RiskConfig, load_risk_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_DATASET = PROJECT_ROOT / "datasets" / "synthetic"


def _copy_dataset(tmp_path: Path) -> Path:
    destination = tmp_path / "dataset"
    shutil.copytree(SYNTHETIC_DATASET, destination)
    return destination / "manifest.csv"


def _read_rows(manifest: Path) -> list[dict[str, str]]:
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_rows(manifest: Path, rows: list[dict[str, str]]) -> None:
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _entry_path(manifest: Path, row: dict[str, str], field: str) -> Path:
    return (manifest.parent / row[field]).resolve()


def _set_hash(row: dict[str, str], field: str, path: Path) -> None:
    row[field] = sha256_file(path)


def _test_rows(manifest: Path) -> list[dict[str, str]]:
    return [row for row in _read_rows(manifest) if row["split"] == "test"]


def test_config_loaders_build_nested_configs(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmark.yaml"
    benchmark_path.write_text(
        "baseline_classes: [0, 2]\n"
        "fixed_cooldown_ms: 2500\n"
        "matching:\n"
        "  minimum_iou: 0.25\n"
        "  important_severities: [HIGH]\n",
        encoding="utf-8",
    )
    risk_path = tmp_path / "risk.yaml"
    risk_path.write_text(
        "features:\n  history_window_ms: 2000\nalerts:\n  minimum_level: HIGH\n",
        encoding="utf-8",
    )

    benchmark_config = load_benchmark_config(benchmark_path)
    risk_config = load_risk_config(risk_path)

    assert benchmark_config.baseline_classes == (0, 2)
    assert benchmark_config.fixed_cooldown_ms == 2500
    assert benchmark_config.matching == MatchingConfig(0.25, ("HIGH",))
    assert risk_config.features.history_window_ms == 2000
    assert risk_config.alerts.minimum_level == "HIGH"


def test_draft_annotation_serialization_requires_explicit_opt_in(tmp_path: Path) -> None:
    draft = VideoAnnotations(
        video_id="draft-video",
        duration_ms=1000,
        data_kind="synthetic",
        annotation_status="draft",
        annotator_id="UNASSIGNED",
        label_source="pending_human_review",
        source_video_sha256="synthetic-placeholder",
        notes="Draft template only",
        events=(),
    )
    path = tmp_path / "draft.json"
    path.write_text(draft.to_json() + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Draft annotation"):
        load_annotations(path)
    loaded = load_annotations(path, require_reviewed=False)
    assert loaded == draft
    assert json.loads(loaded.to_json())["annotation_status"] == "draft"


@pytest.mark.parametrize("leak_field", ("source_group_id", "video_sha256", "tracks_sha256"))
def test_manifest_rejects_cross_split_source_video_and_tracks_leaks(
    tmp_path: Path, leak_field: str
) -> None:
    manifest = _copy_dataset(tmp_path)
    rows = _read_rows(manifest)
    development = next(row for row in rows if row["split"] == "development")
    test = next(row for row in rows if row["split"] == "test")
    development[leak_field] = test[leak_field]
    _write_rows(manifest, rows)

    with pytest.raises(DatasetError, match="Development/test leakage"):
        read_manifest(manifest)


def test_manifest_rejects_duplicate_video_ids_across_rows(tmp_path: Path) -> None:
    manifest = _copy_dataset(tmp_path)
    rows = _read_rows(manifest)
    rows[1]["video_id"] = rows[0]["video_id"]
    _write_rows(manifest, rows)

    with pytest.raises(DatasetError, match="Duplicate video_id"):
        read_manifest(manifest)


def test_missing_manifest_and_required_files_are_errors(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="manifest missing"):
        read_manifest(tmp_path / "absent.csv")

    for field in ("tracks_path", "annotations_path"):
        case_root = tmp_path / field
        manifest = _copy_dataset(case_root)
        row = next(row for row in _read_rows(manifest) if row["split"] == "test")
        _entry_path(manifest, row, field).unlink()
        with pytest.raises(DatasetError, match="file missing"):
            load_dataset(manifest)


@pytest.mark.parametrize(
    ("field", "message"),
    (("tracks_path", "Perception hash mismatch"), ("annotations_path", "Annotation hash mismatch")),
)
def test_tampered_tracks_or_labels_fail_hash_validation(
    tmp_path: Path, field: str, message: str
) -> None:
    manifest = _copy_dataset(tmp_path)
    row = next(row for row in _read_rows(manifest) if row["split"] == "test")
    path = _entry_path(manifest, row, field)
    path.write_bytes(path.read_bytes() + b"\n")

    with pytest.raises(DatasetError, match=message):
        load_dataset(manifest)


def test_incomplete_temporal_coverage_is_not_a_negative_sample(tmp_path: Path) -> None:
    manifest = _copy_dataset(tmp_path)
    rows = _read_rows(manifest)
    row = next(row for row in rows if row["video_id"] == "test_stationary")
    tracks = _entry_path(manifest, row, "tracks_path")
    records = tracks.read_text(encoding="utf-8").splitlines()
    tracks.write_text("\n".join(records[:-1]) + "\n", encoding="utf-8")
    _set_hash(row, "tracks_sha256", tracks)
    _write_rows(manifest, rows)

    with pytest.raises(DatasetError, match="timestamps do not cover"):
        load_dataset(manifest)


def test_real_dataset_rejects_failed_source_perception_metadata(tmp_path: Path) -> None:
    manifest = _copy_dataset(tmp_path)
    rows = _read_rows(manifest)
    row = next(row for row in rows if row["video_id"] == "test_nonrisk")
    video = manifest.parent / "source.mp4"
    video.write_bytes(b"placeholder video bytes for metadata validation")
    video_hash = sha256_file(video)
    annotation = _entry_path(manifest, row, "annotations_path")
    raw = json.loads(annotation.read_text(encoding="utf-8"))
    raw.update(
        data_kind="real",
        annotator_id="independent-reviewer",
        label_source="human_raw_video",
        source_video_sha256=video_hash,
    )
    annotation.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tracks = _entry_path(manifest, row, "tracks_path")
    source_metadata = tracks.parent / "run_metadata.json"
    source_metadata.write_text(
        json.dumps(
            {
                "status": "failed",
                "input": {"sha256": video_hash},
                "frames_processed": 12,
            }
        ),
        encoding="utf-8",
    )
    row.update(
        data_kind="real",
        video_path="source.mp4",
        video_sha256=video_hash,
    )
    _set_hash(row, "tracks_sha256", tracks)
    _set_hash(row, "annotations_sha256", annotation)
    _write_rows(manifest, rows)

    with pytest.raises(DatasetError, match="failed/incomplete"):
        load_dataset(manifest)


def test_import_rollback_preserves_manifest_and_import_manifest_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _copy_dataset(tmp_path)
    original_manifest = manifest.read_bytes()
    sentinel = manifest.parent / ".import_manifest.csv"
    sentinel.write_text("user-owned sentinel\n", encoding="utf-8")
    video = tmp_path / "new-video.mp4"
    tracks = tmp_path / "empty-tracks.jsonl"
    video.write_bytes(b"placeholder source")
    tracks.write_text("", encoding="utf-8")
    monkeypatch.setattr(importing_module, "probe_video", lambda _path: (1000.0, 4.0))
    monkeypatch.setattr(dataset_module, "probe_video", lambda _path: (1000.0, 4.0))

    with pytest.raises(DatasetError, match="no frame records"):
        import_video(
            video,
            tracks,
            manifest.parent,
            video_id="new-real-video",
            source_group_id="new-source-group",
            scene="new-scene",
        )

    assert manifest.read_bytes() == original_manifest
    assert sentinel.read_text(encoding="utf-8") == "user-owned sentinel\n"
    assert not (manifest.parent / "annotations" / "new-real-video.json").exists()
    assert sorted(path.name for path in manifest.parent.glob("*.csv")) == [
        ".import_manifest.csv",
        "manifest.csv",
    ]


def test_seal_labels_updates_reviewed_annotation_hash_without_rewriting_labels(
    tmp_path: Path,
) -> None:
    manifest = _copy_dataset(tmp_path)
    rows = _read_rows(manifest)
    row = next(row for row in rows if row["video_id"] == "test_stationary")
    annotation = _entry_path(manifest, row, "annotations_path")
    raw = json.loads(annotation.read_text(encoding="utf-8"))
    raw["notes"] = "reviewed notes changed before sealing"
    annotation.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_rows(manifest, rows)
    sealed_bytes = annotation.read_bytes()

    seal_labels(manifest)

    sealed_row = next(row for row in _read_rows(manifest) if row["video_id"] == "test_stationary")
    assert sealed_row["annotations_sha256"] == hashlib.sha256(sealed_bytes).hexdigest()
    assert annotation.read_bytes() == sealed_bytes
    load_dataset(manifest)


def test_frozen_protocol_rejects_label_and_config_changes(tmp_path: Path) -> None:
    manifest = _copy_dataset(tmp_path)
    config = BenchmarkConfig(generate_figures=False)
    risk_config = RiskConfig()
    labels_protocol = tmp_path / "labels-protocol.json"
    freeze_protocol(manifest, config, risk_config, labels_protocol)
    check_protocol(labels_protocol, manifest, config, risk_config)

    rows = _read_rows(manifest)
    row = next(row for row in rows if row["video_id"] == "test_stationary")
    annotation = _entry_path(manifest, row, "annotations_path")
    raw = json.loads(annotation.read_text(encoding="utf-8"))
    raw["notes"] += " changed after freeze"
    annotation.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _set_hash(row, "annotations_sha256", annotation)
    _write_rows(manifest, rows)
    with pytest.raises(ValueError, match="Frozen test protocol mismatch"):
        check_protocol(labels_protocol, manifest, config, risk_config)

    config_manifest = _copy_dataset(tmp_path / "config-case")
    config_protocol = tmp_path / "config-protocol.json"
    freeze_protocol(config_manifest, config, risk_config, config_protocol)
    changed_config = replace(config, fixed_risk_threshold=0.9)
    with pytest.raises(ValueError, match="benchmark_config"):
        check_protocol(config_protocol, config_manifest, changed_config, risk_config)
    changed_risk_config = replace(
        risk_config,
        scoring=replace(risk_config.scoring, medium_threshold=0.4),
    )
    with pytest.raises(ValueError, match="risk_config"):
        check_protocol(config_protocol, config_manifest, config, changed_risk_config)


def test_runner_requires_protocol_and_writes_deterministic_metrics_and_matches(
    tmp_path: Path,
) -> None:
    manifest = _copy_dataset(tmp_path)
    config = BenchmarkConfig(generate_figures=False)
    risk_config = RiskConfig()

    failed_output = tmp_path / "without-protocol"
    with pytest.raises(ValueError, match="requires a frozen protocol"):
        benchmark(manifest, failed_output, config, risk_config, split="test")
    failure = json.loads((failed_output / "run_metadata.json").read_text(encoding="utf-8"))
    assert failure["status"] == "failed"
    assert failure["failure_stage"] == "protocol_validation"

    protocol = tmp_path / "test-protocol.json"
    freeze_protocol(manifest, config, risk_config, protocol)
    outputs = (tmp_path / "run-one", tmp_path / "run-two")
    for output in outputs:
        benchmark(manifest, output, config, risk_config, split="test", protocol=protocol)

    assert (outputs[0] / "metrics.json").read_bytes() == (outputs[1] / "metrics.json").read_bytes()
    assert (outputs[0] / "matches.jsonl").read_bytes() == (
        outputs[1] / "matches.jsonl"
    ).read_bytes()
