from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pytest

import sightguard.evaluation.dataset as dataset_module
import sightguard.evaluation.runner as runner_module
from sightguard.data import FrameRecord
from sightguard.evaluation.config import BenchmarkConfig
from sightguard.evaluation.dataset import MANIFEST_FIELDS, sha256_file
from sightguard.evaluation.protocol import check_protocol, freeze_protocol
from sightguard.evaluation.runner import benchmark
from sightguard.evaluation.schema import (
    AlertOpportunity,
    LabeledEvent,
    StrategyAlert,
    VideoAnnotations,
)
from sightguard.risk.config import RiskConfig
from sightguard.risk.schema import RiskLevel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_DATASET = PROJECT_ROOT / "datasets" / "synthetic"


def _copy_synthetic_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    shutil.copytree(SYNTHETIC_DATASET, root)
    return root / "manifest.csv"


def _strategy_alert(**overrides: object) -> StrategyAlert:
    values: dict[str, object] = {
        "video_id": "video-1",
        "strategy": "A",
        "alert_id": "alert-1",
        "event_id": "model-event-1",
        "timestamp_ms": 100.0,
        "risk_level": RiskLevel.MEDIUM,
        "risk_score": 0.5,
        "alert_type": "first",
        "reason": "test alert",
        "source_track_ids": (1,),
        "class_id": 0,
        "bbox_normalized": None,
    }
    values.update(overrides)
    return StrategyAlert(**values)  # type: ignore[arg-type]


def _real_dataset(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "real-dataset"
    tracks_dir = root / "tracks"
    annotations_dir = root / "annotations"
    tracks_dir.mkdir(parents=True)
    annotations_dir.mkdir()
    video = root / "source.mp4"
    video.write_bytes(b"synthetic container bytes for source binding")
    video_hash = sha256_file(video)
    tracks = tracks_dir / "real-video.jsonl"
    records = (
        FrameRecord(0, 0.0, 640, 480, timestamp_source="video_pts"),
        FrameRecord(1, 250.0, 640, 480, timestamp_source="video_pts"),
    )
    tracks.write_text("".join(f"{record.to_json()}\n" for record in records), encoding="utf-8")
    metadata = tracks_dir / "run_metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "status": "completed",
                "input": {"path": str(video), "sha256": video_hash},
                "config": {"confidence_threshold": 0.25},
                "frames_processed": len(records),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    labels = VideoAnnotations(
        video_id="real-video",
        duration_ms=500.0,
        data_kind="real",
        annotation_status="reviewed",
        annotator_id="independent-reviewer",
        label_source="human_raw_video",
        source_video_sha256=video_hash,
        notes="A small local test clip.",
        events=(),
    )
    annotation = annotations_dir / "real-video.json"
    annotation.write_text(labels.to_json() + "\n", encoding="utf-8")
    manifest = root / "manifest.csv"
    row = {
        "video_id": "real-video",
        "split": "test",
        "source_group_id": "real-sequence-1",
        "scene": "test-scene",
        "data_kind": "real",
        "duration_ms": "500",
        "fps": "4",
        "video_path": "source.mp4",
        "tracks_path": tracks.relative_to(root).as_posix(),
        "annotations_path": annotation.relative_to(root).as_posix(),
        "video_sha256": video_hash,
        "tracks_sha256": sha256_file(tracks),
        "annotations_sha256": sha256_file(annotation),
    }
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)
    return manifest, video, metadata


def test_strategy_alert_rejects_invalid_class_frame_and_source_ids() -> None:
    with pytest.raises(ValueError, match="class_id"):
        _strategy_alert(class_id=-1)
    with pytest.raises(ValueError, match="frame_index"):
        _strategy_alert(frame_index=-1)
    with pytest.raises(ValueError, match="source IDs"):
        _strategy_alert(source_track_ids=(-1,))


def test_alert_opportunity_must_start_before_exclusive_video_end() -> None:
    opportunity = AlertOpportunity(
        opportunity_id="at-video-end",
        kind="initial",
        window_start_ms=1000,
        deadline_ms=1000,
        minimum_severity=RiskLevel.MEDIUM,
    )
    event = LabeledEvent(
        video_id="video-1",
        event_id="event-1",
        event_type="obstacle",
        start_ms=0,
        end_ms=1000,
        severity=RiskLevel.HIGH,
        annotation_status="confirmed",
        alert_opportunities=(opportunity,),
        annotator_id="reviewer",
        notes="boundary test",
        class_ids=(0,),
    )

    with pytest.raises(ValueError, match="start before the exclusive video end"):
        VideoAnnotations(
            video_id="video-1",
            duration_ms=1000,
            data_kind="synthetic",
            annotation_status="reviewed",
            annotator_id="reviewer",
            label_source="synthetic_scenario",
            source_video_sha256="synthetic-hash",
            notes="boundary test",
            events=(event,),
        )


def test_runner_failure_keeps_partial_artifacts_and_identifies_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _copy_synthetic_dataset(tmp_path)
    config = BenchmarkConfig(generate_figures=False)
    risk_config = RiskConfig()
    protocol = tmp_path / "test-protocol.json"
    freeze_protocol(manifest, config, risk_config, protocol)

    def fail_strategy(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("simulated shared strategy failure")

    monkeypatch.setattr(runner_module, "run_strategies", fail_strategy)
    output = tmp_path / "failed-run"
    with pytest.raises(RuntimeError, match="simulated shared strategy failure"):
        benchmark(manifest, output, config, risk_config, split="test", protocol=protocol)

    metadata = json.loads((output / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "failed"
    assert metadata["artifacts_complete"] is False
    assert metadata["failure_video_id"] == "test_escalation"
    assert metadata["failure_strategy"] == "shared_risk_and_policies"
    assert (output / "alerts.jsonl.partial").is_file()
    assert (output / "alerts.jsonl.partial").read_text(encoding="utf-8") == ""
    assert not (output / "alerts.jsonl").exists()
    assert not (output / "risk_frames.jsonl").exists()
    assert not (output / "events.jsonl").exists()


def test_runner_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "existing-output"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("preserve me", encoding="utf-8")
    manifest = _copy_synthetic_dataset(tmp_path)

    with pytest.raises(FileExistsError, match="must be empty"):
        benchmark(
            manifest,
            output,
            BenchmarkConfig(generate_figures=False),
            RiskConfig(),
            split="development",
        )

    assert marker.read_text(encoding="utf-8") == "preserve me"


def test_frozen_protocol_binds_real_perception_metadata_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, _video, metadata = _real_dataset(tmp_path)
    monkeypatch.setattr(dataset_module, "probe_video", lambda _path: (500.0, 4.0))
    assert len(dataset_module.load_dataset(manifest, check_videos=True)) == 1

    config = BenchmarkConfig(generate_figures=False)
    risk_config = RiskConfig()
    protocol_path = tmp_path / "real-protocol.json"
    frozen = freeze_protocol(manifest, config, risk_config, protocol_path)
    original_hash = sha256_file(metadata)
    assert frozen["inputs"]["real-video"]["perception_metadata_sha256"] == original_hash
    check_protocol(protocol_path, manifest, config, risk_config)

    source = json.loads(metadata.read_text(encoding="utf-8"))
    source["config"]["confidence_threshold"] = 0.3
    metadata.write_text(json.dumps(source, sort_keys=True), encoding="utf-8")
    assert sha256_file(metadata) != original_hash

    with pytest.raises(ValueError, match="Frozen test protocol mismatch: inputs"):
        check_protocol(protocol_path, manifest, config, risk_config)
