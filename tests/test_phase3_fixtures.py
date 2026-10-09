import csv
import hashlib
import subprocess
import sys
from pathlib import Path

from sightguard.evaluation.config import BenchmarkConfig
from sightguard.evaluation.dataset import load_dataset, read_manifest
from sightguard.evaluation.matching import evaluate_video
from sightguard.evaluation.metrics import aggregate_metrics
from sightguard.evaluation.strategies import run_strategies
from sightguard.risk.config import RiskConfig
from sightguard.risk.schema import RiskLevel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = PROJECT_ROOT / "datasets" / "synthetic"
MANIFEST = FIXTURE_ROOT / "manifest.csv"
GENERATOR = PROJECT_ROOT / "scripts" / "generate_phase3_fixtures.py"


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_generator_recreates_the_committed_fixture_tree_deterministically(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    for destination in (first, second):
        subprocess.run(
            [sys.executable, str(GENERATOR), "--output-dir", str(destination)],
            check=True,
            capture_output=True,
            text=True,
        )

    first_tree, second_tree = _tree_bytes(first), _tree_bytes(second)
    assert first_tree == second_tree
    assert first_tree == _tree_bytes(FIXTURE_ROOT)


def test_manifest_loads_five_reviewed_synthetic_scenarios_with_separate_splits() -> None:
    entries = read_manifest(MANIFEST)
    videos = load_dataset(MANIFEST)
    by_id = {video.entry.video_id: video for video in videos}

    assert len(entries) == len(videos) == 5
    assert {entry.video_id for entry in entries if entry.split == "development"} == {
        "dev_stationary"
    }
    assert {entry.video_id for entry in entries if entry.split == "test"} == {
        "test_stationary",
        "test_escalation",
        "test_missed",
        "test_nonrisk",
    }
    assert len({entry.source_group_id for entry in entries}) == len(entries)
    assert all(entry.data_kind == "synthetic" and entry.video_path is None for entry in entries)
    assert all(video.labels.annotation_status == "reviewed" for video in videos)
    assert all(video.labels.label_source == "synthetic_scenario" for video in videos)
    assert all(video.labels.data_kind == "synthetic" for video in videos)
    assert all(video.labels.annotator_id != "UNASSIGNED" for video in videos)

    expected = {
        "dev_stationary": (2000, 8),
        "test_stationary": (4000, 16),
        "test_escalation": (2000, 8),
        "test_missed": (4000, 16),
        "test_nonrisk": (3000, 12),
    }
    for video_id, (duration_ms, frame_count) in expected.items():
        video = by_id[video_id]
        assert video.entry.duration_ms == duration_ms
        assert video.entry.fps == 4
        assert len(video.records) == frame_count
        assert [frame.frame_index for frame in video.records] == list(range(frame_count))
        assert [frame.timestamp_ms for frame in video.records] == [
            frame_index * 250 for frame_index in range(frame_count)
        ]
        assert video.records[-1].timestamp_ms + 250 >= duration_ms
        assert video.labels.source_video_sha256 == video.entry.video_sha256

    assert by_id["dev_stationary"].records[0].detections[0].class_id == 1
    assert len(by_id["dev_stationary"].labels.events) == 1


def test_scenario_tracks_and_labels_match_predeclared_fixture_definitions() -> None:
    videos = {video.entry.video_id: video for video in load_dataset(MANIFEST)}

    stationary = videos["test_stationary"]
    assert all(len(frame.detections) == 1 for frame in stationary.records)
    assert all(frame.detections[0].class_id == 0 for frame in stationary.records)
    assert stationary.records[0].detections[0].bbox_xyxy == (224, 96, 416, 456)
    stationary_event = stationary.labels.events[0]
    assert stationary_event.severity == RiskLevel.HIGH
    assert [
        (op.kind, op.window_start_ms, op.deadline_ms, op.minimum_severity)
        for op in stationary_event.alert_opportunities
    ] == [("initial", 0, 750, RiskLevel.MEDIUM)]

    escalation = videos["test_escalation"]
    expected_sizes = (
        (20, 40),
        (20, 40),
        (30, 60),
        (45, 90),
        (68, 136),
        (102, 204),
        (102, 204),
        (102, 204),
    )
    actual_sizes = tuple(
        (
            frame.detections[0].bbox_xyxy[2] - frame.detections[0].bbox_xyxy[0],
            frame.detections[0].bbox_xyxy[3] - frame.detections[0].bbox_xyxy[1],
        )
        for frame in escalation.records
    )
    assert actual_sizes == expected_sizes
    escalation_event = escalation.labels.events[0]
    assert escalation_event.severity == RiskLevel.HIGH
    assert [
        (op.kind, op.window_start_ms, op.deadline_ms, op.minimum_severity)
        for op in escalation_event.alert_opportunities
    ] == [
        ("initial", 0, 750, RiskLevel.MEDIUM),
        ("escalation", 1000, 1500, RiskLevel.HIGH),
    ]
    assert len(escalation_event.target_regions) == len(escalation.records)

    missed = videos["test_missed"]
    assert all(frame.detections == () for frame in missed.records)
    assert missed.labels.events[0].severity == RiskLevel.HIGH
    assert missed.labels.events[0].alert_opportunities[0].minimum_severity == RiskLevel.MEDIUM

    nonrisk = videos["test_nonrisk"]
    assert all(len(frame.detections) == 1 for frame in nonrisk.records)
    assert nonrisk.labels.events == ()


def test_synthetic_test_split_has_the_predeclared_strategy_count_fixture() -> None:
    benchmark_config = BenchmarkConfig(generate_figures=False)
    risk_config = RiskConfig()
    per_strategy = {name: [] for name in ("A", "B", "C")}

    for video in load_dataset(MANIFEST):
        if video.entry.split != "test":
            continue
        run = run_strategies(
            video.records,
            video.entry.video_id,
            risk_config,
            benchmark_config,
        )
        for strategy in per_strategy:
            per_strategy[strategy].append(
                evaluate_video(
                    video.labels,
                    run.alerts[strategy],
                    benchmark_config.matching,
                    strategy=strategy,
                )
            )

    expected = {
        "A": (2, 34, 2, 22, 12, 36),
        "B": (2, 2, 2, 1, 1, 4),
        "C": (3, 1, 1, 0, 1, 4),
    }
    for strategy, results in per_strategy.items():
        metrics = aggregate_metrics(results)
        observed = tuple(
            metrics[key] for key in ("tp", "fp", "fn", "duplicates", "unsupported", "total_alerts")
        )
        assert observed == expected[strategy]
        assert metrics["tp"] + metrics["fn"] == metrics["opportunities"]
        assert metrics["fp"] == metrics["duplicates"] + metrics["unsupported"]


def test_manifest_hashes_match_generated_tracks_and_annotation_files() -> None:
    with MANIFEST.open(encoding="utf-8", newline="") as stream:
        rows = tuple(csv.DictReader(stream))

    assert len(rows) == 5
    for row in rows:
        tracks_path = MANIFEST.parent / row["tracks_path"]
        annotations_path = MANIFEST.parent / row["annotations_path"]
        assert hashlib.sha256(tracks_path.read_bytes()).hexdigest() == row["tracks_sha256"]
        assert (
            hashlib.sha256(annotations_path.read_bytes()).hexdigest() == row["annotations_sha256"]
        )
