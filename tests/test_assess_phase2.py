import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from sightguard.cli import main
from sightguard.data import Detection, FrameRecord
from sightguard.pipeline.assess import assess_tracks
from sightguard.risk.config import RiskConfig, load_risk_config
from sightguard.risk.engine import AssessmentEngine

FIXTURES = Path(__file__).parent / "fixtures" / "phase2"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_risk_configuration() -> None:
    assert load_risk_config(Path("configs/risk.yaml")) == RiskConfig()
    assert load_risk_config() == RiskConfig()


@pytest.mark.parametrize(
    "text",
    [
        "[]",
        "unknown: 1",
        "schema_version: '8'",
        "features: []",
        "features: {motion_min_interval_ms: .nan}",
        "features: {history_window_ms: -1}",
        "features: {focus_center_x: 0.1}",
        "features: {focus_top_y: 1}",
        "scoring: {spatial_weight: 0.8}",
        "scoring: {medium_threshold: 0.8}",
        "scoring: {bottom_start: 1}",
        "scoring: {scale_rate_full_score: 0}",
        "association: {confirmation_observations: 1}",
        "association: {geometry_min_iou: 2}",
        "alerts: {minimum_level: UNKNOWN}",
        "alerts: {repeat_enabled: 1}",
        "alerts: {max_alerts_per_frame: false}",
        "alerts: {ordinary_interval_ms: -1}",
    ],
)
def test_invalid_risk_config(tmp_path: Path, text: str) -> None:
    path = tmp_path / "risk.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        load_risk_config(path)


def test_hand_specified_regression_and_export_determinism(tmp_path: Path) -> None:
    input_path = FIXTURES / "static_occlusion.jsonl"
    before = input_path.read_bytes()
    expected = json.loads((FIXTURES / "static_occlusion.expected.json").read_text(encoding="utf-8"))
    first, second = tmp_path / "first", tmp_path / "second"
    metadata = assess_tracks(input_path, first, RiskConfig())
    assess_tracks(input_path, second, RiskConfig())
    for name in ("risk_frames.jsonl", "events.jsonl", "alerts.jsonl"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
        assert all(row["schema_version"] == "2.0" for row in read_jsonl(first / name))
    frames, changes, alerts = (
        read_jsonl(first / name) for name in ("risk_frames.jsonl", "events.jsonl", "alerts.jsonl")
    )
    assert [r["events"][0]["state"] for r in frames] == expected["frame_states"]
    for row, expected_score in zip(frames, expected["scores"], strict=True):
        if expected_score is None:
            assert row["observations"] == []
        else:
            assert row["observations"][0]["assessment"]["score"] == pytest.approx(expected_score)
    assert [
        [c["timestamp_ms"], c["event_id"], c["action"], c["state"]] for c in changes
    ] == expected["event_changes"]
    assert [
        [
            a["timestamp_ms"],
            a["alert_id"],
            a["event_id"],
            a["alert_type"],
            a["risk_level"],
            a["source_track_ids"],
        ]
        for a in alerts
    ] == expected["alerts"]
    assert frames[-1]["events"][0]["source_track_ids"] == expected["final_source_track_ids"]
    assert metadata["status"] == "completed" and metadata["frames"] == 5
    assert metadata["alerts"] == 1 and metadata["censored_risk_events"] == 1
    assert metadata["config_sha256"] and metadata["source_sha256"]
    assert input_path.read_bytes() == before
    assert (
        main(
            [
                "assess",
                "--tracks",
                str(input_path),
                "--config",
                "configs/risk.yaml",
                "--output-dir",
                str(tmp_path / "cli"),
            ]
        )
        == 0
    )
    with pytest.raises(FileExistsError):
        assess_tracks(input_path, first, RiskConfig())


@pytest.mark.parametrize(
    "text", ["", "not json", '{"schema_version":"1.0"}', '{"frame_index":0,"timestamp_ms":null}']
)
def test_invalid_input_metadata(tmp_path: Path, text: str) -> None:
    path, output = tmp_path / "tracks.jsonl", tmp_path / "out"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        assess_tracks(path, output, RiskConfig())
    metadata = json.loads((output / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "failed" and metadata["error"]


def test_missing_input_cli(tmp_path: Path) -> None:
    assert (
        main(
            [
                "assess",
                "--tracks",
                str(tmp_path / "missing.jsonl"),
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
        == 1
    )


def test_partial_source_failure_and_invalid_bbox_exports(tmp_path: Path) -> None:
    valid = Detection(3, 0, "person", 0.9, (35, 20, 65, 95))
    invalid = Detection(None, 0, "person", 0.9, (3, 3, 3, 5))
    frames = [
        FrameRecord(0, 0, 100, 100, (valid,)),
        FrameRecord(1, 250, 100, 100, (valid,)),
        FrameRecord(2, 500, 100, 100, (invalid,)),
        FrameRecord(3, 750, 100, 100, status="processing_error", error="test failure"),
    ]
    path, output = tmp_path / "tracks.jsonl", tmp_path / "out"
    path.write_text("\n".join(f.to_json() for f in frames) + "\n", encoding="utf-8")
    (tmp_path / "run_metadata.json").write_text('{"status":"failed"}', encoding="utf-8")
    metadata = assess_tracks(path, output, RiskConfig())
    rows = read_jsonl(output / "risk_frames.jsonl")
    assert len(rows) == 4
    assert rows[2]["observations"][0]["error"].startswith("invalid_bbox")
    assert rows[2]["observations"][0]["assessment"] is None
    assert rows[-1]["input_status"] == "processing_error"
    assert rows[-1]["events"][0]["state"] == "ACTIVE"
    assert metadata["input"]["source_run_status"] == "failed"
    assert metadata["source_error_frames"] == 1 and metadata["invalid_detections"] == 1


def test_time_order_and_finished_session() -> None:
    engine = AssessmentEngine()
    engine.step(FrameRecord(0, 100, 100, 100))
    with pytest.raises(ValueError, match="decrease"):
        engine.step(FrameRecord(1, 50, 100, 100))
    with pytest.raises(ValueError, match="contiguous"):
        engine.step(FrameRecord(2, 200, 100, 100))
    assert engine.finish() == ()
    with pytest.raises(ValueError, match="finished"):
        engine.step(FrameRecord(1, 200, 100, 100))


def test_assessment_does_not_import_vision_libraries(tmp_path: Path) -> None:
    code = """
import builtins, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'ultralytics', 'cv2', 'numpy', 'gradio'}:
        raise AssertionError('Forbidden core dependency: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from sightguard.cli import main
raise SystemExit(main(['assess', '--tracks', sys.argv[1], '--output-dir', sys.argv[2]]))
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(FIXTURES / "static_occlusion.jsonl"),
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_scoring_quality_never_scales_risk() -> None:
    config = RiskConfig()
    risk_rows = []
    for confidence in (0.05, 0.99):
        engine = AssessmentEngine(
            replace(config, alerts=replace(config.alerts, limited_quality_threshold=0))
        )
        row = engine.step(
            FrameRecord(0, 0, 100, 100, (Detection(1, 0, "person", confidence, (35, 20, 65, 95)),))
        )
        risk_rows.append(row.frame.observations[0].assessment)
    assert risk_rows[0].score == risk_rows[1].score
    assert risk_rows[0].quality.score < risk_rows[1].quality.score


def test_export_prefix_causality_with_stable_video_identity(tmp_path: Path) -> None:
    lines = (FIXTURES / "static_occlusion.jsonl").read_text(encoding="utf-8").splitlines()
    prefix = tmp_path / "prefix.jsonl"
    full = tmp_path / "full.jsonl"
    prefix.write_text("\n".join(lines[:3]) + "\n", encoding="utf-8")
    full.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assess_tracks(prefix, tmp_path / "short", RiskConfig(), stream_id="same-clip")
    assess_tracks(full, tmp_path / "long", RiskConfig(), stream_id="same-clip")
    for name in ("risk_frames.jsonl", "alerts.jsonl", "events.jsonl"):
        short = [
            row for row in read_jsonl(tmp_path / "short" / name) if row.get("action") != "censored"
        ]
        long = [
            row
            for row in read_jsonl(tmp_path / "long" / name)
            if row["timestamp_ms"] <= 250 and row.get("action") != "censored"
        ]
        assert short == long


def test_threshold_jitter_and_optional_causal_smoothing() -> None:
    import math

    from sightguard.risk.schema import RiskLevel
    from sightguard.risk.scoring import RiskScorer

    engine = AssessmentEngine()
    features = engine.step(
        FrameRecord(0, 0, 100, 100, (Detection(1, 0, "person", 0.9, (35, 20, 65, 95)),))
    )
    base = features.frame.observations[0].features
    scoring = replace(RiskConfig().scoring, spatial_weight=1, appearance_weight=0, motion_weight=0)
    scorer = RiskScorer(scoring)
    series = [
        (0, 0.64, RiskLevel.MEDIUM),
        (100, 0.66, RiskLevel.MEDIUM),
        (200, 0.64, RiskLevel.MEDIUM),
        (300, 0.66, RiskLevel.MEDIUM),
        (400, 0.68, RiskLevel.HIGH),
        (500, 0.64, RiskLevel.HIGH),
        (900, 0.59, RiskLevel.HIGH),
        (1100, 0.58, RiskLevel.HIGH),
        (1200, 0.64, RiskLevel.HIGH),
        (1500, 0.58, RiskLevel.HIGH),
        (1900, 0.57, RiskLevel.MEDIUM),
    ]
    for timestamp, score, expected in series:
        observation = replace(base, spatial=replace(base.spatial, focus_relevance=score))
        assert scorer.assess("test", observation, timestamp).level == expected
    smooth = RiskScorer(replace(scoring, smoothing_tau_ms=1000))
    smooth.assess("test", replace(base, spatial=replace(base.spatial, focus_relevance=0)), 0)
    increased = smooth.assess(
        "test", replace(base, spatial=replace(base.spatial, focus_relevance=1)), 500
    )
    assert increased.raw_score == 1
    assert increased.score == pytest.approx(1 - math.exp(-0.5))
    assert increased.level == RiskLevel.MEDIUM
