"""Report gates must retain misses, false alarms and absent audio as separate failures."""

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def report_module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "android/tools"))
    return importlib.import_module("summarize_furniture_validation")


def fixture_report(groups, count=4):
    return {
        "device": "synthetic",
        "sdk": 36,
        "model_sha256": "fixture",
        "evidence_level": "unit only",
        "metric": "synthetic pixels",
        "model_load_ms": 0,
        "cases": [
            {
                "case": {"image": str(i), "gt_fractions": {g: 0.1 for g in groups}},
                "pixels": {g: {"tp": 8, "fp": 1, "fn": 1, "tn": 10} for g in groups},
                "observations": [
                    {
                        "events": [f"semantic:1:0:{g}:1" for g in groups],
                        "latency_ms": 100,
                        "source_age_ms": 110,
                        "measure_ms": 1,
                        "pss_kb": 100,
                    }
                ],
            }
            for i in range(count)
        ],
        "audio": [{"phase": "onStart", "event": f"semantic:1:0:{g}:1"} for g in groups],
    }


def test_gate_requires_pixels_temporal_events_audio_and_sample_count(report_module):
    report = fixture_report(report_module.GROUPS)
    assert all(g["pilot_gate_pass"] for g in report_module.summarize(report)["groups"].values())
    report["audio"] = []
    result = report_module.summarize(report)["groups"]["TABLE"]
    assert not result["pilot_gate_pass"]
    assert "no_real_model_event_audio_onStart" in result["rejection_reasons"]
    report["cases"] = report["cases"][:3]
    assert (
        "fewer_than_four_gt_positive_images"
        in report_module.summarize(report)["groups"]["TABLE"]["rejection_reasons"]
    )


def test_false_events_and_misses_are_retained_and_groups_do_not_leak(report_module):
    report = fixture_report(report_module.GROUPS)
    report["cases"][0]["case"]["gt_fractions"]["TABLE"] = 0
    report["cases"][1]["observations"][0]["events"] = ["semantic:1:0:CHAIR:1"]
    result = report_module.summarize(report)["groups"]
    assert result["TABLE"]["false_event_images"] == ["0"]
    assert result["TABLE"]["missed_positive_images"] == ["1"]
    assert not result["TABLE"]["pilot_gate_pass"]
    assert result["CHAIR"]["pilot_gate_pass"]


def test_empty_report_cannot_pass(report_module):
    report = fixture_report(report_module.GROUPS, count=0)
    result = report_module.summarize(report)
    assert result["fresh_observation_fraction"] is None
    assert all(not g["pilot_gate_pass"] for g in result["groups"].values())
