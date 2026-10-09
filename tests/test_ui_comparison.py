from dataclasses import replace

from sightguard.evaluation.schema import AlertOpportunity, LabeledEvent, VideoAnnotations
from sightguard.risk.schema import RiskLevel
from sightguard.ui.app import _comparison, _event_choices, _metrics_table, _scored_metrics
from sightguard.ui.services import BenchmarkView


def _view() -> BenchmarkView:
    event = LabeledEvent(
        "v",
        "G",
        "obstacle",
        0,
        2000,
        RiskLevel.HIGH,
        "confirmed",
        (
            AlertOpportunity("initial", "initial", 0, 500, RiskLevel.MEDIUM),
            AlertOpportunity("upgrade", "escalation", 1000, 1500, RiskLevel.HIGH),
        ),
        "independent-author",
        "scripted unit fixture",
        (0,),
    )
    labels = VideoAnnotations(
        "v",
        2000,
        "synthetic",
        "reviewed",
        "independent-author",
        "synthetic_scenario",
        "0" * 64,
        "unit fixture",
        (event,),
    )
    # B and C intentionally reuse the same model event ID, which is not a GT identity.
    alerts = tuple(
        {
            "video_id": "v",
            "strategy": strategy,
            "alert_id": alert,
            "event_id": "E",
            "timestamp_ms": timestamp,
            "risk_level": severity,
            "risk_score": 0.8,
            "alert_type": kind,
            "reason": "test",
            "class_id": 0,
            "source_track_ids": [1],
            "frame_index": 0,
        }
        for strategy, alert, timestamp, severity, kind in (
            ("B", "B1", 0, "MEDIUM", "trigger"),
            ("C", "C1", 250, "MEDIUM", "first"),
            ("C", "C2", 1250, "HIGH", "escalation"),
            ("C", "C3", 1750, "HIGH", "repeat"),
        )
    )
    matches = tuple(
        {
            "video_id": "v",
            "strategy": row["strategy"],
            "alert_id": row["alert_id"],
            "timestamp_ms": row["timestamp_ms"],
            "gt_event_id": "G",
            "opportunity_id": "initial" if row["timestamp_ms"] < 1000 else "upgrade",
            "status": "duplicate" if row["alert_id"] == "C3" else "matched",
            "reason": "test",
        }
        for row in alerts
    )
    return BenchmarkView(
        "unit-run",
        {"inputs": [{"video_id": "v"}]},
        {"C": {"tp": 2, "fp": 1, "fn": 0, "event_recall": 1, "first_response_delay_mean_ms": None}},
        alerts,
        matches,
        {},
        {"v": labels},
        (),
    )


def test_model_selector_does_not_merge_same_ids_from_different_strategies() -> None:
    view = _view()
    _, rows, matches = _comparison(view, "v", ["B", "C"], "model|C|E")
    assert rows["strategy"].tolist() == ["C", "C", "C"]
    assert rows["alert_id"].tolist() == ["C1", "C2", "C3"]
    assert set(matches["strategy"]) == {"C"}
    choices = dict(_event_choices(view, "v"))
    assert "model|B|E" in choices.values()
    assert "model|C|E" in choices.values()


def test_gt_selector_displays_necessary_upgrade_and_duplicate_separately() -> None:
    _, rows, matches = _comparison(_view(), "v", ["B", "C"], "gt|G")
    assert len(rows) == 4
    assert rows["match_status"].tolist() == ["matched", "matched", "matched", "duplicate"]
    assert len(matches) == 4


def test_missing_verified_gt_does_not_show_persisted_match_classifications() -> None:
    view = replace(_view(), labels={})
    _, rows, matches = _comparison(view, "v", ["B", "C"])
    assert rows["match_status"].tolist() == ["unscored"] * 4
    assert matches.empty
    assert view.metrics["C"]["tp"] == 2  # Historical source is preserved, not rewritten.


def test_accuracy_requires_labels_for_inputs_with_no_detections_too() -> None:
    view = replace(_view(), metadata={"inputs": [{"video_id": "v"}, {"video_id": "missed"}]})
    assert not _scored_metrics(view)
    assert view.metrics["C"]["tp"] == 2


def test_accuracy_is_hidden_without_gt_and_preserved_when_verified() -> None:
    view = _view()
    assert _scored_metrics(view) == view.metrics
    assert _scored_metrics(replace(view, labels={})) == {}


def test_displayed_counts_and_nulls_preserve_metric_source() -> None:
    view = _view()
    table = _metrics_table(view)
    assert table.iloc[0]["tp"] == 2
    assert table.iloc[0]["fp"] == 1
    assert table.iloc[0]["fn"] == 0
    assert table.iloc[0]["first_response_delay_mean_ms"] == "N/A"
    assert view.metrics["C"]["first_response_delay_mean_ms"] is None
