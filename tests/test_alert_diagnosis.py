from sightguard.evaluation.alert_diagnosis import diagnose_video
from sightguard.ui.plots import lifecycle_plot, risk_plot


def test_joins_count_alerted_events_not_all_observed_events() -> None:
    samples = [
        {
            "frame_index": i,
            "timestamp_ms": i * 50,
            "observations": [
                {
                    "event_id": eid,
                    "source_track_id": 1,
                    "class_name": "person",
                    "assessment": {"score": score, "level": "MEDIUM"},
                }
                for eid, score in (("E1", 0.4), ("E2", 0.2))
            ],
        }
        for i in range(2)
    ]
    alert = {
        "event_id": "E1",
        "timestamp_ms": 50,
        "alert_type": "first",
        "alert_id": "A1",
        "risk_score": 0.4,
        "risk_level": "MEDIUM",
    }
    events = [
        {"event_id": "E1", "timestamp_ms": 0, "state": "CANDIDATE"},
        {"event_id": "E1", "timestamp_ms": 50, "state": "ACTIVE"},
    ]
    result = diagnose_video("v", 100, samples, events, [alert])
    assert result["event_count"] == 2
    assert result["alerted_event_count"] == 1
    assert result["alerts_per_minute"] == 600
    event = result["events"][0]
    assert event["right_censored"]
    assert event["alerts"][0]["state_at_alert"] == "ACTIVE"
    assert event["alerts"][0]["observation"]["assessment"]["score"] == 0.4
    assert len(lifecycle_plot([event]).data) == 4
    figure = risk_plot(tuple(samples), (alert,), "E1")
    assert len(figure.data) == 2
    assert figure.data[0].y == (0.4, 0.4)
    assert figure.data[1].x == (0.05,)
    assert figure.data[1].marker.symbol == "circle"


def test_missing_observations_break_curve_and_orphan_alert_is_explicit() -> None:
    frames = tuple(
        {
            "timestamp_ms": t,
            "frame_index": i,
            "observations": []
            if i == 1
            else [{"event_id": "E", "assessment": {"score": 0.2, "level": "LOW"}}],
        }
        for i, t in enumerate((0, 50, 100))
    )
    figure = risk_plot(frames)
    assert figure.data[0].y == (0.2, None, 0.2)
    assert figure.data[0].connectgaps is False
    orphan = {"event_id": "health", "timestamp_ms": 50, "alert_type": "perception_limited"}
    result = diagnose_video("v", 150, list(frames), [], [orphan])
    assert result["alerted_event_count"] == 1
    assert result["events"][0]["alerts"][0]["observation_missing"]
    health_plot = risk_plot((), (orphan,), "health")
    assert health_plot.data[0].y == (1.04,)
    assert "无风险分数" in health_plot.data[0].name


def test_short_health_candidate_is_not_a_risk_event_or_alert() -> None:
    events = [
        {"event_id": "H1", "event_kind": "perception", "timestamp_ms": 10, "state": "CANDIDATE"},
        {"event_id": "H1", "event_kind": "perception", "timestamp_ms": 60, "state": "RESOLVED"},
    ]
    result = diagnose_video("v", 100, [], events, [])
    assert result["risk_event_count"] == 0
    assert result["perception_event_count"] == 1
    assert result["alert_count"] == result["alerted_event_count"] == 0
    event = result["events"][0]
    assert event["end_ms"] - event["start_ms"] == 50
    assert not event["right_censored"]
