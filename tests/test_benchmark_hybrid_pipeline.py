"""CPU-only tests for the standalone hybrid pipeline benchmark helpers."""

from __future__ import annotations

import io
import sys
import threading
from collections import defaultdict, deque
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from benchmark_hybrid_pipeline import (  # noqa: E402
    instrument_assessment_step,
    observe_auxiliary,
)


class FakeDecision:
    def __init__(
        self,
        result_id: str,
        *,
        timestamp_ms: float = 0.0,
        candidate_risk_level: str | None = "HIGH",
        error: str | None = None,
    ) -> None:
        self.result_id = result_id
        self.timestamp_ms = timestamp_ms
        self.payload = {
            "result_id": result_id,
            "timestamp_ms": timestamp_ms,
            "candidate_risk_level": candidate_risk_level,
            "inference_latency_ms": 450.0,
            "valid": True,
            "stale": False,
            "error": error,
        }

    def to_dict(self) -> dict:
        return dict(self.payload)


class FakeManager:
    def __init__(self, decision: FakeDecision, completed_at: float) -> None:
        self.condition = threading.Condition()
        self.latest = decision
        self.latest_completed = completed_at
        self.model = {"device": "cuda", "precision": "fp16"}
        self.auxiliary = "vision"
        self.model_latencies = {"vision": deque([450.0])}
        self.fps_before = deque([9.0])
        self.fps_after = deque([8.0])
        self.calls = 1
        self.timeouts = 0
        self.stale_count = 0
        self.peak_vram_mb = 512.0
        self.status = "Ready"
        self.error = None


def test_auxiliary_candidate_freshness_uses_real_decision_fields() -> None:
    manager = FakeManager(FakeDecision("fresh"), completed_at=100.25)
    session = SimpleNamespace(origin=100.0)
    counts = defaultdict(
        lambda: {
            "completed": 0,
            "fresh": 0,
            "valid": 0,
            "candidate": 0,
            "expired_or_stale": 0,
            "unknown_abstentions": 0,
            "output_errors": 0,
            "errors": 0,
        }
    )
    rows: list[dict] = []
    observed: set[str] = set()

    observe_auxiliary(
        manager,
        session,
        "measurement",
        io.StringIO(),
        observed,
        rows,
        counts,
        stale_after_ms=2000,
    )

    assert rows[0]["completion_age_ms"] == 250.0
    assert rows[0]["fresh_at_completion"] is True
    assert rows[0]["candidate"] is True
    assert counts["measurement"]["candidate"] == 1


def test_expired_candidate_is_not_counted_as_fresh_candidate() -> None:
    manager = FakeManager(FakeDecision("expired"), completed_at=102.5)
    session = SimpleNamespace(origin=100.0)
    counts = defaultdict(
        lambda: {
            "completed": 0,
            "fresh": 0,
            "valid": 0,
            "candidate": 0,
            "expired_or_stale": 0,
            "unknown_abstentions": 0,
            "output_errors": 0,
            "errors": 0,
        }
    )
    rows: list[dict] = []

    observe_auxiliary(
        manager,
        session,
        "measurement",
        io.StringIO(),
        set(),
        rows,
        counts,
        stale_after_ms=2000,
    )

    assert rows[0]["completion_age_ms"] == 2500.0
    assert rows[0]["fresh_at_completion"] is False
    assert rows[0]["expired_or_stale"] is True
    assert rows[0]["candidate"] is False
    assert counts["measurement"]["expired_or_stale"] == 1


def test_unknown_abstention_is_fresh_but_not_an_error_or_candidate() -> None:
    decision = FakeDecision(
        "unknown", candidate_risk_level=None, error="Insufficient evidence (UNKNOWN)"
    )
    manager = FakeManager(decision, completed_at=100.25)
    session = SimpleNamespace(origin=100.0)
    counts = defaultdict(
        lambda: {
            "completed": 0,
            "fresh": 0,
            "valid": 0,
            "candidate": 0,
            "expired_or_stale": 0,
            "unknown_abstentions": 0,
            "output_errors": 0,
            "errors": 0,
        }
    )
    rows: list[dict] = []

    observe_auxiliary(
        manager,
        session,
        "measurement",
        io.StringIO(),
        set(),
        rows,
        counts,
        stale_after_ms=2000,
    )

    assert rows[0]["fresh_at_completion"] is True
    assert rows[0]["unknown_abstention"] is True
    assert rows[0]["candidate"] is False
    assert counts["measurement"]["errors"] == 0
    assert counts["measurement"]["unknown_abstentions"] == 1


def test_assessment_step_timing_preserves_result_and_frame_identity() -> None:
    returned = object()
    record = SimpleNamespace(frame_index=7, timestamp_ms=700.0)
    session = SimpleNamespace(engine=SimpleNamespace(step=lambda _record: returned))
    phase = {"value": "measurement"}
    rows: list[dict] = []
    by_frame: dict[int, dict] = {}

    instrument_assessment_step(session, phase, rows, by_frame, threading.Lock())
    result = session.engine.step(record)

    assert result is returned
    assert by_frame[7]["timestamp_ms"] == 700.0
    assert by_frame[7]["phase"] == "measurement"
    assert by_frame[7]["elapsed_ms"] >= 0
    assert rows == [by_frame[7]]
