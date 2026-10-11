from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from threading import Event, Thread
from time import monotonic

import cv2
import numpy as np
import pytest

from sightguard.config import PerceptionConfig
from sightguard.data import Detection, FrameRecord
from sightguard.hybrid.config import HybridConfig, ModelConfig, load_hybrid_config
from sightguard.hybrid.input import build_input
from sightguard.hybrid.manager import IntelligenceManager
from sightguard.hybrid.schema import DecisionResult, fuse, parse_output
from sightguard.live.config import LiveConfig
from sightguard.live.runtime import CapturedFrame, LiveResult, LiveSession
from sightguard.risk.config import RiskConfig
from sightguard.risk.engine import AssessmentEngine


class MockDecisionProvider:
    """Bounded provider fake with explicit gates for cancellation and late results."""

    def __init__(
        self,
        source: str,
        *,
        level: str = "HIGH",
        levels: list[str] | None = None,
        load_error: Exception | None = None,
        infer_error: Exception | None = None,
        infer_gate: Event | None = None,
        wait_for_cancel: bool = False,
    ) -> None:
        self.source = source
        self.level = level
        self.levels = levels
        self.load_error = load_error
        self.infer_error = infer_error
        self.infer_gate = infer_gate
        self.wait_for_cancel = wait_for_cancel
        self.load_started = Event()
        self.infer_started = Event()
        self.unloaded = Event()
        self.load_calls = 0
        self.infer_calls = 0
        self.unload_calls = 0
        self.states: list[dict] = []
        self.images: list[np.ndarray | None] = []

    def load(self, cancel: Event) -> dict:
        self.load_calls += 1
        self.load_started.set()
        if cancel.is_set():
            raise InterruptedError("mock load cancelled")
        if self.load_error is not None:
            raise self.load_error
        return {
            "model_id": f"mock/{self.source}",
            "revision": "fixture-r1",
            "device": "cpu",
            "precision": "fp32",
            "format": "mock",
            "load_ms": 1.0,
        }

    def infer(self, state: dict, image: np.ndarray | None, cancel: Event) -> dict:
        self.infer_calls += 1
        self.states.append(state)
        self.images.append(None if image is None else image.copy())
        self.infer_started.set()
        if self.infer_gate is not None:
            # No fake can block a test indefinitely. A cancelled old-generation call can
            # deliberately finish late so the manager's generation guard is exercised.
            if not self.infer_gate.wait(3.0):
                raise TimeoutError("mock inference gate deadline")
        if self.wait_for_cancel and cancel.is_set():
            raise InterruptedError("mock inference cancelled")
        if self.infer_error is not None:
            raise self.infer_error
        level = self.level
        if self.levels is not None:
            level = self.levels[min(self.infer_calls - 1, len(self.levels) - 1)]
        return {
            "raw_output": {"answers": {"risk": {"choice": level, "confidence": 0.73}}},
            "peak_vram_mb": 0.0,
        }

    def unload(self) -> None:
        self.unload_calls += 1
        self.unloaded.set()


def _config(
    *,
    output_dir: str = "outputs/hybrid",
    stale_after_ms: float = 10_000,
    retain_images: bool = False,
) -> HybridConfig:
    return HybridConfig(
        models={
            "laya": ModelConfig(
                python="mock-python",
                path="mock-laya",
                model_id="fixture/laya",
                revision="laya-r1",
            ),
            "vision": ModelConfig(
                python="mock-python",
                path="mock-vision",
                model_id="fixture/vision",
                revision="vision-r1",
            ),
        },
        frequency_hz=3.0,
        timeout_s=0.1,
        load_timeout_s=0.1,
        stale_after_ms=stale_after_ms,
        output_dir=output_dir,
        retain_images=retain_images,
    )


def _wait_for(predicate, timeout: float = 2.5):
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        value = predicate()
        if value:
            return value
        Event().wait(0.01)
    raise AssertionError("condition was not met before the deadline")


def _result_builder(stream_id: str = "hybrid-test"):
    engine = AssessmentEngine(RiskConfig(), stream_id=stream_id)

    def build(
        frame_index: int,
        *,
        detection: bool = True,
        captured_at: float | None = None,
        timestamp_ms: float | None = None,
        bbox: tuple[int, int, int, int] | None = None,
    ) -> LiveResult:
        timestamp = frame_index * 250.0 if timestamp_ms is None else timestamp_ms
        while engine.last_index < frame_index - 1:
            skipped = engine.last_index + 1
            engine.step(
                FrameRecord(
                    skipped,
                    skipped * 250.0,
                    640,
                    480,
                    (),
                    timestamp_source="monotonic",
                )
            )
        detections = (
            (Detection(7, 0, "person", 0.9, bbox or (280, 300, 360, 450)),) if detection else ()
        )
        record = FrameRecord(
            frame_index,
            timestamp,
            640,
            480,
            detections,
            timestamp_source="monotonic",
        )
        decision = engine.step(record)
        image = np.full((480, 640, 3), frame_index % 255, dtype=np.uint8)
        captured = CapturedFrame(
            frame_index,
            timestamp,
            monotonic() if captured_at is None else captured_at,
            image,
        )
        return LiveResult(captured, record, decision, 12.0, monotonic(), 8.0)

    return engine, build


def _manager(
    root: Path,
    providers: list[MockDecisionProvider] | None = None,
    *,
    provider_factory=None,
    config: HybridConfig | None = None,
    session_id: str = "session-a",
) -> IntelligenceManager:
    providers = providers if providers is not None else []

    def create(source: str) -> MockDecisionProvider:
        provider = (
            provider_factory(source, len(providers))
            if provider_factory is not None
            else MockDecisionProvider(source)
        )
        providers.append(provider)
        return provider

    return IntelligenceManager(config or _config(), root, session_id, provider_factory=create)


def _wait_status(manager: IntelligenceManager, status: str) -> dict:
    def matching_snapshot():
        value = manager.snapshot()
        return value if value["model_status"] == status else None

    return _wait_for(matching_snapshot)


def test_hybrid_config_resolves_local_paths_and_serializes(tmp_path: Path) -> None:
    config_path = tmp_path / "configs" / "hybrid.yaml"
    config_path.parent.mkdir()
    config_path.write_text(
        """schema_version: '1.0'
frequency_hz: 2
models:
  laya:
    python: models/hybrid/structured/python.exe
    path: models/hybrid/structured
    model_id: fixture/laya
    revision: rev-1
  vision:
    python: models/hybrid/vision/python.exe
    path: models/hybrid/vision
    model_id: fixture/vision
    revision: rev-2
""",
        encoding="utf-8",
    )

    config = load_hybrid_config(config_path, tmp_path)

    assert config.models["laya"].python == str(
        (tmp_path / "models/hybrid/structured/python.exe").resolve()
    )
    assert config.models["vision"].path == str((tmp_path / "models/hybrid/vision").resolve())
    serialized = json.loads(json.dumps(config.to_dict()))
    assert serialized["schema_version"] == "1.0"
    assert serialized["models"]["laya"]["revision"] == "rev-1"


def test_hybrid_config_rejects_invalid_timing_and_model_contract() -> None:
    with pytest.raises(ValueError, match="frequency_hz"):
        HybridConfig(models=_config().models, frequency_hz=4)
    with pytest.raises(ValueError, match="requires schema"):
        HybridConfig(models={"laya": _config().models["laya"]})
    with pytest.raises(ValueError, match="ONNX baseline"):
        ModelConfig(
            python="python",
            path="model",
            model_id="fixture",
            revision="rev",
            backend="onnx",
        )


def test_decision_contract_round_trip_and_output_parsing() -> None:
    raw = {"answers": {"risk": {"choice": "MEDIUM", "confidence": 0.61}}}
    level, score, error = parse_output(raw, "laya")
    assert (level, score, error) == ("MEDIUM", 0.61, None)
    vision_level, _, vision_error = parse_output(
        {
            "answers": {
                "risk": {
                    "choice": (
                        "Strong concern deserving heightened attention; research category only."
                    )
                }
            }
        },
        "vision",
    )
    assert vision_level == "HIGH" and vision_error is None
    assert parse_output({"answers": {"risk": {"choice": "UNKNOWN"}}}, "laya")[0] is None
    assert parse_output({"answers": {"risk": {"choice": "MAYBE"}}}, "laya")[2]
    assert parse_output({"answers": {"risk": {"choice": "LOW", "confidence": True}}}, "laya")[2]
    assert fuse("LOW", "HIGH", "conservative_max") == "HIGH"
    assert fuse("LOW", "HIGH", "consensus") is None

    value = DecisionResult(
        "r1",
        "s1",
        "laya",
        "fixture/laya",
        "rev1",
        4,
        1000.0,
        "E1",
        raw,
        "MEDIUM",
        0.61,
        "uncalibrated upstream choice",
        12.3,
        "started",
        "completed",
    )
    restored = DecisionResult.from_json(value.to_json())
    assert asdict(restored) == asdict(value)
    with pytest.raises(ValueError, match="Unsupported decision schema"):
        DecisionResult.from_json('{"schema_version":"2.0"}')


def test_semantic_input_uses_features_without_rule_label_or_final_score() -> None:
    _, build = _result_builder()
    result = build(0)

    state, event_id = build_input(result, max_objects=4)

    assert event_id is not None
    assert state["objects"]
    assert state["objects"][0]["event_id"] == event_id
    assert "assessment" not in json.dumps(state).lower()
    assert "rule_level" not in json.dumps(state).lower()
    assert "candidate_risk_level" not in json.dumps(state).lower()
    assert "image" not in state


@pytest.mark.parametrize(
    ("mode", "auxiliary", "expected_source"),
    [
        ("rule", "laya", "laya"),
        ("laya", "vision", "laya"),
        ("vision", "laya", "vision"),
        ("hybrid", "laya", "laya"),
        ("hybrid", "vision", "vision"),
    ],
)
def test_four_modes_select_only_the_selected_auxiliary(
    tmp_path: Path, mode: str, auxiliary: str, expected_source: str
) -> None:
    providers: list[MockDecisionProvider] = []
    manager = _manager(tmp_path, providers)
    try:
        if mode == "rule":
            manager.select(mode, auxiliary, load=False)
            snapshot = _wait_status(manager, "Unavailable")
            assert snapshot["mode"] == "rule"
            assert providers == []
            return
        manager.select(mode, auxiliary, load=True)
        snapshot = _wait_status(manager, "Ready")
        assert snapshot["mode"] == mode
        assert snapshot["auxiliary"] == expected_source
        assert [provider.source for provider in providers] == [expected_source]
    finally:
        manager.close()


def test_vision_gets_original_frame_and_laya_gets_structured_input_only(tmp_path: Path) -> None:
    providers: list[MockDecisionProvider] = []
    manager = _manager(
        tmp_path,
        providers,
        config=_config(retain_images=True),
    )
    _, build = _result_builder()
    try:
        vision_result = build(1)
        manager.select("vision", load=True)
        _wait_status(manager, "Ready")
        manager.submit(vision_result)
        _wait_for(lambda: manager.snapshot()["latest_result"])
        vision_provider = providers[-1]
        assert len(vision_provider.images) == 1
        assert np.array_equal(vision_provider.images[0], vision_result.captured.image)
        retained = cv2.imread(str(manager.output / "frame-1.png"))
        assert np.array_equal(retained, vision_result.captured.image)

        manager.select("laya", load=True)
        _wait_status(manager, "Ready")
        laya_result = build(2)
        manager.submit(laya_result)
        _wait_for(lambda: manager.snapshot()["latest_result"])
        assert providers[-1].source == "laya"
        assert providers[-1].images == [None]
        assert "assessment" not in json.dumps(providers[-1].states[0]).lower()
    finally:
        manager.close()


def test_generation_switch_discards_a_late_result_and_loads_new_source(tmp_path: Path) -> None:
    old_gate = Event()
    created: list[MockDecisionProvider] = []

    def factory(source: str, index: int) -> MockDecisionProvider:
        provider = MockDecisionProvider(
            source,
            level="HIGH" if index == 0 else "MEDIUM",
            infer_gate=old_gate if index == 0 else None,
        )
        created.append(provider)
        return provider

    manager = _manager(tmp_path, provider_factory=factory)
    _, build = _result_builder()
    try:
        manager.select("laya", load=True)
        _wait_status(manager, "Ready")
        assert _wait_for(lambda: len(created) == 1)
        manager.submit(build(1))
        assert _wait_for(lambda: created[0].infer_started.is_set())
        manager.select("vision", auxiliary="vision", load=True)
        manager.submit(build(2))
        old_gate.set()
        snapshot = _wait_for(
            lambda: (
                value
                if (value := manager.snapshot()).get("latest_result")
                and value["latest_result"]["source"] == "vision"
                else None
            )
        )
        assert snapshot["latest_result"]["candidate_risk_level"] == "MEDIUM"
        assert snapshot["mode"] == "vision"
        assert len(created) == 2
        assert created[0].unloaded.is_set()
    finally:
        old_gate.set()
        manager.close()


def test_result_is_invalidated_when_its_event_disappears_during_inference(tmp_path: Path) -> None:
    gate = Event()
    providers: list[MockDecisionProvider] = []
    manager = _manager(
        tmp_path,
        providers,
        provider_factory=lambda source, _index: MockDecisionProvider(source, infer_gate=gate),
    )
    engine, build = _result_builder()
    try:
        manager.select("laya", load=True)
        _wait_status(manager, "Ready")
        observed = build(1, timestamp_ms=100)
        manager.submit(observed)
        assert _wait_for(lambda: providers[0].infer_started.is_set())
        disappeared = build(2, detection=False, timestamp_ms=200)
        manager.submit(disappeared)
        gate.set()
        latest = _wait_for(lambda: manager.snapshot()["latest_result"])
        assert latest["event_id"] == providers[0].states[0]["objects"][0]["event_id"]
        assert latest["stale"] is True
        assert latest["valid"] is False
        assert manager.snapshot()["performance"]["stale"] >= 1
    finally:
        gate.set()
        manager.close()
        engine.finish()


def test_unknown_candidate_does_not_unload_provider_and_later_inference_recovers(
    tmp_path: Path,
) -> None:
    providers: list[MockDecisionProvider] = []
    manager = _manager(
        tmp_path,
        providers,
        provider_factory=lambda source, _index: MockDecisionProvider(
            source, levels=["UNKNOWN", "HIGH"]
        ),
    )
    _, build = _result_builder()
    try:
        manager.select("laya", load=True)
        _wait_status(manager, "Ready")

        manager.submit(build(1))
        unknown = _wait_for(
            lambda: (
                value
                if (value := manager.snapshot()).get("latest_result")
                and value["latest_result"]["frame_id"] == 1
                else None
            )
        )
        assert unknown["latest_result"]["valid"] is False
        assert unknown["model_status"] == "Ready"
        assert providers[0].unload_calls == 0

        manager.submit(build(2))
        recovered = _wait_for(
            lambda: (
                value
                if (value := manager.snapshot()).get("latest_result")
                and value["latest_result"]["frame_id"] == 2
                and value["latest_result"]["valid"]
                else None
            )
        )
        assert recovered["latest_result"]["candidate_risk_level"] == "HIGH"
        assert recovered["model_status"] == "Ready"
        assert providers[0].infer_calls == 2
        assert providers[0].unload_calls == 0
    finally:
        manager.close()


def test_unload_invalidates_in_flight_result_and_releases_provider(tmp_path: Path) -> None:
    gate = Event()
    providers: list[MockDecisionProvider] = []
    manager = _manager(
        tmp_path,
        providers,
        provider_factory=lambda source, _index: MockDecisionProvider(source, infer_gate=gate),
    )
    _, build = _result_builder()
    try:
        manager.select("laya", load=True)
        _wait_status(manager, "Ready")
        manager.submit(build(1))
        assert _wait_for(lambda: providers[0].infer_started.is_set())

        manager.unload()
        gate.set()
        _wait_for(lambda: providers[0].unloaded.is_set())
        snapshot = _wait_status(manager, "Unavailable")

        assert snapshot["latest_result"] is None
        assert providers[0].unload_calls == 1
        assert manager.thread.is_alive()
    finally:
        gate.set()
        manager.close()


@pytest.mark.parametrize(
    ("failure", "expected_timeout"),
    [
        (TimeoutError("mock timeout"), 1),
        (RuntimeError("mock failure"), 0),
        (MemoryError("mock OOM"), 0),
    ],
)
def test_timeout_exception_and_oom_fail_shadow_worker_without_blocking(
    tmp_path: Path, failure: Exception, expected_timeout: int
) -> None:
    providers: list[MockDecisionProvider] = []
    manager = _manager(
        tmp_path,
        providers,
        provider_factory=lambda source, _index: MockDecisionProvider(source, infer_error=failure),
    )
    _, build = _result_builder()
    try:
        manager.select("laya", load=True)
        _wait_status(manager, "Ready")
        started = monotonic()
        manager.submit(build(1))
        assert monotonic() - started < 0.1
        latest = _wait_for(
            lambda: (
                value
                if (value := manager.snapshot()).get("latest_result")
                and value["latest_result"].get("error")
                else None
            )
        )
        assert failure.__class__.__name__ in latest["latest_result"]["error"]
        assert latest["performance"]["timeouts"] == expected_timeout
        assert manager.thread.is_alive()
        assert providers[0].unloaded.is_set()
    finally:
        manager.close()


def test_load_oom_is_reported_as_failed_and_provider_is_released(tmp_path: Path) -> None:
    providers: list[MockDecisionProvider] = []
    manager = _manager(
        tmp_path,
        providers,
        provider_factory=lambda source, _index: MockDecisionProvider(
            source, load_error=MemoryError("mock load OOM")
        ),
    )
    try:
        manager.select("vision", load=True)
        snapshot = _wait_status(manager, "Failed")
        assert "mock load OOM" in snapshot["error"]
        assert providers[0].unloaded.is_set()
        assert manager.thread.is_alive()
    finally:
        manager.close()


def test_sessions_write_separate_logs_with_reproducibility_metadata(tmp_path: Path) -> None:
    first_providers: list[MockDecisionProvider] = []
    second_providers: list[MockDecisionProvider] = []
    first = _manager(tmp_path, first_providers, session_id="session-one")
    second = _manager(tmp_path, second_providers, session_id="session-two")
    _, build_first = _result_builder("session-one")
    _, build_second = _result_builder("session-two")
    try:
        first.select("laya", load=True)
        second.select("vision", load=True)
        _wait_status(first, "Ready")
        _wait_status(second, "Ready")
        first.submit(build_first(1))
        second.submit(build_second(2))
        first_result = _wait_for(lambda: first.snapshot()["latest_result"])
        second_result = _wait_for(lambda: second.snapshot()["latest_result"])
        assert first_result["session_id"] == "session-one"
        assert second_result["session_id"] == "session-two"
        assert first.output != second.output
        assert first.snapshot()["auxiliary"] == "laya"
        assert second.snapshot()["auxiliary"] == "vision"
        first_dir, second_dir = first.output, second.output
    finally:
        first.close()
        second.close()

    for output, session_id in ((first_dir, "session-one"), (second_dir, "session-two")):
        metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["session_id"] == session_id
        assert metadata["actual_alert_controller"] == "Rule Engine"
        assert metadata["shadow_only"] is True
        assert metadata["config_sha256"]
        assert (output / "rule_decisions.jsonl").is_file()
        assert (output / "comparisons.jsonl").is_file()
        assert (output / "performance.json").is_file()


def test_live_rule_alerts_continue_while_shadow_inference_is_blocked(tmp_path: Path) -> None:
    gate = Event()
    providers: list[MockDecisionProvider] = []
    manager = _manager(
        tmp_path,
        providers,
        provider_factory=lambda source, _index: MockDecisionProvider(source, infer_gate=gate),
    )
    manager.select("laya", load=True)
    _wait_status(manager, "Ready")
    sizes = ((20, 40), (20, 40), (30, 60), (45, 90), (68, 136), (102, 204))

    class SizePerceptor:
        def process(self, image: np.ndarray) -> tuple[Detection, ...]:
            index = int(image[0, 0, 0])
            width, height = sizes[index]
            return (
                Detection(
                    7,
                    0,
                    "person",
                    0.9,
                    (320 - width // 2, 336 - height, 320 + width // 2, 336),
                ),
            )

    session = LiveSession(
        PerceptionConfig(),
        RiskConfig(),
        LiveConfig(),
        perceptor_factory=lambda _config: SizePerceptor(),
    )
    session.intelligence = manager
    session.origin = monotonic()
    session.started = True
    session.threads = (Thread(target=session._infer, daemon=True),)
    worker = session.threads[0]
    worker.start()

    baseline = AssessmentEngine(RiskConfig(), stream_id="rule-baseline")
    actual_decisions = []
    expected_decisions = []
    try:
        for index, (width, height) in enumerate(sizes):
            record = FrameRecord(
                index,
                index * 250.0,
                640,
                480,
                (
                    Detection(
                        7,
                        0,
                        "person",
                        0.9,
                        (320 - width // 2, 336 - height, 320 + width // 2, 336),
                    ),
                ),
                timestamp_source="monotonic",
            )
            image = np.full((480, 640, 3), index, dtype=np.uint8)
            session.frames.publish(CapturedFrame(index, record.timestamp_ms, monotonic(), image))
            expected_decisions.append(baseline.step(record))
            result = _wait_for(
                lambda index=index: (
                    value
                    if (value := session.snapshot()[0]) is not None
                    and value.record.frame_index == index
                    else None
                )
            )
            actual_decisions.append(result.decision)
            if index == 0:
                assert _wait_for(lambda: providers[0].infer_started.is_set())
                assert not gate.is_set()

        expected = [
            (alert.alert_type, alert.timestamp_ms, alert.reason)
            for decision in expected_decisions
            for alert in decision.alerts
        ]
        actual = [
            (alert.alert_type, alert.timestamp_ms, alert.reason)
            for decision in actual_decisions
            for alert in decision.alerts
        ]
        assert expected == actual
        assert [alert.alert_type for decision in actual_decisions for alert in decision.alerts] == [
            "first",
            "escalation",
        ]
        assert len(providers[0].states) == 1
        assert not gate.is_set()
    finally:
        session.stop_event.set()
        with session.frames.condition:
            session.frames.condition.notify_all()
        worker.join(timeout=2)
        gate.set()
        manager.close()
        baseline.finish()
    assert not worker.is_alive()


def test_live_session_creates_hybrid_manager_lazily_and_reuses_it(tmp_path: Path) -> None:
    import yaml

    config_path = tmp_path / "configs" / "hybrid.yaml"
    config_path.parent.mkdir()
    config_path.write_text(yaml.safe_dump(_config().to_dict()), encoding="utf-8")
    camera_calls = []
    session = LiveSession(
        PerceptionConfig(),
        RiskConfig(),
        LiveConfig(),
        camera_factory=lambda index: camera_calls.append(index),
    )
    try:
        assert session.intelligence is None
        manager = session.enable_intelligence(tmp_path)
        assert session.enable_intelligence(tmp_path) is manager
        assert session.started is False
        assert camera_calls == []
        assert manager.snapshot()["actual_alert_controller"] == "Rule Engine"
    finally:
        session.close()
