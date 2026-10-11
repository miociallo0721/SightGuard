"""Single semantic owner and capacity-one input; no access to real alert mutation."""

import hashlib
import json
import logging
from collections import deque
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from threading import Condition, Event, Thread
from time import monotonic
from typing import TYPE_CHECKING
from uuid import uuid4

import cv2
import numpy as np

from sightguard.hybrid.config import HybridConfig
from sightguard.hybrid.input import build_input
from sightguard.hybrid.provider import (
    DecisionProvider,
    ProcessDecisionProvider,
    RuleDecisionProvider,
)
from sightguard.hybrid.schema import DecisionResult, fuse, parse_output

if TYPE_CHECKING:
    from sightguard.live.runtime import LiveResult

LOGGER = logging.getLogger(__name__)


def utc() -> str:
    return datetime.now(UTC).isoformat()


class IntelligenceManager:
    """Selected auxiliary only. UI commands invalidate a generation, never touch models."""

    def __init__(
        self,
        config: HybridConfig,
        root: Path,
        session_id: str,
        *,
        provider_factory: Callable[[str], DecisionProvider] | None = None,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self.config, self.root, self.session_id, self.clock = config, root, session_id, clock
        self.condition = Condition()
        self.stop = Event()
        self.cancel = Event()
        self.generation = 0
        self.mode, self.auxiliary, self.want_load = "rule", "laya", False
        self.pending: LiveResult | None = None
        self.current: LiveResult | None = None
        self.latest: DecisionResult | None = None
        self.comparison: dict | None = None
        self.status, self.error = "Unavailable", None
        self.model: dict = {}
        self.latencies: deque[float] = deque(maxlen=256)
        self.model_latencies = {name: deque(maxlen=256) for name in config.models}
        self.model_calls = dict.fromkeys(config.models, 0)
        self.model_timeouts = dict.fromkeys(config.models, 0)
        self.rule_provider = RuleDecisionProvider()
        self.fps_before: deque[float] = deque(maxlen=120)
        self.fps_after: deque[float] = deque(maxlen=120)
        self.calls = self.timeouts = self.stale_count = 0
        self.peak_vram_mb = None
        self.output = root / config.output_dir / f"session-{uuid4().hex}"
        self.output.mkdir(parents=True, exist_ok=False)
        self.metadata = {
            "schema_version": "1.0",
            "session_id": session_id,
            "created_at": utc(),
            "actual_alert_controller": "Rule Engine",
            "shadow_only": True,
            "config": config.to_dict(),
            "config_sha256": hashlib.sha256(
                json.dumps(config.to_dict(), sort_keys=True).encode()
            ).hexdigest(),
            "rule_log_scope": "sampled source frames used for semantic tasks, not all frames",
            "models": {},
            "input_source": "original local camera + immutable Phase 2 features",
        }
        self._write_metadata()
        self.factory = provider_factory or self._provider
        self.thread = Thread(target=self._work, name="sightguard-semantic", daemon=True)
        self.thread.start()

    def _provider(self, source: str) -> DecisionProvider:
        return ProcessDecisionProvider(
            source,
            self.config.models[source],
            timeout_s=self.config.timeout_s,
            load_timeout_s=self.config.load_timeout_s,
            log_path=self.output / f"{source}-host.log",
        )

    def _write_metadata(self) -> None:
        (self.output / "metadata.json").write_text(
            json.dumps(self.metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def record_source(self, source: dict) -> None:
        """Called before auxiliary loading, never on the fast inference path."""
        self.metadata["source_configuration"] = source
        path = self.root / source["perception_config"]["model_path"]
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            self.metadata["yolo_sha256"] = digest.hexdigest()
        self._write_metadata()

    def _append(self, filename: str, payload: dict) -> None:
        with (self.output / filename).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")

    def select(self, mode: str, auxiliary: str = "laya", load: bool = True) -> None:
        if mode not in {"rule", "laya", "vision", "hybrid"} or auxiliary not in {"laya", "vision"}:
            raise ValueError("Unknown research mode/model")
        if type(load) is not bool:
            raise ValueError("load must be boolean")
        source = mode if mode in {"laya", "vision"} else auxiliary
        with self.condition:
            if self.stop.is_set():
                raise RuntimeError("Intelligence session closed")
            self.cancel.set()
            self.cancel = Event()
            self.generation += 1
            self.mode, self.auxiliary = mode, source
            self.want_load = load and mode != "rule"
            self.pending = self.latest = self.comparison = None
            self.model, self.error = {}, None
            self.status = "Loading" if self.want_load else "Unavailable"
            self.condition.notify_all()

    def unload(self) -> None:
        self.select(self.mode, self.auxiliary, load=False)

    def submit(self, result: "LiveResult") -> None:
        """Fast path does no serialization, image encoding, disk I/O, or model work."""
        with self.condition:
            if self.stop.is_set():
                return
            self.current = result
            values = self.fps_after if self.want_load else self.fps_before
            if result.record.status == "ok":
                values.append(result.update_fps)
            if self.want_load:
                self.pending = result
                self.condition.notify_all()

    def _still_valid(self, result: DecisionResult) -> bool:
        current = self.current
        if current is None or current.record.status != "ok":
            return False
        if current.record.timestamp_ms - result.timestamp_ms > self.config.stale_after_ms:
            return False
        if result.event_id is None:
            return not current.decision.frame.observations
        return any(
            event.event_id == result.event_id
            and event.observed_now
            and event.state.value not in {"RESOLVED", "LOST"}
            for event in current.decision.frame.events
        )

    def snapshot(self) -> dict:
        with self.condition:
            latest = self.latest
            if (
                latest
                and latest.valid
                and (
                    not self._still_valid(latest)
                    or (self.clock() - self.latest_completed) * 1000 > self.config.stale_after_ms
                )
            ):
                latest = replace(latest, valid=False, stale=True, error="Result no longer current")
                self.latest = latest
                self.stale_count += 1
                self.comparison = {
                    **(self.comparison or {}),
                    "status": "Unavailable: stale",
                    "agreement": None,
                    "auxiliary_level": None,
                    "fusion_level": None,
                }
            samples = list(self.model_latencies[self.auxiliary])
            return {
                "schema_version": "1.0",
                "mode": self.mode,
                "auxiliary": self.auxiliary,
                "actual_alert_controller": "Rule Engine",
                "model_status": self.status,
                "model": dict(self.model),
                "latest_result": latest.to_dict() if latest else None,
                "comparison": dict(self.comparison) if self.comparison else None,
                "performance": {
                    "calls": self.calls,
                    "timeouts": self.timeouts,
                    "stale": self.stale_count,
                    "p50_ms": float(np.percentile(samples, 50)) if samples else None,
                    "p95_ms": float(np.percentile(samples, 95)) if samples else None,
                    "load_ms": self.model.get("load_ms"),
                    "peak_vram_mb": self.peak_vram_mb,
                    "fast_fps_before": float(np.mean(self.fps_before)) if self.fps_before else None,
                    "fast_fps_after": float(np.mean(self.fps_after)) if self.fps_after else None,
                    "fps_scope": "rolling 120 risk updates; before/after means, not a benchmark",
                    "by_model": {
                        name: {
                            "calls": self.model_calls[name],
                            "timeouts": self.model_timeouts[name],
                            "p50_ms": float(np.percentile(values, 50)) if values else None,
                            "p95_ms": float(np.percentile(values, 95)) if values else None,
                            "load_ms": self.metadata["models"].get(name, {}).get("load_ms"),
                        }
                        for name, values in self.model_latencies.items()
                    },
                },
                "error": self.error,
            }

    def _work(self) -> None:
        provider, loaded_generation, last_call = None, -1, 0.0
        try:
            while not self.stop.is_set():
                with self.condition:
                    self.condition.wait_for(
                        lambda observed_generation=loaded_generation: (
                            self.stop.is_set()
                            or self.generation != observed_generation
                            or self.pending is not None
                        ),
                        timeout=0.2,
                    )
                    generation, source, cancel = self.generation, self.auxiliary, self.cancel
                    wanted = self.want_load
                if self.stop.is_set():
                    break
                if loaded_generation != generation:
                    if provider is not None:
                        provider.unload()
                        provider = None
                    loaded_generation, last_call = generation, 0.0
                    if wanted:
                        try:
                            provider = self.factory(source)
                            model = provider.load(cancel)
                            with self.condition:
                                if generation == self.generation:
                                    self.model, self.status = model, "Ready"
                            self.metadata["models"][source] = model
                            self._write_metadata()
                        except Exception as exc:
                            with self.condition:
                                if generation == self.generation:
                                    self.status = (
                                        "Not Installed"
                                        if isinstance(exc, FileNotFoundError)
                                        else "Failed"
                                    )
                                    self.error = f"{type(exc).__name__}: {exc}"
                                    if isinstance(exc, TimeoutError):
                                        self.timeouts += 1
                                        self.model_timeouts[source] += 1
                            self._append("errors.jsonl", {"source": source, "error": str(exc)})
                            if provider:
                                provider.unload()
                            provider = None
                with self.condition:
                    if provider is None:
                        self.pending = None
                        self.condition.wait(timeout=0.2)
                        continue
                    remaining = 1 / self.config.frequency_hz - (self.clock() - last_call)
                    if remaining > 0:
                        self.condition.wait(timeout=min(remaining, 0.2))
                        continue
                    packet, self.pending = self.pending, None
                if packet is None:
                    continue
                # Reject an already-obsolete sample before spending model time on it.
                if packet.record.status != "ok" or packet.captured is None:
                    continue
                if (self.clock() - packet.captured.captured_at) * 1000 > self.config.stale_after_ms:
                    with self.condition:
                        self.stale_count += 1
                    continue
                last_call = self.clock()
                try:
                    if not self._infer(provider, packet, source, generation, cancel):
                        provider = None
                except Exception as exc:
                    # Logging/protocol failures affect only the research worker.
                    LOGGER.exception("Semantic path failed; rule alerts continue")
                    with self.condition:
                        if generation == self.generation:
                            self.status, self.error = "Failed", str(exc)
                    provider.unload()
                    provider = None
        except Exception as exc:
            LOGGER.exception("Semantic worker stopped; rule alerts continue")
            with self.condition:
                self.status, self.error = "Failed", str(exc)
        finally:
            if provider:
                provider.unload()

    def _infer(
        self,
        provider: DecisionProvider,
        packet: "LiveResult",
        source: str,
        generation: int,
        cancel: Event,
    ) -> bool:
        state, event_id = build_input(packet, self.config.max_objects)
        started, wall_start = self.clock(), utc()
        raw, level, score, error = None, None, None, None
        failed = False
        with self.condition:
            if generation != self.generation:
                return True
            self.status = "Running"
            self.calls += 1
            self.model_calls[source] += 1
        rule_event = next((e for e in packet.decision.frame.events if e.event_id == event_id), None)
        rule_level = (
            rule_event.assessment.level.value if rule_event and rule_event.assessment else None
        )
        self._append(
            "rule_decisions.jsonl",
            {
                "session_id": self.session_id,
                "frame_id": packet.record.frame_index,
                "timestamp_ms": packet.record.timestamp_ms,
                "decision": self.rule_provider.infer(packet, None, cancel)["raw_output"],
                "actual_alert_controller": "Rule Engine",
            },
        )
        image = packet.captured.image
        input_record = {
            "session_id": self.session_id,
            "source": source,
            "state": state,
            "event_id": event_id,
            "image_pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
        }
        if source == "vision" and self.config.retain_images:
            filename = f"frame-{packet.record.frame_index}.png"
            if not cv2.imwrite(str(self.output / filename), image):
                raise OSError("Cannot retain original input image")
            input_record["image_file"] = filename
        self._append("inputs.jsonl", input_record)
        try:
            reply = provider.infer(state, image if source == "vision" else None, cancel)
            raw = reply["raw_output"]
            level, score, error = parse_output(raw, source)
            with self.condition:
                self.peak_vram_mb = reply.get("peak_vram_mb")
        except Exception as exc:
            failed = True
            error = f"{type(exc).__name__}: {exc}"
            with self.condition:
                if isinstance(exc, TimeoutError):
                    self.timeouts += 1
                    self.model_timeouts[source] += 1
            # Process-backed timeouts/cancellation release resources, no runaway queue.
            provider.unload()
        completed = self.clock()
        model = self.config.models[source]
        result = DecisionResult(
            uuid4().hex,
            self.session_id,
            source,
            model.model_id,
            model.revision,
            packet.record.frame_index,
            packet.record.timestamp_ms,
            event_id,
            raw,
            level,
            score,
            "upstream choice score; uncalibrated research comparison, not collision probability",
            (completed - started) * 1000,
            wall_start,
            utc(),
            valid=error is None,
            error=error,
        )
        with self.condition:
            stale = (
                generation != self.generation
                or cancel.is_set()
                or not self._still_valid(result)
                or (completed - packet.captured.captured_at) * 1000 > self.config.stale_after_ms
            )
            result = replace(result, stale=stale, valid=result.valid and not stale)
            if stale:
                self.stale_count += 1
            self.latencies.append(result.inference_latency_ms)
            self.model_latencies[source].append(result.inference_latency_ms)
            comparison = {
                "session_id": self.session_id,
                "frame_id": result.frame_id,
                "event_id": event_id,
                "timestamp_ms": result.timestamp_ms,
                "rule_level": rule_level,
                "auxiliary_level": level if result.valid else None,
                "agreement": rule_level == level if result.valid and rule_level else None,
                "status": (
                    "Agreement"
                    if result.valid and rule_level == level
                    else "Disagreement"
                    if result.valid and rule_level
                    else "Unavailable: " + (error or "stale/incomparable")
                ),
                "fusion_level": fuse(rule_level, level, self.config.fusion)
                if result.valid and self.mode == "hybrid"
                else None,
                "fusion_policy": self.config.fusion,
                "actual_alert_controller": "Rule Engine",
            }
            if generation == self.generation:
                self.latest, self.latest_completed = result, completed
                self.comparison = comparison
                self.error = error
                self.status = "Failed" if failed else "Ready"
        self._append(f"{source}_decisions.jsonl", result.to_dict())
        self._append("comparisons.jsonl", comparison)
        return not failed

    def close(self) -> None:
        self.stop.set()
        self.cancel.set()
        with self.condition:
            self.condition.notify_all()
        self.thread.join(timeout=8)
        if self.thread.is_alive():
            raise RuntimeError("Semantic provider failed to honour cancellation")
        with self.condition:
            self.want_load = False
            self.status = "Unavailable"
            self.pending = None
        (self.output / "performance.json").write_text(
            json.dumps(self.snapshot(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
