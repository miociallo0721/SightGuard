"""Cancellation terminates the owned auxiliary process, never the camera/YOLO worker."""

import base64
import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
from time import monotonic
from typing import TYPE_CHECKING, Protocol

import cv2
import numpy as np

from sightguard.hybrid.config import ModelConfig
from sightguard.hybrid.schema import questions

if TYPE_CHECKING:
    from sightguard.live.runtime import LiveResult


class DecisionProvider(Protocol):
    def load(self, cancel: Event) -> dict: ...
    def infer(self, state: dict, image: np.ndarray | None, cancel: Event) -> dict: ...
    def unload(self) -> None: ...


class RuleDecisionProvider:
    """Read-only projection of an already computed Phase 2 result; no second risk engine."""

    def load(self, cancel: Event) -> dict:
        return {"model_id": "SightGuard Rule Engine", "revision": "phase2", "device": "cpu"}

    def infer(self, state: "LiveResult", image: np.ndarray | None, cancel: Event) -> dict:
        return {"raw_output": asdict(state.decision), "actual_alert_controller": "Rule Engine"}

    def unload(self) -> None:
        pass


class ProcessDecisionProvider:
    """All lifecycle methods belong to the semantic thread. Bounded JSONL transport."""

    def __init__(
        self,
        source: str,
        config: ModelConfig,
        *,
        timeout_s: float,
        load_timeout_s: float,
        log_path: Path,
    ) -> None:
        self.source, self.config = source, config
        self.timeout_s, self.load_timeout_s = timeout_s, load_timeout_s
        self.log_path = log_path
        self.process = None
        self.reader = None
        self.stderr = None
        self.responses: Queue = Queue(maxsize=1)

    def _read(self, process: subprocess.Popen, responses: Queue) -> None:
        try:
            for line in process.stdout:
                responses.put(json.loads(line), timeout=1)
        except Exception as exc:
            try:
                responses.put({"ok": False, "error": f"Host protocol: {exc}"}, timeout=1)
            except Exception:
                pass

    def _exchange(self, payload: dict, cancel: Event, timeout: float) -> dict:
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("Auxiliary host not running")
        self.process.stdin.write(json.dumps(payload, allow_nan=False) + "\n")
        self.process.stdin.flush()
        deadline = monotonic() + timeout
        while monotonic() < deadline:
            if cancel.is_set():
                raise InterruptedError("Auxiliary operation cancelled")
            try:
                answer = self.responses.get(timeout=min(0.05, max(0.001, deadline - monotonic())))
            except Empty:
                if self.process.poll() is not None:
                    raise RuntimeError(
                        f"Auxiliary host exited: {self.process.returncode}"
                    ) from None
                continue
            if not answer.get("ok"):
                raise RuntimeError(answer.get("error", "Auxiliary host failed"))
            return answer
        raise TimeoutError("Auxiliary inference/load timeout")

    def load(self, cancel: Event) -> dict:
        if not Path(self.config.python).is_file() or not Path(self.config.path).is_dir():
            raise FileNotFoundError("Local model or isolated Python not installed")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.stderr = self.log_path.open("a", encoding="utf-8")
        self.responses = Queue(maxsize=1)
        temporary = self.log_path.parent / "tmp"
        temporary.mkdir(exist_ok=True)
        environment = {**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"}
        for name in ("TEMP", "TMP", "TMPDIR", "HF_HOME", "TORCH_HOME"):
            environment[name] = str(temporary.resolve())
        self.process = subprocess.Popen(
            [self.config.python, "-u", str(Path(__file__).with_name("model_host.py"))],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr,
            text=True,
            encoding="utf-8",
            env=environment,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        self.reader = Thread(target=self._read, args=(self.process, self.responses), daemon=True)
        self.reader.start()
        return self._exchange(
            {"source": self.source, "config": asdict(self.config)}, cancel, self.load_timeout_s
        )["metadata"]

    def infer(self, state: dict, image: np.ndarray | None, cancel: Event) -> dict:
        payload = {"state": state, "questions": questions()}
        if self.source == "vision":
            payload["questions"]["risk"]["criteria"] = list(
                payload["questions"]["risk"]["criteria"].values()
            )
            if image is None or not image.size:
                raise ValueError("Vision requires the associated original camera frame")
            ok, encoded = cv2.imencode(".png", image)
            if not ok:
                raise ValueError("Cannot encode source camera image")
            payload["image"] = base64.b64encode(encoded).decode("ascii")
        return self._exchange(payload, cancel, self.timeout_s)

    def unload(self) -> None:
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=3)
            if self.reader is not None:
                self.reader.join(timeout=2)
            for stream in (self.process.stdin, self.process.stdout):
                if stream:
                    stream.close()
            self.process = None
        if self.stderr is not None:
            self.stderr.close()
            self.stderr = None
