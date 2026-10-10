"""Run an offline smoke inference with a locally exported Laya ONNX checkpoint.

Example (from the repository root):

    models/hybrid/env-structured/Scripts/python.exe scripts/verify_laya_structured.py \
        --model-dir models/hybrid/laya-structured-int8

The model directory must already contain the Laya config/tokenizer files and the ONNX
file. This script never downloads a model or falls back to a remote checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
import traceback
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _smoke_input() -> tuple[dict[str, Any], dict[str, Any]]:
    state = {
        "scenario": "synthetic structured-perception smoke fixture",
        "objects": [
            {
                "category": "pedestrian",
                "relative_position": "ahead-left",
                "motion_trend": "moving toward the projected walking corridor",
                "observed_duration_seconds": 0.8,
                "detection_confidence": 0.82,
            }
        ],
        "temporal_features": {
            "approach_trend": "increasing",
            "time_to_contact_seconds": None,
            "validity": "insufficient samples for a reliable estimate",
        },
        "perception_reliability": "single synthetic example; not ground truth",
    }
    questions = {
        "candidate_risk_level": {
            "type": "choice",
            "instructions": (
                "For research triage only, which candidate risk category best matches this "
                "structured observation? This is not a calibrated collision probability or "
                "a safety guarantee."
            ),
            "criteria": {
                "none": "No relevant near-term concern is indicated by these observations.",
                "monitor": "A possible concern is present; continue observing for more evidence.",
                "caution": "A notable possible concern warrants heightened attention.",
                "urgent": "A strong immediate concern is indicated by the available observation.",
            },
        }
    }
    return state, questions


def _write_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    temporary = root / "models/hybrid/tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    for key in ("TEMP", "TMP", "TMPDIR", "HF_HOME", "TORCH_HOME"):
        os.environ[key] = str(temporary)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help=(
            "Existing local Laya checkpoint directory containing rl_agent_config.json "
            "and tokenizer/."
        ),
    )
    parser.add_argument(
        "--onnx-path",
        type=Path,
        default=Path("laya.int8.onnx"),
        help=(
            "Existing ONNX file; a relative path is resolved under --model-dir "
            "(default: laya.int8.onnx)."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=("onnx", "pytorch"),
        default="onnx",
        help="Inference backend; ONNX is the default, PyTorch uses the local eager CPU checkpoint.",
    )
    parser.add_argument(
        "--repeat-count",
        type=int,
        default=3,
        help="Number of predictions to time, including the first (default: 3).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Record JSON path (default: outputs/hybrid/p0/laya-structured-<UTC timestamp>.json).",
    )
    args = parser.parse_args()

    started_at = datetime.now(UTC)
    timestamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    output_path = args.output or Path("outputs/hybrid/p0") / f"laya-structured-{timestamp}.json"
    model_dir = args.model_dir.expanduser().resolve()
    onnx_path = args.onnx_path.expanduser()
    if not onnx_path.is_absolute():
        onnx_path = model_dir / onnx_path
    onnx_path = onnx_path.resolve()

    # Keep every Hugging Face/Transformers path offline. The upstream agents accept a local
    # checkpoint path, so no network lookup is needed.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    record: dict[str, Any] = {
        "schema_version": 1,
        "status": "error",
        "started_at_utc": started_at.isoformat(),
        "input_provenance": "synthetic smoke fixture; no camera footage or ground-truth label",
        "runtime": {
            "python": sys.version,
            "laya": _package_version("laya"),
            "onnxruntime": _package_version("onnxruntime"),
            "numpy": _package_version("numpy"),
            "transformers": _package_version("transformers"),
            "logical_cpu_count": os.cpu_count(),
        },
        "model": {"local_directory": str(model_dir), "backend": args.backend},
        "load_latency_ms": None,
        "inference_latency_ms": None,
        "raw_output": None,
        "error": None,
    }

    try:
        if args.repeat_count < 1:
            raise ValueError("--repeat-count must be at least 1")
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Local model directory does not exist: {model_dir}")
        required_files = [model_dir / "rl_agent_config.json"]
        if args.backend == "onnx":
            record["model"]["onnx_file"] = str(onnx_path)
            required_files.append(onnx_path)
        else:
            required_files.append(model_dir / "model.safetensors")
        tokenizer_candidates = [
            model_dir / "tokenizer" / "tokenizer.json",
            model_dir / "tokenizer.json",
        ]
        if not any(path.is_file() for path in tokenizer_candidates):
            raise FileNotFoundError(f"No local tokenizer.json found under {model_dir}")
        for required_file in required_files:
            if not required_file.is_file():
                raise FileNotFoundError(
                    f"Required local model file does not exist: {required_file}"
                )

        manifest_path = model_dir / "model_manifest.json"
        if manifest_path.is_file():
            record["model"]["manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))

        record["model"]["backend"] = args.backend
        record["model"]["sha256"] = {}
        if args.backend == "onnx":
            record["model"]["sha256"][str(onnx_path)] = _sha256(onnx_path)
        else:
            checkpoint = model_dir / "model.safetensors"
            record["model"]["sha256"][str(checkpoint)] = _sha256(checkpoint)
        for candidate in (model_dir / "rl_agent_config.json", *tokenizer_candidates):
            if candidate.is_file():
                record["model"]["sha256"][str(candidate)] = _sha256(candidate)

        state, questions = _smoke_input()
        record["input"] = {"state": state, "questions": questions}

        load_started = time.perf_counter()
        if args.backend == "onnx":
            import onnxruntime as ort
            from laya.onnx_agent import ONNXAgent

            record["runtime"]["onnxruntime_available_providers"] = ort.get_available_providers()
            record["runtime"]["onnxruntime_thread_config"] = (
                "upstream ONNXAgent creates default SessionOptions; intra_op_num_threads "
                "is left at ORT default (0)"
            )
            agent = ONNXAgent(str(model_dir), onnx_path=str(onnx_path))
            session_providers = agent.session.get_providers()
            record["runtime"]["onnxruntime_session_providers"] = session_providers
            record["runtime"]["execution_device"] = (
                "CPU" if "CPUExecutionProvider" in session_providers else "unknown"
            )
        else:
            import laya

            agent = laya.load(
                str(model_dir),
                device="cpu",
                compile=False,
                backend="eager",
            )
            record["runtime"]["execution_device"] = str(agent.device)
            if hasattr(agent, "model") and agent.model is not None:
                record["runtime"]["model_parameter_dtype"] = str(
                    next(agent.model.parameters()).dtype
                )
        record["load_latency_ms"] = round((time.perf_counter() - load_started) * 1000, 3)

        latencies_ms: list[float] = []
        for _ in range(args.repeat_count):
            inference_started = time.perf_counter()
            result = agent.system_one(state, questions)
            latencies_ms.append(round((time.perf_counter() - inference_started) * 1000, 3))

        record["inference_latency_ms"] = {
            "samples": latencies_ms,
            "first_call": latencies_ms[0],
            "median_all_calls": round(statistics.median(latencies_ms), 3),
            "repeat_count": args.repeat_count,
        }
        record["raw_output"] = result
        record["status"] = "success"
    except Exception as exc:  # The failure itself is a first-class P0 result.
        record["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }

    record["finished_at_utc"] = datetime.now(UTC).isoformat()
    record["record_file"] = str(output_path.resolve())
    _write_record(output_path, record)
    print(json.dumps(record, ensure_ascii=False, indent=2))
    print(f"\nSaved P0 record to {output_path.resolve()}")
    return 0 if record["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
