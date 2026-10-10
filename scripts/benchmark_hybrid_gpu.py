"""Reproducible, standalone benchmark for the pinned Laya hybrid models.

Run this script with the model-specific isolated Python environment, for example::

    models/hybrid/env-vision-gpu/Scripts/python.exe scripts/benchmark_hybrid_gpu.py \
        --model vision --device cuda --precision fp16 \
        --output outputs/hybrid/gpu-benchmark/vision-cuda-fp16

The output directory must not already exist. This script only reads local model files and
never downloads or exports weights. Results include inference failures and invalid outputs.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
VISION_MODEL_ID = "thaitea/laya-vision"
VISION_REVISION = "f2fe3c12cb6d04c59d8a190250bf3fb40fc828dc"
VISION_CODE_REVISION = "9e1e2419d855ad3e1a2af4d4bd1ef6be5418842c"
STRUCTURED_MODEL_ID = "convaiinnovations/laya"
STRUCTURED_REVISION = "7b928d828b7b0e022f929d9bd2e44165aa270148"
STRUCTURED_CODE_REVISION = "68804629e8ccd9d616d48a40e87de9fabbeae069"
SEED = 20261010
STALE_AFTER_MS = 2000.0


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        json_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def json_safe(value: Any) -> Any:
    """Convert model outputs and runtime metadata into strict JSON values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"non_finite_float": repr(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        return json_safe(value.model_dump())
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except (ValueError, RuntimeError):
            pass
    return {"unserializable_type": type(value).__qualname__, "repr": repr(value)}


def package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def configure_local_runtime(root: Path) -> dict[str, str]:
    """Keep temporary files and caches for this process on the project drive."""
    hybrid = root / "models" / "hybrid"
    cache = hybrid / "cache" / "gpu-benchmark"
    temp = hybrid / "tmp"
    paths = {
        "TEMP": temp,
        "TMP": temp,
        "TMPDIR": temp,
        "HF_HOME": cache / "huggingface",
        "HF_HUB_CACHE": cache / "huggingface" / "hub",
        "HUGGINGFACE_HUB_CACHE": cache / "huggingface" / "hub",
        "TORCH_HOME": cache / "torch",
        "PIP_CACHE_DIR": cache / "pip",
        "UV_CACHE_DIR": cache / "uv",
    }
    for path in set(paths.values()):
        path.mkdir(parents=True, exist_ok=True)
    resolved = {name: str(path.resolve()) for name, path in paths.items()}
    os.environ.update(resolved)
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "PYTHONUTF8": "1",
        }
    )
    return resolved


def nvidia_smi_snapshot() -> dict[str, Any]:
    """Read device and process memory when the local driver exposes it."""
    record: dict[str, Any] = {"available": False, "gpu_rows": [], "process_rows": []}
    try:
        gpu = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.free,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
        if gpu.returncode == 0:
            record["gpu_rows"] = [line.strip() for line in gpu.stdout.splitlines() if line.strip()]
            record["available"] = True
        else:
            record["gpu_error"] = gpu.stderr.strip() or f"nvidia-smi exit {gpu.returncode}"
    except (OSError, subprocess.TimeoutExpired) as exc:
        record["gpu_error"] = f"{type(exc).__name__}: {exc}"

    try:
        processes = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
        if processes.returncode == 0:
            record["process_rows"] = [
                line.strip() for line in processes.stdout.splitlines() if line.strip()
            ]
            record["current_process_memory_mb"] = next(
                (
                    line.split(",")[-1].strip()
                    for line in record["process_rows"]
                    if line.split(",", 1)[0].strip() == str(os.getpid())
                ),
                None,
            )
        else:
            record["process_error"] = (
                processes.stderr.strip() or f"nvidia-smi exit {processes.returncode}"
            )
    except (OSError, subprocess.TimeoutExpired) as exc:
        record["process_error"] = f"{type(exc).__name__}: {exc}"
    return record


def start_gpu_device_sampler(output_dir: Path) -> Any:
    """Use the pipeline benchmark's shared sampler for observed device-wide memory."""
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from benchmark_hybrid_pipeline import NvidiaSmiSampler

    sampler = NvidiaSmiSampler(output_dir / "gpu_device_samples.jsonl", 0.25)
    sampler.start()
    return sampler


def summarize_gpu_device_samples(path: Path) -> dict[str, Any]:
    devices: dict[tuple[str, str], dict[str, Any]] = {}
    errors = 0
    sample_count = 0
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            sample_count += 1
            sample = json.loads(line)
            if sample.get("error"):
                errors += 1
            for device in sample.get("devices", []):
                key = (str(device.get("index", "unknown")), str(device.get("name", "unknown")))
                summary = devices.setdefault(
                    key,
                    {
                        "index": key[0],
                        "name": key[1],
                        "samples_with_memory": 0,
                        "sampled_device_memory_peak_mib": None,
                    },
                )
                used = device.get("memory_used_mib")
                if isinstance(used, (int, float)) and math.isfinite(used):
                    summary["samples_with_memory"] += 1
                    current = summary["sampled_device_memory_peak_mib"]
                    summary["sampled_device_memory_peak_mib"] = (
                        float(used) if current is None else max(float(current), float(used))
                    )
    return {
        "sample_file": str(path.resolve()),
        "sample_count": sample_count,
        "sample_errors": errors,
        "sampling_interval_s": 0.25,
        "scope": (
            "sampled whole-device memory usage, not process-exclusive; the maximum is a "
            "sampled lower bound and may include unrelated GPU activity"
        ),
        "devices": list(devices.values()),
    }


def torch_cuda_memory(torch: Any, device: str) -> dict[str, int] | None:
    if device != "cuda" or not torch.cuda.is_available():
        return None
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "free_bytes": int(torch.cuda.mem_get_info()[0]),
        "total_bytes": int(torch.cuda.mem_get_info()[1]),
    }


def synchronize(torch: Any, device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(p * len(ordered)) - 1))
    return round(ordered[index], 3)


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "p50_ms": None, "p95_ms": None, "mean_ms": None}
    return {
        "count": len(values),
        "p50_ms": round(statistics.median(values), 3),
        "p95_ms": percentile(values, 0.95),
        "mean_ms": round(statistics.fmean(values), 3),
    }


def validate_vision_model(model_dir: Path) -> dict[str, Any]:
    files = [
        model_dir / "vlm_agent_config.json",
        model_dir / "backbone" / "config.json",
        model_dir / "model.safetensors",
        model_dir / "processor" / "processor_config.json",
        model_dir / "processor" / "tokenizer.json",
        model_dir / "processor" / "tokenizer_config.json",
        model_dir / "sightguard_model_manifest.json",
    ]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing local Vision model files: " + ", ".join(missing))
    manifest_path = model_dir / "sightguard_model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("model_id") != VISION_MODEL_ID or manifest.get("revision") != VISION_REVISION:
        raise ValueError("Local Vision manifest does not match the pinned model revision.")
    expected = manifest.get("model_safetensors_sha256")
    actual = sha256_file(model_dir / "model.safetensors")
    if expected != actual:
        raise ValueError("Vision model.safetensors SHA-256 does not match its manifest.")
    return {
        "model_id": VISION_MODEL_ID,
        "revision": VISION_REVISION,
        "model_license": manifest.get("license"),
        "code_revision": VISION_CODE_REVISION,
        "local_directory": str(model_dir),
        "model_manifest": manifest,
        "files": {
            str(path.relative_to(model_dir)): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        },
    }


def validate_structured_model(model_dir: Path, onnx_path: Path, precision: str) -> dict[str, Any]:
    manifest_path = model_dir / "model_manifest.json"
    config_path = model_dir / "rl_agent_config.json"
    tokenizer_path = model_dir / "tokenizer" / "tokenizer.json"
    required = [manifest_path, config_path, tokenizer_path, onnx_path]
    if precision == "fp32":
        required.append(model_dir / "laya.onnx.data")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing local Structured model files: " + ", ".join(missing))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    revision = manifest.get("huggingface_revision")
    if manifest.get("model_id") != STRUCTURED_MODEL_ID or revision != STRUCTURED_REVISION:
        raise ValueError("Local Structured manifest does not match the pinned model revision.")
    expected = manifest.get("onnx_export", {}).get("sha256") if precision == "int8" else None
    actual = sha256_file(onnx_path)
    if expected and expected != actual:
        raise ValueError("Laya int8 ONNX SHA-256 does not match its manifest.")
    files = [manifest_path, config_path, tokenizer_path, onnx_path]
    external_data = model_dir / "laya.onnx.data"
    if precision == "fp32" and external_data.is_file():
        files.append(external_data)
    return {
        "model_id": STRUCTURED_MODEL_ID,
        "revision": STRUCTURED_REVISION,
        "model_license": manifest.get("license"),
        "code_revision": STRUCTURED_CODE_REVISION,
        "local_directory": str(model_dir),
        "backend": "ONNX Runtime",
        "model_manifest": manifest,
        "files": {
            str(path.relative_to(model_dir)): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        },
        "quantization": manifest.get("quantization") if precision == "int8" else None,
    }


def build_fixed_input() -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Reuse the fixed P0 synthetic RGB image and a stable semantic evidence object."""
    # This sibling script owns the established deterministic P0 illustration generator.
    from verify_laya_vision import make_synthetic_image

    image = make_synthetic_image()
    state = {
        "frame_id": 0,
        "timestamp_ms": 1000.0,
        "image_size": [image.width, image.height],
        "perception_status": "ok",
        "objects": [
            {
                "event_id": "synthetic-event-1",
                "track_id": 1,
                "class_id": 0,
                "class_name": "pedestrian",
                "bbox_xyxy": [228, 145, 300, 405],
                "detection_confidence": 0.82,
                "features": {
                    "center_xy": [0.5156, 0.5371],
                    "area": 0.0703,
                    "focus_relevance": 0.82,
                    "observation_duration_ms": 800.0,
                    "scale_change_per_s": 0.03,
                    "center_change_per_s": 0.01,
                    "inward_change_per_s": 0.0,
                    "motion_interval_ms": 200.0,
                    "motion_reliable": False,
                    "continuity": 1.0,
                    "track_id_missing": False,
                    "interrupted": False,
                    "limitations": ["synthetic fixture", "not ground truth"],
                },
            }
        ],
        "task_scope": "first object is focus event; others are context; no navigation",
        "limitations": "Synthetic semantic input; not collision truth or a safety guarantee",
    }
    # Use the project's own versioned task schema and adapt only the upstream Vision shape.
    src_dir = str(ROOT / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from sightguard.hybrid.schema import questions

    structured_questions = questions()
    vision_questions = {
        "risk": {
            **structured_questions["risk"],
            "criteria": list(structured_questions["risk"]["criteria"].values()),
        }
    }
    return image, state, {"structured": structured_questions, "vision": vision_questions}


def input_record(
    image: Any, state: dict[str, Any], question_sets: dict[str, Any]
) -> dict[str, Any]:
    state_hash = sha256_json(state)
    question_hash = sha256_json(question_sets)
    pixels_hash = hashlib.sha256(image.tobytes()).hexdigest()
    combined = hashlib.sha256(
        (pixels_hash + state_hash + question_hash).encode("ascii")
    ).hexdigest()
    return {
        "kind": "fixed synthetic illustration from verify_laya_vision.make_synthetic_image",
        "generator_sha256": sha256_file(Path(__file__).with_name("verify_laya_vision.py")),
        "width": image.width,
        "height": image.height,
        "rgb_pixel_sha256": pixels_hash,
        "state_sha256": state_hash,
        "questions_sha256": question_hash,
        "combined_sha256": combined,
        "state": state,
        "questions_by_backend": question_sets,
        "scope": "identical synthetic state and concern task; Vision uses upstream list criteria",
    }


def code_hashes(model: str, runtime_support_path: Path | None = None) -> dict[str, str]:
    result = {Path(__file__).name: sha256_file(Path(__file__))}
    helper = ROOT / "src" / "sightguard" / "hybrid" / "schema.py"
    result["src/sightguard/hybrid/schema.py"] = sha256_file(helper)
    config_path = ROOT / "configs" / "hybrid.yaml"
    if config_path.is_file():
        result["configs/hybrid.yaml"] = sha256_file(config_path)
    if model == "vision":
        local_code = ROOT / "models" / "hybrid" / "laya-vision-source" / "laya"
        for name in ("vlm.py", "common.py", "agent.py"):
            path = local_code / name
            if path.is_file():
                result[f"laya/{name}"] = sha256_file(path)
    else:
        # The installed Laya package is reported and hashed again after import.
        result["upstream_code_revision"] = STRUCTURED_CODE_REVISION
    if runtime_support_path and runtime_support_path.is_file():
        result["src/sightguard/hybrid/runtime_support.py"] = sha256_file(runtime_support_path)
    return result


def environment_record(
    torch: Any, args: argparse.Namespace, paths: dict[str, str]
) -> dict[str, Any]:
    environment: dict[str, Any] = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "torch": str(torch.__version__),
        "torch_module": str(Path(torch.__file__).resolve()),
        "torch_cuda_runtime": torch.version.cuda,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "numpy": package_version("numpy"),
        "pillow": package_version("pillow"),
        "transformers": package_version("transformers"),
        "runtime_paths": paths,
        "thread_settings": {"torch_intra_op": 2, "torch_inter_op": 1},
        "seed": SEED,
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        environment["cuda_device"] = {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": int(properties.total_memory),
        }
    environment["nvidia_smi_before_load"] = nvidia_smi_snapshot()
    if args.model == "vision":
        environment["laya"] = package_version("laya")
    return environment


def parse_model_output(raw: Any, source: str, parse_function: Callable) -> dict[str, Any]:
    started = time.perf_counter()
    candidate, score, parse_error = parse_function(raw, source)
    elapsed = (time.perf_counter() - started) * 1000
    raw_choice: Any = None
    if isinstance(raw, dict):
        answers = raw.get("answers", raw)
        if isinstance(answers, dict):
            answer = answers.get("risk")
            if isinstance(answer, dict):
                raw_choice = answer.get("choice")
    # Vision's list-criteria API returns the UNKNOWN description string, which the
    # shared parser maps back to the UNKNOWN label before it reports abstention.
    valid_unknown = candidate is None and parse_error == "Insufficient evidence (UNKNOWN)"
    valid = parse_error is None or valid_unknown
    return {
        "elapsed_ms": round(elapsed, 3),
        "status": "valid_unknown" if valid_unknown else "valid" if valid else "invalid",
        "candidate_risk_level": candidate,
        "candidate_score": score,
        "raw_choice": json_safe(raw_choice),
        "error": None if valid_unknown else parse_error,
    }


def attach_call_timing(
    started: float,
    completed: float,
    call_events: list[dict[str, Any]],
) -> dict[str, Any]:
    if not call_events:
        return {
            "predict_call_ms": round((completed - started) * 1000, 3),
            "preprocessing_before_first_model_call_ms": None,
            "model_forward_ms": None,
            "between_model_calls_ms": None,
            "model_postprocessing_after_last_call_ms": None,
            "model_component_calls": [],
        }
    event_total = sum(event["duration_ms"] for event in call_events)
    between = sum(
        max(0.0, call_events[index + 1]["started"] - call_events[index]["completed"]) * 1000
        for index in range(len(call_events) - 1)
    )
    return {
        "predict_call_ms": round((completed - started) * 1000, 3),
        "preprocessing_before_first_model_call_ms": round(
            max(0.0, (call_events[0]["started"] - started) * 1000), 3
        ),
        "model_forward_ms": round(event_total, 3),
        "between_model_calls_ms": round(between, 3),
        "model_postprocessing_after_last_call_ms": round(
            max(0.0, (completed - call_events[-1]["completed"]) * 1000), 3
        ),
        "model_component_calls": [
            {
                "component": event.get("component", "model_call"),
                "duration_ms": round(event["duration_ms"], 3),
            }
            for event in call_events
        ],
    }


def vision_parameter_profile(agent: Any) -> dict[str, Any]:
    dtypes: dict[str, int] = {}
    devices: dict[str, int] = {}
    parameter_count = 0
    for parameter in agent.model.parameters():
        count = int(parameter.numel())
        dtype = str(parameter.dtype).removeprefix("torch.")
        device = str(parameter.device)
        dtypes[dtype] = dtypes.get(dtype, 0) + count
        devices[device] = devices.get(device, 0) + count
        parameter_count += count
    return {
        "parameter_count": parameter_count,
        "parameter_dtype_counts": dtypes,
        "parameter_device_counts": devices,
    }


def run_vision(
    args: argparse.Namespace, record: dict[str, Any], torch: Any
) -> tuple[Any, list[dict[str, Any]]]:
    if args.precision == "int8":
        raise ValueError("Laya Vision has no pinned int8 checkpoint; use fp32 or fp16.")
    if args.device == "cpu" and args.precision == "fp16":
        raise ValueError("Laya Vision CPU inference is restricted to fp32.")
    model_dir = (args.model_dir or ROOT / "models" / "hybrid" / "vision").resolve()
    record["model"] = validate_vision_model(model_dir)
    import laya

    record["environment"]["laya_module"] = str(Path(laya.__file__).resolve())
    record["code_hashes"]["installed/laya/vlm.py"] = sha256_file(
        Path(laya.__file__).with_name("vlm.py")
    )
    record["model_load_started_at"] = utc_now()
    load_start = time.perf_counter()
    # The pinned upstream loader always starts on CPU/FP32; placement and precision follow.
    src_dir = str(ROOT / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from sightguard.hybrid.runtime_support import load_vision

    agent = load_vision(laya, str(model_dir))
    record["weight_reader"] = "pread" if os.name == "nt" else "mmap"
    if args.full_fp16:
        if args.device != "cuda" or args.precision != "fp16":
            raise ValueError("--full-fp16 requires --model vision --device cuda --precision fp16.")
        agent.model.to(device="cuda", dtype=torch.float16)
        agent.device = torch.device("cuda")
        agent.cfg["dtype"] = "fp16"
        agent._torch_dtype = lambda: torch.float16
        record["precision_adapter"] = "explicit whole-model FP16 attempt, including readout"
    else:
        src_dir = str(ROOT / "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        from sightguard.hybrid.runtime_support import configure_vision

        record["precision_adapter"] = configure_vision(agent, torch, args.device, args.precision)
    agent.model.eval()
    synchronize(torch, args.device)
    record["model_load_latency_ms"] = round((time.perf_counter() - load_start) * 1000, 3)
    record["model_loaded_at"] = utc_now()
    record["model_parameters"] = vision_parameter_profile(agent)
    record["gpu_memory_after_load"] = torch_cuda_memory(torch, args.device)
    image, state, question_sets = record["_fixed_input"]
    questions = question_sets["vision"]
    note = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
    model_calls: list[dict[str, Any]] = []

    def wrap_model_call(component: str, original: Callable) -> Callable:
        def measured(*call_args: Any, **call_kwargs: Any) -> Any:
            synchronize(torch, args.device)
            call_started = time.perf_counter()
            try:
                return_value = original(*call_args, **call_kwargs)
                synchronize(torch, args.device)
                return return_value
            finally:
                call_completed = time.perf_counter()
                model_calls.append(
                    {
                        "component": component,
                        "started": call_started,
                        "completed": call_completed,
                        "duration_ms": (call_completed - call_started) * 1000,
                    }
                )

        return measured

    original_forward = agent.model.forward
    original_encode_images = agent.model.encode_images
    agent.model.forward = wrap_model_call("vision_decision_forward", original_forward)
    agent.model.encode_images = wrap_model_call("vision_image_encoder", original_encode_images)
    samples: list[dict[str, Any]] = []
    try:

        def predict() -> Any:
            return agent.predict(
                {"image": image, "note": note},
                questions,
                n_permutations=1,
                batch_size=1,
                prefix_cache=False,
            )

        samples = execute_inference_series(
            predict,
            torch=torch,
            device=args.device,
            source="vision",
            record=record,
            model_forward_calls=model_calls,
        )
    finally:
        agent.model.forward = original_forward
        agent.model.encode_images = original_encode_images
    return agent, samples


def make_ort_session_options(ort: Any, profile_prefix: Path | None, profiling: bool) -> Any:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if profiling:
        options.enable_profiling = True
        if profile_prefix is not None:
            options.profile_file_prefix = str(profile_prefix)
    return options


class TimedOrtSession:
    """Proxy that synchronizes CUDA around actual ORT session.run calls."""

    def __init__(self, session: Any, torch: Any, device: str):
        self.session = session
        self.torch = torch
        self.device = device
        self.call_events: list[dict[str, float]] = []

    def run(self, *run_args: Any, **run_kwargs: Any) -> Any:
        synchronize(self.torch, self.device)
        started = time.perf_counter()
        try:
            result = self.session.run(*run_args, **run_kwargs)
            synchronize(self.torch, self.device)
            return result
        finally:
            completed = time.perf_counter()
            self.call_events.append(
                {
                    "component": "onnx_session_run",
                    "started": started,
                    "completed": completed,
                    "duration_ms": (completed - started) * 1000,
                }
            )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.session, name)


def execute_inference_series(
    predict: Callable[[], Any],
    *,
    torch: Any,
    device: str,
    source: str,
    record: dict[str, Any],
    model_forward_calls: list[dict[str, float]] | None = None,
) -> list[dict[str, Any]]:
    from sightguard.hybrid.schema import parse_output

    all_samples: list[dict[str, Any]] = []
    batches = [("cold_first_call", 1), ("warmup", record["request"]["warmup"])]
    for phase, count in batches:
        for index in range(count):
            if phase == "cold_first_call" and index > 0:
                continue
            if model_forward_calls is not None:
                model_forward_calls.clear()
                call_events = model_forward_calls
            else:
                session_proxy = record["_timed_ort_session"]
                session_proxy.call_events.clear()
                call_events = session_proxy.call_events
            synchronize(torch, device)
            started_at = utc_now()
            started = time.perf_counter()
            try:
                raw = predict()
                synchronize(torch, device)
                completed = time.perf_counter()
                timing = attach_call_timing(started, completed, call_events)
                parsed = parse_model_output(raw, source, parse_output)
                end_to_end = timing["predict_call_ms"] + parsed["elapsed_ms"]
                sample = {
                    "phase": phase,
                    "phase_index": index,
                    "started_at": started_at,
                    "inference_ok": True,
                    "timing": timing,
                    "risk_parse": parsed,
                    "end_to_end_ms": round(end_to_end, 3),
                    "end_to_end_under_2s": end_to_end < STALE_AFTER_MS,
                    "gpu_memory": torch_cuda_memory(torch, device),
                    "raw_output": json_safe(raw),
                }
            except Exception as exc:
                completed = time.perf_counter()
                timing = attach_call_timing(started, completed, call_events)
                sample = {
                    "phase": phase,
                    "phase_index": index,
                    "started_at": started_at,
                    "inference_ok": False,
                    "timing": timing,
                    "gpu_memory": torch_cuda_memory(torch, device),
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                }
            sample["completed_at"] = utc_now()
            all_samples.append(sample)
            record["inference_records"].append(sample)
            if not sample["inference_ok"]:
                # A failed cold/warmup call invalidates the steady-state sequence.
                record["series_stopped_after_error"] = phase
                return all_samples

    stable: list[dict[str, Any]] = []
    for index in range(record["request"]["iterations"]):
        if model_forward_calls is not None:
            model_forward_calls.clear()
            call_events = model_forward_calls
        else:
            session_proxy = record["_timed_ort_session"]
            session_proxy.call_events.clear()
            call_events = session_proxy.call_events
        synchronize(torch, device)
        started_at = utc_now()
        started = time.perf_counter()
        try:
            raw = predict()
            synchronize(torch, device)
            completed = time.perf_counter()
            timing = attach_call_timing(started, completed, call_events)
            parsed = parse_model_output(raw, source, parse_output)
            end_to_end = timing["predict_call_ms"] + parsed["elapsed_ms"]
            sample = {
                "phase": "steady_state",
                "phase_index": index,
                "started_at": started_at,
                "inference_ok": True,
                "timing": timing,
                "risk_parse": parsed,
                "end_to_end_ms": round(end_to_end, 3),
                "end_to_end_under_2s": end_to_end < STALE_AFTER_MS,
                "gpu_memory": torch_cuda_memory(torch, device),
                "raw_output": json_safe(raw),
            }
        except Exception as exc:
            completed = time.perf_counter()
            timing = attach_call_timing(started, completed, call_events)
            sample = {
                "phase": "steady_state",
                "phase_index": index,
                "started_at": started_at,
                "inference_ok": False,
                "timing": timing,
                "gpu_memory": torch_cuda_memory(torch, device),
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            }
        sample["completed_at"] = utc_now()
        all_samples.append(sample)
        stable.append(sample)
        record["inference_records"].append(sample)
    record["steady_state_summary"] = summarize_samples(stable)
    return all_samples


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [sample for sample in samples if sample.get("inference_ok")]
    summary: dict[str, Any] = {
        "requested": len(samples),
        "inference_successes": len(successful),
        "inference_errors": len(samples) - len(successful),
        "valid_outputs": sum(
            sample.get("risk_parse", {}).get("status") in {"valid", "valid_unknown"}
            for sample in successful
        ),
        "valid_unknown_outputs": sum(
            sample.get("risk_parse", {}).get("status") == "valid_unknown" for sample in successful
        ),
        "invalid_outputs": sum(
            sample.get("risk_parse", {}).get("status") == "invalid" for sample in successful
        ),
        "valid_non_unknown_under_2s": sum(
            sample.get("risk_parse", {}).get("status") == "valid"
            and sample.get("end_to_end_under_2s", False)
            for sample in successful
        ),
        "valid_unknown_under_2s": sum(
            sample.get("risk_parse", {}).get("status") == "valid_unknown"
            and sample.get("end_to_end_under_2s", False)
            for sample in successful
        ),
        "latency_ms": {},
    }
    phase_keys = {
        "predict_call": "predict_call_ms",
        "preprocessing_before_first_model_call": "preprocessing_before_first_model_call_ms",
        "model_forward": "model_forward_ms",
        "between_model_calls": "between_model_calls_ms",
        "model_postprocessing_after_last_call": "model_postprocessing_after_last_call_ms",
        "risk_result_parsing": None,
        "end_to_end_including_risk_parse": None,
    }
    for label, key in phase_keys.items():
        if key is None and label == "risk_result_parsing":
            values = [
                sample["risk_parse"]["elapsed_ms"]
                for sample in successful
                if "risk_parse" in sample
            ]
        elif key is None:
            values = [sample["end_to_end_ms"] for sample in successful if "end_to_end_ms" in sample]
        else:
            values = [
                sample["timing"][key]
                for sample in successful
                if sample.get("timing", {}).get(key) is not None
            ]
        summary["latency_ms"][label] = summarize(values)
    return summary


def run_structured(
    args: argparse.Namespace,
    record: dict[str, Any],
    torch: Any,
    output_dir: Path,
) -> tuple[Any, list[dict[str, Any]], Any]:
    if args.precision == "fp16":
        raise ValueError("No pinned Laya Structured FP16 ONNX export exists; use fp32 or int8.")
    model_dir = (args.model_dir or ROOT / "models" / "hybrid" / "laya-structured-source").resolve()
    default_name = "laya.int8.onnx" if args.precision == "int8" else "laya.onnx"
    onnx_path = (args.onnx or model_dir / default_name).resolve()
    record["model"] = validate_structured_model(model_dir, onnx_path, args.precision)

    import onnxruntime as ort

    record["environment"]["onnxruntime"] = str(ort.__version__)
    record["environment"]["onnxruntime_available_providers"] = list(ort.get_available_providers())
    preload = getattr(ort, "preload_dlls", None)
    if callable(preload):
        try:
            preload()
            record["environment"]["onnxruntime_preload_dlls"] = "success"
        except Exception as exc:
            record["environment"]["onnxruntime_preload_dlls"] = {
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
        if "CUDAExecutionProvider" not in record["environment"]["onnxruntime_available_providers"]:
            raise RuntimeError(
                "CUDA requested but ONNX Runtime does not expose CUDAExecutionProvider."
            )
        torch.cuda.init()
    if (
        args.device == "cuda"
        and "CUDAExecutionProvider" not in record["environment"]["onnxruntime_available_providers"]
    ):
        raise RuntimeError("CUDA requested but ONNX Runtime has no CUDAExecutionProvider.")

    from laya.onnx_agent import ONNXAgent

    record["environment"]["laya"] = package_version("laya")
    record["environment"]["laya_onnx_agent_module"] = str(
        Path(sys.modules[ONNXAgent.__module__].__file__).resolve()
    )
    agent_module = Path(sys.modules[ONNXAgent.__module__].__file__).resolve()
    record["code_hashes"]["installed/laya/onnx_agent.py"] = sha256_file(agent_module)
    record["model_load_started_at"] = utc_now()
    load_started = time.perf_counter()
    original_session_constructor = ort.InferenceSession
    original_get_providers = ort.get_available_providers
    initial_profile = output_dir / "ort_profile_initial_cpu"

    def bounded_cpu_session(*session_args: Any, **session_kwargs: Any) -> Any:
        session_kwargs["sess_options"] = make_ort_session_options(
            ort,
            initial_profile if args.device == "cpu" else None,
            profiling=args.device == "cpu",
        )
        session_kwargs["providers"] = ["CPUExecutionProvider"]
        return original_session_constructor(*session_args, **session_kwargs)

    ort.InferenceSession = bounded_cpu_session
    ort.get_available_providers = lambda: ["CPUExecutionProvider"]
    try:
        agent = ONNXAgent(str(model_dir), onnx_path=str(onnx_path))
    finally:
        ort.InferenceSession = original_session_constructor
        ort.get_available_providers = original_get_providers
    initial_session = agent.session
    target_session_start = time.perf_counter()
    target_profile_prefix: Path | None = None
    if args.device == "cuda":
        agent.session = None
        del initial_session
        gc.collect()
        target_profile_prefix = output_dir / "ort_profile_cuda"
        options = make_ort_session_options(ort, target_profile_prefix, profiling=True)
        agent.session = ort.InferenceSession(
            str(onnx_path),
            sess_options=options,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
    else:
        agent.session = initial_session
    target_session_ms = (time.perf_counter() - target_session_start) * 1000
    raw_session = agent.session
    session_providers = list(raw_session.get_providers())
    provider_options = raw_session.get_provider_options()
    record["runtime_execution"] = {
        "requested_device": args.device,
        "precision": args.precision,
        "session_providers_configured": session_providers,
        "provider_options": json_safe(provider_options),
        "session_setup": (
            "official ONNXAgent initialized with bounded CPU session; "
            "CUDA run releases it and creates requested CUDA+CPU session"
            if args.device == "cuda"
            else "official ONNXAgent initialized with bounded CPU session and profiling"
        ),
        "actual_node_execution_evidence": (
            "ORT profiling Node events; provider listing is recorded only as configuration"
        ),
    }
    record["model_load_latency_ms"] = round((time.perf_counter() - load_started) * 1000, 3)
    record["target_session_initialization_ms"] = round(target_session_ms, 3)
    record["model_loaded_at"] = utc_now()
    record["ort_profile_prefix"] = str(target_profile_prefix or initial_profile)
    proxy = TimedOrtSession(raw_session, torch, args.device)
    agent.session = proxy
    record["_timed_ort_session"] = proxy

    _, state, question_sets = record["_fixed_input"]
    questions = question_sets["structured"]

    def predict() -> Any:
        return agent.system_one(state, questions)

    samples = execute_inference_series(
        predict,
        torch=torch,
        device=args.device,
        source="laya",
        record=record,
    )
    return agent, samples, raw_session


def summarize_ort_profile(profile_path: Path) -> dict[str, Any]:
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    provider_totals: dict[str, dict[str, Any]] = {}
    events_seen = 0
    for event in payload:
        if event.get("cat") != "Node":
            continue
        args = event.get("args") or {}
        provider = args.get("provider")
        if not provider:
            continue
        events_seen += 1
        name = str(args.get("op_name") or event.get("name") or "unknown")
        duration = float(event.get("dur", 0.0))
        key = (provider, name)
        group = groups.setdefault(
            key, {"provider": provider, "node_name": name, "count": 0, "duration_us": 0.0}
        )
        group["count"] += 1
        group["duration_us"] += duration
        provider_record = provider_totals.setdefault(
            provider, {"node_count": 0, "duration_us": 0.0}
        )
        provider_record["node_count"] += 1
        provider_record["duration_us"] += duration
    for value in provider_totals.values():
        value["duration_us"] = round(value["duration_us"], 3)
    nodes = sorted(
        ({**group, "duration_us": round(group["duration_us"], 3)} for group in groups.values()),
        key=lambda item: (item["provider"], item["node_name"]),
    )
    providers = set(provider_totals)
    total_node_duration = sum(item["duration_us"] for item in provider_totals.values())
    total_node_events = sum(item["node_count"] for item in provider_totals.values())
    dominant_provider = max(
        provider_totals,
        key=lambda provider: provider_totals[provider]["duration_us"],
        default=None,
    )
    if not providers:
        conclusion = "no_profiled_node_events"
    elif providers == {"CUDAExecutionProvider"}:
        conclusion = "cuda_nodes_only"
    elif "CUDAExecutionProvider" in providers and "CPUExecutionProvider" in providers:
        conclusion = "cuda_and_cpu_mixed"
    elif "CUDAExecutionProvider" in providers:
        conclusion = "cuda_nodes_with_other_provider"
    else:
        conclusion = "cpu_nodes_only"
    return {
        "profile_file": str(profile_path),
        "node_event_count": events_seen,
        "node_provider_counts_and_durations": provider_totals,
        "node_names_by_provider": nodes,
        "execution_conclusion_from_profiled_nodes": conclusion,
        "dominant_node_provider_by_profiled_duration": dominant_provider,
        "cuda_node_count_share": round(
            provider_totals.get("CUDAExecutionProvider", {}).get("node_count", 0)
            / total_node_events,
            4,
        )
        if total_node_events
        else None,
        "cuda_node_duration_share": round(
            provider_totals.get("CUDAExecutionProvider", {}).get("duration_us", 0.0)
            / total_node_duration,
            4,
        )
        if total_node_duration
        else None,
    }


def finalize_ort_profile(record: dict[str, Any], raw_session: Any) -> None:
    try:
        profile_name = raw_session.end_profiling()
        profile_path = Path(profile_name).resolve()
        record["onnxruntime_profile"] = summarize_ort_profile(profile_path)
    except Exception as exc:
        record["onnxruntime_profile"] = {
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.iterations < 1:
        parser.error("--iterations must be at least 1")
    if args.warmup < 0:
        parser.error("--warmup must be zero or greater")
    if args.full_fp16 and not (
        args.model == "vision" and args.device == "cuda" and args.precision == "fp16"
    ):
        parser.error("--full-fp16 requires --model vision --device cuda --precision fp16")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("vision", "laya"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--precision", choices=("fp32", "fp16", "int8"), required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New result directory; existing directories are rejected.",
    )
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument(
        "--onnx", type=Path, help="Optional existing local ONNX path for Laya Structured."
    )
    parser.add_argument(
        "--full-fp16",
        action="store_true",
        help="Attempt Vision all-half parameters, including its FP32 readout.",
    )
    args = parser.parse_args()
    validate_args(parser, args)
    return args


def markdown_report(record: dict[str, Any]) -> str:
    request = record.get("request", {})
    model = record.get("model", {})
    steady = record.get("steady_state_summary", {})
    stable_successes = steady.get("inference_successes", 0)
    lines = [
        "# Hybrid GPU feasibility benchmark",
        "",
        f"- Status: **{record.get('status', 'unknown')}**",
        f"- Model: `{request.get('model')}` ({model.get('model_id', 'unresolved')})",
        f"- Device / precision: `{request.get('device')}` / `{request.get('precision')}`",
        f"- Started: {record.get('started_at_utc', 'unknown')}",
        f"- Load time: {record.get('model_load_latency_ms', 'unavailable')} ms",
        f"- Stable iterations: {stable_successes} / {request.get('iterations', 0)} successful",
        "",
        "## Steady-state latency",
        "",
    ]
    for label, values in steady.get("latency_ms", {}).items():
        lines.append(
            f"- {label.replace('_', ' ')}: P50 {values.get('p50_ms')} ms, "
            f"P95 {values.get('p95_ms')} ms, mean {values.get('mean_ms')} ms "
            f"(n={values.get('count', 0)})."
        )
    cold = next(
        (
            sample
            for sample in record.get("inference_records", [])
            if sample.get("phase") == "cold_first_call"
        ),
        None,
    )
    if cold:
        lines.extend(["", "## Cold call", ""])
        if cold.get("inference_ok"):
            lines.append(
                f"- End-to-end: {cold.get('end_to_end_ms')} ms; risk parse: "
                f"{cold.get('risk_parse', {}).get('status')}."
            )
        else:
            error = cold.get("error", {})
            lines.append(f"- Failed: `{error.get('type')}: {error.get('message')}`")
    if request.get("device") == "cuda":
        lines.extend(["", "## GPU memory", ""])
        if request.get("model") == "vision":
            lines.append(
                f"- Vision CUDA allocator after load: `{record.get('gpu_memory_after_load')}`"
            )
            peaks = [
                sample.get("gpu_memory")
                for sample in record.get("inference_records", [])
                if sample.get("gpu_memory")
            ]
            lines.append(
                f"- Vision CUDA allocator samples: {len(peaks)}; final record in `benchmark.json`."
            )
        else:
            lines.append(
                "- ONNX Runtime allocations are not represented by the PyTorch allocator. "
                "See the before/after `nvidia-smi` snapshots; Windows may not expose "
                "per-process memory."
            )
        lines.append(
            f"- nvidia-smi after run: `{record.get('environment', {}).get('nvidia_smi_after_run')}`"
        )
        observation = record.get("gpu_device_memory_observation")
        if observation:
            for device in observation.get("devices", []):
                lines.append(
                    f"- {device.get('name')} sampled device memory peak: "
                    f"{device.get('sampled_device_memory_peak_mib')} MiB "
                    f"(n={device.get('samples_with_memory')})."
                )
            lines.append(f"- Memory sample scope: {observation.get('scope')}")
    if record.get("onnxruntime_profile"):
        profile = record["onnxruntime_profile"]
        lines.extend(["", "## ONNX Runtime node placement", ""])
        profile_conclusion = profile.get("execution_conclusion_from_profiled_nodes", "unavailable")
        lines.append(f"- Profile evidence: **{profile_conclusion}**")
        for provider, counts in profile.get("node_provider_counts_and_durations", {}).items():
            lines.append(
                f"- {provider}: {counts.get('node_count')} node events, "
                f"{counts.get('duration_us')} µs total profiled node time."
            )
        lines.append(
            "- Configured provider names are not treated as proof of GPU execution; "
            "see node names and timings in `benchmark.json`."
        )
    if record.get("model_parameters"):
        lines.extend(["", "## Vision parameter placement", ""])
        lines.append(
            f"- Parameter dtype counts: "
            f"`{record['model_parameters'].get('parameter_dtype_counts')}`"
        )
        lines.append(
            f"- Parameter device counts: "
            f"`{record['model_parameters'].get('parameter_device_counts')}`"
        )
    if steady:
        lines.extend(["", "## Output validity and 2-second window", ""])
        lines.append(
            "- Valid non-UNKNOWN parsed outputs completed within 2 s: "
            f"{steady.get('valid_non_unknown_under_2s', 0)}."
        )
        lines.append(
            "- Valid UNKNOWN outputs completed within 2 s: "
            f"{steady.get('valid_unknown_under_2s', 0)}; UNKNOWN is valid abstention, "
            "not a risk candidate."
        )
        lines.append(
            "- The fixed synthetic call-time check does not establish fresh-scene "
            "performance or a safety guarantee."
        )
    error = record.get("error")
    if error:
        lines.extend(["", "## Run error", "", f"- `{error.get('type')}: {error.get('message')}`"])
    if record.get("inference_records"):
        errors = [
            sample for sample in record["inference_records"] if not sample.get("inference_ok")
        ]
        invalid = [
            sample
            for sample in record["inference_records"]
            if sample.get("risk_parse", {}).get("status") == "invalid"
        ]
        lines.extend(
            [
                "",
                "## Stability",
                "",
                f"- Inference exceptions: {len(errors)}; invalid result schemas: {len(invalid)}.",
            ]
        )
        for sample in errors:
            error = sample.get("error", {})
            lines.append(
                f"- {sample.get('phase')} #{sample.get('phase_index')}: "
                f"`{error.get('type')}: {error.get('message')}`"
            )
        for sample in invalid:
            lines.append(
                f"- Invalid output {sample.get('phase')} #{sample.get('phase_index')}: "
                f"`{sample.get('risk_parse', {}).get('error')}`"
            )
    lines.extend(
        [
            "",
            "Full per-iteration outputs, hashes, environment, profile evidence, and tracebacks "
            "are in `benchmark.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(output_dir: Path, record: dict[str, Any]) -> None:
    record.pop("_fixed_input", None)
    record.pop("_timed_ort_session", None)
    json_path = output_dir / "benchmark.json"
    report_path = output_dir / "report.md"
    json_path.write_text(
        json.dumps(json_safe(record), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(markdown_report(record), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output.expanduser()
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists; refusing to overwrite: {output_dir}"
        )
    output_dir.mkdir(parents=True)

    random.seed(SEED)
    record: dict[str, Any] = {
        "schema_version": "sightguard.hybrid.gpu-benchmark.v1",
        "started_at_utc": utc_now(),
        "status": "failed",
        "request": {
            "model": args.model,
            "device": args.device,
            "precision": args.precision,
            "full_fp16": args.full_fp16,
            "iterations": args.iterations,
            "warmup": args.warmup,
            "output_directory": str(output_dir),
            "model_dir_override": str(args.model_dir.resolve()) if args.model_dir else None,
            "onnx_override": str(args.onnx.resolve()) if args.onnx else None,
        },
        "stale_after_ms": STALE_AFTER_MS,
        "environment": {"runtime_paths": {}},
        "code_hashes": code_hashes(args.model),
        "inference_records": [],
        "error": None,
    }
    agent: Any = None
    raw_session: Any = None
    gpu_sampler: Any = None
    paths: dict[str, str] = {}
    try:
        paths = configure_local_runtime(ROOT)
        record["environment"]["runtime_paths"] = paths
        if args.device == "cuda":
            gpu_sampler = start_gpu_device_sampler(output_dir)
            record["gpu_device_samples_path"] = str(output_dir / "gpu_device_samples.jsonl")
        src_dir = str(ROOT / "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        image, state, question_sets = build_fixed_input()
        record["input"] = input_record(image, state, question_sets)
        record["_fixed_input"] = (image, state, question_sets)

        # Load the model runtime's selected Torch first. ORT's Windows CUDA DLL preload
        # must see this CUDA-enabled runtime before creating an ONNX session.
        import torch

        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
        if args.device == "cuda":
            torch.cuda.init()
        record["environment"].update(environment_record(torch, args, paths))
        record["environment"]["git"] = git_record()
        if args.model == "vision":
            agent, _ = run_vision(args, record, torch)
        else:
            agent, _, raw_session = run_structured(args, record, torch, output_dir)
            finalize_ort_profile(record, raw_session)
        record["environment"]["nvidia_smi_after_run"] = nvidia_smi_snapshot()
        if args.model == "vision":
            record["gpu_memory_peak"] = torch_cuda_memory(torch, args.device)
        successful = [item for item in record["inference_records"] if item.get("inference_ok")]
        invalid = [
            item for item in successful if item.get("risk_parse", {}).get("status") == "invalid"
        ]
        errors = [item for item in record["inference_records"] if not item.get("inference_ok")]
        if errors:
            record["status"] = "failed" if not successful else "partial_failure"
        elif invalid:
            record["status"] = "invalid_output"
        elif successful:
            record["status"] = "success"
        else:
            record["status"] = "failed"
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        try:
            import torch

            record["environment"]["nvidia_smi_after_run"] = nvidia_smi_snapshot()
            if torch.cuda.is_available():
                record["gpu_memory_peak"] = torch_cuda_memory(torch, args.device)
        except Exception:
            pass
    finally:
        if raw_session is not None:
            try:
                raw_session.end_profiling()
            except Exception:
                pass
        record.pop("_timed_ort_session", None)
        if agent is not None:
            try:
                agent.session = None
            except Exception:
                pass
        agent = None
        raw_session = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                record["gpu_memory_after_unload"] = torch_cuda_memory(torch, args.device)
        except Exception:
            pass
        if gpu_sampler is not None:
            try:
                gpu_sampler.close()
                record["gpu_device_memory_observation"] = summarize_gpu_device_samples(
                    output_dir / "gpu_device_samples.jsonl"
                )
                record["gpu_device_memory_observation"]["sampler_reported_errors"] = list(
                    gpu_sampler.errors
                )
            except Exception as exc:
                record["gpu_device_memory_observation"] = {
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
        record["finished_at_utc"] = utc_now()
        record["elapsed_wall_ms"] = round(
            (
                datetime.fromisoformat(record["finished_at_utc"])
                - datetime.fromisoformat(record["started_at_utc"])
            ).total_seconds()
            * 1000,
            3,
        )
        try:
            write_outputs(output_dir, record)
        except Exception as exc:
            print(
                f"Could not write benchmark artifacts: {type(exc).__name__}: {exc}", file=sys.stderr
            )
            raise

    print(
        json.dumps(
            {
                "status": record["status"],
                "output_directory": str(output_dir),
                "json": str(output_dir / "benchmark.json"),
                "report": str(output_dir / "report.md"),
            },
            ensure_ascii=False,
        )
    )
    return 0 if record["status"] == "success" else 2


def git_record() -> dict[str, Any]:
    def read_git(*arguments: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *arguments],
                cwd=ROOT,
                capture_output=True,
                check=False,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return completed.stdout.strip() if completed.returncode == 0 else None

    return {
        "commit": read_git("rev-parse", "HEAD"),
        "branch": read_git("rev-parse", "--abbrev-ref", "HEAD"),
        "working_tree_status": read_git("status", "--short"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
