"""Run one offline P0 inference with a locally downloaded Laya Vision checkpoint."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MODEL_ID = "thaitea/laya-vision"
MODEL_REVISION = "f2fe3c12cb6d04c59d8a190250bf3fb40fc828dc"
MODEL_LICENSE = "CC-BY-NC-SA-4.0"
CODE_REPOSITORY = "https://github.com/r33drichards/laya-vision"
CODE_REVISION = "9e1e2419d855ad3e1a2af4d4bd1ef6be5418842c"
CODE_LICENSE = "Apache-2.0"
MANIFEST_NAME = "sightguard_model_manifest.json"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def configure_local_runtime(root: Path) -> dict[str, str]:
    hybrid_dir = root / "models" / "hybrid"
    cache_dir = hybrid_dir / "cache"
    temp_dir = hybrid_dir / "tmp"
    hub_cache_dir = cache_dir / "hub"
    torch_home_dir = cache_dir / "torch"
    pip_cache_dir = cache_dir / "pip"
    for directory in (cache_dir, temp_dir, hub_cache_dir, torch_home_dir, pip_cache_dir):
        directory.mkdir(parents=True, exist_ok=True)

    paths = {
        "TEMP": str(temp_dir.resolve()),
        "TMP": str(temp_dir.resolve()),
        "TMPDIR": str(temp_dir.resolve()),
        "UV_CACHE_DIR": str(cache_dir.resolve()),
        "HF_HOME": str(cache_dir.resolve()),
        "HF_HUB_CACHE": str(hub_cache_dir.resolve()),
        "HUGGINGFACE_HUB_CACHE": str(hub_cache_dir.resolve()),
        "TORCH_HOME": str(torch_home_dir.resolve()),
        "PIP_CACHE_DIR": str(pip_cache_dir.resolve()),
    }
    os.environ.update(paths)
    return paths


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def required_model_files(model_dir: Path) -> list[Path]:
    return [
        model_dir / "vlm_agent_config.json",
        model_dir / "backbone" / "config.json",
        model_dir / "model.safetensors",
        model_dir / "processor" / "processor_config.json",
        model_dir / "processor" / "tokenizer.json",
        model_dir / "processor" / "tokenizer_config.json",
    ]


def validate_local_model(model_dir: Path) -> dict[str, Any]:
    missing = [str(path) for path in required_model_files(model_dir) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Local Laya Vision files are missing; this script never downloads them: "
            + ", ".join(missing)
        )

    manifest_path = model_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing {manifest_path}; download and verify the pinned "
            "model before running inference."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("model_id") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            "Local model manifest does not identify the pinned Laya Vision checkpoint."
        )

    weights_path = model_dir / "model.safetensors"
    actual_hash = sha256_file(weights_path)
    if manifest.get("model_safetensors_sha256") != actual_hash:
        raise ValueError("model.safetensors SHA-256 does not match the local source manifest.")
    return {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "license": MODEL_LICENSE,
        "local_path": str(model_dir),
        "weight_file": weights_path.name,
        "weight_size_bytes": weights_path.stat().st_size,
        "weight_sha256": actual_hash,
        "manifest": manifest,
        "checkpoint_layout": (
            "model.safetensors is loaded as the full encoder and decision checkpoint; "
            "backbone/config.json supplies architecture only, and processor/ "
            "supplies image/tokenizer assets."
        ),
    }


def make_synthetic_image() -> Any:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (512, 512), (142, 190, 220))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 511, 210), fill=(181, 213, 229))
    draw.polygon([(0, 225), (158, 220), (275, 512), (0, 512)], fill=(74, 81, 88))
    draw.polygon([(352, 220), (511, 225), (511, 512), (238, 512)], fill=(74, 81, 88))
    draw.polygon([(158, 220), (352, 220), (238, 512), (275, 512)], fill=(196, 190, 174))
    draw.line((255, 480, 255, 405), fill=(235, 219, 168), width=5)
    draw.line((255, 365, 255, 315), fill=(235, 219, 168), width=4)
    draw.rectangle((198, 165, 314, 225), fill=(168, 173, 176))
    draw.rectangle((210, 145, 302, 165), fill=(148, 153, 157))
    draw.ellipse((241, 250, 271, 280), fill=(45, 40, 36))
    draw.polygon([(236, 280), (276, 280), (285, 355), (228, 355)], fill=(36, 89, 150))
    draw.line((237, 300, 208, 345), fill=(45, 40, 36), width=10)
    draw.line((274, 300, 300, 341), fill=(45, 40, 36), width=10)
    draw.line((240, 352, 225, 403), fill=(38, 43, 49), width=12)
    draw.line((270, 352, 284, 403), fill=(38, 43, 49), width=12)
    draw.line((0, 232, 188, 232), fill=(220, 214, 201), width=5)
    draw.line((324, 232, 511, 232), fill=(220, 214, 201), width=5)
    return image


def choose_device(requested: str, torch: Any) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return requested


def gpu_memory(torch: Any) -> dict[str, int] | None:
    if not torch.cuda.is_available():
        return None
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def profile_parameters(agent: Any) -> dict[str, Any]:
    dtype_counts: dict[str, int] = {}
    parameter_count = 0
    for parameter in agent.model.parameters():
        dtype = str(parameter.dtype).removeprefix("torch.")
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + parameter.numel()
        parameter_count += parameter.numel()
    return {"parameter_count": parameter_count, "parameter_dtype_counts": dtype_counts}


def nvidia_driver_version() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return next((line.strip() for line in result.stdout.splitlines() if line.strip()), None)


def run_attempt(
    *,
    torch: Any,
    laya: Any,
    model_dir: Path,
    image: Any,
    device: str,
    precision: str,
) -> dict[str, Any]:
    attempt: dict[str, Any] = {
        "device": device,
        "precision_requested": precision,
        "load_started_at": utc_now(),
        "load_latency_ms": None,
        "inference_started_at": None,
        "inference_completed_at": None,
        "inference_latency_ms": None,
        "gpu_memory_before_load": gpu_memory(torch),
        "error": None,
    }
    load_start = time.perf_counter()
    try:
        # The official loader reads the local checkpoint, processor, and config files only.
        agent = laya.load_vlm(str(model_dir), device="cpu", dtype="fp32")
        selected_precision = precision
        if selected_precision == "auto":
            selected_precision = "fp16" if device == "cuda" else "fp32"
        dtype = torch.float16 if selected_precision == "fp16" else torch.float32
        if device == "cpu" and selected_precision == "fp16":
            raise ValueError("FP16 CPU inference was not requested as a fallback; use fp32 on CPU.")

        agent.model.to(device="cpu", dtype=dtype)
        agent.model.to(torch.device(device))
        agent.device = torch.device(device)
        agent.cfg["dtype"] = selected_precision
        # This pinned upstream revision maps only "bf16" and "fp32" in _torch_dtype().
        # Override that selector in memory so its image preprocessing follows this FP16 model.
        agent._torch_dtype = lambda: dtype
        agent.source = {"id": MODEL_ID, "revision": MODEL_REVISION}
        attempt["precision"] = selected_precision
        attempt["load_latency_ms"] = round((time.perf_counter() - load_start) * 1000, 3)
        attempt["model_profile"] = profile_parameters(agent)
        attempt["gpu_memory_after_load"] = gpu_memory(torch)

        question = {
            "scene": {
                "type": "choice",
                "instructions": "Which description best fits the visible forward walking path?",
                "criteria": [
                    "A pedestrian is on the walking path",
                    "A bicycle is on the walking path",
                    "A vehicle is blocking the path",
                    "The walking path appears clear",
                    "The image is too unclear to tell",
                ],
            }
        }
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        attempt["inference_started_at"] = utc_now()
        inference_start = time.perf_counter()
        output = agent.predict(
            {
                "image": image,
                "note": "Synthetic illustration for the local P0 scene task.",
            },
            question,
            n_permutations=1,
            batch_size=1,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        attempt["inference_latency_ms"] = round((time.perf_counter() - inference_start) * 1000, 3)
        attempt["inference_completed_at"] = utc_now()
        attempt["raw_output"] = output
        attempt["gpu_memory_after_inference"] = gpu_memory(torch)
        attempt["model_profile"] = profile_parameters(agent)
        attempt["success"] = True
    except (
        Exception
    ) as error:  # Record the real exception so the caller can decide about a CPU retry.
        attempt["load_latency_ms"] = (
            round((time.perf_counter() - load_start) * 1000, 3)
            if attempt["load_latency_ms"] is None
            else attempt["load_latency_ms"]
        )
        attempt["success"] = False
        attempt["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    return attempt


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    default_model_dir = root / "models" / "hybrid" / "vision"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=default_model_dir)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--precision", choices=("auto", "fp16", "fp32"), default="auto")
    parser.add_argument(
        "--image",
        type=Path,
        help="Optional local image; omitted uses a generated test illustration.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="JSON output path; omitted creates a timestamped file under outputs/hybrid/p0.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    runtime_paths = configure_local_runtime(root)
    output_path = args.output or (
        root / "outputs" / "hybrid" / "p0" / f"vision_p0_{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    )
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_dir = args.model_dir.resolve()

    record: dict[str, Any] = {
        "schema_version": "sightguard.hybrid.p0.vision.v1",
        "created_at": utc_now(),
        "status": "failed",
        "source": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_license": MODEL_LICENSE,
            "code_repository": CODE_REPOSITORY,
            "code_revision": CODE_REVISION,
            "code_license": CODE_LICENSE,
            "runtime_import_name": "laya",
        },
        "request": {
            "device": args.device,
            "precision": args.precision,
            "model_dir": str(model_dir),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "runtime_paths": runtime_paths,
            "environment_lock_path": str(
                (root / "models" / "hybrid" / "env-vision" / "requirements-lock.txt").resolve()
            ),
            "shared_torch_pth_path": str(
                (
                    root
                    / "models"
                    / "hybrid"
                    / "env-vision"
                    / "Lib"
                    / "site-packages"
                    / "sightguard_shared_cpu_torch.pth"
                ).resolve()
            ),
        },
        "input": None,
        "attempts": [],
        "raw_output": None,
        "error": None,
    }

    try:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        model_record = validate_local_model(model_dir)
        record["model"] = model_record

        import huggingface_hub
        import laya
        import numpy
        import PIL
        import safetensors
        import torch
        import transformers

        record["environment"].update(
            {
                "torch": str(torch.__version__),
                "torch_module": str(Path(torch.__file__).resolve()),
                "torch_cuda_runtime": torch.version.cuda,
                "transformers": transformers.__version__,
                "huggingface_hub": huggingface_hub.__version__,
                "safetensors": safetensors.__version__,
                "numpy": numpy.__version__,
                "pillow": PIL.__version__,
                "laya_module": str(Path(laya.__file__).resolve()),
                "nvidia_driver": nvidia_driver_version(),
                "cuda_available": bool(torch.cuda.is_available()),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "gpu_total_memory_bytes": (
                    int(torch.cuda.get_device_properties(0).total_memory)
                    if torch.cuda.is_available()
                    else None
                ),
            }
        )

        if args.image:
            from PIL import Image

            image_path = args.image.resolve()
            image_bytes = image_path.read_bytes()
            image = Image.open(image_path).convert("RGB")
            image_source = {
                "kind": "local_file",
                "path": str(image_path),
                "file_sha256": sha256_bytes(image_bytes),
            }
        else:
            image = make_synthetic_image()
            image_source = {"kind": "synthetic_illustration", "generator": "make_synthetic_image()"}
        record["input"] = {
            **image_source,
            "size": list(image.size),
            "rgb_pixel_sha256": sha256_bytes(image.tobytes()),
            "context": "A simple synthetic illustration for a local P0 scene-classification check.",
            "question": {
                "type": "choice",
                "instructions": "Which description best fits the visible forward walking path?",
                "criteria": [
                    "A pedestrian is on the walking path",
                    "A bicycle is on the walking path",
                    "A vehicle is blocking the path",
                    "The walking path appears clear",
                    "The image is too unclear to tell",
                ],
            },
        }

        device = choose_device(args.device, torch)
        precision = args.precision
        if precision == "auto":
            precision = "fp16" if device == "cuda" else "fp32"
        if precision == "fp16" and device == "cpu":
            raise ValueError("FP16 CPU inference is not supported by this verification script.")

        attempt = run_attempt(
            torch=torch,
            laya=laya,
            model_dir=model_dir,
            image=image,
            device=device,
            precision=precision,
        )
        record["attempts"].append(attempt)
        if not attempt["success"] and args.device == "auto" and device == "cuda":
            gc.collect()
            torch.cuda.empty_cache()
            fallback = run_attempt(
                torch=torch,
                laya=laya,
                model_dir=model_dir,
                image=image,
                device="cpu",
                precision="fp32",
            )
            fallback["fallback_reason"] = attempt["error"]
            record["attempts"].append(fallback)
            attempt = fallback
        if not attempt["success"]:
            record["error"] = attempt["error"]
        else:
            record["status"] = "success"
            record["selected_device"] = attempt["device"]
            record["selected_precision"] = attempt["precision"]
            record["load_latency_ms"] = attempt["load_latency_ms"]
            record["inference_latency_ms"] = attempt["inference_latency_ms"]
            record["raw_output"] = attempt["raw_output"]
    except Exception as error:
        record["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }

    output_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({"status": record["status"], "output": str(output_path)}, ensure_ascii=False))
    return 0 if record["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
