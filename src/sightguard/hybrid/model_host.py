"""Standalone subprocess entry: one isolated upstream package, local-only weights.

Run by its absolute filename with the configured environment Python. Do not import
SightGuard here: the two upstream packages both use the name `laya`.
"""

import base64
import contextlib
import gc
import hashlib
import io
import json
import os
import sys
import time
from importlib.metadata import version
from pathlib import Path


def send(payload: dict) -> None:
    print(json.dumps(payload, allow_nan=False), flush=True)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    request = json.loads(sys.stdin.readline())
    config, source = request["config"], request["source"]
    try:
        started = time.perf_counter()
        with contextlib.redirect_stdout(sys.stderr):
            import laya
            import torch

            torch.set_num_threads(2)
            torch.set_num_interop_threads(1)
            device = config["device"]
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA runtime unavailable in selected isolated environment")
            model_dir = Path(config["path"])
            if not model_dir.is_dir():
                raise FileNotFoundError(f"Missing local model: {model_dir}")
            manifest_name = (
                "sightguard_model_manifest.json" if source == "vision" else "model_manifest.json"
            )
            manifest = json.loads((model_dir / manifest_name).read_text(encoding="utf-8"))
            revision = manifest.get("revision", manifest.get("huggingface_revision"))
            if manifest.get("model_id") != config["model_id"] or revision != config["revision"]:
                raise ValueError("Local manifest and configured model identity/revision disagree")
            weight = (
                Path(config["onnx_path"])
                if source != "vision" and config["backend"] == "onnx"
                else model_dir / "model.safetensors"
            )
            weight_hash = hash_file(weight)
            expected_hash = (
                manifest.get("model_safetensors_sha256")
                if source == "vision"
                else manifest.get("onnx_export", {}).get("sha256")
                if config["backend"] == "onnx"
                else manifest.get("source_checkpoint_sha256")
            )
            if expected_hash != weight_hash:
                raise ValueError("Local weight hash disagrees with source/export manifest")
            if source == "vision":
                from runtime_support import configure_vision, load_vision

                agent = load_vision(laya, str(model_dir))
                profile = configure_vision(agent, torch, device, config["precision"])
                precision = profile["precision"]
            elif config["backend"] == "onnx":
                from laya.onnx_agent import ONNXAgent

                agent = ONNXAgent(str(model_dir), onnx_path=str(weight))
                # Use the same official graph/session contract, with bounded CPU threads.
                import onnxruntime as ort

                options = ort.SessionOptions()
                options.intra_op_num_threads = 2
                options.inter_op_num_threads = 1
                agent.session = (
                    None  # Release the default session before allocating its replacement.
                )
                gc.collect()
                agent.session = ort.InferenceSession(
                    str(weight), options, providers=["CPUExecutionProvider"]
                )
                device, precision = "cpu", "int8"
            else:
                if config["precision"] != "fp32":
                    raise ValueError("Structured eager baseline currently supports FP32 only")
                agent = laya.load(str(model_dir), device=device, compile=False, backend="eager")
                precision = str(next(agent.model.parameters()).dtype)
            metadata = {
                "model_id": config["model_id"],
                "revision": config["revision"],
                "device": device,
                "precision": precision,
                "backend": config["backend"],
                "weight_sha256": weight_hash,
                "weight_file": str(weight),
                "python": sys.version,
                "torch": str(torch.__version__),
                "laya_module": str(laya.__file__),
                "load_ms": (time.perf_counter() - started) * 1000,
                "license": config.get("license"),
                "code_revision": config.get("code_revision"),
                "laya_version": version("laya"),
                "transformers": version("transformers"),
                "code_module_sha256": hash_file(Path(laya.__file__)),
                "torch_cuda_runtime": torch.version.cuda,
                "onnxruntime": version("onnxruntime") if config["backend"] == "onnx" else None,
            }
            if source == "vision":
                metadata.update(profile)
                metadata["weight_reader"] = "pread" if os.name == "nt" else "mmap"
            if device == "cuda":
                torch.cuda.synchronize()
                metadata["gpu_name"] = torch.cuda.get_device_name(0)
        send({"ok": True, "metadata": metadata})
    except Exception as exc:
        send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        return
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                if device == "cuda":
                    torch.cuda.synchronize()
                if source == "vision":
                    from PIL import Image

                    image = Image.open(io.BytesIO(base64.b64decode(request["image"]))).convert(
                        "RGB"
                    )
                    raw = agent.predict(
                        {"image": image, "note": json.dumps(request["state"])},
                        request["questions"],
                        n_permutations=1,
                        batch_size=1,
                    )
                else:
                    raw = agent.system_one(request["state"], request["questions"])
                if device == "cuda":
                    torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated() / 1024**2 if device == "cuda" else None
            send({"ok": True, "raw_output": raw, "peak_vram_mb": peak})
        except Exception as exc:
            send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
