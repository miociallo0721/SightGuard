"""Export the local SightGuard SegFormer MiT-B0 and compare ONNX with PyTorch."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

CHECKPOINT = ROOT / "models" / "path_relevance" / "segformer_mit_b0_ade20k_160k.pth"
IMAGE = ROOT / "android" / "validation" / "validation.jpg"
OUTPUT = ROOT / "android" / "spatial-models" / "segformer_mit_b0_ade20k_256.onnx"
REPORT = ROOT / "android" / "segformer-validation" / "segformer_mit_b0_ade20k_256.json"
INPUT_BINARY = ROOT / "android" / "segformer-validation" / "segformer_mit_b0_ade20k_256.tensor.bin"
LOGITS_BINARY = ROOT / "android" / "segformer-validation" / "segformer_mit_b0_ade20k_256.logits.bin"
CHECKPOINT_SHA256 = "8ffa8fdac140ffaef9119150aca65e9fcb4e7a165e60b80be0102cc805d73a9c"
INPUT_SIZE = 256
OPSET = 17


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _installed_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not_installed"


def _rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except ImportError:
        return None


def _normalized_letterbox(image_bgr: Any, cv2: Any, np: Any) -> tuple[Any, dict[str, Any]]:
    """Create normalized RGB NCHW input with mean-colored center padding and no distortion."""
    source_height, source_width = image_bgr.shape[:2]
    scale = min(INPUT_SIZE / source_width, INPUT_SIZE / source_height)
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(
        image_rgb,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )
    rgb_float = resized.astype(np.float32) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = ((rgb_float - mean) / std).transpose(2, 0, 1)

    pad_left = (INPUT_SIZE - resized_width) // 2
    pad_top = (INPUT_SIZE - resized_height) // 2
    tensor = np.zeros((1, 3, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
    tensor[
        0,
        :,
        pad_top : pad_top + resized_height,
        pad_left : pad_left + resized_width,
    ] = normalized
    metadata = {
        "method": "center_letterbox_preserve_aspect",
        "sourceWidth": int(source_width),
        "sourceHeight": int(source_height),
        "targetWidth": INPUT_SIZE,
        "targetHeight": INPUT_SIZE,
        "resizedWidth": resized_width,
        "resizedHeight": resized_height,
        "scale": scale,
        "padLeft": pad_left,
        "padTop": pad_top,
        "padRight": INPUT_SIZE - resized_width - pad_left,
        "padBottom": INPUT_SIZE - resized_height - pad_top,
        "padValueAfterNormalization": [0.0, 0.0, 0.0],
        "interpolation": "cv2.INTER_AREA",
        "channelOrder": "RGB",
        "normalization": {
            "rangeBeforeNormalization": "[0,1]",
            "mean": mean.tolist(),
            "std": std.tolist(),
        },
        "tensorShape": list(tensor.shape),
        "tensorDtype": str(tensor.dtype),
        "tensorSha256": hashlib.sha256(tensor.tobytes(order="C")).hexdigest(),
    }
    return np.ascontiguousarray(tensor), metadata


def _timed_runs(call: Any, warmup_runs: int, timed_runs: int) -> dict[str, Any]:
    for _ in range(warmup_runs):
        call()
    samples = []
    for _ in range(timed_runs):
        started = time.perf_counter()
        call()
        samples.append((time.perf_counter() - started) * 1000.0)
    ordered = sorted(samples)
    return {
        "warmupRuns": warmup_runs,
        "timedRuns": timed_runs,
        "medianMs": ordered[len(ordered) // 2],
        "minimumMs": ordered[0],
        "maximumMs": ordered[-1],
        "samplesMs": samples,
    }


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def export_and_validate(
    *,
    checkpoint: Path = CHECKPOINT,
    image_path: Path = IMAGE,
    output_path: Path = OUTPUT,
    report_path: Path = REPORT,
    input_binary_path: Path = INPUT_BINARY,
    logits_binary_path: Path = LOGITS_BINARY,
    warmup_runs: int = 2,
    timed_runs: int = 5,
) -> dict[str, Any]:
    if warmup_runs < 0 or timed_runs < 1:
        raise ValueError("warmup_runs must be non-negative and timed_runs must be positive")
    checkpoint = checkpoint.resolve()
    image_path = image_path.resolve()
    output_path = output_path.resolve()
    report_path = report_path.resolve()
    input_binary_path = input_binary_path.resolve()
    logits_binary_path = logits_binary_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "schemaVersion": "sightguard.segformer-onnx-validation.v1",
        "status": "running",
        "startedAtUtc": datetime.now(UTC).isoformat(),
        "architecture": "SegFormer MiT-B0 ADE20K MMSegmentation checkpoint",
        "checkpoint": {
            "path": str(checkpoint),
            "sourceUrl": (
                "https://download.openmmlab.com/mmsegmentation/v0.5/segformer/"
                "segformer_mit-b0_512x512_160k_ade20k/"
                "segformer_mit-b0_512x512_160k_ade20k_20210726_101530-8ffa8fda.pth"
            ),
            "expectedSha256": CHECKPOINT_SHA256,
        },
        "image": {"path": str(image_path)},
        "output": {
            "path": str(output_path),
            "format": "ONNX",
            "opset": OPSET,
            "input": {"name": "pixel_values", "shape": [1, 3, INPUT_SIZE, INPUT_SIZE]},
            "outputName": "class_logits",
            "dynamicAxes": False,
            "outputSemantics": "raw ADE20K class logits at the decoder's 1/4 input resolution",
        },
        "software": {
            "python": platform.python_version(),
            "torch": _installed_version("torch"),
            "transformers": _installed_version("transformers"),
            "numpy": _installed_version("numpy"),
            "opencv": _installed_version("opencv-python"),
            "onnx": _installed_version("onnx"),
            "onnxruntime": _installed_version("onnxruntime"),
        },
        "runtime": {"device": "cpu", "processRssBeforeModelBytes": _rss_bytes()},
    }
    temporary_path: Path | None = None
    try:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"SegFormer checkpoint not found: {checkpoint}")
        checkpoint_hash = _sha256(checkpoint)
        report["checkpoint"]["sha256"] = checkpoint_hash
        if checkpoint_hash != CHECKPOINT_SHA256:
            raise ValueError(
                "Local checkpoint SHA-256 does not match the expected OpenMMLab artifact: "
                f"{checkpoint_hash}"
            )
        if not image_path.is_file():
            raise FileNotFoundError(f"Validation image not found: {image_path}")
        report["image"].update(
            {
                "sha256": _sha256(image_path),
                "source": "existing Ultralytics bundled bus.jpg validation photograph",
            }
        )

        missing = [
            name
            for name in ("torch", "transformers", "numpy", "opencv-python", "onnx", "onnxruntime")
            if _installed_version(name) == "not_installed"
        ]
        if missing:
            raise RuntimeError(f"Missing export dependencies: {', '.join(missing)}")

        import cv2
        import numpy as np
        import onnx
        import onnxruntime as ort
        import torch

        torch.set_num_threads(1)
        from sightguard.path_relevance.model import SegformerB0Perceptor

        if timed_runs < 1:
            raise ValueError("At least one timed run is required")
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"Could not decode validation image: {image_path}")
        input_tensor, preprocess = _normalized_letterbox(image_bgr, cv2, np)
        report["image"].update(
            {"width": int(image_bgr.shape[1]), "height": int(image_bgr.shape[0])}
        )
        report["preprocessing"] = preprocess

        perceptor = SegformerB0Perceptor(checkpoint, device="cpu", input_width=INPUT_SIZE)
        report["runtime"]["processRssAfterModelLoadBytes"] = _rss_bytes()

        class ExportWrapper(torch.nn.Module):
            def __init__(self, encoder: Any, decoder: Any) -> None:
                super().__init__()
                self.encoder = encoder
                self.decoder = decoder

            def forward(self, pixel_values: Any) -> Any:
                output = self.encoder(
                    pixel_values=pixel_values,
                    output_hidden_states=True,
                    return_dict=True,
                )
                return self.decoder(output.hidden_states)

        wrapper = ExportWrapper(perceptor.encoder, perceptor.decoder).eval()
        input_tensor_torch = torch.from_numpy(input_tensor)
        with tempfile.NamedTemporaryFile(
            prefix=f".{output_path.stem}.", suffix=".onnx", dir=output_path.parent, delete=False
        ) as temp_file:
            temporary_path = Path(temp_file.name)

        torch.onnx.export(
            wrapper,
            (input_tensor_torch,),
            str(temporary_path),
            input_names=["pixel_values"],
            output_names=["class_logits"],
            opset_version=OPSET,
            do_constant_folding=True,
            dynamic_axes=None,
            dynamo=False,
        )
        onnx.checker.check_model(str(temporary_path))

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            str(temporary_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        session_input = session.get_inputs()[0]
        if session_input.shape != [1, 3, INPUT_SIZE, INPUT_SIZE]:
            raise RuntimeError(f"ONNX input is not static RGB NCHW: {session_input.shape}")
        if session_input.name != "pixel_values":
            raise RuntimeError(f"Unexpected ONNX input name: {session_input.name}")

        ort_input = {session_input.name: input_tensor}
        with torch.inference_mode():
            torch_logits = wrapper(input_tensor_torch).cpu().numpy()
        ort_logits = session.run(["class_logits"], ort_input)[0]
        if torch_logits.shape != ort_logits.shape:
            raise RuntimeError(
                f"PyTorch and ONNX output shapes differ: {torch_logits.shape} vs {ort_logits.shape}"
            )
        if not np.isfinite(torch_logits).all() or not np.isfinite(ort_logits).all():
            raise RuntimeError("Model returned non-finite class logits")

        delta = np.abs(torch_logits.astype(np.float64) - ort_logits.astype(np.float64))
        torch_argmax = np.argmax(torch_logits, axis=1)
        ort_argmax = np.argmax(ort_logits, axis=1)
        argmax_match = float(np.mean(torch_argmax == ort_argmax))
        allclose = bool(np.allclose(torch_logits, ort_logits, rtol=1e-3, atol=1e-4))
        report["onnx"] = {
            "sha256": _sha256(temporary_path),
            "sizeBytes": temporary_path.stat().st_size,
            "checkerPassed": True,
            "inputName": session_input.name,
            "inputShape": session_input.shape,
            "inputType": session_input.type,
            "outputName": session.get_outputs()[0].name,
            "outputShape": list(ort_logits.shape),
            "outputType": session.get_outputs()[0].type,
            "providers": session.get_providers(),
        }
        report["comparison"] = {
            "sameInputTensor": True,
            "pytorchOutputShape": list(torch_logits.shape),
            "onnxOutputShape": list(ort_logits.shape),
            "maxAbsoluteError": float(delta.max(initial=0.0)),
            "meanAbsoluteError": float(delta.mean()),
            "allCloseRtol1e3Atol1e4": allclose,
            "argmaxPixelAgreement": argmax_match,
            "argmaxPixelAgreementPercent": 100.0 * argmax_match,
        }

        # Preserve exact desktop inputs and PyTorch reference logits for the
        # Android instrumentation comparison. Both files are contiguous
        # little-endian float32 in NCHW order.
        input_binary = np.ascontiguousarray(input_tensor.astype("<f4", copy=False))
        logits_binary = np.ascontiguousarray(torch_logits.astype("<f4", copy=False))
        input_binary_path.parent.mkdir(parents=True, exist_ok=True)
        logits_binary_path.parent.mkdir(parents=True, exist_ok=True)
        input_binary.tofile(input_binary_path)
        logits_binary.tofile(logits_binary_path)
        report["referenceBinaries"] = {
            "byteOrder": "little-endian",
            "dtype": "float32",
            "layout": "contiguous NCHW",
            "inputTensor": {
                "path": str(input_binary_path),
                "shape": list(input_binary.shape),
                "sizeBytes": input_binary_path.stat().st_size,
                "sha256": _sha256(input_binary_path),
            },
            "pytorchLogits": {
                "path": str(logits_binary_path),
                "shape": list(logits_binary.shape),
                "sizeBytes": logits_binary_path.stat().st_size,
                "sha256": _sha256(logits_binary_path),
            },
        }

        def run_pytorch() -> Any:
            with torch.inference_mode():
                return wrapper(input_tensor_torch)

        def run_onnx() -> Any:
            return session.run(["class_logits"], ort_input)[0]

        report["timing"] = {
            "pytorchCpu": _timed_runs(run_pytorch, warmup_runs, timed_runs),
            "onnxruntimeCpu": _timed_runs(run_onnx, warmup_runs, timed_runs),
        }
        report["runtime"]["processRssAfterOnnxSessionBytes"] = _rss_bytes()
        os.replace(temporary_path, output_path)
        temporary_path = None
        report["output"]["sha256"] = report["onnx"]["sha256"]
        report["output"]["sizeBytes"] = output_path.stat().st_size
        report["status"] = "completed"
        report["finishedAtUtc"] = datetime.now(UTC).isoformat()
        _write_report(report_path, report)
        return report
    except Exception as exc:
        report["status"] = "failed"
        report["finishedAtUtc"] = datetime.now(UTC).isoformat()
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        _write_report(report_path, report)
        raise
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--image", type=Path, default=IMAGE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--input-binary", type=Path, default=INPUT_BINARY)
    parser.add_argument("--logits-binary", type=Path, default=LOGITS_BINARY)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--timed-runs", type=int, default=5)
    args = parser.parse_args()
    report = export_and_validate(
        checkpoint=args.checkpoint,
        image_path=args.image,
        output_path=args.output,
        report_path=args.report,
        input_binary_path=args.input_binary,
        logits_binary_path=args.logits_binary,
        warmup_runs=args.warmup_runs,
        timed_runs=args.timed_runs,
    )
    comparison = report["comparison"]
    print(
        f"ONNX export complete: shape={report['onnx']['outputShape']}, "
        f"max_abs={comparison['maxAbsoluteError']:.6g}, "
        f"argmax_match={comparison['argmaxPixelAgreementPercent']:.4f}%"
    )


if __name__ == "__main__":
    main()
