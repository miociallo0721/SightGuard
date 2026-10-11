"""Export pinned local Depth Anything V2 Small; real CPU Torch/ORT parity."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from export_segformer import _installed_version, _sha256, _timed_runs, _write_report

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
SIZE = 280
REPORT = ROOT / "android/parity-validation/v03-depth-python/depth_validation.json"
MANIFEST = ROOT / "android/app/src/main/assets/depth_manifest.json"


def write_manifest(report: dict[str, Any]) -> dict[str, Any]:
    """Package the verified export contract without importing model runtimes."""
    if report.get("status") != "completed" or not report["comparison"]["passed"]:
        raise ValueError("manifest requires completed real numeric validation")
    onnx = report["onnx"]
    checkpoint = report["checkpoint"]
    asset = MANIFEST.parent / "depth_anything_v2_small_280.onnx"
    if not asset.is_file() or _sha256(asset) != onnx["sha256"]:
        raise ValueError("packaged ONNX does not match validated export")
    preprocess = report["preprocessing"]
    manifest = {
        "schema_version": "sightguard.depth-model-manifest.v1",
        "version": "sightguard.depth.v03",
        "model_id": checkpoint["model_id"],
        "model_revision": checkpoint["model_revision"],
        "model_asset": asset.name,
        "onnx_sha256": onnx["sha256"],
        "onnx_size_bytes": onnx["sizeBytes"],
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "config_sha256": checkpoint["config_sha256"],
        "processor_sha256": checkpoint["processor_sha256"],
        "checkpoint_source": checkpoint["checkpoint_url"],
        "license": "Apache-2.0",
        "license_basis": checkpoint["license"],
        "readme_sha256": checkpoint["readmeSha256"],
        "opset": onnx["opset"],
        "input_size": SIZE,
        "input": {
            "name": onnx["inputName"],
            "shape": onnx["inputShape"],
            "dtype": "float32",
            "layout": "NCHW",
            "channel_order": "RGB",
        },
        "output": {
            "name": onnx["outputName"],
            "shape": onnx["outputShape"],
            "dtype": "float32",
            "layout": "NHW",
            "normalization": "none",
        },
        "dynamic_axes": False,
        "depth_semantics": report["semantics"]["meaning"],
        "depth_scale_type": "frame_relative_uncalibrated",
        "metric_distance": False,
        "semantics_source": report["semantics"]["paperSource"],
        "preprocessing": {
            "method": "center_letterbox_preserve_aspect",
            "target_width": SIZE,
            "target_height": SIZE,
            "scale": "min(280/sourceWidth, 280/sourceHeight)",
            "resize_rounding": "nearest integer (Python round in reference)",
            "pad_left_top": "floor((280-resizedWidth)/2), floor((280-resizedHeight)/2)",
            "channel_order": "RGB",
            "rescale_factor": 1 / 255,
            "mean": preprocess["normalization"]["mean"],
            "std": preprocess["normalization"]["std"],
            "padding_after_normalization": [0.0, 0.0, 0.0],
            "python_interpolation": "OpenCV INTER_AREA",
            "android_interpolation": "Bitmap bilinear; bitmap preprocessing differs",
            "numeric_parity_scope": "exact serialized float32 input tensor",
            "pc_difference": preprocess["pcDifference"],
        },
        "spatial_restore": "crop letterbox padding then resize to exact source frame",
        "tool_versions": {"python": report["python"], **report["software"]},
        "validation": {
            "report_schema": report["schemaVersion"],
            "report_sha256": _sha256(REPORT),
            "finished_at_utc": report["finishedAtUtc"],
            "comparison": report["comparison"],
            "input_tensor_sha256": report["inputTensorSha256"],
        },
    }
    _write_report(MANIFEST, manifest)
    return manifest


def export_and_validate(warmup_runs: int = 2, timed_runs: int = 5) -> dict[str, Any]:
    """Keep original checkpoint and report failures without fabricated output."""
    if warmup_runs < 0 or timed_runs < 1:
        raise ValueError("invalid run counts")
    directory = ROOT / "android/parity-validation/v03-depth-python"
    directory.mkdir(parents=True, exist_ok=True)
    output = ROOT / "android/spatial-models/depth_anything_v2_small_280.onnx"
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path = directory / "depth_validation.json"
    report: dict[str, Any] = {
        "schemaVersion": "sightguard.depth-onnx-validation.v1",
        "status": "running",
        "failure": None,
        "startedAtUtc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "byteOrder": "little-endian",
        "tensorLayout": "input NCHW; output NHW",
        "reproduction": (
            ". android/tools/task_environment.ps1; "
            '$env:PYTHONPATH = "$PWD/.venv/Lib/site-packages"; '
            "& android/.export-venv/Scripts/python.exe android/tools/export_depth.py"
        ),
    }
    _write_report(report_path, report)
    try:
        import cv2
        import numpy as np
        import onnx
        import onnxruntime as ort
        import torch

        from sightguard.perception.depth.model import DepthEstimator
        from sightguard.perception.depth.schema import DepthModelConfig

        torch.set_num_threads(1)
        config = DepthModelConfig(
            checkpoint_dir=ROOT / "models/depth_anything_v2_small", device="cpu"
        )
        estimator = DepthEstimator(config)
        estimator.initialize()
        report["checkpoint"] = {
            **config.to_dict(),
            **estimator.metadata,
            "license": "Apache-2.0 (local pinned README front matter)",
            "readmeSha256": _sha256(config.checkpoint_dir / "README.md"),
        }
        image = ROOT / "android/validation/validation.jpg"
        bgr = cv2.imread(str(image))
        if bgr is None:
            raise ValueError("validation image failed to decode")
        h, w = bgr.shape[:2]
        scale = min(SIZE / w, SIZE / h)
        rw, rh = round(w * scale), round(h * scale)
        left, top = (SIZE - rw) // 2, (SIZE - rh) // 2
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (rw, rh), interpolation=cv2.INTER_AREA)
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        tensor = np.zeros((1, 3, SIZE, SIZE), dtype=np.float32)
        tensor[0, :, top : top + rh, left : left + rw] = (
            (resized.astype(np.float32) / 255.0 - mean) / std
        ).transpose(2, 0, 1)
        report["preprocessing"] = {
            "method": "center_letterbox_preserve_aspect",
            "sourceWidth": w,
            "sourceHeight": h,
            "targetWidth": SIZE,
            "targetHeight": SIZE,
            "resizedWidth": rw,
            "resizedHeight": rh,
            "scale": scale,
            "padLeft": left,
            "padTop": top,
            "padRight": SIZE - rw - left,
            "padBottom": SIZE - rh - top,
            "padValueAfterNormalization": [0, 0, 0],
            "channelOrder": "RGB",
            "interpolation": "cv2.INTER_AREA",
            "normalization": {"mean": mean.tolist(), "std": std.tolist()},
            "pcDifference": (
                "PC DPTImageProcessor preserves aspect by variable dimensions; "
                "Android uses static mean-padded letterbox"
            ),
        }
        report["image"] = {
            "sha256": _sha256(image),
            "source": "existing Ultralytics bus photograph",
            "path": str(image),
        }

        class Wrapper(torch.nn.Module):
            def __init__(self, model: Any) -> None:
                super().__init__()
                self.model = model

            def forward(self, pixel_values: Any) -> Any:
                return self.model(pixel_values=pixel_values).predicted_depth

        wrapper = Wrapper(estimator.model).eval()
        x = torch.from_numpy(tensor)
        started = time.perf_counter()
        torch.onnx.export(
            wrapper,
            (x,),
            str(output),
            input_names=["pixel_values"],
            output_names=["predicted_depth"],
            opset_version=17,
            dynamic_axes=None,
            dynamo=False,
        )
        onnx.checker.check_model(str(output))
        report["exportMs"] = (time.perf_counter() - started) * 1000
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        started = time.perf_counter()
        session = ort.InferenceSession(str(output), options, providers=["CPUExecutionProvider"])
        report["ortLoadMs"] = (time.perf_counter() - started) * 1000
        with torch.inference_mode():
            reference = wrapper(x).numpy()
        actual = session.run(None, {"pixel_values": tensor})[0]
        if reference.shape != (1, SIZE, SIZE) or actual.shape != reference.shape:
            raise RuntimeError("unexpected output shape")
        if not np.isfinite(reference).all() or not np.isfinite(actual).all():
            raise RuntimeError("nonfinite depth")
        error = np.abs(reference.astype(np.float64) - actual.astype(np.float64))
        passed = bool(np.allclose(reference, actual, atol=1e-4, rtol=1e-3))
        report["comparison"] = {
            "sameInputTensor": True,
            "maxAbsoluteError": float(error.max()),
            "meanAbsoluteError": float(error.mean()),
            "rtol": 1e-3,
            "atol": 1e-4,
            "passed": passed,
        }
        report["onnx"] = {
            "path": str(output),
            "sha256": _sha256(output),
            "sizeBytes": output.stat().st_size,
            "opset": 17,
            "checkerPassed": True,
            "inputName": "pixel_values",
            "inputShape": session.get_inputs()[0].shape,
            "outputName": "predicted_depth",
            "outputShape": list(actual.shape),
            "dtype": "float32",
            "providers": session.get_providers(),
        }
        report["semantics"] = {
            "meaning": "relative_inverse_depth_larger_is_nearer",
            "scale": "frame_relative_uncalibrated",
            "metricDistance": False,
            "paperSource": "https://arxiv.org/html/2406.09414v2#S5.SS2",
            "paperBasis": "Section 5.2 states affine-invariant inverse depth",
            "basis": (
                "pinned PC schema SEMANTICS and official DepthAnything relative model head; "
                "no scene metric calibration"
            ),
            "min": float(reference.min()),
            "max": float(reference.max()),
            "mean": float(reference.mean()),
        }
        report["referenceBinaries"] = {}
        for name, value in (
            ("depth_input.tensor.bin", tensor),
            ("depth_output.tensor.bin", reference),
            ("depth_ort_output.tensor.bin", actual),
        ):
            path = directory / name
            np.asarray(value, dtype="<f4").tofile(path)
            report["referenceBinaries"][name] = {
                "sha256": _sha256(path),
                "shape": list(value.shape),
                "sizeBytes": path.stat().st_size,
            }
        shutil.copy2(image, directory / "depth_input.jpg")

        def torch_run() -> Any:
            with torch.inference_mode():
                return wrapper(x)

        report["timing"] = {
            "pytorchCpu": _timed_runs(torch_run, warmup_runs, timed_runs),
            "onnxruntimeCpu": _timed_runs(
                lambda: session.run(None, {"pixel_values": tensor}), warmup_runs, timed_runs
            ),
        }
        report["software"] = {
            name: _installed_version(name)
            for name in ("torch", "transformers", "onnx", "onnxruntime", "numpy", "opencv-python")
        }
        report["inputTensorSha256"] = hashlib.sha256(tensor.tobytes()).hexdigest()
        if not passed:
            raise RuntimeError("Torch/ORT numeric parity failed")
        report["status"] = "completed"
        report["finishedAtUtc"] = datetime.now(UTC).isoformat()
        _write_report(report_path, report)
        shutil.copy2(output, MANIFEST.parent / output.name)
        write_manifest(report)
        return report
    except Exception as exc:
        report.update(status="failed", failure={"type": type(exc).__name__, "message": str(exc)})
        report["finishedAtUtc"] = datetime.now(UTC).isoformat()
        _write_report(report_path, report)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--timed-runs", type=int, default=5)
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="regenerate packaged manifest from completed validation; no inference",
    )
    arguments = parser.parse_args()
    if arguments.manifest_only:
        result = write_manifest(json.loads(REPORT.read_text(encoding="utf-8")))
        print(json.dumps({"manifest": str(MANIFEST), "onnx_sha256": result["onnx_sha256"]}))
    else:
        result = export_and_validate(arguments.warmup_runs, arguments.timed_runs)
        print(json.dumps(result["comparison"]))
