"""Export YOLO26n to the Android app's fixed-size ONNX contract and validate it."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
import ultralytics
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / "models" / "yolo26n.pt"
ASSETS = ROOT / "android" / "app" / "src" / "main" / "assets"
VALIDATION = ROOT / "android" / "validation"
ANDROID_TEST_ASSETS = ROOT / "android" / "app" / "src" / "androidTest" / "assets"
IMAGE_SIZE = 320
OPSET = 17


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def letterbox_rgb(image_bgr: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """RGB centered letterbox using half-pixel bilinear resize without antialiasing."""
    height, width = image_bgr.shape[:2]
    scale = min(IMAGE_SIZE / width, IMAGE_SIZE / height)
    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(
        rgb, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
    )
    pad_left = (IMAGE_SIZE - resized_width) // 2
    pad_top = (IMAGE_SIZE - resized_height) // 2
    canvas = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 114, dtype=np.uint8)
    canvas[pad_top : pad_top + resized_height, pad_left : pad_left + resized_width] = resized
    tensor = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    metadata = {
        "source_width": int(width),
        "source_height": int(height),
        "target_width": IMAGE_SIZE,
        "target_height": IMAGE_SIZE,
        "scale": scale,
        "actual_scale_x": resized_width / width,
        "actual_scale_y": resized_height / height,
        "resized_width": resized_width,
        "resized_height": resized_height,
        "pad_left": pad_left,
        "pad_top": pad_top,
        "pad_right": IMAGE_SIZE - resized_width - pad_left,
        "pad_bottom": IMAGE_SIZE - resized_height - pad_top,
        "pad_rgb": [114, 114, 114],
    }
    return tensor, metadata


def flatten_tensors(value: Any) -> list[np.ndarray]:
    if isinstance(value, torch.Tensor):
        return [value.detach().cpu().numpy()]
    if isinstance(value, (tuple, list)):
        return [array for item in value for array in flatten_tensors(item)]
    if isinstance(value, dict):
        return [array for item in value.values() for array in flatten_tensors(item)]
    raise TypeError(f"Unsupported model output type: {type(value)!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Reuse the existing Android ONNX asset without exporting or changing its hash.",
    )
    args = parser.parse_args()
    if not WEIGHTS.is_file():
        raise FileNotFoundError(f"Missing weights: {WEIGHTS}")
    ASSETS.mkdir(parents=True, exist_ok=True)
    VALIDATION.mkdir(parents=True, exist_ok=True)
    ANDROID_TEST_ASSETS.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(WEIGHTS), task="detect")
    model_path = ASSETS / "yolo26n.onnx"
    if not args.validate_only:
        model.export(
            format="onnx",
            imgsz=IMAGE_SIZE,
            batch=1,
            dynamic=False,
            simplify=False,
            opset=OPSET,
            nms=False,
            device="cpu",
        )
        exported = WEIGHTS.with_suffix(".onnx")
        if not exported.is_file():
            raise FileNotFoundError(f"Ultralytics did not create expected export: {exported}")
        shutil.copy2(exported, model_path)
    elif not model_path.is_file():
        raise FileNotFoundError(f"Cannot validate; ONNX asset is missing: {model_path}")

    # Prefer the package-bundled bus sample, so validation never downloads media.
    image_path = VALIDATION / "validation.jpg"
    test_image_path = ANDROID_TEST_ASSETS / "validation.jpg"
    if args.validate_only and not image_path.is_file() and test_image_path.is_file():
        shutil.copy2(test_image_path, image_path)
    elif not args.validate_only:
        source_image = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"
        if not source_image.is_file():
            raise FileNotFoundError(f"Missing bundled Ultralytics validation image: {source_image}")
        shutil.copy2(source_image, image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Missing validation image: {image_path}")
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read validation image: {image_path}")
    input_tensor, prep_info = letterbox_rgb(bgr)

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()
    onnx_outputs = session.run(None, {input_meta.name: input_tensor})

    # Call the exported YOLO network's PyTorch graph on the exact same tensor.
    model.model.eval()
    # Match ONNX's native end-to-end YOLO26 head. Outside export mode
    # Ultralytics returns training-style multi-scale predictions by default.
    head = model.model.model[-1]
    head.end2end = True
    head.export = True
    with torch.inference_mode():
        torch_output = model.model(torch.from_numpy(input_tensor))
    torch_outputs = flatten_tensors(torch_output)
    if len(torch_outputs) != len(onnx_outputs):
        raise RuntimeError(
            f"Output count differs: PyTorch={len(torch_outputs)}, ONNX={len(onnx_outputs)}"
        )
    comparisons = []
    for index, (pt, onnx_value) in enumerate(zip(torch_outputs, onnx_outputs, strict=True)):
        if pt.shape != onnx_value.shape:
            raise RuntimeError(f"Output {index} shape differs: {pt.shape} vs {onnx_value.shape}")
        delta = np.abs(pt.astype(np.float64) - onnx_value.astype(np.float64))
        comparisons.append(
            {
                "output_index": index,
                "shape": list(onnx_value.shape),
                "max_absolute_error": float(delta.max(initial=0.0)),
                "mean_absolute_error": float(delta.mean()),
                "allclose_rtol_1e-3_atol_1e-4": bool(
                    np.allclose(pt, onnx_value, rtol=1e-3, atol=1e-4)
                ),
            }
        )

    def detections_above_threshold(
        output: np.ndarray, threshold: float = 0.25
    ) -> list[dict[str, Any]]:
        rows = output[0]
        selected = rows[rows[:, 4] >= threshold][:20]
        return [
            {
                "xyxy": [float(value) for value in row[:4]],
                "score": float(row[4]),
                "class_index": int(row[5]),
                "class_name": model.names[int(row[5])],
            }
            for row in selected
        ]

    def raw_output_summary(output: np.ndarray) -> dict[str, Any]:
        rows = output[0]
        ranked = rows[np.argsort(rows[:, 4])[::-1]]
        return {
            "row_count": int(rows.shape[0]),
            "count_at_or_above_0_25": int(np.count_nonzero(rows[:, 4] >= 0.25)),
            "android_min_score": 0.35,
            "count_at_or_above_android_min_score": int(
                np.count_nonzero(rows[:, 4] >= 0.35)
            ),
            "top_10_by_score": [
                {
                    "rank": index + 1,
                    "score": float(row[4]),
                    "class_index": int(row[5]),
                    "class_name": model.names[int(row[5])],
                    "xyxy": [float(value) for value in row[:4]],
                }
                for index, row in enumerate(ranked[:10])
            ],
        }

    graph = onnx.load(str(model_path))
    onnx.checker.check_model(graph)
    metadata = {
        "format_version": 1,
        "model": "YOLO26n",
        "task": "detect",
        "weights_sha256": sha256(WEIGHTS),
        "onnx_sha256": sha256(model_path),
        "input": {
            "name": input_meta.name,
            "shape": list(input_meta.shape),
            "dtype": input_meta.type,
            "layout": "NCHW",
            "color_order": "RGB",
            "normalization": "float32 / 255.0",
            "preprocessing": (
                "centered letterbox to 320x320; OpenCV INTER_LINEAR; "
                "pad RGB=(114,114,114)"
            ),
        },
        "outputs": [
            {"name": item.name, "shape": list(item.shape), "dtype": item.type}
            for item in output_meta
        ],
        "preprocessing_details": {
            "source_order": "OpenCV decoded BGR converted to RGB",
            "resize_ratio": "min(320/source_width, 320/source_height)",
            "scaled_dimensions": "round(source_dimension * ratio)",
            "interpolation": "OpenCV cv2.INTER_LINEAR (half-pixel bilinear; no antialiasing)",
            "android_interpolation": (
                "Bitmap.createScaledBitmap(..., filter=true); expected to be closer to "
                "OpenCV bilinear than antialiased Pillow downsampling, though not pixel-identical"
            ),
            "cross_runtime_tolerance": {
                "comparison": "object-level; pixel-identical resized images are not assumed",
                "minimum_bbox_iou": 0.80,
                "maximum_confidence_delta": 0.15,
            },
            "padding": "centered; integer left/top=floor(remaining/2); RGB=(114,114,114)",
            "tensor": "RGB uint8 to float32, NCHW, divide by 255",
        },
        "output_semantics": (
            "YOLO26 native end-to-end [x1,y1,x2,y2,score,class_index], "
            "maximum 300 detections; no extra NMS during export."
        ),
        "class_names": {str(key): value for key, value in model.names.items()},
        "export": {
            "ultralytics": ultralytics.__version__,
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "numpy": np.__version__,
            "opset": OPSET,
            "imgsz": IMAGE_SIZE,
            "dynamic": False,
            "simplify": False,
            "nms": False,
        },
    }
    (ASSETS / "model_manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    validation = {
        "format_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "validation_image": {
            "file": image_path.name,
            "sha256": sha256(image_path),
            "width": int(bgr.shape[1]),
            "height": int(bgr.shape[0]),
            "source": "ultralytics package bundled assets/bus.jpg",
        },
        "preprocessing": prep_info,
        "preprocessing_method": {
            "source_order": "OpenCV decoded BGR converted to RGB",
            "resize_ratio": "min(320/source_width, 320/source_height)",
            "scaled_dimensions": "round(source_dimension * ratio)",
            "interpolation": "OpenCV cv2.INTER_LINEAR (half-pixel bilinear; no antialiasing)",
            "android_interpolation": (
                "Bitmap.createScaledBitmap(..., filter=true); expected to be closer to "
                "OpenCV bilinear than antialiased Pillow downsampling, though not pixel-identical"
            ),
            "cross_runtime_tolerance": {
                "comparison": "object-level; pixel-identical resized images are not assumed",
                "minimum_bbox_iou": 0.80,
                "maximum_confidence_delta": 0.15,
            },
            "padding": "centered; integer left/top=floor(remaining/2); RGB=(114,114,114)",
            "tensor": "RGB uint8 to float32, NCHW, divide by 255",
        },
        "input": {"name": input_meta.name, "shape": list(input_meta.shape)},
        "outputs": [
            {"name": item.name, "shape": list(item.shape), "dtype": item.type}
            for item in output_meta
        ],
        "comparison": comparisons,
        "raw_output_summary": {
            "pytorch": raw_output_summary(torch_outputs[0]),
            "onnxruntime": raw_output_summary(onnx_outputs[0]),
        },
        "detections": {
            "confidence_threshold": 0.25,
            "maximum_records": 20,
            "pytorch": detections_above_threshold(torch_outputs[0]),
            "onnxruntime": detections_above_threshold(onnx_outputs[0]),
        },
        "versions": {
            "python": sys.version,
            "ultralytics": ultralytics.__version__,
            "torch": torch.__version__,
            "opencv": cv2.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
        },
        "weights_sha256": sha256(WEIGHTS),
        "onnx_sha256": sha256(model_path),
    }
    validation_path = VALIDATION / "python_model_validation.json"
    validation_path.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    # Package only the comparison reference and image in the instrumentation APK.
    # Keep them out of main/assets so they never ship with the product app.
    shutil.copy2(image_path, ANDROID_TEST_ASSETS / "validation.jpg")
    shutil.copy2(validation_path, ANDROID_TEST_ASSETS / "validation_reference.json")
    summary = {
        "input": metadata["input"],
        "outputs": metadata["outputs"],
        "validation": comparisons,
        "raw_output_summary": validation["raw_output_summary"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
