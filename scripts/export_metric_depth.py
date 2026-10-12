"""Export official static aspect-preserving metric graphs, then deduplicate shared weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from sightguard.research.resources import check_resources, lower_own_priority
from sightguard.research.settings import research_cache_root

ROOT = Path(__file__).resolve().parents[1]
MODELS = research_cache_root()
SHAPES = [(280, 378), (378, 280), (280, 504), (504, 280)]


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def export(height: int, width: int, image_path: Path, output: Path) -> None:
    import torch

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(MODELS / "Depth-Anything-V2/metric_depth"))
    from depth_anything_v2.dpt import DepthAnythingV2

    check_resources(output_dir=output)
    model = DepthAnythingV2(
        encoder="vits", features=64, out_channels=[48, 96, 192, 384], max_depth=20
    ).eval()
    weights = MODELS / "depth_anything_v2_metric_hypersim_vits.pth"
    model.load_state_dict(torch.load(weights, weights_only=True, map_location="cpu"), strict=True)
    image = cv2.imread(str(image_path))
    # Rotate/reframe the public diagnostic image only to create each source aspect.
    # The actual model tensor uses the official Resize/Normalize/PrepareForNet contract.
    image = cv2.resize(image, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
    tensor, _ = model.image2tensor(image, 280)
    tensor = tensor.cpu()
    if tuple(tensor.shape) != (1, 3, height, width):
        raise ValueError(f"official preprocessing shape mismatch: {tensor.shape}")
    start = time.perf_counter()
    with torch.inference_mode():
        expected = model(tensor).numpy()
        torch.onnx.export(
            model,
            tensor,
            str(output / f"metric_{height}x{width}.onnx"),
            input_names=["pixel_values"],
            output_names=["predicted_metric_z_m"],
            opset_version=17,
            dynamo=False,
            do_constant_folding=True,
        )
    np.savez_compressed(
        output / f"parity_{height}x{width}.npz",
        input=tensor.numpy(),
        expected=expected,
        source_image=image,
    )
    print(
        json.dumps(
            {
                "export_shape": [1, 3, height, width],
                "export_ms": (time.perf_counter() - start) * 1000,
            }
        ),
        flush=True,
    )


def pack(output: Path) -> None:
    import onnx

    offsets: dict[str, tuple[int, int]] = {}
    assets = ROOT / "android/app/src/main/assets/metric_research"
    assets.mkdir(parents=True, exist_ok=True)
    weights = assets / "metric_weights.bin"
    graphs = []
    with weights.open("wb") as stream:
        for height, width in SHAPES:
            check_resources(output_dir=output)
            graph = onnx.load(str(output / f"metric_{height}x{width}.onnx"))
            for tensor in graph.graph.initializer:
                if len(tensor.raw_data) < 4096:
                    continue
                data = tensor.raw_data
                digest = hashlib.sha256(data).hexdigest()
                if digest not in offsets:
                    offsets[digest] = (stream.tell(), len(data))
                    stream.write(data)
                offset, length = offsets[digest]
                tensor.ClearField("raw_data")
                tensor.data_location = onnx.TensorProto.EXTERNAL
                tensor.ClearField("external_data")
                for key, value in [
                    ("location", "metric_weights.bin"),
                    ("offset", str(offset)),
                    ("length", str(length)),
                ]:
                    item = tensor.external_data.add()
                    item.key, item.value = key, value
            path = assets / f"metric_{height}x{width}.onnx"
            onnx.save_model(graph, str(path))
            graphs.append(
                {
                    "file": path.name,
                    "sha256": sha256(path),
                    "input_shape": [1, 3, height, width],
                    "output_shape": [1, height, width],
                }
            )
            del graph
    manifest = {
        "schema_version": "sightguard.metric-depth-model.v1",
        "shadow_only": True,
        "default_enabled": False,
        "automatic_speech": False,
        "model_id": "depth-anything/Depth-Anything-V2-Metric-Hypersim-Small",
        "revision": "3bc65d4e14a6786a61acec16453c50e12bf5f338",
        "checkpoint_sha256": "b782898d8a3e8be1f639de33837ed85e9b4b73e40f8f5e5cd99067588d722545",
        "official_source_git_sha": "a561b849ebae10a6f5ef49e26c83cbbcd36c71bf",
        "license": "Apache-2.0",
        "quantity": "predicted_optical_axis_z_m",
        "max_depth_m": 20,
        "opset": 17,
        "input_size": 280,
        "preprocessing": {
            "resize": "official_lower_bound_preserve_aspect_multiple_14",
            "interpolation": "OpenCV_INTER_CUBIC_half_pixel_a_minus_0_75",
            "channel_order": "RGB",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "padding": "NONE",
            "unsupported_aspect": "ABSTAIN_no_warp_or_letterbox",
        },
        "shared_weights": {
            "file": weights.name,
            "sha256": sha256(weights),
            "bytes": weights.stat().st_size,
        },
        "graphs": graphs,
    }
    (assets / "metric_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("shared_weights_MB", weights.stat().st_size / 1024**2, flush=True)


def parity(output: Path) -> None:
    import onnxruntime as ort

    records = []
    assets = ROOT / "android/app/src/main/assets/metric_research"
    for height, width in SHAPES:
        check_resources(output_dir=output)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.enable_cpu_mem_arena = False
        start = time.perf_counter()
        session = ort.InferenceSession(
            str(assets / f"metric_{height}x{width}.onnx"),
            options,
            providers=["CPUExecutionProvider"],
        )
        load_ms = (time.perf_counter() - start) * 1000
        with np.load(output / f"parity_{height}x{width}.npz") as data:
            start = time.perf_counter()
            actual = session.run(None, {"pixel_values": data["input"]})[0]
            elapsed = (time.perf_counter() - start) * 1000
            difference = np.abs(actual - data["expected"])
            record = {
                "shape": [height, width],
                "load_ms": load_ms,
                "inference_ms": elapsed,
                "max_error_m": float(difference.max()),
                "mean_error_m": float(difference.mean()),
                "passed": bool(np.allclose(actual, data["expected"], rtol=1e-4, atol=1e-4)),
                "ort_version": ort.__version__,
                "providers": session.get_providers(),
                "scope": "PC_CPU_not_Android_device_validation",
            }
            records.append(record)
            # Exact common tensor and expected native depth for Android ORT parity.
            fixture = ROOT / "android/app/src/androidTest/assets/metric_research"
            fixture.mkdir(parents=True, exist_ok=True)
            data["input"].astype("<f4").tofile(fixture / f"input_{height}x{width}.bin")
            data["expected"].astype("<f4").tofile(fixture / f"expected_{height}x{width}.bin")
            cv2.imwrite(str(fixture / f"source_{height}x{width}.png"), data["source_image"])
        print(json.dumps(record), flush=True)
        del session
    (output / "onnx_parity.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    if not all(record["passed"] for record in records):
        raise ValueError("PyTorch_ONNX_parity_failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["export", "pack", "parity"], required=True)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--output", type=Path, default=MODELS / "export")
    args = parser.parse_args()
    lower_own_priority()
    cv2.setNumThreads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.phase == "export":
        export(args.height, args.width, args.image, args.output)
    elif args.phase == "pack":
        pack(args.output)
    else:
        parity(args.output)


if __name__ == "__main__":
    main()
