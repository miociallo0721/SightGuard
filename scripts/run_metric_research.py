"""Serial real ONNX semantics/YOLO and official metric-depth evaluation on fixed samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from sightguard.research.resources import check_resources, lower_own_priority
from sightguard.research.settings import load_research_settings, research_cache_root
from sightguard.research.wall_metric import (
    MetricDepthField,
    WallMetricConfig,
    WallMetricEstimator,
    depth_errors,
)
from sightguard.scene.spatial_grid.schema import FrameKey

ROOT = Path(__file__).resolve().parents[1]
MODELS = research_cache_root()
DELIVERY = ROOT / "android/deliveries/metric-corridor-research"


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def letterbox(image: np.ndarray, size: int, *, normalized: bool) -> tuple[np.ndarray, tuple]:
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    rw, rh = round(width * scale), round(height * scale)
    dx, dy = (size - rw) // 2, (size - rh) // 2
    rgb = cv2.resize(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        (rw, rh),
        interpolation=cv2.INTER_AREA if normalized else cv2.INTER_LINEAR,
    )
    if normalized:
        values = rgb.astype(np.float32) / 255 - np.array([0.485, 0.456, 0.406], np.float32)
        values /= np.array([0.229, 0.224, 0.225], np.float32)
        tensor = np.zeros((1, 3, size, size), np.float32)
    else:
        values = rgb.astype(np.float32) / 255
        tensor = np.full((1, 3, size, size), 114 / 255, np.float32)
    tensor[0, :, dy : dy + rh, dx : dx + rw] = values.transpose(2, 0, 1)
    return tensor, (scale, dx, dy, rw, rh)


def guard(output: Path, settings: dict) -> dict:
    return check_resources(
        minimum_mb=settings["resources"]["minimum_free_ram_mb"],
        stop_utc=settings["resources"]["operations_stop_utc"],
        output_dir=output,
    )


def semantics(samples: list[dict], output: Path, settings: dict) -> None:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = settings["resources"]["ort_threads"]
    options.inter_op_num_threads = 1
    options.enable_cpu_mem_arena = False
    assets = ROOT / "android/app/src/main/assets"
    seg = ort.InferenceSession(
        str(assets / "segformer_mit_b0_ade20k_256.onnx"),
        options,
        providers=["CPUExecutionProvider"],
    )
    yolo = ort.InferenceSession(
        str(assets / "yolo26n.onnx"), options, providers=["CPUExecutionProvider"]
    )
    metadata = {
        "segformer_sha256": sha256(assets / "segformer_mit_b0_ade20k_256.onnx"),
        "yolo_sha256": sha256(assets / "yolo26n.onnx"),
        "ort": ort.__version__,
    }
    cache = output / "semantics"
    cache.mkdir(parents=True, exist_ok=True)
    with (output / "semantic_results.jsonl").open("w", encoding="utf-8") as records:
        for sample in samples:
            resources = guard(output, settings)
            start = time.perf_counter()
            image = cv2.imread(sample["rgb_path"])
            h, w = image.shape[:2]
            tensor, (_, dx, dy, rw, rh) = letterbox(image, 256, normalized=True)
            infer_start = time.perf_counter()
            logits = seg.run(None, {seg.get_inputs()[0].name: tensor})[0][0]
            seg_ms = (time.perf_counter() - infer_start) * 1000
            if logits.shape != (150, 64, 64):
                raise ValueError("SegFormer ADE20K model contract mismatch")
            classes = logits.argmax(axis=0).astype(np.uint8)
            confidence = 1 / np.exp(logits - logits.max(axis=0)).sum(axis=0)
            # Native 64x64 cell predictions, excluding padding, mapped to source pixels.
            crop = (
                slice(round(dy / 4), round((dy + rh) / 4)),
                slice(round(dx / 4), round((dx + rw) / 4)),
            )
            classes = cv2.resize(classes[crop], (w, h), interpolation=cv2.INTER_NEAREST)
            confidence = cv2.resize(confidence[crop], (w, h), interpolation=cv2.INTER_NEAREST)
            tensor, (scale, dx, dy, _, _) = letterbox(image, 320, normalized=False)
            infer_start = time.perf_counter()
            boxes = yolo.run(None, {yolo.get_inputs()[0].name: tensor})[0][0]
            yolo_ms = (time.perf_counter() - infer_start) * 1000
            detections = []
            foreground = np.zeros((h, w), bool)
            for x1, y1, x2, y2, score, class_id in boxes:
                if score < 0.35:
                    continue
                x1, x2 = np.clip([(x1 - dx) / scale, (x2 - dx) / scale], 0, w)
                y1, y2 = np.clip([(y1 - dy) / scale, (y2 - dy) / scale], 0, h)
                if x2 <= x1 or y2 <= y1:
                    continue
                detections.append(
                    {
                        "bbox_xyxy": list(map(float, (x1, y1, x2, y2))),
                        "confidence": float(score),
                        "class_id": int(class_id),
                    }
                )
                foreground[int(y1) : int(np.ceil(y2)), int(x1) : int(np.ceil(x2))] = True
            ground = (np.isin(classes, [3, 6, 11]) & (confidence >= 0.35)).astype(np.uint8)
            _, components = cv2.connectedComponents(ground, connectivity=8)
            elapsed = (time.perf_counter() - start) * 1000
            np.savez_compressed(
                cache / f"{sample['sample_id']}.npz",
                classes=classes,
                confidence=confidence.astype(np.float32),
                components=components,
                foreground=foreground,
                processing_ms=elapsed,
            )
            record = {
                **sample,
                **metadata,
                "segformer_latency_ms": seg_ms,
                "yolo_latency_ms": yolo_ms,
                "processing_ms": elapsed,
                "detections": detections,
                "resources": resources,
                "class_pixels_055": {
                    str(i): int(((classes == i) & (confidence >= 0.55)).sum())
                    for i in np.unique(classes)
                },
            }
            records.write(json.dumps(record) + "\n")
            records.flush()
            print(sample["sample_id"], "seg_ms", round(seg_ms, 1), flush=True)


def metric(samples: list[dict], domain: str, output: Path, input_size: int, settings: dict) -> None:
    import torch
    import torch.nn.functional as functional

    torch.set_num_threads(settings["resources"]["torch_threads"])
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(settings["resources"]["gpu_memory_fraction"])
    sys.path.insert(0, str(MODELS / "Depth-Anything-V2/metric_depth"))
    from depth_anything_v2.dpt import DepthAnythingV2

    guard(output, settings)
    weights = MODELS / f"depth_anything_v2_metric_{domain}_vits.pth"
    maximum = 20 if domain == "hypersim" else 80
    start = time.perf_counter()
    model = DepthAnythingV2(
        encoder="vits", features=64, out_channels=[48, 96, 192, 384], max_depth=maximum
    )
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    model.eval().cuda()
    torch.cuda.synchronize()
    load_ms = (time.perf_counter() - start) * 1000
    cache = output / f"{domain}_{input_size}"
    cache.mkdir(parents=True, exist_ok=True)
    estimator = WallMetricEstimator(WallMetricConfig(**settings["wall"]))
    metadata = {
        "model": domain,
        "max_depth_m": maximum,
        "weights_sha256": sha256(weights),
        "torch": torch.__version__,
        "input_size": input_size,
        "load_ms": load_ms,
        "context": "offline_RGBD_replay_not_CameraX",
        "quantity": "predicted_optical_axis_z_m",
        "wall_semantics_ground_truth_available": False,
        "age_basis": "simulated_serial_consumer_age_from_measured_processing_times",
    }
    with (output / f"metric_{domain}_{input_size}.jsonl").open("w", encoding="utf-8") as records:
        for sample in samples:
            resources = guard(output, settings)
            image = cv2.imread(sample["rgb_path"])
            h, w = image.shape[:2]
            start = time.perf_counter()
            tensor, _ = model.image2tensor(image, input_size)
            torch.cuda.synchronize()
            infer_start = time.perf_counter()
            with torch.inference_mode():
                native = model(tensor)
                torch.cuda.synchronize()
                inference_ms = (time.perf_counter() - infer_start) * 1000
                aligned = functional.interpolate(
                    native[:, None], (h, w), mode="bilinear", align_corners=True
                )[0, 0]
                prediction = aligned.cpu().numpy()
            latency = (time.perf_counter() - start) * 1000
            sem = np.load(output / "semantics" / f"{sample['sample_id']}.npz")
            key = FrameKey(sample["sequence"], sample["rgb_index"], sample["timestamp_ms"], w, h)
            field = MetricDepthField(
                key,
                prediction,
                domain,
                maximum,
                source_age_ms=latency,
                model_latency_ms=inference_ms,
            )
            wall = estimator.estimate(
                key,
                sem["classes"],
                sem["confidence"],
                field,
                semantic_key=key,
                semantic_age_ms=latency + float(sem["processing_ms"]),
                foreground_mask=sem["foreground"],
            )
            truth = cv2.imread(sample["depth_path"], cv2.IMREAD_UNCHANGED).astype(np.float32) / 5000
            overall = depth_errors(prediction, truth) if sample["depth_association_valid"] else None
            mask, _, _, _ = estimator.mask(
                key, sem["classes"], sem["confidence"], sem["foreground"]
            )
            wall_truth = np.where(mask, truth, 0)
            wall_error = (
                depth_errors(prediction, wall_truth) if sample["depth_association_valid"] else None
            )
            bins = {}
            for low, high in [(0, 1), (1, 2), (2, 3), (3, 5), (5, 10)]:
                bins[f"{low}-{high}m"] = (
                    depth_errors(
                        prediction, np.where((truth >= low) & (truth < high) & mask, truth, 0)
                    )
                    if sample["depth_association_valid"]
                    else None
                )
            np.savez_compressed(
                cache / f"{sample['sample_id']}.npz",
                depth=prediction,
                latency_ms=latency,
                inference_ms=inference_ms,
            )
            record = {
                **sample,
                **metadata,
                "input_shape": list(tensor.shape),
                "inference_ms": inference_ms,
                "processing_ms": latency,
                "gpu_peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
                "gpu_peak_reserved_mb": torch.cuda.max_memory_reserved() / 1024**2,
                "resources": resources,
                "wall": wall.to_dict(),
                "whole_frame_errors": overall,
                "predicted_wall_region_errors": wall_error,
                "wall_distance_bins": bins,
                "wall_truth_valid_fraction": float(((truth > 0) & mask).sum() / max(1, mask.sum())),
                "config": asdict(estimator.config),
            }
            records.write(json.dumps(record) + "\n")
            records.flush()
            sem.close()
            print(
                sample["sample_id"],
                domain,
                round(inference_ms, 1),
                wall.status,
                "median",
                wall.median_m,
                flush=True,
            )


def relative(samples: list[dict], output: Path, settings: dict) -> None:
    """Frozen Android relative model baseline, with no conversion to meters."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = settings["resources"]["ort_threads"]
    options.inter_op_num_threads = 1
    options.enable_cpu_mem_arena = False
    model = ROOT / "android/app/src/main/assets/depth_anything_v2_small_280.onnx"
    session = ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
    cache = output / "relative_280"
    cache.mkdir(parents=True, exist_ok=True)
    with (output / "relative_results.jsonl").open("w", encoding="utf-8") as records:
        for sample in samples:
            resources = guard(output, settings)
            image = cv2.imread(sample["rgb_path"])
            h, w = image.shape[:2]
            start = time.perf_counter()
            tensor, (_, dx, dy, rw, rh) = letterbox(image, 280, normalized=True)
            infer_start = time.perf_counter()
            native = session.run(None, {session.get_inputs()[0].name: tensor})[0][0]
            inference_ms = (time.perf_counter() - infer_start) * 1000
            aligned = cv2.resize(
                native[dy : dy + rh, dx : dx + rw], (w, h), interpolation=cv2.INTER_LINEAR
            )
            elapsed = (time.perf_counter() - start) * 1000
            np.savez_compressed(
                cache / f"{sample['sample_id']}.npz",
                depth=aligned,
                latency_ms=elapsed,
                inference_ms=inference_ms,
            )
            record = {
                **sample,
                "quantity": "relative_inverse_depth_larger_is_nearer",
                "metric_error_evaluation": "NOT_APPLICABLE_NO_METRIC_UNITS",
                "onnx_sha256": sha256(model) if not records.tell() else None,
                "inference_ms": inference_ms,
                "processing_ms": elapsed,
                "p10_p50_p90": np.quantile(aligned, [0.1, 0.5, 0.9]).tolist(),
                "resources": resources,
            }
            records.write(json.dumps(record) + "\n")
            records.flush()
            print(sample["sample_id"], "relative_ms", round(inference_ms, 1), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=["semantics", "hypersim", "vkitti", "relative"], required=True
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/metric_corridor_research.yaml"
    )
    parser.add_argument("--manifest", type=Path, default=MODELS / "sample-manifest.json")
    parser.add_argument("--output", type=Path, default=DELIVERY / "evaluation")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--limit", type=int, help="Diagnostic subset only, explicitly reported")
    args = parser.parse_args()
    settings = load_research_settings(args.config)
    lower_own_priority()
    cv2.setNumThreads(1)
    samples = json.loads(args.manifest.read_text(encoding="utf-8"))["samples"]
    if args.limit:
        samples = samples[: args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    if args.phase == "semantics":
        semantics(samples, args.output, settings)
    elif args.phase == "relative":
        relative(samples, args.output, settings)
    else:
        metric(samples, args.phase, args.output, args.input_size, settings)


if __name__ == "__main__":
    main()
