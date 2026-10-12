"""Serial, offline real-model phases; GT is never passed to model inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from sightguard.research.metric3d_adapter import prepare_metric3d, restore_metric3d_depth
from sightguard.research.resources import check_resources, lower_own_priority

METRIC3D_SHA = "b34b2a2be9148054991cef7e417930e1320602ba7bc503b0ee4e7888543728f6"
METRIC3D_SOURCE = "eb5b6fac0dc155e4e52f576e304fbf11655ff339"
ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def metric3d(samples: list[dict], args: argparse.Namespace) -> None:
    import torch

    source = args.model_cache / "Metric3D"
    actual_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True
    ).strip()
    weights = args.model_cache / "metric_depth_vit_small_800k.pth"
    if actual_sha != METRIC3D_SOURCE or sha256(weights) != METRIC3D_SHA:
        raise ValueError("Metric3D source/weight version mismatch")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(0.65)
    start = time.perf_counter()
    model = torch.hub.load(str(source), "metric3d_vit_small", source="local", pretrain=False)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state["model_state_dict"], strict=False)
    # Official hub uses strict=False. Permit only the upstream unused mask token;
    # it is read only when a masking argument is supplied (never in inference).
    if (
        set(incompatible.missing_keys) != {"depth_model.encoder.mask_token"}
        or incompatible.unexpected_keys
    ):
        raise ValueError(f"unexpected checkpoint incompatibility: {incompatible}")
    del state
    model.cuda().eval()
    torch.cuda.synchronize()
    metadata = {
        "source_sha": actual_sha,
        "weights_sha256": METRIC3D_SHA,
        "load_ms": (time.perf_counter() - start) * 1000,
        "input_hw": [616, 1064],
        "dtype": "float32",
        "quantity": "predicted_optical_axis_z_m_focal_decanonicalized",
        "weight_license": "not_independently_verified_for_redistribution_research_only",
        "torch": torch.__version__,
        "missing_weight_keys": incompatible.missing_keys,
        "mask_token_inference_invariance": "checked_on_first_sample",
    }
    folder = args.output / "metric3d"
    folder.mkdir(exist_ok=False)
    with (args.output / "metric3d-results.jsonl").open("x", encoding="utf-8") as log:
        for sample in samples:
            resources = check_resources(stop_utc=args.stop_utc, output_dir=args.output)
            image = cv2.imread(sample["rgb_path"])
            if image is None or sha256(Path(sample["rgb_path"])) != sample["rgb_sha256"]:
                raise ValueError("input RGB missing or changed")
            start = time.perf_counter()
            values, geometry = prepare_metric3d(image[:, :, ::-1], sample["intrinsic"][0])
            tensor = torch.from_numpy(values).cuda()
            torch.cuda.synchronize()
            infer_start = time.perf_counter()
            with torch.inference_mode():
                canonical, _, extra = model.inference({"input": tensor})
                if not log.tell():
                    token = model.depth_model.encoder.mask_token
                    saved_token = token.clone()
                    token.fill_(1e6)
                    check, _, check_extra = model.inference({"input": tensor})
                    token.copy_(saved_token)
                    if not torch.equal(canonical, check):
                        raise ValueError("missing upstream mask token affects real inference")
                    del check, check_extra, saved_token
            torch.cuda.synchronize()
            infer_ms = (time.perf_counter() - infer_start) * 1000
            depth = restore_metric3d_depth(canonical.squeeze().cpu().numpy(), geometry)
            normal = extra["prediction_normal"][0, :3].cpu().numpy()
            top, bottom, left, right = geometry.padding_tblr
            normal = normal[:, top : normal.shape[1] - bottom, left : normal.shape[2] - right]
            normal = cv2.resize(normal.transpose(1, 2, 0), geometry.source_hw[::-1])
            processing_ms = (time.perf_counter() - start) * 1000
            np.savez_compressed(
                folder / f"{sample['sample_id']}.npz",
                depth=depth,
                normals=normal.astype(np.float16),
            )
            record = {
                "sample_id": sample["sample_id"],
                "metadata": metadata,
                "inference_ms": infer_ms,
                "processing_ms": processing_ms,
                "source_focal_x": sample["intrinsic"][0],
                "scaled_focal_x": geometry.scaled_focal_x,
                "peak_gpu_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
                "peak_gpu_reserved_mb": torch.cuda.max_memory_reserved() / 1024**2,
                "resources": resources,
            }
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(sample["sample_id"], "metric3d_ms", round(infer_ms, 1), flush=True)
            del tensor, canonical, extra, normal


def baselines(samples: list[dict], args: argparse.Namespace) -> None:
    # Existing adapters only receive RGB + source metadata; no GT arrays.
    reserved = {
        "semantics": ("semantics", "semantic_results.jsonl"),
        "relative": ("relative_280", "relative_results.jsonl"),
    }.get(args.phase, (args.phase, f"{args.phase}-results.jsonl"))
    if any((args.output / name).exists() for name in reserved):
        raise FileExistsError("refusing to overwrite previous model results")
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_metric_research import relative, semantics

    settings = {
        "resources": {
            "ort_threads": 2,
            "minimum_free_ram_mb": 1536,
            "operations_stop_utc": args.stop_utc,
        }
    }
    if args.phase == "semantics":
        semantics(samples, args.output, settings)
    elif args.phase == "relative":
        relative(samples, args.output, settings)
    else:
        import torch

        source = args.existing_cache / "Depth-Anything-V2"
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True
        ).strip()
        if revision != "a561b849ebae10a6f5ef49e26c83cbbcd36c71bf":
            raise ValueError("Depth Anything source revision mismatch")
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.cuda.set_per_process_memory_fraction(0.65)
        sys.path.insert(0, str(source / "metric_depth"))
        from depth_anything_v2.dpt import DepthAnythingV2

        size = 280 if args.phase == "hypersim280" else 518
        model = DepthAnythingV2(
            encoder="vits", features=64, out_channels=[48, 96, 192, 384], max_depth=20
        )
        weights = args.existing_cache / "depth_anything_v2_metric_hypersim_vits.pth"
        if sha256(weights) != "b782898d8a3e8be1f639de33837ed85e9b4b73e40f8f5e5cd99067588d722545":
            raise ValueError("Hypersim weight checksum mismatch")
        model.load_state_dict(
            torch.load(weights, map_location="cpu", weights_only=True), strict=True
        )
        model.cuda().eval()
        folder = args.output / args.phase
        folder.mkdir(exist_ok=False)
        with (args.output / f"{args.phase}-results.jsonl").open("x", encoding="utf-8") as log:
            for sample in samples:
                resources = check_resources(stop_utc=args.stop_utc, output_dir=args.output)
                image = cv2.imread(sample["rgb_path"])
                tensor, _ = model.image2tensor(image, size)
                torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.inference_mode():
                    native = model(tensor)
                    restored = torch.nn.functional.interpolate(
                        native[:, None], image.shape[:2], mode="bilinear", align_corners=True
                    )[0, 0]
                torch.cuda.synchronize()
                ms = (time.perf_counter() - start) * 1000
                np.savez_compressed(
                    folder / f"{sample['sample_id']}.npz", depth=restored.cpu().numpy()
                )
                log.write(
                    json.dumps(
                        {
                            "sample_id": sample["sample_id"],
                            "inference_ms": ms,
                            "resources": resources,
                            "input_size": size,
                        }
                    )
                    + "\n"
                )
                log.flush()
                print(sample["sample_id"], args.phase, round(ms, 1), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--existing-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-utc", required=True)
    parser.add_argument(
        "--phase",
        choices=["metric3d", "semantics", "relative", "hypersim280", "hypersim518"],
        required=True,
    )
    args = parser.parse_args()
    lower_own_priority()
    cv2.setNumThreads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    check_resources(stop_utc=args.stop_utc, output_dir=args.output)
    samples = json.loads(args.manifest.read_text(encoding="utf-8"))["samples"]
    if args.phase != "metric3d":
        samples = [sample for sample in samples if sample["dataset"] == "NYU"]
    if not samples:
        raise ValueError("no frozen samples")
    for sample in samples:
        if sha256(Path(sample["rgb_path"])) != sample["rgb_sha256"]:
            raise ValueError("frozen input RGB checksum mismatch")
    if args.phase == "metric3d":
        metric3d(samples, args)
    else:
        baselines(samples, args)


if __name__ == "__main__":
    main()
