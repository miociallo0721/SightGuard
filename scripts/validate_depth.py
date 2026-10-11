"""Check saved depth, replay original alerts, and optionally benchmark real backends."""

from __future__ import annotations

import argparse
import json
import sys
import time
import weakref
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def benchmark() -> dict[str, Any]:
    import cv2
    import numpy as np

    from sightguard.perception.depth.model import DepthEstimator
    from sightguard.perception.depth.schema import DepthModelConfig
    from sightguard.pipeline.depth import percentiles

    capture = cv2.VideoCapture(str(ROOT / "datasets/real_world/processed/urbanego_run8_05.mp4"))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise OSError("V05 frame unavailable")
    context = dict(
        video_id="urbanego_run8_05",
        frame_index=0,
        timestamp_ms=0,
        source_width=1280,
        source_height=720,
    )
    results = {}
    for device, samples in (("cpu", 10), ("cuda:0", 60)):
        estimator = DepthEstimator(DepthModelConfig(device=device))
        started = time.perf_counter()
        try:
            estimator.initialize()
            model_reference = weakref.ref(estimator.model)
            model_bytes = sum(p.numel() * p.element_size() for p in estimator.model.parameters())
            warmups = [estimator.infer(frame, **context).result.to_dict() for _ in range(2)]
            if any(r["status"] != "valid" for r in warmups):
                raise RuntimeError("real warmup failed")
            observations = []
            for _ in range(samples):
                prediction = estimator.infer(frame, **context)
                if prediction.result.status != "valid":
                    raise RuntimeError(f"real prediction failed: {prediction.result.error}")
                observations.append(
                    {
                        "timings_ms": prediction.result.timings_ms,
                        "memory": estimator.resource_stats(),
                    }
                )
            near = float(np.median(prediction.depth_map[200:500, 1100:1200]))
            far = float(np.median(prediction.depth_map[50:100, 520:660]))
            assert near > far
            before = estimator.resource_stats()
            estimator.close()
            after = estimator.resource_stats()
            assert model_reference() is None
            if device.startswith("cuda"):
                assert before["allocated_mib"] - after["allocated_mib"] >= model_bytes / 2**20
            results[device] = {
                "status": "completed",
                "model": estimator.metadata,
                "warmup": warmups,
                "samples": samples,
                "timings_ms": {
                    name: percentiles([r["timings_ms"][name] for r in observations])
                    for name in ("preprocess", "forward", "postprocess", "inference")
                },
                "memory_observations": observations,
                "before_unload": before,
                "after_unload": after,
                "wall_seconds": time.perf_counter() - started,
                "direction_probe": {
                    "near_wall_median": near,
                    "far_car_median": far,
                    "scope": "RGB-selected regions; direction sanity check, not depth GT",
                },
                "cpu_threads": estimator.torch.get_num_threads(),
                "scope": "one decoded true V05 frame repeatedly inferred; no saving or decoding",
            }
        except Exception as exc:
            results[device] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        finally:
            estimator.close()
        print(device, results[device]["status"], flush=True)
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--benchmarks", action="store_true")
    parser.add_argument(
        "--artifact-baseline",
        type=Path,
        help="Optional pre-run JSON path-to-SHA256 map for historical artifacts",
    )
    parser.add_argument("--workspace-dependencies", action="store_true")
    args = parser.parse_args()
    if args.workspace_dependencies:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        sys.path.append(str(ROOT / ".venv/Lib/site-packages"))
    import cv2

    from sightguard.perception.depth.artifacts import load_depth_map, sha256_file, write_json
    from sightguard.pipeline.assess import assess_tracks
    from sightguard.pipeline.depth import read_jsonl
    from sightguard.risk.config import load_risk_config

    args.output_dir.mkdir(parents=True, exist_ok=False)
    run = json.loads((args.run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    report: dict[str, Any] = {
        "run_sha256": sha256_file(args.run_dir / "run_metadata.json"),
        "artifacts": [],
        "baseline_replay": [],
    }
    for video in run["videos"]:
        video_id = video["video_id"]
        directory = args.run_dir / video_id
        rows = read_jsonl(directory / "depth_frames.jsonl")
        scenes = read_jsonl(ROOT / "outputs/path_relevance/v04" / video_id / "scene_frames.jsonl")
        assert [(r["frame_index"], r["timestamp_ms"]) for r in rows] == [
            (r["frame_index"], r["timestamp_ms"]) for r in scenes
        ]
        for row in rows:
            depth = load_depth_map(directory, row)
            assert list(depth.shape) == row["depth_map_shape"]
        images = [
            cv2.imread(str(directory / "previews" / f"{i:06d}.jpg")) for i in (0, 100, 200, 290)
        ]
        if any(img is None for img in images):
            raise OSError("preview decode failed")
        if not cv2.imwrite(str(args.output_dir / f"{video_id}_contact.jpg"), cv2.vconcat(images)):
            raise OSError("contact sheet save failed")
        report["artifacts"].append(
            {
                "video_id": video_id,
                "arrays_verified": len(rows),
                "alignment": "exact v04 frame indices and timestamps",
            }
        )
        baseline = ROOT / "outputs/real_world/v01" / video_id
        output = args.output_dir / "risk_replay" / video_id
        replay = assess_tracks(
            baseline / "perception/tracks.jsonl",
            output,
            load_risk_config(ROOT / "configs/risk.yaml"),
            stream_id=video_id,
        )
        equal = {
            name: read_jsonl(output / name) == read_jsonl(baseline / "assessment" / name)
            for name in ("risk_frames.jsonl", "events.jsonl", "alerts.jsonl")
        }
        assert all(equal.values())
        report["baseline_replay"].append(
            {
                "video_id": video_id,
                "frames": replay["frames"],
                "alerts": replay["alerts"],
                "equal": equal,
            }
        )
    original = {
        reference["path"]: reference["sha256"]
        for video in run["videos"]
        for reference in video["provenance"]["references"].values()
    }
    if args.artifact_baseline:
        original.update(json.loads(args.artifact_baseline.read_text()))
    report["existing_artifacts_checked"] = len(original)
    report["existing_artifacts_unchanged"] = all(
        sha256_file(Path(p)) == h for p, h in original.items()
    )
    assert report["existing_artifacts_unchanged"]
    if args.benchmarks:
        report["benchmarks"] = benchmark()
    write_json(args.output_dir / "validation.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
