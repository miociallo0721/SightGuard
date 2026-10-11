"""Read-only source verification, same-source ablations and isolated alert replay."""

import argparse
import json
from pathlib import Path

from sightguard.perception.depth.artifacts import sha256_file, write_json
from sightguard.pipeline.assess import assess_tracks
from sightguard.risk.config import load_risk_config
from sightguard.scene.spatial_grid.artifacts import fresh_directory, read_rows


def validate(output: Path, fusion_root: Path, frozen_hashes: Path) -> dict:
    fresh_directory(output)
    frozen = json.loads(frozen_hashes.read_text(encoding="utf-8"))
    mismatches = [path for path, digest in frozen.items() if sha256_file(Path(path)) != digest]
    assert not mismatches, mismatches
    result = {
        "frozen_files_checked": len(frozen),
        "frozen_mismatches": mismatches,
        "videos": [],
        "total_frames": 0,
        "total_alerts": 0,
        "production_audio_called": False,
        "audio_check": "isolated assess_tracks has no audio adapter",
        "fusion_root": str(fusion_root),
        "failures": [],
    }
    for directory in sorted(fusion_root.glob("urbanego_run8_*")):
        video_id = directory.name
        frames = read_rows(directory / "spatial_frames.jsonl")
        original_depth = read_rows(
            Path("outputs/depth_integration/v03") / video_id / "depth_frames.jsonl"
        )
        original_scene = read_rows(
            Path("outputs/path_relevance/v04") / video_id / "scene_frames.jsonl"
        )
        video_metadata = json.loads((directory / "run_metadata.json").read_text(encoding="utf-8"))
        semantic_path = next(
            Path(path)
            for path in video_metadata["input_sha256"]
            if Path(path).name == "semantic_frames.jsonl"
        )
        semantic = read_rows(semantic_path)
        assert len(frames) == len(original_depth) == len(semantic) == 30
        for index, (frame, depth) in enumerate(zip(frames, original_depth, strict=True)):
            assert frame["key"]["video_id"] == depth["video_id"] == video_id
            assert frame["key"]["frame_index"] == depth["frame_index"] == index * 10
            assert frame["key"]["timestamp_ms"] == depth["timestamp_ms"] == index * 500
            assert len(frame["cells"]) == 225
            assert not frame["production_decision_changed"]
        metrics = read_rows(directory / "ablation_metrics.jsonl")
        keys = {name: [row["key"] for row in metrics if row["ablation"] == name] for name in "ABCD"}
        assert keys["A"] == keys["B"] == keys["C"] == keys["D"]
        assert all(row["states"] == {"UNKNOWN": 225} for row in metrics if row["ablation"] == "B")
        baseline = Path("outputs/real_world/v01") / video_id
        replay = output / "replay" / video_id
        assess_tracks(
            baseline / "perception/tracks.jsonl",
            replay,
            load_risk_config(Path("configs/risk.yaml")),
            stream_id=video_id,
        )
        for name in ("risk_frames.jsonl", "events.jsonl", "alerts.jsonl"):
            assert read_rows(replay / name) == read_rows(baseline / "assessment" / name), (
                video_id,
                name,
            )
        alerts = len(read_rows(replay / "alerts.jsonl"))
        result["total_alerts"] += alerts
        result["total_frames"] += len(read_rows(replay / "risk_frames.jsonl"))
        result["videos"].append(
            {
                "video_id": video_id,
                "fusion_frames": len(frames),
                "alerts": alerts,
                "scene_status_equal_to_v04": all(
                    a["status"] == b["status"]
                    for a, b in zip(semantic, original_scene, strict=True)
                ),
            }
        )
    assert result["total_frames"] == 2400
    assert result["total_alerts"] == 257
    assert all(sha256_file(Path(path)) == digest for path, digest in frozen.items())
    write_json(output / "validation.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fusion", type=Path, default=Path("outputs/spatial_grid/v02"))
    parser.add_argument(
        "--frozen", type=Path, default=Path("outputs/spatial_grid_audit/v01/frozen_hashes.json")
    )
    args = parser.parse_args()
    print(validate(args.output, args.fusion, args.frozen))


if __name__ == "__main__":
    main()
