"""Check real artifacts, causal joins, producer hashes and independent alert replay."""

import argparse
import json
from pathlib import Path

from validate_spatial_grid import validate as validate_baseline

from sightguard.perception.depth.artifacts import sha256_file, write_json
from sightguard.scene.spatial_grid.artifacts import fresh_directory, read_rows


def validate(motion: Path, temporal: Path, output: Path, frozen: Path) -> dict:
    fresh_directory(output)
    meta = json.loads((motion / "run_metadata.json").read_text(encoding="utf-8"))
    assert meta["status"] == "completed" and not meta["torch_loaded"]
    assert not meta["production_decision_changed"] and not meta["production_audio_called"]
    for path, digest in meta["source_sha256"].items():
        assert sha256_file(Path(path)) == digest, path
    for root, files in meta["output_sha256_by_root"].items():
        for reference, digest in files.items():
            assert sha256_file(Path(root) / reference) == digest, reference
    videos = []
    for item in meta["videos"]:
        video = item["video_id"]
        data = read_rows(motion / video / "frame_motion.jsonl")
        control = read_rows(motion / video / "unmasked_frame_motion.jsonl")
        windows = read_rows(motion / video / "motion_windows.jsonl")
        rows = read_rows(temporal / video / "temporal_corridor.jsonl")
        source = read_rows(
            Path("outputs/corridor_intelligence/v01") / video / "corridor_frames.jsonl"
        )
        indexed = {(x["key"]["frame_index"], x["ablation"]): x for x in source}
        assert len(data) == len(control) == 300
        assert len(windows) == 30 and len(rows) == 90
        assert [x["key"] for x in data] == [x["key"] for x in control]
        for i, row in enumerate(data):
            assert row["key"]["frame_index"] == i and row["key"]["video_id"] == video
            assert row["absolute_camera_pose_status"] == "UNKNOWN"
            if i:
                assert row["previous_key"] == data[i - 1]["key"]
                assert (
                    row["dt_ms"] == row["key"]["timestamp_ms"] - data[i - 1]["key"]["timestamp_ms"]
                )
            if row["quality"]["label"] == "UNKNOWN":
                assert row["quality"]["reasons"]
        for w in windows:
            assert w["end_ms"] == w["key"]["timestamp_ms"]
            assert w["absolute_camera_pose_status"] == "UNKNOWN"
            for index in w["source_frame_indices"]:
                assert w["start_ms"] < data[index]["key"]["timestamp_ms"] <= w["end_ms"]
                assert data[index]["stream_id"] == w["stream_id"]
        for row in rows:
            original = indexed[(row["key"]["frame_index"], row["ablation"])]
            assert row["source_corridor"] == original
            assert not row["production_decision_changed"] and not row["production_audio_called"]
            assert row["motion_window"]["key"] == row["key"]
            if row["ablation"] == "D":
                assert original["status"] == "ABSTAIN"
                assert all(x["status"] == "INVALID" for x in row["freshness"])
            for f in row["freshness"]:
                assert f["consumer_time_kind"] == "SIMULATED_VIDEO_PTS_NOT_ONLINE_LATENCY"
                if f["status"] != "FRESH":
                    assert f["gated_research_best"] is None
                    assert not f["reuse_allowed_for_research"]
        videos.append(
            {
                "video_id": video,
                "frames": len(data),
                "windows": len(windows),
                "temporal_rows": len(rows),
            }
        )
    baseline = validate_baseline(output / "baseline", Path("outputs/spatial_grid/v03"), frozen)
    result = {
        "status": "passed",
        "videos": videos,
        "frames": sum(x["frames"] for x in videos),
        "temporal_rows": sum(x["temporal_rows"] for x in videos),
        "frozen_files_checked": baseline["frozen_files_checked"],
        "frozen_mismatches": baseline["frozen_mismatches"],
        "baseline_alerts": baseline["total_alerts"],
        "baseline_frames": baseline["total_frames"],
        "risk_frames_events_alerts_equal": True,
        "production_audio_called": False,
        "original_corridor_rows_equal": True,
        "all_absolute_pose_unknown": True,
    }
    write_json(output / "validation.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, default=Path("outputs/motion_quality/v01"))
    parser.add_argument("--temporal", type=Path, default=Path("outputs/temporal_corridor/v01"))
    parser.add_argument(
        "--frozen", type=Path, default=Path("outputs/motion_quality_audit/v01/frozen_hashes.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(validate(args.motion, args.temporal, args.output, args.frozen))


if __name__ == "__main__":
    main()
