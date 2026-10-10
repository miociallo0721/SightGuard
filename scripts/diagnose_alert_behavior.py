"""Reproduce diagnostics from existing artifacts without running inference."""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from sightguard.evaluation.alert_diagnosis import diagnose_video


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("datasets/real_world/manifests/dataset.json")
    )
    parser.add_argument("--runs", type=Path, default=Path("outputs/real_world/v01"))
    parser.add_argument("--output", type=Path, default=Path("outputs/alert_behavior_diagnosis/v01"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    results, provenance, summary, index = [], {}, [], []
    for path in (
        args.manifest,
        Path(__file__),
        Path("configs/risk.yaml"),
        Path("src/sightguard/evaluation/alert_diagnosis.py"),
    ):
        provenance[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for video in manifest["videos"]:
        video_id = video["video_id"]
        root = args.runs / video_id
        data = {}
        for name in ("risk_frames", "events", "alerts"):
            path = root / "assessment" / f"{name}.jsonl"
            data[name] = [
                json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
            ]
        for path in [
            root / "perception" / "tracks.jsonl",
            root / "assessment" / "run_metadata.json",
            *(root / "assessment" / f"{n}.jsonl" for n in data),
        ]:
            provenance[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        result = diagnose_video(
            video_id, video["duration"] * 1000, data["risk_frames"], data["events"], data["alerts"]
        )
        track_path = root / "perception" / "tracks.jsonl"
        track_frames = {
            row["frame_index"]: row
            for row in (
                json.loads(line) for line in track_path.read_text(encoding="utf-8").splitlines()
            )
        }
        mismatches = []
        for frame in data["risk_frames"]:
            source = track_frames.get(frame["frame_index"])
            if source is None or abs(source["timestamp_ms"] - frame["timestamp_ms"]) > 0.001:
                mismatches.append(
                    {"frame_index": frame["frame_index"], "reason": "source_frame_time"}
                )
                continue
            for observation in frame["observations"]:
                position = observation["detection_index"]
                detections = source["detections"]
                if (
                    position >= len(detections)
                    or detections[position]["track_id"] != observation["source_track_id"]
                    or detections[position]["class_id"] != observation["class_id"]
                ):
                    mismatches.append(
                        {"frame_index": frame["frame_index"], "reason": "source_detection_identity"}
                    )
        result["source_alignment_mismatches"] = mismatches
        results.append(result)
        summary.append(
            {
                k: v
                for k, v in result.items()
                if k not in ("events", "same_track_multiple_events", "source_alignment_mismatches")
            }
        )
        for event in result["events"]:
            if (
                event["alert_count"]
                or event["score_changes"]
                or event["event_kind"] == "perception"
            ):
                index.append(
                    {
                        k: event[k]
                        for k in (
                            "video_id",
                            "event_id",
                            "event_kind",
                            "start_ms",
                            "end_ms",
                            "right_censored",
                            "alert_count",
                            "source_track_ids",
                            "class_name",
                        )
                    }
                )
                index[-1]["alert_times_ms"] = [a["timestamp_ms"] for a in event["alerts"]]
                index[-1]["flags"] = (
                    (
                        ["perception_episode_no_alert"]
                        if event["event_kind"] == "perception" and not event["alert_count"]
                        else []
                    )
                    + (["multiple_alerts"] if event["alert_count"] > 1 else [])
                    + (["score_level_change_or_delta_ge_0.15"] if event["score_changes"] else [])
                    + (["multiple_track_ids"] if len(event["source_track_ids"]) > 1 else [])
                    + (
                        ["active_reentry"]
                        if sum(t["state"] == "ACTIVE" for t in event["transitions"]) > 1
                        else []
                    )
                )
    current_track_realerts = []
    classes = Counter()
    for video in results:
        by_track = defaultdict(list)
        for event in video["events"]:
            for alert in event["alerts"]:
                observation = alert["observation"]
                if observation:
                    classes[observation["class_name"]] += 1
                    if (
                        alert["alert_type"] == "first"
                        and observation["source_track_id"] is not None
                    ):
                        by_track[observation["source_track_id"]].append(
                            {
                                "event_id": event["event_id"],
                                "timestamp_ms": alert["timestamp_ms"],
                                "score": alert["risk_score"],
                                "alert_id": alert["alert_id"],
                            }
                        )
        current_track_realerts.extend(
            {"video_id": video["video_id"], "track_id": track, "first_alerts": rows}
            for track, rows in by_track.items()
            if len(rows) > 1
        )
    payload = {
        "schema_version": "sightguard.alert-diagnosis.v1",
        "ground_truth_used": False,
        "audio_playback_count": None,
        "audio_note": "Offline assessment has no playback log; not zero playback evidence",
        "input_sha256": provenance,
        "videos": results,
        "alert_class_counts": dict(classes),
        "current_track_realerts": current_track_realerts,
    }
    (args.output / "diagnosis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "anomaly_index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.output / "score_trace.jsonl").open("w", encoding="utf-8") as handle:
        for video in manifest["videos"]:
            path = args.runs / video["video_id"] / "assessment" / "risk_frames.jsonl"
            for line in path.read_text(encoding="utf-8").splitlines():
                frame = json.loads(line)
                for observation in frame["observations"]:
                    handle.write(
                        json.dumps(
                            {
                                "video_id": video["video_id"],
                                "timestamp_ms": frame["timestamp_ms"],
                                "frame_index": frame["frame_index"],
                                **observation,
                            }
                        )
                        + "\n"
                    )
    with (args.output / "video_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
