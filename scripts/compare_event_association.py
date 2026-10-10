"""Compare two Phase 2 event-association replays over the same tracks JSONL."""

import argparse
import csv
import hashlib
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sightguard.data import FrameRecord, replay_records
from sightguard.decision.events import Association, EventManager
from sightguard.risk.config import load_risk_config
from sightguard.risk.engine import AssessmentEngine
from sightguard.risk.schema import SpatialFeatures


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _event_signatures(changes: list[dict[str, Any]]) -> dict[str, str]:
    signatures = {}
    for change in changes:
        if change["event_kind"] == "risk" and change["action"] == "created":
            snapshot = change["snapshot"]
            signatures[change["event_id"]] = "|".join(
                (
                    f"{change['timestamp_ms']:.3f}",
                    str(change["source_track_id"]),
                    str(snapshot["class_id"]),
                )
            )
    return signatures


def _states(frame: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {event["event_id"]: event for event in frame["events"]}


def _measure_association_ms(tracks_path: Path, config: Any, video_id: str) -> list[float]:
    class TimedEventManager(EventManager):
        def __init__(self) -> None:
            super().__init__(config.association, video_id)
            self.elapsed_ms: list[float] = []

        def associate(
            self, frame: FrameRecord, spatial: dict[int, SpatialFeatures]
        ) -> list[Association]:
            started = time.perf_counter_ns()
            try:
                return super().associate(frame, spatial)
            finally:
                self.elapsed_ms.append((time.perf_counter_ns() - started) / 1_000_000)

    engine = AssessmentEngine(config, stream_id=video_id)
    manager = TimedEventManager()
    engine.manager = manager
    for record in replay_records(tracks_path):
        engine.step(record)
    return manager.elapsed_ms


def _timing_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"calls": 0, "total_ms": 0.0, "mean_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    ordered = sorted(values)
    return {
        "calls": len(values),
        "total_ms": sum(values),
        "mean_ms": statistics.fmean(values),
        "p95_ms": ordered[max(0, int(0.95 * len(ordered) + 0.999999) - 1)],
        "max_ms": ordered[-1],
    }


def _run_summary(
    run_dir: Path, video_id: str, duration_ms: float, association_ms: list[float]
) -> dict[str, Any]:
    frames = _read_jsonl(run_dir / "risk_frames.jsonl")
    changes = _read_jsonl(run_dir / "events.jsonl")
    alerts = _read_jsonl(run_dir / "alerts.jsonl")
    risk_event_ids = {change["event_id"] for change in changes if change["event_kind"] == "risk"}
    final_by_event: dict[str, dict[str, Any]] = {}
    interval_by_event: dict[str, list[float]] = defaultdict(list)
    for change in changes:
        if change["event_kind"] != "risk":
            continue
        final_by_event[change["event_id"]] = change["snapshot"]
        interval_by_event[change["event_id"]].append(change["timestamp_ms"])

    event_spans = {}
    for event_id, timestamps in interval_by_event.items():
        created_at = next(
            change["timestamp_ms"]
            for change in changes
            if change["event_id"] == event_id and change["action"] == "created"
        )
        resolved_at = next(
            (
                change["timestamp_ms"]
                for change in changes
                if change["event_id"] == event_id and change["action"] == "resolved"
            ),
            max(timestamps),
        )
        event_spans[event_id] = (created_at, resolved_at)

    same_frame_competitions = 0
    association_conflicts = 0
    conflict_cases = []
    previous_states: dict[str, dict[str, Any]] = {}
    for frame in frames:
        current_states = _states(frame)
        track_events: dict[int, set[str]] = defaultdict(set)
        for observation in frame["observations"]:
            track_id = observation["source_track_id"]
            if track_id is not None and observation["event_id"] is not None:
                track_events[track_id].add(observation["event_id"])
        same_frame_competitions += sum(len(event_ids) > 1 for event_ids in track_events.values())

        for observation in frame["observations"]:
            track_id = observation["source_track_id"]
            event_id = observation["event_id"]
            if track_id is None or event_id is None:
                continue
            other_claims = [
                state
                for other_id, state in current_states.items()
                if other_id != event_id
                and state["state"] != "RESOLVED"
                and (state["last_track_id"] == track_id or track_id in state["source_track_ids"])
            ]
            if other_claims:
                association_conflicts += 1
            if observation["association_method"] == "new" and other_claims:
                conflict_cases.append(
                    {
                        "video_id": video_id,
                        "frame_index": frame["frame_index"],
                        "timestamp_ms": frame["timestamp_ms"],
                        "event_id": event_id,
                        "track_id": track_id,
                        "class_name": observation["class_name"],
                        "association_before": [
                            {
                                "event_id": claim["event_id"],
                                "state": previous_states.get(claim["event_id"], {}).get("state"),
                                "last_track_id": previous_states.get(claim["event_id"], {}).get(
                                    "last_track_id"
                                ),
                            }
                            for claim in other_claims
                        ],
                        "association_after": {
                            "event_id": event_id,
                            "state": current_states[event_id]["state"],
                            "method": observation["association_method"],
                        },
                        "reason": "new_event_shares_a_track_id_with_another_live_event",
                        "alerts_for_new_event": [
                            alert for alert in alerts if alert["event_id"] == event_id
                        ],
                    }
                )
        previous_states = current_states

    multi_track_events = {
        event_id: list(snapshot["source_track_ids"])
        for event_id, snapshot in final_by_event.items()
        if len(snapshot["source_track_ids"]) > 1
    }
    first_alerts_by_track: dict[int, set[str]] = defaultdict(set)
    for alert in alerts:
        if alert["alert_type"] == "first":
            for track_id in alert["source_track_ids"]:
                first_alerts_by_track[track_id].add(alert["event_id"])
    first_alert_track_repeats = sum(
        len(event_ids) > 1 for event_ids in first_alerts_by_track.values()
    )
    overlapping_first_alert_pairs = []
    for track_id, event_ids in first_alerts_by_track.items():
        ordered_ids = sorted(event_ids)
        for left_index, left_id in enumerate(ordered_ids):
            for right_id in ordered_ids[left_index + 1 :]:
                left_span, right_span = event_spans[left_id], event_spans[right_id]
                if left_span[0] <= right_span[1] and right_span[0] <= left_span[1]:
                    overlapping_first_alert_pairs.append(
                        {
                            "track_id": track_id,
                            "event_ids": [left_id, right_id],
                            "lifecycles_ms": {left_id: left_span, right_id: right_span},
                            "interpretation": (
                                "candidate_only; shared tracker ID is not physical identity "
                                "ground truth"
                            ),
                        }
                    )
    counts = Counter(alert["alert_type"] for alert in alerts)
    duration_minutes = duration_ms / 60_000
    return {
        "video_id": video_id,
        "risk_event_count": len(risk_event_ids),
        "alert_count": len(alerts),
        "first_alerts": counts["first"],
        "escalation_alerts": counts["escalation"],
        "repeat_alerts": counts["repeat"],
        "perception_limited_alerts": counts["perception_limited"],
        "alerts_per_minute": len(alerts) / duration_minutes if duration_minutes else None,
        "same_frame_track_competitions": same_frame_competitions,
        "cross_event_track_conflict_observations": association_conflicts,
        "new_event_conflicts": len(conflict_cases),
        "first_alert_same_track_repeats": first_alert_track_repeats,
        "suspected_overlapping_same_track_first_alert_pairs": len(overlapping_first_alert_pairs),
        "overlapping_first_alert_pairs": overlapping_first_alert_pairs,
        "events_with_track_id_changes": len(multi_track_events),
        "multi_track_event_ids": multi_track_events,
        "association_timing": _timing_summary(association_ms),
        "conflict_cases": conflict_cases,
        "event_signatures": _event_signatures(changes),
        "changes": changes,
        "frames": frames,
        "alerts": alerts,
    }


def _association_changes(
    video_id: str, baseline: dict[str, Any], experimental: dict[str, Any]
) -> list[dict[str, Any]]:
    base_frames = {frame["frame_index"]: frame for frame in baseline["frames"]}
    v2_frames = {frame["frame_index"]: frame for frame in experimental["frames"]}
    cases = []
    previous_base_states: dict[str, dict[str, Any]] = {}
    previous_v2_states: dict[str, dict[str, Any]] = {}
    for frame_index in sorted(base_frames.keys() & v2_frames.keys()):
        base_frame, v2_frame = base_frames[frame_index], v2_frames[frame_index]
        if abs(base_frame["timestamp_ms"] - v2_frame["timestamp_ms"]) > 0.001:
            raise ValueError(f"Timestamp mismatch in {video_id} frame {frame_index}")
        base_obs = {row["detection_index"]: row for row in base_frame["observations"]}
        v2_obs = {row["detection_index"]: row for row in v2_frame["observations"]}
        base_states, v2_states = _states(base_frame), _states(v2_frame)
        base_alerts = [
            a for a in baseline["alerts"] if a["timestamp_ms"] == base_frame["timestamp_ms"]
        ]
        v2_alerts = [
            a for a in experimental["alerts"] if a["timestamp_ms"] == v2_frame["timestamp_ms"]
        ]
        for detection_index in sorted(base_obs.keys() & v2_obs.keys()):
            left, right = base_obs[detection_index], v2_obs[detection_index]
            base_owner = baseline["event_signatures"].get(left["event_id"])
            v2_owner = experimental["event_signatures"].get(right["event_id"])
            if base_owner == v2_owner and left["association_method"] == right["association_method"]:
                continue
            cases.append(
                {
                    "video_id": video_id,
                    "timestamp_ms": base_frame["timestamp_ms"],
                    "frame_index": frame_index,
                    "detection_index": detection_index,
                    "track_id": left["source_track_id"],
                    "class_name": left["class_name"],
                    "baseline": {
                        "event_id": left["event_id"],
                        "event_signature": base_owner,
                        "state_before": previous_base_states.get(left["event_id"], {}).get("state"),
                        "state_after_frame": base_states.get(left["event_id"], {}).get("state"),
                        "association_method": left["association_method"],
                    },
                    "experimental": {
                        "event_id": right["event_id"],
                        "event_signature": v2_owner,
                        "state_before": previous_v2_states.get(right["event_id"], {}).get("state"),
                        "state_after_frame": v2_states.get(right["event_id"], {}).get("state"),
                        "association_method": right["association_method"],
                    },
                    "baseline_alerts_at_timestamp": base_alerts,
                    "experimental_alerts_at_timestamp": v2_alerts,
                }
            )
        previous_base_states, previous_v2_states = base_states, v2_states
    return cases


def _alerts_by_signature(run: dict[str, Any]) -> dict[tuple[str | None, str], list[dict[str, Any]]]:
    signatures = run["event_signatures"]
    grouped: dict[tuple[str | None, str], list[dict[str, Any]]] = defaultdict(list)
    for alert in run["alerts"]:
        grouped[(signatures.get(alert["event_id"]), alert["alert_type"])].append(alert)
    return {
        key: sorted(records, key=lambda alert: (alert["timestamp_ms"], alert["alert_id"]))
        for key, records in grouped.items()
    }


def _alert_behavior_signature(alert: dict[str, Any]) -> tuple[Any, ...]:
    return (
        alert["timestamp_ms"],
        alert["risk_level"],
        alert["risk_score"],
        tuple(alert["source_track_ids"]),
        alert["quality_score"],
        alert["reason"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("datasets/real_world/manifests/dataset.json")
    )
    parser.add_argument("--tracks-root", type=Path, default=Path("outputs/real_world/v01"))
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--experimental", type=Path, required=True)
    parser.add_argument("--baseline-config", type=Path, default=Path("configs/risk.yaml"))
    parser.add_argument(
        "--experimental-config", type=Path, default=Path("configs/risk-association-v2.yaml")
    )
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.report_dir.exists():
        raise FileExistsError(f"Refusing to overwrite report directory: {args.report_dir}")
    args.report_dir.mkdir(parents=True)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    baseline_config, experimental_config = (
        load_risk_config(args.baseline_config),
        load_risk_config(args.experimental_config),
    )
    baseline_settings, experimental_settings = (
        baseline_config.to_dict(),
        experimental_config.to_dict(),
    )
    baseline_settings["association"]["allow_class_switch_for_track_id"] = True
    if baseline_settings != experimental_settings:
        raise ValueError("Experimental configuration must differ only by the class-switch flag")
    videos = []
    case_rows = []
    creation_changes = []
    alert_changes = []
    all_timings: dict[str, list[float]] = {"baseline": [], "experimental": []}
    for video in manifest["videos"]:
        video_id, duration_ms = video["video_id"], video["duration"] * 1000
        tracks_path = args.tracks_root / video_id / "perception" / "tracks.jsonl"
        base_dir, v2_dir = args.baseline / video_id, args.experimental / video_id
        for directory in (base_dir, v2_dir):
            if not all(
                (directory / name).is_file()
                for name in (
                    "risk_frames.jsonl",
                    "events.jsonl",
                    "alerts.jsonl",
                    "run_metadata.json",
                )
            ):
                raise FileNotFoundError(f"Incomplete assessment output: {directory}")
        base_timing = _measure_association_ms(tracks_path, baseline_config, video_id)
        v2_timing = _measure_association_ms(tracks_path, experimental_config, video_id)
        all_timings["baseline"].extend(base_timing)
        all_timings["experimental"].extend(v2_timing)
        base = _run_summary(base_dir, video_id, duration_ms, base_timing)
        v2 = _run_summary(v2_dir, video_id, duration_ms, v2_timing)
        tracks_sha256 = _sha256(tracks_path)
        source_metadata_sha256 = None
        strategy_metadata = {}
        for strategy, directory, config in (
            ("baseline", base_dir, baseline_config),
            ("experimental", v2_dir, experimental_config),
        ):
            metadata = json.loads((directory / "run_metadata.json").read_text(encoding="utf-8"))
            if metadata["input"]["sha256"] != tracks_sha256:
                raise ValueError(f"{strategy} tracks hash mismatch for {video_id}")
            if metadata["config"] != config.to_dict():
                raise ValueError(f"{strategy} effective config mismatch for {video_id}")
            source_hash = metadata["input"].get("source_metadata_sha256")
            if source_metadata_sha256 is None:
                source_metadata_sha256 = source_hash
            elif source_metadata_sha256 != source_hash:
                raise ValueError(f"{strategy} source perception metadata mismatch for {video_id}")
            strategy_metadata[f"{strategy}_assessment_metadata_sha256"] = _sha256(
                directory / "run_metadata.json"
            )
        if len(base["frames"]) != len(v2["frames"]):
            raise ValueError(f"Frame count mismatch in {video_id}")
        cases = _association_changes(video_id, base, v2)
        case_rows.extend(cases)

        base_signatures, v2_signatures = (
            set(base["event_signatures"].values()),
            set(v2["event_signatures"].values()),
        )
        for strategy, run, other_signatures in (
            ("baseline_only", base, v2_signatures),
            ("experimental_only", v2, base_signatures),
        ):
            for change in run["changes"]:
                if change["event_kind"] == "risk" and change["action"] == "created":
                    signature = run["event_signatures"].get(change["event_id"])
                    if signature and signature not in other_signatures:
                        creation_changes.append(
                            {
                                "video_id": video_id,
                                "strategy": strategy,
                                "timestamp_ms": change["timestamp_ms"],
                                "event_id": change["event_id"],
                                "track_id": change["source_track_id"],
                                "class_id_at_creation": change["snapshot"]["class_id"],
                                "signature": signature,
                                "interpretation": (
                                    "event_creation_pattern_differs; physical identity is "
                                    "unverified"
                                ),
                            }
                        )
        base_alerts, v2_alerts = _alerts_by_signature(base), _alerts_by_signature(v2)
        for key in sorted(base_alerts.keys() | v2_alerts.keys(), key=str):
            baseline_records, experimental_records = (
                base_alerts.get(key, []),
                v2_alerts.get(key, []),
            )
            records_differ = [_alert_behavior_signature(alert) for alert in baseline_records] != [
                _alert_behavior_signature(alert) for alert in experimental_records
            ]
            if len(baseline_records) != len(experimental_records) or records_differ:
                alert_changes.append(
                    {
                        "video_id": video_id,
                        "event_signature": key[0],
                        "alert_type": key[1],
                        "baseline_count": len(baseline_records),
                        "experimental_count": len(experimental_records),
                        "alert_records_differ": records_differ,
                        "timestamp_deltas_ms_by_order": (
                            [
                                right["timestamp_ms"] - left["timestamp_ms"]
                                for left, right in zip(
                                    baseline_records, experimental_records, strict=True
                                )
                            ]
                            if len(baseline_records) == len(experimental_records)
                            else None
                        ),
                        "baseline_alerts": baseline_records,
                        "experimental_alerts": experimental_records,
                    }
                )
        videos.append(
            {
                "video_id": video_id,
                "duration_ms": duration_ms,
                "input_tracks_sha256": tracks_sha256,
                "source_perception_metadata_sha256": source_metadata_sha256,
                **strategy_metadata,
                "baseline": {
                    key: value
                    for key, value in base.items()
                    if key not in ("frames", "changes", "alerts")
                },
                "experimental": {
                    key: value
                    for key, value in v2.items()
                    if key not in ("frames", "changes", "alerts")
                },
                "association_changes": len(cases),
            }
        )

    provenance = {
        "manifest_sha256": _sha256(args.manifest),
        "baseline_config_sha256": _sha256(args.baseline_config),
        "experimental_config_sha256": _sha256(args.experimental_config),
        "comparator_sha256": _sha256(Path(__file__)),
        "ground_truth_used": False,
    }
    total_duration_ms = sum(video["duration_ms"] for video in videos)
    aggregate = {}
    for strategy in ("baseline", "experimental"):
        aggregate[strategy] = {
            "risk_event_count": sum(video[strategy]["risk_event_count"] for video in videos),
            "first_alerts": sum(video[strategy]["first_alerts"] for video in videos),
            "escalation_alerts": sum(video[strategy]["escalation_alerts"] for video in videos),
            "alert_count": sum(video[strategy]["alert_count"] for video in videos),
            "alerts_per_minute": (
                sum(video[strategy]["alert_count"] for video in videos)
                / (total_duration_ms / 60_000)
                if total_duration_ms
                else None
            ),
            "same_frame_track_competitions": sum(
                video[strategy]["same_frame_track_competitions"] for video in videos
            ),
            "cross_event_track_conflict_observations": sum(
                video[strategy]["cross_event_track_conflict_observations"] for video in videos
            ),
            "new_event_conflicts": sum(video[strategy]["new_event_conflicts"] for video in videos),
            "suspected_overlapping_same_track_first_alert_pairs": sum(
                video[strategy]["suspected_overlapping_same_track_first_alert_pairs"]
                for video in videos
            ),
            "events_with_track_id_changes": sum(
                video[strategy]["events_with_track_id_changes"] for video in videos
            ),
            "association_timing": _timing_summary(all_timings[strategy]),
        }
    report = {
        "schema_version": "sightguard.event-association-comparison.v1",
        "provenance": provenance,
        "aggregate": aggregate,
        "videos": videos,
        "event_creation_differences": creation_changes,
        "alert_differences": alert_changes,
    }
    (args.report_dir / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.report_dir / "association_cases.jsonl").open(
        "x", encoding="utf-8", newline="\n"
    ) as stream:
        for case in case_rows:
            stream.write(json.dumps(case, ensure_ascii=False) + "\n")
    fieldnames = (
        "video_id",
        "strategy",
        "risk_event_count",
        "first_alerts",
        "escalation_alerts",
        "alerts_per_minute",
        "same_frame_track_competitions",
        "cross_event_track_conflict_observations",
        "new_event_conflicts",
        "first_alert_same_track_repeats",
        "suspected_overlapping_same_track_first_alert_pairs",
        "events_with_track_id_changes",
        "association_calls",
        "association_mean_ms",
        "association_p95_ms",
        "association_max_ms",
    )
    with (args.report_dir / "summary.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for video in videos:
            for strategy_name in ("baseline", "experimental"):
                values = video[strategy_name]
                timing = values["association_timing"]
                writer.writerow(
                    {
                        "video_id": video["video_id"],
                        "strategy": strategy_name,
                        "risk_event_count": values["risk_event_count"],
                        "first_alerts": values["first_alerts"],
                        "escalation_alerts": values["escalation_alerts"],
                        "alerts_per_minute": values["alerts_per_minute"],
                        "same_frame_track_competitions": values["same_frame_track_competitions"],
                        "cross_event_track_conflict_observations": values[
                            "cross_event_track_conflict_observations"
                        ],
                        "new_event_conflicts": values["new_event_conflicts"],
                        "first_alert_same_track_repeats": values["first_alert_same_track_repeats"],
                        "suspected_overlapping_same_track_first_alert_pairs": values[
                            "suspected_overlapping_same_track_first_alert_pairs"
                        ],
                        "events_with_track_id_changes": values["events_with_track_id_changes"],
                        "association_calls": timing["calls"],
                        "association_mean_ms": timing["mean_ms"],
                        "association_p95_ms": timing["p95_ms"],
                        "association_max_ms": timing["max_ms"],
                    }
                )
    print(
        json.dumps(
            {
                "report": str(args.report_dir),
                "videos": len(videos),
                "association_cases": len(case_rows),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
