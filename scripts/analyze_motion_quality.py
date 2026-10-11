"""Supplementary saved-evidence associations and real-video failure contact sheet."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from sightguard.perception.depth.artifacts import sha256_file, write_json
from sightguard.scene.spatial_grid.artifacts import fresh_directory, read_rows


def association(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    result = {}
    for value in sorted({x[field] for x in rows}):
        subset = [x for x in rows if x[field] == value]
        result[str(value)] = {
            "pairs": len(subset),
            "lowest_cost_changed": sum(x["change"]["lowest_cost_changed"] for x in subset),
            "status_changed": sum(x["change"]["status_changed"] for x in subset),
            "best_identity_or_null_changed": sum(
                x["change"]["best_supported_identity_changed"] for x in subset
            ),
        }
    return result


def analyze(motion: Path, temporal: Path, output: Path) -> dict[str, Any]:
    fresh_directory(output)
    metadata = json.loads((motion / "run_metadata.json").read_text(encoding="utf-8"))
    samples, cases, tiles, windows, input_hashes = [], [], [], [], {}
    videos = []
    for item in metadata["videos"]:
        video_id = item["video_id"]
        motion_path = motion / video_id / "frame_motion.jsonl"
        temporal_path = temporal / video_id / "temporal_corridor.jsonl"
        data = read_rows(motion_path)
        joined = read_rows(temporal_path)
        input_hashes[str(motion_path)] = sha256_file(motion_path)
        input_hashes[str(temporal_path)] = sha256_file(temporal_path)
        source = data[0]["provenance"]
        tracks = read_rows(Path(source["tracks_reference"]))
        detected = {
            x["frame_index"]: sum(
                d["confidence"] >= data[0]["config"]["detection_confidence"]
                for d in x["detections"]
            )
            for x in tracks
        }
        valid = [x for x in data if x["estimation_status"] == "VALID"]
        video_windows = read_rows(motion / video_id / "motion_windows.jsonl")
        windows.extend(video_windows)
        videos.append(
            {
                "video_id": video_id,
                "window_labels": dict(Counter(x["quality"]["label"] for x in video_windows)),
                "valid_frame_rate_p50": float(
                    np.median([x["motion_diagonals_per_second"] for x in valid])
                ),
                "valid_frame_rate_p95": float(
                    np.percentile([x["motion_diagonals_per_second"] for x in valid], 95)
                ),
                "confident_box_count_p50": float(np.median(list(detected.values()))),
                "confident_box_count_p95": float(np.percentile(list(detected.values()), 95)),
            }
        )
        for row in joined:
            if row["ablation"] not in "BC" or row["change"]["types"] == ["INITIAL_OBSERVATION"]:
                continue
            rate = row["motion_window"]["motion_rate_p95"]
            samples.append(
                {
                    "ablation": row["ablation"],
                    "change": row["change"],
                    "rate_bin": "UNAVAILABLE"
                    if rate is None
                    else "AT_LEAST_0.5_DIAG_S"
                    if rate >= 0.5
                    else "BELOW_0.5_DIAG_S",
                    "detection_bin": "AT_LEAST_3_BOXES"
                    if detected[row["key"]["frame_index"]] >= 3
                    else "FEWER_THAN_3_BOXES",
                }
            )
        unknown = next((x for x in data[1:] if x["quality"]["label"] == "UNKNOWN"), data[0])
        sampled = next(
            (
                x
                for x in data[10:]
                if x["key"]["frame_index"] % 10 == 0 and x["quality"]["label"] == "LIMITED"
            ),
            data[100],
        )
        cap = cv2.VideoCapture(source["video_reference"])
        try:
            for selected in (unknown, sampled):
                key = selected["key"]
                cap.set(cv2.CAP_PROP_POS_FRAMES, key["frame_index"])
                ok, image = cap.read()
                if not ok or not np.isclose(
                    cap.get(cv2.CAP_PROP_POS_MSEC), key["timestamp_ms"], atol=0.01, rtol=0
                ):
                    raise ValueError("failure preview source PTS mismatch")
                for point in selected["flow_samples"]:
                    color = (90, 230, 90) if point["inlier"] else (70, 70, 240)
                    a, b = (
                        tuple(round(x) for x in point["previous_xy"]),
                        tuple(round(x) for x in point["current_xy"]),
                    )
                    cv2.line(image, a, b, color, 2)
                    cv2.circle(image, b, 3, color, -1)
                name = f"{video_id}_{key['frame_index']:06d}.jpg"
                if not cv2.imwrite(str(output / name), image):
                    raise OSError("failure preview write failed")
                tile = np.full((325, 480, 3), 24, dtype=np.uint8)
                tile[55:] = cv2.resize(image, (480, 270))
                cv2.putText(
                    tile,
                    f"{video_id[-2:]} f{key['frame_index']} {selected['quality']['label']}",
                    (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (230, 230, 230),
                    1,
                )
                cv2.putText(
                    tile,
                    ",".join(selected["quality"]["reasons"])[:62],
                    (8, 43),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.36,
                    (190, 200, 240),
                    1,
                )
                tiles.append(tile)
                cases.append(
                    {
                        "key": key,
                        "image": name,
                        "quality": selected["quality"],
                        "matches": selected["match_count"],
                        "inlier_ratio": selected["inlier_ratio"],
                        "background_available_fraction": selected["background_available_fraction"],
                        "video_sha256": source["video_sha256"],
                        "motion_source": str(motion_path),
                    }
                )
        finally:
            cap.release()
    sheet = np.vstack([np.hstack(tiles[i : i + 4]) for i in range(0, len(tiles), 4)])
    if not cv2.imwrite(str(output / "failure_contact_sheet.jpg"), sheet):
        raise OSError("contact sheet write failed")
    result = {
        "schema_version": "sightguard.motion-analysis.v1",
        "videos": videos,
        "window_quality": dict(Counter(x["quality"]["label"] for x in windows)),
        "associations": {
            name: {
                "by_observed_window_rate": association(
                    [x for x in samples if x["ablation"] == name], "rate_bin"
                ),
                "by_confident_detection_count": association(
                    [x for x in samples if x["ablation"] == name], "detection_bin"
                ),
            }
            for name in "BC"
        },
        "cases": cases,
        "source_sha256": input_hashes,
        "script_sha256": sha256_file(Path(__file__)),
        "warning": (
            "quality and detections are proxies, not causal attribution "
            "or independently verified passability"
        ),
    }
    write_json(output / "analysis.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, default=Path("outputs/motion_quality/v01"))
    parser.add_argument("--temporal", type=Path, default=Path("outputs/temporal_corridor/v01"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.motion, args.temporal, args.output)
    print(
        json.dumps(
            {name: result[name] for name in ("videos", "window_quality", "associations", "cases")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
