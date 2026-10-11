"""Validate corridor artifacts and independently replay unchanged baseline alerts."""

import argparse
import json
import math
from pathlib import Path

from validate_spatial_grid import validate as validate_baseline

from sightguard.perception.depth.artifacts import sha256_file, write_json
from sightguard.scene.spatial_grid.artifacts import fresh_directory, read_rows


def validate(output: Path, corridor: Path, spatial: Path, frozen: Path) -> dict:
    fresh_directory(output)
    metadata = json.loads((corridor / "run_metadata.json").read_text(encoding="utf-8"))
    if metadata["status"] != "completed" or metadata["torch_loaded"]:
        raise ValueError("incomplete or model-loading corridor run")
    for reference, digest in metadata["output_files_sha256"].items():
        if sha256_file(corridor / reference) != digest:
            raise ValueError("corridor output integrity failure")
    for reference, digest in metadata["source_sha256"].items():
        if sha256_file(Path(reference)) != digest:
            raise ValueError("corridor producer code hash mismatch")
    videos = []
    total_relations = 0
    for video in metadata["videos"]:
        video_id = video["video_id"]
        directory = corridor / video_id
        frames = read_rows(directory / "corridor_frames.jsonl")
        original = read_rows(spatial / video_id / "spatial_frames.jsonl")
        groups = {name: [x for x in frames if x["ablation"] == name] for name in "ABCD"}
        expected_keys = [x["key"] for x in original]
        for rows in groups.values():
            assert [x["key"] for x in rows] == expected_keys
        for position in range(len(original)):
            candidates = None
            for name in "ABCD":
                row = groups[name][position]
                assert not row["production_decision_changed"]
                geometry = [x["candidate"] for x in row["candidates"]]
                assert candidates is None or candidates == geometry
                candidates = geometry
                for candidate in row["candidates"]:
                    total = candidate["total_cost"]
                    assert (total is None) == (candidate["evaluation_status"] == "INELIGIBLE")
                    if total is not None:
                        assert math.isfinite(total)
                        assert math.isclose(
                            total,
                            sum(
                                x
                                for x in candidate["weighted_contributions"].values()
                                if x is not None
                            ),
                            abs_tol=1e-12,
                        )
                    if name in "AB":
                        assert candidate["cost_components"]["depth"] is None
                    for relation in candidate["object_relations"]:
                        assert relation["key"] == row["key"] and relation["event_reference"] is None
                if row["status"] in ("ABSTAIN", "TIED"):
                    assert row["best_supported_candidate"] is None
        relations = read_rows(directory / "object_relations.jsonl")
        total_relations += len(relations)
        assert len(frames) == 120
        assert len(read_rows(directory / "candidate_corridors.jsonl")) == 360
        videos.append(
            {
                "video_id": video_id,
                "source_frames": len(original),
                "assessment_rows": len(frames),
                "object_corridor_relation_rows_all_ablations": len(relations),
            }
        )
    baseline = validate_baseline(output / "baseline", spatial, frozen)
    result = {
        "status": "passed",
        "videos": videos,
        "source_frames": sum(x["source_frames"] for x in videos),
        "assessment_rows": sum(x["assessment_rows"] for x in videos),
        "object_corridor_relation_rows_all_ablations": total_relations,
        "baseline_frames": baseline["total_frames"],
        "baseline_alerts": baseline["total_alerts"],
        "frozen_files_checked": baseline["frozen_files_checked"],
        "frozen_mismatches": baseline["frozen_mismatches"],
        "risk_events_alerts_equal": True,
        "production_audio_called": False,
        "production_decision_changed": False,
        "ground_truth_navigation_validated": False,
        "program_checks": [
            "exact frame identities",
            "same geometry in ABCD",
            "nullable eligibility costs",
            "cost sums",
            "source/output hashes",
            "unchanged risk/events/alerts replay",
        ],
        "browser_checks": "recorded separately after real browser inspection",
    }
    write_json(output / "validation.json", result)
    write_json(corridor / "validation.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corridor", type=Path, default=Path("outputs/corridor_intelligence/v01"))
    parser.add_argument("--spatial", type=Path, default=Path("outputs/spatial_grid/v03"))
    parser.add_argument(
        "--frozen", type=Path, default=Path("outputs/corridor_audit/v01/frozen_hashes.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(validate(args.output, args.corridor, args.spatial, args.frozen))


if __name__ == "__main__":
    main()
