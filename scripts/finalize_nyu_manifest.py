"""Verify extracted frozen NYUv2 samples without fetching more network data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extracted", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    selection_path = args.extracted / "frozen-selection.json"
    frozen = json.loads(selection_path.read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (args.extracted / "extraction.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    completed = {r["sample_id"]: r for r in records if r["status"] == "complete"}
    samples, missing = [], []
    for selected in frozen["samples"]:
        sid = selected["sample_id"]
        if sid not in completed:
            missing.append(sid)
            continue
        record = completed[sid]
        if any(record[key] != selected[key] for key in selected):
            raise ValueError("extracted scene differs from frozen selection")
        for name in ("rgb", "gt"):
            if digest(Path(record[f"{name}_path"])) != record[f"{name}_sha256"]:
                raise ValueError("extracted artifact checksum mismatch")
        samples.append(
            {
                **record,
                "dataset": "NYU",
                "sequence": record["scene"],
                "rgb_index": record["index"],
                "timestamp_ms": 0.0,
                "time_contract": "unrelated_still_separate_session_no_temporal_claim",
                "intrinsic": [
                    518.85790117450188,
                    519.46961112127485,
                    325.58244941119034,
                    253.73616633400465,
                ],
                "intrinsic_source": "official_toolbox_nyu_depth_v2/camera_params.m",
                "distortion_correction": "not_applied_published_registered_rgb_used",
            }
        )
    if missing and not args.allow_partial:
        raise ValueError(f"frozen samples missing: {missing}")
    if not samples:
        raise ValueError("no verified samples")
    result = {
        **frozen,
        "frozen_selection_sha256": digest(selection_path),
        "planned_samples": len(frozen["samples"]),
        "completed_samples": len(samples),
        "missing_samples": missing,
        "extraction_failures": [r for r in records if r["status"] != "complete"],
        "samples": samples,
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({"planned": result["planned_samples"], "complete": len(samples)}))


if __name__ == "__main__":
    main()
