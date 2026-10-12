"""Bounded ADB batches keep locked-device tests short; no wake, unlock, or power-policy changes."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def run_recorded(command: list[str], log_path: Path) -> subprocess.CompletedProcess[bytes]:
    """A stalled test must not leave an unattended instrumentation process running."""
    try:
        result = subprocess.run(command, capture_output=True, timeout=100)
    except subprocess.TimeoutExpired as error:
        log_path.write_bytes((error.stdout or b"") + (error.stderr or b"") + b"\nHOST_TIMEOUT\n")
        subprocess.run(
            [*command[:3], "shell", "am", "force-stop", "org.sightguard.app"],
            check=True,
            timeout=20,
        )
        raise RuntimeError("Instrumentation timed out; only this app was stopped") from error
    log_path.write_bytes(result.stdout + result.stderr)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--adb", default="D:/Android/Sdk/platform-tools/adb.exe")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    command = [args.adb, "-s", args.serial]
    reports = []
    for start in range(0, len(manifest["cases"]), args.batch_size):
        count = min(args.batch_size, len(manifest["cases"]) - start)
        run = run_recorded(
            [
                *command,
                "shell",
                "am",
                "instrument",
                "-w",
                "-r",
                "-e",
                "case_start",
                str(start),
                "-e",
                "case_count",
                str(count),
                "-e",
                "class",
                "org.sightguard.app.FurnitureDatasetInstrumentedTest",
                "org.sightguard.app.test/androidx.test.runner.AndroidJUnitRunner",
            ],
            args.output / f"batch-{start:03d}.log",
        )
        if run.returncode != 0 or b"OK (1 test)" not in run.stdout:
            raise RuntimeError(
                f"Android batch {start} failed; preserved output, no success inferred"
            )
        data = subprocess.check_output(
            [
                *command,
                "exec-out",
                "run-as",
                "org.sightguard.app",
                "cat",
                "files/furniture-dataset-actual.json",
            ],
            timeout=20,
        )
        (args.output / f"batch-{start:03d}.json").write_bytes(data)
        report = json.loads(data)
        if report["case_start"] != start or len(report["cases"]) != count:
            raise RuntimeError("Missing or stale batch report")
        reports.append(report)
        print(f"completed {start + count}/{len(manifest['cases'])}", flush=True)
    merged = dict(reports[0])
    merged["cases"] = [case for report in reports for case in report["cases"]]
    merged["audio"] = [event for report in reports for event in report["audio"]]
    merged["batch_model_load_ms"] = [report["model_load_ms"] for report in reports]
    merged["protocol"] = (
        "Separate short instrumentation processes; not a continuous phone lifecycle test"
    )
    merged["batch_size"] = args.batch_size
    (args.output / "combined.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
