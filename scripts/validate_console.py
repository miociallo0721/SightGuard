"""Opt-in local HTTP acceptance checks; never downloads model weights."""

import argparse
import json
from pathlib import Path

from gradio_client import Client, handle_file
from gradio_client.exceptions import AppError

from sightguard.evaluation.dataset import sha256_file
from sightguard.ui.services import ConsoleStore


def hashes(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)): sha256_file(path)
        for path in directory.rglob("*")
        if path.is_file()
    }


def find_table(result, columns: set[str]) -> dict:
    for value in result if isinstance(result, tuple) else (result,):
        if isinstance(value, dict) and columns.issubset(value.get("headers", ())):
            return value
    raise AssertionError(f"Missing expected table {columns}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:7860")
    parser.add_argument("--benchmark-run", default="phase3_final")
    parser.add_argument("--analysis-run", default="tracking_verified")
    parser.add_argument("--run-analysis", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/console_validation/report.json")
    )
    args = parser.parse_args()
    store = ConsoleStore(Path.cwd())
    benchmark_view = store.load_benchmark(args.benchmark_run)
    source = store.root / "outputs" / args.benchmark_run
    before = hashes(source)
    client = Client(args.url, verbose=False, analytics_enabled=False, download_files=False)
    checks = []
    analytics = client.predict(args.benchmark_run, "opportunity_recall", api_name="/load_analytics")
    table = find_table(analytics, {"strategy", "tp", "fp", "fn"})
    for row in table["data"]:
        name = row[table["headers"].index("strategy")]
        for field in ("tp", "fp", "fn", "event_recall", "opportunity_recall"):
            assert row[table["headers"].index(field)] == benchmark_view.metrics[name][field]
    checks.append("HTTP analytics counts/recalls equal stored metrics")
    comparison = client.predict(args.benchmark_run, api_name="/load_comparison")
    assert "Synthetic Data" in comparison[0]
    first_alerts = find_table(comparison, {"strategy", "alert_id", "match_status"})
    assert set(row[first_alerts["headers"].index("strategy")] for row in first_alerts["data"]) == {
        "A",
        "B",
        "C",
    }
    missed = client.predict("test_missed", ["A", "B", "C"], api_name="/comparison_video")
    missed_table = find_table(missed, {"status", "opportunity_id"})
    assert len(missed_table["data"]) == 3
    assert all(
        row[missed_table["headers"].index("status")] == "missed_opportunity"
        for row in missed_table["data"]
    )
    checks.append("HTTP comparison preserves the complete perception miss")
    errors = client.predict("C", "test_missed", api_name="/filter_errors")
    assert len(errors["data"]) == 1
    assert errors["data"][0][errors["headers"].index("status")] == "missed_opportunity"
    client.predict("synthetic/manifest.csv", api_name="/preview_videos")
    preview = client.predict(
        "synthetic/manifest.csv",
        "test_escalation",
        ["A", "B", "C"],
        "benchmark.yaml",
        "risk.yaml",
        api_name="/preview_strategies",
    )
    assert "不提供 TP/FP/FN" in preview[0]
    checks.append("HTTP strategy preview is unscored without Ground Truth")
    dataset = client.predict("synthetic/manifest.csv", api_name="/browse_dataset")
    assert len(find_table(dataset, {"video_id", "split"})["data"]) == 5
    validation = client.predict("synthetic/manifest.csv", api_name="/validate_dataset")
    assert "失败" not in validation and "could not" not in validation.lower()
    checks.append("HTTP manifest browsing and existing validator execute")
    analysis = client.predict(args.analysis_run, api_name="/load_analysis")
    assert "已加载" in analysis[0]
    frame = client.predict(305, api_name="/select_frame")
    assert "frame 3" in frame[-1] and "300.000 ms" in frame[-1]
    view = store.load_analysis(args.analysis_run)
    milestone = next(row for row in view.events if row.get("state") == "ACTIVE")
    located = client.predict(
        f"{milestone['event_id']}|{milestone['timestamp_ms']}", api_name="/locate_event"
    )
    assert located[0] == milestone["timestamp_ms"]
    client.predict(api_name="/export_analysis")
    try:
        invalid = client.predict("../../AGENTS.md", api_name="/load_analysis")
    except AppError as error:
        # Gradio rejects unregistered dropdown values before the callback runs.
        assert "not in the list of choices" in str(error)
    else:
        assert "操作失败" in invalid[0]
    checks.append("HTTP frame/event selection, ZIP export and invalid-path handling execute")
    generated_run = None
    if args.run_analysis:
        existing = set(store.analysis_runs())
        fixture = store.root / "examples" / "tracking_fixture.mp4"
        original_hash = sha256_file(fixture)
        result = client.predict(
            handle_file(str(fixture)), "default.yaml", "risk.yaml", api_name="/analyze_video"
        )
        assert "已加载" in result[0], result[0]
        added = set(store.analysis_runs()) - existing
        assert len(added) == 1
        generated_run = added.pop()
        generated = store.load_analysis(generated_run)
        assert generated.records and generated.risk_frames
        assert generated.stats["detection_count"] > 0
        assert sha256_file(fixture) == original_hash
        checks.append("HTTP MP4 upload invokes real CPU YOLO/BoT-SORT and Phase 2 replay")
    assert hashes(source) == before
    checks.append("All existing benchmark files remained byte-identical")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "status": "passed",
                "checks": checks,
                "generated_analysis_run": generated_run,
                "limitation": "Repeated bus.jpg fixture; not a real dynamic road evaluation",
            },
            stream,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Passed {len(checks)} HTTP acceptance groups; report: {args.output}")


if __name__ == "__main__":
    main()
