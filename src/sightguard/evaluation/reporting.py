"""Portable CSV/JSON reports and static scientific figures (no UI dependencies)."""

import csv
import json
from pathlib import Path
from typing import Any

from sightguard.evaluation.dataset import DatasetVideo
from sightguard.evaluation.schema import EvaluationResult, StrategyAlert


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def flat_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "response_delays_ms"}


def render_figures(
    output: Path,
    videos: tuple[DatasetVideo, ...],
    alerts: tuple[StrategyAlert, ...],
    results: tuple[EvaluationResult, ...],
    comparison: dict[str, dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = output / "figures"
    directory.mkdir()
    strategies = tuple(comparison)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for axis, (field, title) in zip(
        axes,
        (
            ("event_recall", "Important event recall"),
            ("opportunity_recall", "Opportunity recall"),
            ("invalid_alerts_per_hour", "Invalid alerts / video hour"),
        ),
        strict=True,
    ):
        values = [comparison[name][field] for name in strategies]
        axis.bar(strategies, [0 if value is None else value for value in values])
        axis.set_title(title)
        for i, value in enumerate(values):
            if value is None:
                axis.text(i, 0, "N/A", ha="center")
        if "recall" in field:
            axis.set_ylim(0, 1.05)
    kind = ",".join(sorted({video.entry.data_kind for video in videos}))
    fig.suptitle(f"SightGuard offline evaluation ({kind}); no real-world safety claim")
    for suffix in ("png", "svg"):
        fig.savefig(directory / f"comparison.{suffix}", dpi=150)
    plt.close(fig)
    colors = {
        "matched": "green",
        "duplicate": "orange",
        "false_positive": "red",
        "ignored": "gray",
        "perception_limited": "purple",
    }
    for video in videos:
        fig, axis = plt.subplots(figsize=(11, 4), constrained_layout=True)
        for event in video.labels.events:
            for opportunity in event.alert_opportunities:
                axis.plot(
                    [opportunity.window_start_ms / 1000, opportunity.deadline_ms / 1000],
                    [0, 0],
                    linewidth=8,
                    alpha=0.35,
                    color="green",
                )
                axis.annotate(
                    f"{event.event_id}:{opportunity.opportunity_id}",
                    (opportunity.window_start_ms / 1000, 0),
                    fontsize=7,
                )
        for index, name in enumerate(strategies, 1):
            result = next(
                row
                for row in results
                if row.video_id == video.entry.video_id and row.strategy == name
            )
            status = {row.alert_id: row.status for row in result.matches if row.alert_id}
            opportunity_lookup = {
                (event.event_id, opportunity.opportunity_id): opportunity
                for event in video.labels.events
                for opportunity in event.alert_opportunities
            }
            for match in result.matches:
                if match.status == "missed_opportunity":
                    opportunity = opportunity_lookup[(match.gt_event_id, match.opportunity_id)]
                    axis.scatter(opportunity.deadline_ms / 1000, index, c="black", marker="x")
            for alert in alerts:
                if alert.video_id == video.entry.video_id and alert.strategy == name:
                    axis.scatter(
                        alert.timestamp_ms / 1000,
                        index,
                        c=colors.get(status[alert.alert_id], "black"),
                        marker="|",
                    )
        axis.set_yticks(range(len(strategies) + 1), ["GT opportunities", *strategies])
        axis.set_xlim(0, video.entry.duration_ms / 1000)
        axis.set_xlabel("Video time (seconds)")
        axis.set_title(f"{video.entry.video_id} ({video.entry.data_kind})")
        for suffix in ("png", "svg"):
            fig.savefig(directory / f"timeline_{video.entry.video_id}.{suffix}", dpi=150)
        plt.close(fig)
