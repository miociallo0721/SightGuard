"""Publication-style figures from fixed public samples; candidate colors never mean safety."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle

from sightguard.research.resources import check_resources, lower_own_priority
from sightguard.research.settings import research_cache_root
from sightguard.research.wall_metric import WallMetricEstimator
from sightguard.scene.spatial_grid.schema import FrameKey

ROOT = Path(__file__).resolve().parents[1]
DELIVERY = ROOT / "android/deliveries/metric-corridor-research"


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    lower_own_priority()
    plt.switch_backend("Agg")
    check_resources(output_dir=DELIVERY)
    cv2.setNumThreads(1)
    cache = research_cache_root()
    samples = []
    for name in ["sample-manifest.json", "extension-manifest.json"]:
        samples.extend(json.loads((cache / name).read_text(encoding="utf-8"))["samples"])
    selected = [
        [s for s in samples if s["sequence"] == sequence][50]
        for sequence in ["freiburg1_rpy", "freiburg3_nostructure_notexture_far"]
    ]
    fig, axes = plt.subplots(2, 5, figsize=(16, 6.9), constrained_layout=True)
    estimator = WallMetricEstimator()
    for row, sample in enumerate(selected):
        folder = DELIVERY / "evaluation"
        check_resources(output_dir=DELIVERY)
        image = cv2.cvtColor(cv2.imread(sample["rgb_path"]), cv2.COLOR_BGR2RGB)
        truth = cv2.imread(sample["depth_path"], cv2.IMREAD_UNCHANGED).astype(np.float32) / 5000
        with np.load(folder / "semantics" / f"{sample['sample_id']}.npz") as data:
            key = FrameKey(
                sample["sequence"], sample["rgb_index"], sample["timestamp_ms"], 640, 480
            )
            wall, _, coverage, _ = estimator.mask(
                key, data["classes"], data["confidence"], data["foreground"]
            )
        axes[row, 0].imshow(image)
        axes[row, 0].set_title(sample["sample_id"], fontsize=9)
        axes[row, 1].imshow(image)
        overlay = np.zeros((*wall.shape, 4))
        overlay[wall] = [0.98, 0.43, 0.1, 0.6]
        axes[row, 1].imshow(overlay)
        axes[row, 1].set_title(f"Predicted central wall, coverage={coverage:.2f}", fontsize=9)
        depth_plot = axes[row, 2].imshow(
            np.ma.masked_where(truth <= 0, truth), vmin=0, vmax=5, cmap="magma"
        )
        axes[row, 2].set_title("Registered RGB-D truth Z (m)", fontsize=9)
        for col, size in [(3, 280), (4, 518)]:
            with np.load(folder / f"hypersim_{size}" / f"{sample['sample_id']}.npz") as data:
                prediction = data["depth"]
            valid = wall & (truth > 0)
            mae = np.abs(prediction[valid] - truth[valid]).mean() if valid.any() else np.nan
            result = next(
                r
                for r in read_rows(folder / f"metric_hypersim_{size}.jsonl")
                if r["sample_id"] == sample["sample_id"]
            )
            axes[row, col].imshow(prediction, vmin=0, vmax=5, cmap="magma")
            error_text = f"mask MAE={mae:.2f}m" if np.isfinite(mae) else "no valid wall pixels"
            axes[row, col].set_title(
                f"Hypersim {size} raw Z; {error_text}\n{result['wall']['status']}", fontsize=9
            )
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])
    fig.colorbar(
        depth_plot,
        ax=axes[:, 2:].ravel().tolist(),
        shrink=0.7,
        label="Optical-axis Z (m), display clipped at 5m",
    )
    fig.suptitle(
        (
            "Fixed uniform sample #50 in two sequences; predicted mask ha"
            "s NO semantic ground truth\nTUM RGB-D / Sturm et al., IROS 2"
            "012, CC BY 4.0. No phone tape-distance validation."
        ),
        fontsize=11,
    )
    fig.savefig(DELIVERY / "metric_depth_examples.png", dpi=150)
    plt.close(fig)

    corridors = read_rows(DELIVERY / "corridor_candidates.jsonl")
    chosen = [
        next(r for r in corridors if r["status"] == status)
        for status in ["SUPPORTED_CANDIDATE", "TIED"]
    ]
    chosen.append(
        next(
            r
            for r in corridors
            if any(
                "known_obstacle_covers_lower_band" in c["rejection_reasons"]
                for c in r["candidates"]
            )
        )
    )
    by_id = {s["sample_id"]: s for s in samples}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for axis, record in zip(axes, chosen, strict=True):
        sample = by_id[record["sample_id"]]
        image = cv2.cvtColor(cv2.imread(sample["rgb_path"]), cv2.COLOR_BGR2RGB)
        folder = DELIVERY / (
            "evaluation"
            if sample["sequence"] in {s["sequence"] for s in selected}
            or sample["sequence"] == "freiburg1_xyz"
            else "evaluation-extension"
        )
        source = next(
            r
            for r in read_rows(folder / "semantic_results.jsonl")
            if r["sample_id"] == record["sample_id"]
        )
        axis.imshow(image)
        for candidate in record["candidates"]:
            if candidate["eligible"]:
                axis.add_patch(
                    Polygon(
                        candidate["image_polygon_xy"],
                        closed=True,
                        facecolor="#337da8",
                        edgecolor="#15577d",
                        alpha=0.22,
                        hatch="//",
                    )
                )
        for detection in source["detections"]:
            x1, y1, x2, y2 = detection["bbox_xyxy"]
            axis.add_patch(
                Rectangle(
                    (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="#d86924", linewidth=1.2
                )
            )
        axis.set_title(
            f"{record['status']} / strict D: {record['strict_d_status']}\n"
            f"{record['sample_id']}\n{record['best_candidate']} / "
            f"support={record['temporal_support']}",
            fontsize=9,
        )
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(
        (
            "First record of each reported category; blue hatched bands a"
            "re IMAGE CANDIDATES ONLY\nNo body alignment, physical width,"
            " safety conclusion, or walking instruction. TUM RGB-D, CC BY"
            " 4.0."
        ),
        fontsize=11,
    )
    fig.savefig(DELIVERY / "corridor_examples.png", dpi=150)
    plt.close(fig)
    supported = [r for r in corridors if r["status"] == "SUPPORTED_CANDIDATE"]
    fig, axes = plt.subplots(
        int(np.ceil(len(supported) / 4)),
        4,
        figsize=(12, 2.45 * int(np.ceil(len(supported) / 4))),
        constrained_layout=True,
    )
    for axis, record in zip(axes.ravel(), supported, strict=False):
        sample = by_id[record["sample_id"]]
        axis.imshow(cv2.cvtColor(cv2.imread(sample["rgb_path"]), cv2.COLOR_BGR2RGB))
        candidate = next(
            c for c in record["candidates"] if c["candidate_id"] == record["best_candidate"]
        )
        axis.add_patch(
            Polygon(
                candidate["image_polygon_xy"],
                closed=True,
                fill=False,
                edgecolor="#33b0d8",
                linewidth=1.4,
            )
        )
        axis.set_title(f"{record['sample_id']}\n{record['best_candidate']}", fontsize=6.5)
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in axes.ravel()[len(supported) :]:
        axis.set_visible(False)
    fig.suptitle(
        (
            "ALL model-supported image bands for independent visual revie"
            "w; NOT verified ground or passage\nTUM RGB-D / Sturm et al.,"
            " IROS 2012, CC BY 4.0"
        ),
        fontsize=10,
    )
    fig.savefig(DELIVERY / "corridor_all_supported_review.png", dpi=150)
    plt.close(fig)
    (DELIVERY / "figure_sources.json").write_text(
        json.dumps(
            {
                "depth_samples": [s["sample_id"] for s in selected],
                "depth_selection": "fixed_uniform_position_50_not_selected_by_error",
                "corridor_samples": [r["sample_id"] for r in chosen],
                "corridor_selection": (
                    "first_supported_first_tied_first_lower_bbox_rejection_illustration_only"
                ),
                "license": "CC-BY-4.0",
                "attribution": "TUM RGB-D / Sturm et al. IROS 2012",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Saved two figures with fixed/explicit selection and source manifest")


if __name__ == "__main__":
    main()
