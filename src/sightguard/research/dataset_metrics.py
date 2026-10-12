"""Independent pixel-weighted depth/semantic evaluation; never model inputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DepthErrorTotals:
    valid_truth: int = 0
    compared: int = 0
    absolute_sum: float = 0
    squared_sum: float = 0
    relative_sum: float = 0

    def add(self, prediction: np.ndarray, truth: np.ndarray, region: np.ndarray) -> None:
        if prediction.shape != truth.shape or region.shape != truth.shape or region.dtype != bool:
            raise ValueError("aligned depth and boolean evaluation region required")
        valid_truth = region & np.isfinite(truth) & (truth > 0)
        valid = valid_truth & np.isfinite(prediction) & (prediction > 0)
        self.valid_truth += int(np.count_nonzero(valid_truth))
        self.compared += int(np.count_nonzero(valid))
        error = np.abs(prediction[valid].astype(np.float64) - truth[valid])
        self.absolute_sum += float(error.sum())
        self.squared_sum += float(np.square(error).sum())
        self.relative_sum += float((error / truth[valid]).sum())

    def summary(self) -> dict:
        count = self.compared
        return {
            "valid_truth_pixels": self.valid_truth,
            "compared_pixels": count,
            "valid_output_fraction": count / self.valid_truth if self.valid_truth else None,
            "mae_m": self.absolute_sum / count if count else None,
            "rmse_m": (self.squared_sum / count) ** 0.5 if count else None,
            "abs_rel": self.relative_sum / count if count else None,
        }


def binary_semantic_counts(predicted: np.ndarray, truth: np.ndarray, labeled: np.ndarray) -> dict:
    if predicted.shape != truth.shape or labeled.shape != truth.shape:
        raise ValueError("semantic evaluation geometry mismatch")
    tp = int(np.count_nonzero(predicted & truth & labeled))
    fp = int(np.count_nonzero(predicted & ~truth & labeled))
    fn = int(np.count_nonzero(~predicted & truth & labeled))
    return {"true_positive": tp, "false_positive": fp, "false_negative": fn}


def summarize_semantic_counts(counts: dict) -> dict:
    tp, fp, fn = (counts[name] for name in ("true_positive", "false_positive", "false_negative"))
    return {
        **counts,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "iou": tp / (tp + fp + fn) if tp + fp + fn else None,
    }
