"""Independent reference values, undefined metrics and invalid-output accounting."""

import numpy as np
import pytest

from sightguard.research.dataset_metrics import (
    DepthErrorTotals,
    binary_semantic_counts,
    summarize_semantic_counts,
)


def test_pixel_weighted_totals_and_invalid_prediction_coverage():
    totals = DepthErrorTotals()
    totals.add(np.array([2.0, 2.0, np.nan, 8.0]), np.array([1.0, 4.0, 3.0, 0.0]), np.ones(4, bool))
    totals.add(np.array([4.0]), np.array([2.0]), np.ones(1, bool))
    result = totals.summary()
    assert result["valid_truth_pixels"] == 4
    assert result["valid_output_fraction"] == 0.75
    assert result["mae_m"] == pytest.approx(5 / 3)
    assert result["rmse_m"] == pytest.approx(3**0.5)
    assert result["abs_rel"] == pytest.approx(2.5 / 3)


def test_empty_truth_is_undefined_not_zero_error():
    totals = DepthErrorTotals()
    totals.add(np.ones(2), np.zeros(2), np.ones(2, bool))
    assert totals.summary()["mae_m"] is None
    assert totals.summary()["valid_output_fraction"] is None


def test_unlabeled_pixels_excluded_from_semantic_precision():
    counts = binary_semantic_counts(
        np.array([True, True, False, True]),
        np.array([True, False, True, False]),
        np.array([True, True, True, False]),
    )
    assert counts == {"true_positive": 1, "false_positive": 1, "false_negative": 1}
    result = summarize_semantic_counts(counts)
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["iou"] == pytest.approx(1 / 3)


def test_geometry_mismatch_rejected():
    with pytest.raises(ValueError):
        DepthErrorTotals().add(np.ones(2), np.ones(3), np.ones(3, bool))
