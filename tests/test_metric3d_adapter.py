"""Known canonical focal scale and source-coordinate restoration checks."""

import numpy as np
import pytest

from sightguard.research.metric3d_adapter import prepare_metric3d, restore_metric3d_depth


@pytest.mark.parametrize("shape", [(480, 640), (640, 480), (616, 1064)])
def test_constant_depth_obeys_scaled_focal_and_source_geometry(shape):
    rgb = np.zeros((*shape, 3), dtype=np.uint8)
    tensor, geometry = prepare_metric3d(rgb, 500)
    assert tensor.shape == (1, 3, 616, 1064)
    assert tensor.dtype == np.float32
    canonical = np.full((616, 1064), 2, dtype=np.float32)
    depth = restore_metric3d_depth(canonical, geometry)
    assert depth.shape == shape
    np.testing.assert_allclose(depth, 2 * geometry.scaled_focal_x / 1000)


def test_padding_is_removed_without_horizontal_mirroring():
    _, geometry = prepare_metric3d(np.zeros((4, 8, 3), np.uint8), 1000, input_hw=(8, 8))
    assert geometry.padding_tblr == (2, 2, 0, 0)
    canonical = np.full((8, 8), 100, np.float32)
    canonical[2:6] = np.arange(8, dtype=np.float32)
    restored = restore_metric3d_depth(canonical, geometry)
    np.testing.assert_allclose(restored, np.tile(np.arange(8), (4, 1)))


@pytest.mark.parametrize("focal", [0, -1, float("nan"), float("inf")])
def test_unknown_or_invalid_intrinsics_abstain(focal):
    with pytest.raises(ValueError, match="focal"):
        prepare_metric3d(np.zeros((10, 10, 3), np.uint8), focal)


def test_nonfinite_depth_rejected():
    _, geometry = prepare_metric3d(np.zeros((4, 4, 3), np.uint8), 500, input_hw=(4, 4))
    with pytest.raises(ValueError, match="nonfinite"):
        restore_metric3d_depth(np.full((4, 4), np.nan), geometry)
