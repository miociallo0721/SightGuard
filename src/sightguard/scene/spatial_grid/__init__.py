"""Serializable image-space evidence; no navigation or alert side effects."""

from sightguard.scene.spatial_grid.engine import SpatialGridFusion
from sightguard.scene.spatial_grid.schema import (
    EvidenceSource,
    FrameKey,
    SpatialGridConfig,
    SpatialGridFrame,
    SpatialGridInput,
)

__all__ = [
    "EvidenceSource",
    "FrameKey",
    "SpatialGridConfig",
    "SpatialGridFrame",
    "SpatialGridFusion",
    "SpatialGridInput",
]
