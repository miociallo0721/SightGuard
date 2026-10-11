"""Strict adapter to existing SpatialGridFrame; no replacement grid schema."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from sightguard.scene.spatial_grid.engine import grid_bounds
from sightguard.scene.spatial_grid.schema import (
    SCHEMA_VERSION,
    BboxCoverage,
    DepthEvidence,
    EvidenceSource,
    FrameKey,
    GroundEvidence,
    ObjectGridRelation,
    SpatialGridCell,
    SpatialGridConfig,
    SpatialGridFrame,
)


def _fraction(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0 <= value <= 1
    )


def validate_depth(evidence: DepthEvidence, area: int | None = None) -> None:
    if type(evidence.valid_count) is not int or evidence.valid_count < 0:
        raise ValueError("invalid depth valid_count")
    if area is not None and evidence.valid_count > area:
        raise ValueError("depth valid_count exceeds cell pixels")
    if not all(
        _fraction(x)
        for x in (
            evidence.invalid_fraction,
            evidence.negative_fraction,
            evidence.nonfinite_fraction,
        )
    ):
        raise ValueError("invalid depth fractions")
    if evidence.frame_percentile is not None and not _fraction(evidence.frame_percentile):
        raise ValueError("invalid relative depth percentile")
    for value in (evidence.median, evidence.q10, evidence.q90, evidence.spread_fraction):
        if value is not None and (isinstance(value, bool) or not math.isfinite(value) or value < 0):
            raise ValueError("invalid depth summary")
    if evidence.valid_count:
        if any(x is None for x in (evidence.q10, evidence.median, evidence.q90)):
            raise ValueError("incomplete depth summary")
        if not evidence.q10 <= evidence.median <= evidence.q90:
            raise ValueError("unordered depth quantiles")
    elif any(
        x is not None
        for x in (evidence.q10, evidence.median, evidence.q90, evidence.frame_percentile)
    ):
        raise ValueError("unobserved depth cannot have numeric summary")


def validate_grid(frame: SpatialGridFrame, expected_key: FrameKey | None = None) -> None:
    if frame.schema_version != SCHEMA_VERSION or frame.production_decision_changed:
        raise ValueError("unsupported or production-modifying spatial evidence")
    if frame.coordinate_space != "source_image_pixels_half_open":
        raise ValueError("unsupported grid coordinate space")
    if frame.depth_semantics != "relative_inverse_depth_larger_is_nearer":
        raise ValueError("unsupported depth semantics")
    if expected_key is not None and frame.key != expected_key:
        raise ValueError("grid video/frame/timestamp/dimension mismatch")
    config = SpatialGridConfig(rows=frame.rows, cols=frame.cols)
    if set(frame.source_status) != {"semantic", "depth", "detection"}:
        raise ValueError("incomplete source status contract")
    if any(not math.isfinite(x) or x < 0 for x in frame.frame_depth_quantiles):
        raise ValueError("invalid frame relative depth quantiles")
    if tuple(sorted(frame.frame_depth_quantiles)) != frame.frame_depth_quantiles:
        raise ValueError("unordered frame relative depth quantiles")
    bounds = grid_bounds(frame.key.width, frame.key.height, config)
    if len(frame.cells) != len(bounds):
        raise ValueError("incomplete grid")
    for name, source in frame.sources.items():
        if source is not None and source.status != frame.source_status.get(name):
            raise ValueError("source status/provenance mismatch")
        if source is not None and source.status == "valid" and source.key != frame.key:
            raise ValueError(f"{name} source identity mismatch")
    for index, (cell, expected) in enumerate(zip(frame.cells, bounds, strict=True)):
        if (
            cell.cell_id != index
            or (cell.row, cell.col) != divmod(index, frame.cols)
            or tuple(cell.bounds_xyxy) != expected
        ):
            raise ValueError("cell identity/order/bounds mismatch")
        if cell.source_status != frame.source_status:
            raise ValueError("inconsistent source status")
        for value in (
            cell.ground.support_fraction,
            cell.ground.candidate_fraction,
            cell.ground.mean_confidence,
        ):
            if value is not None and not _fraction(value):
                raise ValueError("invalid ground summary")
        if any(not _fraction(x) for x in cell.ground.class_fractions.values()):
            raise ValueError("invalid ground class fraction")
        if sum(cell.ground.class_fractions.values()) > 1 + 1e-9:
            raise ValueError("ground class fractions exceed one")
        if cell.ground.status == "supported" and frame.source_status.get("semantic") != "valid":
            raise ValueError("unknown semantics cannot support ground")
        validate_depth(cell.depth, (expected[2] - expected[0]) * (expected[3] - expected[1]))
        for coverage in cell.detection_coverage:
            if (
                type(coverage.detection_index) is not int
                or coverage.detection_index < 0
                or not all(
                    _fraction(x)
                    for x in (coverage.confidence, coverage.cell_fraction, coverage.bbox_fraction)
                )
            ):
                raise ValueError("invalid detection coverage")
    ids = [relation.detection_index for relation in frame.object_relations]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate object identities")
    objects = {x.detection_index: x for x in frame.object_relations}
    for cell in frame.cells:
        for coverage in cell.detection_coverage:
            if coverage.detection_index not in objects:
                raise ValueError("coverage references missing object")
            if not any(
                x["cell_id"] == cell.cell_id
                for x in objects[coverage.detection_index].cell_coverage
            ):
                raise ValueError("cell references absent reverse object coverage")
    for relation in frame.object_relations:
        if (
            type(relation.detection_index) is not int
            or relation.detection_index < 0
            or (relation.confidence is not None and not _fraction(relation.confidence))
            or (relation.ground_fraction is not None and not _fraction(relation.ground_fraction))
        ):
            raise ValueError("invalid object metadata")
        validate_depth(relation.depth)
        seen = set()
        for coverage in relation.cell_coverage:
            cell_id = coverage["cell_id"]
            if type(cell_id) is not int or not 0 <= cell_id < len(frame.cells) or cell_id in seen:
                raise ValueError("invalid object cell reference")
            seen.add(cell_id)
            matched = [
                x
                for x in frame.cells[cell_id].detection_coverage
                if x.detection_index == relation.detection_index
            ]
            if (
                len(matched) != 1
                or matched[0].track_id != relation.track_id
                or matched[0].class_id != relation.class_id
                or not math.isclose(
                    matched[0].cell_fraction, coverage["cell_fraction"], abs_tol=1e-9
                )
                or not math.isclose(
                    matched[0].bbox_fraction, coverage["bbox_fraction"], abs_tol=1e-9
                )
            ):
                raise ValueError("object/cell relation mismatch")


def parse_spatial_frame(
    row: dict[str, Any], expected_key: FrameKey | None = None
) -> SpatialGridFrame:
    """Restore existing immutable interfaces and reject malformed saved inputs."""
    try:
        key = FrameKey(**row["key"])
        sources = {
            name: EvidenceSource(
                **{**source, "key": FrameKey(**source["key"]), "reasons": tuple(source["reasons"])}
            )
            if source
            else None
            for name, source in row["sources"].items()
        }
        cells = []
        for item in row["cells"]:
            ground = GroundEvidence(
                **{
                    **item["ground"],
                    "bottom_connected_component_ids": tuple(
                        item["ground"]["bottom_connected_component_ids"]
                    ),
                }
            )
            cells.append(
                SpatialGridCell(
                    **{
                        **item,
                        "ground": ground,
                        "depth": DepthEvidence(**item["depth"]),
                        "bounds_xyxy": tuple(item["bounds_xyxy"]),
                        "detection_coverage": tuple(
                            BboxCoverage(**x) for x in item["detection_coverage"]
                        ),
                        "conflict_reasons": tuple(item["conflict_reasons"]),
                        "unknown_reasons": tuple(item["unknown_reasons"]),
                    }
                )
            )
        relations = tuple(
            ObjectGridRelation(
                **{
                    **item,
                    "depth": DepthEvidence(**item["depth"]),
                    "bbox_xyxy": tuple(item["bbox_xyxy"])
                    if item["bbox_xyxy"] is not None
                    else None,
                    "clipped_bbox_xyxy": tuple(item["clipped_bbox_xyxy"])
                    if item["clipped_bbox_xyxy"] is not None
                    else None,
                    "cell_coverage": tuple(item["cell_coverage"]),
                    "candidate_component_ids": tuple(item["candidate_component_ids"]),
                    "reasons": tuple(item["reasons"]),
                }
            )
            for item in row["object_relations"]
        )
        frame = SpatialGridFrame(
            **{
                **row,
                "key": key,
                "sources": sources,
                "cells": tuple(cells),
                "object_relations": relations,
                "failures": tuple(row["failures"]),
                "frame_depth_quantiles": tuple(row["frame_depth_quantiles"]),
                "limitations": tuple(row["limitations"]),
            }
        )
        validate_grid(frame, expected_key)
        return frame
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError(f"malformed SpatialGridFrame: {exc}") from exc


def reference_inside(root: Path, reference: str) -> Path:
    path = (root / reference).resolve()
    if Path(reference).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError("artifact reference escapes source directory")
    return path
