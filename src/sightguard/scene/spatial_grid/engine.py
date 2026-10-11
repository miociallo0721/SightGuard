"""Independent NumPy evidence aggregation; no model, risk, UI or audio imports."""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from sightguard.scene.spatial_grid.schema import (
    GROUND_CLASSES,
    BboxCoverage,
    DepthEvidence,
    EvidenceSource,
    GroundEvidence,
    ObjectGridRelation,
    SpatialGridCell,
    SpatialGridConfig,
    SpatialGridFrame,
    SpatialGridInput,
)


def grid_bounds(width: int, height: int, config: SpatialGridConfig) -> tuple[tuple[int, ...], ...]:
    """Partition each pixel exactly once, including non-divisible dimensions."""
    if width < config.cols or height < config.rows:
        raise ValueError("source image must contain at least one pixel per cell")
    return tuple(
        (
            c * width // config.cols,
            r * height // config.rows,
            (c + 1) * width // config.cols,
            (r + 1) * height // config.rows,
        )
        for r in range(config.rows)
        for c in range(config.cols)
    )


def _depth_stats(array: np.ndarray | None, sorted_frame: np.ndarray) -> DepthEvidence:
    if array is None:
        return DepthEvidence(0, 1.0, 0.0, 0.0, None, None, None, None, None, "unknown")
    finite = np.isfinite(array)
    negative = finite & (array < 0)
    valid = array[finite & ~negative]
    size = array.size
    if not valid.size:
        return DepthEvidence(
            0,
            1.0,
            float(negative.sum() / size),
            float((~finite).sum() / size),
            None,
            None,
            None,
            None,
            None,
            "unknown",
        )
    q10, median, q90 = (float(x) for x in np.quantile(valid, (0.1, 0.5, 0.9)))

    def frame_quantile(fraction: float) -> float:
        position = fraction * (sorted_frame.size - 1)
        lower = int(position)
        upper = min(lower + 1, sorted_frame.size - 1)
        return float(sorted_frame[lower]) + (position - lower) * (
            float(sorted_frame[upper]) - float(sorted_frame[lower])
        )

    low, high = frame_quantile(0.1), frame_quantile(0.9)
    scale = float(high - low)
    # Mid-rank handles constant maps without inventing a near/far ordering.
    percentile = float(
        (
            np.searchsorted(sorted_frame, median, side="left")
            + np.searchsorted(sorted_frame, median, side="right")
        )
        / (2 * sorted_frame.size)
    )
    return DepthEvidence(
        int(valid.size),
        float(1 - valid.size / size),
        float(negative.sum() / size),
        float((~finite).sum() / size),
        median,
        q10,
        q90,
        percentile if scale > 0 else None,
        (q90 - q10) / scale if scale > 0 else None,
        "limited" if valid.size < size or scale == 0 else "valid",
    )


def _source_status(source: EvidenceSource | None, data: SpatialGridInput) -> tuple[str, str | None]:
    if source is None:
        return "unknown", "missing_source"
    if source.key != data.key:
        return "unknown", "frame_identity_time_or_size_mismatch"
    if source.status == "excluded":
        return "excluded", None
    if source.status != "valid":
        return "unknown", "source_unknown"
    return "valid", None


def _parse_detection(detection: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    """Reject malformed geometry/confidence; missing track ID is not a failure."""
    if not isinstance(detection, dict):
        raise ValueError("detection_not_object")
    confidence = detection.get("confidence")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (float, int))
        or not math.isfinite(confidence)
        or not 0 <= confidence <= 1
    ):
        raise ValueError("invalid_detection_confidence")
    bbox = detection.get("bbox_xyxy")
    if (
        not isinstance(bbox, (list, tuple))
        or len(bbox) != 4
        or any(
            isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
            for x in bbox
        )
    ):
        raise ValueError("invalid_bbox")
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        raise ValueError("invalid_bbox_extent")
    clipped = (max(0.0, x1), max(0.0, y1), min(float(width), x2), min(float(height), y2))
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        raise ValueError("bbox_outside_image")
    class_id = detection.get("class_id")
    track_id = detection.get("track_id")
    if type(class_id) is not int or class_id < 0:
        raise ValueError("invalid_class_id")
    if track_id is not None and (type(track_id) is not int or track_id < 0):
        raise ValueError("invalid_track_id")
    return {
        **detection,
        "bbox_xyxy": tuple(bbox),
        "clipped": clipped,
        "confidence": float(confidence),
        "track_id": track_id,
    }


class SpatialGridFusion:
    """Stateless exact-frame fusion retaining independent evidence dimensions."""

    def __init__(self, config: SpatialGridConfig | None = None) -> None:
        self.config = config or SpatialGridConfig()

    def fuse(
        self, data: SpatialGridInput, *, timings: dict[str, float] | None = None
    ) -> SpatialGridFrame:
        config = self.config
        key = data.key
        bounds = grid_bounds(key.width, key.height, config)
        sources = {
            "semantic": data.semantic_source,
            "depth": data.depth_source,
            "detection": data.detection_source,
        }
        status: dict[str, str] = {}
        failures: list[str] = []
        for name, source in sources.items():
            status[name], reason = _source_status(source, data)
            if reason:
                failures.append(f"{name}:{reason}")
        shape = (key.height, key.width)
        maps = (data.class_map, data.confidence_map, data.candidate_components)
        if status["semantic"] == "valid" and (
            any(not isinstance(x, np.ndarray) or x.shape != shape for x in maps)
            or data.class_map.dtype.kind not in "iu"
            or np.any(data.class_map > 149)
            or np.any(data.class_map < 0)
            or data.candidate_components.dtype.kind not in "iu"
            or np.any(data.candidate_components < 0)
        ):
            status["semantic"] = "unknown"
            failures.append("semantic:invalid_or_misaligned_arrays")
        if status["depth"] == "valid" and (
            not isinstance(data.depth_map, np.ndarray)
            or data.depth_map.shape != shape
            or data.depth_map.dtype != np.float32
        ):
            status["depth"] = "unknown"
            failures.append("depth:invalid_or_misaligned_array")
        semantic_valid = status["semantic"] == "valid"
        depth = data.depth_map if status["depth"] == "valid" else None
        sorted_depth = (
            # Quantile medians are float64. Cast the frame ranking array once,
            # avoiding NumPy copying a full float32 map for each scalar search.
            np.sort(depth[np.isfinite(depth) & (depth >= 0)]).astype(np.float64)
            if depth is not None
            else np.array([], dtype=np.float32)
        )
        ground = np.zeros(shape, dtype=bool)
        confidence_valid = np.zeros(shape, dtype=bool)
        if semantic_valid:
            confidence_valid = (
                np.isfinite(data.confidence_map)
                & (data.confidence_map >= 0)
                & (data.confidence_map <= 1)
            )
            ground = np.isin(data.class_map, tuple(GROUND_CLASSES)) & confidence_valid
            ground &= data.confidence_map >= config.min_pixel_confidence
        coverage: list[list[BboxCoverage]] = [[] for _ in bounds]
        relations: list[ObjectGridRelation] = []
        started = time.perf_counter()
        if status["detection"] == "valid":
            for index, item in enumerate(data.detections):
                try:
                    detection = _parse_detection(item, key.width, key.height)
                except ValueError as exc:
                    failures.append(f"detection:{index}:{exc}")
                    relations.append(
                        ObjectGridRelation(
                            index,
                            None,
                            None,
                            None,
                            None,
                            None,
                            (),
                            _depth_stats(None, sorted_depth),
                            None,
                            (),
                            None,
                            "unknown",
                            "unknown",
                            (str(exc),),
                            data.detection_source.reference,
                        )
                    )
                    continue
                x1, y1, x2, y2 = detection["clipped"]
                area = (x2 - x1) * (y2 - y1)
                candidate = detection["confidence"] >= config.min_detection_confidence
                distribution: list[dict[str, float | int]] = []
                for cell_id, (cx1, cy1, cx2, cy2) in enumerate(bounds):
                    overlap = max(0, min(x2, cx2) - max(x1, cx1)) * max(
                        0, min(y2, cy2) - max(y1, cy1)
                    )
                    if overlap <= 0:
                        continue
                    cell_fraction = overlap / ((cx2 - cx1) * (cy2 - cy1))
                    bbox_fraction = overlap / area
                    coverage[cell_id].append(
                        BboxCoverage(
                            index,
                            detection["track_id"],
                            detection["class_id"],
                            detection.get("class_name"),
                            detection["confidence"],
                            cell_fraction,
                            bbox_fraction,
                            candidate,
                            "limited_bbox_proxy",
                        )
                    )
                    distribution.append(
                        {
                            "cell_id": cell_id,
                            "cell_fraction": cell_fraction,
                            "bbox_fraction": bbox_fraction,
                        }
                    )
                # Pixel-center inclusion follows continuous bbox coverage; thin boxes may
                # have cell associations but no raster pixels for numerical statistics.
                rx1, ry1 = max(0, math.ceil(x1 - 0.5)), max(0, math.ceil(y1 - 0.5))
                rx2, ry2 = math.ceil(x2 - 0.5), math.ceil(y2 - 0.5)
                region = np.s_[ry1:ry2, rx1:rx2]
                region_depth = (
                    depth[region] if depth is not None and rx2 > rx1 and ry2 > ry1 else None
                )
                ground_fraction = (
                    float(ground[region].mean())
                    if semantic_valid and rx2 > rx1 and ry2 > ry1
                    else None
                )
                component_ids = (
                    tuple(int(x) for x in np.unique(data.candidate_components[region]) if x > 0)
                    if semantic_valid
                    else ()
                )
                center = (x1 + x2) / (2 * key.width)
                reasons = ["bbox_region_includes_background", "not_3d_contact_or_occupancy"]
                if not candidate:
                    reasons.append("low_detection_confidence")
                if ground_fraction is None:
                    reasons.append("semantic_relation_unknown")
                relation_depth = _depth_stats(region_depth, sorted_depth)
                if not relation_depth.valid_count:
                    reasons.append("depth_relation_unknown")
                relations.append(
                    ObjectGridRelation(
                        index,
                        detection["track_id"],
                        detection["class_id"],
                        detection["confidence"],
                        detection["bbox_xyxy"],
                        detection["clipped"],
                        tuple(distribution),
                        relation_depth,
                        ground_fraction,
                        component_ids,
                        "image_left"
                        if center < 1 / 3
                        else "image_right"
                        if center > 2 / 3
                        else "image_center",
                        "candidate" if candidate else "limited",
                        "limited",
                        tuple(reasons),
                        data.detection_source.reference,
                    )
                )
        if timings is not None:
            timings["object_coverage"] = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        cells = []
        provenance = {
            name: source.reference if source else None for name, source in sources.items()
        }
        provenance.update(
            {
                f"{name}_sha256": source.sha256 if source else None
                for name, source in sources.items()
            }
        )
        for cell_id, (x1, y1, x2, y2) in enumerate(bounds):
            region = np.s_[y1:y2, x1:x2]
            unknown = [
                f"{name}_{'not_used' if value == 'excluded' else 'unknown'}"
                for name, value in status.items()
                if value != "valid"
            ]
            if any(reason.startswith("detection:") for reason in failures):
                unknown.append("invalid_detection_records_in_frame")
            conflict: list[str] = []
            support = float(ground[region].mean()) if semantic_valid else None
            if semantic_valid:
                classes = {
                    GROUND_CLASSES[k]: float((data.class_map[region] == k).mean())
                    for k in GROUND_CLASSES
                }
                components = data.candidate_components[region]
                ids = tuple(int(x) for x in np.unique(components) if x > 0)
                supported_conf = data.confidence_map[region][ground[region]]
                candidate_fraction = float((components > 0).mean())
                mean_conf = float(supported_conf.mean()) if supported_conf.size else None
                if not confidence_valid[region].all():
                    unknown.append("semantic_invalid_confidence_pixels")
            else:
                classes, ids, candidate_fraction, mean_conf = {}, (), None, None
            ground_known = support is not None and support >= config.ground_fraction_threshold
            if not ground_known:
                unknown.append("ground_semantics_insufficient")
            depth_stats = _depth_stats(depth[region] if depth is not None else None, sorted_depth)
            if depth_stats.quality != "valid":
                unknown.append("depth_numeric_unknown_or_limited")
            if (
                depth_stats.spread_fraction is not None
                and depth_stats.spread_fraction >= config.depth_spread_fraction
            ):
                conflict.append("relative_depth_variation_not_object_boundary")
            occupancies = coverage[cell_id]
            # Max overlap retains all boxes but does not sum their overlaps as a probability.
            occupancy = any(
                x.candidate and x.cell_fraction >= config.occupancy_coverage_threshold
                for x in occupancies
            )
            if ground_known and occupancy:
                conflict.append("ground_and_bbox_candidate_overlap")
            state = (
                "MIXED"
                if ground_known and occupancy
                else "GROUND_EVIDENCE"
                if ground_known
                else "OCCUPANCY_EVIDENCE"
                if occupancy
                else "UNKNOWN"
            )
            scene_quality = data.semantic_source.quality if semantic_valid else "unknown"
            if scene_quality != "high":
                unknown.append("scene_quality_limited_or_unknown")
            cells.append(
                SpatialGridCell(
                    cell_id,
                    cell_id // config.cols,
                    cell_id % config.cols,
                    (x1, y1, x2, y2),
                    state,
                    "unknown"
                    if state == "UNKNOWN"
                    else "limited"
                    if unknown or conflict
                    else "valid",
                    GroundEvidence(
                        "supported" if ground_known else "unknown",
                        support,
                        classes,
                        mean_conf,
                        candidate_fraction,
                        ids,
                        scene_quality,
                    ),
                    depth_stats,
                    tuple(occupancies),
                    dict(status),
                    tuple(conflict),
                    tuple(unknown),
                    dict(provenance),
                )
            )
        if timings is not None:
            timings["grid_aggregation"] = (time.perf_counter() - started) * 1000
        frame_status = "unknown" if not any(x == "valid" for x in status.values()) else "limited"
        return SpatialGridFrame(
            key,
            config.rows,
            config.cols,
            frame_status,
            sources,
            status,
            tuple(cells),
            tuple(relations),
            tuple(failures),
            tuple(float(x) for x in np.quantile(sorted_depth, (0.1, 0.5, 0.9)))
            if sorted_depth.size
            else (),
        )
