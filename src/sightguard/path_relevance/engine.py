"""Image-space candidate corridor estimation and object association.

The output describes semantic overlap with model-estimated ground regions. It is
not a metric route, collision prediction, or safety guarantee.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from sightguard.path_relevance.model import SegmentationOutput

WALKABLE_ADE20K_CLASSES = {3: "floor", 6: "road", 11: "sidewalk"}
PATH_RELEVANCE_MEDIUM_MIN = 0.20
PATH_RELEVANCE_HIGH_MIN = 0.55


@dataclass(frozen=True)
class CorridorCandidate:
    corridor_id: int
    bbox_xyxy: tuple[float, float, float, float]
    polygon_xy: tuple[tuple[float, float], ...]
    area_fraction: float
    vertical_extent_fraction: float
    mean_segmentation_confidence: float


@dataclass(frozen=True)
class CorridorEstimate:
    status: str
    quality: str
    confidence: float
    reasons: tuple[str, ...]
    corridors: tuple[CorridorCandidate, ...]
    mask: np.ndarray
    width: int
    height: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "quality": self.quality,
            "confidence": self.confidence,
            "reasons": list(self.reasons),
            "corridors": [asdict(item) for item in self.corridors],
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True)
class ObjectPathAssessment:
    track_id: int | None
    detection_index: int
    class_id: int
    class_name: str
    detection_confidence: float | None
    event_id: str | None
    path_relevance_score: float | None
    relevance_level: str
    corridor_intersection: float | None
    estimated_spatial_relation: str
    contact_point_status: str
    contact_point_confidence: float
    scene_quality: str
    confidence: float
    uncertainty: float
    reasons: tuple[str, ...]
    evidence_source: tuple[str, ...]
    bbox_xyxy: tuple[float, float, float, float]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["reasons"] = list(self.reasons)
        result["evidence_source"] = list(self.evidence_source)
        result["bbox_xyxy"] = list(self.bbox_xyxy)
        return result


def estimate_corridors(
    segmentation: SegmentationOutput,
    *,
    min_pixel_confidence: float = 0.35,
    min_area_fraction: float = 0.005,
    bottom_start_fraction: float = 0.82,
    min_vertical_extent_fraction: float = 0.35,
) -> CorridorEstimate:
    """Keep semantic walkable components connected from the lower image edge.

    All coordinates remain in image space. Multiple supported components are
    retained; no centerline, camera calibration, or metric depth is assumed.
    """
    import cv2

    class_map = segmentation.class_map
    confidence = segmentation.confidence_map
    if (
        class_map.ndim != 2
        or confidence.shape != class_map.shape
        or not np.isfinite(confidence).all()
        or class_map.size == 0
    ):
        raise ValueError("semantic maps must be non-empty, aligned, and finite")
    if not 0 <= min_pixel_confidence <= 1:
        raise ValueError("min_pixel_confidence must be between 0 and 1")
    if not 0 < min_area_fraction <= 1 or not 0 < bottom_start_fraction < 1:
        raise ValueError("corridor area and bottom thresholds are invalid")
    if not 0 < min_vertical_extent_fraction <= 1:
        raise ValueError("min_vertical_extent_fraction must be between 0 and 1")

    height, width = class_map.shape
    semantic_ground = np.isin(class_map, tuple(WALKABLE_ADE20K_CLASSES))
    supported = semantic_ground & (confidence >= min_pixel_confidence)
    # A small close reconnects quantization gaps in the 1/32-resolution model map.
    supported = cv2.morphologyEx(
        supported.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
    )
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(supported, connectivity=8)
    min_area = max(4, round(height * width * min_area_fraction))
    bottom_row = min(height - 1, round(height * bottom_start_fraction))
    accepted: list[CorridorCandidate] = []
    corridor_mask = np.zeros((height, width), dtype=np.uint8)
    source_h, source_w = height, width
    for label in range(1, component_count):
        x, y, component_width, component_height, area = (int(value) for value in stats[label])
        if area < min_area:
            continue
        component = labels == label
        touches_lower_region = bool(component[bottom_row:, :].any())
        vertical_extent = component_height / height
        if not touches_lower_region or vertical_extent < min_vertical_extent_fraction:
            continue
        corridor_mask[component] = 1
        contours, _ = cv2.findContours(
            component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        simplified = cv2.approxPolyDP(
            contour, epsilon=max(1.0, min(width, height) * 0.008), closed=True
        )
        polygon = tuple((float(point[0][0]), float(point[0][1])) for point in simplified)
        mean_confidence = float(confidence[component].mean())
        accepted.append(
            CorridorCandidate(
                corridor_id=len(accepted),
                bbox_xyxy=(
                    float(x),
                    float(y),
                    float(x + component_width),
                    float(y + component_height),
                ),
                polygon_xy=polygon,
                area_fraction=float(area / (height * width)),
                vertical_extent_fraction=float(vertical_extent),
                mean_segmentation_confidence=mean_confidence,
            )
        )

    if not accepted:
        lower_evidence = bool(semantic_ground[bottom_row:, :].any())
        reason = "no_supported_ground_component_reaches_lower_frame"
        if lower_evidence:
            reason = "ground_evidence_is_fragmented_or_does_not_extend_forward"
        return CorridorEstimate(
            status="unknown",
            quality="unknown",
            confidence=0.0,
            reasons=(reason, "camera_pose_unobserved", "metric_depth_unavailable"),
            corridors=(),
            mask=corridor_mask,
            width=source_w,
            height=source_h,
        )

    selected_confidence = float(confidence[corridor_mask.astype(bool)].mean())
    selected_extent = max(item.vertical_extent_fraction for item in accepted)
    extent_quality = min(1.0, selected_extent / 0.7)
    confidence_score = float(np.clip(selected_confidence * extent_quality, 0, 1))
    quality = "high" if confidence_score >= 0.65 else "limited"
    reasons = ["semantic_ground_component_reaches_lower_frame"]
    if len(accepted) > 1:
        reasons.append("multiple_disconnected_ground_candidates")
    if quality == "limited":
        reasons.append("limited_semantic_or_forward_extent_confidence")
    reasons.extend(("camera_pose_unobserved", "metric_depth_unavailable"))
    return CorridorEstimate(
        status="valid",
        quality=quality,
        confidence=confidence_score,
        reasons=tuple(reasons),
        corridors=tuple(accepted),
        mask=corridor_mask,
        width=source_w,
        height=source_h,
    )


def _contact_prior(class_name: str) -> tuple[float, str]:
    name = class_name.casefold()
    if name == "person":
        return 0.60, "bbox_bottom_center_heuristic_person"
    if name in {"chair", "bench", "couch", "bed", "dining table", "toilet"}:
        return 0.45, "bbox_bottom_center_heuristic_furniture"
    if name in {"car", "truck", "bus", "motorcycle", "bicycle"}:
        return 0.35, "bbox_bottom_center_heuristic_vehicle"
    return 0.20, "bbox_bottom_center_heuristic_unclassified"


def _unknown_object(
    detection: dict[str, Any], index: int, event_id: str | None, reason: str, scene_quality: str
) -> ObjectPathAssessment:
    bbox = tuple(float(value) for value in detection.get("bbox_xyxy", (0, 0, 0, 0)))
    contact_prior, contact_status = _contact_prior(str(detection.get("class_name", "unknown")))
    return ObjectPathAssessment(
        track_id=detection.get("track_id"),
        detection_index=index,
        class_id=int(detection.get("class_id", -1)),
        class_name=str(detection.get("class_name", "unknown")),
        detection_confidence=(
            float(detection["confidence"])
            if isinstance(detection.get("confidence"), (int, float))
            else None
        ),
        event_id=event_id,
        path_relevance_score=None,
        relevance_level="UNKNOWN",
        corridor_intersection=None,
        estimated_spatial_relation="unknown",
        contact_point_status=contact_status,
        contact_point_confidence=contact_prior,
        scene_quality=scene_quality,
        confidence=0.0,
        uncertainty=1.0,
        reasons=(reason,),
        evidence_source=("segformer_mit_b0_ade20k", "yolo_botsort_bbox"),
        bbox_xyxy=bbox,
    )


def associate_detection(
    detection: dict[str, Any],
    detection_index: int,
    estimate: CorridorEstimate,
    *,
    event_id: str | None = None,
    frame_width: int,
    frame_height: int,
    medium_threshold: float = PATH_RELEVANCE_MEDIUM_MIN,
    high_threshold: float = PATH_RELEVANCE_HIGH_MIN,
) -> ObjectPathAssessment:
    """Measure a detection's overlap with supported semantic ground regions."""
    if estimate.status != "valid" or not estimate.corridors:
        return _unknown_object(
            detection, detection_index, event_id, "candidate_corridor_unknown", estimate.quality
        )
    if not 0 <= medium_threshold < high_threshold <= 1:
        raise ValueError("path relevance thresholds must satisfy 0 <= medium < high <= 1")
    raw_bbox = detection.get("bbox_xyxy")
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        return _unknown_object(
            detection, detection_index, event_id, "invalid_detection_bbox", estimate.quality
        )
    x1, y1, x2, y2 = (float(value) for value in raw_bbox)
    if not all(np.isfinite((x1, y1, x2, y2))) or x2 <= x1 or y2 <= y1:
        return _unknown_object(
            detection, detection_index, event_id, "degenerate_detection_bbox", estimate.quality
        )
    if frame_width <= 0 or frame_height <= 0:
        return _unknown_object(
            detection, detection_index, event_id, "invalid_frame_dimensions", estimate.quality
        )

    mask_height, mask_width = estimate.mask.shape
    scale_x, scale_y = mask_width / frame_width, mask_height / frame_height
    left = int(np.clip(np.floor(x1 * scale_x), 0, mask_width))
    right = int(np.clip(np.ceil(x2 * scale_x), 0, mask_width))
    top = int(np.clip(np.floor(y1 * scale_y), 0, mask_height))
    bottom = int(np.clip(np.ceil(y2 * scale_y), 0, mask_height))
    if right <= left or bottom <= top:
        return _unknown_object(
            detection, detection_index, event_id, "bbox_outside_frame", estimate.quality
        )

    crop = estimate.mask[top:bottom, left:right]
    intersection = float(crop.mean()) if crop.size else 0.0
    class_name = str(detection.get("class_name", "unknown"))
    contact_prior, contact_status = _contact_prior(class_name)
    lower_start = max(0, round(crop.shape[0] * (0.72 if class_name == "person" else 0.65)))
    lower_contact_zone = crop[lower_start:, :]
    contact_support = float(lower_contact_zone.mean()) if lower_contact_zone.size else 0.0
    relevance = float(np.clip(0.70 * intersection + 0.30 * contact_support, 0.0, 1.0))

    if intersection >= 0.15:
        relation = "corridor_overlap"
    elif intersection > 0.01:
        relation = "partial_corridor_overlap"
    else:
        center_x = (left + right) // 2
        center_y = (top + bottom) // 2
        nearest_distance = min(
            max(candidate.bbox_xyxy[0] - center_x, 0, center_x - candidate.bbox_xyxy[2])
            + max(candidate.bbox_xyxy[1] - center_y, 0, center_y - candidate.bbox_xyxy[3])
            for candidate in estimate.corridors
        )
        relation = (
            "adjacent_to_corridor"
            if nearest_distance <= max(2, right - left)
            else "outside_corridor"
        )

    if relevance >= high_threshold:
        level = "HIGH"
    elif relevance >= medium_threshold:
        level = "MEDIUM"
    else:
        level = "LOW"
    if estimate.quality == "limited":
        level = "UNKNOWN" if relevance < medium_threshold else level
    detection_confidence = detection.get("confidence")
    detection_factor = (
        float(np.clip(detection_confidence, 0.0, 1.0))
        if isinstance(detection_confidence, (int, float)) and np.isfinite(detection_confidence)
        else 1.0
    )
    confidence = float(
        np.clip(
            estimate.confidence * (0.55 + 0.45 * contact_prior) * detection_factor,
            0.0,
            1.0,
        )
    )
    reasons = ["bbox_overlap_with_semantic_ground_component"]
    if contact_support > 0:
        reasons.append("lower_bbox_region_overlaps_candidate_corridor")
    if contact_prior < 0.5:
        reasons.append("contact_point_is_class_dependent_bbox_heuristic")
    if estimate.quality == "limited":
        reasons.append("scene_quality_limited")
    return ObjectPathAssessment(
        track_id=detection.get("track_id"),
        detection_index=detection_index,
        class_id=int(detection.get("class_id", -1)),
        class_name=class_name,
        detection_confidence=(
            float(detection_factor) if detection.get("confidence") is not None else None
        ),
        event_id=event_id,
        path_relevance_score=None if level == "UNKNOWN" else relevance,
        relevance_level=level,
        corridor_intersection=intersection,
        estimated_spatial_relation=relation,
        contact_point_status=contact_status,
        contact_point_confidence=contact_prior,
        scene_quality=estimate.quality,
        confidence=confidence,
        uncertainty=float(1.0 - confidence),
        reasons=tuple(reasons),
        evidence_source=("segformer_mit_b0_ade20k", "yolo_botsort_bbox"),
        bbox_xyxy=(x1, y1, x2, y2),
    )


def scene_is_fresh(
    scene_timestamp_ms: float, detection_timestamp_ms: float, *, max_age_ms: float
) -> bool:
    """Reject stale or future scene results before spatial association."""
    values = (scene_timestamp_ms, detection_timestamp_ms, max_age_ms)
    if not all(np.isfinite(values)) or max_age_ms < 0:
        return False
    age_ms = detection_timestamp_ms - scene_timestamp_ms
    return 0 <= age_ms <= max_age_ms
