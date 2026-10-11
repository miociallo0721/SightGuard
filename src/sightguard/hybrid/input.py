"""Build semantic evidence without leaking final rule scores or annotated images."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sightguard.live.runtime import LiveResult


def _rounded(value):
    """Compact semantic copy only; full precision remains in sampled rule logs."""
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return {key: _rounded(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_rounded(item) for item in value]
    return value


def build_input(result: "LiveResult", max_objects: int = 8) -> tuple[dict, str | None]:
    observations = [o for o in result.decision.frame.observations if o.features is not None]
    observations.sort(key=lambda o: o.features.spatial.focus_relevance, reverse=True)
    objects = []
    for observation in observations[:max_objects]:
        detection = result.record.detections[observation.detection_index]
        features = observation.features
        objects.append(
            {
                "event_id": observation.event_id,
                "track_id": observation.source_track_id,
                "class_id": observation.class_id,
                "class_name": observation.class_name,
                "bbox_xyxy": detection.bbox_xyxy,
                "detection_confidence": detection.confidence,
                "features": {
                    "center_xy": features.spatial.center_xy,
                    "area": features.spatial.area,
                    "focus_relevance": features.spatial.focus_relevance,
                    "observation_duration_ms": features.observation_duration_ms,
                    "scale_change_per_s": features.scale_change_per_s,
                    "center_change_per_s": features.center_change_per_s,
                    "inward_change_per_s": features.inward_change_per_s,
                    "motion_interval_ms": features.motion_interval_ms,
                    "motion_reliable": features.quality.motion_reliable,
                    "continuity": features.quality.continuity,
                    "track_id_missing": features.quality.track_id_missing,
                    "interrupted": features.quality.interrupted,
                    "limitations": features.quality.limitations,
                },
            }
        )
    # One concern task focuses on the highest focus-relevance observed event, not highest rule risk.
    event_id = objects[0]["event_id"] if objects else None
    return _rounded(
        {
            "frame_id": result.record.frame_index,
            "timestamp_ms": result.record.timestamp_ms,
            "image_size": [result.record.frame_width, result.record.frame_height],
            "perception_status": result.record.status,
            "objects": objects,
            "task_scope": "first object is focus event; others are context; no navigation",
            "limitations": "Monocular evidence; session-local identity; not collision truth",
        }
    ), event_id
