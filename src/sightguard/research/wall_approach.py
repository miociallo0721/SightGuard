"""Wall-specific visual expansion research with a calibrated pure-rotation counter-model."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np

from sightguard.scene.spatial_grid.schema import FrameKey


@dataclass(frozen=True)
class ApproachConfig:
    minimum_features: int = 24
    minimum_inlier_fraction: float = 0.75
    minimum_quadrants: int = 3
    maximum_fb_error_px: float = 1
    maximum_residual_px: float = 1.25
    maximum_rotation_rad: float = 0.015
    minimum_expansion_per_second: float = 0.04
    minimum_confirmations: int = 3
    maximum_pair_ms: float = 400
    maximum_source_age_ms: float = 500
    maximum_rotation_model_residual_px: float = 1
    minimum_mask_iou: float = 0.5

    def __post_init__(self) -> None:
        if (
            self.minimum_features < 3
            or not 1 <= self.minimum_quadrants <= 4
            or self.minimum_confirmations < 1
        ):
            raise ValueError("invalid approach support thresholds")
        for value in (self.minimum_inlier_fraction, self.minimum_mask_iou):
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError("invalid approach fraction")
        for value in (
            self.maximum_fb_error_px,
            self.maximum_residual_px,
            self.maximum_rotation_rad,
            self.minimum_expansion_per_second,
            self.maximum_pair_ms,
            self.maximum_source_age_ms,
            self.maximum_rotation_model_residual_px,
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("invalid approach time or geometry threshold")


def bearing_vectors(
    points: np.ndarray, intrinsic: np.ndarray, distortion: np.ndarray
) -> np.ndarray:
    normalized = cv2.undistortPoints(
        points.reshape(-1, 1, 2).astype(np.float64), intrinsic, distortion
    ).reshape(-1, 2)
    rays = np.column_stack([normalized, np.ones(len(points))])
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def rotation_counter_model(
    previous: np.ndarray,
    current: np.ndarray,
    intrinsic: np.ndarray,
    distortion: np.ndarray,
) -> tuple[float, float]:
    """Wahba fit of bearing rays. Small residual means rotation explains the flow."""
    before = bearing_vectors(previous, intrinsic, distortion)
    after = bearing_vectors(current, intrinsic, distortion)
    u, _, vt = np.linalg.svd(after.T @ before)
    rotation = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
    predicted = before @ rotation.T
    normalized_error = predicted[:, :2] / predicted[:, 2:] - after[:, :2] / after[:, 2:]
    pixel_error = np.linalg.norm(normalized_error * [intrinsic[0, 0], intrinsic[1, 1]], axis=1)
    angle = math.acos(float(np.clip((np.trace(rotation) - 1) / 2, -1, 1)))
    return float(np.median(pixel_error)), angle


class WallApproachEstimator:
    def __init__(self, config: ApproachConfig | None = None) -> None:
        self.config = config or ApproachConfig()
        self.previous_key: FrameKey | None = None
        self.confirmations = 0

    def reset(self) -> None:
        self.previous_key = None
        self.confirmations = 0

    def evaluate(
        self,
        previous_key: FrameKey | None,
        current_key: FrameKey,
        previous_image: np.ndarray | None,
        current_image: np.ndarray,
        previous_wall: np.ndarray | None,
        current_wall: np.ndarray,
        *,
        source_age_ms: float,
        wall_present: bool | None,
        intrinsic: np.ndarray | None,
        distortion: np.ndarray | None = None,
        fixed_lens_verified: bool = False,
        depth_support: str = "UNAVAILABLE",
        motion_quality: str = "UNKNOWN",
        imu_sync: str = "MOTION_SYNC_UNKNOWN",
    ) -> dict[str, Any]:
        cfg = self.config
        base = {
            "schema_version": "sightguard.wall-approach-research.v1",
            "key": asdict(current_key),
            "previous_key": asdict(previous_key) if previous_key else None,
            "wall_present": wall_present,
            "possible_near": False,
            "approaching_candidate": False,
            "source_age_ms": source_age_ms,
            "depth_support": depth_support,
            "motion_quality": motion_quality,
            "imu_sync": imu_sync,
            "camera_motion_only": True,
            "body_approach_verified": False,
            "automatic_alert_enabled": False,
            "confidence": None,
            "config": asdict(cfg),
        }

        def reject(reason: str, **details: Any) -> dict[str, Any]:
            self.confirmations = 0
            self.previous_key = current_key
            return {
                **base,
                **details,
                "state": "UNKNOWN" if wall_present is None else "ABSTAIN",
                "abstain_reason": reason,
                "temporal_support": 0,
            }

        if wall_present is not True:
            return reject("wall_not_confirmed")
        if not math.isfinite(source_age_ms) or not 0 <= source_age_ms <= cfg.maximum_source_age_ms:
            return reject("source_stale")
        if previous_key is None or previous_image is None or previous_wall is None:
            return reject("need_previous_wall_frame")
        interval = current_key.timestamp_ms - previous_key.timestamp_ms
        if (
            current_key.video_id != previous_key.video_id
            or current_key.frame_index <= previous_key.frame_index
            or (current_key.width, current_key.height) != (previous_key.width, previous_key.height)
            or not 10 <= interval <= cfg.maximum_pair_ms
        ):
            return reject("source_sequence_geometry_or_interval_mismatch")
        if self.previous_key is not None and self.previous_key != previous_key:
            self.confirmations = 0
        if intrinsic is None or not fixed_lens_verified:
            return reject("intrinsics_or_fixed_lens_unverified")
        if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
            return reject("invalid_intrinsics")
        if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
            return reject("invalid_focal_length")
        if distortion is not None and (
            distortion.size not in (4, 5, 8, 12, 14) or not np.isfinite(distortion).all()
        ):
            return reject("invalid_distortion")
        if motion_quality in ("RAPID_IMAGE_MOTION", "DISCONTINUITY"):
            return reject("rapid_motion_or_discontinuity")
        shape = (current_key.height, current_key.width)
        if (
            previous_image.shape[:2] != shape
            or current_image.shape[:2] != shape
            or previous_wall.shape != shape
            or current_wall.shape != shape
        ):
            return reject("image_mask_geometry_mismatch")
        intersection = np.count_nonzero(previous_wall & current_wall)
        union = np.count_nonzero(previous_wall | current_wall)
        iou = intersection / max(1, union)
        if iou < cfg.minimum_mask_iou:
            return reject("wall_region_association_failed", mask_iou=iou)
        gray = [
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
            for image in (previous_image, current_image)
        ]
        mask = cv2.erode(previous_wall.astype(np.uint8) * 255, np.ones((5, 5), np.uint8))
        points = cv2.goodFeaturesToTrack(gray[0], 200, 0.01, 8, mask=mask)
        if points is None or len(points) < cfg.minimum_features:
            return reject(
                "low_texture_or_insufficient_wall_features",
                feature_count=0 if points is None else len(points),
            )
        following, status, _ = cv2.calcOpticalFlowPyrLK(
            gray[0], gray[1], points, None, winSize=(21, 21), maxLevel=3
        )
        if following is None:
            return reject("forward_flow_failed")
        back, back_status, _ = cv2.calcOpticalFlowPyrLK(
            gray[1], gray[0], following, None, winSize=(21, 21), maxLevel=3
        )
        if back is None:
            return reject("backward_flow_failed")
        before, after = points[:, 0], following[:, 0]
        valid = status[:, 0].astype(bool) & back_status[:, 0].astype(bool)
        valid &= np.linalg.norm(back[:, 0] - before, axis=1) <= cfg.maximum_fb_error_px
        valid &= np.isfinite(after).all(axis=1)
        valid &= (after[:, 0] >= 0) & (after[:, 0] < current_key.width)
        valid &= (after[:, 1] >= 0) & (after[:, 1] < current_key.height)
        indices = np.flatnonzero(valid)
        xy = after[indices].astype(int)
        valid[indices] &= current_wall[xy[:, 1], xy[:, 0]]
        before, after = before[valid], after[valid]
        return self.evaluate_tracks(
            base,
            previous_key,
            current_key,
            before,
            after,
            intrinsic,
            np.zeros(5) if distortion is None else distortion,
            feature_count=len(points),
            mask_iou=iou,
        )

    def evaluate_tracks(
        self,
        base: dict[str, Any],
        previous_key: FrameKey,
        current_key: FrameKey,
        before: np.ndarray,
        after: np.ndarray,
        intrinsic: np.ndarray,
        distortion: np.ndarray,
        *,
        feature_count: int,
        mask_iou: float,
    ) -> dict[str, Any]:
        """Geometric kernel exposed for Level 1 synthetic positive and negative tests."""
        cfg = self.config
        details: dict[str, Any] = {
            "feature_count": feature_count,
            "match_count": len(before),
            "mask_iou": mask_iou,
        }
        reason = None
        interval = current_key.timestamp_ms - previous_key.timestamp_ms
        if (
            current_key.video_id != previous_key.video_id
            or current_key.frame_index <= previous_key.frame_index
            or not 10 <= interval <= cfg.maximum_pair_ms
        ):
            reason = "source_sequence_geometry_or_interval_mismatch"
        elif (
            before.shape != after.shape
            or before.ndim != 2
            or before.shape[1] != 2
            or not np.isfinite(before).all()
            or not np.isfinite(after).all()
        ):
            reason = "invalid_feature_tracks"
        elif len(before) < cfg.minimum_features:
            reason = "insufficient_valid_matches"
        else:
            cv2.setRNGSeed(41)
            matrix, inliers = cv2.estimateAffinePartial2D(
                before,
                after,
                method=cv2.RANSAC,
                ransacReprojThreshold=cfg.maximum_residual_px,
                maxIters=1000,
                confidence=0.99,
            )
            if matrix is None or inliers is None:
                reason = "wall_transform_fit_failed"
            else:
                good = inliers[:, 0].astype(bool)
                a, b = before[good], after[good]
                scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
                residual = float(
                    np.median(np.linalg.norm(a @ matrix[:, :2].T + matrix[:, 2] - b, axis=1))
                )
                quadrants = len(
                    {(x >= current_key.width / 2, y >= current_key.height / 2) for x, y in a}
                )
                rotation_residual, rotation_angle = rotation_counter_model(
                    a, b, intrinsic, distortion
                )
                rate = (scale - 1) / ((current_key.timestamp_ms - previous_key.timestamp_ms) / 1000)
                foe = None
                if abs(np.linalg.det(np.eye(2) - matrix[:, :2])) > 1e-8:
                    foe = np.linalg.solve(np.eye(2) - matrix[:, :2], matrix[:, 2])
                details.update(
                    inlier_count=int(good.sum()),
                    inlier_fraction=float(good.mean()),
                    quadrants=quadrants,
                    scale_ratio=scale,
                    expansion_per_second=rate,
                    fit_residual_px=residual,
                    rotation_model_residual_px=rotation_residual,
                    rotation_model_angle_rad=rotation_angle,
                    focus_of_expansion_xy=foe.tolist() if foe is not None else None,
                )
                if good.sum() < cfg.minimum_features or good.mean() < cfg.minimum_inlier_fraction:
                    reason = "insufficient_wall_consensus"
                elif quadrants < cfg.minimum_quadrants or residual > cfg.maximum_residual_px:
                    reason = "insufficient_spatial_coverage_or_mixed_planes"
                elif rotation_residual <= cfg.maximum_rotation_model_residual_px:
                    reason = "rotation_or_stationary_model_explains_flow"
                elif rotation_angle > cfg.maximum_rotation_rad:
                    reason = "rotation_or_mixed_motion"
                elif rate < cfg.minimum_expansion_per_second:
                    reason = "no_sufficient_expansion_receding_or_lateral"
                elif foe is None or not (
                    0.2 * current_key.width <= foe[0] <= 0.8 * current_key.width
                    and 0.2 * current_key.height <= foe[1] <= 0.8 * current_key.height
                ):
                    reason = "off_axis_motion_or_unstable_expansion_center"
        self.confirmations = 0 if reason else self.confirmations + 1
        self.previous_key = current_key
        confirmed = not reason and self.confirmations >= cfg.minimum_confirmations
        return {
            **base,
            **details,
            "state": "WALL_APPROACHING_CANDIDATE"
            if confirmed
            else "WALL_PRESENT"
            if not reason
            else "ABSTAIN",
            "approaching_candidate": confirmed,
            "temporal_support": self.confirmations,
            "abstain_reason": reason or (None if confirmed else "confirming_visual_expansion"),
            "quality": "research_geometric_support" if not reason else "rejected",
        }
