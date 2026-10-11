"""Adjacent-frame sparse LK and robust image similarity transform, CPU only."""

from __future__ import annotations

import math
import time
from dataclasses import asdict
from typing import Any

import cv2
import numpy as np

from sightguard.scene.motion.schema import (
    FrameMotionEstimate,
    MotionConfig,
    MotionFrame,
    MotionQuality,
)

ALGORITHM = "shi_tomasi_bidirectional_lk_ransac_partial_affine"


def background_mask(frame: MotionFrame, config: MotionConfig) -> np.ndarray | None:
    """Exclude confident boxes; remaining pixels are only candidate background."""
    tracks = frame.tracks
    if tracks is None or tracks.status != "ok":
        return None
    key = frame.key
    if (tracks.frame_index, tracks.timestamp_ms, tracks.frame_width, tracks.frame_height) != (
        key.frame_index,
        key.timestamp_ms,
        key.width,
        key.height,
    ):
        return None
    mask = np.full((key.height, key.width), 255, np.uint8)
    margin = config.mask_margin_pixels
    for detection in tracks.detections:
        if detection.confidence < config.detection_confidence:
            continue
        x1, y1, x2, y2 = detection.bbox_xyxy
        if x2 <= x1 or y2 <= y1:
            continue
        left, top = max(0, math.floor(x1 - margin)), max(0, math.floor(y1 - margin))
        right, bottom = (
            min(key.width, math.ceil(x2 + margin)),
            min(key.height, math.ceil(y2 + margin)),
        )
        if right > left and bottom > top:
            mask[top:bottom, left:right] = 0
    return mask


class FrameMotionEstimator:
    def __init__(self, config: MotionConfig | None = None) -> None:
        self.config = config or MotionConfig()

    def estimate(
        self,
        previous: MotionFrame | None,
        current: MotionFrame,
        *,
        masked: bool = True,
        previous_estimate: FrameMotionEstimate | None = None,
        timings: dict[str, float] | None = None,
    ) -> FrameMotionEstimate:
        cfg = self.config
        stage = timings if timings is not None else {}
        base: dict[str, Any] = {
            "key": current.key,
            "stream_id": current.stream_id,
            "previous_key": previous.key if previous else None,
            "dt_ms": current.key.timestamp_ms - previous.key.timestamp_ms if previous else None,
            "masking_mode": "BOX_EXCLUDED" if masked else "UNMASKED_COMPARISON",
            "estimation_status": "UNKNOWN",
            "feature_count": 0,
            "match_count": 0,
            "inlier_count": 0,
            "inlier_ratio": None,
            "background_available_fraction": None,
            "background_feature_coverage": None,
            "transform_previous_to_current": None,
            "translation_source_pixels": None,
            "rotation_2d_degrees": None,
            "scale_2d_ratio": None,
            "median_displacement_source_pixels": None,
            "motion_diagonals_per_second": None,
            "residual_inlier_median_pixels": None,
            "residual_inlier_p95_pixels": None,
            "residual_all_p95_pixels": None,
            "blur_variance": None,
            "flow_samples": (),
            "algorithm": ALGORITHM,
            "config": asdict(cfg),
            "provenance": {
                **current.provenance,
                "previous": previous.provenance if previous else None,
                "timestamp_source": current.timestamp_source,
                "background_status": "unverified_even_after_box_exclusion",
            },
        }

        def finish(label: str, *reasons: str) -> FrameMotionEstimate:
            return FrameMotionEstimate(**base, quality=MotionQuality(label, tuple(reasons)))

        if not current.stream_id:
            return finish("UNKNOWN", "stream_identity_missing")
        if current.reasons:
            return finish("UNKNOWN", *current.reasons)
        if current.image is None:
            return finish("UNKNOWN", "frame_decode_failed")
        if previous is None:
            return finish("UNKNOWN", "no_previous_frame")
        if previous.image is None or previous.reasons:
            return finish("UNKNOWN", "previous_frame_unavailable")
        if previous.key.video_id != current.key.video_id or previous.stream_id != current.stream_id:
            return finish("UNKNOWN", "sequence_identity_mismatch")
        if previous.key.frame_index + 1 != current.key.frame_index:
            return finish("UNKNOWN", "missing_or_duplicate_frame_index")
        if (previous.key.width, previous.key.height) != (current.key.width, current.key.height):
            return finish("UNKNOWN", "image_dimensions_changed")
        if previous.timestamp_source != "video_pts" or current.timestamp_source != "video_pts":
            return finish("UNKNOWN", "unverified_timestamp_source")
        if not cfg.min_dt_ms <= base["dt_ms"] <= cfg.max_dt_ms:
            return finish("UNKNOWN", "frame_time_interval_invalid")
        for frame in (previous, current):
            image = frame.image
            if (
                image.dtype != np.uint8
                or image.ndim not in (2, 3)
                or image.shape[:2] != (frame.key.height, frame.key.width)
                or (image.ndim == 3 and image.shape[2] != 3)
            ):
                return finish("UNKNOWN", "invalid_image_contract")
        try:
            t = time.perf_counter()
            width = min(current.key.width, cfg.resize_width)
            height = max(1, round(current.key.height * width / current.key.width))
            size = (width, height)
            gray = []
            masks = []
            for frame in (previous, current):
                image = frame.image
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
                gray.append(cv2.resize(image, size, interpolation=cv2.INTER_AREA))
                mask = (
                    background_mask(frame, cfg) if masked else np.full(image.shape, 255, np.uint8)
                )
                if mask is None:
                    return finish("UNKNOWN", "detection_mask_source_unavailable_or_misaligned")
                masks.append(cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST))
            bg_fraction = min(float(np.mean(mask > 0)) for mask in masks)
            base["background_available_fraction"] = bg_fraction
            base["blur_variance"] = min(
                float(cv2.Laplacian(image, cv2.CV_64F).var()) for image in gray
            )
            stage["prepare_and_mask"] = (time.perf_counter() - t) * 1000
            if bg_fraction < cfg.min_background_fraction:
                return finish("UNKNOWN", "insufficient_unmasked_background_area")
            if base["blur_variance"] < cfg.blur_unknown_variance:
                return finish("UNKNOWN", "severe_image_degradation_proxy")
            t = time.perf_counter()
            points = cv2.goodFeaturesToTrack(
                gray[0],
                cfg.max_features,
                cfg.feature_quality,
                cfg.feature_min_distance,
                mask=masks[0],
            )
            stage["feature_extraction"] = (time.perf_counter() - t) * 1000
            base["feature_count"] = len(points) if points is not None else 0
            if base["feature_count"] < cfg.min_features:
                return finish("UNKNOWN", "insufficient_features")
            t = time.perf_counter()
            lk = {
                "winSize": (cfg.lk_window, cfg.lk_window),
                "maxLevel": cfg.lk_levels,
                "criteria": (
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    cfg.lk_iterations,
                    cfg.lk_epsilon,
                ),
            }
            forward, status, error = cv2.calcOpticalFlowPyrLK(gray[0], gray[1], points, None, **lk)
            if forward is None or status is None or error is None:
                return finish("UNKNOWN", "optical_flow_failed")
            backward, back_status, _ = cv2.calcOpticalFlowPyrLK(
                gray[1], gray[0], forward, None, **lk
            )
            if backward is None or back_status is None:
                return finish("UNKNOWN", "backward_flow_failed")
            a, b, back = points.reshape(-1, 2), forward.reshape(-1, 2), backward.reshape(-1, 2)
            valid = (
                (status.ravel() > 0)
                & (back_status.ravel() > 0)
                & (error.ravel() <= cfg.lk_max_error)
            )
            valid &= (
                np.isfinite(a).all(axis=1)
                & np.isfinite(b).all(axis=1)
                & np.isfinite(back).all(axis=1)
            )
            valid &= np.linalg.norm(a - back, axis=1) <= cfg.fb_max_error
            valid &= (b[:, 0] >= 0) & (b[:, 0] < width) & (b[:, 1] >= 0) & (b[:, 1] < height)
            indices = np.flatnonzero(valid)
            if len(indices):
                xy = b[indices].astype(int)
                valid[indices] &= masks[1][xy[:, 1], xy[:, 0]] > 0
            a, b = a[valid], b[valid]
            base["match_count"] = len(a)
            stage["optical_flow"] = (time.perf_counter() - t) * 1000
            if len(a) < cfg.min_matches:
                return finish("UNKNOWN", "insufficient_valid_matches")
            t = time.perf_counter()
            cv2.setRNGSeed(cfg.rng_seed)
            matrix, inliers = cv2.estimateAffinePartial2D(
                a,
                b,
                method=cv2.RANSAC,
                ransacReprojThreshold=cfg.ransac_threshold_pixels,
                maxIters=cfg.ransac_iterations,
                confidence=cfg.ransac_confidence,
                refineIters=10,
            )
            stage["global_transform"] = (time.perf_counter() - t) * 1000
            if (
                matrix is None
                or inliers is None
                or matrix.shape != (2, 3)
                or not np.isfinite(matrix).all()
            ):
                return finish("UNKNOWN", "ransac_transform_failed_or_nonfinite")
            inliers = inliers.ravel().astype(bool)
            count = int(inliers.sum())
            base["inlier_count"] = count
            base["inlier_ratio"] = count / len(a)
            scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
            rotation = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))
            base["rotation_2d_degrees"], base["scale_2d_ratio"] = rotation, scale
            if (
                not cfg.min_scale <= scale <= cfg.max_scale
                or abs(rotation) > cfg.max_rotation_degrees_per_frame
            ):
                return finish("UNKNOWN", "transform_outside_image_model_bounds")
            predicted = a @ matrix[:, :2].T + matrix[:, 2]
            residual = np.linalg.norm(predicted - b, axis=1)
            base["residual_all_p95_pixels"] = float(np.percentile(residual, 95))
            if count:
                base["residual_inlier_median_pixels"] = float(np.median(residual[inliers]))
                base["residual_inlier_p95_pixels"] = float(np.percentile(residual[inliers], 95))
            bins = {
                (min(3, int(x / width * 4)), min(2, int(y / height * 3))) for x, y in a[inliers]
            }
            base["background_feature_coverage"] = len(bins) / 12
            conversion = np.array([current.key.width / width, current.key.height / height])
            source_matrix = matrix.copy()
            source_matrix[:, :2] = np.diag(conversion) @ matrix[:, :2] @ np.diag(1 / conversion)
            source_matrix[:, 2] = matrix[:, 2] * conversion
            base["transform_previous_to_current"] = tuple(
                tuple(float(x) for x in row) for row in source_matrix
            )
            base["translation_source_pixels"] = tuple(float(x) for x in source_matrix[:, 2])
            corners = np.array(
                [
                    [0, 0],
                    [width - 1, 0],
                    [0, height - 1],
                    [width - 1, height - 1],
                    [width / 2, height / 2],
                ]
            )
            displacements = (corners @ matrix[:, :2].T + matrix[:, 2] - corners) * conversion
            magnitude = float(np.median(np.linalg.norm(displacements, axis=1)))
            base["median_displacement_source_pixels"] = magnitude
            normalized = magnitude / math.hypot(current.key.width, current.key.height)
            rate = normalized / (base["dt_ms"] / 1000)
            base["motion_diagonals_per_second"] = rate
            base["flow_samples"] = tuple(
                {
                    "previous_xy": [float(x) for x in p * conversion],
                    "current_xy": [float(x) for x in q * conversion],
                    "inlier": bool(i),
                }
                for p, q, i in zip(a[:80], b[:80], inliers[:80], strict=True)
            )
            t = time.perf_counter()
            reasons = []
            if count < cfg.min_matches or base["inlier_ratio"] < cfg.min_inlier_ratio:
                reasons.append("insufficient_ransac_consensus")
            if base["background_feature_coverage"] < cfg.min_feature_coverage:
                reasons.append("insufficient_background_feature_coverage")
            if not count or base["residual_inlier_p95_pixels"] > cfg.residual_p95_limit:
                reasons.append("image_model_residual_too_large")
            if normalized > cfg.max_frame_displacement_fraction:
                reasons.append("image_displacement_outside_model_bounds")
            if reasons:
                result = finish("UNKNOWN", *reasons)
            else:
                base["estimation_status"] = "VALID"
                if (
                    len(a) < cfg.preferred_matches
                    or base["inlier_ratio"] < cfg.preferred_inlier_ratio
                    or base["background_feature_coverage"] < cfg.preferred_feature_coverage
                    or base["residual_inlier_p95_pixels"] > cfg.residual_p95_preferred
                    or base["blur_variance"] < cfg.blur_limited_variance
                ):
                    reasons.append("limited_match_coverage_consensus_or_image_quality")
                if (
                    previous_estimate is not None
                    and previous_estimate.key == previous.key
                    and previous_estimate.stream_id == current.stream_id
                    and previous_estimate.motion_diagonals_per_second is not None
                    and previous_estimate.estimation_status == "VALID"
                    and abs(rate - previous_estimate.motion_diagonals_per_second)
                    > cfg.discontinuity_rate_delta
                ):
                    reasons.append("image_motion_discontinuity")
                label = (
                    "LIMITED"
                    if reasons
                    else (
                        "STABLE_IMAGE_MOTION"
                        if rate <= cfg.stable_diagonals_per_second
                        else "RAPID_IMAGE_MOTION"
                        if rate >= cfg.rapid_diagonals_per_second
                        else "MODERATE_IMAGE_MOTION"
                    )
                )
                result = finish(label, *reasons)
            stage["motion_quality"] = (time.perf_counter() - t) * 1000
            return result
        except cv2.error as exc:
            base["provenance"]["opencv_error"] = str(exc)
            return finish("UNKNOWN", "opencv_estimation_error")
