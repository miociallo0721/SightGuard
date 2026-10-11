"""Camera/listener angles: left negative, front 0; OpenAL front is negative Z."""

import math


def normalize_azimuth(angle: float) -> float:
    if isinstance(angle, bool) or not isinstance(angle, (int, float)) or not math.isfinite(angle):
        raise ValueError("Angle must be finite")
    return (angle + 180) % 360 - 180


def to_openal(
    azimuth_deg: float, elevation_deg: float = 0, radius: float = 1
) -> tuple[float, float, float]:
    azimuth = math.radians(normalize_azimuth(azimuth_deg))
    if not math.isfinite(elevation_deg) or not -90 <= elevation_deg <= 90:
        raise ValueError("Invalid elevation")
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("Invalid virtual radius")
    elevation = math.radians(elevation_deg)
    return (
        radius * math.cos(elevation) * math.sin(azimuth),
        radius * math.sin(elevation),
        -radius * math.cos(elevation) * math.cos(azimuth),
    )


def camera_azimuth(
    center_x: float,
    *,
    horizontal_fov_deg: float | None = None,
    approximate_half_angle_deg: float = 45,
) -> tuple[float, str]:
    if not math.isfinite(center_x) or not 0 <= center_x <= 1:
        raise ValueError("Center must be normalized to image width")
    offset = 2 * center_x - 1
    if horizontal_fov_deg is None:
        if (
            not math.isfinite(approximate_half_angle_deg)
            or not 0 < approximate_half_angle_deg <= 90
        ):
            raise ValueError("Invalid approximate half angle")
        return offset * approximate_half_angle_deg, "approximate_screen"
    if not math.isfinite(horizontal_fov_deg) or not 0 < horizontal_fov_deg < 180:
        raise ValueError("Invalid calibrated horizontal field of view")
    return math.degrees(
        math.atan(offset * math.tan(math.radians(horizontal_fov_deg / 2)))
    ), "camera_geometry"
