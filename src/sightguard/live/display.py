"""OpenCV presentation of actual engine output on its corresponding captured image."""

import textwrap

import cv2
import numpy as np

from sightguard.live.config import LiveConfig
from sightguard.live.runtime import CapturedFrame, LiveResult
from sightguard.risk.config import RiskConfig
from sightguard.risk.schema import AlertEvent

COLORS = {
    "LOW": (50, 210, 70),
    "MEDIUM": (0, 220, 255),
    "HIGH": (40, 40, 255),
    "UNKNOWN": (150, 150, 150),
}
ALERT_LABELS = {
    "first": "FIRST ALERT",
    "escalation": "RISK ESCALATION",
    "perception_limited": "PERCEPTION LIMITED",
    "repeat": "REPEAT ALERT",
}


def render_live(
    result: LiveResult | None,
    alerts: tuple[tuple[AlertEvent, float], ...],
    risk_config: RiskConfig,
    live_config: LiveConfig,
    now: float,
    *,
    show_details: bool = True,
    error: str | None = None,
    preview: CapturedFrame | None = None,
    include_panel: bool = True,
) -> np.ndarray:
    packet = result.captured if result is not None else preview
    image = (
        packet.image.copy()
        if packet is not None
        else np.zeros((live_config.height, live_config.width, 3), dtype=np.uint8)
    )
    height, width = image.shape[:2]
    stale = packet is None or (now - packet.captured_at) * 1000 > live_config.stale_after_ms
    valid = result is not None and result.record.status == "ok" and not stale and not error
    focus = risk_config.features
    polygon = np.array(
        [
            (
                (focus.focus_center_x - focus.focus_top_half_width) * width,
                focus.focus_top_y * height,
            ),
            (
                (focus.focus_center_x + focus.focus_top_half_width) * width,
                focus.focus_top_y * height,
            ),
            (
                (focus.focus_center_x + focus.focus_bottom_half_width) * width,
                focus.focus_bottom_y * (height - 1),
            ),
            (
                (focus.focus_center_x - focus.focus_bottom_half_width) * width,
                focus.focus_bottom_y * (height - 1),
            ),
        ],
        dtype=np.int32,
    )
    cv2.polylines(image, [polygon], True, (210, 160, 50), 2)
    if valid:
        observations = {o.detection_index: o for o in result.decision.frame.observations}
        events = {e.event_id: e for e in result.decision.frame.events}
        for index, detection in enumerate(result.record.detections):
            observation = observations.get(index)
            assessment = observation.assessment if observation else None
            event = events.get(observation.event_id) if observation else None
            level = assessment.level.value if assessment else "UNKNOWN"
            score = f"{assessment.score:.3f}" if assessment else "N/A"
            color = COLORS[level]
            x1, y1, x2, y2 = (
                int(np.clip(v, 0, limit - 1))
                for v, limit in zip(
                    detection.bbox_xyxy, (width, height, width, height), strict=True
                )
            )
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            identity = detection.track_id if detection.track_id is not None else "?"
            lines = (
                f"{detection.class_name} track:{identity}",
                f"{observation.event_id if observation and observation.event_id else '?'} "
                f"{level} R:{score} {event.state.value if event else 'UNKNOWN'}",
            )
            for offset, line in enumerate(lines):
                y = max(16, y1 - 24) + offset * 17
                cv2.putText(
                    image, line, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 0, 0), 3, cv2.LINE_AA
                )
                cv2.putText(
                    image, line, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, color, 1, cv2.LINE_AA
                )
    if not include_panel:
        if alerts and (now - alerts[-1][1]) * 1000 <= live_config.alert_banner_ms:
            cv2.rectangle(image, (0, 0), (width - 1, height - 1), COLORS["HIGH"], 5)
        return image
    panel_width = 470
    canvas = np.full((max(height, 560), width + panel_width, 3), (22, 25, 29), dtype=np.uint8)
    canvas[:height, :width] = image
    x, y = width + 12, 28

    def line(text: str, color: tuple[int, int, int] = (230, 230, 230)) -> None:
        nonlocal y, canvas
        for part in textwrap.wrap(text, 57) or [""]:
            if y + 8 >= canvas.shape[0]:
                canvas = cv2.copyMakeBorder(
                    canvas, 0, 76, 0, 0, cv2.BORDER_CONSTANT, value=(22, 25, 29)
                )
            cv2.putText(canvas, part, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, color, 1, cv2.LINE_AA)
            y += 19

    line("SightGuard | LIVE PC PROTOTYPE", (210, 190, 80))
    line("Q: quit   H: details")
    line("Image-space risk; not collision prediction")
    if error:
        line(f"ERROR: {error}", COLORS["HIGH"])
    elif stale:
        line("STALE / NO CAMERA FRAME - risk UNKNOWN", COLORS["HIGH"])
    elif result is None:
        line("Waiting for first model / risk update", COLORS["UNKNOWN"])
    elif result.record.status != "ok":
        line(f"PERCEPTION ERROR: {result.record.error}", COLORS["HIGH"])
    else:
        line(f"Perception: {result.decision.frame.perception.status}")
    if packet is not None:
        line(f"Camera frame: {packet.capture_index} | t={packet.timestamp_ms:.1f} ms")
        line(f"Frame age: {max(0, now - packet.captured_at) * 1000:.1f} ms")
    if result is not None:
        line(
            f"Risk updates: {result.update_fps if valid else 0:.2f} FPS | "
            f"last processing: {result.processing_ms:.1f} ms"
        )
        line(f"Processed index: {result.record.frame_index} | timestamp: monotonic")
        if show_details:
            for event in result.decision.frame.events[:6]:
                assessment = event.assessment
                level = assessment.level.value if assessment and valid else "UNKNOWN"
                score = f"{assessment.score:.3f}" if assessment and valid else "N/A"
                line(
                    f"{event.event_id} track:{event.last_track_id} {event.state.value} "
                    f"{level} R:{score}" + (" (history)" if not valid else ""),
                    COLORS[level],
                )
            health = result.decision.frame.perception
            if health.event_id:
                line(f"{health.event_id}: {health.state} ({health.status})", COLORS["UNKNOWN"])
    if alerts:
        alert, arrived_at = alerts[-1]
        active = (now - arrived_at) * 1000 <= live_config.alert_banner_ms
        line("ALERT NOW" if active else "LAST ALERT (history)", COLORS["HIGH"])
        line(
            f"{ALERT_LABELS.get(alert.alert_type, alert.alert_type)} | {alert.event_id}",
            COLORS["HIGH"] if active else COLORS["UNKNOWN"],
        )
        line(f"Reason: {alert.reason}")
        if active:
            cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), COLORS["HIGH"], 5)
        if show_details:
            line(f"Generated at t={alert.timestamp_ms:.1f} ms")
            line("Recent actual alerts (same-level repeats off by default):")
            for previous, _ in alerts[-4:]:
                line(f"{previous.alert_type} {previous.event_id} @{previous.timestamp_ms:.0f}ms")
    else:
        line("No AlertEvent generated yet", COLORS["UNKNOWN"])
    return canvas
