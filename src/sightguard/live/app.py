"""Main-thread OpenCV window and keyboard lifecycle."""

import logging
from time import monotonic

import cv2

from sightguard.config import PerceptionConfig
from sightguard.live.config import LiveConfig
from sightguard.live.display import render_live
from sightguard.live.runtime import LiveSession
from sightguard.risk.config import RiskConfig

LOGGER = logging.getLogger(__name__)
WINDOW = "SightGuard Live - Q quit / H details"


def run_live(
    perception: PerceptionConfig,
    risk: RiskConfig,
    config: LiveConfig,
    *,
    duration_s: float | None = None,
) -> None:
    session = LiveSession(perception, risk, config)
    show_details = True
    terminal_error = None
    LOGGER.info(
        "Camera %d requested %dx%d, device=%s; Q quit, H details",
        config.camera,
        config.width,
        config.height,
        perception.device,
    )
    try:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        session.start()
        while True:
            now = monotonic()
            result, alerts = session.snapshot()
            error, failed_at = session.failure()
            cv2.imshow(
                WINDOW,
                render_live(
                    result,
                    alerts,
                    risk,
                    config,
                    now,
                    show_details=show_details,
                    error=error,
                    preview=session.frames.peek(),
                ),
            )
            key = cv2.waitKey(10) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("h"), ord("H")):
                show_details = not show_details
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if (
                error
                and failed_at is not None
                and (now - failed_at) * 1000 >= config.failure_display_ms
            ):
                terminal_error = error
                break
            if duration_s is not None and now - session.origin >= duration_s:
                break
    except KeyboardInterrupt:
        LOGGER.info("Live session interrupted")
    finally:
        try:
            session.close()
        finally:
            cv2.destroyAllWindows()
        result, alerts = session.snapshot()
        if result is not None:
            LOGGER.info(
                "Last update: processed=%d status=%s objects=%d risk_fps=%.2f "
                "processing_ms=%.1f recent_alerts=%d",
                result.record.frame_index + 1,
                result.record.status,
                len(result.record.detections),
                result.update_fps,
                result.processing_ms,
                len(alerts),
            )
        LOGGER.info("Live session closed; camera and worker resources released")
    if terminal_error:
        raise RuntimeError(terminal_error)
