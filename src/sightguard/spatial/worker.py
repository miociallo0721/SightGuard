"""One audio-context owner, nonblocking producers and bounded diagnostic history."""

import logging
import math
from collections import deque
from collections.abc import Callable
from dataclasses import asdict
from threading import Condition, Event, Lock, Thread
from time import monotonic

from sightguard.spatial.config import SpatialConfig
from sightguard.spatial.coordinates import to_openal
from sightguard.spatial.scheduler import CueScheduler
from sightguard.spatial.schemas import SpatialCue

LOGGER = logging.getLogger(__name__)
_AUDIO_LEASE = Lock()


class AudioWorker:
    def __init__(
        self,
        config: SpatialConfig,
        *,
        backend_factory: Callable | None = None,
        clock_ms: Callable[[], float] | None = None,
    ) -> None:
        self.config = config
        self.origin = monotonic()
        self.clock_ms = clock_ms or (lambda: (monotonic() - self.origin) * 1000)
        self.backend_factory = backend_factory
        self.scheduler = CueScheduler(config.queue_capacity)
        self.condition = Condition()
        self.ready = Event()
        self.stopping = Event()
        self.thread: Thread | None = None
        self.backend_status = None
        self.error: str | None = None
        self.enabled = True
        self.gain = config.gain
        self.force_stop = False
        self.records: deque[dict] = deque(maxlen=512)
        self.enqueued_at: dict[str, float] = {}

    def now_ms(self) -> float:
        return self.clock_ms()

    def start(self) -> None:
        if self.thread is not None:
            raise RuntimeError("AudioWorker cannot be reused")
        self.thread = Thread(target=self._run, name="sightguard-audio", daemon=True)
        self.thread.start()

    def submit(self, cue: SpatialCue) -> bool:
        with self.condition:
            if not self.enabled or self.stopping.is_set() or self.error:
                return False
            now = self.now_ms()
            accepted = self.scheduler.submit(cue, now)
            if accepted:
                pending_ids = {item.cue_id for item in self.scheduler.pending}
                self.enqueued_at = {
                    key: value for key, value in self.enqueued_at.items() if key in pending_ids
                }
                self.enqueued_at[cue.cue_id] = now
                self.condition.notify()
            return accepted

    def stop(self) -> None:
        """Cancel queued and active playback without closing the device."""
        with self.condition:
            self.scheduler.clear()
            self.enqueued_at.clear()
            self.force_stop = True
            self.condition.notify()

    def set_enabled(self, enabled: bool) -> None:
        if type(enabled) is not bool:
            raise ValueError("enabled must be bool")
        with self.condition:
            self.enabled = enabled
            if not enabled:
                self.stop()

    def set_gain(self, gain: float) -> None:
        if (
            isinstance(gain, bool)
            or not isinstance(gain, (int, float))
            or not math.isfinite(gain)
            or not 0 <= gain <= 1
        ):
            raise ValueError("gain must be in 0..1")
        with self.condition:
            self.gain = gain

    def pause_navigation(self, paused: bool) -> None:
        if type(paused) is not bool:
            raise ValueError("paused must be bool")
        with self.condition:
            self.scheduler.navigation_paused = paused
            self.condition.notify()

    def cancel_navigation(self) -> None:
        with self.condition:
            self.scheduler.cancel_navigation()
            self.condition.notify()

    def cancel(self, cue_id: str) -> None:
        with self.condition:
            self.scheduler.cancel(cue_id)
            self.condition.notify()

    def snapshot(self) -> dict:
        with self.condition:
            return {
                "audio_status": asdict(self.backend_status) if self.backend_status else None,
                "ready": self.ready.is_set(),
                "running": bool(self.thread and self.thread.is_alive()),
                "error": self.error,
                "enabled": self.enabled,
                "gain": self.gain,
                "queued": len(self.scheduler.pending),
                "active_cue": asdict(self.scheduler.active) if self.scheduler.active else None,
                "navigation_paused": self.scheduler.navigation_paused,
                "records": list(self.records),
            }

    def _run(self) -> None:
        backend = None
        owns_device = False
        try:
            owns_device = _AUDIO_LEASE.acquire(blocking=False)
            if not owns_device:
                raise RuntimeError(
                    "Audio is already owned by another session; close its Lab or Live session"
                )
            from sightguard.spatial.openal_backend import OpenALBackend
            from sightguard.spatial.sounds import synthesize

            backend = (
                self.backend_factory()
                if self.backend_factory
                else OpenALBackend(
                    self.config.library_path, self.config.device_name, self.config.request_hrtf
                )
            )
            status = backend.open()
            with self.condition:
                self.backend_status = status
            LOGGER.info("Spatial audio initialized: %s", status)
            self.ready.set()
            sounds: dict = {}
            while not self.stopping.is_set():
                playing = backend.is_playing()
                with self.condition:
                    should_stop = self.force_stop
                    self.force_stop = False
                    decision = self.scheduler.update(
                        self.now_ms(), playing=playing and not should_stop
                    )
                    gain = self.gain
                if should_stop or decision.stop:
                    backend.stop()
                if decision.cue is not None:
                    cue = decision.cue
                    if cue.cue_type not in sounds:
                        sounds[cue.cue_type] = synthesize(cue.cue_type)
                    # Recheck deadlines after sound creation, immediately before native submission.
                    now = self.now_ms()
                    with self.condition:
                        permitted = (
                            self.enabled
                            and self.scheduler.active == cue
                            and cue.cue_id not in self.scheduler.cancelled
                            and not (
                                cue.source.value == "navigation"
                                and self.scheduler.navigation_paused
                            )
                        )
                    if permitted and cue.expires_at_ms > now and not self.stopping.is_set():
                        backend.play(
                            sounds[cue.cue_type],
                            48000,
                            to_openal(
                                cue.azimuth_deg, cue.elevation_deg, self.config.virtual_radius
                            ),
                            gain,
                        )
                        backend_playing = backend.is_playing()
                        submitted = self.now_ms()
                        with self.condition:
                            arrived = self.enqueued_at.pop(cue.cue_id, submitted)
                            self.records.append(
                                {
                                    "cue": asdict(cue),
                                    "submitted_at_ms": submitted,
                                    "enqueue_to_submit_ms": max(0, submitted - arrived),
                                    "cue_age_ms": submitted - cue.timestamp_ms,
                                    "backend_reported_playing": backend_playing,
                                    "action": decision.reason,
                                }
                            )
                with self.condition:
                    self.condition.wait(timeout=self.config.poll_interval_ms / 1000)
        except Exception as exc:
            with self.condition:
                self.error = f"{type(exc).__name__}: {exc}"
            LOGGER.warning("Spatial audio unavailable; vision can continue: %s", self.error)
        finally:
            self.ready.set()
            if backend is not None:
                try:
                    backend.close()
                except Exception as exc:
                    with self.condition:
                        self.error = self.error or f"Audio cleanup failed: {exc}"
            if owns_device:
                _AUDIO_LEASE.release()
            with self.condition:
                self.scheduler.clear()
                self.enqueued_at.clear()

    def close(self) -> None:
        self.stopping.set()
        with self.condition:
            self.condition.notify_all()
        if self.thread is not None:
            self.thread.join(self.config.join_timeout_s)
            if self.thread.is_alive():
                raise RuntimeError("Audio backend did not exit within shutdown timeout")
