"""Small, serializable audio-backend interfaces for spatial cues."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True, slots=True)
class AudioStatus:
    """Runtime audio state reported by a backend after opening its device."""

    backend: str
    device: str
    hrtf_supported: bool
    hrtf_enabled: bool
    hrtf_status: str
    hrtf_dataset: str | None
    fallback_reason: str | None


class AudioBackend(Protocol):
    """Audio device interface intended to be used from one dedicated thread."""

    def open(self) -> AudioStatus:
        """Open the backend and report the active device and spatialization state."""

    def play(
        self,
        samples: np.ndarray,
        sample_rate: int,
        position: tuple[float, float, float],
        gain: float,
    ) -> None:
        """Play mono signed 16-bit PCM at a direction relative to the listener."""

    def is_playing(self) -> bool:
        """Return whether the current cue is still playing."""

    def stop(self) -> None:
        """Stop the current cue, if any."""

    def close(self) -> None:
        """Release the device and all backend resources."""
