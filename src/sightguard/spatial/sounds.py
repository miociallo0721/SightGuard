"""Generate short, low-level mono cues without external audio assets."""

from __future__ import annotations

import math

import numpy as np

_CUE_PATTERNS: dict[str, tuple[tuple[float, ...], float]] = {
    # Pulse lengths in seconds and the gap inserted between adjacent pulses.
    "risk_initial": ((0.11, 0.11), 0.10),
    "risk_escalation": ((0.055, 0.055, 0.055), 0.055),
    "perception_limited": ((0.25,), 0.0),
    "navigation": ((0.055, 0.16), 0.075),
    "system_test": ((0.06, 0.06, 0.06), 0.20),
}
_TONE_HZ = 660.0
_PEAK = 0.18
_FADE_SECONDS = 0.006


def synthesize(cue_type: str, sample_rate: int = 48_000) -> np.ndarray:
    """Create a conservative, fade-shaped ``int16`` mono cue.

    Cue categories share one tone and peak level; their pulse rhythm carries the
    distinction. No audio device or optional synthesis dependency is required.
    """
    if cue_type not in _CUE_PATTERNS:
        supported = ", ".join(_CUE_PATTERNS)
        raise ValueError(f"unknown cue_type {cue_type!r}; expected one of: {supported}")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")

    pulse_lengths, gap_seconds = _CUE_PATTERNS[cue_type]
    gap_samples = round(gap_seconds * sample_rate)
    chunks: list[np.ndarray] = []
    fade_samples = max(
        1,
        min(
            round(_FADE_SECONDS * sample_rate),
            round(min(pulse_lengths) * sample_rate / 2),
        ),
    )

    for index, pulse_length in enumerate(pulse_lengths):
        count = max(1, round(pulse_length * sample_rate))
        times = np.arange(count, dtype=np.float64) / sample_rate
        pulse = np.sin(2.0 * math.pi * _TONE_HZ * times)
        fade_in = np.linspace(0.0, 1.0, fade_samples, endpoint=False, dtype=np.float64)
        fade_out = fade_in[::-1]
        pulse[:fade_samples] *= fade_in
        pulse[-fade_samples:] *= fade_out
        chunks.append(pulse)
        if index + 1 < len(pulse_lengths) and gap_samples:
            chunks.append(np.zeros(gap_samples, dtype=np.float64))

    audio = np.concatenate(chunks)
    return np.rint(audio * (_PEAK * np.iinfo(np.int16).max)).astype(np.int16)
