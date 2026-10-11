"""OpenAL Soft audio backend with a ctypes-only runtime dependency."""

from __future__ import annotations

import ctypes
import ctypes.util
import math
import os
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np

from sightguard.spatial.renderer import AudioStatus

# ALC core and ALC_SOFT_HRTF values from OpenAL Soft's AL/alext.h.
_ALC_DEVICE_SPECIFIER = 0x1005
_ALC_ALL_DEVICES_SPECIFIER = 0x1013
_ALC_ENUMERATE_ALL_EXT = b"ALC_ENUMERATE_ALL_EXT"
_ALC_ENUMERATION_EXT = b"ALC_ENUMERATION_EXT"
_ALC_HRTF_SOFT = 0x1992
_ALC_HRTF_STATUS_SOFT = 0x1993
_ALC_HRTF_SPECIFIER_SOFT = 0x1995
_ALC_HRTF_DISABLED_SOFT = 0
_ALC_HRTF_ENABLED_SOFT = 1
_ALC_HRTF_DENIED_SOFT = 2
_ALC_HRTF_REQUIRED_SOFT = 3
_ALC_HRTF_HEADPHONES_DETECTED_SOFT = 4
_ALC_HRTF_UNSUPPORTED_FORMAT_SOFT = 5

# AL core values.
_AL_NO_ERROR = 0
_AL_FORMAT_MONO16 = 0x1101
_AL_BUFFER = 0x1009
_AL_POSITION = 0x1004
_AL_GAIN = 0x100A
_AL_ROLLOFF_FACTOR = 0x1021
_AL_SOURCE_STATE = 0x1010
_AL_PLAYING = 0x1012
_AL_RENDERER = 0xB003
_AL_VERSION = 0xB002

_HRTF_STATUS_NAMES = {
    _ALC_HRTF_DISABLED_SOFT: "disabled",
    _ALC_HRTF_ENABLED_SOFT: "enabled",
    _ALC_HRTF_DENIED_SOFT: "denied",
    _ALC_HRTF_REQUIRED_SOFT: "required",
    _ALC_HRTF_HEADPHONES_DETECTED_SOFT: "headphones_detected",
    _ALC_HRTF_UNSUPPORTED_FORMAT_SOFT: "unsupported_format",
}
_ALC_INTP = ctypes.POINTER(ctypes.c_int)
_ALUINTP = ctypes.POINTER(ctypes.c_uint)
_GLOBAL_CONTEXT_LOCK = threading.Lock()


def _configure(function: Any, argtypes: list[Any], restype: Any) -> Any:
    """Set signatures centrally so OpenAL pointers are never int-truncated."""
    function.argtypes = argtypes
    function.restype = restype
    return function


def _bind_library(library: Any) -> None:
    """Assign explicit ctypes signatures to every symbol used by this module."""
    device = ctypes.c_void_p
    context = ctypes.c_void_p
    char_pointer = ctypes.c_void_p
    al_uint = ctypes.c_uint
    al_int = ctypes.c_int
    al_float = ctypes.c_float
    signatures = {
        "alcOpenDevice": ([ctypes.c_char_p], device),
        "alcCloseDevice": ([device], ctypes.c_ubyte),
        "alcCreateContext": ([device, _ALC_INTP], context),
        "alcMakeContextCurrent": ([context], ctypes.c_ubyte),
        "alcGetProcAddress": ([device, ctypes.c_char_p], ctypes.c_void_p),
        "alcDestroyContext": ([context], None),
        "alcGetString": ([device, al_int], char_pointer),
        "alcGetIntegerv": ([device, al_int, al_int, _ALC_INTP], None),
        "alcGetError": ([device], al_int),
        "alcIsExtensionPresent": ([device, ctypes.c_char_p], ctypes.c_ubyte),
        "alGetString": ([al_int], char_pointer),
        "alGetError": ([], al_int),
        "alDopplerFactor": ([al_float], None),
        "alGenSources": ([al_int, _ALUINTP], None),
        "alDeleteSources": ([al_int, _ALUINTP], None),
        "alSourceStop": ([al_uint], None),
        "alSourcePlay": ([al_uint], None),
        "alSourcei": ([al_uint, al_int, al_int], None),
        "alSourcef": ([al_uint, al_int, al_float], None),
        "alSource3f": ([al_uint, al_int, al_float, al_float, al_float], None),
        "alGetSourcei": ([al_uint, al_int, _ALC_INTP], None),
        "alGenBuffers": ([al_int, _ALUINTP], None),
        "alDeleteBuffers": ([al_int, _ALUINTP], None),
        "alBufferData": ([al_uint, al_int, ctypes.c_void_p, al_int, al_int], None),
    }
    for name, (argtypes, restype) in signatures.items():
        _configure(getattr(library, name), argtypes, restype)


def _library_candidates(library_path: str | Path | None) -> list[str]:
    if library_path is not None:
        return [str(Path(library_path).expanduser())]

    candidates: list[str] = []
    configured_library = os.environ.get("SIGHTGUARD_OPENAL_LIBRARY")
    if configured_library:
        candidates.append(str(Path(configured_library).expanduser()))
    working_directory_dll = Path.cwd() / "models" / "openal-soft" / "soft_oal.dll"
    if working_directory_dll.is_file():
        candidate = str(working_directory_dll)
        if candidate not in candidates:
            candidates.append(candidate)
    repository_dll = Path(__file__).resolve().parents[3] / "models" / "openal-soft" / "soft_oal.dll"
    if repository_dll.is_file() and str(repository_dll) not in candidates:
        candidates.append(str(repository_dll))

    if os.name == "nt":
        names = ("soft_oal.dll",)
    elif os.name == "posix" and sys.platform == "darwin":
        names = ("libopenal.1.dylib", "libopenal.dylib")
    else:
        names = ("libopenal.so.1", "libopenal.so")
    for name in names:
        found = ctypes.util.find_library(name)
        if found and found not in candidates:
            candidates.append(found)
        candidates.append(name)
    return candidates


def _load_library(library_path: str | Path | None = None) -> Any:
    candidates = _library_candidates(library_path)
    errors: list[str] = []
    for candidate in candidates:
        try:
            library = ctypes.CDLL(candidate)
            _bind_library(library)
            return library
        except (OSError, AttributeError) as exc:
            errors.append(f"{candidate}: {exc}")
    detail = "; ".join(errors) if errors else "no library candidates were found"
    raise RuntimeError(f"could not load an OpenAL library ({detail})")


def _read_c_string(pointer: Any) -> str:
    if pointer is None or pointer == 0:
        return ""
    try:
        raw = pointer if isinstance(pointer, bytes) else ctypes.string_at(pointer)
        return raw.decode("utf-8", errors="replace")
    except (TypeError, ValueError, OSError):
        return ""


def _read_multi_string(pointer: Any, limit: int = 65_536) -> list[str]:
    """Decode an ALC double-NUL list without truncating it at its first item."""
    if pointer is None or pointer == 0:
        return []
    try:
        data = ctypes.cast(pointer, ctypes.POINTER(ctypes.c_ubyte))
        values: list[bytes] = []
        current = bytearray()
        saw_terminator = False
        for index in range(limit):
            value = int(data[index])
            if value == 0:
                if not current:
                    if saw_terminator:
                        break
                    saw_terminator = True
                    continue
                values.append(bytes(current))
                current.clear()
                saw_terminator = True
            else:
                current.append(value)
                saw_terminator = False
        return [value.decode("utf-8", errors="replace") for value in values]
    except (TypeError, ValueError, OSError):
        return []


def enumerate_devices(library_path: str | Path | None = None) -> list[str]:
    """List playback devices via ALC enumeration, without opening a context.

    ``library_path`` selects the same OpenAL shared library that should be
    inspected. If ``ALC_ENUMERATE_ALL_EXT`` is present, the list includes all
    endpoints; otherwise the standard ``ALC_ENUMERATION_EXT`` list is used.
    """
    library = _load_library(library_path)
    all_devices = bool(library.alcIsExtensionPresent(None, _ALC_ENUMERATE_ALL_EXT))
    if not all_devices and not bool(library.alcIsExtensionPresent(None, _ALC_ENUMERATION_EXT)):
        return []
    token = _ALC_ALL_DEVICES_SPECIFIER if all_devices else _ALC_DEVICE_SPECIFIER
    return _read_multi_string(library.alcGetString(None, token))


class OpenALBackend:
    """OpenAL backend with explicit HRTF requests and normal-stereo fallback.

    Create, open, use, and close the object on one dedicated worker thread.
    ``position`` is interpreted as a direction and normalized to a fixed
    one-metre radius, so distance does not imply an unverified risk estimate.
    Without ``library_path``, discovery checks ``SIGHTGUARD_OPENAL_LIBRARY``,
    then ``models/openal-soft/soft_oal.dll`` under the working directory, and
    finally platform library discovery (``find_library('openal')`` on Linux).
    """

    def __init__(
        self,
        library_path: str | Path | None = None,
        device_name: str | None = None,
        request_hrtf: bool = True,
    ) -> None:
        self.library_path = library_path
        self.device_name = device_name
        self.request_hrtf = request_hrtf
        self._library: Any | None = None
        self._device: int | None = None
        self._context: int | None = None
        self._source = 0
        self._buffer = 0
        self._owner_thread_id: int | None = None
        self._status: AudioStatus | None = None
        self._thread_context_setter: Any | None = None
        self._thread_local_context = False
        self._global_lock_owned = False

    def open(self) -> AudioStatus:
        if self._context is not None and self._status is not None:
            self._check_thread()
            return self._status

        self._owner_thread_id = threading.get_ident()
        try:
            self._library = _load_library(self.library_path)
            encoded_name = self.device_name.encode("utf-8") if self.device_name else None
            self._device = self._library.alcOpenDevice(encoded_name)
            if not self._device:
                raise RuntimeError("alcOpenDevice failed; no playback device is available")

            hrtf_supported = bool(
                self._library.alcIsExtensionPresent(self._device, b"ALC_SOFT_HRTF")
            )
            thread_context_extension = bool(
                self._library.alcIsExtensionPresent(self._device, b"ALC_EXT_thread_local_context")
            )
            if thread_context_extension:
                setter_address = self._library.alcGetProcAddress(
                    self._device, b"alcSetThreadContext"
                )
                if setter_address:
                    setter_type = ctypes.CFUNCTYPE(ctypes.c_ubyte, ctypes.c_void_p)
                    self._thread_context_setter = setter_type(setter_address)
                    self._thread_local_context = True
            if not self._thread_local_context:
                if not _GLOBAL_CONTEXT_LOCK.acquire(blocking=False):
                    raise RuntimeError(
                        "another OpenAL backend owns the process-wide context; "
                        "ALC_EXT_thread_local_context is unavailable"
                    )
                self._global_lock_owned = True

            attributes: Any = None
            if hrtf_supported:
                requested_value = 1 if self.request_hrtf else 0
                attributes = (ctypes.c_int * 3)(_ALC_HRTF_SOFT, requested_value, 0)

            context = self._library.alcCreateContext(self._device, attributes)
            context_failure_reason: str | None = None
            if not context and hrtf_supported and self.request_hrtf:
                error_code = int(self._library.alcGetError(self._device))
                context_failure_reason = f"HRTF context creation failed (ALC error {error_code})"
                context = self._library.alcCreateContext(self._device, None)
            if not context:
                error_code = int(self._library.alcGetError(self._device))
                raise RuntimeError(f"alcCreateContext failed (ALC error {error_code})")
            self._context = context
            if not self._set_current_context(self._context):
                error_code = int(self._library.alcGetError(self._device))
                raise RuntimeError(
                    f"setting current OpenAL context failed (ALC error {error_code})"
                )

            self._library.alDopplerFactor(0.0)
            source_out = ctypes.c_uint()
            self._library.alGenSources(1, ctypes.byref(source_out))
            self._source = int(source_out.value)
            self._check_al_error("alGenSources")
            if self._source == 0:
                raise RuntimeError("alGenSources returned source id 0")

            hrtf_status_code = _ALC_HRTF_DISABLED_SOFT
            hrtf_enabled = False
            dataset: str | None = None
            if hrtf_supported:
                status_out = ctypes.c_int()
                self._library.alcGetIntegerv(
                    self._device, _ALC_HRTF_STATUS_SOFT, 1, ctypes.byref(status_out)
                )
                hrtf_status_code = int(status_out.value)
                enabled_out = ctypes.c_int()
                self._library.alcGetIntegerv(
                    self._device, _ALC_HRTF_SOFT, 1, ctypes.byref(enabled_out)
                )
                hrtf_enabled = bool(enabled_out.value)
                if hrtf_enabled:
                    dataset = (
                        _read_c_string(
                            self._library.alcGetString(self._device, _ALC_HRTF_SPECIFIER_SOFT)
                        )
                        or None
                    )

            hrtf_status = (
                _HRTF_STATUS_NAMES.get(hrtf_status_code, f"unknown:{hrtf_status_code}")
                if hrtf_supported
                else "unsupported"
            )
            fallback_reason = context_failure_reason if not hrtf_enabled else None
            if self.request_hrtf and not hrtf_supported:
                fallback_reason = "ALC_SOFT_HRTF is unavailable; using normal stereo output"
            elif self.request_hrtf and not hrtf_enabled and fallback_reason is None:
                fallback_reason = (
                    f"HRTF was not enabled ({hrtf_status}); using normal stereo output"
                )
            elif not self.request_hrtf and not hrtf_enabled:
                fallback_reason = "HRTF disabled by request; using normal stereo output"

            device = ""
            if self._library.alcIsExtensionPresent(self._device, _ALC_ENUMERATE_ALL_EXT):
                device = _read_c_string(
                    self._library.alcGetString(self._device, _ALC_ALL_DEVICES_SPECIFIER)
                )
            if not device:
                device = _read_c_string(
                    self._library.alcGetString(self._device, _ALC_DEVICE_SPECIFIER)
                )
            if not device:
                device = self.device_name or "default device"
            version = _read_c_string(self._library.alGetString(_AL_VERSION))
            renderer = _read_c_string(self._library.alGetString(_AL_RENDERER))
            if "alsoft" in version.lower() or "openal soft" in version.lower():
                backend_name = f"OpenAL Soft {version}".strip()
            elif renderer:
                backend_name = f"OpenAL ({renderer})"
            else:
                backend_name = "OpenAL"

            self._status = AudioStatus(
                backend=backend_name,
                device=device,
                hrtf_supported=hrtf_supported,
                hrtf_enabled=hrtf_enabled,
                hrtf_status=hrtf_status,
                hrtf_dataset=dataset,
                fallback_reason=fallback_reason,
            )
            return self._status
        except Exception:
            self._cleanup()
            raise

    def play(
        self,
        samples: np.ndarray,
        sample_rate: int,
        position: tuple[float, float, float],
        gain: float,
    ) -> None:
        self._require_open()
        if not isinstance(samples, np.ndarray) or samples.ndim != 1 or samples.dtype != np.int16:
            raise TypeError("samples must be a one-dimensional numpy int16 array")
        if samples.size == 0:
            raise ValueError("samples must not be empty")
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer")
        if len(position) != 3 or not all(math.isfinite(float(value)) for value in position):
            raise ValueError("position must contain three finite coordinates")
        if not math.isfinite(float(gain)) or gain < 0.0:
            raise ValueError("gain must be finite and non-negative")

        audio = np.ascontiguousarray(samples)
        fixed_position = self._fixed_radius_position(position)
        self.stop()
        if self._buffer:
            old_buffer = ctypes.c_uint(self._buffer)
            self._library.alSourcei(self._source, _AL_BUFFER, 0)
            self._library.alDeleteBuffers(1, ctypes.byref(old_buffer))
            self._buffer = 0

        buffer_out = ctypes.c_uint()
        self._library.alGenBuffers(1, ctypes.byref(buffer_out))
        self._buffer = int(buffer_out.value)
        try:
            self._check_al_error("alGenBuffers")
            if self._buffer == 0:
                raise RuntimeError("alGenBuffers returned buffer id 0")
            self._library.alBufferData(
                self._buffer,
                _AL_FORMAT_MONO16,
                audio.ctypes.data_as(ctypes.c_void_p),
                int(audio.nbytes),
                sample_rate,
            )
            self._check_al_error("alBufferData")
            self._library.alSourcei(self._source, _AL_BUFFER, self._buffer)
            self._library.alSource3f(self._source, _AL_POSITION, *fixed_position)
            self._library.alSourcef(self._source, _AL_GAIN, min(float(gain), 1.0))
            # Position supplies direction only; keep attenuation independent of range.
            self._library.alSourcef(self._source, _AL_ROLLOFF_FACTOR, 0.0)
            self._library.alSourcePlay(self._source)
            self._check_al_error("alSourcePlay")
        except Exception:
            self._discard_buffer()
            raise

    def is_playing(self) -> bool:
        self._require_open()
        state = ctypes.c_int()
        self._library.alGetSourcei(self._source, _AL_SOURCE_STATE, ctypes.byref(state))
        self._check_al_error("alGetSourcei")
        return state.value == _AL_PLAYING

    def stop(self) -> None:
        self._require_open()
        self._library.alSourceStop(self._source)
        self._check_al_error("alSourceStop")

    def close(self) -> None:
        if self._owner_thread_id is not None:
            self._check_thread()
        self._cleanup()

    @staticmethod
    def _fixed_radius_position(position: tuple[float, float, float]) -> tuple[float, float, float]:
        vector = tuple(float(value) for value in position)
        length = math.sqrt(sum(value * value for value in vector))
        if length == 0.0:
            return (0.0, 0.0, -1.0)
        return tuple(value / length for value in vector)  # type: ignore[return-value]

    def _check_al_error(self, operation: str) -> None:
        error_code = int(self._library.alGetError())
        if error_code != _AL_NO_ERROR:
            raise RuntimeError(f"{operation} failed (AL error {error_code})")

    def _set_current_context(self, context: int | None) -> bool:
        if self._thread_local_context:
            return bool(self._thread_context_setter(context))
        return bool(self._library.alcMakeContextCurrent(context))

    def _check_thread(self) -> None:
        if self._owner_thread_id != threading.get_ident():
            raise RuntimeError("OpenAL backend must be used on the same thread that opened it")

    def _require_open(self) -> None:
        self._check_thread()
        if self._library is None or self._context is None or not self._source:
            raise RuntimeError("OpenAL backend is not open")

    def _discard_buffer(self) -> None:
        if self._buffer and self._library is not None:
            try:
                buffer_id = ctypes.c_uint(self._buffer)
                self._library.alSourceStop(self._source)
                self._library.alSourcei(self._source, _AL_BUFFER, 0)
                self._library.alDeleteBuffers(1, ctypes.byref(buffer_id))
            except Exception:
                pass
            self._buffer = 0

    def _cleanup(self) -> None:
        library = self._library
        if library is not None:
            if self._source:
                try:
                    library.alSourceStop(self._source)
                except Exception:
                    pass
            if self._buffer:
                self._discard_buffer()
            if self._source:
                try:
                    source_id = ctypes.c_uint(self._source)
                    library.alDeleteSources(1, ctypes.byref(source_id))
                except Exception:
                    pass
                self._source = 0
            if self._context:
                try:
                    self._set_current_context(None)
                except Exception:
                    pass
                try:
                    library.alcDestroyContext(self._context)
                except Exception:
                    pass
                self._context = None
            if self._device:
                try:
                    library.alcCloseDevice(self._device)
                except Exception:
                    pass
                self._device = None
        self._thread_context_setter = None
        self._thread_local_context = False
        if self._global_lock_owned:
            self._global_lock_owned = False
            try:
                _GLOBAL_CONTEXT_LOCK.release()
            except RuntimeError:
                pass
        self._library = None
        self._owner_thread_id = None
        self._status = None
