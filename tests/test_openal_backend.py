from __future__ import annotations

import ctypes

import numpy as np
import pytest

from sightguard.spatial import openal_backend
from sightguard.spatial.openal_backend import OpenALBackend, enumerate_devices
from sightguard.spatial.sounds import synthesize


class _FakeFunction:
    def __init__(self, function):
        self.function = function
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.function(*args)


class _FakeOpenALLibrary:
    def __init__(
        self,
        *,
        hrtf_supported: bool = True,
        hrtf_enabled: bool = True,
        hrtf_status: int = openal_backend._ALC_HRTF_ENABLED_SOFT,
        fail_hrtf_context: bool = False,
        fail_current_context: bool = False,
        thread_context_supported: bool = False,
        all_devices_name: str | None = "Actual USB Headphones",
        device_specifier_name: str | None = "OpenAL Soft",
        devices: tuple[str, ...] = ("Mock Headphones", "Mock Speakers"),
    ):
        self.hrtf_supported = hrtf_supported
        self.hrtf_enabled = hrtf_enabled
        self.hrtf_status = hrtf_status
        self.fail_hrtf_context = fail_hrtf_context
        self.fail_current_context = fail_current_context
        self.thread_context_supported = thread_context_supported
        self.all_devices_name = all_devices_name
        self.device_specifier_name = device_specifier_name
        self.device_names = devices
        self.context_attributes: list[list[int] | None] = []
        self.playback_source_state = 0
        self.source_position: tuple[float, float, float] | None = None
        self.source_gain: float | None = None
        self.rolloff: float | None = None
        self.buffer_bytes: bytes | None = None
        self.buffer_frequency: int | None = None
        self.doppler_factor: float | None = None
        self.closed_device = False
        self.destroyed_context = False
        self.al_error = 0
        self.alc_error = 7
        self.calls: list[str] = []
        self.thread_context_calls: list[int | None] = []
        self._strings: dict[bytes, ctypes.Array] = {}
        callback_type = ctypes.CFUNCTYPE(ctypes.c_ubyte, ctypes.c_void_p)
        self._thread_context_callback = callback_type(self._set_thread_context)
        self._thread_context_pointer = ctypes.cast(
            self._thread_context_callback, ctypes.c_void_p
        ).value
        self._device_list = ctypes.create_string_buffer(
            b"".join(name.encode("utf-8") + b"\0" for name in devices) + b"\0"
        )

        implementations = {
            "alcOpenDevice": lambda name: self._called("alcOpenDevice", 0x10001),
            "alcCloseDevice": self._close_device,
            "alcCreateContext": self._create_context,
            "alcMakeContextCurrent": self._make_current,
            "alcGetProcAddress": self._get_proc_address,
            "alcDestroyContext": self._destroy_context,
            "alcGetString": self._get_alc_string,
            "alcGetIntegerv": self._get_alc_integer,
            "alcGetError": lambda device: self.alc_error,
            "alcIsExtensionPresent": self._has_extension,
            "alGetString": self._get_al_string,
            "alGetError": self._get_al_error,
            "alDopplerFactor": self._set_doppler,
            "alGenSources": self._gen_source,
            "alDeleteSources": lambda count, output: self.calls.append("alDeleteSources"),
            "alSourceStop": self._stop_source,
            "alSourcePlay": self._play_source,
            "alSourcei": lambda source, token, value: self.calls.append("alSourcei"),
            "alSourcef": self._set_source_float,
            "alSource3f": self._set_source_position,
            "alGetSourcei": self._get_source_state,
            "alGenBuffers": self._gen_buffer,
            "alDeleteBuffers": lambda count, output: self.calls.append("alDeleteBuffers"),
            "alBufferData": self._buffer_data,
        }
        for name, function in implementations.items():
            setattr(self, name, _FakeFunction(function))

    def _called(self, name: str, value: int) -> int:
        self.calls.append(name)
        return value

    def _close_device(self, device):
        self.calls.append("alcCloseDevice")
        self.closed_device = True
        return 1

    def _create_context(self, device, attributes):
        if attributes is None:
            values = None
        else:
            values = list(attributes)
        self.context_attributes.append(values)
        if self.fail_hrtf_context and values and values[:2] == [openal_backend._ALC_HRTF_SOFT, 1]:
            return None
        return self._called("alcCreateContext", 0x20002)

    def _make_current(self, context):
        if self.fail_current_context:
            return 0
        self.calls.append("alcMakeContextCurrent")
        return 1

    def _set_thread_context(self, context):
        self.thread_context_calls.append(context)
        return 1

    def _get_proc_address(self, device, name):
        if name == b"alcSetThreadContext" and self.thread_context_supported:
            return self._thread_context_pointer
        return None

    def _destroy_context(self, context):
        self.calls.append("alcDestroyContext")
        self.destroyed_context = True

    def _has_extension(self, device, extension):
        if extension == b"ALC_SOFT_HRTF":
            return int(self.hrtf_supported)
        if extension == b"ALC_EXT_thread_local_context":
            return int(self.thread_context_supported)
        if extension == openal_backend._ALC_ENUMERATE_ALL_EXT:
            return 1
        if extension == openal_backend._ALC_ENUMERATION_EXT:
            return 1
        return 0

    def _pointer_to_string(self, value: bytes) -> int:
        buffer = ctypes.create_string_buffer(value + b"\0")
        self._strings[value] = buffer
        return ctypes.addressof(buffer)

    def _get_alc_string(self, device, token):
        if device is None and token == openal_backend._ALC_ALL_DEVICES_SPECIFIER:
            return ctypes.addressof(self._device_list)
        if token == openal_backend._ALC_ALL_DEVICES_SPECIFIER:
            if self.all_devices_name is None:
                return None
            return self._pointer_to_string(self.all_devices_name.encode("utf-8"))
        if token == openal_backend._ALC_DEVICE_SPECIFIER:
            if self.device_specifier_name is None:
                return None
            return self._pointer_to_string(self.device_specifier_name.encode("utf-8"))
        if token == openal_backend._ALC_HRTF_SPECIFIER_SOFT:
            return self._pointer_to_string(b"Mock HRTF dataset")
        return None

    def _set_integer(self, output, value: int) -> None:
        ctypes.cast(output, ctypes.POINTER(ctypes.c_int))[0] = value

    def _get_alc_integer(self, device, token, count, output):
        if token == openal_backend._ALC_HRTF_STATUS_SOFT:
            self._set_integer(output, self.hrtf_status)
        elif token == openal_backend._ALC_HRTF_SOFT:
            self._set_integer(output, int(self.hrtf_enabled))

    def _get_al_string(self, token):
        if token == openal_backend._AL_VERSION:
            return self._pointer_to_string(b"1.1 ALSOFT 1.25.2")
        if token == openal_backend._AL_RENDERER:
            return self._pointer_to_string(b"OpenAL Soft")
        return None

    def _get_al_error(self):
        error = self.al_error
        self.al_error = 0
        return error

    def _set_doppler(self, value):
        self.doppler_factor = float(value)

    def _write_uint(self, output, value: int):
        ctypes.cast(output, ctypes.POINTER(ctypes.c_uint))[0] = value

    def _gen_source(self, count, output):
        self._write_uint(output, 11)

    def _gen_buffer(self, count, output):
        self._write_uint(output, 22)

    def _stop_source(self, source):
        self.playback_source_state = 0
        self.calls.append("alSourceStop")

    def _play_source(self, source):
        self.playback_source_state = openal_backend._AL_PLAYING
        self.calls.append("alSourcePlay")

    def _set_source_float(self, source, token, value):
        if token == openal_backend._AL_GAIN:
            self.source_gain = float(value)
        elif token == openal_backend._AL_ROLLOFF_FACTOR:
            self.rolloff = float(value)

    def _set_source_position(self, source, token, x, y, z):
        self.source_position = (float(x), float(y), float(z))

    def _get_source_state(self, source, token, output):
        self._set_integer(output, self.playback_source_state)

    def _buffer_data(self, buffer, audio_format, data, size, frequency):
        self.buffer_bytes = ctypes.string_at(data, size)
        self.buffer_frequency = int(frequency)


def _backend(monkeypatch, library: _FakeOpenALLibrary) -> OpenALBackend:
    monkeypatch.setattr(openal_backend.ctypes, "CDLL", lambda path: library)
    monkeypatch.setattr(openal_backend.Path, "is_file", lambda self: False)
    return OpenALBackend("fake-openal.dll")


def test_open_requests_hrtf_and_reports_actual_dataset(monkeypatch):
    library = _FakeOpenALLibrary()
    backend = _backend(monkeypatch, library)

    status = backend.open()

    assert library.context_attributes == [[openal_backend._ALC_HRTF_SOFT, 1, 0]]
    assert status.backend == "OpenAL Soft 1.1 ALSOFT 1.25.2"
    assert status.device == "Actual USB Headphones"
    assert status.hrtf_supported is True
    assert status.hrtf_enabled is True
    assert status.hrtf_status == "enabled"
    assert status.hrtf_dataset == "Mock HRTF dataset"
    assert status.fallback_reason is None
    assert library.doppler_factor == 0.0

    backend.close()
    assert library.destroyed_context is True
    assert library.closed_device is True


def test_device_name_falls_back_to_original_specifier_then_explicit_name(monkeypatch):
    fallback_library = _FakeOpenALLibrary(all_devices_name=None)
    fallback_backend = _backend(monkeypatch, fallback_library)
    assert fallback_backend.open().device == "OpenAL Soft"
    fallback_backend.close()

    explicit_library = _FakeOpenALLibrary(
        all_devices_name=None,
        device_specifier_name=None,
    )
    monkeypatch.setattr(openal_backend.ctypes, "CDLL", lambda path: explicit_library)
    explicit_backend = OpenALBackend("fake-openal.dll", "Requested Headphones")
    assert explicit_backend.open().device == "Requested Headphones"
    explicit_backend.close()


def test_missing_hrtf_uses_normal_context_and_explains_stereo_fallback(monkeypatch):
    library = _FakeOpenALLibrary(hrtf_supported=False)
    backend = _backend(monkeypatch, library)

    status = backend.open()

    assert library.context_attributes == [None]
    assert status.hrtf_supported is False
    assert status.hrtf_enabled is False
    assert status.hrtf_status == "unsupported"
    assert status.hrtf_dataset is None
    assert "normal stereo" in status.fallback_reason
    backend.close()


def test_hrtf_context_failure_retries_plain_context(monkeypatch):
    library = _FakeOpenALLibrary(
        fail_hrtf_context=True,
        hrtf_enabled=False,
        hrtf_status=openal_backend._ALC_HRTF_DISABLED_SOFT,
    )
    backend = _backend(monkeypatch, library)

    status = backend.open()

    assert library.context_attributes == [
        [openal_backend._ALC_HRTF_SOFT, 1, 0],
        None,
    ]
    assert status.hrtf_supported is True
    assert status.hrtf_enabled is False
    assert "context creation failed" in status.fallback_reason
    backend.close()


def test_play_normalizes_direction_and_uses_requested_gain(monkeypatch):
    library = _FakeOpenALLibrary()
    backend = _backend(monkeypatch, library)
    backend.open()
    samples = np.array([0, 1234, -1234, 0], dtype=np.int16)

    backend.play(samples, 48_000, (3.0, 4.0, 0.0), 0.4)

    assert library.source_position == pytest.approx((0.6, 0.8, 0.0))
    assert library.source_gain == pytest.approx(0.4)
    assert library.rolloff == 0.0
    assert library.buffer_bytes == samples.tobytes()
    assert library.buffer_frequency == 48_000
    assert backend.is_playing() is True
    backend.stop()
    assert backend.is_playing() is False
    backend.close()


def test_open_failure_closes_partially_opened_device(monkeypatch):
    library = _FakeOpenALLibrary(fail_current_context=True)
    backend = _backend(monkeypatch, library)

    with pytest.raises(RuntimeError, match="setting current OpenAL context failed"):
        backend.open()

    assert library.destroyed_context is True
    assert library.closed_device is True


def test_backend_rejects_use_from_a_different_thread(monkeypatch):
    library = _FakeOpenALLibrary()
    backend = _backend(monkeypatch, library)
    backend.open()
    owner = backend._owner_thread_id
    backend._owner_thread_id = owner + 1

    with pytest.raises(RuntimeError, match="same thread"):
        backend.is_playing()

    backend._owner_thread_id = owner
    backend.close()


def test_thread_local_context_avoids_the_process_wide_context_lock(monkeypatch):
    library = _FakeOpenALLibrary(thread_context_supported=True)
    backend = _backend(monkeypatch, library)

    backend.open()
    assert library.thread_context_calls == [0x20002]
    assert openal_backend._GLOBAL_CONTEXT_LOCK.locked() is False

    backend.close()
    assert library.thread_context_calls == [0x20002, None]


def test_global_context_lock_prevents_two_backends_from_switching_contexts(monkeypatch):
    library = _FakeOpenALLibrary(thread_context_supported=False)
    first = _backend(monkeypatch, library)
    second = OpenALBackend("fake-openal.dll")

    first.open()
    with pytest.raises(RuntimeError, match="process-wide context"):
        second.open()

    assert openal_backend._GLOBAL_CONTEXT_LOCK.locked() is True
    first.close()
    assert openal_backend._GLOBAL_CONTEXT_LOCK.locked() is False


def test_enumerate_devices_reads_the_whole_multistring(monkeypatch):
    library = _FakeOpenALLibrary(devices=("Headphones", "Speakers", "USB DAC"))
    monkeypatch.setattr(openal_backend.ctypes, "CDLL", lambda path: library)
    monkeypatch.setattr(openal_backend.Path, "is_file", lambda self: False)

    assert enumerate_devices("fake-openal.dll") == ["Headphones", "Speakers", "USB DAC"]


@pytest.mark.parametrize(
    "cue_type",
    ["risk_initial", "risk_escalation", "perception_limited", "navigation", "system_test"],
)
def test_synthesize_returns_short_low_level_mono_pcm(cue_type):
    samples = synthesize(cue_type)

    assert isinstance(samples, np.ndarray)
    assert samples.ndim == 1
    assert samples.dtype == np.int16
    assert samples.size > 0
    assert np.max(np.abs(samples.astype(np.int32))) <= round(0.18 * np.iinfo(np.int16).max)


def test_synthesize_rejects_unknown_cue_and_invalid_sample_rate():
    with pytest.raises(ValueError, match="unknown cue_type"):
        synthesize("unknown")
    with pytest.raises(ValueError, match="positive integer"):
        synthesize("risk_initial", 0)
