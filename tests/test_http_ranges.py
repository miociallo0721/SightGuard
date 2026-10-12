"""Network-free random-access, identity, budget and corruption checks."""

from datetime import UTC, datetime, timedelta
from io import BytesIO
from urllib.error import URLError

import pytest

from sightguard.research.http_ranges import HttpRangeFile


def reader(tmp_path, monkeypatch, **kwargs):
    data = bytes(range(100))
    instance = HttpRangeFile(
        "https://example.invalid/public.h5",
        len(data),
        '"version-1"',
        tmp_path,
        stop_utc=datetime.now(UTC) + timedelta(minutes=1),
        block_bytes=8,
        **kwargs,
    )
    monkeypatch.setattr(instance, "_fetch", lambda start, end: data[start : end + 1])
    return instance, data


def test_unaligned_reads_seek_and_eof(tmp_path, monkeypatch):
    stream, data = reader(tmp_path, monkeypatch)
    assert stream.read(13) == data[:13]
    assert stream.seek(-3, 1) == 10
    assert stream.read(20) == data[10:30]
    assert stream.seek(-3, 2) == 97
    buffer = bytearray(8)
    assert stream.readinto(buffer) == 3
    assert buffer[:3] == data[-3:]
    assert stream.read() == b""
    stream.seek(101)
    assert stream.read(1) == b""


def test_cached_blocks_are_verified_and_do_not_refetch(tmp_path, monkeypatch):
    stream, data = reader(tmp_path, monkeypatch)
    assert stream.read(8) == data[:8]
    stream.seek(0)
    monkeypatch.setattr(stream, "_fetch", lambda *_: pytest.fail("unexpected network call"))
    assert stream.read(8) == data[:8]
    (stream.cache / "00000000.bin").write_bytes(b"corrupt!")
    stream.seek(0)
    with pytest.raises(ValueError, match="checksum"):
        stream.read(8)


def test_byte_budget_and_deadline(tmp_path, monkeypatch):
    stream, _ = reader(tmp_path, monkeypatch, maximum_bytes=8)
    stream.read(8)
    with pytest.raises(RuntimeError, match="budget"):
        stream.read(1)
    stream.seek(0)
    stream.stop_utc = datetime.now(UTC) - timedelta(seconds=1)
    with pytest.raises(TimeoutError):
        stream.read(1)


@pytest.mark.parametrize("offset,whence", [(-1, 0), (0, 7)])
def test_invalid_seek(tmp_path, monkeypatch, offset, whence):
    stream, _ = reader(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        stream.seek(offset, whence)


def test_source_identity_is_part_of_cache_path(tmp_path, monkeypatch):
    first, _ = reader(tmp_path, monkeypatch)
    second = HttpRangeFile(
        first.url,
        first.size,
        '"version-2"',
        tmp_path,
        stop_utc=first.stop_utc,
    )
    assert first.cache != second.cache


@pytest.mark.parametrize(
    "status,etag,extent",
    [
        (200, '"version-1"', "bytes 0-7/100"),
        (206, '"changed"', "bytes 0-7/100"),
        (206, '"version-1"', "bytes 0-8/100"),
    ],
)
def test_http_source_contract_rejects_ignored_or_changed_ranges(
    tmp_path, monkeypatch, status, etag, extent
):
    stream = HttpRangeFile(
        "https://example.invalid/data",
        100,
        '"version-1"',
        tmp_path,
        stop_utc=datetime.now(UTC) + timedelta(minutes=1),
        block_bytes=8,
    )
    response = BytesIO(bytes(range(8)))
    response.status = status
    response.headers = {"ETag": etag, "Content-Range": extent}
    monkeypatch.setattr("urllib.request.urlopen", lambda *_, **__: response)
    with pytest.raises(ValueError):
        stream.read(8)
    assert not list(stream.cache.glob("*.bin"))


def test_cache_can_be_reopened_without_network(tmp_path, monkeypatch):
    stream, data = reader(tmp_path, monkeypatch)
    assert stream.read(8) == data[:8]
    reopened = HttpRangeFile(
        stream.url, stream.size, stream.etag, tmp_path, stop_utc=stream.stop_utc, block_bytes=8
    )
    monkeypatch.setattr(reopened, "_fetch", lambda *_: pytest.fail("unexpected network call"))
    assert reopened.read(8) == data[:8]


def test_cached_different_block_geometry_is_rejected(tmp_path, monkeypatch):
    stream, _ = reader(tmp_path, monkeypatch)
    stream.read(8)
    changed = HttpRangeFile(
        stream.url, stream.size, stream.etag, tmp_path, stop_utc=stream.stop_utc, block_bytes=16
    )
    with pytest.raises(ValueError, match="extent"):
        changed.read(8)


def test_failed_retries_consume_conservative_network_budget(tmp_path, monkeypatch):
    stream = HttpRangeFile(
        "https://example.invalid/data",
        100,
        '"version-1"',
        tmp_path,
        stop_utc=datetime.now(UTC) + timedelta(minutes=1),
        block_bytes=8,
        maximum_bytes=16,
    )
    attempts = []

    def fail(*args, **kwargs):
        attempts.append(1)
        raise URLError("temporary connection failure")

    monkeypatch.setattr("urllib.request.urlopen", fail)
    with pytest.raises(RuntimeError, match="network byte budget"):
        stream.read(8)
    assert len(attempts) == 2
    assert stream.reserved_network_bytes == 16
    assert not list(stream.cache.glob("*.bin"))
