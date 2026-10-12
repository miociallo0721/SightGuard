"""Bounded byte-range cache tests use no network or model dependencies."""

import importlib
import io
from pathlib import Path

import pytest


@pytest.fixture
def archive_fixture(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "android/tools"))
    module = importlib.import_module("fetch_furniture_validation")
    payload = bytes(range(256)) * 2500
    requests = []

    class Response(io.BytesIO):
        status = 206

        def __init__(self, data, headers):
            super().__init__(data)
            self.headers = headers

    def fake_open(request):
        if request.get_method() == "HEAD":
            return Response(b"", {"Content-Length": str(len(payload)), "ETag": "fixture-v1"})
        start, end = map(int, request.headers["Range"].removeprefix("bytes=").split("-"))
        requests.append((start, end))
        assert request.headers["If-range"] == "fixture-v1"
        return Response(
            payload[start : end + 1], {"Content-Range": f"bytes {start}-{end}/{len(payload)}"}
        )

    monkeypatch.setattr(module, "open_retry", fake_open)
    return module, payload, requests


def test_read_ahead_preserves_zip_offsets_and_reuses_blocks(archive_fixture):
    module, payload, requests = archive_fixture
    archive = module.RangeArchive()
    assert archive.read(30) == payload[:30]
    assert archive.read(400) == payload[30:430]
    assert len(requests) == 1
    archive.seek(262140)
    assert archive.read(30) == payload[262140:262170]
    archive.seek(-50, 2)
    assert archive.read(50) == payload[-50:]
    assert archive.read(10) == b""
    assert sum(end - start + 1 for start, end in requests) == archive.downloaded
    assert len(archive.blocks) <= 8


def test_full_archive_response_is_rejected_before_reading(archive_fixture, monkeypatch):
    module, _, _ = archive_fixture
    archive = module.RangeArchive()

    class FullResponse(io.BytesIO):
        status = 200
        headers = {}

        def read(self, size=-1):
            raise AssertionError("Bulk response must not be consumed")

    monkeypatch.setattr(module, "open_retry", lambda request: FullResponse())
    with pytest.raises(ValueError, match="honor byte range"):
        archive.read(100)
