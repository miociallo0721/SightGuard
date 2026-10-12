"""Bounded, source-bound HTTP ranges for lazy public HDF5 dataset reads."""

from __future__ import annotations

import hashlib
import io
import json
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path


class HttpRangeFile(io.RawIOBase):
    """Read exact ranges without downloading a complete multi-GB dataset.

    Cached blocks are bound to URL, byte length and the server's ETag. A local
    block digest is a reproducibility record, not a publisher checksum.
    """

    def __init__(
        self,
        url: str,
        size: int,
        etag: str,
        cache: Path,
        *,
        stop_utc: datetime,
        maximum_bytes: int = 128 * 1024**2,
        block_bytes: int = 256 * 1024,
    ) -> None:
        super().__init__()
        if not url.startswith("https://") or size <= 0 or not etag:
            raise ValueError("HTTPS source, positive length and ETag are required")
        if stop_utc.utcoffset() is None or maximum_bytes <= 0 or block_bytes <= 0:
            raise ValueError("explicit deadline and positive budgets are required")
        self.url, self.size, self.etag = url, size, etag
        self.stop_utc = stop_utc
        self.maximum_bytes, self.block_bytes = maximum_bytes, block_bytes
        identity = json.dumps([url, size, etag]).encode()
        self.cache = cache / hashlib.sha256(identity).hexdigest()[:20]
        self.cache.mkdir(parents=True, exist_ok=True)
        self.position = 0
        self.downloaded_bytes = 0
        self.reserved_network_bytes = 0
        self.records: dict[str, dict] = {}
        self.manifest = self.cache / "ranges.json"
        if self.manifest.exists():
            old = json.loads(self.manifest.read_text(encoding="utf-8"))
            if (old["url"], old["size"], old["etag"]) != (url, size, etag):
                raise ValueError("cached source identity mismatch")
            self.records = old["blocks"]

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self._checkClosed()
        if whence not in (io.SEEK_SET, io.SEEK_CUR, io.SEEK_END):
            raise ValueError("invalid seek origin")
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[whence]
        if base + offset < 0:
            raise ValueError("negative seek")
        self.position = base + offset
        return self.position

    def _fetch(self, start: int, end: int) -> bytes:
        request = urllib.request.Request(
            self.url,
            headers={"Range": f"bytes={start}-{end}", "If-Match": self.etag},
        )
        for attempt in range(3):
            if datetime.now(UTC) >= self.stop_utc:
                raise TimeoutError("range-reader deadline reached")
            # Reserve the whole requested response, including failed retries.
            # This is a conservative upper bound on transferred response bytes.
            if self.reserved_network_bytes + end - start + 1 > self.maximum_bytes:
                raise RuntimeError("range-reader network byte budget reached")
            self.reserved_network_bytes += end - start + 1
            try:
                with urllib.request.urlopen(request, timeout=12) as response:
                    if response.status != 206:
                        raise ValueError("server did not honor exact byte ranges")
                    if response.headers.get("Content-Range") != f"bytes {start}-{end}/{self.size}":
                        raise ValueError("range or total length mismatch")
                    if response.headers.get("ETag") != self.etag:
                        raise ValueError("source ETag changed")
                    data = response.read(end - start + 2)
                break
            except (urllib.error.URLError, TimeoutError):
                if attempt == 2:
                    raise
        if len(data) != end - start + 1:
            raise ValueError("truncated or oversized range")
        return data

    def _block(self, index: int) -> bytes:
        if datetime.now(UTC) >= self.stop_utc:
            raise TimeoutError("range-reader deadline reached")
        path = self.cache / f"{index:08d}.bin"
        if path.exists():
            data = path.read_bytes()
            record = self.records.get(str(index))
            if record is None or hashlib.sha256(data).hexdigest() != record["sha256"]:
                raise ValueError("cached range checksum mismatch")
            start = index * self.block_bytes
            end = min(start + self.block_bytes, self.size) - 1
            if record["start"] != start or record["end"] != end or len(data) != end - start + 1:
                raise ValueError("cached range extent mismatch; block geometry changed")
            return data
        start = index * self.block_bytes
        end = min(start + self.block_bytes, self.size) - 1
        if self.downloaded_bytes + end - start + 1 > self.maximum_bytes:
            raise RuntimeError("range-reader byte budget reached")
        data = self._fetch(start, end)
        self.downloaded_bytes += len(data)
        self.records[str(index)] = {
            "start": start,
            "end": end,
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        path.write_bytes(data)
        self.manifest.write_text(
            json.dumps(
                {"url": self.url, "size": self.size, "etag": self.etag, "blocks": self.records},
                indent=2,
            ),
            encoding="utf-8",
        )
        return data

    def read(self, size: int = -1) -> bytes:
        self._checkClosed()
        count = min(self.size - self.position, size if size >= 0 else self.size)
        if count <= 0:
            return b""
        if count > self.maximum_bytes:
            raise RuntimeError("single read exceeds byte budget")
        pieces = []
        remaining = count
        while remaining:
            index, offset = divmod(self.position, self.block_bytes)
            block = self._block(index)
            take = min(remaining, len(block) - offset)
            if take <= 0:
                raise ValueError("invalid cached block extent")
            pieces.append(block[offset : offset + take])
            self.position += take
            remaining -= take
        return b"".join(pieces)

    def readinto(self, buffer: bytearray) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)
