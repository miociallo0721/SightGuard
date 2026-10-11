"""Bounded resource checks for unattended experiments, without killing other apps."""

from __future__ import annotations

import ctypes
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path


class ResourceLimit(RuntimeError):
    """A research task must stop or defer without changing formal perception."""


def free_memory_mb() -> float:
    if os.name == "nt":

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("load", ctypes.c_ulong),
                ("total_phys", ctypes.c_ulonglong),
                ("avail_phys", ctypes.c_ulonglong),
                ("total_page", ctypes.c_ulonglong),
                ("avail_page", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("avail_virtual", ctypes.c_ulonglong),
                ("avail_extended", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise ResourceLimit("memory_status_unavailable")
        return status.avail_phys / 1024**2
    with Path("/proc/meminfo").open() as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024
    raise ResourceLimit("memory_status_unavailable")


def check_resources(
    *,
    minimum_mb: float = 1536,
    stop_utc: str | None = None,
    output_dir: Path | None = None,
) -> dict[str, float | str]:
    now = datetime.now(UTC)
    deadline = datetime.fromisoformat(
        stop_utc or os.getenv("SIGHTGUARD_RESEARCH_STOP_UTC", "2026-10-11T22:20:00+00:00")
    )
    if deadline.utcoffset() is None:
        raise ValueError("research deadline must have an explicit UTC offset")
    if now >= deadline:
        raise ResourceLimit("absolute_deadline_reached")
    available = free_memory_mb()
    if available < minimum_mb:
        raise ResourceLimit(f"low_memory:{available:.0f}MB<{minimum_mb:.0f}MB")
    if output_dir is not None and shutil.disk_usage(output_dir).free < 5 * 1024**3:
        raise ResourceLimit("disk_reserve_below_5GB")
    return {"utc": now.isoformat(), "available_ram_mb": available}


def lower_own_priority() -> None:
    """Only adjust the calling research process, never unrelated user processes."""
    if os.name == "nt":
        from ctypes import wintypes

        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        kernel.SetPriorityClass.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        if not kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x00004000):
            raise ResourceLimit("cannot_lower_research_process_priority")
    else:
        os.nice(5)
