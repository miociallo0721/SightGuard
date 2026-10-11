import pytest

from sightguard.research import resources


def test_resource_gate_does_not_continue_with_low_ram(monkeypatch):
    monkeypatch.setattr(resources, "free_memory_mb", lambda: 900)
    with pytest.raises(resources.ResourceLimit, match="low_memory"):
        resources.check_resources(stop_utc="2100-01-01T00:00:00+00:00")


def test_absolute_utc_deadline_is_checked_first(monkeypatch):
    monkeypatch.setattr(resources, "free_memory_mb", lambda: 16000)
    with pytest.raises(resources.ResourceLimit, match="absolute_deadline"):
        resources.check_resources(stop_utc="2020-01-01T00:00:00+00:00")


def test_deadline_cannot_silently_use_machine_local_timezone():
    with pytest.raises(ValueError, match="explicit UTC offset"):
        resources.check_resources(stop_utc="2100-01-01T00:00:00")
