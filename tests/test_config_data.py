import json
from pathlib import Path

import pytest

from sightguard.config import PerceptionConfig, load_config
from sightguard.data import Detection, FrameRecord, replay_records


def test_config_load(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("class_ids: [0, 1, 3]\nconfidence: 0.5\ndevice: cpu\n", encoding="utf-8")
    assert load_config(config).class_ids == (0, 1, 3)
    assert load_config(config).confidence == 0.5
    assert load_config(Path("configs/default.yaml")) == PerceptionConfig()


@pytest.mark.parametrize(
    "text",
    [
        "[]",
        "unknown: 1",
        "confidence: 2",
        "confidence: .nan",
        "class_ids: []",
        "class_ids: [true]",
        "image_size: 0",
        "device: 0",
        "schema_version: '9'",
        "tracker: bytetrack.yaml",
    ],
)
def test_invalid_config(tmp_path: Path, text: str) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_serialization_and_replay(tmp_path: Path) -> None:
    detection = Detection(None, 0, "person", 0.7, (1, 2, 10, 20))
    record = FrameRecord(0, 0, 64, 48, (detection,))
    assert FrameRecord.from_json(record.to_json()) == record
    path = tmp_path / "tracks.jsonl"
    empty = FrameRecord(1, 100, 64, 48)
    failed = FrameRecord(2, 200, 64, 48, status="processing_error", error="inference error")
    path.write_text(
        "\n".join(item.to_json() for item in (record, empty, failed)) + "\n", encoding="utf-8"
    )
    assert list(replay_records(path)) == [record, empty, failed]
    assert json.loads(record.to_json())["detections"][0]["track_id"] is None


def test_invalid_record_and_replay(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        FrameRecord(0, 0, 64, 48, status="processing_error")
    with pytest.raises(ValueError):
        Detection(1, 0, "person", 0.5, (10, 10, 0, 0))
    with pytest.raises(ValueError):
        FrameRecord.from_json(FrameRecord(0, 0, 64, 48).to_json().replace('"1.0"', '"9"'))
    path = tmp_path / "bad.jsonl"
    path.write_text(FrameRecord(1, 100, 64, 48).to_json(), encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        list(replay_records(path))
