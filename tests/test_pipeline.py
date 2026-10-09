import json
from pathlib import Path

import cv2
import pytest

from sightguard.cli import main
from sightguard.config import PerceptionConfig
from sightguard.data import Detection, replay_records
from sightguard.pipeline.video import VideoClock, VideoError, analyze_video


class MockPerceptor:
    def __init__(self, config):
        self.frame = 0

    def process(self, frame):
        self.frame += 1
        if self.frame == 2:
            return ()
        return (
            Detection(None if self.frame == 3 else self.frame, 0, "person", 0.8, (2, 3, 20, 30)),
        )


def test_clock_pts_and_fallback() -> None:
    clock = VideoClock(10)
    assert clock.timestamp(0, 0) == (0, "video_pts")
    assert clock.timestamp(1, 0) == (100, "fps_fallback")
    assert clock.timestamp(2, 250) == (250, "video_pts")
    assert clock.timestamp(3, float("nan")) == (350, "fps_fallback")
    assert clock.timestamp(4, 200) == (450, "fps_fallback")
    with pytest.raises(VideoError):
        VideoClock(0)


def test_pipeline_and_video_isolation(video: Path, tmp_path: Path) -> None:
    for run in ("one", "two"):
        output = tmp_path / run
        metadata = analyze_video(video, output, PerceptionConfig(), perceptor_factory=MockPerceptor)
        records = list(replay_records(output / "tracks.jsonl"))
        assert len(records) == 5
        assert records[0].detections[0].track_id == 1
        assert records[1].status == "ok" and records[1].detections == ()
        assert records[2].detections[0].track_id is None
        assert [r.frame_index for r in records] == list(range(5))
        assert [r.timestamp_ms for r in records] == pytest.approx([0, 100, 200, 300, 400])
        assert metadata["status"] == "completed"
        assert metadata["detections"] == 4
        assert metadata["missing_track_ids"] == 1
        capture = cv2.VideoCapture(str(output / "annotated.mp4"))
        count = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            assert frame.shape[:2] == (48, 64)
            count += 1
        capture.release()
        assert count == 5
        assert main(["replay", "--input", str(output / "tracks.jsonl")]) == 0
    with pytest.raises(FileExistsError):
        analyze_video(video, tmp_path / "one", PerceptionConfig(), perceptor_factory=MockPerceptor)


def test_invalid_input(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        analyze_video(tmp_path / "missing.mp4", tmp_path / "out", PerceptionConfig())
    invalid = tmp_path / "invalid.mp4"
    invalid.write_bytes(b"not a video")
    with pytest.raises(VideoError, match="Cannot open"):
        analyze_video(
            invalid, tmp_path / "out", PerceptionConfig(), perceptor_factory=MockPerceptor
        )
    metadata = json.loads((tmp_path / "out/run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "failed"
    assert main(["analyze", "--input", str(invalid), "--output-dir", str(tmp_path / "cli")]) == 1


def test_perception_failure_preserved(video: Path, tmp_path: Path) -> None:
    class Failing:
        def __init__(self, config):
            pass

        def process(self, frame):
            raise RuntimeError("simulated failure")

    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="simulated"):
        analyze_video(video, output, PerceptionConfig(), perceptor_factory=Failing)
    records = list(replay_records(output / "tracks.jsonl"))
    assert len(records) == 1
    assert records[0].status == "processing_error"
    assert records[0].error == "RuntimeError: simulated failure"
    metadata = json.loads((output / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "failed"


@pytest.mark.parametrize("empty", [False, True])
def test_decode_failure(monkeypatch, video: Path, tmp_path: Path, empty: bool) -> None:
    class BrokenCapture:
        def __init__(self, path):
            self.released = False

        def isOpened(self):
            return True

        def get(self, prop):
            return 10 if prop == cv2.CAP_PROP_FPS else 5

        def read(self):
            return (True, None) if empty else (False, None)

        def release(self):
            self.released = True

    monkeypatch.setattr(cv2, "VideoCapture", BrokenCapture)
    with pytest.raises(VideoError, match="empty frame|read failed"):
        analyze_video(
            video, tmp_path / "broken", PerceptionConfig(), perceptor_factory=MockPerceptor
        )


def test_midstream_decode_failure(monkeypatch, video: Path, tmp_path: Path) -> None:
    original_capture = cv2.VideoCapture

    class TruncatedCapture:
        def __init__(self, path):
            self.capture = original_capture(path)
            self.calls = 0

        def isOpened(self):
            return self.capture.isOpened()

        def get(self, prop):
            return self.capture.get(prop)

        def read(self):
            self.calls += 1
            return self.capture.read() if self.calls <= 2 else (False, None)

        def release(self):
            self.capture.release()

    monkeypatch.setattr(cv2, "VideoCapture", TruncatedCapture)
    output = tmp_path / "truncated"
    with pytest.raises(VideoError, match="frame 2"):
        analyze_video(video, output, PerceptionConfig(), perceptor_factory=MockPerceptor)
    assert len(list(replay_records(output / "tracks.jsonl"))) == 2
    metadata = json.loads((output / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "failed"


def test_encoder_failure(monkeypatch, video: Path, tmp_path: Path) -> None:
    class BrokenWriter:
        def __init__(self, *args):
            pass

        def isOpened(self):
            return False

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoWriter", BrokenWriter)
    output = tmp_path / "encoder_failure"
    with pytest.raises(VideoError, match="encoder"):
        analyze_video(video, output, PerceptionConfig(), perceptor_factory=MockPerceptor)
    metadata = json.loads((output / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "failed"
