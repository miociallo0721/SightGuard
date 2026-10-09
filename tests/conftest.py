from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture
def video(tmp_path: Path) -> Path:
    path = tmp_path / "input.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    assert writer.isOpened(), "MP4 encoder required for the I/O tests"
    for index in range(5):
        writer.write(np.full((48, 64, 3), index * 20, dtype=np.uint8))
    writer.release()
    return path
