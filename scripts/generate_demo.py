"""Generate a synthetic 2-second clip for I/O tests, not model quality evaluation."""

from pathlib import Path

import cv2
import numpy as np


def generate_video(path: Path, frames: int = 20, fps: float = 10.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 240))
    if not writer.isOpened():
        raise RuntimeError("MP4 codec unavailable")
    try:
        for index in range(frames):
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.rectangle(frame, (10 + index * 5, 80), (50 + index * 5, 160), (100, 200, 255), -1)
            writer.write(frame)
    finally:
        writer.release()


if __name__ == "__main__":
    generate_video(Path("examples/demo.mp4"))
    print("Created examples/demo.mp4 (synthetic I/O fixture)")
