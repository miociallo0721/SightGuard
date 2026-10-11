"""Generate local MP4 fixtures for Android's VideoFileSource instrumentation tests.

The clips are transformed derivatives of Ultralytics' locally packaged bus.jpg. They are
decoder/integration fixtures, not moving-scene footage or model-accuracy benchmarks. The
generated MP4s are gitignored; run this script before packaging androidTest assets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

import cv2
import numpy as np

FPS = 12
FRAME_COUNT = 36
LANDSCAPE_SIZE = (640, 360)
PORTRAIT_SIZE = (360, 640)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def packaged_bus_photo() -> Path:
    try:
        from ultralytics.utils import ASSETS
    except ImportError as exc:
        raise RuntimeError("Install the project environment or pass --source bus.jpg") from exc
    return Path(ASSETS) / "bus.jpg"


def render_frame(image: np.ndarray, size: tuple[int, int], index: int) -> np.ndarray:
    width, height = size
    source_height, source_width = image.shape[:2]

    # Fill the frame with a darkened, blurred crop so the full source photo can stay visible.
    fill_scale = max(width / source_width, height / source_height)
    fill = cv2.resize(
        image,
        (round(source_width * fill_scale), round(source_height * fill_scale)),
        interpolation=cv2.INTER_LINEAR,
    )
    left = (fill.shape[1] - width) // 2
    top = (fill.shape[0] - height) // 2
    background = fill[top : top + height, left : left + width]
    background = cv2.GaussianBlur(background, (41, 41), 0)
    background = cv2.convertScaleAbs(background, alpha=0.38, beta=12)
    frame = background.copy()

    progress = index / max(1, FRAME_COUNT - 1)
    zoom = 0.91 + 0.07 * progress
    fit_scale = min(width * 0.86 / source_width, height * 0.86 / source_height) * zoom
    foreground = cv2.resize(
        image,
        (max(1, round(source_width * fit_scale)), max(1, round(source_height * fit_scale))),
        interpolation=cv2.INTER_AREA,
    )
    sway_x = round(math.sin(progress * math.pi * 2) * width * 0.025)
    sway_y = round(math.cos(progress * math.pi * 2) * height * 0.018)
    x = (width - foreground.shape[1]) // 2 + sway_x
    y = (height - foreground.shape[0]) // 2 + sway_y
    x = max(0, min(width - foreground.shape[1], x))
    y = max(0, min(height - foreground.shape[0], y))
    frame[y : y + foreground.shape[0], x : x + foreground.shape[1]] = foreground
    return frame


def write_video(
    path: Path,
    source_image: np.ndarray,
    size: tuple[int, int],
    rotate_for_track: bool,
) -> None:
    width, height = size
    encoded_size = (height, width) if rotate_for_track else size
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, encoded_size
    )
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not open an MP4 writer for {path}")
    try:
        for index in range(FRAME_COUNT):
            frame = render_frame(source_image, size, index)
            if rotate_for_track:
                # Encode the intended upright portrait frame sideways; the MP4 track matrix
                # below describes the clockwise correction expected from Android's retriever.
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            writer.write(frame)
    finally:
        writer.release()


def _boxes(data: bytearray, start: int, end: int):
    offset = start
    while offset + 8 <= end:
        size = struct.unpack_from(">I", data, offset)[0]
        kind = bytes(data[offset + 4 : offset + 8])
        header_size = 8
        if size == 1:
            if offset + 16 > end:
                break
            size = struct.unpack_from(">Q", data, offset + 8)[0]
            header_size = 16
        elif size == 0:
            size = end - offset
        if size < header_size or offset + size > end:
            break
        yield kind, offset, header_size, size
        offset += size


def _find_child(data: bytearray, parent: tuple[int, int, int, int], child_type: bytes):
    _, offset, header_size, size = parent
    for child in _boxes(data, offset + header_size, offset + size):
        if child[0] == child_type:
            return child
    return None


def add_clockwise_rotation_metadata(path: Path, raw_height: int) -> None:
    data = bytearray(path.read_bytes())
    top_level = list(_boxes(data, 0, len(data)))
    moov = next((box for box in top_level if box[0] == b"moov"), None)
    if moov is None:
        raise RuntimeError(f"MP4 has no moov atom: {path}")
    track_boxes = _boxes(data, moov[1] + moov[2], moov[1] + moov[3])
    for track in (box for box in track_boxes if box[0] == b"trak"):
        header = _find_child(data, track, b"tkhd")
        if header is None:
            continue
        version = data[header[1] + header[2]]
        if version == 0:
            matrix_offset = header[1] + header[2] + 40
        elif version == 1:
            matrix_offset = header[1] + header[2] + 52
        else:
            raise RuntimeError(f"Unsupported tkhd version {version} in {path}")

        # ISO BMFF track matrix for a clockwise quarter turn with y-down video coordinates.
        matrix = (0, 1 << 16, 0, -(1 << 16), 0, 0, raw_height << 16, 0, 1 << 30)
        struct.pack_into(">9i", data, matrix_offset, *matrix)
        path.write_bytes(data)
        return
    raise RuntimeError(f"MP4 has no video track header: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        help="Local bus photograph; defaults to Ultralytics assets/bus.jpg",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("android/app/src/androidTest/assets/video-fixtures"),
        help="Output directory (the generated .mp4 files are gitignored)",
    )
    args = parser.parse_args()

    source = (args.source or packaged_bus_photo()).resolve()
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read local source photo: {source}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    clips = [
        ("bus_landscape.mp4", LANDSCAPE_SIZE, False),
        ("bus_portrait.mp4", PORTRAIT_SIZE, False),
        ("bus_rotation_90.mp4", PORTRAIT_SIZE, True),
    ]
    outputs = []
    for filename, output_size, rotated_track in clips:
        path = args.output_dir / filename
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite generated fixture: {path}")
        encoded_size = (output_size[1], output_size[0]) if rotated_track else output_size
        write_video(path, image, output_size, rotated_track)
        if rotated_track:
            add_clockwise_rotation_metadata(path, raw_height=encoded_size[1])
        outputs.append(
            {
                "path": str(path),
                "sha256": sha256(path),
                "encoded_size": list(encoded_size),
                "display_size": list(output_size),
                "rotation_degrees": 90 if rotated_track else 0,
            }
        )

    print(
        json.dumps(
            {
                "source": str(source),
                "source_sha256": sha256(source),
                "opencv_version": cv2.__version__,
                "fps": FPS,
                "frame_count": FRAME_COUNT,
                "generation": (
                    "blurred background, contained bus photograph with slight deterministic "
                    "pan/zoom"
                ),
                "limitations": [
                    "Synthetic derivative of one still photograph; no real camera motion or "
                    "moving objects.",
                    "For decoder/orientation checks and optional real model inference, not a "
                    "performance benchmark.",
                ],
                "outputs": outputs,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
