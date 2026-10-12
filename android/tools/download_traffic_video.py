"""Download and locally trim a CC BY-SA traffic clip for developer-time AVD validation.

This script is not used by the app or its runtime. Run it manually from the repository root
with the project's OpenCV environment; the downloaded source is kept only in a temporary
directory, while the short MP4 and its provenance record go under parity-validation/videos/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import cv2

SOURCE_PAGE = "https://commons.wikimedia.org/wiki/File:Traffic_control_in_Mexico_City.ogv"
SOURCE_REVISION = (
    "https://commons.wikimedia.org/w/index.php?"
    "title=File:Traffic_control_in_Mexico_City.ogv&oldid=982908504"
)
SOURCE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/f/f4/Traffic_control_in_Mexico_City.ogv"
)
LICENSE_NAME = "Creative Commons Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0)"
LICENSE_URL = "https://creativecommons.org/licenses/by-sa/3.0/"
ATTRIBUTION = "Uwebart, 'Traffic control in Mexico City', Wikimedia Commons"
SOURCE_SIZE_BYTES = 18_155_857
MAX_DOWNLOAD_BYTES = 20_000_000
START_MS = 20_000
CLIP_DURATION_MS = 20_000
OUTPUT_NAME = "traffic_control_mexico_city_20s.mp4"
MAX_OUTPUT_BYTES = 20_000_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_source(destination: Path) -> int:
    request = urllib.request.Request(
        SOURCE_URL,
        headers={"User-Agent": "SightGuard local validation fixture/1.0"},
    )
    downloaded = 0
    with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as output:
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > MAX_DOWNLOAD_BYTES:
            raise RuntimeError(
                f"Source is {content_length} bytes, above the "
                f"{MAX_DOWNLOAD_BYTES}-byte download limit"
            )
        while block := response.read(1024 * 1024):
            downloaded += len(block)
            if downloaded > MAX_DOWNLOAD_BYTES:
                raise RuntimeError(f"Download exceeded {MAX_DOWNLOAD_BYTES} bytes")
            output.write(block)
    if downloaded == 0:
        raise RuntimeError("The Wikimedia source download was empty")
    return downloaded


def trim_to_mp4(source: Path, output: Path) -> dict[str, int | float]:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError("OpenCV could not decode the downloaded Ogg/Theora source")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or width <= 0 or height <= 0:
            raise RuntimeError(f"Source has invalid video metadata: {width}x{height} at {fps} fps")

        first_frame = round(START_MS / 1000 * fps)
        frames_to_write = round(CLIP_DURATION_MS / 1000 * fps)
        if frame_count and first_frame + frames_to_write > frame_count:
            raise RuntimeError("The source is too short for the configured 20-second segment")

        writer = cv2.VideoWriter(
            str(output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError("OpenCV could not create the MP4 output")

        try:
            decoded = 0
            written = 0
            first_source_timestamp_ms: float | None = None
            while decoded < first_frame:
                ok, _ = capture.read()
                if not ok:
                    raise RuntimeError(f"Source ended while seeking to frame {first_frame}")
                decoded += 1

            while written < frames_to_write:
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(
                        f"Source ended after writing {written} of {frames_to_write} frames"
                    )
                if first_source_timestamp_ms is None:
                    first_source_timestamp_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
                writer.write(frame)
                written += 1
        finally:
            writer.release()

        return {
            "sourceWidth": width,
            "sourceHeight": height,
            "sourceFps": fps,
            "sourceFrameCount": frame_count,
            "segmentStartFrame": first_frame,
            "segmentFirstFrameTimestampMs": first_source_timestamp_ms or 0.0,
            "outputFrameCount": frames_to_write,
            "outputWidth": width,
            "outputHeight": height,
            "outputFps": fps,
            "outputDurationMs": round(frames_to_write / fps * 1000),
        }
    finally:
        capture.release()


def verify_output(path: Path, expected: dict[str, int | float]) -> None:
    if path.stat().st_size > MAX_OUTPUT_BYTES:
        raise RuntimeError(f"Trimmed MP4 is larger than {MAX_OUTPUT_BYTES} bytes")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError("OpenCV could not reopen the generated MP4")
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if (width, height) != (expected["outputWidth"], expected["outputHeight"]):
            raise RuntimeError(f"Output dimensions changed unexpectedly: {width}x{height}")
        if frame_count != expected["outputFrameCount"]:
            raise RuntimeError(
                f"Output has {frame_count} frames, expected {expected['outputFrameCount']}"
            )
        if abs(fps - float(expected["outputFps"])) > 0.01:
            raise RuntimeError(f"Output FPS changed unexpectedly: {fps}")
        if not capture.read()[0]:
            raise RuntimeError("Generated MP4 contains no decodable first frame")
    finally:
        capture.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "parity-validation" / "videos",
        help="Directory for the short video and its JSON provenance record",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output = output_dir / OUTPUT_NAME
    metadata_path = output.with_suffix(".metadata.json")
    if output.exists() or metadata_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing fixture: {output}")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        with tempfile.TemporaryDirectory(prefix="sightguard-commons-video-") as temporary_dir:
            source = Path(temporary_dir) / "Traffic_control_in_Mexico_City.ogv"
            downloaded_bytes = download_source(source)
            source_hash = sha256_file(source)
            media_info = trim_to_mp4(source, output)

        verify_output(output, media_info)
        metadata = {
            "schemaVersion": 1,
            "downloadedAtUtc": datetime.now(UTC).isoformat(),
            "source": {
                "title": "Traffic control in Mexico City.ogv",
                "description": "Police handling traffic, pedestrians, and drivers in Mexico City.",
                "author": "Uwebart",
                "attribution": ATTRIBUTION,
                "sourcePage": SOURCE_PAGE,
                "sourceRevisionPage": SOURCE_REVISION,
                "downloadUrl": SOURCE_URL,
                "license": LICENSE_NAME,
                "licenseUrl": LICENSE_URL,
                "originalContainerAndCodecs": "Ogg container, Theora video, Vorbis audio",
                "originalDurationMs": 112_800,
                "originalWidth": 480,
                "originalHeight": 360,
                "measuredFps": media_info["sourceFps"],
                "measuredFrameCount": media_info["sourceFrameCount"],
                "measuredDurationMs": round(
                    float(media_info["sourceFrameCount"]) / float(media_info["sourceFps"]) * 1000
                ),
                "originalSizeBytesFromCommons": SOURCE_SIZE_BYTES,
                "downloadedSizeBytes": downloaded_bytes,
                "sha256": source_hash,
            },
            "transformation": {
                "startMsRequested": START_MS,
                "durationMsRequested": CLIP_DURATION_MS,
                "outputFormat": "MP4, MPEG-4 Part 2 video (mp4v), no audio",
                "method": (
                    "OpenCV sequentially decodes and trims frames, then re-encodes without overlays"
                ),
                "changes": [
                    "Trimmed to a 20-second segment starting near 20 seconds.",
                    "Re-encoded from Ogg/Theora to MP4 MPEG-4 Part 2.",
                    "Audio omitted; no detections or annotations were added.",
                ],
                "opencvVersion": cv2.__version__,
            },
            "output": {
                "fileName": output.name,
                "sizeBytes": output.stat().st_size,
                "sha256": sha256_file(output),
                **media_info,
            },
            "limitations": [
                "This is a short excerpt from one 2007 public street recording, not a safety "
                "or collision-prediction validation set.",
                "Attribution and CC BY-SA 3.0 apply; preserve this record with any "
                "redistribution of the excerpt.",
            ],
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    except Exception:
        output.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
        raise

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
