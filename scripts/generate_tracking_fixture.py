"""Create a tracking smoke fixture from Ultralytics' packaged bus image.

Repeated still frames validate IDs and integration, not real motion accuracy.
The input remains local, with no network request and no extra model inference.
"""

import argparse
import json
from pathlib import Path

import cv2
from ultralytics.utils import ASSETS

from sightguard.pipeline.video import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("examples/tracking_fixture.mp4"))
    args = parser.parse_args()
    source = ASSETS / "bus.jpg"
    image = cv2.imread(str(source))
    if image is None:
        raise RuntimeError(f"Packaged sample image unavailable: {source}")
    image = cv2.resize(image, (480, 640))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), 10, (480, 640))
    if not writer.isOpened():
        raise RuntimeError("MP4 encoder unavailable")
    try:
        for _ in range(10):
            writer.write(image)
    finally:
        writer.release()
    provenance = {
        "source": "Ultralytics packaged assets/bus.jpg",
        "source_sha256": sha256_file(source),
        "video_sha256": sha256_file(args.output),
        "generation": "resize to 480x640, repeat 10 frames at 10 FPS, mp4v",
        "limitation": "Repeated still image, not a tracking accuracy benchmark",
    }
    args.output.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(f"Created {args.output}")


if __name__ == "__main__":
    main()
