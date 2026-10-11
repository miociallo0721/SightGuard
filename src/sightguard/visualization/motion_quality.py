"""Real RGB previews and standalone read-only temporal evidence viewer."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import cv2

from sightguard.scene.motion.schema import MotionFrame


def save_sample(
    directory: Path,
    frame: MotionFrame,
    primary: dict[str, Any],
    control: dict[str, Any],
    spatial: dict[str, Any],
    temporal: list[dict[str, Any]],
) -> dict[str, Any]:
    previews = directory / "previews"
    previews.mkdir(exist_ok=True)
    filename = f"previews/{frame.key.frame_index:06d}.jpg"
    if not cv2.imwrite(str(directory / filename), frame.image, [cv2.IMWRITE_JPEG_QUALITY, 88]):
        raise OSError("RGB preview write failed")
    margin = primary["config"]["mask_margin_pixels"]
    boxes = (
        [
            [
                max(0, d.bbox_xyxy[0] - margin),
                max(0, d.bbox_xyxy[1] - margin),
                min(frame.key.width, d.bbox_xyxy[2] + margin),
                min(frame.key.height, d.bbox_xyxy[3] + margin),
            ]
            for d in frame.tracks.detections
            if d.confidence >= primary["config"]["detection_confidence"]
        ]
        if frame.tracks
        else []
    )
    return {
        "image": filename,
        "primary": primary,
        "control": control,
        "spatial": spatial,
        "temporal": temporal,
        "exclusion_boxes": boxes,
    }


def write_viewer(directory: Path, samples: list[dict[str, Any]]) -> None:
    template = Path(__file__).with_suffix(".html").read_text(encoding="utf-8")
    payload = json.dumps(samples, allow_nan=False).replace("<", "\\u003c")
    (directory / "index.html").write_text(
        template.replace("__DATA__", payload).replace("__VIDEO__", html.escape(directory.name)),
        encoding="utf-8",
    )
    root = directory.parent
    videos = sorted(
        path.name for path in root.glob("urbanego_run8_*") if (path / "index.html").exists()
    )
    links = "".join(
        f'<li><a href="{html.escape(name)}/index.html">{html.escape(name)}</a></li>'
        for name in videos
    )
    (root / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Motion Quality v1</title>'
        "<h1>Stage 4A — real video research</h1>"
        f"<p>Read only · absolute pose UNKNOWN · no navigation</p><ul>{links}</ul>",
        encoding="utf-8",
    )
