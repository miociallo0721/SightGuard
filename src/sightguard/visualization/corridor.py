"""Image-band evidence viewer; no arrows, suggested motion or green safe paths."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from sightguard.scene.spatial_grid.schema import SpatialGridFrame

COLORS = ((230, 190, 70), (80, 170, 240), (215, 140, 215))


def save_preview(
    image: np.ndarray, frame: SpatialGridFrame, assessment: dict[str, Any], path: Path
) -> None:
    if not cv2.imwrite(str(path.with_name(path.stem + "_rgb.jpg")), image):
        raise OSError("RGB preview save failed")
    canvas = image.copy()
    for cell in frame.cells:
        x1, y1, x2, y2 = cell.bounds_xyxy
        cv2.rectangle(canvas, (x1, y1), (x2 - 1, y2 - 1), (130, 130, 130), 1)
        if cell.ground.status == "unknown":
            cv2.putText(
                canvas, "?", (x1 + 3, y1 + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1
            )
    for index, candidate in enumerate(assessment["candidates"]):
        color = COLORS[index % len(COLORS)]
        polygon = np.asarray(candidate["candidate"]["polygon_xy"], np.int32)
        cv2.polylines(canvas, [polygon], True, color, 2)
        x, y = polygon[0]
        cv2.putText(
            canvas,
            candidate["candidate"]["candidate_id"],
            (int(x), int(y) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )
    for obj in frame.object_relations:
        if obj.clipped_bbox_xyxy:
            x1, y1, x2, y2 = map(round, obj.clipped_bbox_xyxy)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (230, 230, 230), 1)
            cv2.putText(
                canvas,
                f"bbox track={obj.track_id}",
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )
    cv2.rectangle(canvas, (0, 0), (frame.key.width, 36), (25, 25, 25), -1)
    cv2.putText(
        canvas,
        (
            f"{frame.key.video_id} frame={frame.key.frame_index} "
            f"PTS={frame.key.timestamp_ms:.0f}ms D:{assessment['status']} | IMAGE EVIDENCE ONLY"
        ),
        (8, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (235, 235, 235),
        1,
    )
    if not cv2.imwrite(str(path), canvas):
        raise OSError("evidence preview save failed")


def write_viewer(directory: Path, rows: list[dict[str, Any]], previews: bool) -> None:
    encoded = (
        json.dumps(rows, ensure_ascii=False, allow_nan=False)
        .replace("<", "\\u003c")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    (directory / "evidence.js").write_text(
        "window.CORRIDOR_DATA=" + encoded + ";\n", encoding="utf-8"
    )
    html = Path(__file__).with_name("corridor_viewer.html").read_text(encoding="utf-8")

    if not previews:
        html = html.replace('<img id="image"', '<img hidden id="image"')
    (directory / "index.html").write_text(html, encoding="utf-8")
