"""Blind annotation storage, exact raw PTS, and explicit human review transactions."""

from __future__ import annotations

import bisect
import copy
import csv
import hashlib
import io
import json
import math
import os
import tempfile
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2

from sightguard.evaluation.dataset import (
    MANIFEST_FIELDS,
    _validate_overlaps,
    load_dataset,
    read_manifest,
    sha256_file,
)
from sightguard.evaluation.schema import load_annotations


class AnnotationStore:
    """Browser supplies dataset/video IDs only, never writable filesystem paths."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.datasets = self.root / "datasets"
        self._lock = threading.RLock()
        self._pts: dict[str, tuple[list[float], int, int]] = {}

    def safe(self, path: Path) -> Path:
        resolved = path.resolve()
        if not resolved.is_relative_to(self.datasets.resolve()):
            raise ValueError("Annotation paths must stay inside datasets")
        return resolved

    def manifests(self) -> list[str]:
        result = []
        for path in self.datasets.rglob("manifest.csv"):
            if not path.resolve().is_relative_to(self.datasets.resolve()):
                continue
            try:
                entries = read_manifest(path)
                if entries and all(entry.video_path is not None for entry in entries):
                    result.append(path.relative_to(self.datasets).as_posix())
            except (ValueError, OSError):
                continue
        return sorted(result)

    def entries(self, manifest: str) -> tuple:
        if Path(manifest).is_absolute() or ".." in Path(manifest).parts:
            raise ValueError("Choose a registered dataset")
        path = self.safe(self.datasets / manifest)
        if manifest not in self.manifests():
            raise ValueError("Dataset is not registered")
        entries = read_manifest(path)
        for entry in entries:
            self.safe(entry.annotations_path)
            self.safe(entry.tracks_path)
            if entry.video_path is None:
                raise ValueError("A raw video is required")
            self.safe(entry.video_path)
        return entries

    def _entry(self, manifest: str, video_id: str) -> Any:
        return next(e for e in self.entries(manifest) if e.video_id == video_id)

    @staticmethod
    def review_path(entry: Any) -> Path:
        return entry.annotations_path.with_suffix(".review.json")

    def _etag(self, manifest: str, entry: Any) -> str:
        digest = hashlib.sha256()
        for path in (self.datasets / manifest, entry.annotations_path, self.review_path(entry)):
            path = self.safe(path)
            digest.update(path.read_bytes() if path.exists() else b"")
        return digest.hexdigest()

    def open(self, manifest: str, video_id: str, previous: dict | None = None) -> dict:
        if previous and (previous.get("dirty") or previous.get("visual_dirty")):
            raise ValueError("Unsaved edits: save or explicitly discard before switching")
        with self._lock:
            entry = self._entry(manifest, video_id)
            if sha256_file(entry.video_path) != entry.video_sha256:
                raise ValueError("Raw video SHA256 mismatch")
            if sha256_file(entry.annotations_path) != entry.annotations_sha256:
                raise ValueError("Label hash changed; validate provenance before editing")
            label = load_annotations(entry.annotations_path, require_reviewed=False)
            if (
                label.video_id != video_id
                or label.duration_ms != entry.duration_ms
                or label.source_video_sha256 != entry.video_sha256
            ):
                raise ValueError("Label/video identity mismatch")
            if entry.video_sha256 not in self._pts:
                cap = cv2.VideoCapture(str(entry.video_path))
                timestamps = []
                try:
                    width, height = int(cap.get(3)), int(cap.get(4))
                    expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    while True:
                        ok, _ = cap.read()
                        if not ok:
                            break
                        t = float(cap.get(cv2.CAP_PROP_POS_MSEC))
                        if not math.isfinite(t) or t < 0 or (timestamps and t <= timestamps[-1]):
                            raise ValueError("Raw video PTS unavailable/nonmonotonic")
                        timestamps.append(t)
                    if not timestamps or len(timestamps) != expected:
                        raise ValueError("Raw video incomplete decode")
                finally:
                    cap.release()
                provenance = self.datasets / "real_world/manifests/dataset.json"
                if provenance.exists():
                    rows = json.loads(provenance.read_text(encoding="utf-8"))["videos"]
                    match = next(
                        (r for r in rows if r["processed_sha256"] == entry.video_sha256), None
                    )
                    if match:
                        quality = json.loads(
                            self.safe(Path(match["processing_record"])).read_text(encoding="utf-8")
                        )
                        saved_pts = quality["timestamps_ms"]
                        if (
                            quality["sha256"] != entry.video_sha256
                            or len(saved_pts) != len(timestamps)
                            or any(
                                abs(a - b) > 1 for a, b in zip(saved_pts, timestamps, strict=True)
                            )
                        ):
                            raise ValueError("Stored PTS differs from raw video")
                        timestamps = saved_pts
                self._pts[entry.video_sha256] = timestamps, width, height
            timestamps, width, height = self._pts[entry.video_sha256]
            review_file = self.safe(self.review_path(entry))
            review = (
                json.loads(review_file.read_text(encoding="utf-8")) if review_file.exists() else {}
            )
            if review and (
                review.get("label_sha256") != entry.annotations_sha256
                or review.get("source_video_sha256") != entry.video_sha256
                or review.get("video_id") != entry.video_id
            ):
                raise ValueError("Review evidence is not bound to current labels/video")
            return {
                "manifest": manifest,
                "video_id": video_id,
                "etag": self._etag(manifest, entry),
                "label": json.loads(label.to_json()),
                "pts": timestamps,
                "width": width,
                "height": height,
                "fps": entry.fps,
                "source_group": entry.source_group_id,
                "review": review,
                "frame_index": 0,
                "dirty": False,
                "read_only": entry.split == "test" or label.annotation_status == "reviewed",
            }

    def frame(self, state: dict, index: int) -> tuple[Any, int, float]:
        index = max(0, min(int(index), len(state["pts"]) - 1))
        entry = self._entry(state["manifest"], state["video_id"])
        cap = cv2.VideoCapture(str(entry.video_path))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = cap.read()
            if not ok or abs(cap.get(cv2.CAP_PROP_POS_MSEC) - state["pts"][index]) > 1:
                raise ValueError("Raw frame/PTS seek mismatch")
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), index, state["pts"][index]
        finally:
            cap.release()

    @staticmethod
    def seek(state: dict, time_ms: float) -> int:
        if not 0 <= time_ms <= state["label"]["duration_ms"]:
            raise ValueError("Time outside video")
        return max(0, bisect.bisect_right(state["pts"], time_ms) - 1)

    @staticmethod
    def edit(state: dict, collection: str, item: dict | None, index: int | None = None) -> dict:
        if state["read_only"]:
            raise ValueError("Reviewed/test labels are read-only; create an independent revision")
        if collection not in ("events", "excluded_intervals"):
            raise ValueError("Unsupported label collection")
        result = copy.deepcopy(state)
        values = result["label"][collection]
        if index is None:
            if item is None:
                raise ValueError("Choose an interval to delete")
            values.append(item)
        elif not 0 <= index < len(values):
            raise ValueError("Interval no longer exists")
        elif item is None:
            values.pop(index)
        else:
            values[index] = item
        result["dirty"] = True
        return result

    def validate(self, state: dict, label: dict | None = None) -> Any:
        candidate = label if label is not None else state["label"]
        entry = self._entry(state["manifest"], state["video_id"])
        if (
            candidate["video_id"] != entry.video_id
            or candidate["duration_ms"] != entry.duration_ms
            or candidate["source_video_sha256"] != entry.video_sha256
            or candidate["data_kind"] != entry.data_kind
        ):
            raise ValueError("Immutable video provenance changed")
        with tempfile.TemporaryDirectory(dir=entry.annotations_path.parent) as folder:
            path = Path(folder) / "candidate.json"
            path.write_text(json.dumps(candidate, allow_nan=False), encoding="utf-8")
            labels = load_annotations(path, require_reviewed=False)
            _validate_overlaps(labels)
        for event in labels.events:
            if event.annotation_status != "confirmed" and not event.notes.strip():
                raise ValueError("Uncertain/excluded events require an observable reason")
            if event.annotation_status != "confirmed" and event.alert_opportunities:
                raise ValueError("Uncertain/excluded events must not have alert opportunities")
            if any(op.deadline_ms >= labels.duration_ms for op in event.alert_opportunities):
                raise ValueError("Deadline must be before exclusive video end")
        for interval in labels.excluded_intervals:
            if not interval.reason.removeprefix("uncertain:").strip():
                raise ValueError("Exclusion/uncertainty requires an observable reason")
        return labels

    def save(
        self,
        state: dict,
        annotator: str,
        notes: str,
        *,
        status: str = "draft",
        reviewer: str = "",
        disagreements: str = "",
        watched: bool = False,
        no_risk: bool = False,
    ) -> dict:
        with self._lock:
            entry = self._entry(state["manifest"], state["video_id"])
            if (
                state["read_only"]
                or entry.split == "test"
                or load_annotations(
                    entry.annotations_path, require_reviewed=False
                ).annotation_status
                == "reviewed"
            ):
                raise ValueError("Frozen/reviewed data is read-only")
            if state["etag"] != self._etag(state["manifest"], entry):
                raise ValueError("Concurrent edit detected; reload before saving")
            if not annotator.strip() or annotator == "UNASSIGNED":
                raise ValueError("Explicit human annotator_id required")
            if status not in ("draft", "ready", "reviewed"):
                raise ValueError("Unsupported review status")
            label = copy.deepcopy(state["label"])
            label.update(
                annotator_id=annotator.strip(),
                notes=notes,
                annotation_status="reviewed" if status == "reviewed" else "draft",
                label_source="human_raw_video"
                if entry.data_kind == "real"
                else "synthetic_scenario",
            )
            if status in ("ready", "reviewed") and (
                not watched
                or (not label["events"] and not no_risk and not self.fully_excluded(label))
            ):
                raise ValueError(
                    "Confirm full raw-video review and explicit no-risk if events are empty"
                )
            if status == "reviewed":
                prior = state["review"]
                if (
                    prior.get("status") != "ready"
                    or not reviewer.strip()
                    or reviewer.strip() == annotator.strip()
                    or prior.get("label_sha256") != sha256_file(entry.annotations_path)
                    or label != state["label"] | {"annotation_status": "reviewed"}
                ):
                    raise ValueError("Review requires unchanged ready labels and a second reviewer")
            self.validate(state, label)
            new_bytes = (
                json.dumps(label, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
            ).encode()
            review = {
                "schema_version": "studio-review-1.0",
                "status": status,
                "annotator_id": annotator.strip(),
                "reviewer_id": reviewer.strip(),
                "disagreements": disagreements,
                "full_video_review_confirmed": watched,
                "no_risk_confirmed": no_risk,
                "video_id": entry.video_id,
                "source_video_sha256": entry.video_sha256,
                "label_sha256": hashlib.sha256(new_bytes).hexdigest(),
                "saved_at_utc": datetime.now(UTC).isoformat(),
            }
            manifest_path = self.safe(self.datasets / state["manifest"])
            rows = list(csv.DictReader(io.StringIO(manifest_path.read_text(encoding="utf-8-sig"))))
            for row in rows:
                if row["video_id"] == entry.video_id:
                    row["annotations_sha256"] = review["label_sha256"]
            csv_text = io.StringIO()
            writer = csv.DictWriter(csv_text, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            writes = {
                entry.annotations_path: new_bytes,
                self.safe(self.review_path(entry)): json.dumps(
                    review, ensure_ascii=False, indent=2
                ).encode(),
                manifest_path: csv_text.getvalue().encode(),
            }
            # Stage all files, preserve evidence, and rollback if any replacement fails.
            history = self.safe(entry.annotations_path.parent / "studio_history" / uuid.uuid4().hex)
            history.mkdir(parents=True)
            originals = {p: p.read_bytes() if p.exists() else None for p in writes}
            for i, (path, value) in enumerate(originals.items()):
                if value is not None:
                    (history / f"{i}-{path.name}").write_bytes(value)
            staged = {}
            try:
                for path, content in writes.items():
                    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
                        stream.write(content)
                        stream.flush()
                        os.fsync(stream.fileno())
                        staged[path] = Path(stream.name)
                for path, temporary in staged.items():
                    os.replace(temporary, path)
            except Exception:
                for path, original in originals.items():
                    if original is None:
                        path.unlink(missing_ok=True)
                    else:
                        path.write_bytes(original)
                raise
            finally:
                for temporary in staged.values():
                    temporary.unlink(missing_ok=True)
            return self.open(state["manifest"], state["video_id"])

    @staticmethod
    def fully_excluded(label: dict) -> bool:
        covered = 0.0
        for interval in sorted(label["excluded_intervals"], key=lambda item: item["start_ms"]):
            if interval["start_ms"] > covered:
                return False
            covered = max(covered, interval["end_ms"])
        return covered >= label["duration_ms"]

    def progress(self, manifest: str) -> list[list]:
        result = []
        for entry in self.entries(manifest):
            try:
                state = self.open(manifest, entry.video_id)
                self.validate(state)
                status = state["review"].get(
                    "status",
                    "pending" if state["label"]["annotator_id"] == "UNASSIGNED" else "draft",
                )
                if state["label"]["annotation_status"] == "reviewed":
                    status = "reviewed"
                eligible = False
                if status == "reviewed":
                    self._benchmark_entry(manifest, entry)
                    eligible = True
                result.append(
                    [
                        entry.video_id,
                        status,
                        state["label"]["annotator_id"],
                        state["review"].get("reviewer_id", ""),
                        eligible,
                    ]
                )
            except (ValueError, OSError):
                result.append([entry.video_id, "invalid", "", "", False])
        return result

    def _benchmark_entry(self, manifest: str, entry: Any) -> None:
        path = self.safe(self.datasets / manifest)
        with path.open(encoding="utf-8-sig", newline="") as stream:
            row = next(r for r in csv.DictReader(stream) if r["video_id"] == entry.video_id)
        with tempfile.TemporaryDirectory(dir=path.parent) as folder:
            single = Path(folder) / "manifest.csv"
            for key in ("video_path", "tracks_path", "annotations_path"):
                row[key] = str((path.parent / row[key]).resolve())
            with single.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
                writer.writeheader()
                writer.writerow(row)
            load_dataset(single, require_reviewed=True, check_videos=True)

    def benchmark_gate(self, manifest: str) -> str:
        load_dataset(self.safe(self.datasets / manifest), require_reviewed=True, check_videos=True)
        return "Phase 3 validation passed; use existing Benchmark command with reviewed manifest"

    def visual(self, state: dict, attributes: dict | None = None, annotator: str = "") -> tuple:
        entry = self._entry(state["manifest"], state["video_id"])
        # Only an existing task explicitly bound to this video hash may be edited.
        tasks_path = self.safe(self.datasets / "real_world/annotations/visual_tasks.json")
        tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
        task = next(t for t in tasks["videos"] if t["video_id"] == entry.video_id)
        original_manifest = self.safe(self.datasets / "real_world/annotations/phase3/manifest.csv")
        original = next(e for e in read_manifest(original_manifest) if e.video_id == entry.video_id)
        if original.video_sha256 != entry.video_sha256:
            raise ValueError("Visual task is for another raw video")
        image_path = self.safe(Path(task["image_path"]))
        if sha256_file(image_path) != task["image_sha256"]:
            raise ValueError("Visual task image hash mismatch")
        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        if attributes is not None:
            if not annotator.strip():
                raise ValueError("Visual annotator required")
            if set(attributes) != set(tasks["attributes"]):
                raise ValueError("Visual fields differ from task definition")
            if attributes["position_left_center_right"] not in (None, "left", "center", "right"):
                raise ValueError("Invalid visible position")
            for key in (
                "visible",
                "overlap_visible_passage",
                "occlusion",
                "frame_motion",
                "evidence_sufficient",
            ):
                if attributes[key] is not None and type(attributes[key]) is not bool:
                    raise ValueError("Visual boolean fields allow true/false/unknown only")
            # Per-dataset sidecar isolates testing/review from original Phase 6A tasks.
            if state["read_only"]:
                raise ValueError("Reviewed/test visual annotations are read-only")
            path = self.safe(entry.annotations_path.parent / f"{entry.video_id}.visual.json")
            value = {
                "schema_version": tasks["schema_version"],
                "video_id": entry.video_id,
                "image_sha256": task["image_sha256"],
                "timestamp_ms": task["image_timestamp_ms"],
                "human_attributes": attributes,
                "annotator_id": annotator.strip(),
                "saved_at_utc": datetime.now(UTC).isoformat(),
                "source": "human_raw_frame",
            }
            with self._lock:
                if path.exists():
                    history = self.safe(path.parent / "studio_history" / uuid.uuid4().hex)
                    history.mkdir(parents=True)
                    (history / path.name).write_bytes(path.read_bytes())
                with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
                    temp = Path(stream.name)
                    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False).encode())
                try:
                    os.replace(temp, path)
                finally:
                    temp.unlink(missing_ok=True)
        visual_path = self.safe(entry.annotations_path.parent / f"{entry.video_id}.visual.json")
        if visual_path.exists():
            saved = json.loads(visual_path.read_text(encoding="utf-8"))
            if saved["image_sha256"] != task["image_sha256"]:
                raise ValueError("Saved visual annotation refers to another image")
            task = task | {
                "human_attributes": saved["human_attributes"],
                "annotator_id": saved["annotator_id"],
            }
        return image, task
