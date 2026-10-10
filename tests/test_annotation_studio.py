"""Studio edits remain independent, exact, recoverable, and non-evaluated until review."""

import copy
import csv
import json

import cv2
import numpy as np
import pytest

from sightguard.data import FrameRecord
from sightguard.evaluation.dataset import MANIFEST_FIELDS, sha256_file
from sightguard.evaluation.importing import import_video
from sightguard.evaluation.schema import load_annotations
from sightguard.ui.annotation_store import AnnotationStore
from sightguard.ui.annotation_tab import parse_opportunities, timeline


@pytest.fixture
def studio(tmp_path):
    data = tmp_path / "datasets/sample"
    data.mkdir(parents=True)
    video = data / "raw.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    assert writer.isOpened()
    for i in range(20):
        writer.write(np.full((48, 64, 3), i * 10, dtype=np.uint8))
    writer.release()
    tracks = data / "tracks.jsonl"
    tracks.write_text("\n".join(FrameRecord(i, i * 100, 64, 48).to_json() for i in range(20)))
    (data / "run_metadata.json").write_text(
        json.dumps(
            {"status": "completed", "frames_processed": 20, "input": {"sha256": sha256_file(video)}}
        )
    )
    import_video(video, tracks, data, video_id="real", source_group_id="parent", scene="unlabelled")
    store = AnnotationStore(tmp_path)
    return store, store.open("sample/manifest.csv", "real")


def event():
    return {
        "video_id": "real",
        "event_id": "human-1",
        "event_type": "visible_obstruction",
        "start_ms": 100,
        "end_ms": 1500,
        "severity": "HIGH",
        "annotation_status": "confirmed",
        "annotator_id": "human-a",
        "notes": "raw frame evidence",
        "class_ids": [0],
        "alert_opportunities": [
            {
                "opportunity_id": "initial-1",
                "kind": "initial",
                "window_start_ms": 100,
                "deadline_ms": 500,
                "minimum_severity": "MEDIUM",
            },
            {
                "opportunity_id": "upgrade-1",
                "kind": "escalation",
                "window_start_ms": 800,
                "deadline_ms": 1000,
                "minimum_severity": "HIGH",
            },
        ],
        "target_regions": [
            {"start_ms": 100, "end_ms": 1500, "bbox_normalized": [0.1, 0.1, 0.8, 0.9]}
        ],
    }


def test_keyboard_edited_numeric_cells():
    assert (
        parse_opportunities([["i", "initial", "100.125", "500.5", "MEDIUM"]])[0]["window_start_ms"]
        == 100.125
    )
    with pytest.raises(ValueError):
        parse_opportunities([["i", "initial", "unknown", 500, "MEDIUM"]])


def test_registered_video_identity_raw_frame_pts(studio):
    store, state = studio
    assert state["video_id"] == "real" and state["width"] == 64
    image, index, pts = store.frame(state, 7)
    assert index == 7 and pts == pytest.approx(700) and image.mean() == pytest.approx(68, abs=5)
    assert store.seek(state, 750) == 7
    changed = copy.deepcopy(state)
    changed["pts"] = [0, 40, 105, 180]
    assert (
        store.seek(changed, 110) == 2
    )  # VFR selection uses supplied PTS, never fps multiplication.
    assert not any(k in state for k in ("detections", "risk", "alerts", "model_predictions"))
    with pytest.raises(ValueError):
        store.open("../escape/manifest.csv", "real")
    with pytest.raises(StopIteration):
        store.open("sample/manifest.csv", "other")
    with pytest.raises(ValueError, match="Unsaved"):
        store.open(state["manifest"], "real", state | {"visual_dirty": True})


def test_create_modify_delete_and_timeline(studio):
    store, state = studio
    edited = store.edit(state, "events", event())
    store.validate(edited)
    assert state["label"]["events"] == [] and edited["dirty"]
    image, rows = timeline(edited)
    assert image.shape[1] == 1000 and rows == [("events", 0)]
    revised = event()
    revised["end_ms"] = 1600
    edited = store.edit(edited, "events", revised, 0)
    store.validate(edited)
    deleted = store.edit(edited, "events", None, 0)
    assert deleted["label"]["events"] == []
    interval = store.edit(
        deleted,
        "excluded_intervals",
        {"start_ms": 1100, "end_ms": 1300, "reason": "uncertain: occluded"},
    )
    store.validate(interval)


@pytest.mark.parametrize(
    "changes",
    [
        {"start_ms": 1500, "end_ms": 1400},
        {"end_ms": 3000},
        {"class_ids": []},
        {"event_id": ""},
        {
            "target_regions": [
                {"start_ms": 100, "end_ms": 1500, "bbox_normalized": [0.8, 0.1, 0.1, 0.9]}
            ]
        },
        {"alert_opportunities": []},
    ],
)
def test_invalid_event_rejected(studio, changes):
    store, state = studio
    with pytest.raises(ValueError):
        store.validate(store.edit(state, "events", event() | changes))


def test_illegal_overlaps_and_opportunities(studio):
    store, state = studio
    one = event()
    one["target_regions"] = []
    s = store.edit(state, "events", one)
    with pytest.raises(ValueError, match="Overlapping"):
        store.validate(store.edit(s, "events", one | {"event_id": "two"}))
    one = event()
    one["alert_opportunities"][1]["window_start_ms"] = 50
    with pytest.raises(ValueError):
        store.validate(store.edit(state, "events", one))
    with pytest.raises(ValueError):
        store.validate(
            store.edit(
                state, "excluded_intervals", {"start_ms": 0, "end_ms": 4000, "reason": "bad"}
            )
        )


def test_draft_save_reload_hash_binding_unsaved_switch_and_conflict(studio):
    store, state = studio
    edited = store.edit(state, "events", event())
    with pytest.raises(ValueError, match="Unsaved"):
        store.open(state["manifest"], "real", edited)
    saved = store.save(edited, "human-a", "independent")
    assert saved["label"]["events"][0]["end_ms"] == 1500 and not saved["dirty"]
    assert saved["label"]["annotation_status"] == "draft"
    assert store.progress(state["manifest"])[0][1] == "draft"
    assert list((store.datasets / "sample/annotations/studio_history").glob("*/*"))
    with pytest.raises(ValueError, match="Concurrent"):
        store.save(edited, "human-a", "stale")
    with pytest.raises(ValueError, match="Draft"):
        store.benchmark_gate(state["manifest"])
    with pytest.raises(ValueError):
        load_annotations(store._entry(state["manifest"], "real").annotations_path)


def test_empty_review_requires_explicit_human_confirmation_and_second_reviewer(studio):
    store, state = studio
    with pytest.raises(ValueError):
        store.save(
            state,
            "human-a",
            "negative",
            status="reviewed",
            watched=True,
            no_risk=True,
            reviewer="human-b",
        )
    with pytest.raises(ValueError):
        store.save(state, "human-a", "negative", status="ready", watched=True)
    ready = store.save(state, "human-a", "negative", status="ready", watched=True, no_risk=True)
    assert ready["label"]["annotation_status"] == "draft" and ready["review"]["status"] == "ready"
    with pytest.raises(ValueError):
        store.save(
            ready,
            "human-a",
            "negative",
            status="reviewed",
            watched=True,
            no_risk=True,
            reviewer="human-a",
        )
    done = store.save(
        ready,
        "human-a",
        "negative",
        status="reviewed",
        watched=True,
        no_risk=True,
        reviewer="human-b",
        disagreements="none",
    )
    assert done["read_only"] and done["label"]["events"] == []
    assert "passed" in store.benchmark_gate(state["manifest"])
    with pytest.raises(ValueError):
        store.edit(done, "events", event())
    with pytest.raises(ValueError, match="read-only"):
        store.save(done | {"read_only": False}, "human-a", "attempted downgrade")
    assert store.progress(state["manifest"])[0][-1] is True


def test_failed_atomic_replace_restores_all_evidence(studio, monkeypatch):
    store, state = studio
    entry = store._entry(state["manifest"], "real")
    old = entry.annotations_path.read_bytes()
    manifest = store.datasets / state["manifest"]
    oldmanifest = manifest.read_bytes()
    import sightguard.ui.annotation_store as module

    real_replace = module.os.replace
    count = 0

    def fail_second(a, b):
        nonlocal count
        count += 1
        if count == 2:
            raise OSError("simulated disk error")
        return real_replace(a, b)

    monkeypatch.setattr(module.os, "replace", fail_second)
    with pytest.raises(OSError):
        store.save(state, "human-a", "test")
    assert entry.annotations_path.read_bytes() == old and manifest.read_bytes() == oldmanifest
    assert not store.review_path(entry).exists()
    assert store.open(state["manifest"], "real")["etag"] == state["etag"]


def test_fully_unobservable_video_is_not_forced_into_negative_gt(studio):
    store, state = studio
    edited = store.edit(
        state,
        "excluded_intervals",
        {"start_ms": 0, "end_ms": 2000, "reason": "uncertain: fully occluded"},
    )
    ready = store.save(edited, "human-a", "cannot determine", status="ready", watched=True)
    done = store.save(
        ready, "human-a", "cannot determine", status="reviewed", watched=True, reviewer="human-b"
    )
    assert done["review"]["no_risk_confirmed"] is False
    assert len(done["label"]["excluded_intervals"]) == 1
    invalid = store.edit(
        state, "excluded_intervals", {"start_ms": 0, "end_ms": 100, "reason": "uncertain:"}
    )
    with pytest.raises(ValueError, match="reason"):
        store.validate(invalid)


def test_visual_attributes_are_isolated_from_risk_and_source_tasks(studio):
    store, state = studio
    root = store.datasets / "real_world/annotations"
    root.mkdir(parents=True)
    origin = root / "phase3"
    origin.mkdir()
    (origin / "manifest.csv").write_bytes((store.datasets / state["manifest"]).read_bytes())
    image = root / "frame.png"
    cv2.imwrite(str(image), np.zeros((48, 64, 3), np.uint8))
    fields = [
        "object_class",
        "visible",
        "position_left_center_right",
        "overlap_visible_passage",
        "occlusion",
        "frame_motion",
        "evidence_sufficient",
    ]
    tasks = root / "visual_tasks.json"
    tasks.write_text(
        json.dumps(
            {
                "schema_version": "v01",
                "attributes": fields,
                "videos": [
                    {
                        "video_id": "real",
                        "image_path": str(image),
                        "image_sha256": sha256_file(image),
                        "image_timestamp_ms": 700,
                    }
                ],
            }
        )
    )
    old = tasks.read_bytes()
    label = store._entry(state["manifest"], "real").annotations_path
    oldlabel = label.read_bytes()
    attrs = dict.fromkeys(fields)
    attrs["visible"] = True
    store.visual(state, attrs, "human-v")
    assert tasks.read_bytes() == old and label.read_bytes() == oldlabel
    assert label.with_name("real.visual.json").exists()
    with pytest.raises(ValueError):
        store.visual(state, attrs | {"visible": "yes"}, "human-v")


def test_manifest_paths_outside_dataset_rejected(studio):
    store, state = studio
    manifest = store.datasets / state["manifest"]
    with manifest.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["annotations_path"] = str(store.root / "outside.json")
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="inside datasets"):
        store.entries(state["manifest"])
