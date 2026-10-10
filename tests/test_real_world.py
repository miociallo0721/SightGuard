"""Acquisition failures, temporal provenance, and leakage are never valid negatives."""

import hashlib
import io
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from sightguard.evaluation.real_world import (
    clip_video,
    download_object,
    extract_member,
    group_split,
    inspect_video,
    validate_manifest,
)
from sightguard.evaluation.schema import VideoAnnotations, load_annotations
from sightguard.hybrid.schema import CRITERIA, parse_output


class Response(io.BytesIO):
    headers = {}


def test_download_success_checksum_and_no_overwrite(tmp_path, monkeypatch):
    payload = b"source data"
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: Response(payload))
    target = tmp_path / "source.bin"
    result = download_object(
        "https://official.test/file",
        target,
        expected_size=len(payload),
        expected_md5=hashlib.md5(payload).hexdigest(),
    )
    assert result["sha256"] == hashlib.sha256(payload).hexdigest()
    assert result["status"] == "verified"
    with pytest.raises(FileExistsError):
        download_object(
            "https://other.test/file", target, expected_size=len(payload), expected_md5="0" * 32
        )
    assert json.loads(target.with_suffix(".bin.download.json").read_text())["status"] == "verified"


@pytest.mark.parametrize("response", [b"short", b"wrongdata"])
def test_incomplete_or_checksum_failure_not_verified(tmp_path, monkeypatch, response):
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: Response(response))
    target = tmp_path / "failed.bin"
    with pytest.raises(ValueError):
        download_object(
            "https://official.test/file", target, expected_size=9, expected_md5="0" * 32
        )
    assert not target.exists()
    assert json.loads(target.with_suffix(".bin.download.json").read_text())["status"] == "failed"


def test_oversize_and_archive_traversal_rejected(tmp_path):
    with pytest.raises(ValueError, match="approval"):
        download_object(
            "https://official.test/file",
            tmp_path / "file",
            expected_size=600 * 1024**2,
            expected_md5="0" * 32,
        )
    with pytest.raises(ValueError, match="Unsafe"):
        extract_member(tmp_path / "archive.zip", "../escape", tmp_path / "escape")


def test_network_failure_never_creates_verified_record(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr("urllib.request.urlopen", fail)
    target = tmp_path / "network.bin"
    with pytest.raises(OSError):
        download_object(
            "https://official.test/file", target, expected_size=9, expected_md5="0" * 32
        )
    assert not target.exists()
    assert json.loads(target.with_suffix(".bin.download.json").read_text())["status"] == "failed"


def test_manifest_same_parent_leakage_and_accepted_missing_file(tmp_path):
    row = {
        "schema_version": "sightguard.real-world.v0.1",
        "video_id": "one",
        "parent_source_id": "parent",
        "split": "development",
        "annotation_status": "pending",
        "eligibility_status": "pending_visual_review",
    }
    with pytest.raises(ValueError, match="leakage"):
        validate_manifest([row, {**row, "video_id": "two", "split": "test"}])
    with pytest.raises(ValueError, match="Duplicate"):
        validate_manifest([row, row])
    with pytest.raises(ValueError, match="verified local"):
        validate_manifest(
            [
                {
                    **row,
                    "eligibility_status": "accepted",
                    "acquisition_status": "failed",
                    "filepath": str(tmp_path / "missing.mp4"),
                }
            ]
        )


def test_video_metadata_and_time_mapping(tmp_path: Path):
    source = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    assert writer.isOpened()
    for index in range(20):
        writer.write(np.full((48, 64, 3), index * 10, dtype=np.uint8))
    writer.release()
    info = inspect_video(source)
    assert info["frames"] == 20 and info["fps"] == 10
    clip = clip_video(source, tmp_path / "clip.mp4", 0.5, 1.5, source_info=info)
    assert clip["frames"] == 10
    assert clip["time_mapping"]["source_ms"] == pytest.approx(info["timestamps_ms"][5:15])
    assert clip["duration_ms"] == pytest.approx(1000)
    with pytest.raises(FileExistsError):
        clip_video(source, tmp_path / "clip.mp4", 0, 1)
    with pytest.raises(ValueError):
        inspect_video(tmp_path / "missing.mp4")


def test_group_split_and_duplicate_manifest_rejected():
    rows = [{"video_id": str(i), "parent_source_id": str(i // 2)} for i in range(10)]
    result = group_split(rows, seed=17, test_fraction=0.4)
    assert all(result[str(i)] == result[str(i + 1)] for i in range(0, 10, 2))
    assert group_split(rows, seed=17) == dict.fromkeys([str(i) for i in range(10)], "development")
    with pytest.raises(ValueError):
        validate_manifest([{"schema_version": "wrong"}])


def test_accepted_hash_tampering_and_duplicate_content(tmp_path):
    path = tmp_path / "video.mp4"
    path.write_bytes(b"altered video")
    row = {
        "schema_version": "sightguard.real-world.v0.1",
        "video_id": "one",
        "parent_source_id": "parent",
        "split": "development",
        "annotation_status": "pending",
        "eligibility_status": "accepted",
        "acquisition_status": "verified",
        "review_status": "visual_reviewed",
        "filepath": str(path),
        "processed_sha256": "0" * 64,
    }
    with pytest.raises(ValueError, match="SHA256"):
        validate_manifest([row])
    row["processed_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="Duplicate processed"):
        validate_manifest([row, {**row, "video_id": "two"}])


def test_unreviewed_annotations_and_unknown_are_not_metrics(tmp_path):
    draft = VideoAnnotations(
        "real",
        1000,
        "real",
        "draft",
        "UNASSIGNED",
        "pending_human_review",
        "unavailable",
        "No machine-generated ground truth",
        (),
    )
    path = tmp_path / "draft.json"
    path.write_text(draft.to_json())
    assert load_annotations(path, require_reviewed=False).schema_version == "3.0"
    with pytest.raises(ValueError, match="Draft"):
        load_annotations(path)
    level, _, reason = parse_output(
        {"answers": {"risk": {"choice": CRITERIA["UNKNOWN"]}}}, "vision"
    )
    assert level is None and "UNKNOWN" in reason
