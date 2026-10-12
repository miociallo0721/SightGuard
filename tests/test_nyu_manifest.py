"""Frozen public-data provenance must survive partial downloads and resume."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


def invoke(tmp_path, monkeypatch, *, changed=False, missing=False, partial=False):
    source = Path(__file__).parents[1] / "scripts/finalize_nyu_manifest.py"
    spec = importlib.util.spec_from_file_location("finalize_nyu_manifest", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sample = {"sample_id": "nyu_0001", "index": 1, "scene": "room", "split": "evaluation"}
    frozen = {"labels": {"wall": 21, "floor": 11}, "samples": [sample]}
    if missing:
        frozen["samples"].append({**sample, "sample_id": "nyu_0002", "index": 2})
    (tmp_path / "frozen-selection.json").write_text(json.dumps(frozen))
    artifact = tmp_path / "artifact"
    artifact.write_bytes(b"image or truth")
    record = {**sample, "status": "complete"}
    for name in ("rgb", "gt"):
        record[f"{name}_path"] = str(artifact)
        record[f"{name}_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    (tmp_path / "extraction.jsonl").write_text(json.dumps(record) + "\n")
    if changed:
        artifact.write_bytes(b"changed")
    output = tmp_path / "manifest.json"
    args = [str(source), "--extracted", str(tmp_path), "--output", str(output)]
    if partial:
        args.append("--allow-partial")
    monkeypatch.setattr(sys, "argv", args)
    module.main()
    return json.loads(output.read_text())


def test_offline_manifest_preserves_mapping_scene_and_unrelated_time(tmp_path, monkeypatch):
    result = invoke(tmp_path, monkeypatch)
    assert result["labels"] == {"wall": 21, "floor": 11}
    assert result["completed_samples"] == result["planned_samples"] == 1
    assert result["samples"][0]["intrinsic"][0] == pytest.approx(518.8579011745019)
    assert "unrelated_still" in result["samples"][0]["time_contract"]


def test_changed_input_cannot_enter_evaluation(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="checksum"):
        invoke(tmp_path, monkeypatch, changed=True)


def test_missing_frozen_sample_is_not_silently_dropped(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="missing"):
        invoke(tmp_path, monkeypatch, missing=True)


def test_explicit_partial_result_keeps_missing_ids(tmp_path, monkeypatch):
    result = invoke(tmp_path, monkeypatch, missing=True, partial=True)
    assert result["missing_samples"] == ["nyu_0002"]
    assert result["planned_samples"] == 2
    assert result["completed_samples"] == 1
