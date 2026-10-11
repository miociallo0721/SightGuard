"""Offline unit tests: injected model outputs are explicitly simulated."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sightguard.perception.depth.artifacts import (
    load_depth_map,
    save_prediction,
    sha256_file,
    visualize_depth,
)
from sightguard.perception.depth.model import (
    DepthEstimator,
    DepthModelUnavailableError,
    InvalidDepthOutputError,
    bgr_to_rgb,
    check_depth,
)
from sightguard.perception.depth.schema import SEMANTICS, DepthFrameResult, DepthModelConfig
from sightguard.pipeline.depth import run_depth_batch, sample_indices


class SimulatedOOM(RuntimeError):
    pass


def mock_estimator(monkeypatch, config=None, output=None):
    estimator = DepthEstimator(config or DepthModelConfig())
    estimator.model = SimpleNamespace(to=lambda device: None)
    estimator.torch = SimpleNamespace(
        cuda=SimpleNamespace(
            OutOfMemoryError=SimulatedOOM,
            empty_cache=lambda: None,
            is_available=lambda: False,
        )
    )
    estimator.device_name = "cpu"
    depth = np.arange(48 * 64, dtype=np.float32).reshape(48, 64) if output is None else output
    monkeypatch.setattr(
        estimator, "_predict", lambda rgb: (depth, depth, (48, 64), {"inference": 1.0})
    )
    return estimator


def infer(estimator, frame=None, **overrides):
    values = dict(
        video_id="one",
        frame_index=0,
        timestamp_ms=0.0,
        source_width=64,
        source_height=48,
        timestamp_source="video_pts",
    )
    values.update(overrides)
    return estimator.infer(
        np.zeros((48, 64, 3), dtype=np.uint8) if frame is None else frame, **values
    )


@pytest.mark.parametrize(
    "values",
    [
        {"input_size": 0},
        {"input_size": 281},
        {"input_size": True},
        {"device": "mps"},
        {"min_relative_spread": float("nan")},
        {"min_relative_spread": 0},
        {"model_revision": "main"},
        {"checkpoint_sha256": "0" * 64},
        {"allow_cpu_fallback": "true"},
    ],
)
def test_config_rejects_invalid(values):
    with pytest.raises(ValueError):
        DepthModelConfig(**values)


def test_missing_local_weights_are_explicit_and_no_implicit_load(tmp_path):
    estimator = DepthEstimator(DepthModelConfig(checkpoint_dir=tmp_path))
    assert estimator.model is None
    with pytest.raises(DepthModelUnavailableError, match="missing"):
        estimator.initialize()
    prediction = infer(estimator)
    assert prediction.result.status == "unknown"
    assert prediction.result.inference_status == "model_unavailable"
    assert prediction.depth_map is None


def test_corrupted_checkpoint_rejected_before_runtime_import(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"wrong checkpoint")
    with pytest.raises(DepthModelUnavailableError, match="SHA-256 mismatch"):
        DepthEstimator(DepthModelConfig(checkpoint_dir=tmp_path)).initialize()


def test_bgr_rgb_copy_and_input_validation():
    frame = np.array([[[10, 20, 30]]], dtype=np.uint8)
    converted = bgr_to_rgb(frame)
    assert converted.tolist() == [[[30, 20, 10]]]
    converted[:] = 0
    assert frame.tolist() == [[[10, 20, 30]]]
    for bad in (
        None,
        np.zeros((0, 2, 3), dtype=np.uint8),
        np.zeros((2, 3)),
        np.zeros((2, 2, 3), dtype=np.float32),
    ):
        with pytest.raises(ValueError):
            bgr_to_rgb(bad)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_rejected(monkeypatch, value):
    depth = np.arange(48 * 64, dtype=np.float32).reshape(48, 64)
    depth[4, 3] = value
    prediction = infer(mock_estimator(monkeypatch, output=depth))
    assert prediction.result.inference_status == "completed"
    assert prediction.result.quality_reasons == ("non_finite_output",)
    assert prediction.depth_map is None


def test_constant_output_is_unknown_but_retained(monkeypatch):
    prediction = infer(mock_estimator(monkeypatch, output=np.ones((48, 64), np.float32)))
    assert prediction.result.status == "unknown"
    assert prediction.result.quality_reasons == ("insufficient_output_variation",)
    assert prediction.depth_map is not None


def test_invalid_output_shape_and_input_shape(monkeypatch):
    estimator = mock_estimator(monkeypatch, output=np.ones((2, 3), np.float32))
    assert infer(estimator).result.quality_reasons == ("invalid_output_size",)
    assert infer(estimator, source_width=32).result.inference_status == "image_read_failed"
    monkeypatch.setattr(
        estimator,
        "_predict",
        lambda rgb: (_ for _ in ()).throw(InvalidDepthOutputError("bad shape")),
    )
    assert infer(estimator).result.quality_reasons == ("invalid_output_size",)


def test_no_stale_output_after_failure(monkeypatch):
    estimator = mock_estimator(monkeypatch)
    assert infer(estimator).depth_map is not None
    monkeypatch.setattr(
        estimator, "_predict", lambda rgb: (_ for _ in ()).throw(RuntimeError("broken"))
    )
    failed = infer(estimator, frame_index=1, timestamp_ms=100)
    assert failed.depth_map is None
    assert failed.result.status == "unknown"
    assert failed.result.inference_status == "inference_failed"
    assert "broken" in failed.result.error


def test_cuda_oom_only_one_cpu_retry(monkeypatch):
    estimator = mock_estimator(monkeypatch)
    estimator.device_name = "cuda:0"
    calls = []

    def predict(rgb):
        calls.append(estimator.device_name)
        if len(calls) == 1:
            raise SimulatedOOM("OOM")
        depth = np.arange(48 * 64, dtype=np.float32).reshape(48, 64)
        return depth, depth, (48, 64), {"inference": 1.0}

    monkeypatch.setattr(estimator, "_predict", predict)
    prediction = infer(estimator)
    assert calls == ["cuda:0", "cpu"]
    assert prediction.result.device == "cpu"
    assert prediction.result.runtime_notes == ("cuda_oom_one_cpu_retry",)
    monkeypatch.setattr(
        estimator, "_predict", lambda rgb: (_ for _ in ()).throw(SimulatedOOM("again"))
    )
    assert infer(estimator).depth_map is None


def test_oom_fallback_can_be_disabled(monkeypatch):
    estimator = mock_estimator(monkeypatch, DepthModelConfig(allow_cpu_fallback=False))
    estimator.device_name = "cuda:0"
    monkeypatch.setattr(
        estimator, "_predict", lambda rgb: (_ for _ in ()).throw(SimulatedOOM("OOM"))
    )
    assert infer(estimator).result.quality_reasons == ("cuda_oom_no_retry",)
    assert estimator.device_name == "cuda:0"


def test_lossless_save_and_visualization_separation(monkeypatch, tmp_path):
    prediction = infer(mock_estimator(monkeypatch))
    original = prediction.depth_map.copy()
    frame = np.zeros((48, 64, 3), np.uint8)
    visualize_depth(frame, prediction.depth_map, "diagnostic")
    assert np.array_equal(original, prediction.depth_map)
    result = save_prediction(tmp_path, prediction, frame)
    with np.load(tmp_path / result.depth_data_reference, allow_pickle=False) as data:
        assert data["depth"].dtype == np.float32
        assert np.array_equal(data["depth"], original)
        assert np.array_equal(data["model_output"], prediction.model_output)
    row = json.loads(json.dumps(result.to_dict(), allow_nan=False))
    assert row["depth_semantics"] == SEMANTICS
    assert row["depth_map_shape"] == [48, 64]
    assert row["depth_data_sha256"] == sha256_file(tmp_path / row["depth_data_reference"])
    assert row["model_revision"] == prediction.result.model_revision
    assert row["production_decision_changed"] is False
    assert "meters" not in row
    assert np.array_equal(load_depth_map(tmp_path, row), original)
    with pytest.raises(ValueError, match="UNKNOWN"):
        load_depth_map(tmp_path, {**row, "status": "unknown"})
    with pytest.raises(ValueError, match="escapes"):
        load_depth_map(tmp_path, {**row, "depth_data_reference": "../other.npz"})
    with pytest.raises(ValueError, match="SHA-256"):
        load_depth_map(tmp_path, {**row, "depth_data_sha256": "0" * 64})
    with pytest.raises(FileExistsError):
        save_prediction(tmp_path, prediction, frame)


def test_exact_frame_temporal_contract(monkeypatch):
    result = infer(mock_estimator(monkeypatch), timestamp_ms=500, frame_index=10).result
    assert result.matches_frame("one", 10, 500)
    assert not result.matches_frame("two", 10, 500)
    assert not result.matches_frame("one", 11, 500)
    assert not result.matches_frame("one", 10, 499)
    assert not replace(result, status="unknown").matches_frame("one", 10, 500)
    for t in (float("nan"), float("inf"), -1):
        with pytest.raises(ValueError):
            DepthFrameResult("one", 0, t, 64, 48)


def test_sampling_matches_v04():
    assert sample_indices(300, 20, 0.5) == list(range(0, 300, 10))
    for args in ((300, 0, 0.5), (0, 20, 0.5), (300, 20, 0.075), (300, 20, float("nan"))):
        with pytest.raises(ValueError):
            sample_indices(*args)


def fixtures(video: Path, tmp_path: Path):
    baseline, scene = tmp_path / "baseline", tmp_path / "scene"
    rows = []
    for video_id in ("one", "two"):
        tracks = [
            dict(
                frame_index=i,
                timestamp_ms=i * 100.0,
                timestamp_source="video_pts",
                frame_width=64,
                frame_height=48,
            )
            for i in range(5)
        ]
        risks = [dict(frame_index=i, timestamp_ms=i * 100.0, observations=[]) for i in range(5)]
        for name, values in (
            ("perception/tracks.jsonl", tracks),
            ("assessment/risk_frames.jsonl", risks),
        ):
            p = baseline / video_id / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("\n".join(json.dumps(v) for v in values) + "\n")
        (baseline / video_id / "assessment/alerts.jsonl").write_text("")
        (baseline / video_id / "assessment/run_metadata.json").write_text(
            json.dumps({"stream_id": video_id})
        )
        p = scene / video_id / "scene_frames.jsonl"
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps(dict(video_id=video_id, frame_index=0, timestamp_ms=0.0)) + "\n")
        rows.append(
            dict(
                video_id=video_id,
                filepath=str(video),
                fps=10,
                width=64,
                height=48,
                processed_sha256=sha256_file(video),
            )
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"schema_version": "sightguard.real-world.v0.1", "videos": rows})
    )
    return manifest, baseline, scene


def test_runner_state_isolation_and_output_guard(monkeypatch, video, tmp_path):
    manifest, baseline, scene = fixtures(video, tmp_path)
    estimator = mock_estimator(monkeypatch)
    monkeypatch.setattr(estimator, "initialize", lambda: None)
    monkeypatch.setattr("sightguard.pipeline.depth.DepthEstimator", lambda config: estimator)
    out = tmp_path / "depth"
    result = run_depth_batch(manifest, baseline, scene, out, DepthModelConfig(), warmup_frames=0)
    assert result["status"] == "completed"
    for video_id in ("one", "two"):
        rows = [
            json.loads(v) for v in (out / video_id / "depth_frames.jsonl").read_text().splitlines()
        ]
        assert len(rows) == 1
        assert rows[0]["video_id"] == rows[0]["stream_id"] == video_id
        assert rows[0]["timestamp_ms"] == 0
    with pytest.raises(FileExistsError):
        run_depth_batch(manifest, baseline, scene, out, DepthModelConfig())
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileExistsError):
        run_depth_batch(manifest, baseline, scene, empty, DepthModelConfig())


def test_failed_video_and_missing_model_are_recorded(monkeypatch, video, tmp_path):
    manifest, baseline, scene = fixtures(video, tmp_path)
    result = run_depth_batch(
        manifest,
        baseline,
        scene,
        tmp_path / "unknown",
        DepthModelConfig(checkpoint_dir=tmp_path / "missing"),
    )
    assert result["status"] == "completed_with_unknown_or_failures"
    assert all(v["unknown_frames"] == 1 for v in result["videos"])
    values = json.loads(manifest.read_text())
    values["videos"][0]["filepath"] = str(tmp_path / "invalid.mp4")
    (tmp_path / "invalid.mp4").write_bytes(b"bad")
    values["videos"][0]["processed_sha256"] = sha256_file(tmp_path / "invalid.mp4")
    manifest.write_text(json.dumps(values))
    result = run_depth_batch(
        manifest,
        baseline,
        scene,
        tmp_path / "invalid",
        DepthModelConfig(checkpoint_dir=tmp_path / "missing"),
    )
    assert result["videos"][0]["status"] == "failed"
    assert "opened" in result["videos"][0]["error"]
    assert result["videos"][1]["sampled_frames"] == 1


def test_timestamp_mismatch_is_not_inferred_as_valid(monkeypatch, video, tmp_path):
    manifest, baseline, scene = fixtures(video, tmp_path)
    p = scene / "one/scene_frames.jsonl"
    p.write_text(json.dumps(dict(video_id="one", frame_index=0, timestamp_ms=50)) + "\n")
    result = run_depth_batch(
        manifest,
        baseline,
        scene,
        tmp_path / "mismatch",
        DepthModelConfig(checkpoint_dir=tmp_path / "missing"),
    )
    assert result["videos"][0]["status"] == "failed"
    assert "time/video identity" in result["videos"][0]["error"]
    assert result["videos"][0]["sampled_frames"] == 0


def test_quality_statistics_preserve_negative_interpolation_values():
    depth = np.arange(12, dtype=np.float32).reshape(3, 4) - 2
    reasons, stats = check_depth(depth, (3, 4), 1e-5)
    assert not reasons
    assert stats["min"] == -2


@pytest.mark.parametrize(
    "requested,available,expected",
    [
        ("auto", False, "cpu"),
        ("auto", True, "cuda:0"),
        ("cpu", True, "cpu"),
        ("cuda:0", True, "cuda:0"),
    ],
)
def test_simulated_checkpoint_loading_device_selection_and_single_load(
    monkeypatch,
    tmp_path,
    requested,
    available,
    expected,
):
    import sys

    config = DepthModelConfig(checkpoint_dir=tmp_path, device=requested)
    hashes = {
        "model.safetensors": config.checkpoint_sha256,
        "config.json": config.config_sha256,
        "preprocessor_config.json": config.processor_sha256,
    }
    for name in hashes:
        (tmp_path / name).write_bytes(b"unit-test-only-placeholder")
    monkeypatch.setattr("sightguard.perception.depth.model.sha256_file", lambda p: hashes[p.name])
    calls = []
    model = SimpleNamespace(
        config=SimpleNamespace(depth_estimation_type="relative"),
        eval=lambda: None,
        to=lambda d: calls.append(d),
        parameters=lambda: [SimpleNamespace(numel=lambda: 24)],
    )

    def load(path, **kwargs):
        assert kwargs["local_files_only"] is True
        calls.append("load")
        return model, {"missing_keys": set(), "unexpected_keys": set()}

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            DepthAnythingForDepthEstimation=SimpleNamespace(from_pretrained=load),
            DPTImageProcessor=SimpleNamespace(from_pretrained=lambda *args, **kwargs: "processor"),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            version=SimpleNamespace(cuda="simulated"),
            cuda=SimpleNamespace(
                is_available=lambda: available,
                OutOfMemoryError=SimulatedOOM,
                synchronize=lambda d: None,
                device_count=lambda: int(available),
                get_device_name=lambda i: "simulated GPU",
                empty_cache=lambda: None,
            ),
        ),
    )
    estimator = DepthEstimator(config)
    estimator.initialize()
    estimator.initialize()
    assert estimator.device_name == expected
    assert calls == ["load", expected]
    assert estimator.metadata["loading_info"]["missing_keys"] == []
    estimator.close()
    assert estimator.model is None


def test_depth_imports_do_not_activate_decisions_or_audio():
    import subprocess
    import sys

    source = (
        "import sightguard.pipeline.depth, sys; "
        "assert not any(m.startswith(('sightguard.risk', 'sightguard.decision', "
        "'sightguard.spatial', 'sightguard.ui')) for m in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", source], check=True, timeout=15)
