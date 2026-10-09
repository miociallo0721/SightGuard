import json
from pathlib import Path

import pytest

from sightguard.config import PerceptionConfig
from sightguard.pipeline.video import analyze_video
from sightguard.ui.jobs import AnalysisJobs
from sightguard.ui.services import ConsoleDataError, ConsoleStore


def test_pipeline_progress_reports_actual_frames_without_changing_results(
    video: Path, tmp_path: Path
) -> None:
    class EmptyPerceptor:
        def __init__(self, _config):
            pass

        def process(self, _frame):
            return ()

    observations = []
    metadata = analyze_video(
        video,
        tmp_path / "progress",
        PerceptionConfig(),
        perceptor_factory=EmptyPerceptor,
        on_progress=lambda n, total, timestamp: observations.append((n, total, timestamp)),
    )
    assert metadata["frames_processed"] == 5
    assert [item[0] for item in observations] == [1, 2, 3, 4, 5]
    assert all(item[1] == 5 for item in observations)
    assert [item[2] for item in observations] == pytest.approx(
        [0, 100, 200, 300, 400], rel=0, abs=1e-7
    )


def test_job_rejects_unuploaded_paths_and_preserves_originals(tmp_path: Path, video: Path) -> None:
    jobs = AnalysisJobs(ConsoleStore(tmp_path))
    before = video.read_bytes()
    with pytest.raises(ConsoleDataError, match="上传"):
        jobs.run(str(video), "default.yaml", "risk.yaml")
    assert video.read_bytes() == before
    assert not list(jobs.workspace.glob("analysis-*"))


def test_failed_job_metadata_and_new_run_isolation(tmp_path: Path, monkeypatch) -> None:
    import sightguard.ui.jobs as module

    (tmp_path / "configs").mkdir()
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "local.pt").write_bytes(b"local test model placeholder")
    (tmp_path / "configs" / "default.yaml").write_text("model_path: models/local.pt\n")
    (tmp_path / "configs" / "risk.yaml").write_text("schema_version: '2.0'\n")
    jobs = AnalysisJobs(ConsoleStore(tmp_path))
    source = jobs.uploads / "test.mp4"
    source.write_bytes(b"test upload")

    def fail(*_args, **_kwargs):
        raise RuntimeError("inference failed")

    monkeypatch.setattr(module, "analyze_video", fail)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="inference failed"):
            jobs.run(str(source), "default.yaml", "risk.yaml")
    outputs = list(jobs.workspace.glob("analysis-*"))
    assert len(outputs) == 2
    for output in outputs:
        metadata = json.loads((output / "console_metadata.json").read_text())
        assert metadata["status"] == "failed"
        assert metadata["ground_truth_available"] is False
        assert (output / "input.mp4").read_bytes() == b"test upload"
    assert source.read_bytes() == b"test upload"
