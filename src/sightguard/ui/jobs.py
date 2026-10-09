"""New analysis runs and browser media copies; original research artifacts stay untouched."""

import json
import shutil
import subprocess
import threading
import uuid
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from sightguard.config import load_config
from sightguard.evaluation.dataset import sha256_file
from sightguard.pipeline.assess import assess_tracks
from sightguard.pipeline.video import analyze_video
from sightguard.risk.config import load_risk_config
from sightguard.ui.services import ConsoleDataError, ConsoleStore

ProgressCallback = Callable[[float, str], None]


class AnalysisJobs:
    def __init__(self, store: ConsoleStore) -> None:
        self.store = store
        self.workspace = store.root / "outputs" / "console"
        self.uploads = self.workspace / "uploads"
        self.cache = self.workspace / "media"
        self.lock = threading.Lock()
        for directory in (self.workspace, self.uploads, self.cache):
            resolved = directory.resolve()
            if not resolved.is_relative_to(store.root / "outputs"):
                raise ConsoleDataError("Console output directory escapes project outputs")
            directory.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        upload_path: str,
        perception_config_id: str,
        risk_config_id: str,
        progress: ProgressCallback | None = None,
    ) -> str:
        if not upload_path:
            raise ConsoleDataError("请先上传本地 MP4 文件")
        source = Path(upload_path).resolve()
        if (
            not self.uploads.resolve().is_relative_to(self.store.root / "outputs")
            or not source.is_relative_to(self.uploads.resolve())
            or not source.is_file()
        ):
            raise ConsoleDataError("视频必须来自本控制台的上传组件")
        if source.suffix.lower() != ".mp4" or source.stat().st_size > 500 * 1024 * 1024:
            raise ConsoleDataError("仅接受不超过 500 MiB 的 MP4")
        config = load_config(self.store.config_path(perception_config_id))
        model = (self.store.root / config.model_path).resolve()
        if not model.is_relative_to(self.store.root / "models") or not model.is_file():
            raise ConsoleDataError("配置必须指定 models 目录中的本地权重；不会自动下载")
        config = replace(config, model_path=str(model))
        risk_config = load_risk_config(self.store.config_path(risk_config_id))
        identifier = f"analysis-{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:12]}"
        output = self.workspace / identifier
        output.mkdir(exist_ok=False)
        provenance = {
            "schema_version": "4.0",
            "run_id": identifier,
            "status": "running",
            "source_kind": "unknown",
            "ground_truth_available": False,
            "note": "Uploaded footage; provenance/accuracy not independently annotated",
        }
        try:
            input_copy = output / "input.mp4"
            shutil.copyfile(source, input_copy)
            if progress:
                progress(0, "读取 MP4，初始化本地模型")

            def observed(count: int, total: int | None, timestamp_ms: float) -> None:
                if progress and (count == 1 or count % 5 == 0 or count == total):
                    progress(
                        0.85 * min(count / total, 1) if total else 0,
                        f"检测/跟踪：{count}/{total or '?'} 帧，视频时间 {timestamp_ms:.0f} ms",
                    )

            analyze_video(input_copy, output / "perception", config, on_progress=observed)
            if progress:
                progress(0.88, "视觉分析结束；重放风险评估与事件决策")
            assess_tracks(
                output / "perception" / "tracks.jsonl",
                output / "assessment",
                risk_config,
                stream_id=identifier,
            )
            provenance["status"] = "completed"
            if progress:
                progress(1, "完成：检测、跟踪与风险结果已保存")
        except Exception as exc:
            provenance.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            (output / "console_metadata.json").write_text(
                json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        return (output / "perception").relative_to(self.store.root / "outputs").as_posix()

    def playable_video(self, source: Path | None) -> str | None:
        if source is None:
            return None
        source = source.resolve()
        permitted = (
            self.store.root / "outputs",
            self.store.root / "datasets" / "videos",
            self.store.root / "examples",
        )
        if not any(source.is_relative_to(area) for area in permitted) or not source.is_file():
            raise ConsoleDataError("播放器仅接受已校验的输出视频")
        import imageio_ffmpeg

        target = self.cache / f"{sha256_file(source)}.mp4"
        with self.lock:
            if not target.exists():
                temporary = self.cache / f"{uuid.uuid4().hex}.mp4"
                try:
                    result = subprocess.run(
                        [
                            imageio_ffmpeg.get_ffmpeg_exe(),
                            "-nostdin",
                            "-v",
                            "error",
                            "-n",
                            "-i",
                            str(source),
                            "-an",
                            "-c:v",
                            "libx264",
                            "-pix_fmt",
                            "yuv420p",
                            "-movflags",
                            "+faststart",
                            str(temporary),
                        ],
                        capture_output=True,
                        text=True,
                        timeout=600,
                        check=False,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
                    if result.returncode:
                        raise ConsoleDataError(f"浏览器视频副本编码失败：{result.stderr[-500:]}")
                    temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)
        return str(target)
