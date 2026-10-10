"""Local product showcase with asynchronous video analysis and synchronized replay."""

import json
import logging
import threading
import uuid
import webbrowser
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Annotated
from urllib.parse import urlparse

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from sightguard.decision.demo_guidance import DemoGuidance, load_demo_guidance
from sightguard.evaluation.dataset import sha256_file
from sightguard.ui.jobs import AnalysisJobs
from sightguard.ui.product_live import add_product_live
from sightguard.ui.services import ConsoleStore

LOGGER = logging.getLogger(__name__)
WEB_ROOT = Path(__file__).parent / "web"
MAX_UPLOAD = 500 * 1024 * 1024


def build_product_app(root: Path) -> FastAPI:
    """Expose project-contained artifacts and one model job at a time on localhost."""
    store = ConsoleStore(root)
    jobs = AnalysisJobs(store)
    config_path = store.config_path("demo_guidance.yaml")
    guidance_config = load_demo_guidance(config_path)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        try:
            yield
        finally:
            application.state.live.shutdown()
            application.state.spatial.shutdown()

    app = FastAPI(
        lifespan=lifespan, title="SightGuard · 智能辅助眼镜", docs_url=None, redoc_url=None
    )
    add_product_live(app, store)
    app.mount("/static", StaticFiles(directory=WEB_ROOT), name="static")
    state: dict[str, dict] = {}
    busy = threading.Lock()
    state_lock = threading.RLock()

    def update(identifier: str, **values: object) -> None:
        with state_lock:
            state[identifier].update(values)

    def prepare(identifier: str, run_id: str) -> None:
        import cv2

        view = store.load_analysis(run_id)
        media = jobs.playable_video(view.video_path)
        if not media:
            raise ValueError("该记录没有可播放的视频，请上传新视频。")
        capture = cv2.VideoCapture(media)
        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
        finally:
            capture.release()
        if not fps or fps <= 0:
            raise ValueError("无法读取视频帧率。")
        engine = DemoGuidance(guidance_config)
        frames = []
        for record in view.records:
            frames.append(
                {
                    "frame_index": record.frame_index,
                    "timestamp_ms": record.timestamp_ms,
                    "playback_ms": record.frame_index * 1000 / fps,
                    "status": record.status,
                    "detections": [asdict(item) for item in record.detections],
                    "suggestion": engine.step(record).to_dict(),
                }
            )
        result = {
            "schema_version": "1.0",
            "run_id": run_id,
            "fps": fps,
            "source_kind": view.source_kind,
            "warnings": list(view.warnings),
            "stats": view.stats,
            "frames": frames,
            "video_url": f"/api/jobs/{identifier}/video",
            "guidance_config": asdict(guidance_config),
            "guidance_config_sha256": sha256_file(config_path),
            "guidance_source_sha256": sha256_file(
                Path(__file__).parents[1] / "decision" / "demo_guidance.py"
            ),
            "time_basis": "annotated CFR playback frame index; original timestamps retained",
            "hardware_connected": False,
            "notice": "演示规则不是路径规划；不测距，不确认空闲区域可通行。",
        }
        result_path = jobs.workspace / f"showcase-{identifier}.json"
        result_path.write_text(
            json.dumps(result, ensure_ascii=False, allow_nan=False), encoding="utf-8"
        )
        update(
            identifier,
            status="completed",
            progress=1,
            message="分析完成，开始探索识别结果",
            result=result,
            video=media,
            result_path=str(result_path),
        )

    def execute(identifier: str, source: Path | None = None, run_id: str | None = None) -> None:
        try:
            if source is not None:
                run_id = jobs.run(
                    str(source),
                    "showcase.yaml",
                    "risk.yaml",
                    lambda amount, message: update(
                        identifier, progress=amount * 0.95, message=message
                    ),
                )
            if run_id is None:
                raise ValueError("没有可用的演示记录，请上传视频。")
            update(identifier, progress=0.96, message="准备浏览器视频和同步决策")
            prepare(identifier, run_id)
        except Exception as exc:
            LOGGER.exception("Showcase job %s failed", identifier)
            update(identifier, status="failed", message=f"分析失败：{exc}")
        finally:
            busy.release()

    def create_job() -> str:
        if not busy.acquire(blocking=False):
            raise HTTPException(409, "另一个视频正在分析，请等待完成后重试。")
        identifier = uuid.uuid4().hex
        with state_lock:
            state[identifier] = {"status": "running", "progress": 0, "message": "正在准备视频"}
        return identifier

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        from fastapi.responses import JSONResponse

        if request.url.hostname not in {"127.0.0.1", "localhost", "::1"}:
            return JSONResponse({"detail": "仅接受本机访问"}, status_code=403)
        origin = request.headers.get("origin")
        if request.method == "POST" and origin and urlparse(origin).netloc != request.url.netloc:
            return JSONResponse({"detail": "不接受跨站上传"}, status_code=403)
        return await call_next(request)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(WEB_ROOT / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/api/status")
    def status() -> dict:
        return {
            "service": "ready",
            "model_available": (store.root / "models/yolo26n.pt").is_file(),
            "hardware_connected": False,
            "busy": busy.locked(),
        }

    @app.post("/api/analyze")
    async def analyze(video: Annotated[UploadFile, File()]) -> dict:
        identifier = None
        source = None
        try:
            if Path(video.filename or "").suffix.lower() != ".mp4":
                raise HTTPException(400, "请选择 MP4 视频。")
            identifier = create_job()
            source = jobs.uploads / f"showcase-{identifier}.mp4"
            size = 0
            with source.open("xb") as stream:
                while chunk := await video.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_UPLOAD:
                        raise HTTPException(413, "视频不能超过 500 MB。")
                    stream.write(chunk)
            if size == 0:
                raise HTTPException(400, "视频文件为空。")
            threading.Thread(target=execute, args=(identifier, source), daemon=True).start()
            return {"id": identifier}
        except BaseException:
            if identifier is not None:
                with state_lock:
                    state.pop(identifier, None)
                if source is not None:
                    source.unlink(missing_ok=True)
                busy.release()
            raise
        finally:
            await video.close()

    @app.post("/api/example")
    def example() -> dict:
        run_ids = store.analysis_runs()
        run_id = (
            "tracking_verified" if "tracking_verified" in run_ids else next(iter(run_ids), None)
        )
        if run_id is None:
            raise HTTPException(404, "本机没有已保存的示例，请上传 MP4 视频。")
        identifier = create_job()
        threading.Thread(
            target=execute, kwargs={"identifier": identifier, "run_id": run_id}, daemon=True
        ).start()
        return {"id": identifier}

    def find(identifier: str) -> dict:
        with state_lock:
            value = state.get(identifier)
            if value is None:
                raise HTTPException(404, "任务不存在，服务可能已重新启动。")
            return dict(value)

    @app.get("/api/jobs/{identifier}")
    def job_status(identifier: str) -> dict:
        value = find(identifier)
        return {key: value[key] for key in ("status", "progress", "message")}

    @app.get("/api/jobs/{identifier}/result")
    def job_result(identifier: str) -> dict:
        value = find(identifier)
        if value["status"] != "completed":
            raise HTTPException(409, value["message"])
        return value["result"]

    @app.get("/api/jobs/{identifier}/video")
    def job_video(identifier: str) -> FileResponse:
        value = find(identifier)
        if value["status"] != "completed":
            raise HTTPException(409, "视频尚未准备好")
        return FileResponse(value["video"], media_type="video/mp4")

    @app.get("/api/jobs/{identifier}/download")
    def download(identifier: str) -> FileResponse:
        value = find(identifier)
        if value["status"] != "completed":
            raise HTTPException(409, "结果尚未准备好")
        return FileResponse(value["result_path"], filename="SightGuard-demo-result.json")

    return app


def launch_product_ui(root: Path, port: int = 7860, *, inbrowser: bool = False) -> None:
    """Launch the local showcase; preserve the separate research console entry."""
    import uvicorn

    if not 1 <= port <= 65535:
        raise ValueError("端口必须在 1–65535 之间")
    app = build_product_app(root)
    if inbrowser:
        threading.Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    uvicorn.run(app, host="127.0.0.1", port=port)
