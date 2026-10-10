from pathlib import Path
from threading import Event

from fastapi.testclient import TestClient

from sightguard.ui.jobs import AnalysisJobs
from sightguard.ui.product_app import build_product_app


def client(root: Path) -> TestClient:
    for directory in ("configs", "models", "outputs", "datasets"):
        (root / directory).mkdir()
    (root / "configs/demo_guidance.yaml").write_text('schema_version: "1.0"\n')
    return TestClient(build_product_app(root), base_url="http://127.0.0.1")


def test_local_routes_and_invalid_upload_cleanup(tmp_path: Path) -> None:
    with client(tmp_path) as http:
        assert http.get("/").status_code == 200
        assert http.get("/api/status").json()["hardware_connected"] is False
        assert http.get("/api/jobs/missing/result").status_code == 404
        assert http.post("/api/example").status_code == 404
        assert http.post("/api/analyze", files={"video": ("test.avi", b"x")}).status_code == 400
        assert http.post("/api/analyze", files={"video": ("test.mp4", b"")}).status_code == 400
        assert http.get("/api/status").json()["busy"] is False
        assert not list((tmp_path / "outputs/console/uploads").glob("showcase-*.mp4"))
        assert http.get("/api/status", headers={"host": "evil.example"}).status_code == 403
        assert (
            http.post("/api/example", headers={"origin": "https://evil.example"}).status_code == 403
        )


def test_model_failure_is_reported_and_concurrent_upload_rejected(
    tmp_path: Path, monkeypatch
) -> None:
    started, release, finished = Event(), Event(), Event()

    def fail_run(*args, **kwargs):
        started.set()
        release.wait(5)
        finished.set()
        raise ValueError("model test failure")

    monkeypatch.setattr(AnalysisJobs, "run", fail_run)
    with client(tmp_path) as http:
        job = http.post("/api/analyze", files={"video": ("test.mp4", b"local fixture")}).json()[
            "id"
        ]
        assert started.wait(2)
        assert http.post("/api/analyze", files={"video": ("two.mp4", b"x")}).status_code == 409
        assert http.get(f"/api/jobs/{job}/result").status_code == 409
        release.set()
        assert finished.wait(2)
        import time

        deadline = time.monotonic() + 3
        while http.get("/api/status").json()["busy"] and time.monotonic() < deadline:
            time.sleep(0.01)
        assert http.get("/api/status").json()["busy"] is False
        status = http.get(f"/api/jobs/{job}").json()
        assert status["status"] == "failed"
        assert "model test failure" in status["message"]


def test_live_and_spatial_controls_validate_inputs_and_preserve_owner(tmp_path, monkeypatch):
    from sightguard.ui.live import LiveController, LiveView

    seen = []

    def fake_start(self, owner, *args):
        seen.append((owner, args))
        return LiveView(running=True, audio_status={"enabled": args[-2]})

    monkeypatch.setattr(LiveController, "start", fake_start)
    with client(tmp_path) as http:
        initial = http.get("/api/live/poll")
        assert initial.status_code == 200 and not initial.json()["running"]
        identity = http.cookies.get("sightguard_session")
        assert identity and "HttpOnly" in initial.headers["set-cookie"]
        response = http.post("/api/live/start", json={"camera": 1, "spatial_audio": True})
        assert response.status_code == 200 and response.json()["audio_status"]["enabled"]
        assert seen[0][0] == identity
        assert seen[0][1][-2:] == (True, 0.12)
        assert http.post("/api/live/start", json={"camera": -1}).status_code == 422
        assert http.post("/api/spatial/play", json={"azimuth": 181}).status_code == 422
        assert http.post("/api/live/navigation", json={"action": "play"}).status_code == 400
        assert http.post("/api/live/stop").status_code == 200
        assert http.post("/api/spatial/release").status_code == 200
