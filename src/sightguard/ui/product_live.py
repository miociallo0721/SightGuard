"""Product-page controls sharing the existing live and native-audio adapters."""

from dataclasses import asdict
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt

from sightguard.ui.live import LiveController, camera_choices
from sightguard.ui.services import ConsoleStore
from sightguard.ui.spatial_lab import SpatialLabController


class LiveStart(BaseModel):
    model_config = ConfigDict(extra="forbid")
    camera: StrictInt = Field(default=0, ge=0, le=100)
    spatial_audio: StrictBool = False
    audio_gain: float = Field(default=0.12, ge=0, le=1, allow_inf_nan=False)


class Navigate(BaseModel):
    action: str
    angle: float = Field(default=0, ge=-180, le=180, allow_inf_nan=False)


class IntelligenceAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["rule", "laya", "vision", "hybrid"]
    auxiliary: Literal["laya", "vision"] = "laya"
    action: Literal["select", "load", "unload"] = "select"


class TestSound(BaseModel):
    model_config = ConfigDict(extra="forbid")
    azimuth: float = Field(ge=-180, le=180, allow_inf_nan=False)
    elevation: float = Field(default=0, ge=-45, le=45, allow_inf_nan=False)
    gain: float = Field(default=0.12, ge=0, le=1, allow_inf_nan=False)
    cue_type: str = "system_test"
    device: str | None = None
    request_hrtf: StrictBool = True


def add_product_live(app: FastAPI, store: ConsoleStore) -> None:
    live = LiveController(store)
    spatial = SpatialLabController(store.root)
    app.state.live = live
    app.state.spatial = spatial

    def owner(request: Request, response: Response) -> str:
        value = request.cookies.get("sightguard_session", "")
        if len(value) != 32 or any(char not in "0123456789abcdef" for char in value):
            value = uuid4().hex
            response.set_cookie("sightguard_session", value, httponly=True, samesite="strict")
        return value

    def view(value) -> dict:
        result = asdict(value)
        for key in ("events", "alerts"):
            table = result.get(key)
            if table is not None:
                # Pandas NaN is not valid JSON. Preserve undefined values as null.
                result[key] = table.astype(object).where(table.notna(), None).to_dict("records")
        return result

    def call(method, *args) -> dict:
        try:
            return view(method(*args))
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/live/cameras")
    def cameras() -> dict:
        choices, preferred = camera_choices()
        return {"choices": choices, "preferred": preferred}

    @app.post("/api/live/start")
    def start(body: LiveStart, request: Request, response: Response) -> dict:
        identity = owner(request, response)
        spatial.release(identity)
        return call(
            live.start,
            identity,
            body.camera,
            "auto",
            640,
            480,
            640,
            "default.yaml",
            "risk.yaml",
            body.spatial_audio,
            body.audio_gain,
        )

    @app.get("/api/live/poll")
    def poll(request: Request, response: Response) -> dict:
        return call(live.poll, owner(request, response))

    @app.post("/api/live/stop")
    def stop(request: Request, response: Response) -> dict:
        return call(live.stop, owner(request, response))

    @app.post("/api/live/mute")
    def mute(request: Request, response: Response) -> dict:
        return call(live.toggle_audio, owner(request, response))

    @app.post("/api/live/intelligence")
    def intelligence(body: IntelligenceAction, request: Request, response: Response) -> dict:
        return call(
            live.intelligence,
            owner(request, response),
            body.mode,
            body.auxiliary,
            body.action,
        )

    @app.post("/api/live/navigation")
    def navigate(body: Navigate, request: Request, response: Response) -> dict:
        return call(live.navigation, owner(request, response), body.action, body.angle)

    @app.post("/api/spatial/play")
    def play(body: TestSound, request: Request, response: Response) -> dict:
        identity = owner(request, response)
        live.stop(identity)
        return call(
            spatial.play,
            identity,
            body.azimuth,
            body.elevation,
            body.gain,
            body.cue_type,
            body.device,
            body.request_hrtf,
        )

    @app.get("/api/spatial/poll")
    def spatial_poll(request: Request, response: Response) -> dict:
        return call(spatial.poll, owner(request, response))

    @app.post("/api/spatial/stop")
    def spatial_stop(request: Request, response: Response) -> dict:
        return call(spatial.stop, owner(request, response))

    @app.post("/api/spatial/release")
    def spatial_release(request: Request, response: Response) -> dict:
        return call(spatial.release, owner(request, response))
