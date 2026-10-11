"""Put a real photograph on the AVD wall; no app detector output is injected.

Requires grpcio/grpcio-tools in the export environment. Run after start_avd.ps1
and after starting perception in the app. Emulator-controller proto is supplied
by the installed Android SDK; credentials are read locally and never printed.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
import wave
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--empty-view",
        action="store_true",
        help="Aim the actual virtual camera at the untextured room ceiling.",
    )
    parser.add_argument(
        "--photo", type=Path, help="Existing real photograph to place on the virtual wall."
    )
    parser.add_argument(
        "--rotation",
        nargs=3,
        type=float,
        help="Explicit emulator camera rotation for turn/freshness validation.",
    )
    parser.add_argument(
        "--rapid-turn",
        action="store_true",
        help="Exercise real virtual-camera/IMU turns between alternate headings.",
    )
    parser.add_argument(
        "--motion-suite",
        type=Path,
        help="Record controlled virtual-camera scenarios, never physical-phone acceptance.",
    )
    parser.add_argument(
        "--record-audio", type=Path, help="Capture actual emulator output PCM to WAV."
    )
    parser.add_argument("--seconds", type=float, default=60)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    sdk = Path(os.environ.get("ANDROID_HOME", "D:/Android/Sdk"))
    generated = Path(os.environ.get("TEMP", "D:/Temp/SightGuard")) / "emulator-grpc"
    generated.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            "-I",
            str(sdk / "emulator/lib"),
            f"--python_out={generated}",
            f"--grpc_python_out={generated}",
            str(sdk / "emulator/lib/emulator_controller.proto"),
        ],
        check=True,
    )
    adb = str(sdk / "platform-tools/adb.exe")
    subprocess.run([adb, "-s", "emulator-5554", "emu", "grpc", "8554"], check=True)
    photo = (args.photo or root / "android/validation/validation.jpg").resolve()
    if not photo.is_file():
        raise FileNotFoundError(photo)
    subprocess.run(
        [adb, "-s", "emulator-5554", "emu", "virtualscene-image", "wall", photo.as_posix()],
        check=True,
    )
    running = Path(os.environ["LOCALAPPDATA"]) / "Temp/avd/running"
    configs = []
    for path in running.glob("pid_*.ini"):
        values = dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)
        if values.get("port.adb") == "5555" and "medium_phone" in (
            values.get("avd.id"),
            values.get("avd.name"),
        ):
            configs.append((path.stat().st_mtime, values))
    if not configs:
        raise RuntimeError("Running medium_phone gRPC metadata not found")
    config = max(configs, key=lambda item: item[0])[1]
    sys.path.insert(0, str(generated))
    grpc = importlib.import_module("grpc")
    proto = importlib.import_module("emulator_controller_pb2")
    service = importlib.import_module("emulator_controller_pb2_grpc")
    with grpc.insecure_channel("localhost:" + config["grpc.port"]) as channel:
        controller = service.EmulatorControllerStub(channel)
        auth = (("authorization", "Bearer " + config["grpc.token"]),)
        for target, values in [
            (proto.PhysicalModelValue.POSITION, [-1.807, 0.32, 3.584]),
            (
                proto.PhysicalModelValue.ROTATION,
                [0, -90, 0] if args.empty_view else (args.rotation or [180, -30, 180]),
            ),
        ]:
            controller.setPhysicalModel(
                proto.PhysicalModelValue(
                    target=target,
                    value=proto.ParameterValue(data=values),
                    interpolation=1,
                ),
                metadata=auth,
                timeout=3,
            )
        if args.rapid_turn:
            for angles in [[180, -30, 180], [0, -90, 0]] * 4:
                controller.setPhysicalModel(
                    proto.PhysicalModelValue(
                        target=proto.PhysicalModelValue.ROTATION,
                        value=proto.ParameterValue(data=angles),
                        interpolation=proto.PhysicalModelValue.SMOOTH,
                    ),
                    metadata=auth,
                    timeout=3,
                )
                time.sleep(0.25)
        if args.record_audio:
            args.record_audio.parent.mkdir(parents=True, exist_ok=True)
            stream = controller.streamAudio(
                proto.AudioFormat(
                    samplingRate=24000,
                    channels=proto.AudioFormat.Mono,
                    format=proto.AudioFormat.AUD_FMT_S16,
                ),
                metadata=auth,
                timeout=args.seconds + 2,
            )
            started = time.monotonic()
            with wave.open(str(args.record_audio), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(24000)
                try:
                    for packet in stream:
                        output.writeframes(packet.audio)
                        if time.monotonic() - started >= args.seconds:
                            stream.cancel()
                            break
                except grpc.RpcError as error:
                    if error.code() != grpc.StatusCode.DEADLINE_EXCEEDED:
                        raise
            print("Captured actual emulator output audio to", args.record_audio)
        if args.motion_suite:
            args.motion_suite.mkdir(parents=True, exist_ok=True)
            cases = [
                ("stationary", [0, 0, 0], [0, 0, 0], 8),
                ("slow_z_positive", [0, 0, 0.2], [0, 0, 0], 10),
                ("yaw_only", [0, 0, 0], [0, 8, 0], 8),
                ("pitch_only", [0, 0, 0], [8, 0, 0], 8),
                ("lateral", [0.2, 0, 0], [0, 0, 0], 10),
                ("z_negative", [0, 0, -0.2], [0, 0, 0], 10),
                ("stopped_facing_wall", [0, 0, 0], [0, 0, 0], 8),
                ("low_texture_ceiling", [0, 0, 0], None, 8),
            ]
            reports = []
            for name, delta, angles, seconds in cases:
                subprocess.run([adb, "-s", "emulator-5554", "logcat", "-c"], check=True)
                began = time.time()
                for i in range(seconds * 5):
                    f = i / max(1, seconds * 5 - 1)
                    position = [-1.807 + delta[0] * f, 0.32 + delta[1] * f, 3.584 + delta[2] * f]
                    rotation = (
                        [0, -90, 0]
                        if angles is None
                        else [180 + angles[0] * f, -30 + angles[1] * f, 180 + angles[2] * f]
                    )
                    for target, value in [
                        (proto.PhysicalModelValue.POSITION, position),
                        (proto.PhysicalModelValue.ROTATION, rotation),
                    ]:
                        controller.setPhysicalModel(
                            proto.PhysicalModelValue(
                                target=target,
                                value=proto.ParameterValue(data=value),
                                interpolation=proto.PhysicalModelValue.SMOOTH,
                            ),
                            metadata=auth,
                            timeout=3,
                        )
                    time.sleep(0.2)
                logs = subprocess.check_output(
                    [
                        adb,
                        "-s",
                        "emulator-5554",
                        "logcat",
                        "-d",
                        "-v",
                        "epoch",
                        "SG_WALL_SHADOW:I",
                        "SG_ENV:I",
                        "SG_FRAME:I",
                        "SG_AUDIO:I",
                        "SG_SPACE:I",
                        "SG_IMU:I",
                        "*:S",
                    ]
                )
                (args.motion_suite / f"{name}.log").write_bytes(logs)
                reports.append(
                    dict(
                        case=name,
                        start_epoch_s=began,
                        end_epoch_s=time.time(),
                        virtual_position_delta=delta,
                        virtual_rotation_delta=angles,
                        source=(
                            "actual virtualscene CameraX + emulated IMU, 2D real-photo poster; "
                            "physical translation/body motion NOT validated"
                        ),
                    )
                )
                print("Virtual scenario recorded:", name, flush=True)
            (args.motion_suite / "scenarios.json").write_text(
                json.dumps(
                    dict(
                        schema_version=1,
                        cases=reports,
                        physical_phone_acceptance=False,
                        unavailable=[
                            "controlled human approach",
                            "physical low-texture white wall",
                            "real far wall plus independently moving near object",
                        ],
                    ),
                    indent=2,
                ),
                encoding="utf-8",
            )
    print(
        "Rapid virtual-camera/IMU turns completed; inspect freshness logs."
        if args.rapid_turn
        else "Actual virtual camera aimed at ceiling."
        if args.empty_view
        else "Real photo poster configured; inspect actual CameraX detections in the app."
    )


if __name__ == "__main__":
    main()
