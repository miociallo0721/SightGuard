"""Put a real photograph on the AVD wall; no app detector output is injected.

Requires grpcio/grpcio-tools in the export environment. Run after start_avd.ps1
and after starting perception in the app. Emulator-controller proto is supplied
by the installed Android SDK; credentials are read locally and never printed.
"""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--empty-view", action="store_true",
        help="Aim the actual virtual camera at the untextured room ceiling.",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    sdk = Path(os.environ.get("ANDROID_HOME", "D:/Android/Sdk"))
    generated = Path(os.environ.get("TEMP", "D:/Temp/SightGuard")) / "emulator-grpc"
    generated.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable, "-m", "grpc_tools.protoc", "-I", str(sdk / "emulator/lib"),
            f"--python_out={generated}", f"--grpc_python_out={generated}",
            str(sdk / "emulator/lib/emulator_controller.proto"),
        ],
        check=True,
    )
    adb = str(sdk / "platform-tools/adb.exe")
    subprocess.run([adb, "-s", "emulator-5554", "emu", "grpc", "8554"], check=True)
    photo = root / "android/validation/validation.jpg"
    subprocess.run(
        [adb, "-s", "emulator-5554", "emu", "virtualscene-image", "wall", photo.as_posix()],
        check=True,
    )
    running = Path(os.environ["LOCALAPPDATA"]) / "Temp/avd/running"
    configs = []
    for path in running.glob("pid_*.ini"):
        values = dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)
        if values.get("port.adb") == "5555" and "medium_phone" in (
            values.get("avd.id"), values.get("avd.name")
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
                [0, -90, 0] if args.empty_view else [180, -30, 180],
            ),
        ]:
            controller.setPhysicalModel(
                proto.PhysicalModelValue(
                    target=target, value=proto.ParameterValue(data=values), interpolation=1,
                ),
                metadata=auth, timeout=3,
            )
    print(
        "Actual virtual camera aimed at ceiling." if args.empty_view
        else "Real photo poster configured; inspect actual CameraX detections in the app."
    )


if __name__ == "__main__":
    main()
