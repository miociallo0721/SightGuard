"""Capture emulator output as stereo PCM, then report actual channel RMS.

Start this before triggering one direction's speech; use separate captures for
left/center/right. No playback injection or claimed headphone/HRTF validation.
Requires grpcio and grpcio-tools; all generated files stay under D: by default.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import struct
import subprocess
import sys
import wave
from pathlib import Path


def channel_rms(pcm: bytes) -> tuple[float, float]:
    """RMS of little-endian signed 16-bit interleaved stereo frames."""
    if len(pcm) % 4:
        raise ValueError("PCM must contain complete stereo frames")
    count = len(pcm) // 4
    if not count:
        return 0.0, 0.0
    left = right = 0
    for l_sample, r_sample in struct.iter_unpack("<hh", pcm):
        left += l_sample * l_sample
        right += r_sample * r_sample
    return math.sqrt(left / count), math.sqrt(right / count)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--adb-port", type=int, default=5555)
    parser.add_argument("--metadata-dir", type=Path,
                        default=Path("D:/Temp/SightGuard/emulator-control/running"))
    parser.add_argument("--expected", choices=("left", "center", "right"))
    args = parser.parse_args()
    if not 0 < args.seconds <= 120:
        parser.error("seconds must be in (0, 120]")
    sdk = Path(os.environ.get("ANDROID_HOME", "D:/Android/Sdk"))
    generated = Path("D:/Temp/SightGuard/emulator-audio-grpc")
    generated.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        sys.executable, "-m", "grpc_tools.protoc", "-I", str(sdk / "emulator/lib"),
        f"--python_out={generated}", f"--grpc_python_out={generated}",
        str(sdk / "emulator/lib/emulator_controller.proto"),
    ], check=True)
    running = args.metadata_dir
    candidates = []
    for path in running.glob("pid_*.ini"):
        values = dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)
        if values.get("port.adb") == str(args.adb_port) and "grpc.port" in values:
            candidates.append((path.stat().st_mtime, values))
    if not candidates:
        raise RuntimeError("Running emulator gRPC metadata not found")
    config = max(candidates, key=lambda item: item[0])[1]
    sys.path.insert(0, str(generated))
    grpc = importlib.import_module("grpc")
    proto = importlib.import_module("emulator_controller_pb2")
    service = importlib.import_module("emulator_controller_pb2_grpc")
    auth = (("authorization", "Bearer " + config["grpc.token"]),)
    pcm = bytearray()
    timestamps = []
    with grpc.insecure_channel("localhost:" + config["grpc.port"]) as channel:
        controller = service.EmulatorControllerStub(channel)
        stream = controller.streamAudio(proto.AudioFormat(
            samplingRate=44100, channels=proto.AudioFormat.Stereo,
            format=proto.AudioFormat.AUD_FMT_S16,
        ), metadata=auth, timeout=args.seconds)
        try:
            for packet in stream:
                if (
                    packet.format.channels != proto.AudioFormat.Stereo
                    or packet.format.format != proto.AudioFormat.AUD_FMT_S16
                ):
                    raise RuntimeError("Emulator returned incompatible PCM format")
                pcm.extend(packet.audio)
                timestamps.append(packet.timestamp)
        except grpc.RpcError as error:
            if error.code() != grpc.StatusCode.DEADLINE_EXCEEDED:
                raise RuntimeError(f"Audio capture failed: {error.code().name}") from None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(args.output), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(44100)
        output.writeframes(pcm)
    left, right = channel_rms(pcm)
    audible = max(left, right) > 1
    matched = None
    if args.expected:
        matched = audible and {
            "left": left > 2 * right,
            "right": right > 2 * left,
            "center": abs(left - right) <= .15 * max(left, right),
        }[args.expected]
    report = {
        "schema_version": 1, "capture": "emulator_streamAudio", "sample_rate": 44100,
        "frames": len(pcm) // 4, "left_rms": left, "right_rms": right,
        "audible": audible, "expected": args.expected, "channel_check_passed": matched,
        "first_packet_timestamp_us": timestamps[0] if timestamps else None,
        "last_packet_timestamp_us": timestamps[-1] if timestamps else None,
        "limitation": "Emulator output only; not headphone localization or HRTF validation",
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))
    if matched is False or not audible:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
