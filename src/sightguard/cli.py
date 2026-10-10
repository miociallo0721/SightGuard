"""CLI for explicit local-weight analysis and independent replay validation."""

import argparse
import logging
from dataclasses import replace
from pathlib import Path

from sightguard import __version__
from sightguard.config import load_config
from sightguard.data import replay_records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sightguard")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    analyze = sub.add_parser("analyze", help="Analyze a local MP4 with YOLO26n and BoT-SORT")
    analyze.add_argument("--input", type=Path, required=True)
    analyze.add_argument("--output-dir", type=Path, required=True)
    analyze.add_argument("--config", type=Path)
    analyze.add_argument("--model", help="Override local model path")
    analyze.add_argument("--device", help="cpu or compatible GPU device, e.g. 0")
    replay = sub.add_parser("replay", help="Validate and summarize saved perception JSONL")
    replay.add_argument("--input", type=Path, required=True)
    assess = sub.add_parser("assess", help="Assess saved tracks without rerunning perception")
    assess.add_argument("--tracks", type=Path, required=True)
    assess.add_argument("--config", type=Path)
    assess.add_argument("--output-dir", type=Path, required=True)
    assess.add_argument(
        "--stream-id", help="Stable video identity, e.g. clip-01; defaults to path hash"
    )
    for command in ("benchmark", "validate-labels", "seal-labels", "freeze-protocol"):
        command_parser = sub.add_parser(command, help=f"Phase 3: {command}")
        command_parser.add_argument("--dataset", type=Path, required=True)
        if command in ("benchmark", "freeze-protocol"):
            command_parser.add_argument("--config", type=Path)
            command_parser.add_argument("--risk-config", type=Path)
        if command == "benchmark":
            command_parser.add_argument("--output-dir", type=Path, required=True)
            command_parser.add_argument("--split", choices=("development", "test"), default="test")
            command_parser.add_argument("--data-kind", choices=("real", "synthetic"))
            command_parser.add_argument("--protocol", type=Path)
            command_parser.add_argument(
                "--strategies", nargs="+", choices=("A", "B", "C"), default=["A", "B", "C"]
            )
        elif command == "freeze-protocol":
            command_parser.add_argument("--output", type=Path, required=True)
    importing = sub.add_parser("import-video", help="Reference raw video and create draft labels")
    for flag in ("video", "tracks", "dataset-dir"):
        importing.add_argument(f"--{flag}", type=Path, required=True)
    for flag in ("video-id", "source-group-id", "scene"):
        importing.add_argument(f"--{flag}", required=True)
    importing.add_argument("--split", choices=("development", "test"), default="development")
    ui = sub.add_parser("ui", help="Launch local SightGuard Research Console")
    ui.add_argument("--port", type=int, default=7860)
    ui.add_argument("--open-browser", action="store_true")
    ui.add_argument("--research", action="store_true", help="Open the Gradio research console")
    spatial = sub.add_parser("spatial-demo", help="Local OpenAL HRTF listening lab (no camera)")
    spatial.add_argument("--port", type=int, default=7862)
    spatial.add_argument("--open-browser", action="store_true")
    live = sub.add_parser("live", help="OpenCV live camera risk demo (Q quit, H details)")
    live.add_argument("--camera", type=int)
    live.add_argument("--device", help="auto, cpu, cuda, cuda:0 or GPU index; default auto")
    live.add_argument("--width", type=int)
    live.add_argument("--height", type=int)
    live.add_argument("--imgsz", type=int)
    live.add_argument("--model", help="Override existing local model path; no download")
    live.add_argument("--config", type=Path, help="Perception YAML")
    live.add_argument("--risk-config", type=Path, help="Existing Phase 2 risk YAML")
    live.add_argument("--live-config", type=Path, help="Camera/staleness/display YAML")
    live.add_argument("--duration", type=float, help="Optional auto-exit duration in seconds")
    live.add_argument("--spatial-audio", action="store_true", help="Opt-in native PC spatial audio")
    live.add_argument("--spatial-config", type=Path, help="Audio YAML; does not change risk rules")
    live.add_argument("--audio-device", help="Exact OpenAL playback device name")
    live.add_argument("--audio-volume", type=float, help="Audio gain in 0..1")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        if args.command == "spatial-demo":
            from sightguard.spatial.app import launch_spatial_demo

            launch_spatial_demo(Path.cwd(), args.port, inbrowser=args.open_browser)
        elif args.command == "live":
            import math

            from sightguard.live.app import run_live
            from sightguard.live.config import load_live_config, resolve_device
            from sightguard.risk.config import load_risk_config

            if args.duration is not None and (
                not math.isfinite(args.duration) or args.duration <= 0
            ):
                raise ValueError("duration must be finite and positive")
            perception = load_config(args.config)
            device = args.device or (perception.device if args.config else "auto")
            overrides = {"device": resolve_device(device)}
            if args.model is not None:
                overrides["model_path"] = args.model
            if args.imgsz is not None:
                overrides["image_size"] = args.imgsz
            live_config = load_live_config(args.live_config)
            camera_overrides = {
                name: getattr(args, name)
                for name in ("camera", "width", "height")
                if getattr(args, name) is not None
            }
            spatial_options = {}
            if args.spatial_audio:
                from sightguard.spatial.config import load_spatial_config

                default_audio = Path("configs/spatial.yaml")
                spatial_config = load_spatial_config(
                    args.spatial_config or (default_audio if default_audio.is_file() else None)
                )
                audio_overrides = {}
                if args.audio_device is not None:
                    audio_overrides["device_name"] = args.audio_device
                if args.audio_volume is not None:
                    audio_overrides["gain"] = args.audio_volume
                spatial_options["spatial_config"] = replace(spatial_config, **audio_overrides)
            run_live(
                replace(perception, **overrides),
                load_risk_config(args.risk_config),
                replace(live_config, **camera_overrides),
                duration_s=args.duration,
                **spatial_options,
            )
        elif args.command == "ui":
            if args.research:
                from sightguard.ui.app import launch_ui

                launch_ui(Path.cwd(), args.port, inbrowser=args.open_browser)
            else:
                from sightguard.ui.product_app import launch_product_ui

                launch_product_ui(Path.cwd(), args.port, inbrowser=args.open_browser)
        elif args.command == "import-video":
            from sightguard.evaluation.importing import import_video

            label = import_video(
                args.video,
                args.tracks,
                args.dataset_dir,
                video_id=args.video_id,
                source_group_id=args.source_group_id,
                scene=args.scene,
                split=args.split,
            )
            logging.info(
                "Draft annotation created: %s; review independently before evaluation", label
            )
        elif args.command == "seal-labels":
            from sightguard.evaluation.importing import seal_labels

            seal_labels(args.dataset)
            logging.info("Reviewed annotation hashes sealed")
        elif args.command == "validate-labels":
            from sightguard.evaluation.dataset import load_dataset

            videos = load_dataset(args.dataset, check_videos=True)
            logging.info("Validated %d reviewed videos and split/provenance evidence", len(videos))
        elif args.command in ("benchmark", "freeze-protocol"):
            from sightguard.evaluation.config import load_benchmark_config
            from sightguard.risk.config import load_risk_config

            benchmark_config = load_benchmark_config(args.config)
            risk_config = load_risk_config(args.risk_config)
            if args.command == "freeze-protocol":
                from sightguard.evaluation.protocol import freeze_protocol

                freeze_protocol(args.dataset, benchmark_config, risk_config, args.output)
                logging.info("Frozen protocol: %s", args.output)
            else:
                from sightguard.evaluation.runner import benchmark

                benchmark(
                    args.dataset,
                    args.output_dir,
                    benchmark_config,
                    risk_config,
                    split=args.split,
                    protocol=args.protocol,
                    data_kind=args.data_kind,
                    strategies=tuple(args.strategies),
                )
                logging.info("Benchmark completed: %s", args.output_dir)
        elif args.command == "replay":
            frames = detections = failures = 0
            for record in replay_records(args.input):
                frames += 1
                detections += len(record.detections)
                failures += record.status == "processing_error"
            logging.info(
                "Replay: %d frames, %d detections, %d failed frames", frames, detections, failures
            )
        elif args.command == "assess":
            from sightguard.pipeline.assess import assess_tracks
            from sightguard.risk.config import load_risk_config

            assess_tracks(
                args.tracks,
                args.output_dir,
                load_risk_config(args.config),
                stream_id=args.stream_id,
            )
        else:
            from sightguard.pipeline.video import analyze_video

            config = load_config(args.config)
            overrides = {
                key: value
                for key, value in (("model_path", args.model), ("device", args.device))
                if value is not None
            }
            analyze_video(args.input, args.output_dir, replace(config, **overrides))
    except Exception as exc:
        logging.error("%s: %s", type(exc).__name__, exc)
        return 1
    return 0
