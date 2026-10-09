"""Independent streaming assessment and export of Phase 1 perception records."""

import hashlib
import importlib.metadata
import json
import logging
import platform
import time
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sightguard import __version__
from sightguard.data import replay_records
from sightguard.risk.config import RiskConfig
from sightguard.risk.engine import AssessmentEngine

LOGGER = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assess_tracks(
    tracks_path: Path, output_dir: Path, config: RiskConfig, *, stream_id: str | None = None
) -> dict[str, Any]:
    tracks_path, output_dir = tracks_path.resolve(), output_dir.resolve()
    if not tracks_path.is_file():
        raise FileNotFoundError(tracks_path)
    names = ("risk_frames.jsonl", "events.jsonl", "alerts.jsonl", "run_metadata.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    if any((output_dir / name).exists() for name in names):
        raise FileExistsError("Assessment artifacts already exist; use a new output directory")
    input_hash = _sha256(tracks_path)
    if stream_id is None:
        path_hash = hashlib.sha256(str(tracks_path).encode()).hexdigest()
        stream_id = f"video-{path_hash[:16]}"
    if not isinstance(stream_id, str) or not stream_id.strip():
        raise ValueError("stream_id must be a non-empty string")
    engine = AssessmentEngine(config, stream_id=stream_id)
    root = Path(__file__).resolve().parents[1]
    source_files = sorted(
        [
            *root.glob("risk/*.py"),
            *root.glob("decision/*.py"),
            Path(__file__).resolve(),
            root / "data.py",
        ]
    )
    configuration = config.to_dict()
    metadata: dict[str, Any] = {
        "schema_version": "2.0",
        "algorithm_version": "heuristic-2.0",
        "status": "running",
        "stream_id": engine.stream_id,
        "input": {"path": str(tracks_path), "sha256": input_hash},
        "config": configuration,
        "config_sha256": hashlib.sha256(
            json.dumps(configuration, sort_keys=True, allow_nan=False).encode()
        ).hexdigest(),
        "source_sha256": {str(path.relative_to(root)): _sha256(path) for path in source_files},
        "versions": {
            "python": platform.python_version(),
            "sightguard": __version__,
            "PyYAML": importlib.metadata.version("PyYAML"),
        },
        "started_at_utc": datetime.now(UTC).isoformat(),
        "frames": 0,
        "event_records": 0,
        "alerts": 0,
        "source_error_frames": 0,
        "invalid_detections": 0,
        "perception_status_counts": {"available": 0, "limited": 0, "unavailable": 0},
        "error": None,
        "safety_boundary": "heuristic image-space relative risk, not collision probability",
    }
    source_metadata = tracks_path.parent / "run_metadata.json"
    if source_metadata.is_file():
        try:
            raw = json.loads(source_metadata.read_text(encoding="utf-8"))
            metadata["input"]["source_run_status"] = raw.get("status", "unknown")
            metadata["input"]["source_metadata_sha256"] = _sha256(source_metadata)
        except (ValueError, AttributeError):
            metadata["input"]["source_run_status"] = "invalid_metadata"
    else:
        metadata["input"]["source_run_status"] = "unknown"
    started = time.perf_counter()
    try:
        with ExitStack() as stack:
            streams = {
                name: stack.enter_context(
                    (output_dir / name).open("x", encoding="utf-8", newline="\n")
                )
                for name in names[:3]
            }
            for record in replay_records(tracks_path):
                step = engine.step(record)
                streams["risk_frames.jsonl"].write(step.frame.to_json() + "\n")
                for change in step.changes:
                    streams["events.jsonl"].write(change.to_json() + "\n")
                for alert in step.alerts:
                    streams["alerts.jsonl"].write(alert.to_json() + "\n")
                metadata["frames"] += 1
                metadata["event_records"] += len(step.changes)
                metadata["alerts"] += len(step.alerts)
                metadata["source_error_frames"] += record.status == "processing_error"
                metadata["invalid_detections"] += sum(
                    o.error is not None for o in step.frame.observations
                )
                metadata["perception_status_counts"][step.frame.perception.status] += 1
                if metadata["frames"] % 100 == 0:
                    LOGGER.info(
                        "Assessed %d frames, video time %.1f ms",
                        metadata["frames"],
                        record.timestamp_ms,
                    )
            if metadata["frames"] == 0:
                raise ValueError("Perception JSONL contains no frames")
            final_changes = engine.finish()
            for change in final_changes:
                streams["events.jsonl"].write(change.to_json() + "\n")
            metadata["event_records"] += len(final_changes)
            metadata["censored_risk_events"] = sum(c.event_kind == "risk" for c in final_changes)
            metadata["status"] = "completed"
    except Exception as exc:
        metadata["status"], metadata["error"] = "failed", f"{type(exc).__name__}: {exc}"
        LOGGER.error("Assessment failed: %s", metadata["error"])
        raise
    finally:
        metadata["finished_at_utc"] = datetime.now(UTC).isoformat()
        metadata["processing_seconds"] = time.perf_counter() - started
        (output_dir / "run_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )
    LOGGER.info(
        "Assessed %d frames; emitted %d alerts; outputs: %s",
        metadata["frames"],
        metadata["alerts"],
        output_dir,
    )
    return metadata
