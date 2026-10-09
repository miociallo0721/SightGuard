"""Read-only adapters for persisted SightGuard results and explicit previews.

The services validate persisted artifacts and delegate strategy execution to the
existing Phase 2/3 core. They do not recompute benchmark metrics or mutate source
datasets and completed runs. The only writes are collision-safe ZIP exports.
"""

from __future__ import annotations

import json
import math
import re
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sightguard.config import load_config
from sightguard.data import FrameRecord, replay_records
from sightguard.evaluation.config import BenchmarkConfig, load_benchmark_config
from sightguard.evaluation.dataset import (
    load_dataset,
    read_manifest,
    sha256_file,
)
from sightguard.evaluation.schema import (
    SCHEMA_VERSION,
    StrategyAlert,
    VideoAnnotations,
    load_annotations,
)
from sightguard.evaluation.strategies import run_strategies
from sightguard.risk.config import RiskConfig, load_risk_config
from sightguard.risk.schema import RiskLevel

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


class ConsoleDataError(ValueError):
    """An artifact is missing, corrupt, incomplete, or outside the console boundary."""


@dataclass(frozen=True)
class AnalysisView:
    run_id: str
    records: tuple[FrameRecord, ...]
    risk_frames: tuple[dict[str, Any], ...]
    events: tuple[dict[str, Any], ...]
    alerts: tuple[dict[str, Any], ...]
    metadata: dict[str, Any]
    source_kind: str
    video_path: Path | None
    duration_ms: float
    stats: dict[str, Any]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class BenchmarkView:
    run_id: str
    metadata: dict[str, Any]
    metrics: dict[str, dict[str, Any]]
    alerts: tuple[dict[str, Any], ...]
    matches: tuple[dict[str, Any], ...]
    tables: dict[str, pd.DataFrame]
    labels: dict[str, VideoAnnotations]
    warnings: tuple[str, ...]


_COUNT_METRICS = (
    "tp",
    "fp",
    "fn",
    "duplicates",
    "unsupported",
    "ignored_alerts",
    "perception_alerts",
    "scene_alerts",
    "total_alerts",
    "important_events",
    "labeled_events",
    "eligible_events",
    "excluded_events",
    "detected_events",
    "missed_events",
    "opportunities",
    "matched_initial_count",
    "effective_duration_ms",
    "excluded_duration_ms",
)
_RATIO_METRICS = (
    "event_recall",
    "opportunity_recall",
    "duplicate_rate",
    "on_time_event_rate",
)
_RATE_METRICS = ("invalid_alerts_per_hour", "total_alerts_per_minute")
_OPTIONAL_TABLES = (
    "comparison",
    "per_video",
    "per_scene",
    "per_severity",
    "event_results",
    "per_event_type",
    "error_cases",
)
_RUN_DATA_FILES = (
    "tracks.jsonl",
    "risk_frames.jsonl",
    "events.jsonl",
    "alerts.jsonl",
    "run_metadata.json",
)


def _pandas() -> Any:
    try:
        import pandas
    except ImportError as exc:  # pragma: no cover - exercised only in broken installs
        raise ConsoleDataError("Pandas is required to display Research Console tables") from exc
    return pandas


def _inside(base: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(base.resolve())
        return True
    except (ValueError, OSError):
        return False


def _finite_number(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or type(value) not in (int, float) or not math.isfinite(value):
        raise ConsoleDataError(f"{name} must be a finite number")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ConsoleDataError(f"{name} must be at least {minimum}")
    return result


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ConsoleDataError(f"{name} must be a JSON object with string keys")
    return value


def _reject_non_finite(value: Any, name: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ConsoleDataError(f"{name} contains a non-finite number")
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_non_finite(item, f"{name}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_non_finite(item, f"{name}[{index}]")


def _reject_json_constant(token: str) -> None:
    raise ValueError(f"Non-standard JSON numeric constant {token}")


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
        _reject_non_finite(value, name)
        return _object(value, name)
    except FileNotFoundError as exc:
        raise ConsoleDataError(f"Missing {name}: {path}") from exc
    except (OSError, UnicodeError, ValueError) as exc:
        raise ConsoleDataError(f"Cannot read {name} at {path}: {exc}") from exc


def _read_jsonl(path: Path, name: str) -> list[dict[str, Any]]:
    try:
        stream = path.open(encoding="utf-8")
    except OSError as exc:
        raise ConsoleDataError(f"Cannot open {name} at {path}: {exc}") from exc
    values: list[dict[str, Any]] = []
    with stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ConsoleDataError(f"Blank line in {name} at line {line_number}")
            try:
                value = json.loads(line, parse_constant=_reject_json_constant)
                _reject_non_finite(value, f"{name} line {line_number}")
                values.append(_object(value, f"{name} line {line_number}"))
            except (json.JSONDecodeError, ConsoleDataError, ValueError) as exc:
                raise ConsoleDataError(f"Invalid {name} line {line_number}: {exc}") from exc
    return values


def _validate_risk_frames(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    result = []
    previous_index = -1
    previous_time = -1.0
    for index, row in enumerate(rows, 1):
        if row.get("schema_version") != "2.0":
            raise ConsoleDataError(f"Risk frame line {index} must use schema 2.0")
        frame_index = row.get("frame_index")
        if type(frame_index) is not int or frame_index < 0 or frame_index <= previous_index:
            raise ConsoleDataError(f"Risk frame line {index} has an invalid frame index")
        timestamp = _finite_number(
            row.get("timestamp_ms"), f"Risk frame line {index} timestamp", minimum=0
        )
        if timestamp < previous_time:
            raise ConsoleDataError(f"Risk frame line {index} has decreasing video time")
        if not isinstance(row.get("observations"), list) or not isinstance(row.get("events"), list):
            raise ConsoleDataError(f"Risk frame line {index} is missing observations/events arrays")
        required_text = ("stream_id", "timestamp_source", "input_status")
        if any(not isinstance(row.get(name), str) or not row[name] for name in required_text):
            raise ConsoleDataError(
                f"Risk frame line {index} has invalid stream/source/status fields"
            )
        if not isinstance(row.get("perception"), dict) or not isinstance(
            row["perception"].get("status"), str
        ):
            raise ConsoleDataError(f"Risk frame line {index} has invalid perception status")
        for observation in row["observations"]:
            if not isinstance(observation, dict):
                raise ConsoleDataError(f"Risk frame line {index} has a non-object observation")
            if type(observation.get("class_id")) is not int or observation["class_id"] < 0:
                raise ConsoleDataError(f"Risk frame line {index} has an invalid observation class")
            if (
                type(observation.get("detection_index")) is not int
                or observation["detection_index"] < 0
            ):
                raise ConsoleDataError(f"Risk frame line {index} has an invalid detection index")
        result.append(row)
        previous_index, previous_time = frame_index, timestamp
    return tuple(result)


def _validate_event_changes(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    for index, row in enumerate(rows, 1):
        required_text = ("stream_id", "event_id", "event_kind", "action", "state", "reason")
        if row.get("schema_version") != "2.0" or any(
            not isinstance(row.get(name), str) or not row[name] for name in required_text
        ):
            raise ConsoleDataError(f"Event change line {index} has an invalid schema")
        _finite_number(row.get("timestamp_ms"), f"Event change line {index} timestamp", minimum=0)
    return tuple(rows)


def _validate_phase2_alerts(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    allowed_types = {"first", "escalation", "repeat", "perception_limited"}
    for index, row in enumerate(rows, 1):
        required_text = ("stream_id", "alert_id", "event_id", "alert_type", "reason")
        if row.get("schema_version") != "2.0" or any(
            not isinstance(row.get(name), str) or not row[name] for name in required_text
        ):
            raise ConsoleDataError(f"Analysis alert line {index} has an invalid schema")
        if row["alert_type"] not in allowed_types:
            raise ConsoleDataError(f"Analysis alert line {index} has an invalid alert type")
        source_ids = row.get("source_track_ids")
        if not isinstance(source_ids, list) or any(
            type(track_id) is not int or track_id < 0 for track_id in source_ids
        ):
            raise ConsoleDataError(f"Analysis alert line {index} has invalid source track IDs")
        _finite_number(row.get("timestamp_ms"), f"Analysis alert line {index} timestamp", minimum=0)
        if row.get("risk_level") is not None:
            try:
                RiskLevel(row["risk_level"])
            except (TypeError, ValueError) as exc:
                raise ConsoleDataError(
                    f"Analysis alert line {index} has invalid risk level"
                ) from exc
        score = row.get("risk_score")
        if score is not None:
            score = _finite_number(score, f"Analysis alert line {index} score", minimum=0)
            if score > 1:
                raise ConsoleDataError(f"Analysis alert line {index} score exceeds 1")
    return tuple(rows)


def _validate_analysis_metadata(metadata: dict[str, Any], records: tuple[FrameRecord, ...]) -> None:
    version = metadata.get("schema_version")
    if version == "1.0":
        count_name = "frames_processed"
    elif version == "2.0":
        count_name = "frames"
    else:
        raise ConsoleDataError("Analysis metadata has a missing or unsupported schema_version")
    status = metadata.get("status")
    if not isinstance(status, str) or status not in ("running", "completed", "failed"):
        raise ConsoleDataError("Analysis metadata has an invalid run status")
    frame_count = metadata.get(count_name)
    if type(frame_count) is not int or frame_count < 0:
        raise ConsoleDataError(f"Analysis metadata {count_name} must be a nonnegative integer")
    if frame_count != len(records):
        raise ConsoleDataError(
            f"Analysis metadata {count_name}={frame_count} disagrees with "
            f"the {len(records)} perception records"
        )


def _validate_metrics(value: Any, strategies: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    raw = _object(value, "metrics.json")
    _reject_non_finite(raw, "metrics.json")
    if set(raw) != set(strategies):
        raise ConsoleDataError("metrics.json strategies differ from run metadata")
    validated: dict[str, dict[str, Any]] = {}
    for strategy in strategies:
        metrics = _object(raw[strategy], f"metrics.{strategy}")
        missing = (
            set(_COUNT_METRICS)
            | set(_RATIO_METRICS)
            | set(_RATE_METRICS)
            | {"first_response_delay_mean_ms", "response_delays_ms"}
        ) - set(metrics)
        if missing:
            raise ConsoleDataError(f"metrics.{strategy} is missing fields: {sorted(missing)}")
        for name in _COUNT_METRICS:
            value_number = metrics[name]
            if name not in ("effective_duration_ms", "excluded_duration_ms"):
                if type(value_number) is not int or value_number < 0:
                    raise ConsoleDataError(
                        f"metrics.{strategy}.{name} must be a nonnegative integer count"
                    )
            else:
                _finite_number(value_number, f"metrics.{strategy}.{name}", minimum=0)
            if type(value_number) is bool:
                raise ConsoleDataError(f"metrics.{strategy}.{name} must be an integer count")
        for name in (*_RATIO_METRICS, "first_response_delay_mean_ms", *_RATE_METRICS):
            value_number = metrics[name]
            if value_number is None:
                continue
            number = _finite_number(value_number, f"metrics.{strategy}.{name}", minimum=0)
            if name in _RATIO_METRICS and number > 1:
                raise ConsoleDataError(f"metrics.{strategy}.{name} must be between 0 and 1 or null")
        delays = metrics["response_delays_ms"]
        if not isinstance(delays, list):
            raise ConsoleDataError(f"metrics.{strategy}.response_delays_ms must be an array")
        for index, delay in enumerate(delays):
            _finite_number(delay, f"metrics.{strategy}.response_delays_ms[{index}]", minimum=0)
        validated[strategy] = metrics
    return validated


class ConsoleStore:
    """Load local result artifacts under a project root without trusting UI paths."""

    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()
        roots = {}
        for name in ("outputs", "datasets", "configs"):
            raw_path = self.root / name
            resolved = raw_path.resolve()
            if resolved.parent != self.root or resolved.name != name:
                raise ConsoleDataError(f"Managed {name} directory cannot be a symlink or alias")
            roots[name] = resolved
        examples_path = self.root / "examples"
        self.examples_root = examples_path.resolve()
        if self.examples_root.parent != self.root or self.examples_root.name != "examples":
            raise ConsoleDataError("Managed examples directory cannot be a symlink or alias")
        self.outputs_root = roots["outputs"]
        self.datasets_root = roots["datasets"]
        self.configs_root = roots["configs"]

    def _resolve_id(self, base: Path, identifier: str, kind: str) -> Path:
        if not isinstance(identifier, str) or not identifier.strip():
            raise ConsoleDataError(f"{kind} ID must be a non-empty relative path")
        candidate_part = Path(identifier)
        if candidate_part.is_absolute() or any(part in ("..", "") for part in candidate_part.parts):
            raise ConsoleDataError(f"Invalid {kind} ID: {identifier}")
        candidate = (base / candidate_part).resolve()
        if candidate == base or not _inside(base, candidate):
            raise ConsoleDataError(f"{kind} ID escapes the allowed directory")
        return candidate

    def _run_path(self, run_id: str) -> Path:
        path = self._resolve_id(self.outputs_root, run_id, "Run")
        if not path.is_dir():
            raise ConsoleDataError(f"Run directory does not exist: {run_id}")
        return path

    def _manifest_path(self, manifest_id: str) -> Path:
        path = self._resolve_id(self.datasets_root, manifest_id, "Dataset manifest")
        if path.name.lower() != "manifest.csv" or not path.is_file():
            raise ConsoleDataError(f"Dataset manifest does not exist: {manifest_id}")
        return path

    def _safe_dataset_file(self, path: Path, description: str) -> Path:
        resolved = path.resolve()
        if not _inside(self.datasets_root, resolved):
            raise ConsoleDataError(f"{description} path escapes the managed datasets directory")
        return resolved

    def _safe_perception_file(self, path: Path, description: str = "Perception") -> Path:
        resolved = path.resolve()
        if resolved.suffix.lower() != ".jsonl" or not (
            _inside(self.datasets_root, resolved) or _inside(self.outputs_root, resolved)
        ):
            raise ConsoleDataError(
                f"{description} JSONL path must be inside datasets/ or outputs/; "
                "place external data inside datasets/ first"
            )
        return resolved

    def _safe_video_file(self, path: Path, description: str = "Video") -> Path:
        resolved = path.resolve()
        if not (_inside(self.datasets_root, resolved) or _inside(self.examples_root, resolved)):
            raise ConsoleDataError(
                f"{description} path must be inside datasets/ or examples/; "
                "place external data inside datasets/ first"
            )
        return resolved

    def analysis_runs(self) -> list[str]:
        if not self.outputs_root.is_dir():
            return []
        result: set[str] = set()
        for metadata_path in self.outputs_root.rglob("run_metadata.json"):
            if not _inside(self.outputs_root, metadata_path):
                continue
            run_dir = metadata_path.parent
            try:
                metadata = _read_json(metadata_path, "run metadata")
            except ConsoleDataError:
                continue
            if metadata.get("schema_version") == "3.0" or (run_dir / "metrics.json").exists():
                continue
            has_tracks = (run_dir / "tracks.jsonl").is_file()
            if not has_tracks and any(
                (parent / "tracks.jsonl").is_file()
                for parent in run_dir.parents
                if _inside(self.outputs_root, parent)
            ):
                # The parent is the selectable analysis run; nested Phase 2
                # assessment directories are implementation details, not runs.
                continue
            sibling_perception = run_dir.parent / "perception" / "tracks.jsonl"
            if (
                not has_tracks
                and sibling_perception.is_file()
                and _inside(self.outputs_root, sibling_perception)
            ):
                # Console jobs select their perception folder; assessment is a sibling.
                continue
            has_risk = (run_dir / "risk_frames.jsonl").is_file()
            if not has_risk:
                has_risk = any(
                    child.is_dir() and (child / "risk_frames.jsonl").is_file()
                    for child in run_dir.iterdir()
                )
            if has_tracks or has_risk:
                result.add(run_dir.relative_to(self.outputs_root).as_posix())
        for console_metadata_path in self.outputs_root.rglob("console_metadata.json"):
            if not _inside(self.outputs_root, console_metadata_path):
                continue
            job_dir = console_metadata_path.parent
            perception_dir = job_dir / "perception"
            if (perception_dir / "run_metadata.json").is_file() and (
                perception_dir / "tracks.jsonl"
            ).is_file():
                continue
            try:
                console_metadata = _read_json(console_metadata_path, "console run metadata")
            except ConsoleDataError:
                continue
            if console_metadata.get("status") in ("failed", "running"):
                result.add(job_dir.relative_to(self.outputs_root).as_posix())
        return sorted(result)

    def benchmark_runs(self) -> list[str]:
        if not self.outputs_root.is_dir():
            return []
        result = []
        for metadata_path in self.outputs_root.rglob("run_metadata.json"):
            if not _inside(self.outputs_root, metadata_path):
                continue
            run_dir = metadata_path.parent
            try:
                metadata = _read_json(metadata_path, "run metadata")
            except ConsoleDataError:
                continue
            if (
                metadata.get("schema_version") == "3.0"
                and metadata.get("status") == "completed"
                and metadata.get("artifacts_complete") is True
                and (run_dir / "metrics.json").is_file()
            ):
                result.append(run_dir.relative_to(self.outputs_root).as_posix())
        return sorted(result)

    def dataset_manifests(self) -> list[str]:
        if not self.datasets_root.is_dir():
            return []
        result = []
        for path in self.datasets_root.rglob("manifest.csv"):
            if path.is_file() and _inside(self.datasets_root, path):
                result.append(path.relative_to(self.datasets_root).as_posix())
        return sorted(result)

    def _analysis_tracks_path(
        self, run_dir: Path, metadata: dict[str, Any] | None = None
    ) -> Path | None:
        candidate = run_dir / "tracks.jsonl"
        if candidate.is_file() and _inside(self.outputs_root, candidate):
            return candidate.resolve()
        candidate = run_dir / "perception" / "tracks.jsonl"
        if candidate.is_file() and _inside(self.outputs_root, candidate):
            return candidate.resolve()
        for parent in run_dir.parents:
            if not _inside(self.outputs_root, parent):
                break
            candidate = parent / "tracks.jsonl"
            if candidate.is_file() and _inside(self.outputs_root, candidate):
                return candidate.resolve()
        input_data = metadata.get("input") if isinstance(metadata, dict) else None
        input_path = input_data.get("path") if isinstance(input_data, dict) else None
        if isinstance(input_path, str) and input_path:
            candidate = Path(input_path).expanduser().resolve()
            if (
                candidate.suffix.lower() == ".jsonl"
                and candidate.is_file()
                and (
                    _inside(self.outputs_root, candidate) or _inside(self.datasets_root, candidate)
                )
            ):
                return candidate
        if (
            isinstance(metadata, dict)
            and metadata.get("schema_version") == "4.0"
            and metadata.get("status") in ("failed", "running")
        ):
            return None
        raise ConsoleDataError(f"Analysis run has no safe tracks.jsonl: {run_dir}")

    def _analysis_risk_dir(self, run_dir: Path, tracks_path: Path | None = None) -> Path | None:
        if tracks_path is None:
            return None

        def matches_tracks(assessment_dir: Path) -> bool:
            metadata_path = assessment_dir / "run_metadata.json"
            if not (
                assessment_dir.is_dir()
                and _inside(self.outputs_root, assessment_dir)
                and metadata_path.is_file()
            ):
                return False
            try:
                metadata = _read_json(metadata_path, "assessment metadata")
                input_data = metadata.get("input")
                expected_hash = input_data.get("sha256") if isinstance(input_data, dict) else None
                return (
                    metadata.get("schema_version") == "2.0"
                    and metadata.get("status") == "completed"
                    and isinstance(expected_hash, str)
                    and expected_hash == sha256_file(tracks_path)
                )
            except (ConsoleDataError, OSError):
                return False

        if (run_dir / "risk_frames.jsonl").is_file() and matches_tracks(run_dir):
            return run_dir

        nested = sorted(
            {
                path.parent
                for path in run_dir.rglob("risk_frames.jsonl")
                if _inside(self.outputs_root, path)
            },
            key=lambda path: ("phase2" not in path.name.lower(), path.as_posix()),
        )
        for assessment_dir in nested:
            if matches_tracks(assessment_dir):
                return assessment_dir

        # Console jobs keep Phase 1 perception and Phase 2 assessment as siblings.
        # Only expose this assessment after its stored input hash matches the
        # selected perception file exactly.
        sibling = run_dir.parent / "assessment"
        if matches_tracks(sibling):
            return sibling
        return None

    def _assessment_omission_reason(self, run_dir: Path, tracks_path: Path | None) -> str | None:
        if tracks_path is None:
            return None
        assessment_dirs = {
            path.parent
            for path in run_dir.rglob("risk_frames.jsonl")
            if _inside(self.outputs_root, path)
        }
        sibling = run_dir.parent / "assessment"
        if (sibling / "risk_frames.jsonl").is_file() and _inside(self.outputs_root, sibling):
            assessment_dirs.add(sibling)
        for assessment_dir in assessment_dirs:
            metadata_path = assessment_dir / "run_metadata.json"
            if not metadata_path.is_file():
                continue
            try:
                metadata = _read_json(metadata_path, "assessment metadata")
            except ConsoleDataError:
                continue
            if metadata.get("schema_version") != "2.0" or metadata.get("status") != "completed":
                continue
            input_data = metadata.get("input")
            expected_hash = input_data.get("sha256") if isinstance(input_data, dict) else None
            if not isinstance(expected_hash, str):
                return (
                    "Phase 2 metadata has no verified input hash; assessment artifacts "
                    "were omitted."
                )
            try:
                actual_hash = sha256_file(tracks_path)
            except OSError:
                return "Phase 2 input could not be verified; assessment artifacts were omitted."
            if actual_hash != expected_hash:
                return (
                    "Phase 2 input hash does not match the selected perception JSONL; "
                    "risk frames, events, and alerts were omitted."
                )
        return None

    def _analysis_source_kind(
        self, metadata: dict[str, Any], tracks_path: Path | None, run_dir: Path
    ) -> str:
        for value in (
            metadata.get("data_kind"),
            metadata.get("source_kind"),
            (metadata.get("input") or {}).get("data_kind")
            if isinstance(metadata.get("input"), dict)
            else None,
        ):
            if value in ("real", "synthetic", "unknown"):
                return value
        # A manifest gives explicit provenance. Match by canonical tracks path or hash.
        if tracks_path is not None:
            for manifest_id in self.dataset_manifests():
                try:
                    entries = read_manifest(self._manifest_path(manifest_id))
                except (ConsoleDataError, OSError, ValueError):
                    continue
                for entry in entries:
                    try:
                        candidate = self._safe_perception_file(entry.tracks_path)
                        if (
                            candidate.is_file()
                            and sha256_file(candidate) == entry.tracks_sha256
                            and sha256_file(tracks_path) == entry.tracks_sha256
                        ):
                            return entry.data_kind
                    except (ConsoleDataError, OSError):
                        continue
        tokens = {part.lower() for part in run_dir.parts}
        if any(
            "synthetic" in part or "fixture" in part or "repeatedbus" in part for part in tokens
        ):
            return "synthetic"
        return "unknown"

    def _safe_video_preview(
        self, run_dir: Path, metadata: dict[str, Any], warnings: list[str]
    ) -> Path | None:
        for ancestor in (run_dir, *run_dir.parents):
            if not _inside(self.outputs_root, ancestor):
                break
            for directory in (ancestor, ancestor / "perception"):
                for name in ("annotated.mp4", "annotated_video.mp4"):
                    candidate = directory / name
                    if candidate.is_file() and _inside(self.outputs_root, candidate):
                        return candidate.resolve()
        input_data = metadata.get("input")
        input_path = input_data.get("path") if isinstance(input_data, dict) else None
        if isinstance(input_path, str) and input_path:
            try:
                candidate = Path(input_path).expanduser().resolve()
            except OSError:
                candidate = None
            if (
                candidate is not None
                and candidate.is_file()
                and (
                    _inside(self.outputs_root, candidate)
                    or _inside(self.datasets_root, candidate)
                    or _inside(self.examples_root, candidate)
                )
            ):
                return candidate
            if candidate is not None:
                warnings.append(
                    "The source video is outside the workspace and was omitted from preview."
                )
        else:
            warnings.append("No workspace-local video preview is available for this analysis run.")
        return None

    def load_analysis(self, run_id: str) -> AnalysisView:
        run_dir = self._run_path(run_id)
        metadata_path = run_dir / "run_metadata.json"
        warnings: list[str] = []
        console_metadata_path = run_dir / "console_metadata.json"
        is_perception_child = run_dir.name == "perception"
        if is_perception_child:
            parent_console_metadata = run_dir.parent / "console_metadata.json"
            if parent_console_metadata.is_file():
                console_metadata_path = parent_console_metadata
        console_metadata = (
            _read_json(console_metadata_path, "console run metadata")
            if console_metadata_path.is_file()
            else None
        )
        if metadata_path.is_file():
            metadata = _read_json(metadata_path, "analysis run metadata")
        elif console_metadata is not None and console_metadata.get("schema_version") == "4.0":
            # Failed/running console jobs can have no completed Phase 1 artifact.
            if console_metadata.get("status") == "completed":
                raise ConsoleDataError(
                    "Completed console job is missing its Phase 1 run_metadata.json"
                )
            metadata = dict(console_metadata)
        else:
            raise ConsoleDataError(f"Missing analysis run metadata: {metadata_path}")

        if console_metadata is not None:
            if console_metadata.get("schema_version") != "4.0" or console_metadata.get(
                "status"
            ) not in ("running", "completed", "failed"):
                raise ConsoleDataError("Console job metadata has an invalid schema or status")
            if metadata is not console_metadata:
                analysis_metadata = dict(metadata)
                analysis_metadata["perception_metadata"] = dict(metadata)
                analysis_metadata["console_metadata"] = dict(console_metadata)
                analysis_metadata["status"] = console_metadata["status"]
                analysis_metadata["source_kind"] = console_metadata.get("source_kind", "unknown")
                metadata_for_view = analysis_metadata
                metadata_for_records = metadata
            else:
                metadata_for_view = dict(console_metadata)
                metadata_for_records = console_metadata
        else:
            metadata_for_view = dict(metadata)
            metadata_for_records = metadata

        status = (
            console_metadata.get("status")
            if console_metadata is not None
            else metadata.get("status")
        )
        if status != "completed":
            warnings.append(
                f"Analysis run status is {status or 'unknown'!r}; saved records may be partial."
            )
        if metadata_for_records.get("schema_version") == "4.0":
            # Console job envelopes use their own schema; Phase 1 frame counts
            # are validated whenever a Phase 1 metadata artifact is present.
            if metadata_for_records.get("status") not in ("running", "completed", "failed"):
                raise ConsoleDataError("Console job metadata has an invalid run status")
        tracks_path = self._analysis_tracks_path(run_dir, metadata_for_records)
        records: tuple[FrameRecord, ...] = ()
        if tracks_path is not None:
            try:
                records = tuple(replay_records(tracks_path))
            except (OSError, ValueError, TypeError) as exc:
                raise ConsoleDataError(f"Cannot load perception records: {exc}") from exc
            if metadata_for_records.get("schema_version") in ("1.0", "2.0"):
                _validate_analysis_metadata(metadata_for_records, records)
        elif status == "completed":
            raise ConsoleDataError("Completed analysis run has no safe tracks.jsonl")
        else:
            warnings.append("No perception JSONL is available for this incomplete console job.")

        risk_dir = self._analysis_risk_dir(run_dir, tracks_path)
        risk_frames: tuple[dict[str, Any], ...] = ()
        events: tuple[dict[str, Any], ...] = ()
        alerts: tuple[dict[str, Any], ...] = ()
        if risk_dir is None:
            omission_reason = self._assessment_omission_reason(run_dir, tracks_path)
            warnings.append(
                omission_reason
                or "No saved Phase 2 assessment artifacts were found; scores and events "
                "are unavailable."
            )
        else:
            assessment_metadata_path = risk_dir / "run_metadata.json"
            assessment_metadata = None
            if assessment_metadata_path.is_file() and risk_dir != run_dir:
                assessment_metadata = _read_json(
                    assessment_metadata_path, "assessment run metadata"
                )
                metadata_for_view["assessment_metadata"] = assessment_metadata
            elif assessment_metadata_path.is_file():
                assessment_metadata = metadata_for_records
            if assessment_metadata is not None:
                source_input = assessment_metadata.get("input")
                expected_tracks_hash = (
                    source_input.get("sha256") if isinstance(source_input, dict) else None
                )
                if tracks_path is not None and expected_tracks_hash != sha256_file(tracks_path):
                    warnings.append(
                        "Phase 2 artifacts refer to a different perception JSONL; "
                        "risk frames, events, and alerts were omitted."
                    )
                    risk_dir = None
                elif not isinstance(expected_tracks_hash, str):
                    warnings.append(
                        "Phase 2 metadata has no verified input hash; assessment artifacts "
                        "were omitted."
                    )
                    risk_dir = None
                if risk_dir is not None and assessment_metadata.get("status") != "completed":
                    warnings.append(
                        "Phase 2 assessment did not complete; partial risk and alert "
                        "artifacts were omitted."
                    )
                    risk_dir = None
            if risk_dir is not None:
                risk_path = risk_dir / "risk_frames.jsonl"
                events_path = risk_dir / "events.jsonl"
                alerts_path = risk_dir / "alerts.jsonl"
                for path, key in (
                    (risk_path, "risk frames"),
                    (events_path, "event changes"),
                    (alerts_path, "analysis alerts"),
                ):
                    if not path.is_file():
                        warnings.append(f"Saved {key} artifact is missing.")
                if risk_path.is_file():
                    risk_frames = _validate_risk_frames(_read_jsonl(risk_path, "risk frames"))
                if events_path.is_file():
                    events = _validate_event_changes(_read_jsonl(events_path, "event changes"))
                if alerts_path.is_file():
                    alerts = _validate_phase2_alerts(_read_jsonl(alerts_path, "analysis alerts"))
            if risk_frames and len(risk_frames) != len(records):
                warnings.append(
                    "Risk frame count differs from perception record count; lookup uses "
                    "timestamps and frame indices where available."
                )

        detections = [detection for record in records for detection in record.detections]
        track_ids = {
            detection.track_id for detection in detections if detection.track_id is not None
        }
        risk_event_ids = {
            event.get("event_id")
            for event in events
            if event.get("event_kind") == "risk" and isinstance(event.get("event_id"), str)
        }
        stats = {
            "frame_count": len(records),
            "detection_count": len(detections),
            "tracked_object_count": len(track_ids),
            "risk_event_count": len(risk_event_ids),
            "first_alert_count": sum(alert["alert_type"] == "first" for alert in alerts),
            "escalation_alert_count": sum(alert["alert_type"] == "escalation" for alert in alerts),
            "repeat_alert_count": sum(alert["alert_type"] == "repeat" for alert in alerts),
            "perception_limited_alert_count": sum(
                alert["alert_type"] == "perception_limited" for alert in alerts
            ),
        }
        duration = self._analysis_duration_ms(metadata_for_records, records)
        if tracks_path is not None:
            source_kind = self._analysis_source_kind(metadata_for_view, tracks_path, run_dir)
        else:
            declared_kind = metadata_for_view.get("source_kind")
            source_kind = (
                declared_kind if declared_kind in ("real", "synthetic", "unknown") else "unknown"
            )
        if source_kind == "unknown":
            warnings.append(
                "Source kind is unknown; no real-data provenance was inferred from the video alone."
            )
        video_path = self._safe_video_preview(run_dir, metadata_for_records, warnings)
        return AnalysisView(
            run_id,
            records,
            risk_frames,
            events,
            alerts,
            metadata_for_view,
            source_kind,
            video_path,
            duration,
            stats,
            tuple(dict.fromkeys(warnings)),
        )

    @staticmethod
    def _analysis_duration_ms(metadata: dict[str, Any], records: tuple[FrameRecord, ...]) -> float:
        for value in (metadata.get("duration_ms"), metadata.get("video_duration_ms")):
            if type(value) in (int, float) and math.isfinite(value) and value >= 0:
                return float(value)
        input_data = metadata.get("input")
        if isinstance(input_data, dict):
            value = input_data.get("duration_ms")
            if type(value) in (int, float) and math.isfinite(value) and value >= 0:
                return float(value)
            fps = input_data.get("fps")
            frame_count = input_data.get("reported_frame_count") or metadata.get("frames_processed")
            if (
                type(fps) in (int, float)
                and math.isfinite(fps)
                and fps > 0
                and type(frame_count) is int
                and frame_count >= 0
            ):
                return frame_count * 1000 / fps
        if not records:
            return 0.0
        intervals = [
            current.timestamp_ms - previous.timestamp_ms
            for previous, current in zip(records, records[1:], strict=False)
            if current.timestamp_ms > previous.timestamp_ms
        ]
        interval = sorted(intervals)[len(intervals) // 2] if intervals else 0.0
        return records[-1].timestamp_ms + interval

    def _benchmark_labels(
        self, metadata: dict[str, Any], warnings: list[str]
    ) -> dict[str, VideoAnnotations]:
        labels: dict[str, VideoAnnotations] = {}
        manifest_value = metadata.get("dataset")
        if not isinstance(manifest_value, str) or not manifest_value:
            warnings.append(
                "Dataset manifest reference is missing; ground-truth windows are unavailable."
            )
            return labels
        try:
            raw_manifest = Path(manifest_value)
            manifest_path = (
                raw_manifest.resolve()
                if raw_manifest.is_absolute()
                else (self.root / raw_manifest).resolve()
            )
            if not _inside(self.datasets_root, manifest_path) or not manifest_path.is_file():
                raise ConsoleDataError(
                    "Benchmark dataset manifest is outside the managed datasets "
                    "directory or missing"
                )
            expected_manifest_hash = metadata.get("manifest_sha256")
            if (
                not isinstance(expected_manifest_hash, str)
                or sha256_file(manifest_path) != expected_manifest_hash
            ):
                warnings.append(
                    "Dataset manifest changed since this benchmark; ground-truth "
                    "windows were omitted."
                )
                return labels
            entries = {entry.video_id: entry for entry in read_manifest(manifest_path)}
        except (ConsoleDataError, OSError, ValueError) as exc:
            warnings.append(f"Ground-truth labels are unavailable: {exc}")
            return labels

        inputs = metadata.get("inputs")
        if not isinstance(inputs, list):
            warnings.append(
                "Benchmark input references are missing; ground-truth windows were omitted."
            )
            return labels
        for item in inputs:
            if not isinstance(item, dict):
                warnings.append(
                    "A malformed input reference prevented loading its ground-truth labels."
                )
                continue
            video_id = item.get("video_id")
            entry = entries.get(video_id) if isinstance(video_id, str) else None
            if entry is None:
                warnings.append(
                    f"Ground-truth annotation is not in the referenced manifest for {video_id!r}."
                )
                continue
            try:
                annotation_path = self._safe_dataset_file(entry.annotations_path, "Annotation")
                tracks_path = self._safe_perception_file(entry.tracks_path, "Perception")
                if (
                    item.get("annotations_sha256") != entry.annotations_sha256
                    or item.get("tracks_sha256") != entry.tracks_sha256
                    or item.get("annotations_path") is None
                    or Path(str(item["annotations_path"])).resolve() != annotation_path
                    or Path(str(item.get("tracks_path", ""))).resolve() != tracks_path
                    or not annotation_path.is_file()
                    or sha256_file(annotation_path) != entry.annotations_sha256
                ):
                    warnings.append(
                        f"Ground-truth files changed since the benchmark for {video_id}; "
                        "labels were omitted."
                    )
                    continue
                label = load_annotations(annotation_path, require_reviewed=True)
                if label.video_id != video_id or label.data_kind != metadata["data_kind"]:
                    warnings.append(
                        f"Ground-truth identity/source mismatch for {video_id}; "
                        "labels were omitted."
                    )
                    continue
                labels[video_id] = label
            except (ConsoleDataError, OSError, ValueError, KeyError, TypeError) as exc:
                warnings.append(f"Ground-truth labels for {video_id!r} are unavailable: {exc}")
        return labels

    def load_benchmark(self, run_id: str) -> BenchmarkView:
        run_dir = self._run_path(run_id)
        metadata = _read_json(run_dir / "run_metadata.json", "benchmark run metadata")
        if metadata.get("schema_version") != "3.0":
            raise ConsoleDataError("Selected run is not a Phase 3 benchmark")
        if metadata.get("status") != "completed" or metadata.get("artifacts_complete") is not True:
            raise ConsoleDataError(
                "Benchmark is incomplete or failed; artifacts are not safe to present as results"
            )
        data_kind = metadata.get("data_kind")
        if data_kind not in ("real", "synthetic"):
            raise ConsoleDataError(
                "Benchmark metadata must declare exactly one real or synthetic data kind"
            )
        names = metadata.get("strategies")
        if (
            not isinstance(names, list)
            or not names
            or any(name not in ("A", "B", "C") for name in names)
        ):
            raise ConsoleDataError("Benchmark metadata has invalid strategy names")
        strategies = tuple(names)
        if len(set(strategies)) != len(strategies):
            raise ConsoleDataError("Benchmark metadata contains duplicate strategy names")
        input_rows = metadata.get("inputs")
        video_count = metadata.get("video_count")
        if (
            not isinstance(input_rows, list)
            or type(video_count) is not int
            or video_count != len(input_rows)
            or any(
                not isinstance(item, dict)
                or not isinstance(item.get("video_id"), str)
                or item.get("data_kind") != data_kind
                for item in input_rows
            )
        ):
            raise ConsoleDataError("Benchmark input metadata has invalid video/source entries")
        input_video_ids = [item["video_id"] for item in input_rows]
        if len(set(input_video_ids)) != len(input_video_ids):
            raise ConsoleDataError("Benchmark metadata contains duplicate video IDs")
        for filename in ("metrics.json", "alerts.jsonl", "matches.jsonl"):
            if not (run_dir / filename).is_file() or not _inside(
                self.outputs_root, run_dir / filename
            ):
                raise ConsoleDataError(
                    f"Completed benchmark is missing required artifact {filename}"
                )
        raw_metrics = _read_json(run_dir / "metrics.json", "benchmark metrics")
        metrics = _validate_metrics(raw_metrics, strategies)

        alert_rows = _read_jsonl(run_dir / "alerts.jsonl", "strategy alerts")
        alerts: list[dict[str, Any]] = []
        for index, row in enumerate(alert_rows, 1):
            try:
                if row.get("schema_version") != SCHEMA_VERSION:
                    raise ValueError("unsupported schema version")
                if isinstance(row.get("risk_score"), bool):
                    raise ValueError("risk_score must be numeric or null")
                alert = StrategyAlert(**row)
            except (TypeError, ValueError, KeyError) as exc:
                raise ConsoleDataError(f"Invalid strategy alert line {index}: {exc}") from exc
            if alert.strategy not in strategies:
                raise ConsoleDataError(f"Alert line {index} names an unselected strategy")
            if alert.video_id not in input_video_ids:
                raise ConsoleDataError(f"Alert line {index} refers to an unlisted video")
            alerts.append(json.loads(alert.to_json()))
        expected_alert_count = metadata.get("raw_generated_alerts")
        if type(expected_alert_count) is not int or expected_alert_count != len(alerts):
            raise ConsoleDataError("Raw alert count disagrees with completed run metadata")

        match_rows = _read_jsonl(run_dir / "matches.jsonl", "benchmark matches")
        matches = tuple(self._validate_match(row, index) for index, row in enumerate(match_rows, 1))
        if any(row["strategy"] not in strategies for row in matches):
            raise ConsoleDataError("Match artifact contains an unselected strategy")
        if any(row["video_id"] not in input_video_ids for row in matches):
            raise ConsoleDataError("Match artifact refers to an unlisted video")

        warnings: list[str] = []
        tables: dict[str, pd.DataFrame] = {}
        pandas = _pandas()
        for table_name in _OPTIONAL_TABLES:
            table_path = run_dir / f"{table_name}.csv"
            if not table_path.is_file():
                tables[table_name] = pandas.DataFrame()
                warnings.append(f"Optional report table {table_path.name} is missing.")
                continue
            if not _inside(self.outputs_root, table_path):
                raise ConsoleDataError(
                    f"Report table escapes the output directory: {table_path.name}"
                )
            try:
                tables[table_name] = pandas.read_csv(
                    table_path, encoding="utf-8-sig", low_memory=False
                )
            except (OSError, ValueError, UnicodeError) as exc:
                raise ConsoleDataError(
                    f"Cannot read report table {table_path.name}: {exc}"
                ) from exc
        labels = self._benchmark_labels(metadata, warnings)
        if not labels:
            warnings.append(
                "No validated Ground Truth labels are available; stored metrics are "
                "shown without event windows."
            )
        return BenchmarkView(
            run_id,
            metadata,
            metrics,
            tuple(alerts),
            matches,
            tables,
            labels,
            tuple(dict.fromkeys(warnings)),
        )

    @staticmethod
    def _validate_match(row: dict[str, Any], index: int) -> dict[str, Any]:
        required = (
            "video_id",
            "strategy",
            "alert_id",
            "timestamp_ms",
            "status",
            "gt_event_id",
            "opportunity_id",
            "opportunity_kind",
            "latency_ms",
            "reason",
            "schema_version",
        )
        if any(name not in row for name in required):
            raise ConsoleDataError(f"Match line {index} is missing required fields")
        if row["schema_version"] != SCHEMA_VERSION:
            raise ConsoleDataError(f"Match line {index} has an unsupported schema")
        if row["status"] not in (
            "matched",
            "duplicate",
            "false_positive",
            "missed_opportunity",
            "ignored",
            "perception_limited",
        ):
            raise ConsoleDataError(f"Match line {index} has an invalid status")
        for name in ("video_id", "strategy", "reason"):
            if not isinstance(row[name], str) or not row[name]:
                raise ConsoleDataError(f"Match line {index} has an invalid {name}")
        for name in ("alert_id", "gt_event_id", "opportunity_id", "opportunity_kind"):
            if row[name] is not None and not isinstance(row[name], str):
                raise ConsoleDataError(f"Match line {index} has an invalid {name}")
        for name in ("timestamp_ms", "latency_ms"):
            if row[name] is not None:
                _finite_number(row[name], f"Match line {index} {name}", minimum=0)
        return row

    def browse_dataset(self, manifest_id: str) -> tuple[pd.DataFrame, str]:
        pandas = _pandas()
        try:
            manifest_path = self._manifest_path(manifest_id)
            entries = read_manifest(manifest_path)
        except (ConsoleDataError, OSError, ValueError) as exc:
            return pandas.DataFrame(), f"Dataset manifest could not be read: {exc}"
        rows: list[dict[str, Any]] = []
        warnings: list[str] = []
        for entry in entries:
            try:
                self._safe_perception_file(entry.tracks_path)
                tracks_safe = True
            except ConsoleDataError:
                tracks_safe = False
            annotation_safe = _inside(self.datasets_root, entry.annotations_path)
            video_safe = entry.video_path is None or (
                _inside(self.datasets_root, entry.video_path)
                or _inside(self.examples_root, entry.video_path)
            )
            if not tracks_safe or not annotation_safe:
                warnings.append(
                    f"{entry.video_id}: tracks must be under outputs/ or datasets/, and "
                    "annotations must be under datasets/"
                )
            if entry.video_path is not None and not video_safe:
                warnings.append(
                    f"{entry.video_id}: video is outside datasets/ and examples/ and "
                    "will not be previewed"
                )
            tracks_exists = tracks_safe and entry.tracks_path.is_file()
            annotations_exists = annotation_safe and entry.annotations_path.is_file()
            video_exists = bool(entry.video_path and video_safe and entry.video_path.is_file())
            label_status, event_count, event_types = "missing", None, ""
            if annotations_exists:
                try:
                    label = load_annotations(entry.annotations_path, require_reviewed=False)
                    label_status = label.annotation_status
                    event_count = len(label.events)
                    event_types = ", ".join(sorted({event.event_type for event in label.events}))
                except (OSError, ValueError, TypeError) as exc:
                    label_status = f"invalid: {exc}"
            rows.append(
                {
                    "video_id": entry.video_id,
                    "split": entry.split,
                    "source_group_id": entry.source_group_id,
                    "scene": entry.scene,
                    "data_kind": entry.data_kind,
                    "duration_ms": entry.duration_ms,
                    "fps": entry.fps,
                    "annotation_status": label_status,
                    "event_count": event_count,
                    "event_types": event_types,
                    "tracks_exists": tracks_exists,
                    "annotations_exists": annotations_exists,
                    "video_exists": video_exists,
                }
            )
        summary = f"Loaded {len(rows)} manifest entries from {manifest_id}."
        if warnings:
            summary += " " + " ".join(warnings)
        return pandas.DataFrame.from_records(rows), summary

    def validate_dataset(self, manifest_id: str) -> str:
        try:
            manifest_path = self._manifest_path(manifest_id)
            entries = read_manifest(manifest_path)
            for entry in entries:
                self._safe_perception_file(entry.tracks_path, "Perception")
                self._safe_dataset_file(entry.annotations_path, "Annotation")
                if entry.video_path is not None:
                    self._safe_video_file(entry.video_path, "Original video")
            videos = load_dataset(manifest_path, check_videos=True)
        except Exception as exc:
            return f"Validation failed: {type(exc).__name__}: {exc}"
        real = sum(video.entry.data_kind == "real" for video in videos)
        synthetic = len(videos) - real
        total_duration = sum(video.entry.duration_ms for video in videos)
        groups = len({video.entry.source_group_id for video in videos})
        return (
            f"Validation passed: {len(videos)} reviewed videos, {groups} source groups, "
            f"{real} real, {synthetic} synthetic, {total_duration:.1f} ms total duration."
        )

    def select_frame(
        self, view: AnalysisView, timestamp_ms: float
    ) -> tuple[np.ndarray | None, pd.DataFrame, pd.DataFrame, str]:
        pandas = _pandas()
        requested = _finite_number(timestamp_ms, "Requested timestamp", minimum=0)
        if not view.records:
            return (
                None,
                pandas.DataFrame(),
                pandas.DataFrame(),
                "No perception frames are available.",
            )
        record = min(
            view.records, key=lambda item: (abs(item.timestamp_ms - requested), item.frame_index)
        )
        risk_row = next(
            (row for row in view.risk_frames if row.get("frame_index") == record.frame_index), None
        )
        current_alerts = [
            alert
            for alert in view.alerts
            if alert.get("frame_index") == record.frame_index
            or alert.get("timestamp_ms") == record.timestamp_ms
        ]
        alert_reason_by_event = {
            alert.get("event_id"): alert.get("reason")
            for alert in current_alerts
            if isinstance(alert.get("event_id"), str)
        }
        observations: list[dict[str, Any]] = []
        if risk_row is not None:
            event_states = {
                event.get("event_id"): event.get("state")
                for event in risk_row.get("events", [])
                if isinstance(event, dict)
            }
            for row in risk_row.get("observations", []):
                features = row.get("features") if isinstance(row.get("features"), dict) else {}
                spatial = (
                    features.get("spatial") if isinstance(features.get("spatial"), dict) else {}
                )
                assessment = (
                    row.get("assessment") if isinstance(row.get("assessment"), dict) else {}
                )
                quality = (
                    assessment.get("quality")
                    if isinstance(assessment.get("quality"), dict)
                    else features.get("quality")
                    if isinstance(features.get("quality"), dict)
                    else {}
                )
                bottom_center = spatial.get("bottom_center_xy")
                assessment_reasons = assessment.get("reasons", [])
                event_id = row.get("event_id")
                observations.append(
                    {
                        "detection_index": row.get("detection_index"),
                        "track_id": row.get("source_track_id"),
                        "event_id": row.get("event_id"),
                        "class_id": row.get("class_id"),
                        "class_name": row.get("class_name"),
                        "bbox_normalized": spatial.get("bbox_normalized"),
                        "risk_score": assessment.get("score"),
                        "risk_level": assessment.get("level"),
                        "event_state": event_states.get(event_id),
                        "bbox_area": spatial.get("area"),
                        "bottom_y": (
                            bottom_center[1]
                            if isinstance(bottom_center, (list, tuple)) and len(bottom_center) == 2
                            else None
                        ),
                        "focus_relevance": spatial.get("focus_relevance"),
                        "spatial_score": assessment.get("spatial_score"),
                        "appearance_score": assessment.get("appearance_score"),
                        "motion_score": assessment.get("motion_score"),
                        "quality_score": quality.get("score"),
                        "observation_duration_ms": features.get("observation_duration_ms"),
                        "observation_interval_ms": features.get("observation_interval_ms"),
                        "motion_interval_ms": features.get("motion_interval_ms"),
                        "scale_change_per_s": features.get("scale_change_per_s"),
                        "inward_change_per_s": features.get("inward_change_per_s"),
                        "decision_reasons": assessment_reasons,
                        "decision_reason": alert_reason_by_event.get(
                            event_id, "No alert emitted for this observation/frame."
                        ),
                        "observation_error": row.get("error"),
                    }
                )
        else:
            for index, detection in enumerate(record.detections):
                x1, y1, x2, y2 = detection.bbox_xyxy
                observations.append(
                    {
                        "detection_index": index,
                        "track_id": detection.track_id,
                        "event_id": None,
                        "class_id": detection.class_id,
                        "class_name": detection.class_name,
                        "bbox_normalized": (
                            x1 / record.frame_width,
                            y1 / record.frame_height,
                            x2 / record.frame_width,
                            y2 / record.frame_height,
                        ),
                        "risk_score": None,
                        "risk_level": None,
                        "event_state": None,
                        "bbox_area": (
                            (x2 - x1) * (y2 - y1) / (record.frame_width * record.frame_height)
                        ),
                        "bottom_y": y2 / record.frame_height,
                        "focus_relevance": None,
                        "spatial_score": None,
                        "appearance_score": None,
                        "motion_score": None,
                        "quality_score": None,
                        "observation_duration_ms": None,
                        "observation_interval_ms": None,
                        "motion_interval_ms": None,
                        "scale_change_per_s": None,
                        "inward_change_per_s": None,
                        "decision_reasons": [],
                        "decision_reason": "No alert emitted for this observation/frame.",
                        "observation_error": None,
                    }
                )
        preview_path = view.video_path
        if preview_path is not None and not (
            _inside(self.outputs_root, preview_path)
            or _inside(self.datasets_root, preview_path)
            or _inside(self.examples_root, preview_path)
        ):
            preview_path = None
        image = self._read_video_frame(preview_path, record.frame_index)
        if preview_path is None:
            image_message = "; video preview unavailable"
        elif image is None:
            image_message = "; selected video frame could not be decoded"
        else:
            image_message = "; preview frame decoded"
        info = (
            f"Requested {requested:.3f} ms; selected frame {record.frame_index} at "
            f"{record.timestamp_ms:.3f} ms ({record.status}, {len(observations)} observations, "
            f"{len(current_alerts)} alerts){image_message}."
        )
        return (
            image,
            pandas.DataFrame.from_records(observations),
            pandas.DataFrame.from_records(current_alerts),
            info,
        )

    @staticmethod
    def _read_video_frame(video_path: Path | None, frame_index: int) -> np.ndarray | None:
        if video_path is None or not video_path.is_file():
            return None
        try:
            import cv2

            capture = cv2.VideoCapture(str(video_path))
            try:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok or frame is None:
                    return None
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            finally:
                capture.release()
        except (ImportError, OSError, ValueError, RuntimeError):
            return None

    def export_analysis(self, run_id: str) -> Path:
        view = self.load_analysis(run_id)
        run_dir = self._run_path(run_id)
        export_dir = self.outputs_root / "console" / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        if not _inside(self.outputs_root, export_dir):
            raise ConsoleDataError("Export directory escapes outputs root")
        archive = export_dir / f"{uuid.uuid4().hex}.zip"
        created_by_call = False
        try:
            with zipfile.ZipFile(archive, mode="x", compression=zipfile.ZIP_DEFLATED) as bundle:
                created_by_call = True
                tracks_metadata = view.metadata.get("perception_metadata")
                if not isinstance(tracks_metadata, dict):
                    tracks_metadata = view.metadata
                tracks_path = self._analysis_tracks_path(run_dir, tracks_metadata)
                if tracks_path is not None:
                    bundle.write(tracks_path, arcname="tracks.jsonl")
                metadata_path = run_dir / "run_metadata.json"
                if not metadata_path.is_file() and (run_dir / "perception").is_dir():
                    metadata_path = run_dir / "perception" / "run_metadata.json"
                if metadata_path.is_file() and _inside(self.outputs_root, metadata_path):
                    bundle.write(metadata_path, arcname="run_metadata.json")
                console_metadata_path = run_dir / "console_metadata.json"
                if not console_metadata_path.is_file() and run_dir.name == "perception":
                    console_metadata_path = run_dir.parent / "console_metadata.json"
                if console_metadata_path.is_file() and _inside(
                    self.outputs_root, console_metadata_path
                ):
                    bundle.write(console_metadata_path, arcname="console_metadata.json")
                risk_dir = self._analysis_risk_dir(run_dir, tracks_path)
                if risk_dir is not None:
                    for name in (
                        "risk_frames.jsonl",
                        "events.jsonl",
                        "alerts.jsonl",
                        "run_metadata.json",
                    ):
                        source = risk_dir / name
                        if source.is_file() and _inside(self.outputs_root, source):
                            bundle.write(source, arcname=f"assessment/{name}")
                video_path = view.video_path
                if video_path is not None and (
                    _inside(self.outputs_root, video_path)
                    or _inside(self.datasets_root, video_path)
                    or _inside(self.examples_root, video_path)
                ):
                    bundle.write(video_path, arcname=f"preview/{video_path.name}")
            return archive
        except Exception:
            if created_by_call:
                archive.unlink(missing_ok=True)
            raise

    def configs(self, kind: str) -> list[str]:
        loaders = {
            "risk": load_risk_config,
            "benchmark": load_benchmark_config,
            "perception": load_config,
        }
        if kind not in loaders:
            raise ConsoleDataError("Configuration kind must be risk, benchmark, or perception")
        if not self.configs_root.is_dir():
            return []
        loader = loaders[kind]
        result = []
        for path in sorted((*self.configs_root.rglob("*.yaml"), *self.configs_root.rglob("*.yml"))):
            if not path.is_file() or not _inside(self.configs_root, path):
                continue
            try:
                loader(path)
            except (OSError, ValueError, TypeError):
                # Configs for the other supported categories share this folder.
                # Only advertise files that pass the selected schema loader.
                continue
            result.append(path.relative_to(self.configs_root).as_posix())
        return result

    def config_path(self, identifier: str) -> Path:
        path = self._resolve_id(self.configs_root, identifier, "Configuration")
        if path.suffix.lower() not in (".yaml", ".yml") or not path.is_file():
            raise ConsoleDataError("Configuration must name an existing YAML file under configs/")
        return path

    def strategy_preview(
        self,
        manifest_id: str,
        video_id: str,
        strategies: tuple[str, ...],
        benchmark_config_id: str,
        risk_config_id: str,
    ) -> BenchmarkView:
        if (
            not strategies
            or len(set(strategies)) != len(strategies)
            or set(strategies) - {"A", "B", "C"}
        ):
            raise ConsoleDataError("Choose one or more unique strategies from A, B, C")
        manifest_path = self._manifest_path(manifest_id)
        try:
            entries = read_manifest(manifest_path)
        except (OSError, ValueError) as exc:
            raise ConsoleDataError(f"Cannot read dataset manifest: {exc}") from exc
        entry = next((candidate for candidate in entries if candidate.video_id == video_id), None)
        if entry is None:
            raise ConsoleDataError(f"Video {video_id!r} is not present in {manifest_id}")
        tracks_path = self._safe_perception_file(entry.tracks_path, "Perception")
        if not tracks_path.is_file() or not re.fullmatch(r"[0-9a-f]{64}", entry.tracks_sha256):
            raise ConsoleDataError("Perception JSONL or its SHA-256 provenance is missing")
        if sha256_file(tracks_path) != entry.tracks_sha256:
            raise ConsoleDataError("Perception JSONL hash differs from the selected manifest")
        try:
            records = tuple(replay_records(tracks_path))
        except (OSError, ValueError, TypeError) as exc:
            raise ConsoleDataError(f"Invalid perception JSONL: {exc}") from exc
        if not records:
            raise ConsoleDataError("Cannot preview strategies without perception frames")
        risk_config_path = self.config_path(risk_config_id)
        benchmark_config_path = self.config_path(benchmark_config_id)
        try:
            risk_config: RiskConfig = load_risk_config(risk_config_path)
            benchmark_config: BenchmarkConfig = load_benchmark_config(benchmark_config_path)
            run = run_strategies(records, video_id, risk_config, benchmark_config, strategies)
        except (OSError, ValueError, TypeError, RuntimeError) as exc:
            raise ConsoleDataError(f"Strategy preview failed: {type(exc).__name__}: {exc}") from exc
        warnings = (
            "Preview mode evaluates behavior only. Independent Ground Truth was not loaded; "
            "no TP/FP/FN or accuracy metrics were calculated.",
        )
        metadata = {
            "schema_version": "console-preview-1.0",
            "status": "completed",
            "mode": "strategy_preview",
            "ground_truth_evaluated": False,
            "data_kind": entry.data_kind,
            "split": entry.split,
            "video_id": entry.video_id,
            "source_group_id": entry.source_group_id,
            "scene": entry.scene,
            "duration_ms": entry.duration_ms,
            "fps": entry.fps,
            "strategies": tuple(sorted(strategies)),
            "benchmark_config": benchmark_config.to_dict(),
            "risk_config": risk_config.to_dict(),
            "tracks_sha256": entry.tracks_sha256,
            "manifest_id": manifest_id,
        }
        alert_rows = tuple(
            json.loads(alert.to_json()) for name in sorted(run.alerts) for alert in run.alerts[name]
        )
        return BenchmarkView(
            run_id=f"preview:{manifest_id}:{video_id}",
            metadata=metadata,
            metrics={},
            alerts=alert_rows,
            matches=(),
            tables={"alerts": _pandas().DataFrame.from_records(alert_rows)},
            labels={},
            warnings=warnings,
        )
