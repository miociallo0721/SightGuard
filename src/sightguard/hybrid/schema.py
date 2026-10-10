"""Versioned, UI-independent decision contracts and explicit ordinal mapping."""

import json
import math
from dataclasses import asdict, dataclass
from typing import Any

LEVELS = ("LOW", "MEDIUM", "HIGH")
CRITERIA = {
    "LOW": "No notable concern indicated by the observations; research category only.",
    "MEDIUM": "Possible concern needing continued attention; research category only.",
    "HIGH": "Strong concern deserving heightened attention; research category only.",
    "UNKNOWN": "Insufficient or unclear evidence to select a concern category.",
}


def questions() -> dict:
    return {
        "risk": {
            "type": "choice",
            "instructions": "Choose a research concern category; no safety prediction.",
            "criteria": CRITERIA,
        }
    }


@dataclass(frozen=True)
class DecisionResult:
    result_id: str
    session_id: str
    source: str
    model_name: str
    model_revision: str
    frame_id: int
    timestamp_ms: float
    event_id: str | None
    raw_output: Any
    candidate_risk_level: str | None
    candidate_score: float | None
    confidence_semantics: str
    inference_latency_ms: float
    inference_started_at: str
    inference_completed_at: str
    stale: bool = False
    valid: bool = True
    error: str | None = None
    schema_version: str = "1.0"

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)

    @classmethod
    def from_json(cls, value: str) -> "DecisionResult":
        data = json.loads(value)
        if data.get("schema_version") != "1.0":
            raise ValueError("Unsupported decision schema")
        return cls(**data)


def parse_output(raw: dict, source: str) -> tuple[str | None, float | None, str | None]:
    """Read upstream typed choice, never treat its confidence as calibrated risk probability."""
    if not isinstance(raw, dict):
        return None, None, "Model output is not a mapping"
    answers = raw.get("answers", raw)
    answer = answers.get("risk") if isinstance(answers, dict) else None
    if not isinstance(answer, dict):
        return None, None, "Missing typed risk answer"
    choice = answer.get("choice")
    # Vision may return the text of a list candidate rather than its label.
    if source == "vision" and choice in CRITERIA.values():
        choice = next(key for key, value in CRITERIA.items() if value == choice)
    if choice not in (*LEVELS, "UNKNOWN"):
        return None, None, f"Unrecognized candidate: {choice!r}"
    score = answer.get("answer_confidence", answer.get("confidence"))
    if score is not None:
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
        ):
            return None, None, "Non-finite or non-numeric candidate score"
        score = float(score)
    return (
        (None if choice == "UNKNOWN" else choice),
        score,
        ("Insufficient evidence (UNKNOWN)" if choice == "UNKNOWN" else None),
    )


def fuse(rule: str | None, auxiliary: str | None, policy: str) -> str | None:
    if rule not in LEVELS or auxiliary not in LEVELS:
        return None
    if policy == "consensus":
        return rule if rule == auxiliary else None
    if policy != "conservative_max":
        raise ValueError("Unknown fusion policy")
    return max((rule, auxiliary), key=LEVELS.index)
