"""CPU-only unit coverage for GPU benchmark record interpretation."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from benchmark_hybrid_gpu import (
    attach_call_timing,
    parse_model_output,
    summarize_ort_profile,
)  # noqa: E402

from sightguard.hybrid.schema import CRITERIA, parse_output


def test_ort_profile_reports_mixed_cpu_cuda_node_providers(tmp_path: Path) -> None:
    profile = tmp_path / "ort_profile.json"
    profile.write_text(
        json.dumps(
            [
                {
                    "cat": "Node",
                    "name": "MatMul_kernel_time",
                    "dur": 12,
                    "args": {"provider": "CUDAExecutionProvider", "op_name": "MatMul"},
                },
                {
                    "cat": "Node",
                    "name": "QuantizeLinear_kernel_time",
                    "dur": 34,
                    "args": {"provider": "CPUExecutionProvider", "op_name": "QuantizeLinear"},
                },
                {"cat": "Session", "name": "session_init", "dur": 500, "args": {}},
            ]
        ),
        encoding="utf-8",
    )

    summary = summarize_ort_profile(profile)

    assert summary["execution_conclusion_from_profiled_nodes"] == "cuda_and_cpu_mixed"
    assert summary["dominant_node_provider_by_profiled_duration"] == "CPUExecutionProvider"
    assert summary["node_provider_counts_and_durations"] == {
        "CUDAExecutionProvider": {"node_count": 1, "duration_us": 12.0},
        "CPUExecutionProvider": {"node_count": 1, "duration_us": 34.0},
    }
    assert summary["cuda_node_count_share"] == 0.5
    assert summary["cuda_node_duration_share"] == pytest.approx(12 / 46, abs=0.0001)
    assert {item["node_name"] for item in summary["node_names_by_provider"]} == {
        "MatMul",
        "QuantizeLinear",
    }


def test_vision_phase_timing_counts_encoder_and_forward_separately() -> None:
    events = [
        {
            "component": "vision_image_encoder",
            "started": 10.02,
            "completed": 10.12,
            "duration_ms": 100.0,
        },
        {
            "component": "vision_decision_forward",
            "started": 10.15,
            "completed": 10.20,
            "duration_ms": 50.0,
        },
    ]

    timing = attach_call_timing(10.0, 10.30, events)

    assert timing["preprocessing_before_first_model_call_ms"] == pytest.approx(20.0)
    assert timing["model_forward_ms"] == pytest.approx(150.0)
    assert timing["between_model_calls_ms"] == pytest.approx(30.0)
    assert timing["model_postprocessing_after_last_call_ms"] == pytest.approx(100.0)
    assert [item["component"] for item in timing["model_component_calls"]] == [
        "vision_image_encoder",
        "vision_decision_forward",
    ]


@pytest.mark.parametrize(
    ("source", "choice"),
    [
        ("laya", "UNKNOWN"),
        ("vision", CRITERIA["UNKNOWN"]),
    ],
)
def test_unknown_abstention_is_valid_output(source: str, choice: str) -> None:
    output = {"answers": {"risk": {"choice": choice, "answer_confidence": 0.0}}}

    parsed = parse_model_output(output, source, parse_output)

    assert parsed["status"] == "valid_unknown"
    assert parsed["candidate_risk_level"] is None
    assert parsed["error"] is None
