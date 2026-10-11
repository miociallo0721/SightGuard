"""Diagnostic fixtures must preserve abstention and input distinctions."""

import importlib
import sys
from pathlib import Path


def investigation_module():
    scripts = str(Path(__file__).resolve().parents[1] / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    return importlib.import_module("investigate_vision_unknown")


def test_recursive_non_finite_output_is_rejected() -> None:
    module = investigation_module()
    assert module.finite_output({"answers": [{"confidence": 0.1}]})
    assert not module.finite_output({"answers": [{"confidence": float("nan")}]})
    assert not module.finite_output({"logits": [float("inf")]})


def test_fixture_variants_and_unknown_candidate_are_preserved() -> None:
    from sightguard.hybrid.schema import CRITERIA

    module = investigation_module()
    first = module.create_cases()
    second = module.create_cases()
    assert len(first) == 6
    assert [row["image"].tobytes() for row in first] == [row["image"].tobytes() for row in second]
    assert len({row["image"].tobytes() for row in first[:4]}) == 4
    assert CRITERIA["UNKNOWN"] in module.task_questions()["risk"]["criteria"]
    assert module.task_questions()["risk"]["type"] == "choice"
