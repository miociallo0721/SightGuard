"""Small offline API/choice investigation; preserves abstention and raw outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from benchmark_hybrid_gpu import (
    build_fixed_input,
    configure_local_runtime,
    sha256_file,
    synchronize,
    torch_cuda_memory,
    utc_now,
    validate_vision_model,
    vision_parameter_profile,
)

ROOT = Path(__file__).resolve().parents[1]


def finite_output(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_output(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_output(item) for item in value)
    return True


def create_cases() -> list[dict[str, Any]]:
    """Self-authored deterministic diagrams plus the pinned upstream tutorial photo."""
    from PIL import Image, ImageDraw, ImageFilter

    cases = []
    for name in ("single_red_ball", "front_box", "open_floor", "obscured_box"):
        image = Image.new("RGB", (640, 480), (230, 235, 240))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 180, 639, 479), fill=(190, 185, 175))
        if name == "single_red_ball":
            draw.ellipse((240, 220, 400, 380), fill=(230, 20, 25))
        elif name in {"front_box", "obscured_box"}:
            draw.rectangle((210, 160, 430, 400), fill=(130, 75, 30), outline=(40, 25, 10), width=5)
            draw.line((210, 160, 320, 220, 430, 160), fill=(40, 25, 10), width=5)
        if name == "obscured_box":
            draw.rectangle((0, 0, 420, 479), fill=(85, 85, 85))
            image = image.filter(ImageFilter.GaussianBlur(25))
        cases.append(
            {"id": name, "image": image, "source": "self-authored PIL diagram; not a photo"}
        )
    photo = ROOT / "models/hybrid/laya-vision-source/site-docs/tutorials/example.jpg"
    cases.append(
        {
            "id": "upstream_dj_photo",
            "image": Image.open(photo).convert("RGB"),
            "source": str(photo.relative_to(ROOT)),
            "source_sha256": sha256_file(photo),
        }
    )
    image, state, questions = build_fixed_input()
    cases.append(
        {
            "id": "original_fixture",
            "image": image,
            "source": "existing P0 fixture",
            "state": state,
            "questions": questions["vision"],
        }
    )
    return cases


def task_questions() -> dict[str, dict[str, Any]]:
    from sightguard.hybrid.schema import questions

    risk = questions()["risk"]
    return {
        "attribute": {
            "type": "choice",
            "instructions": "What is the color of the main foreground object?",
            "criteria": [
                "red",
                "brown",
                "black",
                "There is no foreground object",
                "Cannot determine its color",
            ],
        },
        "presence": {
            "type": "choice",
            "instructions": (
                "Describe only the visible foreground, without inferring collision danger."
            ),
            "criteria": [
                "A foreground object is visible",
                "No foreground object is visible",
                "The image is too unclear to determine object presence",
            ],
        },
        "risk": {**risk, "criteria": list(risk["criteria"].values())},
        "observation": {
            "type": "choice",
            "instructions": (
                "Which statement describes the central lower part of the image? "
                "Judge visible occupancy only, not walking safety."
            ),
            "criteria": [
                "An object occupies the central lower area",
                "The central lower area is visibly empty",
                "The central lower area cannot be assessed from this image",
            ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--precision", choices=("fp32", "fp16"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure_local_runtime(ROOT)
    sys.path.insert(0, str(ROOT / "src"))
    import laya
    import torch

    from sightguard.hybrid.runtime_support import configure_vision, load_vision
    from sightguard.hybrid.schema import CRITERIA, parse_output

    args.output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "schema_version": "sightguard.vision-investigation.v1",
        "started_at": utc_now(),
        "device": args.device,
        "precision": args.precision,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "script_sha256": sha256_file(Path(__file__)),
        "records": [],
        "limitations": [
            "Small diagnostic set, no risk accuracy claim",
            "Synthetic scenes are diagrams",
            "UNKNOWN is an explicit project candidate, not an automatic upstream threshold",
        ],
    }
    hooks = []
    try:
        report["model"] = validate_vision_model(ROOT / "models/hybrid/vision")
        load_start = time.perf_counter()
        agent = load_vision(laya, str(ROOT / "models/hybrid/vision"))
        report["adaptation"] = configure_vision(agent, torch, args.device, args.precision)
        agent.model.eval()
        synchronize(torch, args.device)
        report["load_ms"] = (time.perf_counter() - load_start) * 1000
        report["parameters"] = vision_parameter_profile(agent)
        report["memory_after_load"] = torch_cuda_memory(torch, args.device)
        tensor_evidence = []

        def tensor_description(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                return {
                    "device": str(value.device),
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                }
            if isinstance(value, dict):
                return {key: tensor_description(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [tensor_description(item) for item in value]
            return None

        def observe(name: str):
            def hook(module: Any, inputs: tuple, kwargs: dict, output: Any) -> None:
                if len(tensor_evidence) < 12:
                    tensor_evidence.append(
                        {
                            "module": name,
                            "inputs": tensor_description(inputs),
                            "kwargs": tensor_description(kwargs),
                            "output": tensor_description(output),
                        }
                    )

            return hook

        hooks.append(agent.model.register_forward_hook(observe("decision_model"), with_kwargs=True))
        hooks.append(
            agent.model.scorer.register_forward_hook(observe("fp32_scorer"), with_kwargs=True)
        )
        hooks.append(
            agent.model.encoder.vision_model.register_forward_hook(
                observe("vision_tower"), with_kwargs=True
            )
        )
        for case in create_cases():
            image = case.pop("image")
            image.save(args.output / f"{case['id']}.png")
            note = json.dumps(case.pop("state", {}), sort_keys=True)
            q = case.pop("questions", task_questions())
            if case["id"] == "upstream_dj_photo":
                note = "listing photo for a used DJ setup"
                q["category"] = {
                    "type": "choice",
                    "instructions": "What kind of item is this?",
                    "criteria": ["electronics", "clothing", "furniture", "food", "other"],
                }
                q["record"] = {
                    "type": "noul",
                    "instructions": "Is there a vinyl record on one of the turntables?",
                }
                q["condition"] = {
                    "type": "score",
                    "instructions": "What condition is the equipment in?",
                    "criteria": [
                        "poor: broken or missing parts",
                        "fair: heavy wear",
                        "good: light wear",
                        "like new",
                    ],
                }
            synchronize(torch, args.device)
            start = time.perf_counter()
            logits: dict = {}
            raw = agent.predict(
                {"image": image, "note": note},
                q,
                n_permutations=1,
                batch_size=1,
                prefix_cache=False,
                _raw_logits=logits,
            )
            synchronize(torch, args.device)
            elapsed_ms = (time.perf_counter() - start) * 1000
            normalized = {}
            for name, answer in raw["answers"].items():
                choice = answer.get("choice")
                candidates = q[name].get("criteria", [])
                normalized[name] = {
                    "type": answer["type"],
                    "choice": choice,
                    "choice_in_requested_candidates": choice in candidates
                    if answer["type"] == "choice"
                    else None,
                    "project_unknown": choice == CRITERIA["UNKNOWN"],
                }
            case.update(
                image_pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                size=list(image.size),
                context=note,
                questions=q,
                raw=raw,
                raw_logits={key: value.tolist() for key, value in logits.items()},
                normalized=normalized,
                finite=finite_output(raw),
                risk_parse=list(parse_output(raw, "vision")),
                latency_ms=elapsed_ms,
            )
            report["records"].append(case)
            print(case["id"], elapsed_ms, normalized, flush=True)
        report["tensor_execution_evidence"] = tensor_evidence
        report["memory_after_inference"] = torch_cuda_memory(torch, args.device)
        if args.device == "cuda":
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as prof:
                agent.predict({"image": image, "note": note}, q, batch_size=1, prefix_cache=False)
                torch.cuda.synchronize()
            prof.export_chrome_trace(str(args.output / "cuda_trace.json"))
            report["cuda_kernel_events"] = sum(
                event.device_type == torch.autograd.DeviceType.CUDA for event in prof.events()
            )
        report["status"] = (
            "success" if all(row["finite"] for row in report["records"]) else "non_finite_output"
        )
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        print(report["error"], file=sys.stderr)
    finally:
        for hook in hooks:
            hook.remove()
        report["finished_at"] = utc_now()
        (args.output / "investigation.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )
    return 0 if report["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
