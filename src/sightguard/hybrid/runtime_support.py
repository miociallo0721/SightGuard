"""Precision adaptation for the pinned Vision readout; imports no model framework."""

import os
from typing import Any


def load_vision(laya: Any, model_dir: str) -> Any:
    """Use the documented reader backend to avoid Windows mmap native failures.

    Scoped to the single-owner isolated model process. Neither upstream source nor
    the checkpoint changes; every tensor still comes from the same safetensors file.
    """
    if os.name != "nt":
        return laya.load_vlm(model_dir, device="cpu", dtype="fp32")
    from safetensors import torch as storage

    original = storage.load_file

    def read_file(filename: str, device: str = "cpu", **kwargs: Any) -> dict:
        kwargs.setdefault("backend", "pread")
        return original(filename, device=device, **kwargs)

    storage.load_file = read_file
    try:
        return laya.load_vlm(model_dir, device="cpu", dtype="fp32")
    finally:
        storage.load_file = original


def configure_vision(agent: Any, torch: Any, device: str, precision: str) -> dict:
    """Keep the upstream float32 readout unchanged; cast only the CUDA encoder.

    The pinned VLMDecisionModel._readout explicitly calls h.float(), so converting
    its Transformer/head weights to float16 causes a dtype mismatch. This is a
    mixed precision adapter, not a claim that every parameter uses FP16.
    """
    if precision not in {"fp32", "fp16"}:
        raise ValueError("Vision requires fp32 or fp16")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA runtime unavailable in selected isolated environment")
    if device == "cpu" and precision != "fp32":
        raise ValueError("Vision CPU requires FP32")
    dtype = torch.float16 if precision == "fp16" else torch.float32
    agent.model.to(device=device, dtype=torch.float32)
    if precision == "fp16":
        agent.model.encoder.to(dtype=dtype)
    agent.device = torch.device(device)
    agent._torch_dtype = lambda: dtype
    counts: dict[str, int] = {}
    devices: dict[str, int] = {}
    for parameter in agent.model.parameters():
        key = str(parameter.dtype)
        counts[key] = counts.get(key, 0) + parameter.numel()
        key = str(parameter.device)
        devices[key] = devices.get(key, 0) + parameter.numel()
    return {
        "precision": "fp16_encoder_fp32_readout" if precision == "fp16" else "fp32",
        "parameter_dtype_counts": counts,
        "parameter_device_counts": devices,
    }
