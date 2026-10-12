"""Export the evaluated frozen YOLOE prompt profile; verify same-tensor ONNX parity."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
from export_model import flatten_tensors, letterbox_rgb, sha256
from ultralytics import YOLOE

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "android/deliveries/yoloe26-quick"
ASSETS = ROOT / "android/app/src/main/assets"


def main() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    model = YOLOE(str(WORK / "yoloe26n-fixed.pt"))
    assert len(model.names) == 90 and model.task == "detect"
    target = Path(
        model.export(
            format="onnx",
            imgsz=320,
            batch=1,
            dynamic=False,
            simplify=False,
            opset=17,
            nms=False,
            agnostic_nms=True,
            end2end=True,
            device="cpu",
        )
    )
    onnx.checker.check_model(onnx.load(str(target)))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(target), options, providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].shape == [1, 3, 320, 320]
    assert len(session.get_outputs()) == 1 and session.get_outputs()[0].shape[-1] == 6
    fixtures = [
        ROOT / "android/app/src/androidTest/assets/validation.jpg",
        ROOT / "datasets/imported/ade20k-furniture-v06/ADE_val_00000009.jpg",
        ROOT / "datasets/imported/ade20k-furniture-v06-holdout/ADE_val_00000166.jpg",
    ]
    report = []
    model.model.eval()
    head = model.model.model[-1]
    head.end2end = True
    head.export = True
    head.agnostic_nms = True
    # Export fuses/reparameterizes a copy; original prompt model must also use matching head.
    for image in fixtures:
        values, prep = letterbox_rgb(cv2.imread(str(image)))
        out = session.run(None, {session.get_inputs()[0].name: values})[0]
        with torch.inference_mode():
            pt = flatten_tensors(model.model(torch.from_numpy(values)))[0]
        # Top-K may reorder tied low-score rows; compare meaningful detections by class/geometry.
        selected = pt[0][pt[0, :, 4] >= 0.35]
        matches = []
        for row in selected:
            candidates = out[0][out[0, :, 5] == row[5]]
            assert len(candidates) > 0
            best = candidates[np.abs(candidates[:, :4] - row[:4]).max(axis=1).argmin()]
            delta = np.abs(best - row)
            assert delta[:4].max() < 0.02 and delta[4] < 0.001, delta
            matches.append(float(delta.max()))
        assert len(selected) == int((out[0, :, 4] >= 0.35).sum())
        report.append(
            {
                "image": image.name,
                "image_sha256": sha256(image),
                "preprocessing": prep,
                "detections": out[0][out[0, :, 4] >= 0.35].tolist(),
                "matched": len(matches),
                "max_matched_delta": max(matches, default=0.0),
            }
        )
        if image.name == "validation.jpg":
            values.astype("<f4").tofile(WORK / "input.tensor.bin")
            out.astype("<f4").tofile(WORK / "reference.output.bin")
    manifest = {
        "schema_version": "sightguard.yoloe26.v1",
        "model_name": "YOLOE-26n fixed 90 prompts",
        "onnx_sha256": sha256(target),
        "source_weights_sha256": sha256(WORK / "yoloe-26n-seg.pt"),
        "prompt_sha256": sha256(WORK / "prompts.npz"),
        "input_shape": [1, 3, 320, 320],
        "output_shape": session.get_outputs()[0].shape,
        "names": list(model.names.values()),
        "app_class_ids": list(range(80)) + [60, 60, 56, 56, 56, 80, 80, 81, 82, 83],
        "min_score": 0.35,
        "opset": 17,
        "runtime": "ONNX Runtime CPU",
        "validation": report,
        "source": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26n-seg.pt",
    }
    (WORK / "export-validation.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    shutil.copy2(target, ASSETS / "yoloe26n.onnx")
    (ASSETS / "yoloe_model_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    test_assets = ROOT / "android/app/src/androidTest/assets"
    shutil.copy2(WORK / "input.tensor.bin", test_assets / "yoloe_input.bin")
    shutil.copy2(WORK / "reference.output.bin", test_assets / "yoloe_output.bin")
    shutil.copy2(fixtures[2], test_assets / "yoloe_table.jpg")
    print(
        json.dumps(
            {
                "sha": manifest["onnx_sha256"],
                "shape": manifest["output_shape"],
                "validation": report,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
