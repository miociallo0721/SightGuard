"""Real ORT CPU scene inference; no Torch import or synthetic masks."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "android/parity-validation/v03-semantics-python"
ASSETS = ROOT / "android/app/src/androidTest/assets"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    model = ROOT / "android/spatial-models/segformer_mit_b0_ade20k_256.onnx"
    sources = json.loads((OUT / "sources.json").read_text(encoding="utf8"))
    labels = json.loads(
        (ROOT / "android/app/src/main/assets/ade20k_labels.json").read_text(encoding="utf8")
    )["labels"]
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
    report = {
        "schemaVersion": "sightguard.v03-real-semantics.v1",
        "createdAtUtc": datetime.now(UTC).isoformat(),
        "modelSha256": sha(model),
        "modelPath": str(model),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "onnxruntime": ort.__version__,
        "torchImported": "torch" in sys.modules,
        "minimumConfidence": 0.35,
        "minimumRegionArea": 0.003,
        "timeSource": "static real photographs; inference latency uses perf_counter",
        "limitations": [
            "Class outputs are model predictions, not independent ground truth or risk estimates.",
            "OpenCV INTER_LINEAR resize differs from Android Bitmap bilinear; "
            "same-tensor parity isolates runtime differences.",
            "Selection is a small convenience sample, not an accuracy benchmark.",
        ],
        "samples": [],
        "downloadAttempts": sources,
    }
    for source in sources:
        if source["status"] != "downloaded":
            continue
        path = OUT / source["file"]
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot decode {path}")
        h, w = image.shape[:2]
        scale = min(256 / w, 256 / h)
        rw, rh = max(1, int(w * scale + 0.5)), max(1, int(h * scale + 0.5))
        dx, dy = (256 - rw) // 2, (256 - rh) // 2
        rgb = (
            cv2.resize(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (rw, rh), interpolation=cv2.INTER_LINEAR
            ).astype(np.float32)
            / 255
        )
        normalized = (rgb - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        tensor = np.zeros((1, 3, 256, 256), dtype=np.float32)
        tensor[0, :, dy : dy + rh, dx : dx + rw] = normalized.transpose(2, 0, 1)
        start = time.perf_counter()
        logits = session.run(["class_logits"], {"pixel_values": tensor})[0]
        elapsed = (time.perf_counter() - start) * 1000
        assert logits.shape == (1, 150, 64, 64) and np.isfinite(logits).all()
        flat = logits[0].reshape(150, -1)
        best = flat.argmax(axis=0)
        maximum = flat.max(axis=0)
        # Match Kotlin: float32 subtraction followed by double exp/sum.
        confidence = (1 / np.exp((flat - maximum).astype(np.float64)).sum(axis=0)).astype(
            np.float32
        )
        cells = []
        counts: dict[int, int] = {}
        for pixel in range(4096):
            left, right = np.clip(
                [(pixel % 64 * 4 - dx) / rw, ((pixel % 64 + 1) * 4 - dx) / rw], 0, 1
            )
            top, bottom = np.clip(
                [(pixel // 64 * 4 - dy) / rh, ((pixel // 64 + 1) * 4 - dy) / rh], 0, 1
            )
            if right <= left or bottom <= top:
                continue
            cls = int(best[pixel])
            counts[cls] = counts.get(cls, 0) + 1
            if confidence[pixel] >= 0.35:
                cells.append(
                    dict(
                        classId=cls,
                        confidence=float(confidence[pixel]),
                        left=float(left),
                        top=float(top),
                        right=float(right),
                        bottom=float(bottom),
                    )
                )
        distribution = []
        for cls, count in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
            accepted = [cell for cell in cells if cell["classId"] == cls]
            distribution.append(
                dict(
                    classId=cls,
                    rawLabel=labels[cls]["rawLabel"],
                    argmaxCells=count,
                    acceptedCells=len(accepted),
                    acceptedAreaFraction=sum(
                        (c["right"] - c["left"]) * (c["bottom"] - c["top"]) for c in accepted
                    ),
                )
            )
        sample = dict(
            file=source["file"],
            imageSha256=sha(path),
            width=w,
            height=h,
            resizedWidth=rw,
            resizedHeight=rh,
            padLeft=dx,
            padTop=dy,
            preprocessing=(
                "RGB ImageNet mean/std; center letterbox; zero normalized padding; INTER_LINEAR"
            ),
            tensorSha256=hashlib.sha256(tensor.tobytes()).hexdigest(),
            inferenceMs=elapsed,
            distribution=distribution,
            outputLogitsSha256=hashlib.sha256(logits.tobytes()).hexdigest(),
        )
        (OUT / (path.stem + ".cells.json")).write_text(
            json.dumps(cells, indent=2) + "\n", encoding="utf8"
        )
        shutil.copyfile(path, ASSETS / source["file"])
        if source["file"] == "v03-room.jpg":
            for name, values in [
                ("v03-semantic.tensor.bin", tensor),
                ("v03-semantic.logits.bin", logits),
            ]:
                binary = OUT / name
                values.astype("<f4").tofile(binary)
                shutil.copyfile(binary, ASSETS / name)
            sample["sameTensorReference"] = {
                "inputFile": "v03-semantic.tensor.bin",
                "logitsFile": "v03-semantic.logits.bin",
                "dtype": "float32",
                "byteOrder": "little",
                "inputShape": [1, 3, 256, 256],
                "outputShape": [1, 150, 64, 64],
            }
        report["samples"].append(sample)
        print(source["file"], [(d["rawLabel"], d["acceptedCells"]) for d in distribution[:10]])
    assert "torch" not in sys.modules
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    shutil.copyfile(OUT / "report.json", ASSETS / "v03-semantic-reference.json")
    shutil.copyfile(OUT / "sources.json", ASSETS / "v03-semantic-sources.json")


if __name__ == "__main__":
    main()
