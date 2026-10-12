"""Export a separate resolution experiment with unchanged frozen SegFormer weights.

Never changes Android assets or the existing 256 export. CPU-only export and numerical
PyTorch/ONNX comparison reuse the established exporter and its checkpoint SHA gate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import export_segformer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, default=export_segformer.IMAGE)
    parser.add_argument("--size", type=int, choices=(320, 384, 512), default=384)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    # This isolated CLI process owns the module; the default exporter on disk is untouched.
    export_segformer.INPUT_SIZE = args.size
    result = export_segformer.export_and_validate(
        checkpoint=args.checkpoint,
        image_path=args.image,
        output_path=args.output / f"segformer_{args.size}.onnx",
        report_path=args.output / "export-validation.json",
        input_binary_path=args.output / "input.tensor.bin",
        logits_binary_path=args.output / "reference.logits.bin",
        warmup_runs=1,
        timed_runs=3,
    )
    print(result)


if __name__ == "__main__":
    main()
