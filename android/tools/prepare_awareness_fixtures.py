"""Restore hash-pinned real-scene test fixtures; network is opt-in and never part of JVM tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def fetch(url: str) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": "SightGuard-research-validation/0.4"}
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        return response.read()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    assets = ROOT / "android/app/src/androidTest/assets"
    manifest = json.loads((assets / "awareness-scenes.json").read_text(encoding="utf-8"))
    cache = ROOT / "android/deliveries/accessible-awareness/real-scenes"
    output = assets / "awareness"
    output.mkdir(parents=True, exist_ok=True)
    failures = []
    for case in manifest["cases"]:
        name = case["file"]
        candidate = next(
            (p for p in (output / name, cache / name, assets / name) if p.is_file()), None
        )
        try:
            if candidate is None:
                if not args.download:
                    raise FileNotFoundError(f"{name}: use --download or restore the local cache")
                source = case["source"]
                if "row_index" in source:
                    query = urllib.parse.urlencode(
                        dict(
                            dataset=source["dataset"],
                            config="default",
                            split=source["split"],
                            offset=source["row_index"],
                            length=1,
                        )
                    )
                    row = json.loads(fetch("https://datasets-server.huggingface.co/rows?" + query))[
                        "rows"
                    ][0]
                    url = row["row"]["image"]["src"]
                    if source["revision"] not in url:
                        raise ValueError(
                            "Dataset revision changed; do not replace validation bytes silently"
                        )
                else:
                    url = source.get("url", source.get("downloadUrl"))
                data = fetch(url)
            else:
                data = candidate.read_bytes()
            actual = hashlib.sha256(data).hexdigest()
            if actual != case["sha256"]:
                raise ValueError(f"Input hash differs: {actual}")
            destination = output / name
            if candidate != destination:
                destination.write_bytes(data)
            print(f"Verified {name}")
        except (OSError, ValueError, KeyError) as error:
            failures.append(dict(file=name, failure=str(error)))
    report = ROOT / "android/deliveries/accessible-awareness/fixture-preparation.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(
            dict(schema_version=1, failures=failures, count=len(manifest["cases"])), indent=2
        ),
        encoding="utf-8",
    )
    if failures:
        raise SystemExit(json.dumps(failures, ensure_ascii=False))


if __name__ == "__main__":
    main()
