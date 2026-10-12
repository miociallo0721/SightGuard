"""Static APK content integrity only; never Android inference or playback acceptance."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DELIVERY = ROOT / "android/deliveries/metric-corridor-research"


def entry_hash(archive: zipfile.ZipFile, name: str) -> str:
    digest = hashlib.sha256()
    with archive.open(name) as handle:
        while chunk := handle.read(1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    with zipfile.ZipFile(DELIVERY / "SightGuard-metric-corridor-experimental.apk") as new:
        with zipfile.ZipFile(DELIVERY / "baseline/SightGuard-0.4-stable.apk") as old:
            preserved = {}
            for name in old.namelist():
                if name.startswith(("assets/", "res/raw/")) and not name.endswith("/"):
                    preserved[name] = entry_hash(old, name) == entry_hash(new, name)
            if not all(preserved.values()):
                raise ValueError("baseline model/config/audio asset changed")
        manifest = json.loads(new.read("assets/metric_research/metric_manifest.json"))
        models = []
        for record in [manifest["shared_weights"], *manifest["graphs"]]:
            name = "assets/metric_research/" + record.get("file", "metric_weights.bin")
            actual = entry_hash(new, name)
            models.append({"asset": name, "sha256": actual, "matches": actual == record["sha256"]})
        if not all(record["matches"] for record in models):
            raise ValueError("packaged metric asset hash mismatch")
        settings = json.loads(new.read("assets/metric_research_config.json"))
        assert settings["shadow_only"] and not settings["default_enabled"]
        assert not settings["automatic_speech"]
        permissions = (DELIVERY / "apk-permissions.txt").read_text(encoding="utf-8-sig")
        assert "android.permission.INTERNET" not in permissions
        baseline_signer = (DELIVERY / "apk-signature-baseline.txt").read_text()
        signer = (DELIVERY / "apk-signature-experimental.txt").read_text()
        certificate = next(s for s in signer.splitlines() if "certificate SHA-256 digest:" in s)
        assert certificate in baseline_signer
        report = {
            "scope": "static_APK_content_not_device_acceptance",
            "all_baseline_assets_unchanged": True,
            "baseline_asset_count": len(preserved),
            "preserved_assets": preserved,
            "metric_assets": models,
            "default_enabled": False,
            "automatic_speech": False,
            "INTERNET_permission": False,
            "same_debug_certificate": certificate,
        }
    (DELIVERY / "apk-contract-validation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("Static APK checks passed; preserved baseline assets", len(preserved))


if __name__ == "__main__":
    main()
