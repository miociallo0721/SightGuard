"""Opt-in real camera/YOLO/auxiliary smoke. Never downloads or changes model config."""

import argparse
import json
import os
import time
from dataclasses import replace
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--seconds-per-mode", type=float, default=22)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    temporary = root / "models" / "hybrid" / "tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    for key in ("TEMP", "TMP", "TMPDIR", "HF_HOME", "TORCH_HOME"):
        os.environ[key] = str(temporary)
    from sightguard.config import load_config
    from sightguard.live.config import load_live_config
    from sightguard.live.runtime import LiveSession
    from sightguard.risk.config import load_risk_config

    session = LiveSession(
        replace(load_config(root / "configs/default.yaml"), device="cpu"),
        load_risk_config(root / "configs/risk.yaml"),
        replace(load_live_config(root / "configs/live.yaml"), camera=args.camera),
    )
    summary = {"schema_version": "1.0", "camera": args.camera, "modes": []}
    try:
        session.start()
        time.sleep(3)
        manager = session.enable_intelligence(root)
        time.sleep(2)  # Collect rule-only risk updates before enabling an auxiliary.
        engine = session.engine
        for mode, auxiliary in (("laya", "laya"), ("vision", "vision"), ("hybrid", "laya")):
            manager.select(mode, auxiliary)
            deadline = time.monotonic() + args.seconds_per_mode
            while time.monotonic() < deadline:
                error, _ = session.failure()
                if error:
                    raise RuntimeError(error)
                time.sleep(0.1)
            snapshot = manager.snapshot()
            snapshot["same_rule_engine"] = session.engine is engine
            result, alerts = session.snapshot()
            snapshot["actual_recent_alerts"] = [a.to_json() for a, _ in alerts]
            snapshot["risk_fps"] = result.update_fps if result else None
            summary["modes"].append(snapshot)
            print(
                json.dumps(
                    {
                        "mode": mode,
                        "status": snapshot["model_status"],
                        "performance": snapshot["performance"],
                    }
                ),
                flush=True,
            )
        manager.select("rule")
        summary["rule_return"] = manager.snapshot()
        summary["output_directory"] = str(manager.output)
        summary["status"] = (
            "success"
            if all(
                m["same_rule_engine"]
                and m["latest_result"] is not None
                and m["latest_result"]["raw_output"] is not None
                for m in summary["modes"]
            )
            else "failed"
        )
    except Exception as exc:
        summary["status"], summary["error"] = "failed", f"{type(exc).__name__}: {exc}"
    finally:
        session.close()
    output = root / "outputs/hybrid/p0/live-smoke.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{summary['status']}: {output}")
    return 0 if summary["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
