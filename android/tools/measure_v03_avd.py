"""Measure five real CameraX configurations on the running AVD; no detector injection."""

from __future__ import annotations

import json
import re
import statistics
import time
from pathlib import Path

from avd_ui import adb, dump, tap

OUTPUT = Path(__file__).resolve().parents[1] / "deliveries/v03-final/performance"
CONFIGS = [
    ("yolo", False, False, False),
    ("yolo_semantics", False, True, False),
    ("yolo_depth", True, False, False),
    ("yolo_both", True, True, False),
    ("all_grid_imu_audio", True, True, True),
]


def switch(label: str, enabled: bool) -> None:
    nodes = [n for n in dump().iter("node") if n.get("content-desc") == label]
    if not nodes:
        raise RuntimeError(f"Switch absent: {label}")
    tree = dump()
    parents = {child: parent for parent in tree.iter() for child in parent}
    node = next(n for n in tree.iter("node") if n.get("content-desc") == label)
    while node.get("checkable") != "true" and node in parents:
        node = parents[node]
    if node.get("checkable") != "true":
        raise RuntimeError(f"Toggle state not exposed: {label}")
    checked = node.get("checked") == "true"
    if checked != enabled:
        tap(label, True)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    reports = []
    for name, depth, semantics, grid in CONFIGS:
        tap("空间感知")
        for label, enabled in [
            ("启用相对深度", depth),
            ("启用环境语义", semantics),
            ("启用空间网格", grid),
            ("启用运动传感器", True),
        ]:
            switch(label, enabled)
        tap("返回画面")
        time.sleep(10)
        adb("logcat", "-c")
        print(f"Collecting {name}: 40 s actual CameraX", flush=True)
        time.sleep(40)
        log = adb("logcat", "-d", "-v", "threadtime").decode("utf-8", "replace")
        (OUTPUT / f"{name}.log").write_text(log, encoding="utf-8")
        memory = adb("shell", "dumpsys", "meminfo", "org.sightguard.app").decode()
        (OUTPUT / f"{name}-memory.txt").write_text(memory, encoding="utf-8")
        frames = re.findall(
            r"SG_FRAME.*frame=(\d+) sensorMs=(\d+).*latencyMs=(\d+) yoloMs=(\d+) fps=([\d.]+)", log
        )
        space = re.findall(r"SG_SPACE.*", log)
        pss = re.search(r"TOTAL PSS:\s*(\d+)", memory)
        row = {
            "name": name,
            "warmup_seconds": 10,
            "sample_seconds": 40,
            "sampled_frame_logs": len(frames),
            "pss_kib": int(pss[1]) if pss else None,
            "source": "actual AVD CameraX virtualscene, Android16 API36, CPU",
            "depth": depth,
            "semantics": semantics,
            "grid": grid,
            "spatial_log_samples": space,
            "physical_phone_verified": False,
        }
        if len(frames) >= 2:
            row["inference_fps_from_source_time"] = (
                (int(frames[-1][0]) - int(frames[0][0]))
                * 1000
                / (int(frames[-1][1]) - int(frames[0][1]))
            )
            row["median_analysis_ms"] = statistics.median(int(f[2]) for f in frames)
            row["median_yolo_ms"] = statistics.median(int(f[3]) for f in frames)
            row["median_ui_fps"] = statistics.median(float(f[4]) for f in frames)
        else:
            row["failure"] = "Insufficient real frame samples; no performance inferred"
        reports.append(row)
        (OUTPUT / "report.json").write_text(
            json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            json.dumps(
                {k: v for k, v in row.items() if k != "spatial_log_samples"}, ensure_ascii=False
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
