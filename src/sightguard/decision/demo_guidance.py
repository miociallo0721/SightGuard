"""Image-region suggestions for classroom demos, not traversability or navigation."""

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from sightguard.data import FrameRecord


@dataclass(frozen=True)
class DemoGuidanceConfig:
    schema_version: str = "1.0"
    center_left: float = 0.35
    center_right: float = 0.65
    min_bottom: float = 0.55
    min_area: float = 0.015
    stop_area: float = 0.30
    side_difference: float = 0.035
    confirmation_ms: float = 350

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError("Unsupported demo guidance schema")
        for name, value in asdict(self).items():
            if name == "schema_version":
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
            if name != "confirmation_ms" and value > 1:
                raise ValueError(f"{name} must be at most one")
        if not 0 < self.center_left < self.center_right < 1:
            raise ValueError("Invalid center region")
        if self.min_area >= self.stop_area:
            raise ValueError("stop_area must exceed min_area")


def load_demo_guidance(path: Path) -> DemoGuidanceConfig:
    """Load versioned thresholds independently from perception dependencies."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Demo guidance must be a YAML mapping")
    return DemoGuidanceConfig(**raw)


@dataclass(frozen=True)
class DemoSuggestion:
    action: str
    label: str
    reason: str
    speech: str
    vibration: tuple[int, ...]
    schema_version: str = "1.0"
    mode: str = "image_region_demo"
    hardware_connected: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class DemoGuidance:
    """One video session; only observed boxes and video timestamps affect suggestions."""

    def __init__(self, config: DemoGuidanceConfig) -> None:
        self.config = config
        self.pending = "unknown"
        self.since = 0.0
        self.previous_time = -1.0
        self.observations = 0

    def step(self, record: FrameRecord) -> DemoSuggestion:
        if record.timestamp_ms < self.previous_time:
            raise ValueError("Video timestamps must not move backwards")
        self.previous_time = record.timestamp_ms
        cfg = self.config
        occupied = [0.0, 0.0, 0.0]
        central = False
        urgent = False
        valid = 0
        for detection in record.detections:
            x1, y1, x2, y2 = detection.bbox_xyxy
            x1, x2 = (max(0.0, min(1.0, x / record.frame_width)) for x in (x1, x2))
            y1, y2 = (max(0.0, min(1.0, y / record.frame_height)) for y in (y1, y2))
            area = (x2 - x1) * (y2 - y1)
            if area <= 0:
                continue
            valid += 1
            if y2 < cfg.min_bottom or area < cfg.min_area:
                continue
            borders = (0.0, cfg.center_left, cfg.center_right, 1.0)
            for index in range(3):
                overlap = max(0.0, min(x2, borders[index + 1]) - max(x1, borders[index]))
                occupied[index] += overlap * (y2 - y1)
            overlaps_center = x1 < cfg.center_right and x2 > cfg.center_left
            central |= overlaps_center
            urgent |= overlaps_center and area >= cfg.stop_area
        if record.status != "ok" or not valid:
            action, reason = "unknown", "当前没有足够的有效目标信息，无法判断通行情况。"
        elif urgent:
            action, reason = "stop", "中央目标占据较大画面区域，演示规则优先给出停止提示。"
        elif not central:
            action, reason = "forward", "已检测目标未触发中央关注区规则；未验证前方可通行。"
        elif occupied[0] + cfg.side_difference < occupied[2]:
            action, reason = "left", "中央关注区有目标，左侧已检测框占用较少；未验证左侧可通行。"
        elif occupied[2] + cfg.side_difference < occupied[0]:
            action, reason = "right", "中央关注区有目标，右侧已检测框占用较少；未验证右侧可通行。"
        else:
            action, reason = "stop", "中央关注区有目标，两侧视觉证据不足以给出明确方向。"
        if action != self.pending:
            self.pending, self.since, self.observations = action, record.timestamp_ms, 1
        else:
            self.observations += 1
        if action not in ("stop", "unknown") and (
            record.timestamp_ms - self.since < cfg.confirmation_ms or self.observations < 2
        ):
            action, reason = "unknown", "正在确认连续画面中的目标分布。"
        labels = {
            "unknown": "暂不判断",
            "stop": "停止",
            "left": "左转",
            "right": "右转",
            "forward": "前进",
        }
        patterns = {
            "unknown": (),
            "stop": (600,),
            "left": (150, 150, 150),
            "right": (150, 150, 150, 150, 150),
            "forward": (150,),
        }
        label = labels[action]
        return DemoSuggestion(action, label, reason, f"演示建议：{label}。", patterns[action])
