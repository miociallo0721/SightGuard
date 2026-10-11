"""ADB actions grounded in current UI XML; writes screenshots and dumps only to D:."""

from __future__ import annotations

import argparse
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

ADB = r"D:\Android\Sdk\platform-tools\adb.exe"


def adb(*args: str) -> bytes:
    return subprocess.check_output([ADB, *args])


def dump() -> ET.Element:
    adb("shell", "uiautomator", "dump", "/sdcard/sightguard-ui.xml")
    return ET.fromstring(adb("exec-out", "cat", "/sdcard/sightguard-ui.xml"))


def tap(label: str, description: bool = False) -> None:
    root = dump()
    attribute = "content-desc" if description else "text"
    matches = [node for node in root.iter("node") if node.get(attribute) == label]
    if not matches:
        raise ValueError(f"UI label not visible: {label}")
    node = matches[0]
    coordinates = list(map(int, re.findall(r"\d+", node.get("bounds", ""))))
    if (
        len(coordinates) != 4
        or coordinates[2] <= coordinates[0]
        or coordinates[3] <= coordinates[1]
    ):
        raise ValueError(f"UI node not visible: {label}")
    adb(
        "shell",
        "input",
        "tap",
        str((coordinates[0] + coordinates[2]) // 2),
        str((coordinates[1] + coordinates[3]) // 2),
    )


def save_screenshot(path: Path) -> None:
    if path.resolve().drive.upper() != "D:":
        raise ValueError("Screenshots must stay on D:")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(adb("exec-out", "screencap", "-p"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tap")
    parser.add_argument("--description", action="store_true")
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args()
    if args.tap:
        tap(args.tap, args.description)
    elif args.screenshot:
        save_screenshot(args.screenshot)
    else:
        for node in dump().iter("node"):
            if node.get("text") or node.get("content-desc"):
                print(node.get("text"), node.get("content-desc"), node.get("bounds"))
