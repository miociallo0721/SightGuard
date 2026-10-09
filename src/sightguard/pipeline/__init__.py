"""Video analysis and independent JSONL replay."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from sightguard.data import replay_records

if TYPE_CHECKING:
    from sightguard.pipeline.assess import assess_tracks
    from sightguard.pipeline.video import analyze_video

__all__ = ["analyze_video", "assess_tracks", "replay_records"]


def __getattr__(name: str) -> Callable[..., Any]:
    # Keep the original exports while allowing assessment without CV/ML imports.
    if name == "analyze_video":
        from sightguard.pipeline.video import analyze_video

        return analyze_video
    if name == "assess_tracks":
        from sightguard.pipeline.assess import assess_tracks

        return assess_tracks
    raise AttributeError(name)
