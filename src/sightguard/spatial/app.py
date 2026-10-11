"""Standalone local launcher for the spatial-audio research lab."""

import os
from pathlib import Path

# Disable Gradio analytics/version checks and Hugging Face telemetry before import.
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import gradio as gr

from sightguard.ui.spatial_lab import build_spatial_lab


def launch_spatial_demo(root: Path, port: int = 7862, inbrowser: bool = False) -> tuple:
    """Launch a local page; startup telemetry is disabled and audio stays local."""
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("Spatial demo port must be between 1 and 65535")
    app = build_spatial_lab(Path(root))
    return app.launch(
        server_name="127.0.0.1",
        server_port=port,
        share=False,
        inbrowser=inbrowser,
        show_error=False,
        enable_monitoring=False,
        footer_links=[],
        theme=gr.themes.Base(primary_hue="cyan", neutral_hue="slate"),
    )
