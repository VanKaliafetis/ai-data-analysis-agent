"""save_chart tool — matplotlib figure persistence."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from dataagent.tools.registry import ToolContext, register


@register("save_chart")
def save_chart(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    """Return chart files already created by run_python.

    Chart generation happens inside the subprocess sandbox. Because matplotlib
    figures live inside that subprocess, this tool cannot directly access the
    current figure from the main process. Instead, it reports files that already
    exist in the run charts directory.
    """
    context.charts_dir.mkdir(parents=True, exist_ok=True)

    chart_files = sorted(
        list(context.charts_dir.glob("*.png"))
        + list(context.charts_dir.glob("*.jpg"))
        + list(context.charts_dir.glob("*.jpeg"))
        + list(context.charts_dir.glob("*.svg"))
        + list(context.charts_dir.glob("*.pdf"))
    )

    return {
        "ok": True,
        "chart_files": [str(path) for path in chart_files],
        "filenames": [path.name for path in chart_files],
        "message": (
            "Chart files returned from the current run directory."
            if chart_files
            else "No chart files found yet. Use run_python to generate and save charts."
        ),
    }