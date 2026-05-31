"""finish tool — agent stop condition.

The agent loop detects this tool call by name and exits the iteration loop.
This function just echoes the arguments back so they can be logged and displayed.
"""
from __future__ import annotations

from typing import Any

from dataagent.tools.registry import ToolContext, register


@register("finish")
def finish(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    """Echo finish arguments back to the loop (which handles display and chart opening)."""
    return {
        "answer": arguments.get("answer", ""),
        "chart_files": arguments.get("chart_files", []),
    }
