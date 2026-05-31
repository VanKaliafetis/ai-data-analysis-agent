"""Tool registry — maps tool names to callables and holds per-call runtime context.

Usage:
    @register("my_tool")
    def my_tool(arguments: dict, context: ToolContext) -> dict:
        ...

    result = dispatch("my_tool", {"arg": "val"}, context)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ToolFn = Callable[[dict[str, Any], "ToolContext"], dict[str, Any]]

_REGISTRY: dict[str, ToolFn] = {}


@dataclass
class ToolContext:
    """Runtime context injected into every tool call by the agent loop."""

    file_path: Path
    charts_dir: Path
    run_id: str
    sandbox_timeout: int
    sandbox_memory_mb: int


def register(name: str) -> Callable[[ToolFn], ToolFn]:
    """Decorator that registers a function under the given tool name."""
    def decorator(fn: ToolFn) -> ToolFn:
        _REGISTRY[name] = fn
        return fn
    return decorator


def dispatch(name: str, arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    """Look up and call a registered tool. Returns an error dict for unknown tools."""
    if name not in _REGISTRY:
        return {"error": f"Unknown tool: '{name}'. Available: {sorted(_REGISTRY)}"}
    return _REGISTRY[name](arguments, context)


def all_schemas_registered() -> list[str]:
    return sorted(_REGISTRY.keys())
