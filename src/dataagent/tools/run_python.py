"""run_python tool — sandboxed Python code execution."""
from __future__ import annotations

from typing import Any

from dataagent.sandbox.executor import execute
from dataagent.tools.registry import ToolContext, register


@register("run_python")
def run_python(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    """Execute agent-generated code in the subprocess sandbox."""
    code = str(arguments.get("code", ""))

    if "\\n" in code:
        try:
            code = code.encode("utf-8").decode("unicode_escape")
        except Exception:
            pass

    timeout = int(arguments.get("timeout") or context.sandbox_timeout)

    result = execute(
        code=code,
        data_file=context.file_path,
        charts_dir=context.charts_dir,
        timeout=timeout,
        memory_mb=context.sandbox_memory_mb,
    )

    chart_files = [str(path) for path in result.chart_files]

    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "timed_out": result.timed_out,
        "sandbox_violation": result.sandbox_violation,
        "chart_files": chart_files,
        "chart_count": len(chart_files),
        "ok": (
            not result.timed_out
            and not result.stderr
            and not result.sandbox_violation
        ),
        "debug": {
            "executed_code": code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "sandbox_violation": result.sandbox_violation,
            "timed_out": result.timed_out,
            "charts_dir": str(context.charts_dir),
            "chart_files": chart_files,
            "chart_count": len(chart_files),
        },
    }