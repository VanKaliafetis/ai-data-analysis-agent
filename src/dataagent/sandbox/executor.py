"""Subprocess-based Python executor with safety guardrails.

This is a demo sandbox for a local portfolio project, not a production security boundary.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

BLOCKED_PATTERNS: list[str] = [
    "os.system",
    "os.popen",
    "os.exec",
    "subprocess",
    "socket",
    "urllib",
    "requests",
    "httpx",
    "ftplib",
    "smtplib",
    "ctypes",
    "cffi",
    "__import__",
    "importlib.import_module",
]


@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    timed_out: bool
    return_code: int | None = None
    sandbox_violation: str | None = None
    chart_files: list[Path] = field(default_factory=list)


def execute(
    code: str,
    data_file: Path,
    charts_dir: Path,
    timeout: int = 30,
    memory_mb: int = 1024,
) -> ExecutionResult:
    blocked = _check_blocked_patterns(code)

    if blocked:
        return ExecutionResult(
            stdout="",
            stderr="",
            timed_out=False,
            return_code=None,
            sandbox_violation=f"Blocked unsafe pattern: {blocked}",
        )

    data_file = data_file.resolve()
    charts_dir = charts_dir.resolve()
    charts_dir.mkdir(parents=True, exist_ok=True)

    wrapper = _build_wrapper(code, data_file, charts_dir, memory_mb)

    with tempfile.TemporaryDirectory(prefix="dataagent_sandbox_") as tmp:
        script = Path(tmp) / "runner.py"
        script.write_text(wrapper, encoding="utf-8")

        before = _snapshot_chart_files(charts_dir)

        env = os.environ.copy()
        env.update(
            {
                "PYTHONIOENCODING": "utf-8",
                "MPLBACKEND": "Agg",
                "DATAAGENT_DATA_FILE": str(data_file),
                "DATAAGENT_CHARTS_DIR": str(charts_dir),
            }
        )

        try:
            proc = subprocess.run(
                [sys.executable, str(script)],
                cwd=tmp,
                env=env,
                text=True,
                capture_output=True,
                timeout=timeout,
            )

        except subprocess.TimeoutExpired as exc:
            return ExecutionResult(
                stdout=exc.stdout or "",
                stderr=exc.stderr or f"Execution timed out after {timeout}s",
                timed_out=True,
                return_code=None,
            )

        after = _snapshot_chart_files(charts_dir)

        chart_files = [
            path
            for path, stat in after.items()
            if path not in before
            or before[path]["mtime_ns"] != stat["mtime_ns"]
            or before[path]["size"] != stat["size"]
        ]

        chart_files = [
            path
            for path in chart_files
            if path.exists() and path.stat().st_size > 0
        ]

        return ExecutionResult(
            stdout=proc.stdout,
            stderr=proc.stderr,
            timed_out=False,
            return_code=proc.returncode,
            chart_files=sorted(chart_files, key=lambda p: p.name),
        )


def _snapshot_chart_files(directory: Path) -> dict[Path, dict[str, int]]:
    snapshot: dict[Path, dict[str, int]] = {}

    for pattern in ("*.png", "*.jpg", "*.jpeg", "*.svg", "*.pdf"):
        for path in directory.glob(pattern):
            if not path.is_file():
                continue

            stat = path.stat()
            snapshot[path] = {
                "mtime_ns": stat.st_mtime_ns,
                "size": stat.st_size,
            }

    return snapshot


def _build_wrapper(
    code: str,
    data_file: Path,
    charts_dir: Path,
    memory_mb: int,
) -> str:
    payload = json.dumps(code)

    return textwrap.dedent(
        f"""
from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

try:
    import resource

    requested_mb = {memory_mb}
    if requested_mb > 0:
        limit = requested_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
except Exception:
    pass

import matplotlib
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import pandas as pd

try:
    import seaborn as sns
except Exception:
    sns = None

try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    px = None
    go = None

DATA_FILE = Path({str(data_file)!r})
CHARTS_DIR = Path({str(charts_dir)!r})
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

_chart_counter = 0


def _safe_chart_name(filename: str | None = None) -> str:
    global _chart_counter

    _chart_counter += 1

    if not filename:
        filename = f"chart_{{_chart_counter}}.png"

    safe = Path(str(filename)).name
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", safe).strip("._")

    if not safe:
        safe = f"chart_{{_chart_counter}}.png"

    if not safe.lower().endswith((".png", ".jpg", ".jpeg", ".svg", ".pdf")):
        safe += ".png"

    stem = Path(safe).stem
    suffix = Path(safe).suffix

    candidate = safe
    duplicate_index = 2

    while (CHARTS_DIR / candidate).exists():
        candidate = f"{{stem}}_{{duplicate_index}}{{suffix}}"
        duplicate_index += 1

    return candidate


def save_chart(filename: str = "chart.png", dpi: int = 150) -> str:
    safe = _safe_chart_name(filename)
    path = CHARTS_DIR / safe

    figure_numbers = plt.get_fignums()

    if figure_numbers:
        figure = plt.figure(figure_numbers[-1])
    else:
        figure = plt.gcf()

    try:
        figure.tight_layout()
    except Exception:
        pass

    figure.savefig(path, dpi=dpi, bbox_inches="tight")

    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"Chart was not saved successfully: {{path}}")

    print(f"[chart] {{path}}", flush=True)

    return str(path)


def save_plotly_chart(
    fig,
    filename: str = "plotly_chart.png",
    width: int = 1000,
    height: int = 650,
) -> str:
    safe = _safe_chart_name(filename)
    path = CHARTS_DIR / safe

    fig.write_image(str(path), width=width, height=height)

    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"Plotly chart was not saved successfully: {{path}}")

    print(f"[chart] {{path}}", flush=True)

    return str(path)


def _auto_save_open_figures() -> None:
    figure_numbers = list(plt.get_fignums())

    print(f"[debug] matplotlib figure count: {{len(figure_numbers)}}", flush=True)

    for index, figure_number in enumerate(figure_numbers, start=1):
        figure = plt.figure(figure_number)
        path = CHARTS_DIR / _safe_chart_name(f"auto_chart_{{index}}.png")

        try:
            figure.tight_layout()
        except Exception:
            pass

        figure.savefig(path, dpi=150, bbox_inches="tight")

        if path.exists() and path.stat().st_size > 0:
            print(f"[chart] {{path}}", flush=True)


def _patched_show(*args, **kwargs):
    if plt.get_fignums():
        return save_chart("shown_chart.png")

    return None


plt.show = _patched_show

if DATA_FILE.suffix.lower() == ".csv":
    df = pd.read_csv(DATA_FILE)
else:
    df = pd.read_excel(DATA_FILE)

user_code = json.loads({payload!r})
user_code = textwrap.dedent(user_code).strip()

globals_dict = {{
    "__name__": "__main__",
    "df": df,
    "pd": pd,
    "plt": plt,
    "sns": sns,
    "px": px,
    "go": go,
    "save_chart": save_chart,
    "save_plotly_chart": save_plotly_chart,
    "DATA_FILE": DATA_FILE,
    "CHARTS_DIR": CHARTS_DIR,
}}

try:
    exec(
        compile(user_code, "<agent_code>", "exec"),
        globals_dict,
        globals_dict,
    )

    _auto_save_open_figures()

finally:
    plt.close("all")
"""
    )


def _check_blocked_patterns(code: str) -> str | None:
    lowered = code.lower()

    for pattern in BLOCKED_PATTERNS:
        if pattern.lower() in lowered:
            return pattern

    return None