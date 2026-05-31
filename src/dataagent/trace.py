"""Structured JSONL logging — one record per model call, tool call, or session event.

Every run writes to runs/<run_id>/trace.jsonl.
Token costs are logged as $0.00 (free-tier providers).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class Tracer:
    """Appends newline-delimited JSON records to a trace file."""

    def __init__(self, trace_file: Path) -> None:
        self._file = trace_file
        self._file.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, **fields: Any) -> None:
        record = {"event": event, "timestamp": _utcnow(), **fields}
        with self._file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    def session_start(self, **kw: Any) -> None:
        self.log("session_start", **kw)

    def session_end(self, **kw: Any) -> None:
        self.log("session_end", **kw)

    def model_call(self, **kw: Any) -> None:
        self.log("model_call", cost_usd=0.0, **kw)

    def tool_call(self, **kw: Any) -> None:
        self.log("tool_call", **kw)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()
