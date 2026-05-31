"""read_cell_range tool — CSV/Excel row-column slicer."""
from __future__ import annotations

from typing import Any

import pandas as pd

from dataagent.tools.registry import ToolContext, register


@register("read_cell_range")
def read_cell_range(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    """Read a row/column slice from CSV or Excel files."""
    start_row = int(arguments.get("start_row", 0))
    end_row = arguments.get("end_row")
    start_col = int(arguments.get("start_col", 0))
    end_col = arguments.get("end_col")
    sheet_name = arguments.get("sheet_name", 0)

    if context.file_path.suffix.lower() == ".csv":
        df = pd.read_csv(context.file_path)
    elif context.file_path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(context.file_path, sheet_name=sheet_name)
    else:
        raise ValueError(f"Unsupported file type: {context.file_path.suffix}")

    end_row = int(end_row) if end_row is not None else min(start_row + 20, len(df))
    end_col = int(end_col) if end_col is not None else min(start_col + 10, len(df.columns))

    sliced = df.iloc[start_row:end_row, start_col:end_col]

    return {
        "ok": True,
        "data": sliced.to_dict(orient="records"),
        "columns": list(sliced.columns),
        "shape": list(sliced.shape),
        "start_row": start_row,
        "end_row": end_row,
        "start_col": start_col,
        "end_col": end_col,
    }