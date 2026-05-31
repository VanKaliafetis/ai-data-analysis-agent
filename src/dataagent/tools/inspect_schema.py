"""inspect_schema tool — dataset introspection."""
from __future__ import annotations

import json
from typing import Any

import pandas as pd

from dataagent.tools.registry import ToolContext, register


@register("inspect_schema")
def inspect_schema(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    """Return shape, dtypes, head, null counts, and describe() for the dataset."""
    arguments = arguments or {}
    head_rows: int = arguments.get("head_rows", 5)
    include_stats: bool = arguments.get("include_stats", True)

    df = pd.read_csv(context.file_path) if str(context.file_path).endswith(".csv") else pd.read_excel(context.file_path)

    result = {
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "null_counts": df.isnull().sum().to_dict(),
        "head": df.head(head_rows).to_dict(orient="records"),
    }

    if include_stats:
        result["describe"] = df.describe().to_dict()

    return result
