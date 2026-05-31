"""Canonical JSON schemas for all agent tools (OpenAI function-calling style).

These are the single source of truth. Provider implementations translate
them to their own format (Groq passes through unchanged; Gemini converts
to FunctionDeclaration). The agent loop passes ALL_TOOLS to provider.chat().
"""
from __future__ import annotations

INSPECT_SCHEMA: dict = {
    "name": "inspect_schema",
    "description": (
        "Inspect the loaded dataset. Returns shape, column dtypes, first 5 rows, "
        "null counts per column, and basic statistics (describe())."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "head_rows": {
                "type": "integer",
                "description": "Number of rows to show (default: 5)",
            },
            "include_stats": {
                "type": "boolean",
                "description": "Include statistics (default: true)",
            },
        },
        "required": [],
    },
}

RUN_PYTHON: dict = {
    "name": "run_python",
    "description": (
        "Execute Python code in a sandboxed subprocess. "
        "The dataset is pre-loaded as `df` (a pandas DataFrame). "
        "To produce a chart, build it with matplotlib then call save_chart(). "
        "Returns stdout, stderr/traceback if it errored, and chart file paths."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute. Must be self-contained.",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default: 30)",
            },
        },
        "required": ["code"],
    },
}

READ_CELL_RANGE: dict = {
    "name": "read_cell_range",
    "description": (
        "Read a specific row/column slice of the data file. "
        "Use when the file has metadata rows before the real header, "
        "or when you need a specific region of an Excel sheet."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "start_row": {
                "type": "integer",
                "description": "0-indexed start row (inclusive)",
            },
            "end_row": {
                "type": ["integer", "null"],
                "description": "0-indexed end row (inclusive)",
            },
            "columns": {
                "type": ["array", "null"],
                "items": {"type": ["string", "integer"]},
                "description": "Column names or positions",
            },
            "header_row": {
                "type": ["integer", "null"],
                "description": "Row index to use as header",
            },
            "sheet_name": {
                "type": ["string", "null"],
                "description": "Excel sheet name",
            },
        },
        "required": ["start_row"],
    },
}

SAVE_CHART: dict = {
    "name": "save_chart",
    "description": (
        "Save the current matplotlib figure to a file in the run output directory."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": "Filename e.g. 'age_distribution.png'",
            },
            "dpi": {
                "type": "integer",
                "description": "Resolution in DPI (default: 150)",
            },
        },
        "required": ["filename"],
    },
}

FINISH: dict = {
    "name": "finish",
    "description": (
        "Call this when the analysis is complete. "
        "Provide the final written answer, referencing any chart filenames produced. "
        "This is the only way to end the analysis loop."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "Complete written answer to the user's question.",
            },
            "chart_files": {
                "type": "array",
                "items": {"type": "string"},
                "default": [],
                "description": "Filenames of charts produced during analysis, in display order.",
            },
        },
        "required": ["answer"],
    },
}

ALL_TOOLS: list[dict] = [INSPECT_SCHEMA, RUN_PYTHON, READ_CELL_RANGE, SAVE_CHART, FINISH]
