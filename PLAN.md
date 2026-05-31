# DataAgent — Architecture Plan

## Problem Statement

A CLI REPL that accepts a CSV or Excel file and answers natural-language questions about it by
writing and executing Python (pandas/matplotlib) in a sandboxed subprocess, iterating on errors,
and producing a written answer with embedded chart references.

---

## File Layout

```
dataagent/
├── pyproject.toml
├── .env.example
├── .gitignore
├── README.md
├── PLAN.md
├── Dockerfile
├── docker-compose.yml
│
├── data/                          # bundled sample datasets
│   ├── titanic.csv
│   ├── iris.csv
│   ├── nyc_taxi_sample.csv
│   └── housing.csv                # messy dataset (high null %, mixed types)
│
├── dataagent/
│   ├── __init__.py
│   ├── __main__.py                # REPL entry point  (`python -m dataagent`)
│   ├── config.py                  # settings loaded from .env
│   ├── agent.py                   # agent loop
│   │
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py                # LLMProvider ABC + shared dataclasses
│   │   ├── groq.py                # Groq (OpenAI-compat) implementation
│   │   └── gemini.py              # Gemini native SDK implementation
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── registry.py            # tool registration + dispatch
│   │   ├── schemas.py             # canonical JSON schemas for all tools
│   │   ├── inspect_schema.py
│   │   ├── run_python.py
│   │   ├── read_cell_range.py
│   │   ├── save_chart.py
│   │   └── finish.py
│   │
│   ├── sandbox/
│   │   ├── __init__.py
│   │   └── executor.py            # subprocess harness + safety checks
│   │
│   └── trace.py                   # structured JSONL logging
│
├── eval/
│   ├── __init__.py
│   ├── harness.py                 # runs cases against multiple providers
│   ├── assertions.py              # reusable property-based checkers
│   └── cases/
│       ├── titanic.py
│       ├── iris.py
│       ├── nyc_taxi.py
│       └── housing.py
│
└── runs/                          # gitignored — one subdir per run
    └── <run-id>/
        ├── trace.jsonl
        └── charts/
```

---

## Provider Interface Contract

```python
# dataagent/providers/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

@dataclass
class Message:
    role: str                          # "system" | "user" | "assistant" | "tool"
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None    # for role="tool" replies
    name: str | None = None            # tool name, for role="tool" replies

@dataclass
class LLMResponse:
    message: Message
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model: str
    provider: str

class LLMProvider(ABC):
    """Implement this to add a new LLM backend. Swap via config.py."""

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        tools: list[dict],             # provider-translated tool schemas
    ) -> LLMResponse: ...

    @property
    @abstractmethod
    def model_id(self) -> str: ...

    @property
    @abstractmethod
    def provider_name(self) -> str: ...
```

### Schema Translation

Tools are defined once in canonical (OpenAI-style) JSON schema in `tools/schemas.py`.

- **Groq** — passes schemas through unchanged (OpenAI-compatible API).
- **Gemini** — `GeminiProvider.chat()` converts schemas to `google.genai` `FunctionDeclaration`
  format before the call and converts `FunctionCall` responses back to `ToolCall` dataclasses.
  No agent code ever sees the provider difference.

---

## Tool Catalogue & Schemas

### `inspect_schema`
Inspect the loaded dataset: shape, dtypes, head, null counts, describe().
```json
{
  "name": "inspect_schema",
  "description": "Inspect the loaded dataset. Returns shape, column dtypes, first N rows, null counts per column, and basic statistics (describe()). Always call this before writing analysis code.",
  "parameters": {
    "type": "object",
    "properties": {
      "head_rows":      {"type": "integer", "default": 5,    "description": "Rows to show in head()"},
      "include_stats":  {"type": "boolean", "default": true, "description": "Include df.describe() output"}
    },
    "required": []
  }
}
```

### `run_python`
Execute pandas/matplotlib code in the sandbox. `df` is pre-loaded.
```json
{
  "name": "run_python",
  "description": "Execute Python code in a sandboxed subprocess. The dataset is pre-loaded as `df` (pandas DataFrame). Use matplotlib to produce charts (plt.savefig is intercepted — use save_chart instead). Returns stdout, stderr/traceback, and a list of chart file paths.",
  "parameters": {
    "type": "object",
    "properties": {
      "code":    {"type": "string",  "description": "Python code to execute. Must be self-contained."},
      "timeout": {"type": "integer", "default": 30, "description": "Hard timeout in seconds"}
    },
    "required": ["code"]
  }
}
```

### `read_cell_range`
Read a sub-range of the file. Handles CSVs with metadata headers and Excel files.
```json
{
  "name": "read_cell_range",
  "description": "Read a specific row/column slice of the data file. Use when the file has metadata rows before the real header, or when you need a specific region of an Excel sheet.",
  "parameters": {
    "type": "object",
    "properties": {
      "start_row":  {"type": "integer", "description": "0-indexed start row (inclusive)"},
      "end_row":    {"type": ["integer","null"], "default": null, "description": "0-indexed end row (inclusive). Null = read to end."},
      "columns":    {"type": ["array","null"],   "default": null, "description": "Column names or 0-indexed positions to include. Null = all."},
      "header_row": {"type": ["integer","null"], "default": 0,    "description": "Row index to use as column header. Null = no header."},
      "sheet_name": {"type": ["string","null"],  "default": null, "description": "Excel sheet name (ignored for CSV)."}
    },
    "required": ["start_row"]
  }
}
```

### `save_chart`
Save the current matplotlib figure to the run output directory.
```json
{
  "name": "save_chart",
  "description": "Save the current matplotlib figure to a file in the run's output directory. Call this after building a chart in run_python.",
  "parameters": {
    "type": "object",
    "properties": {
      "filename": {"type": "string",  "description": "Filename, e.g. 'age_distribution.png'. Extension determines format."},
      "dpi":      {"type": "integer", "default": 150, "description": "Resolution in dots per inch."}
    },
    "required": ["filename"]
  }
}
```

### `finish`
Signal analysis is complete. This is the agent's only stop condition.
```json
{
  "name": "finish",
  "description": "Call this when the analysis is complete. Provide the final written answer. Reference any chart files by the filename passed to save_chart.",
  "parameters": {
    "type": "object",
    "properties": {
      "answer":      {"type": "string", "description": "Complete written answer to the user's question."},
      "chart_files": {"type": "array",  "items": {"type": "string"}, "default": [], "description": "Filenames of charts produced, in display order."}
    },
    "required": ["answer"]
  }
}
```

---

## Agent Loop

```
REPL:
  load file → build df
  print welcome

  loop forever:
    question = input("> ")
    run_agent(question, df, session_messages)

AGENT (max_iterations=10):
  messages = session_messages + [user(question)]

  for i in range(max_iterations):
    response = provider.chat(messages, tools)
    log(model_call_event)
    messages.append(response.message)

    if not response.message.tool_calls:
      # Model answered directly without calling finish — treat as implicit finish
      display(response.message.content)
      break

    for tool_call in response.message.tool_calls:
      result = dispatch_tool(tool_call, context)
      log(tool_call_event)
      messages.append(tool_result_message(tool_call, result))

      if tool_call.name == "finish":
        display_answer(result["answer"])
        open_charts(result.get("chart_files", []))
        append session_messages with assistant + tool exchange
        return

  else:
    # Hit iteration cap
    display("Reached maximum iterations. Partial answer:")
    display(last_assistant_content or "No answer produced.")
```

**Stop conditions (priority order):**
1. Agent calls `finish` — normal path
2. Agent produces a content-only response with no tool calls — treated as finish
3. `max_iterations` exceeded — graceful degradation with partial output
4. Provider raises non-retriable error — surface to user with trace path

**Error recovery:**
When `run_python` returns a non-empty `stderr` / traceback, the result is appended as a
`tool` message. The model sees the full traceback and is expected to correct the code.
Up to 3 consecutive failed code attempts trigger a soft warning in the tool result
("This is your 3rd attempt — consider a different approach.").

---

## Sandbox Design & Threat Model

### Implementation (Option A — subprocess + static analysis)

1. **Static pre-check** — scan the code string before execution:
   - Block `os.system`, `os.popen`, `os.execv*`, `subprocess`, `socket`, `urllib`,
     `requests`, `httpx`, `ftplib`, `smtplib`, `importlib.import_module` with dynamic args,
     `ctypes`, `cffi`, `__import__` called with a string variable.
   - Block `open(...)` calls whose path argument is not a string literal under the allowed paths.
   - Raise `SandboxViolation` and return the error to the model rather than executing.

2. **Subprocess isolation**:
   - `subprocess.run(..., timeout=timeout)` — hard wall-clock limit.
   - On POSIX: `resource.setrlimit(RLIMIT_CPU, (cpu_sec, cpu_sec))` and
     `RLIMIT_AS` (address space, e.g. 1 GB) set in the child pre-exec hook.
   - On Windows: `subprocess` `CREATE_NO_WINDOW` flag; no `resource` module — timeout only.

3. **Filesystem access**:
   - Data file passed to subprocess via a temp copy in a controlled staging dir (read-only intent).
   - Chart output directory is the only writable path; injected as `CHART_DIR` env var.
   - No other paths are mounted or accessible by policy (not enforced at kernel level).

4. **No network**:
   - The blocklist catches `socket`, `requests`, `urllib` etc.
   - Not enforced at kernel/iptables level — documented limitation.

### Threat Model

| Threat | Mitigated? | Mechanism |
|---|---|---|
| Infinite loops / CPU exhaustion | Yes | timeout + RLIMIT_CPU (POSIX) |
| Memory exhaustion | Partial | RLIMIT_AS on POSIX; timeout-kill on Windows |
| Shell escape via os.system | Yes | static blocklist |
| Network exfiltration via requests | Yes | static blocklist |
| Reading arbitrary files | Partial | blocklist for dynamic `open()` calls; static literals may slip through |
| Import tricks (`__import__`, `importlib`) | Partial | blocklist covers common patterns; creative escapes possible |
| ctypes / cffi native code | Yes | explicitly blocked |
| Writing outside chart dir | Partial | no kernel enforcement; relies on model compliance |

**This sandbox is suitable for a demo tool running trusted-user-generated code. It is NOT
suitable for running untrusted third-party code in production. For production use, replace
the subprocess executor with gVisor, Pyodide (WASM), or a dedicated sandbox container.**

---

## Structured Logging (JSONL Trace)

File: `./runs/<run-id>/trace.jsonl`

Each line is one JSON object with an `event` field:

```jsonc
// Session lifecycle
{"event":"session_start","run_id":"...","file":"titanic.csv","provider":"groq","model":"llama-3.3-70b-versatile","timestamp":"..."}
{"event":"session_end","run_id":"...","total_input_tokens":0,"total_output_tokens":0,"total_latency_ms":0,"iterations":0,"timestamp":"..."}

// Per model call
{"event":"model_call","run_id":"...","question_idx":0,"iteration":1,"provider":"groq","model":"llama-3.3-70b-versatile","input_tokens":512,"output_tokens":128,"latency_ms":340.2,"cost_usd":0.0,"timestamp":"..."}

// Per tool call
{"event":"tool_call","run_id":"...","iteration":1,"name":"inspect_schema","arguments":{},"result_preview":"shape: (891, 12)...","latency_ms":12.1,"error":null,"timestamp":"..."}
```

Token costs are logged as `0.0` (free tier). Counts are logged for benchmarking.

---

## Eval Harness Design

`eval/harness.py` runs a list of `EvalCase` objects against a list of providers and produces
a comparison table.

```python
@dataclass
class EvalCase:
    id: str
    dataset: str                       # filename in data/
    question: str
    assertions: list[Callable[[str], bool]]  # answer → pass/fail
    max_iterations: int = 10
```

**Assertions are property-based, not exact-match:**
```python
# Example: Titanic mean age
lambda ans: bool(re.search(r'\b(2[89]|3[0-5])\b', ans))  # number between 28–35
```

**Output:** a Markdown + CSV table:

| Case | Groq 70B | Gemini 2.5 Flash | Groq 8B |
|---|---|---|---|
| titanic_mean_age | PASS (1.2s) | PASS (2.1s) | FAIL |
| ... | | | |

---

## Configuration

`.env` keys:
```
GROQ_API_KEY=
GEMINI_API_KEY=
PROVIDER=groq                      # groq | gemini | ollama
GROQ_MODEL=llama-3.3-70b-versatile
GEMINI_MODEL=gemini-2.5-flash
OLLAMA_MODEL=llama3.1:8b
OLLAMA_BASE_URL=http://localhost:11434
MAX_ITERATIONS=10
SANDBOX_TIMEOUT=30
SANDBOX_MEMORY_MB=1024
LOG_LEVEL=INFO
```

---

## Milestone Checklist

- [ ] **Step 2** — PLAN.md approved
- [ ] **Step 3** — Repo scaffold: pyproject.toml, dirs, empty modules, .gitignore, .env.example, README skeleton
- [ ] **Step 4** — Groq provider + agent loop + `inspect_schema` + `finish` → "how many rows?" works end-to-end
- [ ] **Step 5** — `run_python` sandbox → real analysis question (e.g. "plot age distribution") works
- [ ] **Step 6a** — `read_cell_range` tool
- [ ] **Step 6b** — `save_chart` tool
- [ ] **Step 7** — Gemini provider → same question, same behaviour, different provider
- [ ] **Step 8** — Eval harness + first 5 eval cases (Titanic + Iris)
- [ ] **Step 9** — Cross-provider eval: Groq 70B vs Gemini 2.5 Flash vs Groq 8B, real numbers
- [ ] **Step 10** — README: architecture diagram, screenshots, eval table, latency benchmarks, failure analysis
