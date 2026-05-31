"""Core agent loop — multi-turn tool-use loop with error recovery.

The loop continues until:
  1. Agent calls finish() — normal end
  2. Agent produces a non-tool response — implicit finish
  3. MAX_ITERATIONS exceeded — graceful degradation
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import pandas as pd

import dataagent.tools  # noqa: F401 - imports register all tools
from dataagent.providers.base import Message
from dataagent.tools.registry import ToolContext, dispatch
from dataagent.tools.schemas import ALL_TOOLS

if TYPE_CHECKING:
    from dataagent.config import Config
    from dataagent.providers.base import LLMProvider
    from dataagent.trace import Tracer


@dataclass
class AgentSession:
    """Mutable state for one REPL session."""

    file_path: Path
    provider: "LLMProvider"
    config: "Config"
    tracer: "Tracer"
    df: pd.DataFrame
    messages: list[Message] = field(default_factory=list)
    run_id: str = ""
    charts_dir: Path = field(default_factory=Path)
    event_callback: Callable[[str], None] | None = None


def run_agent(session: AgentSession, question: str) -> str:
    """Run one question through the agent loop. Returns only the final answer."""
    iteration = 0
    last_assistant_content = ""
    chart_requested = _question_requests_chart(question)
    latest_chart_files: list[str] = []
    latest_tool_error = ""
    chart_tool_correction_sent = False

    try:
        if not session.messages:
            session.messages.append(
                Message(
                    role="system",
                    content=(
                        "You are a careful data-analysis agent.\n\n"
                        "Rules:\n"
                        "1. Use the provided tools only through formal tool calls.\n"
                        "2. Never write raw tool-call syntax in your normal text.\n"
                        "3. First inspect the schema unless the current conversation already "
                        "contains enough schema context.\n"
                        "4. Use run_python for calculations that require filtering, grouping, "
                        "aggregation, or plotting.\n"
                        "5. If the user asks for a chart, plot, graph, heatmap, histogram, "
                        "boxplot, scatter plot, line chart, bar chart, or visualization, "
                        "you must call run_python and generate matplotlib code.\n"
                        "6. All chart code must create a matplotlib figure and save it by "
                        "calling save_chart('clear_descriptive_name.png').\n"
                        "7. Do not claim a chart was created unless run_python returns a "
                        "non-empty chart_files list.\n"
                        "8. For follow-up questions, use the previous conversation context.\n"
                        "9. End non-chart answers by calling finish with a clear "
                        "natural-language answer.\n"
                    ),
                )
            )

        if chart_requested:
            session.messages.append(
                Message(
                    role="user",
                    content=_chart_instruction(question, session.df),
                )
            )
        else:
            session.messages.append(Message(role="user", content=question))

        _emit_event(session, "Started agent analysis.")

        for iteration in range(session.config.max_iterations):
            _emit_event(session, f"Calling model for iteration {iteration + 1}.")

            response = session.provider.chat(session.messages, ALL_TOOLS)

            session.tracer.model_call(
                provider=response.provider,
                model=response.model,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                latency_ms=response.latency_ms,
            )

            _emit_event(session, "Model returned a response.")

            response.message = _clean_assistant_message(response.message)
            session.messages.append(response.message)
            last_assistant_content = response.message.content or ""

            if not response.message.tool_calls:
                if chart_requested and not chart_tool_correction_sent:
                    chart_tool_correction_sent = True
                    session.messages.append(
                        Message(
                            role="user",
                            content=(
                                "You did not call run_python. This is a chart request. "
                                "You must now call run_python with matplotlib code. "
                                "The code must save the PNG using save_chart()."
                            ),
                        )
                    )
                    continue

                _emit_event(session, "Generated final answer.")
                answer = _clean_user_visible_answer(last_assistant_content)
                return _enforce_chart_truth(
                    answer,
                    chart_requested,
                    latest_chart_files,
                    latest_tool_error,
                )

            for tool_call in response.message.tool_calls:
                context = _tool_context(session)

                _emit_event(session, f"Running tool: `{tool_call.name}`.")

                tool_result = dispatch(tool_call.name, tool_call.arguments, context)

                session.tracer.tool_call(
                    name=tool_call.name,
                    result_keys=list(tool_result.keys()),
                )

                _emit_event(session, f"Completed tool: `{tool_call.name}`.")

                if tool_call.name == "run_python":
                    chart_files = _valid_chart_files(tool_result.get("chart_files", []))
                    latest_chart_files = chart_files

                    tool_result["chart_files"] = chart_files
                    tool_result["chart_count"] = len(chart_files)

                    stderr = str(tool_result.get("stderr", "")).strip()
                    stdout = str(tool_result.get("stdout", "")).strip()
                    sandbox_violation = tool_result.get("sandbox_violation")
                    executed_code = str(
                        tool_result.get("debug", {}).get("executed_code", "")
                    ).strip()

                    if stderr:
                        latest_tool_error = stderr
                        _emit_event(session, f"run_python stderr: {stderr[:300]}")

                    if sandbox_violation:
                        latest_tool_error = str(sandbox_violation)
                        _emit_event(
                            session,
                            f"Sandbox violation: {sandbox_violation}",
                        )

                    if not latest_tool_error and stdout:
                        latest_tool_error = f"No chart file was returned. stdout:\n{stdout}"

                    if not latest_tool_error and executed_code:
                        latest_tool_error = (
                            "No chart file was returned.\n\n"
                            f"Executed code:\n{executed_code}"
                        )

                    if chart_files:
                        latest_tool_error = ""

                        for chart_file in chart_files:
                            _emit_event(
                                session,
                                f"Chart file created: {Path(chart_file).name}",
                            )

                        if chart_requested:
                            if len(chart_files) == 1:
                                return "Chart generated successfully."

                            return f"{len(chart_files)} charts generated successfully."

                    else:
                        _emit_event(
                            session,
                            "run_python completed with no chart files.",
                        )

                    if chart_requested:
                        if sandbox_violation:
                            return (
                                "I tried to generate the chart, but the sandbox "
                                f"blocked the code: {sandbox_violation}"
                            )

                        if stderr:
                            return (
                                "I tried to generate the chart, but Python returned "
                                f"this error: {stderr}"
                            )

                        return _enforce_chart_truth(
                            "",
                            chart_requested,
                            latest_chart_files,
                            latest_tool_error,
                        )

                session.messages.append(
                    Message(
                        role="tool",
                        content=json.dumps(tool_result),
                        tool_call_id=tool_call.id,
                        name=tool_call.name,
                    )
                )

                if tool_call.name == "finish":
                    answer = str(tool_result.get("answer", ""))
                    _emit_event(session, "Generated final answer.")
                    answer = _clean_user_visible_answer(answer)
                    return _enforce_chart_truth(
                        answer,
                        chart_requested,
                        latest_chart_files,
                        latest_tool_error,
                    )

        _emit_event(session, "Maximum iterations reached.")

        answer = (
            f"Max iterations ({session.config.max_iterations}) reached. "
            f"Last response: {_clean_user_visible_answer(last_assistant_content)}"
        )

        return _enforce_chart_truth(
            answer,
            chart_requested,
            latest_chart_files,
            latest_tool_error,
        )

    finally:
        session.tracer.log(
            "iteration_complete",
            question=question,
            iterations=iteration + 1,
        )

        _emit_event(session, f"Run completed after {iteration + 1} iteration(s).")


def _chart_instruction(question: str, df: pd.DataFrame) -> str:
    columns = ", ".join(str(column) for column in df.columns)

    return (
        f"{question}\n\n"
        "This is a chart-generation request.\n"
        "You must respond by calling the run_python tool only.\n"
        "Do not answer in normal text.\n"
        "Do not call finish before run_python.\n\n"
        "Write Python matplotlib code that uses the provided dataframe named df.\n"
        "The sandbox already provides these names: df, pd, plt, save_chart, "
        "DATA_FILE, CHARTS_DIR.\n\n"
        "Requirements for the code:\n"
        "- Use only columns that exist in df.\n"
        "- Create one or more matplotlib figures.\n"
        "- Save every chart by calling save_chart('descriptive_filename.png').\n"
        "- Do not use plt.show().\n"
        "- Do not write files manually outside save_chart().\n"
        "- If a requested column name is approximate, choose the closest matching "
        "column from the available columns.\n"
        "- The code argument must contain real multiline Python code, not escaped "
        "\\n text.\n\n"
        f"Available dataframe columns: {columns}"
    )


def _question_requests_chart(question: str) -> bool:
    lowered = question.lower()

    chart_words = [
        "chart",
        "plot",
        "graph",
        "histogram",
        "visualisation",
        "visualization",
        "heatmap",
        "scatter",
        "scatter plot",
        "bar chart",
        "line chart",
        "boxplot",
        "box plot",
        "distribution",
        "visualise",
        "visualize",
    ]

    return any(word in lowered for word in chart_words)


def _valid_chart_files(raw_chart_files: object) -> list[str]:
    if not raw_chart_files:
        return []

    valid_files: list[str] = []

    for raw_path in raw_chart_files:
        path = Path(str(raw_path))

        if (
            path.exists()
            and path.is_file()
            and path.suffix.lower() == ".png"
            and path.stat().st_size > 0
        ):
            valid_files.append(str(path))

    return valid_files


def _enforce_chart_truth(
    answer: str,
    chart_requested: bool,
    latest_chart_files: list[str],
    latest_tool_error: str = "",
) -> str:
    if not chart_requested:
        return answer

    if _valid_chart_files(latest_chart_files):
        return answer

    if latest_tool_error:
        return (
            "I could not create a valid PNG chart file.\n\n"
            f"Python/tool debug:\n{latest_tool_error}"
        )

    return (
        "I could not create a valid PNG chart file, so there is no chart to display yet."
    )


def _tool_context(session: AgentSession) -> ToolContext:
    return ToolContext(
        file_path=session.file_path,
        charts_dir=session.charts_dir,
        run_id=session.run_id,
        sandbox_timeout=session.config.sandbox_timeout,
        sandbox_memory_mb=session.config.sandbox_memory_mb,
    )


def _emit_event(
    session: AgentSession,
    message: str,
) -> None:
    if session.event_callback is None:
        return

    try:
        session.event_callback(message)
    except Exception:
        pass


def _clean_assistant_message(message: Message) -> Message:
    """Hide raw tool syntax from user-visible assistant content."""
    if message.tool_calls:
        message.content = _remove_raw_tool_syntax(message.content or "")

    return message


def _clean_user_visible_answer(text: str) -> str:
    """Remove accidental raw tool-call syntax from returned answers."""
    cleaned = _remove_raw_tool_syntax(text)

    if cleaned.strip():
        return cleaned.strip()

    if _looks_like_raw_tool_call(text):
        return (
            "I ran the analysis, but the model returned a raw tool call instead "
            "of a final answer. Please ask the question again or rephrase it."
        )

    return text.strip()


def _remove_raw_tool_syntax(text: str) -> str:
    """Remove malformed tool-call snippets from displayed text."""
    if not text:
        return ""

    patterns = [
        r"<\|python_tag\|>.*",
        r"<function=[a-zA-Z_][a-zA-Z0-9_]*>.*?</function>",
    ]

    cleaned = text

    for pattern in patterns:
        cleaned = re.sub(
            pattern,
            "",
            cleaned,
            flags=re.DOTALL,
        )

    return cleaned.strip()


def _looks_like_raw_tool_call(text: str) -> bool:
    return "<|python_tag|>" in text or "<function=" in text