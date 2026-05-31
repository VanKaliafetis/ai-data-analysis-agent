"""Streamlit UI for DataAgent.

Run with:
    uv run streamlit run src/dataagent/ui.py

Or:
    python -m streamlit run src/dataagent/ui.py
"""
from __future__ import annotations

import textwrap
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from dataagent.agent import AgentSession, run_agent
from dataagent.config import load_config
from dataagent.providers.gemini import GeminiProvider
from dataagent.providers.groq import GroqProvider
from dataagent.trace import Tracer

load_dotenv()

SUPPORTED_SUFFIXES = {".csv", ".xlsx", ".xls"}


def load_dataframe(file_path: Path) -> pd.DataFrame:
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(file_path)

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)

    raise ValueError(f"Unsupported file type: {suffix}")


def save_uploaded_file(uploaded_file) -> Path:
    uploads_dir = Path("outputs/uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)

    file_path = uploads_dir / uploaded_file.name

    with file_path.open("wb") as file:
        file.write(uploaded_file.getbuffer())

    return file_path


def get_available_local_files() -> list[Path]:
    data_dir = Path("data")

    if not data_dir.exists():
        return []

    return sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def add_agent_event(message: str) -> None:
    if "agent_events" not in st.session_state:
        st.session_state.agent_events = []

    st.session_state.agent_events.append(
        {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "message": message,
        }
    )


def show_agent_activity_sidebar() -> None:
    st.divider()
    st.subheader("Agent Activity")

    events = st.session_state.get("agent_events", [])

    if not events:
        st.caption("No agent activity yet.")
        return

    st.caption(f"{len(events)} events recorded this session.")

    for event in reversed(events[-20:]):
        timestamp = event.get("timestamp", "")
        message = event.get("message", "")
        st.markdown(f"- `{timestamp}` {message}")


def build_provider(provider_name: str):
    config = load_config()

    if provider_name == "groq":
        return GroqProvider(
            api_key=config.groq_api_key,
            model=config.groq_model,
        )

    if provider_name == "gemini":
        return GeminiProvider(
            api_key=config.gemini_api_key,
            model=config.gemini_model,
        )

    raise ValueError(f"Unsupported provider: {provider_name}")


def create_agent_session(
    file_path: Path,
    provider_name: str,
) -> AgentSession:
    config = load_config()
    provider = build_provider(provider_name)
    dataframe = load_dataframe(file_path)

    run_id = uuid.uuid4().hex[:8]
    run_dir = Path("outputs/runs") / run_id
    charts_dir = run_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    trace_file = run_dir / "trace.jsonl"
    tracer = Tracer(trace_file)

    tracer.session_start(
        run_id=run_id,
        file=str(file_path),
        provider=provider.provider_name,
    )

    add_agent_event(f"Created agent session `{run_id}`.")

    return AgentSession(
        file_path=file_path,
        provider=provider,
        config=config,
        tracer=tracer,
        df=dataframe,
        run_id=run_id,
        charts_dir=charts_dir,
        event_callback=add_agent_event,
    )


def reset_session() -> None:
    for key in [
        "agent_session",
        "active_file_path",
        "active_provider",
        "messages",
        "auto_chart_paths",
        "auto_chart_run_id",
        "dataset_summary",
        "agent_events",
    ]:
        if key in st.session_state:
            del st.session_state[key]


def make_dataset_summary(df: pd.DataFrame) -> dict[str, Any]:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    duplicate_rows = int(df.duplicated().sum())

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "missing_columns": missing.to_dict(),
        "duplicate_rows": duplicate_rows,
    }


def show_dataframe_summary(df: pd.DataFrame) -> dict[str, Any]:
    summary = make_dataset_summary(df)

    st.subheader("Dataset summary")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Rows", summary["rows"])
    col2.metric("Columns", summary["columns"])
    col3.metric("Numeric columns", len(summary["numeric_columns"]))
    col4.metric("Duplicate rows", summary["duplicate_rows"])

    st.subheader("Dataset preview")
    st.dataframe(
        df,
        use_container_width=True,
        height=520,
    )

    with st.expander("Numeric statistics"):
        numeric = df.select_dtypes(include="number")

        if numeric.empty:
            st.info("No numeric columns found.")
        else:
            st.dataframe(numeric.describe().T, use_container_width=True)

    return summary


def create_auto_charts(df: pd.DataFrame, run_id: str) -> list[Path]:
    charts_dir = Path("outputs/figures") / run_id
    charts_dir.mkdir(parents=True, exist_ok=True)

    chart_paths: list[Path] = []

    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False).head(12)

    if not missing.empty:
        path = charts_dir / "missing_values.png"

        fig, ax = plt.subplots(figsize=(9, 4))
        missing.plot(kind="bar", ax=ax)
        ax.set_title("Missing values by column")
        ax.set_ylabel("Missing count")
        ax.set_xlabel("Column")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

        chart_paths.append(path)

    numeric = df.select_dtypes(include="number")

    useful_numeric_cols = [
        col
        for col in numeric.columns
        if df[col].nunique(dropna=True) > 10 and "id" not in col.lower()
    ]

    if useful_numeric_cols:
        numeric_col = useful_numeric_cols[0]
        path = charts_dir / f"{numeric_col}_distribution.png"

        fig, ax = plt.subplots(figsize=(8, 4))
        numeric[numeric_col].dropna().hist(ax=ax, bins=30)
        ax.set_title(f"Distribution of {numeric_col}")
        ax.set_xlabel(numeric_col)
        ax.set_ylabel("Frequency")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

        chart_paths.append(path)

    categorical = df.select_dtypes(exclude="number")

    if not categorical.empty:
        usable_cols = [
            col
            for col in categorical.columns
            if 1 < df[col].nunique(dropna=True) <= 20
        ]

        if usable_cols:
            cat_col = usable_cols[0]
            path = charts_dir / f"{cat_col}_top_values.png"

            counts = df[cat_col].value_counts(dropna=False).head(10)

            fig, ax = plt.subplots(figsize=(8, 4))
            counts.plot(kind="bar", ax=ax)
            ax.set_title(f"Top values in {cat_col}")
            ax.set_xlabel(cat_col)
            ax.set_ylabel("Count")
            fig.tight_layout()
            fig.savefig(path, dpi=150)
            plt.close(fig)

            chart_paths.append(path)

    if numeric.shape[1] >= 2:
        path = charts_dir / "correlation_heatmap.png"
        corr = numeric.corr(numeric_only=True)

        fig, ax = plt.subplots(figsize=(8, 6))
        image = ax.imshow(corr.values)
        ax.set_title("Numeric correlation heatmap")
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(corr.index)))
        ax.set_yticklabels(corr.index)
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

        chart_paths.append(path)

    return chart_paths


def show_auto_charts(df: pd.DataFrame, run_id: str) -> list[Path]:
    if (
        "auto_chart_paths" not in st.session_state
        or st.session_state.get("auto_chart_run_id") != run_id
    ):
        st.session_state.auto_chart_paths = create_auto_charts(df, run_id)
        st.session_state.auto_chart_run_id = run_id

    chart_paths = st.session_state.auto_chart_paths

    st.subheader("Automatic charts")

    if not chart_paths:
        st.info("No automatic charts were generated for this dataset.")
        return []

    columns = st.columns(2)

    for index, chart_path in enumerate(chart_paths):
        with columns[index % 2]:
            st.image(str(chart_path), caption=chart_path.name, use_container_width=True)

    return chart_paths


def show_generated_agent_charts(charts_dir: Path) -> list[Path]:
    if not charts_dir.exists():
        return []

    chart_files = sorted(
        list(charts_dir.glob("*.png"))
        + list(charts_dir.glob("*.jpg"))
        + list(charts_dir.glob("*.jpeg"))
    )

    if not chart_files:
        return []

    st.subheader("Agent-generated charts")

    for chart_file in chart_files:
        st.image(str(chart_file), caption=chart_file.name, use_container_width=True)

    return chart_files


def build_report_markdown(
    file_path: Path,
    provider_name: str,
    summary: dict[str, Any],
    messages: list[dict[str, str]],
    auto_charts: list[Path],
    agent_charts: list[Path],
    run_id: str,
) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# DataAgent Report",
        "",
        f"Generated: {timestamp}",
        f"Run ID: `{run_id}`",
        f"Provider: `{provider_name}`",
        f"Dataset: `{file_path}`",
        "",
        "## Dataset Summary",
        "",
        f"- Rows: {summary['rows']}",
        f"- Columns: {summary['columns']}",
        f"- Numeric columns: {len(summary['numeric_columns'])}",
        f"- Categorical columns: {len(summary['categorical_columns'])}",
        f"- Duplicate rows: {summary['duplicate_rows']}",
        "",
    ]

    if summary["missing_columns"]:
        lines.append("### Missing Values")
        lines.append("")

        for column, count in summary["missing_columns"].items():
            lines.append(f"- `{column}`: {count}")

        lines.append("")

    lines.extend(["## Conversation", ""])

    for message in messages:
        role = message["role"].title()
        content = message["content"]

        lines.extend([f"### {role}", "", content, ""])

    all_charts = auto_charts + agent_charts

    if all_charts:
        lines.extend(["## Charts", ""])

        for chart in all_charts:
            lines.append(f"- `{chart}`")

        lines.append("")

    return "\n".join(lines)


def save_report(
    report_markdown: str,
    run_id: str,
) -> Path:
    reports_dir = Path("outputs/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    report_path = reports_dir / f"report_{run_id}.md"

    with report_path.open("w", encoding="utf-8") as file:
        file.write(report_markdown)

    return report_path


def show_report_export(
    file_path: Path,
    provider_name: str,
    summary: dict[str, Any],
    auto_charts: list[Path],
    agent_charts: list[Path],
    run_id: str,
) -> None:
    st.subheader("Export report")

    report_markdown = build_report_markdown(
        file_path=file_path,
        provider_name=provider_name,
        summary=summary,
        messages=st.session_state.get("messages", []),
        auto_charts=auto_charts,
        agent_charts=agent_charts,
        run_id=run_id,
    )

    report_path = save_report(report_markdown, run_id)

    st.download_button(
        label="Download Markdown report",
        data=report_markdown,
        file_name=f"dataagent_report_{run_id}.md",
        mime="text/markdown",
    )

    st.caption(f"Saved locally to `{report_path}`")


def submit_question(
    session: AgentSession,
    question: str,
) -> None:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    add_agent_event("Received user question.")

    try:
        answer = run_agent(session, question)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        add_agent_event("Displayed final answer.")

    except Exception as exc:
        add_agent_event(f"Agent error: {exc}")

        error_message = textwrap.dedent(
            f"""
            Error: {exc}

            Try a simpler question, switch provider, or check your API key/rate limits.
            """
        ).strip()

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": error_message,
            }
        )


def main() -> None:
    st.set_page_config(
        page_title="DataAgent",
        page_icon="📊",
        layout="wide",
    )

    if "agent_events" not in st.session_state:
        st.session_state.agent_events = []

    st.title("📊 DataAgent")
    st.caption("Ask questions about CSV / Excel files using the agent loop.")

    config = load_config()

    with st.sidebar:
        st.header("Settings")

        provider_name = st.selectbox(
            "Provider",
            options=["groq", "gemini"],
            index=0,
        )

        if provider_name == "groq" and not config.groq_api_key:
            st.warning("GROQ_API_KEY is missing in `.env`.")

        if provider_name == "gemini" and not config.gemini_api_key:
            st.warning("GEMINI_API_KEY is missing in `.env`.")

        st.divider()
        st.subheader("Choose data")

        uploaded_file = st.file_uploader(
            "Upload CSV or Excel",
            type=["csv", "xlsx", "xls"],
        )

        local_files = get_available_local_files()
        local_file_options = ["None"] + [str(path) for path in local_files]

        selected_local_file = st.selectbox(
            "Or use a local file from data/",
            options=local_file_options,
        )

        if st.button("Reset conversation"):
            reset_session()
            st.rerun()

        show_agent_activity_sidebar()

    file_path: Path | None = None

    if uploaded_file is not None:
        file_path = save_uploaded_file(uploaded_file)

    elif selected_local_file != "None":
        file_path = Path(selected_local_file)

    if file_path is None:
        st.info("Upload a CSV/Excel file or select one from the data folder.")
        return

    if (
        "active_file_path" not in st.session_state
        or st.session_state.active_file_path != str(file_path)
        or st.session_state.get("active_provider") != provider_name
    ):
        reset_session()
        st.session_state.agent_events = []
        st.session_state.active_file_path = str(file_path)
        st.session_state.active_provider = provider_name
        st.session_state.messages = []

    st.success(f"Loaded file: `{file_path}`")

    try:
        dataframe = load_dataframe(file_path)
    except Exception as exc:
        st.error(f"Could not load file: {exc}")
        return

    summary = show_dataframe_summary(dataframe)
    st.session_state.dataset_summary = summary

    if "agent_session" not in st.session_state:
        try:
            st.session_state.agent_session = create_agent_session(
                file_path=file_path,
                provider_name=provider_name,
            )
        except Exception as exc:
            st.error(str(exc))
            return

    session: AgentSession = st.session_state.agent_session

    st.divider()

    auto_charts = show_auto_charts(dataframe, session.run_id)

    st.divider()
    st.subheader("Ask the agent")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    with st.form("question_form", clear_on_submit=True):
        question = st.text_input(
            "Ask a question about the data...",
            placeholder="Example: Show survival rate by passenger class and create a chart",
            label_visibility="collapsed",
        )

        submitted = st.form_submit_button("Ask")

    if submitted and question.strip():
        with st.spinner("Agent is analysing the data..."):
            submit_question(session, question.strip())

        st.rerun()

    agent_charts = show_generated_agent_charts(session.charts_dir)

    st.divider()

    show_report_export(
        file_path=file_path,
        provider_name=provider_name,
        summary=summary,
        auto_charts=auto_charts,
        agent_charts=agent_charts,
        run_id=session.run_id,
    )


if __name__ == "__main__":
    main()