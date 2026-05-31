"""Entry point: `python -m dataagent <file.csv>` launches an interactive REPL."""
from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

import pandas as pd
from rich.console import Console

from dataagent import __version__
from dataagent.agent import AgentSession, run_agent
from dataagent.config import load_config
from dataagent.providers.groq import GroqProvider
from dataagent.trace import Tracer


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="dataagent",
        description="AI agent for natural-language data analysis (Groq / Gemini, free tier)",
    )
    parser.add_argument(
        "file",
        nargs="?",
        help="CSV or Excel file to analyse",
    )
    parser.add_argument(
        "--provider",
        default=None,
        metavar="PROVIDER",
        help="LLM provider override: groq | gemini",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"dataagent {__version__}",
    )
    args = parser.parse_args()

    if args.file is None:
        parser.print_help()
        sys.exit(0)

    console = Console(legacy_windows=False)

    # Load config
    config = load_config()
    if args.provider:
        config.provider = args.provider

    # Load file
    file_path = Path(args.file)
    if not file_path.exists():
        console.print(f"[red]Error: File not found: {file_path}[/red]")
        sys.exit(1)

    try:
        if str(file_path).endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
    except Exception as e:
        console.print(f"[red]Error loading file: {e}[/red]")
        sys.exit(1)

    console.print(f"[green]✓ Loaded {file_path.name}[/green] — {df.shape[0]} rows, {df.shape[1]} cols")

    # Initialize provider
    if config.provider == "groq":
        if not config.groq_api_key:
            console.print("[red]Error: GROQ_API_KEY not set in .env[/red]")
            sys.exit(1)
        provider = GroqProvider(config.groq_api_key, config.groq_model)
    else:
        console.print(f"[red]Error: provider '{config.provider}' not yet implemented[/red]")
        sys.exit(1)

    # Setup session
    run_id = str(uuid.uuid4())[:8]
    charts_dir = config.runs_dir / run_id / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    trace_file = config.runs_dir / run_id / "trace.jsonl"
    tracer = Tracer(trace_file)
    tracer.session_start(run_id=run_id, file=str(file_path), provider=provider.provider_name)

    session = AgentSession(
        file_path=file_path,
        provider=provider,
        config=config,
        tracer=tracer,
        df=df,
        run_id=run_id,
        charts_dir=charts_dir,
    )

    console.print(f"[cyan]Run ID: {run_id}[/cyan]")
    console.print("[cyan]Type your question or 'quit' to exit.[/cyan]\n")

    # REPL loop
    while True:
        try:
            question = console.input("[bold cyan]>[/bold cyan] ").strip()
            if not question:
                continue
            if question.lower() in ("quit", "exit", "q"):
                break

            answer = run_agent(session, question)
            console.print(f"\n[bold green]Answer:[/bold green]\n{answer}\n")

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

    tracer.session_end(run_id=run_id)
    console.print(f"[cyan]Trace saved to: {trace_file}[/cyan]")


if __name__ == "__main__":
    main()

