"""Real eval / benchmark harness.

Runs hand-written data-analysis questions against one or more providers and saves:
- JSON results
- CSV results
- Markdown benchmark table

Example:
    python -m eval.harness --providers groq --limit 1
    python -m eval.harness --providers groq --sleep-between-cases 20
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
from dotenv import load_dotenv

from dataagent.agent import AgentSession, run_agent
from dataagent.config import load_config
from dataagent.providers.groq import GroqProvider
from dataagent.trace import Tracer

load_dotenv()


@dataclass
class EvalCase:
    name: str
    dataset: str
    question: str
    assertion: Callable[[str], bool]


def contains_number_between(text: str, low: float, high: float) -> bool:
    numbers = re.findall(r"\d+(?:\.\d+)?", text)

    for number in numbers:
        value = float(number)

        if low <= value <= high:
            return True

    return False


EVAL_CASES: list[EvalCase] = [
    EvalCase(
        name="titanic_row_count",
        dataset="titanic.csv",
        question="How many rows are in this dataset?",
        assertion=lambda answer: contains_number_between(answer, 880, 900),
    ),
    EvalCase(
        name="titanic_average_age",
        dataset="titanic.csv",
        question="What is the average passenger age?",
        assertion=lambda answer: contains_number_between(answer, 28, 32),
    ),
]


def build_provider(provider_name: str):
    config = load_config()

    if provider_name == "groq":
        return GroqProvider(
            api_key=config.groq_api_key,
            model=config.groq_model,
        )

    raise ValueError(f"Unsupported provider: {provider_name}")


def load_dataset(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    raise ValueError(f"Unsupported dataset type: {path.suffix}")


def make_session(
    provider_name: str,
    dataset_path: Path,
    results_dir: Path,
) -> AgentSession:
    config = load_config()
    provider = build_provider(provider_name)
    dataframe = load_dataset(dataset_path)

    run_id = f"eval_{uuid.uuid4().hex[:8]}"
    charts_dir = results_dir / run_id / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    trace_file = results_dir / run_id / "trace.jsonl"
    tracer = Tracer(trace_file)

    tracer.session_start(
        run_id=run_id,
        file=str(dataset_path),
        provider=provider.provider_name,
    )

    return AgentSession(
        file_path=dataset_path,
        provider=provider,
        config=config,
        tracer=tracer,
        df=dataframe,
        run_id=run_id,
        charts_dir=charts_dir,
    )


def run_eval(
    providers: list[str],
    data_dir: Path,
    results_dir: Path,
    limit: int | None = None,
    sleep_between_cases: int = 20,
) -> list[dict]:
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = int(time.time())
    cases = EVAL_CASES[:limit] if limit else EVAL_CASES
    results: list[dict] = []

    for provider_name in providers:
        for case in cases:
            dataset_path = data_dir / case.dataset

            if not dataset_path.exists():
                print(f"[SKIP] Missing dataset: {dataset_path}")
                continue

            print(f"\nRunning {case.name} with {provider_name}...")

            started = time.time()

            try:
                session = make_session(
                    provider_name=provider_name,
                    dataset_path=dataset_path,
                    results_dir=results_dir,
                )

                answer = run_agent(session, case.question)
                latency_seconds = round(time.time() - started, 2)
                passed = case.assertion(answer)

                result = {
                    "provider": provider_name,
                    "case": case.name,
                    "dataset": case.dataset,
                    "question": case.question,
                    "passed": passed,
                    "latency_seconds": latency_seconds,
                    "answer": answer,
                    "error": "",
                }

                print(f"PASS={passed} LATENCY={latency_seconds}s")

            except Exception as exc:
                latency_seconds = round(time.time() - started, 2)

                result = {
                    "provider": provider_name,
                    "case": case.name,
                    "dataset": case.dataset,
                    "question": case.question,
                    "passed": False,
                    "latency_seconds": latency_seconds,
                    "answer": "",
                    "error": str(exc),
                }

                print(f"[ERROR] {case.name}: {exc}")

            results.append(result)
            save_results(results_dir, timestamp, results)

            if sleep_between_cases > 0:
                print(f"Sleeping {sleep_between_cases}s to avoid rate limits...")
                time.sleep(sleep_between_cases)

    return results


def save_results(
    results_dir: Path,
    timestamp: int,
    results: list[dict],
) -> None:
    if not results:
        return

    json_path = results_dir / f"eval_{timestamp}.json"
    csv_path = results_dir / f"eval_{timestamp}.csv"
    md_path = results_dir / f"eval_{timestamp}.md"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    with md_path.open("w", encoding="utf-8") as file:
        file.write("# Eval / Benchmark Results\n\n")
        file.write("| Provider | Case | Passed | Latency |\n")
        file.write("|---|---|---:|---:|\n")

        for result in results:
            file.write(
                f"| {result['provider']} "
                f"| {result['case']} "
                f"| {result['passed']} "
                f"| {result['latency_seconds']}s |\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--providers",
        nargs="+",
        default=["groq"],
    )

    parser.add_argument(
        "--data-dir",
        default="data",
    )

    parser.add_argument(
        "--results-dir",
        default="outputs/reports",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--sleep-between-cases",
        type=int,
        default=20,
    )

    args = parser.parse_args()

    run_eval(
        providers=args.providers,
        data_dir=Path(args.data_dir),
        results_dir=Path(args.results_dir),
        limit=args.limit,
        sleep_between_cases=args.sleep_between_cases,
    )


if __name__ == "__main__":
    main()