"""Eval cases for the NYC Taxi sample dataset (data/nyc_taxi_sample.csv).

Use any small NYC TLC trip-record CSV sample (e.g. first 5000 rows of a monthly file).
Download: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
"""
from __future__ import annotations

from eval.assertions import answer_not_empty, mentions_any_of
from eval.harness import EvalCase

CASES: list[EvalCase] = [
    EvalCase(
        id="taxi_avg_fare",
        dataset="nyc_taxi_sample.csv",
        question="What is the average fare amount in this dataset?",
        assertions=[answer_not_empty],
        tags=["stats"],
    ),
    EvalCase(
        id="taxi_payment_breakdown",
        dataset="nyc_taxi_sample.csv",
        question="What proportion of trips were paid by credit card vs cash?",
        assertions=[mentions_any_of("credit", "card", "cash"), answer_not_empty],
        tags=["groupby"],
    ),
    EvalCase(
        id="taxi_busiest_hour",
        dataset="nyc_taxi_sample.csv",
        question="Which hour of the day had the most pickups? Show a bar chart.",
        assertions=[answer_not_empty],
        tags=["time", "chart"],
    ),
]
