"""Eval cases for the Ames Housing dataset (data/housing.csv).

High null count, mixed types, 81 columns — tests the agent's ability to handle messy data.
Download: https://raw.githubusercontent.com/datasciencedojo/datasets/master/AmesHousing.csv
"""
from __future__ import annotations

from eval.assertions import answer_not_empty, mentions_any_of
from eval.harness import EvalCase

CASES: list[EvalCase] = [
    EvalCase(
        id="housing_null_cols",
        dataset="housing.csv",
        question="Which columns have the most missing values, and how many are missing in each?",
        assertions=[answer_not_empty],
        tags=["nulls", "schema"],
    ),
    EvalCase(
        id="housing_avg_price",
        dataset="housing.csv",
        question="What is the average sale price?",
        assertions=[answer_not_empty],
        tags=["stats"],
    ),
    EvalCase(
        id="housing_price_vs_area",
        dataset="housing.csv",
        question="Is there a correlation between lot area and sale price? Show a scatter plot.",
        assertions=[answer_not_empty],
        tags=["correlation", "chart"],
    ),
    EvalCase(
        id="housing_top_neighborhoods",
        dataset="housing.csv",
        question="Which 3 neighborhoods have the highest median sale price?",
        assertions=[answer_not_empty],
        tags=["groupby", "topN"],
    ),
]
