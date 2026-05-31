"""Eval cases for the Titanic dataset (data/titanic.csv).

Download: https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
"""
from __future__ import annotations

from eval.assertions import answer_not_empty, mentions_any_of, mentions_number_in_range
from eval.harness import EvalCase

CASES: list[EvalCase] = [
    EvalCase(
        id="titanic_row_count",
        dataset="titanic.csv",
        question="How many rows are in this dataset?",
        assertions=[mentions_number_in_range(888, 892), answer_not_empty],
        tags=["basic"],
    ),
    EvalCase(
        id="titanic_mean_age",
        dataset="titanic.csv",
        question="What is the mean age of passengers?",
        assertions=[mentions_number_in_range(28.0, 32.0), answer_not_empty],
        tags=["stats"],
    ),
    EvalCase(
        id="titanic_survival_rate",
        dataset="titanic.csv",
        question="What percentage of passengers survived?",
        assertions=[mentions_number_in_range(35.0, 42.0), answer_not_empty],
        tags=["stats"],
    ),
    EvalCase(
        id="titanic_survival_by_class",
        dataset="titanic.csv",
        question="Which passenger class had the highest survival rate?",
        assertions=[mentions_any_of("first", "1st", "class 1", "pclass 1"), answer_not_empty],
        tags=["groupby"],
    ),
]
