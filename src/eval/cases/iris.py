"""Eval cases for the Iris dataset (data/iris.csv).

Download: https://raw.githubusercontent.com/datasciencedojo/datasets/master/Iris.csv
"""
from __future__ import annotations

from eval.assertions import answer_not_empty, mentions_any_of, mentions_number_in_range
from eval.harness import EvalCase

CASES: list[EvalCase] = [
    EvalCase(
        id="iris_species_count",
        dataset="iris.csv",
        question="How many unique species are in this dataset?",
        assertions=[mentions_number_in_range(3, 3), answer_not_empty],
        tags=["basic"],
    ),
    EvalCase(
        id="iris_longest_petals",
        dataset="iris.csv",
        question="Which species has the longest average petal length?",
        assertions=[mentions_any_of("virginica"), answer_not_empty],
        tags=["groupby"],
    ),
    EvalCase(
        id="iris_sepal_vs_petal",
        dataset="iris.csv",
        question="Is there a correlation between sepal length and petal length? Show a scatter plot.",
        assertions=[answer_not_empty],
        tags=["correlation", "chart"],
    ),
]
