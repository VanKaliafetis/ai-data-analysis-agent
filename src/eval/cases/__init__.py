"""All eval cases, collected from per-dataset modules."""
from eval.cases import housing, iris, nyc_taxi, titanic

ALL_CASES = titanic.CASES + iris.CASES + nyc_taxi.CASES + housing.CASES

__all__ = ["ALL_CASES"]
