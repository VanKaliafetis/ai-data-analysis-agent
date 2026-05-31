"""Reusable property-based assertion helpers for eval cases.

Each helper returns a callable(answer: str) -> bool.
The callable's __name__ is set so failure messages are readable.
"""
from __future__ import annotations

import re


def mentions_number_in_range(lo: float, hi: float) -> callable:
    """Pass if the answer contains at least one number between lo and hi (inclusive)."""
    def check(answer: str) -> bool:
        for m in re.finditer(r"[-+]?\d+(?:[.,]\d+)?", answer):
            try:
                val = float(m.group().replace(",", ""))
                if lo <= val <= hi:
                    return True
            except ValueError:
                continue
        return False

    check.__name__ = f"mentions_number_in_range({lo}, {hi})"
    return check


def mentions_any_of(*phrases: str) -> callable:
    """Pass if the answer contains at least one of the given phrases (case-insensitive)."""
    lower = [p.lower() for p in phrases]

    def check(answer: str) -> bool:
        a = answer.lower()
        return any(p in a for p in lower)

    check.__name__ = f"mentions_any_of({phrases})"
    return check


def mentions_all_of(*phrases: str) -> callable:
    """Pass if the answer contains every one of the given phrases (case-insensitive)."""
    lower = [p.lower() for p in phrases]

    def check(answer: str) -> bool:
        a = answer.lower()
        return all(p in a for p in lower)

    check.__name__ = f"mentions_all_of({phrases})"
    return check


def answer_not_empty(answer: str) -> bool:
    """Pass if the answer is non-empty after stripping whitespace."""
    return bool(answer.strip())


answer_not_empty.__name__ = "answer_not_empty"
