"""Typo detection (pandas) — a different problem from
categorical_standardization.py, which only merges values that are
*identical* once lowercased/stripped ("Manhattan" vs "MANHATTAN"). A typo
like "Manhattn" doesn't normalize to the same string as "Manhattan" at all,
so it needs fuzzy (edit-distance) matching instead of exact matching.

Flags only — never rewrites a value. Fuzzy matching is inherently less
certain than exact-match case/whitespace standardization, so an automatic
"correction" here risks silently turning one real category into another;
a human should look at what's flagged before anything gets changed.

Distance is computed on the already-lowercased/stripped form of each
distinct value, so a pure case/whitespace variant ("MANHATTAN") is never
also reported as a "typo" of ("Manhattan") — that's categorical_standardization's
job, not this one's.
"""

from __future__ import annotations

import pandas as pd

from ..constants import MAX_TYPO_EDIT_DISTANCE
from .types import AnomalyResult


def levenshtein_distance(a: str, b: str) -> int:
    """Classic O(len(a) * len(b)) dynamic-programming edit distance. No
    external dependency — these are short category strings compared only
    pairwise across a column's distinct values, never per-row."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous_row = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current_row = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            insert_cost = current_row[j - 1] + 1
            delete_cost = previous_row[j] + 1
            substitute_cost = previous_row[j - 1] + (ca != cb)
            current_row[j] = min(insert_cost, delete_cost, substitute_cost)
        previous_row = current_row
    return previous_row[-1]


def find_typo_groups(
    series: pd.Series, max_distance: int = MAX_TYPO_EDIT_DISTANCE
) -> dict[str, list[str]]:
    """Returns {common_normalized_value: [rarer_normalized_values_within_max_distance]}.
    Only ever proposes the *less frequent* of a pair as the suspected typo of
    the more frequent one — two equally-common values are left alone, since
    that's more likely two genuinely different short categories than a typo.
    """
    non_null = series.dropna().astype(str).str.strip().str.lower()
    if non_null.empty:
        return {}

    counts = non_null.value_counts()  # sorted most-frequent first
    values = counts.index.tolist()

    assigned: set[str] = set()
    groups: dict[str, list[str]] = {}
    for i, common in enumerate(values):
        if common in assigned:
            continue
        for rare in values[i + 1 :]:
            if rare in assigned or counts[rare] >= counts[common]:
                continue
            if levenshtein_distance(common, rare) <= max_distance:
                groups.setdefault(common, []).append(rare)
                assigned.add(rare)
    return groups


def detect_typos(df: pd.DataFrame, column_types: dict[str, str]) -> dict[str, AnomalyResult]:
    results: dict[str, AnomalyResult] = {}
    for col, dtype in column_types.items():
        if dtype != "categorical" or col not in df.columns:
            continue
        groups = find_typo_groups(df[col])
        if not groups:
            continue

        rare_values = {v for variants in groups.values() for v in variants}
        normalized = df[col].astype(str).str.strip().str.lower()
        mask = df[col].notna() & normalized.isin(rare_values)
        affected = int(mask.sum())
        if not affected:
            continue

        canonical, variants = next(iter(groups.items()))
        summary = (
            f"{len(groups)} possible typo group(s) (e.g. {variants[0]!r} looks like a typo of "
            f"{canonical!r}, edit distance <= {MAX_TYPO_EDIT_DISTANCE}), {affected} rows affected "
            "— flagged, not changed"
        )
        results[col] = AnomalyResult(
            check_name=f"typos:{col}",
            passed=False,
            summary=summary,
            affected_row_count=affected,
            details=df.loc[mask, [col]],
        )
    return results
