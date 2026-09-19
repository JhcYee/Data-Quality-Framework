"""Case/whitespace variant grouping (pandas) — only for columns confirmed as
dtype="categorical", never free-text "string" columns, where forcing
case/whitespace variants together would be wrong (e.g. collapsing distinct
complaint-description sentences that happen to share casing quirks).

Groups values that are identical once lowercased and stripped, and suggests
the most frequent variant in each group as the canonical form. The tool only
reports this; it never rewrites the data.
"""

from __future__ import annotations

import pandas as pd

from .types import AnomalyResult


def find_variant_groups(series: pd.Series) -> dict[str, list[str]]:
    non_null = series.dropna().astype(str)
    groups: dict[str, list[str]] = {}
    for v in non_null.unique():
        key = v.strip().lower()
        groups.setdefault(key, []).append(v)
    return {k: v for k, v in groups.items() if len(v) > 1}


def suggest_canonical_forms(series: pd.Series, variant_groups: dict[str, list[str]]) -> dict[str, str]:
    counts = series.astype(str).value_counts()
    mapping: dict[str, str] = {}
    for variants in variant_groups.values():
        canonical = max(variants, key=lambda v: int(counts.get(v, 0)))
        for v in variants:
            mapping[v] = canonical
    return mapping


def detect_categorical_variants(
    df: pd.DataFrame, column_types: dict[str, str]
) -> dict[str, AnomalyResult]:
    results: dict[str, AnomalyResult] = {}
    for col, dtype in column_types.items():
        if dtype != "categorical":
            continue
        variant_groups = find_variant_groups(df[col])
        if not variant_groups:
            continue
        affected = sum(
            int(df[col].astype(str).isin(variants).sum())
            for variants in variant_groups.values()
        )
        example = next(iter(variant_groups.values()))
        summary = (
            f"{len(variant_groups)} value groups differ only by case/whitespace "
            f"(e.g. {example}), affecting {affected} rows"
        )
        results[col] = AnomalyResult(
            check_name=f"categorical_standardization:{col}",
            passed=False,
            summary=summary,
            affected_row_count=affected,
        )
    return results
