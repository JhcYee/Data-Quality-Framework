"""IQR-based outlier detection (pandas) — naturally vectorized/statistical,
stays in pandas rather than the SQL layer.

compute_outlier_bounds is shared with expectations.py's auto-generated
baseline suite (same IQR_MULTIPLIER), so an outlier flag here and a baseline
range-expectation failure there agree on the same rows instead of silently
disagreeing.

Skips columns with too few distinct values — a 1-5 rating or a 0/1 flag
confirmed as "integer" isn't a continuous measurement, and IQR fences on a
handful of discrete values produce nonsense. Flags, never drops.
"""

from __future__ import annotations

import pandas as pd

from ..constants import IQR_MULTIPLIER, MIN_DISTINCT_FOR_OUTLIER_CHECK
from .types import AnomalyResult


def compute_outlier_bounds(series: pd.Series) -> tuple[float, float] | None:
    non_null = series.dropna()
    if non_null.nunique() < MIN_DISTINCT_FOR_OUTLIER_CHECK:
        return None
    q1, q3 = non_null.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr == 0:
        return None
    return float(q1 - IQR_MULTIPLIER * iqr), float(q3 + IQR_MULTIPLIER * iqr)


def outlier_mask(series: pd.Series, bounds: tuple[float, float]) -> pd.Series:
    lo, hi = bounds
    return series.notna() & ((series.astype("float64") < lo) | (series.astype("float64") > hi))


def detect_outliers(df: pd.DataFrame, column_types: dict[str, str]) -> dict[str, AnomalyResult]:
    results: dict[str, AnomalyResult] = {}
    for col, dtype in column_types.items():
        if dtype not in ("integer", "float"):
            continue
        bounds = compute_outlier_bounds(df[col])
        if bounds is None:
            continue
        mask = outlier_mask(df[col], bounds)
        n = int(mask.sum())
        lo, hi = bounds
        summary = (
            f"{n} values outside [{lo:.2f}, {hi:.2f}]"
            if n
            else f"No outliers (IQR bounds [{lo:.2f}, {hi:.2f}])"
        )
        results[col] = AnomalyResult(
            check_name=f"outliers:{col}",
            passed=n == 0,
            summary=summary,
            affected_row_count=n,
        )
    return results
