"""Per-column profiling of the *confirmed* DataFrame. Powers the UI preview
and feeds the auto-generated GX baseline suite (expectations.py) and the
IQR-based outlier detector (anomalies/outliers.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

TOP_N_CATEGORIES = 10


@dataclass
class ColumnProfile:
    dtype: str
    null_count: int
    null_pct: float
    unique_count: int
    unique_pct: float
    min: Any = None
    max: Any = None
    mean: float | None = None
    top_values: list[tuple[str, int]] = field(default_factory=list)
    samples: list[str] = field(default_factory=list)


@dataclass
class DatasetProfile:
    n_rows: int
    n_columns: int
    columns: dict[str, ColumnProfile]


def _profile_column(series: pd.Series, dtype: str) -> ColumnProfile:
    n = len(series)
    non_null = series.dropna()
    null_count = n - len(non_null)
    null_pct = null_count / n if n else 0.0
    unique_count = int(non_null.nunique())
    unique_pct = unique_count / len(non_null) if len(non_null) else 0.0

    col_min = col_max = mean = None
    top_values: list[tuple[str, int]] = []

    if dtype in ("integer", "float") and len(non_null):
        col_min = float(non_null.min())
        col_max = float(non_null.max())
        mean = float(non_null.mean())
    elif dtype == "datetime" and len(non_null):
        col_min = str(non_null.min())
        col_max = str(non_null.max())
    elif dtype == "categorical" and len(non_null):
        counts = non_null.value_counts().head(TOP_N_CATEGORIES)
        top_values = [(str(idx), int(cnt)) for idx, cnt in counts.items()]

    samples = [str(v) for v in non_null.unique()[:5]]

    return ColumnProfile(
        dtype=dtype,
        null_count=int(null_count),
        null_pct=float(null_pct),
        unique_count=unique_count,
        unique_pct=float(unique_pct),
        min=col_min,
        max=col_max,
        mean=mean,
        top_values=top_values,
        samples=samples,
    )


def profile_dataset(df: pd.DataFrame, column_types: dict[str, str]) -> DatasetProfile:
    columns = {
        col: _profile_column(df[col], column_types.get(col, "string"))
        for col in df.columns
    }
    return DatasetProfile(n_rows=len(df), n_columns=len(df.columns), columns=columns)
