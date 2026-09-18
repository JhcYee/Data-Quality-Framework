"""Null detection (pandas) — vectorized/statistical, not constraint-shaped,
so it stays in pandas rather than the SQL layer. Operates on null counts
already computed by profiling.py, which itself runs on sentinel-normalized
data (see ingestion.py), so a column full of "N/A" strings is correctly
counted here, not silently reported as 0% null.

Flags at any null rate by default — a column nobody said would have nulls
having one at all is exactly the kind of thing worth surfacing. A column the
user explicitly marked "nulls expected" in Schema Confirmation gets the more
lenient HIGH_NULL_THRESHOLD instead, so routine partial-completion columns
(e.g. an optional survey field) don't get flagged for every single null.
"""

from __future__ import annotations

from ..constants import HIGH_NULL_THRESHOLD, STRICT_NULL_THRESHOLD
from ..profiling import DatasetProfile
from .types import AnomalyResult


def detect_nulls(
    profile: DatasetProfile,
    nulls_expected_columns: set[str] | None = None,
    lenient_threshold: float = HIGH_NULL_THRESHOLD,
    strict_threshold: float = STRICT_NULL_THRESHOLD,
) -> list[AnomalyResult]:
    nulls_expected_columns = nulls_expected_columns or set()
    results = []
    for col, cp in profile.columns.items():
        expected = col in nulls_expected_columns
        threshold = lenient_threshold if expected else strict_threshold
        flagged = cp.null_pct > threshold
        note = " (marked as expecting some nulls)" if expected else ""
        summary = f"{cp.null_pct:.1%} null ({cp.null_count} of {profile.n_rows} rows){note}"
        results.append(
            AnomalyResult(
                check_name=f"nulls:{col}",
                passed=not flagged,
                summary=summary,
                affected_row_count=cp.null_count,
            )
        )
    return results
