"""Null detection (pandas) — vectorized/statistical, not constraint-shaped,
so it stays in pandas rather than the SQL layer. Operates on null counts
already computed by profiling.py, which itself runs on sentinel-normalized
data (see ingestion.py), so a column full of "N/A" strings is correctly
counted here, not silently reported as 0% null.
"""

from __future__ import annotations

from ..constants import HIGH_NULL_THRESHOLD
from ..profiling import DatasetProfile
from .types import AnomalyResult


def detect_nulls(
    profile: DatasetProfile, threshold: float = HIGH_NULL_THRESHOLD
) -> list[AnomalyResult]:
    results = []
    for col, cp in profile.columns.items():
        flagged = cp.null_pct > threshold
        summary = f"{cp.null_pct:.1%} null ({cp.null_count} of {profile.n_rows} rows)"
        results.append(
            AnomalyResult(
                check_name=f"nulls:{col}",
                passed=not flagged,
                summary=summary,
                affected_row_count=cp.null_count,
            )
        )
    return results
