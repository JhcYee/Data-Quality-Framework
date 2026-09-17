from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class AnomalyResult:
    check_name: str
    passed: bool
    summary: str
    affected_row_count: int
    details: pd.DataFrame | None = None
