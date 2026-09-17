"""Cross-column logical consistency checks via the SQL layer — a
"WHERE end < start" query is exactly a CHECK-constraint-style check, so it
runs as real SQL. suggest_pairs() only *suggests* likely (start, end) column
pairs by name pattern; the user confirms which pairs to actually check in
the Rules step, the same "infer as suggestion, let the user confirm" pattern
used for dtype guessing and primary-key detection.
"""

from __future__ import annotations

from ..sql_engine import SQLEngine
from .types import AnomalyResult

START_KEYWORDS = ["start", "open", "created", "begin", "issued", "signup", "submitted", "registered"]
END_KEYWORDS = ["end", "close", "closed", "finish", "finished", "resolved", "completed"]


def _quote(col: str) -> str:
    return '"' + col.replace('"', '""') + '"'


def suggest_pairs(column_types: dict[str, str]) -> list[tuple[str, str]]:
    datetime_cols = [c for c, t in column_types.items() if t == "datetime"]
    pairs = []
    for a in datetime_cols:
        for b in datetime_cols:
            if a == b:
                continue
            a_l, b_l = a.lower(), b.lower()
            if any(k in a_l for k in START_KEYWORDS) and any(k in b_l for k in END_KEYWORDS):
                pairs.append((a, b))
    return pairs


def detect_consistency(
    engine: SQLEngine, table_name: str, start_col: str, end_col: str
) -> AnomalyResult:
    sql = (
        f"SELECT * FROM {_quote(table_name)} "
        f"WHERE {_quote(end_col)} < {_quote(start_col)} "
        f"AND {_quote(start_col)} IS NOT NULL AND {_quote(end_col)} IS NOT NULL"
    )
    violations = engine.execute(sql)
    n = len(violations)
    summary = (
        f"{n} rows where '{end_col}' is before '{start_col}'"
        if n
        else f"All rows satisfy '{end_col}' >= '{start_col}'"
    )
    return AnomalyResult(
        check_name=f"consistency:{end_col}>={start_col}",
        passed=n == 0,
        summary=summary,
        affected_row_count=n,
        details=violations,
    )
