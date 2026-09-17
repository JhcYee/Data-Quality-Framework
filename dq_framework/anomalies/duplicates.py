"""Duplicate detection via the SQL layer — GROUP BY ... HAVING COUNT(*) > 1
is exactly a UNIQUE-constraint-style check, so this runs as real SQL against
the active SQLEngine rather than in pandas.

Key-column duplicate detection reuses the column(s) the user explicitly
marked as primary/unique key in Schema Confirmation — no separate heuristic
guessing of "a likely ID column" here. When more than one column was marked,
they're treated together as a single composite key.
"""

from __future__ import annotations

import pandas as pd

from ..sql_engine import SQLEngine
from .types import AnomalyResult


def _quote(col: str) -> str:
    return '"' + col.replace('"', '""') + '"'


def _group_duplicates(engine: SQLEngine, table_name: str, columns: list[str]) -> pd.DataFrame:
    cols_sql = ", ".join(_quote(c) for c in columns)
    sql = (
        f"SELECT {cols_sql}, COUNT(*) AS dup_count "
        f"FROM {_quote(table_name)} "
        f"GROUP BY {cols_sql} HAVING COUNT(*) > 1"
    )
    return engine.execute(sql)


def detect_exact_duplicates(engine: SQLEngine, table_name: str, columns: list[str]) -> AnomalyResult:
    dupes = _group_duplicates(engine, table_name, columns)
    affected = int(dupes["dup_count"].sum()) if not dupes.empty else 0
    summary = (
        f"{len(dupes)} distinct duplicate row groups, {affected} rows total"
        if not dupes.empty
        else "No exact duplicate rows"
    )
    return AnomalyResult(
        check_name="duplicates:exact_rows",
        passed=dupes.empty,
        summary=summary,
        affected_row_count=affected,
        details=dupes,
    )


def detect_key_duplicates(
    engine: SQLEngine, table_name: str, key_columns: list[str]
) -> AnomalyResult | None:
    if not key_columns:
        return None
    dupes = _group_duplicates(engine, table_name, key_columns)
    affected = int(dupes["dup_count"].sum()) if not dupes.empty else 0
    key_desc = ", ".join(key_columns)
    summary = (
        f"{len(dupes)} duplicate key groups on ({key_desc}), {affected} rows total"
        if not dupes.empty
        else f"No duplicate keys on ({key_desc})"
    )
    return AnomalyResult(
        check_name=f"duplicates:key[{key_desc}]",
        passed=dupes.empty,
        summary=summary,
        affected_row_count=affected,
        details=dupes,
    )
