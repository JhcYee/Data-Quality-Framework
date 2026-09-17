"""Referential integrity via the SQL layer — a LEFT JOIN anti-join is
exactly a FOREIGN-KEY-style check, so it runs as real SQL against the active
SQLEngine. Only runs when a reference/lookup file was uploaded and a
(main_column, reference_column) pair was picked in the Rules step; no
reference file means the check is skipped entirely, never faked with a
hardcoded dataset-specific valid-value list.

Callers are responsible for coercing the reference file's join column to the
same confirmed type as the main file's join column before loading it into
the engine — a string-vs-int mismatch would otherwise spuriously flag every
row as orphaned.
"""

from __future__ import annotations

from ..sql_engine import SQLEngine
from .types import AnomalyResult


def _quote(col: str) -> str:
    return '"' + col.replace('"', '""') + '"'


def detect_referential_integrity(
    engine: SQLEngine,
    main_table: str,
    main_col: str,
    reference_table: str,
    reference_col: str,
) -> AnomalyResult:
    sql = (
        f"SELECT m.* FROM {_quote(main_table)} m "
        f"LEFT JOIN {_quote(reference_table)} r "
        f"ON m.{_quote(main_col)} = r.{_quote(reference_col)} "
        f"WHERE m.{_quote(main_col)} IS NOT NULL AND r.{_quote(reference_col)} IS NULL"
    )
    orphans = engine.execute(sql)
    n = len(orphans)
    summary = (
        f"{n} rows where '{main_col}' has no match in reference column '{reference_col}'"
        if n
        else f"All non-null '{main_col}' values found in reference column '{reference_col}'"
    )
    return AnomalyResult(
        check_name=f"referential:{main_col}->{reference_col}",
        passed=n == 0,
        summary=summary,
        affected_row_count=n,
        details=orphans,
    )
