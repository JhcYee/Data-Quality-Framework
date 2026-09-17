"""SQLite/DuckDB dual engine for constraint-style checks (duplicate-key
UNIQUE checks, cross-column CHECK-style consistency checks, FOREIGN-KEY-style
referential integrity anti-joins). Both engines execute the same plain SQL
against a small shared interface, so the choice of engine never changes a
result — only how fast it runs.

Auto-selection favors DuckDB (columnar, fast on large scans/joins) once a
dataset crosses a size/row threshold, and SQLite (stdlib, no extra process)
otherwise; the caller can always override.
"""

from __future__ import annotations

import sqlite3
from typing import Protocol

import pandas as pd

from .constants import SQL_ENGINE_ROW_THRESHOLD, SQL_ENGINE_SIZE_THRESHOLD_BYTES

ENGINE_CHOICES = ["auto", "sqlite", "duckdb"]


class SQLEngine(Protocol):
    def load(self, df: pd.DataFrame, table_name: str) -> None: ...
    def execute(self, sql: str) -> pd.DataFrame: ...
    def close(self) -> None: ...


class SQLiteEngine:
    name = "sqlite"

    def __init__(self) -> None:
        self._conn = sqlite3.connect(":memory:")

    def load(self, df: pd.DataFrame, table_name: str) -> None:
        # Extension dtypes (Int64, boolean, string) round-trip through
        # sqlite fine via to_sql; pd.NA becomes SQL NULL.
        df.to_sql(table_name, self._conn, if_exists="replace", index=False)

    def execute(self, sql: str) -> pd.DataFrame:
        return pd.read_sql_query(sql, self._conn)

    def close(self) -> None:
        self._conn.close()


class DuckDBEngine:
    name = "duckdb"

    def __init__(self) -> None:
        import duckdb

        self._conn = duckdb.connect(":memory:")

    def load(self, df: pd.DataFrame, table_name: str) -> None:
        # register() exposes the DataFrame as a view with no copy.
        self._conn.register(table_name, df)

    def execute(self, sql: str) -> pd.DataFrame:
        return self._conn.execute(sql).df()

    def close(self) -> None:
        self._conn.close()


def select_engine_kind(
    n_rows: int, size_bytes: int, override: str = "auto"
) -> str:
    if override != "auto":
        return override
    if n_rows > SQL_ENGINE_ROW_THRESHOLD or size_bytes > SQL_ENGINE_SIZE_THRESHOLD_BYTES:
        return "duckdb"
    return "sqlite"


def create_engine(kind: str) -> SQLEngine:
    if kind == "duckdb":
        return DuckDBEngine()
    if kind == "sqlite":
        return SQLiteEngine()
    raise ValueError(f"Unknown SQL engine kind '{kind}'")
