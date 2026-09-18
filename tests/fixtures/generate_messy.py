"""Shared generator for deliberately messy synthetic data, used to build both
the small, committed `messy_sample.csv` (fixed seed, for deterministic unit
tests) and the larger scale fixture generated on the fly at test-run time
(not committed, to avoid repo bloat) — both carry the same kinds of injected
issues (nulls incl. sentinels, exact + key duplicates, outliers, casing
variants, a referential mismatch, a consistency violation), just at
different row counts, so the scale test checks correctness, not only speed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BOROUGHS = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
BOROUGH_VARIANTS = ["Manhattan", "MANHATTAN", "manhattan ", "Brooklyn", "BROOKLYN", "Queens", "Bronx", "Staten Island"]
STATUSES = ["Open", "Closed", "Pending"]
NULL_SENTINEL_STRINGS = ["N/A", "None", "-", "Unknown"]
# "ZZ" is deliberately not in generate_borough_reference_df() — a referential-integrity mismatch.
BOROUGH_CODES = ["MN", "BK", "QN", "BX", "SI", "ZZ"]


def generate_messy_df(n_rows: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    ids = list(range(1, n_rows + 1))
    n_dup_keys = max(1, n_rows // 50)
    for pos in rng.choice(n_rows, size=n_dup_keys, replace=False):
        ids[pos] = ids[(pos + 1) % n_rows]

    ages = rng.integers(18, 70, size=n_rows).astype(object)
    for pos in rng.choice(n_rows, size=max(1, n_rows // 100), replace=False):
        ages[pos] = 999  # outlier
    null_positions = rng.choice(n_rows, size=max(1, n_rows // 20), replace=False)
    for i, pos in enumerate(null_positions):
        ages[pos] = rng.choice(NULL_SENTINEL_STRINGS) if i % 2 == 0 else ""

    signup_dates = pd.date_range("2020-01-01", periods=n_rows, freq="D")
    close_dates = pd.Series(
        signup_dates + pd.to_timedelta(rng.integers(-5, 30, size=n_rows), unit="D")
    ).astype(object)
    for pos in rng.choice(n_rows, size=max(1, n_rows // 25), replace=False):
        close_dates.iloc[pos] = None  # a few open/unresolved cases

    df = pd.DataFrame(
        {
            "id": ids,
            "name": [f"Person {i}" for i in range(n_rows)],
            "age": ages,
            "borough": rng.choice(BOROUGH_VARIANTS, size=n_rows),
            "signup_date": signup_dates.astype(str),
            "close_date": close_dates.astype(str),
            "status": rng.choice(STATUSES, size=n_rows),
            "score": rng.normal(70, 10, size=n_rows),
            "borough_code": rng.choice(BOROUGH_CODES, size=n_rows),
            # Looks numeric (heuristic guesses "integer"), but leading zeros
            # matter — a deliberately-wrong dtype guess to exercise the
            # Schema Confirmation override path.
            "zip_code": [f"{z:05d}" for z in rng.integers(501, 999, size=n_rows)],
        }
    )

    # A deliberate misspelling (not a case/whitespace variant) of a common
    # borough, to exercise fuzzy typo detection specifically — distinct from
    # the case/whitespace variants above, which normalize identically and
    # are handled by categorical_standardization instead.
    queens_positions = df.index[df["borough"] == "Queens"]
    if len(queens_positions):
        df.loc[queens_positions[0], "borough"] = "Qeens"

    n_exact_dupes = max(1, n_rows // 50)
    dupe_rows = df.sample(n=n_exact_dupes, random_state=seed)
    df = pd.concat([df, dupe_rows], ignore_index=True)
    return df


def generate_borough_reference_df() -> pd.DataFrame:
    return pd.DataFrame({"code": ["MN", "BK", "QN", "BX", "SI"], "name": BOROUGHS})


def write_fixture(path: Path, n_rows: int = 200, seed: int = 42) -> None:
    generate_messy_df(n_rows, seed).to_csv(path, index=False)


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "messy_sample.csv"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    write_fixture(out, n_rows=n)
    print(f"Wrote {out} ({n} rows)")
