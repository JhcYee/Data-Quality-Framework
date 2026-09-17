"""Turns the user-reviewed/overridden dtype map from the Confirm Schema step
into a final, coerced DataFrame — the single authoritative typed source
everything downstream (profiling, both SQL engines, outlier detection, the
GX baseline suite) reads from.

Values that fail to coerce under the *confirmed* type become null and are
counted as a coercion-failure anomaly right here. By the time Cleaning runs
later in the pipeline, those are already ordinary nulls — Cleaning has no
separate "resolve coercion failures" step.

"categorical" is tracked as our own semantic tag (not pandas' Categorical
dtype, which raises on values outside a fixed category set — too brittle for
messy real data); categorical columns are kept as pandas' nullable "string"
dtype underneath, same as "string" columns, and the tag alone is what makes
categorical_standardization.py and the GX baseline's in_set expectations
treat them differently from free text.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .constants import DTYPE_CHOICES

BOOLEAN_MAP = {
    "true": True, "t": True, "yes": True, "y": True, "1": True,
    "false": False, "f": False, "no": False, "n": False, "0": False,
}

# A later upload's column-name set must overlap a saved schema's by at least
# this fraction (Jaccard similarity) to be considered "the same dataset".
SCHEMA_MATCH_THRESHOLD = 0.5


@dataclass
class ConfirmedSchema:
    dataset_name: str
    column_types: dict[str, str]  # col -> one of DTYPE_CHOICES
    key_columns: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            {
                "dataset_name": self.dataset_name,
                "column_types": self.column_types,
                "key_columns": self.key_columns,
            },
            indent=2,
        )

    @staticmethod
    def from_json(text: str) -> "ConfirmedSchema":
        data = json.loads(text)
        return ConfirmedSchema(
            dataset_name=data["dataset_name"],
            column_types=data["column_types"],
            key_columns=data.get("key_columns", []),
        )


@dataclass
class SchemaConfirmationResult:
    confirmed_df: pd.DataFrame
    schema: ConfirmedSchema
    coercion_failure_counts: dict[str, int]


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or "dataset"


def _str_or_none(v):
    return str(v).strip() if pd.notna(v) else None


def coerce_column(series: pd.Series, dtype: str) -> tuple[pd.Series, int]:
    if dtype not in DTYPE_CHOICES:
        raise ValueError(f"Unknown dtype '{dtype}', expected one of {DTYPE_CHOICES}")

    non_null_mask = series.notna()

    if dtype in ("string", "categorical"):
        return series.astype("string"), 0

    str_series = series.map(_str_or_none)

    if dtype in ("integer", "float"):
        numeric = pd.to_numeric(str_series, errors="coerce")
        n_failures = int((non_null_mask & numeric.isna()).sum())
        if dtype == "integer":
            coerced = numeric.round().astype("Int64")
        else:
            coerced = numeric.astype("float64")
        return coerced, n_failures

    if dtype == "boolean":
        # pandas' default "str" dtype represents a missing value as a float
        # nan internally, not Python None, even though _str_or_none returns
        # None — so this must check pd.notna(), never `v is not None`.
        lowered = str_series.map(lambda v: v.lower() if pd.notna(v) else None)
        mapped = lowered.map(BOOLEAN_MAP)
        n_failures = int((non_null_mask & mapped.isna()).sum())
        return mapped.astype("boolean"), n_failures

    if dtype == "datetime":
        coerced = pd.to_datetime(str_series, errors="coerce", format="mixed")
        n_failures = int((non_null_mask & coerced.isna()).sum())
        return coerced, n_failures

    raise AssertionError("unreachable")  # pragma: no cover


def confirm_schema(
    raw_df: pd.DataFrame,
    column_types: dict[str, str],
    key_columns: list[str] | None = None,
    dataset_name: str = "dataset",
) -> SchemaConfirmationResult:
    confirmed_df = pd.DataFrame(index=raw_df.index)
    failure_counts: dict[str, int] = {}

    for col in raw_df.columns:
        dtype = column_types.get(col, "string")
        coerced, n_failures = coerce_column(raw_df[col], dtype)
        confirmed_df[col] = coerced
        if n_failures:
            failure_counts[col] = n_failures

    schema = ConfirmedSchema(
        dataset_name=dataset_name,
        column_types=dict(column_types),
        key_columns=list(key_columns or []),
    )
    return SchemaConfirmationResult(confirmed_df, schema, failure_counts)


# --------------------------------------------------------------------------
# Persistence — local disk only (see plan's "Known Limitations": this is a
# local/dev-scoped convenience, not durable storage for shared hosting).
# --------------------------------------------------------------------------


def schema_path(schemas_dir: Path, dataset_name: str) -> Path:
    return schemas_dir / f"{_slugify(dataset_name)}_confirmed_schema.json"


def save_confirmed_schema(schema: ConfirmedSchema, schemas_dir: Path) -> Path:
    schemas_dir.mkdir(parents=True, exist_ok=True)
    path = schema_path(schemas_dir, schema.dataset_name)
    path.write_text(schema.to_json())
    return path


def list_saved_schemas(schemas_dir: Path) -> list[ConfirmedSchema]:
    if not schemas_dir.exists():
        return []
    schemas = []
    for path in schemas_dir.glob("*_confirmed_schema.json"):
        try:
            schemas.append(ConfirmedSchema.from_json(path.read_text()))
        except (json.JSONDecodeError, KeyError):
            continue
    return schemas


def find_matching_schema(
    columns: list[str], schemas_dir: Path
) -> ConfirmedSchema | None:
    """'Matching dataset' = column-name-set overlap above threshold, not an
    exact positional match — a later upload can add/drop a few columns and
    still pre-fill from the closest prior schema."""
    incoming = set(columns)
    best: ConfirmedSchema | None = None
    best_score = 0.0
    for schema in list_saved_schemas(schemas_dir):
        existing = set(schema.column_types.keys())
        union = incoming | existing
        if not union:
            continue
        score = len(incoming & existing) / len(union)
        if score >= SCHEMA_MATCH_THRESHOLD and score > best_score:
            best, best_score = schema, score
    return best
