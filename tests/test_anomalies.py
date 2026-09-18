from pathlib import Path

import pandas as pd
import pytest

from dq_framework.anomalies.categorical_standardization import (
    detect_categorical_variants,
    find_variant_groups,
    suggest_canonical_forms,
)
from dq_framework.anomalies.consistency import detect_consistency, suggest_pairs
from dq_framework.anomalies.duplicates import detect_exact_duplicates, detect_key_duplicates
from dq_framework.anomalies.nulls import detect_nulls
from dq_framework.anomalies.outliers import compute_outlier_bounds, detect_outliers
from dq_framework.anomalies.referential import detect_referential_integrity
from dq_framework.anomalies.schema_drift import detect_schema_drift
from dq_framework.anomalies.typos import detect_typos
from dq_framework.ingestion import load_file
from dq_framework.profiling import profile_dataset
from dq_framework.schema_confirmation import ConfirmedSchema, confirm_schema
from dq_framework.sql_engine import SQLiteEngine
from tests.fixtures.generate_messy import generate_borough_reference_df

FIXTURE = Path(__file__).parent / "fixtures" / "messy_sample.csv"

COLUMN_TYPES = {
    "id": "integer",
    "name": "string",
    "age": "integer",
    "borough": "categorical",
    "signup_date": "datetime",
    "close_date": "datetime",
    "status": "categorical",
    "score": "float",
    "borough_code": "categorical",
    "zip_code": "string",
}


@pytest.fixture(scope="module")
def confirmed():
    ir = load_file("messy_sample.csv", FIXTURE.read_bytes())
    return confirm_schema(ir.raw_df, COLUMN_TYPES, key_columns=["id"], dataset_name="messy_sample")


def test_nulls_detects_age_nulls(confirmed):
    profile = profile_dataset(confirmed.confirmed_df, COLUMN_TYPES)
    results = {r.check_name: r for r in detect_nulls(profile)}
    assert results["nulls:age"].affected_row_count > 0


def test_exact_duplicates_detected(confirmed):
    engine = SQLiteEngine()
    try:
        engine.load(confirmed.confirmed_df, "main")
        result = detect_exact_duplicates(engine, "main", list(confirmed.confirmed_df.columns))
    finally:
        engine.close()
    assert not result.passed
    assert result.affected_row_count > 0


def test_key_duplicates_detected(confirmed):
    engine = SQLiteEngine()
    try:
        engine.load(confirmed.confirmed_df, "main")
        result = detect_key_duplicates(engine, "main", ["id"])
    finally:
        engine.close()
    assert result is not None
    assert not result.passed


def test_key_duplicates_none_when_no_key_columns():
    engine = SQLiteEngine()
    try:
        engine.load(pd.DataFrame({"a": [1, 2]}), "main")
        result = detect_key_duplicates(engine, "main", [])
    finally:
        engine.close()
    assert result is None


def test_outlier_bounds_and_detection():
    # Needs >= MIN_DISTINCT_FOR_OUTLIER_CHECK distinct values before IQR
    # fences apply at all (see outliers.py's low-cardinality guard).
    series = pd.Series(list(range(1, 13)) + [999])
    bounds = compute_outlier_bounds(series)
    assert bounds is not None
    lo, hi = bounds
    assert 999 > hi


def test_outliers_skip_low_cardinality_column():
    ratings = pd.Series([1, 2, 3, 4, 5] * 20)  # only 5 distinct values
    assert compute_outlier_bounds(ratings) is None


def test_detect_outliers_flags_injected_age_outlier(confirmed):
    results = detect_outliers(confirmed.confirmed_df, COLUMN_TYPES)
    assert "age" in results
    assert results["age"].affected_row_count > 0


def test_categorical_variant_grouping():
    series = pd.Series(["Manhattan", "MANHATTAN", "manhattan ", "Brooklyn"])
    groups = find_variant_groups(series)
    assert "manhattan" in groups
    assert len(groups["manhattan"]) == 3
    mapping = suggest_canonical_forms(series, groups)
    assert mapping["MANHATTAN"] == mapping["manhattan "] == mapping["Manhattan"]


def test_detect_categorical_variants_on_fixture(confirmed):
    results = detect_categorical_variants(confirmed.confirmed_df, COLUMN_TYPES)
    assert "borough" in results
    assert results["borough"].affected_row_count > 0


def test_detect_typos_on_fixture(confirmed):
    """messy_sample.csv has a deliberate "Qeens" misspelling of "Queens",
    distinct from the case/whitespace variants (which categorical_standardization
    already catches, not this detector)."""
    results = detect_typos(confirmed.confirmed_df, COLUMN_TYPES)
    assert "borough" in results
    assert not results["borough"].passed
    assert results["borough"].affected_row_count >= 1
    assert results["borough"].details is not None
    assert "qeens" in results["borough"].details["borough"].str.lower().values


def test_consistency_detects_close_before_signup(confirmed):
    engine = SQLiteEngine()
    try:
        engine.load(confirmed.confirmed_df, "main")
        result = detect_consistency(engine, "main", "signup_date", "close_date")
    finally:
        engine.close()
    assert result.affected_row_count > 0


def test_suggest_pairs_finds_signup_close():
    pairs = suggest_pairs(COLUMN_TYPES)
    assert ("signup_date", "close_date") in pairs


def test_referential_integrity_detects_orphans(confirmed):
    reference_df = generate_borough_reference_df()
    engine = SQLiteEngine()
    try:
        engine.load(confirmed.confirmed_df, "main")
        engine.load(reference_df, "reference")
        result = detect_referential_integrity(engine, "main", "borough_code", "reference", "code")
    finally:
        engine.close()
    assert result.affected_row_count > 0  # 'ZZ' codes have no match


def test_schema_drift_no_prior_schema():
    current = ConfirmedSchema("x", {"a": "string"}, [])
    result = detect_schema_drift(current, None)
    assert result.passed


def test_schema_drift_detects_type_change():
    current = ConfirmedSchema("x", {"a": "integer"}, [])
    previous = ConfirmedSchema("x", {"a": "string"}, [])
    result = detect_schema_drift(current, previous)
    assert not result.passed
    assert "a" in result.summary
