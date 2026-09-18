import time
from pathlib import Path

import pandas as pd
import pytest

from dq_framework import expectations, pipeline
from dq_framework.cleaning import CleaningOptions
from dq_framework.ingestion import load_file
from dq_framework.profiling import profile_dataset
from dq_framework.schema_confirmation import coerce_column, confirm_schema
from dq_framework.sql_engine import SQLiteEngine, select_engine_kind, create_engine
from tests.fixtures.generate_messy import generate_borough_reference_df, generate_messy_df

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


def _run_pipeline(df: pd.DataFrame, engine_kind: str = "sqlite"):
    confirmed = confirm_schema(df, COLUMN_TYPES, key_columns=["id"], dataset_name="messy_sample")
    profile_before = profile_dataset(confirmed.confirmed_df, COLUMN_TYPES)

    engine = create_engine(engine_kind)
    try:
        reference_df = generate_borough_reference_df()
        coerced_code, _ = coerce_column(reference_df["code"], "categorical")
        reference_df = reference_df.assign(code=coerced_code)
        outcome = pipeline.run_anomaly_detection(
            confirmed.confirmed_df,
            confirmed.schema,
            profile_before,
            engine,
            reference_df=reference_df,
            referential_pairs=[("borough_code", "code")],
            consistency_pairs=[("signup_date", "close_date")],
        )
    finally:
        engine.close()

    cleaning_result = pipeline.run_cleaning(
        confirmed.confirmed_df,
        confirmed.schema,
        CleaningOptions(key_columns=["id"]),
        referential_flag_indices=outcome.referential_flag_indices,
        consistency_flag_indices=outcome.consistency_flag_indices,
        typo_flag_indices=outcome.typo_flag_indices,
    )
    profile_after = profile_dataset(cleaning_result.cleaned_df, COLUMN_TYPES)

    return confirmed, profile_before, outcome, cleaning_result, profile_after


def test_full_pipeline_small_fixture():
    ir = load_file("messy_sample.csv", FIXTURE.read_bytes())
    confirmed, profile_before, outcome, cleaning_result, profile_after = _run_pipeline(ir.raw_df)

    assert any(not r.passed for r in outcome.results)  # the fixture is deliberately messy
    assert len(cleaning_result.log) > 0
    assert cleaning_result.cleaned_df["age"].isna().sum() == 0

    assert any(r.check_name == "typos:borough" for r in outcome.results)
    assert "_typo_suspected" in cleaning_result.cleaned_df.columns
    assert cleaning_result.cleaned_df["_typo_suspected"].sum() >= 1
    # Flagged, never rewritten — the misspelling itself must still be present.
    assert "Qeens" in cleaning_result.cleaned_df["borough"].values

    reports = pipeline.generate_reports(
        "messy_sample",
        outcome.results,
        [],
        cleaning_result.log,
        profile_before,
        cleaning_result.cleaned_df,
        after_profile=profile_after,
    )
    for name in ("cleaned_dataset.csv", "changelog.md", "dq_report.html", "dq_scorecard.xlsx"):
        assert name in reports
        assert len(reports[name]) > 0


def test_gx_baseline_suite_does_not_trivially_pass_everything():
    """A baseline fit exactly to the raw data would trivially pass 100% of
    its own rules — this asserts the deliberate tolerance actually catches
    something on data with real, known injected issues."""
    ir = load_file("messy_sample.csv", FIXTURE.read_bytes())
    confirmed = confirm_schema(ir.raw_df, COLUMN_TYPES, key_columns=["id"], dataset_name="messy_sample")
    profile = profile_dataset(confirmed.confirmed_df, COLUMN_TYPES)

    gx_mod = expectations.get_gx()
    outcomes = pipeline.run_validation(confirmed.confirmed_df, confirmed.schema, profile, gx_module=gx_mod)

    assert len(outcomes) > 0
    assert any(not o.success for o in outcomes), "baseline suite passed everything — it has no real tolerance"


def test_gx_baseline_suite_skips_not_null_for_nulls_expected_columns():
    """The baseline suite's not_null gating must agree with anomalies/nulls.py's
    pass/fail logic on the same column — both keyed off the user's explicit
    'nulls expected' intent, not off how null the data already happens to be."""
    ir = load_file("messy_sample.csv", FIXTURE.read_bytes())
    column_types_with_age_expected = dict(COLUMN_TYPES)
    confirmed = confirm_schema(
        ir.raw_df,
        column_types_with_age_expected,
        key_columns=["id"],
        dataset_name="messy_sample",
        nulls_expected_columns=["age"],
    )
    profile = profile_dataset(confirmed.confirmed_df, COLUMN_TYPES)

    gx_mod = expectations.get_gx()
    outcomes = pipeline.run_validation(confirmed.confirmed_df, confirmed.schema, profile, gx_module=gx_mod)

    assert not any(o.expectation_type == "expect_column_values_to_not_be_null" and o.column == "age" for o in outcomes)
    # close_date wasn't marked expected — it should still get the rule.
    assert any(
        o.expectation_type == "expect_column_values_to_not_be_null" and o.column == "close_date" for o in outcomes
    )


@pytest.mark.slow
def test_pipeline_at_scale():
    """~500k-row fixture with the same injected issue types as the small
    fixture — checks correctness at scale, not only wall-clock speed."""
    n_rows = 500_000
    df = generate_messy_df(n_rows=n_rows, seed=7)

    start = time.time()
    engine_kind = select_engine_kind(len(df), df.memory_usage(deep=True).sum())
    assert engine_kind == "duckdb"  # crosses the auto-select threshold

    confirmed, profile_before, outcome, cleaning_result, profile_after = _run_pipeline(df, engine_kind=engine_kind)
    elapsed = time.time() - start

    assert len(cleaning_result.cleaned_df) <= len(df)
    assert cleaning_result.cleaned_df["age"].isna().sum() == 0
    assert not cleaning_result.cleaned_df["id"].duplicated().any() or "_duplicate_key" in cleaning_result.cleaned_df.columns

    reports = pipeline.generate_reports(
        "scale_test",
        outcome.results,
        [],
        cleaning_result.log,
        profile_before,
        cleaning_result.cleaned_df,
        after_profile=profile_after,
    )
    for name in ("cleaned_dataset.csv", "changelog.md", "dq_report.html", "dq_scorecard.xlsx"):
        assert len(reports[name]) > 0

    assert elapsed < 300, f"pipeline took {elapsed:.1f}s at {n_rows} rows — investigate before shipping"
