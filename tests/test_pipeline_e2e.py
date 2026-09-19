import time
from pathlib import Path

import pandas as pd
import pytest

from dq_framework import expectations, pipeline
from dq_framework.ingestion import load_file
from dq_framework.profiling import profile_dataset
from dq_framework.recommendations import build_recommendations
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
    profile = profile_dataset(confirmed.confirmed_df, COLUMN_TYPES)

    engine = create_engine(engine_kind)
    try:
        reference_df = generate_borough_reference_df()
        coerced_code, _ = coerce_column(reference_df["code"], "categorical")
        reference_df = reference_df.assign(code=coerced_code)
        outcome = pipeline.run_anomaly_detection(
            confirmed.confirmed_df,
            confirmed.schema,
            profile,
            engine,
            reference_df=reference_df,
            referential_pairs=[("borough_code", "code")],
            consistency_pairs=[("signup_date", "close_date")],
        )
    finally:
        engine.close()

    recommendations = build_recommendations(
        outcome.results, len(confirmed.confirmed_df), coercion_failure_counts=confirmed.coercion_failure_counts
    )
    return confirmed, profile, outcome, recommendations


def test_full_pipeline_small_fixture():
    ir = load_file("messy_sample.csv", FIXTURE.read_bytes())
    confirmed, profile, outcome, recommendations = _run_pipeline(ir.raw_df)

    assert any(not r.passed for r in outcome.results)  # the fixture is deliberately messy
    assert len(recommendations) > 0

    issues = {r.issue for r in recommendations}
    assert {"Duplicate rows", "Missing values", "Possible typos", "Referential integrity break"} <= issues

    reports = pipeline.generate_reports("messy_sample", outcome.results, [], recommendations, profile)
    assert set(reports) == {"recommended_actions.md", "dq_report.html", "dq_scorecard.xlsx"}
    assert all(len(data) > 0 for data in reports.values())
    assert "Qeens" in reports["dq_report.html"].decode() or "qeens" in reports["dq_report.html"].decode()


def test_failed_gx_rules_show_up_as_recommended_actions():
    """End to end through real Great Expectations: a rule the user adds (email
    must be unique) fails, and that failure appears in the recommendations."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "email": ["a@x.com", "b@x.com", "a@x.com", "c@x.com"],
            "status": ["ok", "ok", "ok", "ok"],
        }
    )
    types = {"id": "integer", "email": "string", "status": "categorical"}
    confirmed = confirm_schema(df, types, key_columns=["id"], dataset_name="t")
    profile = profile_dataset(confirmed.confirmed_df, types)
    outcomes = pipeline.run_validation(
        confirmed.confirmed_df,
        confirmed.schema,
        profile,
        [expectations.ExpectationSpec("unique", "email")],
    )
    # Only the custom rule is tagged custom; the auto-generated ones aren't.
    assert [(o.expectation_type, o.column) for o in outcomes if o.source == "custom"] == [
        ("expect_column_values_to_be_unique", "email")
    ]
    assert sum(o.source == "auto-generated" for o in outcomes) == len(outcomes) - 1
    recs = build_recommendations([], len(df), validation_outcomes=outcomes)
    (rec,) = recs
    assert (rec.column, rec.issue) == ("email", "Values that should be unique repeat")
    assert rec.affected_rows == 2  # both rows sharing a@x.com


def test_audit_never_modifies_the_confirmed_data():
    """The tool audits; running the whole pipeline must leave the data it was
    given exactly as it found it."""
    ir = load_file("messy_sample.csv", FIXTURE.read_bytes())
    confirmed = confirm_schema(ir.raw_df, COLUMN_TYPES, key_columns=["id"], dataset_name="messy_sample")
    before = confirmed.confirmed_df.copy()
    _run_pipeline(ir.raw_df)
    pd.testing.assert_frame_equal(confirmed.confirmed_df, before)


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

    confirmed, profile, outcome, recommendations = _run_pipeline(df, engine_kind=engine_kind)
    elapsed = time.time() - start

    assert len(recommendations) > 0
    assert any(r.issue == "Duplicate keys" for r in recommendations)

    reports = pipeline.generate_reports("scale_test", outcome.results, [], recommendations, profile)
    assert all(len(data) > 0 for data in reports.values())

    assert elapsed < 300, f"pipeline took {elapsed:.1f}s at {n_rows} rows — investigate before shipping"


def _small_validation_setup():
    df = pd.DataFrame({"id": range(30), "s": ["a", "b", "c"] * 10, "n": [float(i) for i in range(30)]})
    types = {"id": "integer", "s": "categorical", "n": "float"}
    confirmed = confirm_schema(df, types, key_columns=["id"], dataset_name="t")
    return confirmed, profile_dataset(confirmed.confirmed_df, types)


def test_rules_that_cannot_run_are_reported_as_errors_not_failures_or_crashes():
    from dq_framework.expectations import ExpectationSpec

    confirmed, profile = _small_validation_setup()
    outcomes = pipeline.run_validation(
        confirmed.confirmed_df,
        confirmed.schema,
        profile,
        [
            ExpectationSpec("regex", "s", {"regex": "("}),  # invalid pattern
            ExpectationSpec("not_null", "ghost"),  # column doesn't exist
            ExpectationSpec("between", "n", {}),  # no bounds: can't even be built
            ExpectationSpec("unique", "s"),  # runs fine and fails for real
        ],
    )
    by_status = {}
    for o in outcomes:
        by_status.setdefault(o.status, []).append((o.expectation_type.removeprefix("expect_"), o.column))
    assert sorted(by_status["ERROR"]) == [
        ("column_values_to_be_between", "n"),
        ("column_values_to_match_regex", "s"),
        ("column_values_to_not_be_null", "ghost"),
    ]
    assert ("column_values_to_be_unique", "s") in by_status["FAIL"]
    assert all(o.error and o.source == "custom" for o in outcomes if o.status == "ERROR")

    recs = build_recommendations([], 30, validation_outcomes=outcomes)
    could_not_run = [r for r in recs if r.issue == "Validation rule could not run"]
    assert len(could_not_run) == 3
    assert any("Invalid regular expression" in r.finding for r in could_not_run)
    assert any(r.issue == "Values that should be unique repeat" for r in recs)


def test_referential_check_survives_a_type_mismatch_between_the_two_files():
    df = pd.DataFrame({"code": ["1", "2", "x", None]})
    types = {"code": "string"}
    confirmed = confirm_schema(df, types, dataset_name="t")
    profile = profile_dataset(confirmed.confirmed_df, types)
    for kind in ("sqlite", "duckdb"):
        engine = create_engine(kind)
        try:
            outcome = pipeline.run_anomaly_detection(
                confirmed.confirmed_df, confirmed.schema, profile, engine,
                reference_df=pd.DataFrame({"c": [1, 2, 3]}),  # ints vs. the main file's strings
                referential_pairs=[("code", "c")],
            )
        finally:
            engine.close()
        (ref,) = [r for r in outcome.results if r.check_name.startswith("referential")]
        assert ref.affected_row_count == 1  # only "x" has no match


def test_typo_matching_skips_columns_with_too_many_distinct_values():
    from dq_framework.anomalies.typos import detect_typos, find_typo_groups
    from dq_framework.constants import MAX_TYPO_DISTINCT_VALUES

    values = [f"value_{i}" for i in range(MAX_TYPO_DISTINCT_VALUES + 50)]
    df = pd.DataFrame({"c": values})
    assert find_typo_groups(df["c"]) == {}
    result = detect_typos(df, {"c": "categorical"})["c"]
    assert result.passed and "Skipped" in result.summary
