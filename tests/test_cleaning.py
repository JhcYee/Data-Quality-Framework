from pathlib import Path

import pandas as pd

from dq_framework.cleaning import CleaningOptions, clean_dataset
from dq_framework.ingestion import load_file
from dq_framework.schema_confirmation import confirm_schema

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


def _confirmed():
    ir = load_file("messy_sample.csv", FIXTURE.read_bytes())
    return confirm_schema(ir.raw_df, COLUMN_TYPES, key_columns=["id"], dataset_name="messy_sample")


def test_drop_exact_duplicates_reduces_row_count():
    confirmed = _confirmed()
    before = len(confirmed.confirmed_df)
    options = CleaningOptions(key_columns=["id"], drop_exact_duplicates=True)
    result = clean_dataset(confirmed.confirmed_df, COLUMN_TYPES, options)
    assert len(result.cleaned_df) < before
    assert any(e.change_type == "drop_exact_duplicates" for e in result.log)


def test_no_op_when_disabled_leaves_duplicates():
    confirmed = _confirmed()
    before = len(confirmed.confirmed_df)
    options = CleaningOptions(key_columns=["id"], drop_exact_duplicates=False, apply_categorical_standardization=False)
    result = clean_dataset(confirmed.confirmed_df, COLUMN_TYPES, options)
    assert len(result.cleaned_df) == before


def test_cleaning_is_reversible_not_accumulated():
    """Re-running with different options from the same source df must give
    independent results, not chain on top of a prior cleaning pass."""
    confirmed = _confirmed()
    original_len = len(confirmed.confirmed_df)

    with_drop = clean_dataset(confirmed.confirmed_df, COLUMN_TYPES, CleaningOptions(key_columns=["id"], drop_exact_duplicates=True))
    assert len(with_drop.cleaned_df) < original_len

    without_drop = clean_dataset(confirmed.confirmed_df, COLUMN_TYPES, CleaningOptions(key_columns=["id"], drop_exact_duplicates=False))
    assert len(without_drop.cleaned_df) == original_len  # unaffected by the earlier call

    # The source DataFrame itself must never be mutated.
    assert len(confirmed.confirmed_df) == original_len


def test_key_duplicates_flagged_not_dropped_by_default():
    confirmed = _confirmed()
    options = CleaningOptions(key_columns=["id"], drop_exact_duplicates=False, drop_key_duplicates=False)
    result = clean_dataset(confirmed.confirmed_df, COLUMN_TYPES, options)
    assert "_duplicate_key" in result.cleaned_df.columns
    assert result.cleaned_df["_duplicate_key"].sum() > 0
    assert len(result.cleaned_df) == len(confirmed.confirmed_df)  # nothing dropped


def test_outlier_flag_default():
    confirmed = _confirmed()
    options = CleaningOptions(key_columns=["id"], outlier_action="flag", drop_exact_duplicates=False)
    result = clean_dataset(confirmed.confirmed_df, COLUMN_TYPES, options)
    assert "_is_outlier" in result.cleaned_df.columns
    assert result.cleaned_df["_is_outlier"].sum() > 0
    assert (result.cleaned_df["age"] == 999).any()  # untouched


def test_outlier_cap():
    confirmed = _confirmed()
    options = CleaningOptions(key_columns=["id"], outlier_action="cap", drop_exact_duplicates=False)
    result = clean_dataset(confirmed.confirmed_df, COLUMN_TYPES, options)
    assert not (result.cleaned_df["age"] == 999).any()


def test_outlier_drop():
    confirmed = _confirmed()
    before = len(confirmed.confirmed_df)
    options = CleaningOptions(key_columns=["id"], outlier_action="drop", drop_exact_duplicates=False)
    result = clean_dataset(confirmed.confirmed_df, COLUMN_TYPES, options)
    assert len(result.cleaned_df) < before


def test_categorical_standardization_reduces_variants():
    confirmed = _confirmed()
    options = CleaningOptions(key_columns=["id"], apply_categorical_standardization=True, drop_exact_duplicates=False)
    result = clean_dataset(confirmed.confirmed_df, COLUMN_TYPES, options)
    variants = {v.strip().lower() for v in result.cleaned_df["borough"].unique()}
    raw_variants = {v.strip().lower() for v in confirmed.confirmed_df["borough"].unique()}
    assert len(variants) == len(raw_variants)  # same underlying groups
    assert result.cleaned_df["borough"].nunique() < confirmed.confirmed_df["borough"].nunique()


def test_null_imputation_numeric_and_categorical():
    confirmed = _confirmed()
    options = CleaningOptions(key_columns=["id"], drop_exact_duplicates=False)
    result = clean_dataset(confirmed.confirmed_df, COLUMN_TYPES, options)
    assert result.cleaned_df["age"].isna().sum() == 0
    assert any(e.column == "age" and e.change_type == "impute_nulls" for e in result.log)


def test_datetime_nulls_flagged_not_imputed():
    confirmed = _confirmed()
    options = CleaningOptions(key_columns=["id"], drop_exact_duplicates=False)
    result = clean_dataset(confirmed.confirmed_df, COLUMN_TYPES, options)
    assert "_close_date_was_null" in result.cleaned_df.columns
    assert result.cleaned_df["_close_date_was_null"].sum() > 0
    # still null, not guessed
    assert result.cleaned_df.loc[result.cleaned_df["_close_date_was_null"], "close_date"].isna().all()


def test_null_strategy_skip_leaves_column_untouched():
    confirmed = _confirmed()
    options = CleaningOptions(key_columns=["id"], drop_exact_duplicates=False, null_strategy_overrides={"age": "skip"})
    before_nulls = confirmed.confirmed_df["age"].isna().sum()
    result = clean_dataset(confirmed.confirmed_df, COLUMN_TYPES, options)
    assert result.cleaned_df["age"].isna().sum() == before_nulls


def test_referential_and_consistency_flags():
    confirmed = _confirmed()
    idx = confirmed.confirmed_df.index[:3]
    options = CleaningOptions(key_columns=["id"], drop_exact_duplicates=False)
    result = clean_dataset(
        confirmed.confirmed_df,
        COLUMN_TYPES,
        options,
        referential_flag_indices={"referential:borough_code->code": pd.Index(idx)},
        consistency_flag_indices={"consistency:close_date>=signup_date": pd.Index(idx)},
    )
    assert result.cleaned_df.loc[idx, "_referential_issue"].all()
    assert result.cleaned_df.loc[idx, "_consistency_issue"].all()


def test_evidence_based_reasons_contain_numbers():
    confirmed = _confirmed()
    options = CleaningOptions(key_columns=["id"], drop_exact_duplicates=True)
    result = clean_dataset(confirmed.confirmed_df, COLUMN_TYPES, options)
    for entry in result.log:
        assert any(ch.isdigit() for ch in entry.reason), f"reason has no evidence: {entry.reason}"
