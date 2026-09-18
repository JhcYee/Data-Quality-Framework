from pathlib import Path

import pandas as pd

from dq_framework.schema_confirmation import (
    ConfirmedSchema,
    coerce_column,
    coerce_custom_value,
    confirm_schema,
    find_matching_schema,
    save_confirmed_schema,
)


def test_coerce_column_string_preserves_leading_zeros():
    series = pd.Series(["00501", "00692", None])
    coerced, n_failures = coerce_column(series, "string")
    assert n_failures == 0
    assert coerced.iloc[0] == "00501"


def test_coerce_column_integer_strips_leading_zeros_and_counts_failures():
    series = pd.Series(["1", "2", "not a number", None])
    coerced, n_failures = coerce_column(series, "integer")
    assert n_failures == 1  # "not a number" fails; the pre-existing None doesn't count
    assert coerced.iloc[0] == 1


def test_coerce_column_boolean():
    series = pd.Series(["Yes", "no", "YES", "maybe", None])
    coerced, n_failures = coerce_column(series, "boolean")
    assert n_failures == 1  # "maybe"
    assert coerced.iloc[0] == True  # noqa: E712
    assert coerced.iloc[1] == False  # noqa: E712


def test_coerce_column_datetime():
    series = pd.Series(["2020-01-01", "not a date", None])
    coerced, n_failures = coerce_column(series, "datetime")
    assert n_failures == 1


def test_confirm_schema_composite_key_and_failure_counts():
    df = pd.DataFrame({"a": ["1", "x", "3"], "b": ["y", "z", "w"]})
    result = confirm_schema(df, {"a": "integer", "b": "string"}, key_columns=["a", "b"], dataset_name="t")
    assert result.coercion_failure_counts == {"a": 1}
    assert result.schema.key_columns == ["a", "b"]


def test_confirm_schema_nulls_expected_columns():
    df = pd.DataFrame({"a": ["1", "2"], "b": ["x", "y"]})
    result = confirm_schema(
        df, {"a": "integer", "b": "string"}, dataset_name="t", nulls_expected_columns=["b"]
    )
    assert result.schema.nulls_expected_columns == ["b"]


def test_confirmed_schema_json_round_trip_includes_nulls_expected():
    schema = ConfirmedSchema("orders", {"a": "integer", "b": "string"}, key_columns=["a"], nulls_expected_columns=["b"])
    restored = ConfirmedSchema.from_json(schema.to_json())
    assert restored.nulls_expected_columns == ["b"]
    assert restored.key_columns == ["a"]


def test_confirmed_schema_from_json_defaults_nulls_expected_when_absent():
    """Old saved schemas (from before this field existed) must still load."""
    old_json = '{"dataset_name": "orders", "column_types": {"a": "integer"}, "key_columns": ["a"]}'
    restored = ConfirmedSchema.from_json(old_json)
    assert restored.nulls_expected_columns == []


def test_save_and_find_matching_schema(tmp_path: Path):
    schema = ConfirmedSchema("orders", {"id": "integer", "name": "string"}, ["id"])
    save_confirmed_schema(schema, tmp_path)
    found = find_matching_schema(["id", "name"], tmp_path)
    assert found is not None
    assert found.column_types == schema.column_types


def test_find_matching_schema_partial_overlap(tmp_path: Path):
    schema = ConfirmedSchema("orders", {"id": "integer", "name": "string", "status": "categorical"}, ["id"])
    save_confirmed_schema(schema, tmp_path)
    # intersection={id,name}=2, union={id,name,status,new_col}=4 -> Jaccard 0.5, at the match threshold.
    found = find_matching_schema(["id", "name", "new_col"], tmp_path)
    assert found is not None


def test_find_matching_schema_no_match_returns_none(tmp_path: Path):
    schema = ConfirmedSchema("orders", {"id": "integer"}, [])
    save_confirmed_schema(schema, tmp_path)
    found = find_matching_schema(["totally", "different", "columns"], tmp_path)
    assert found is None


def test_coerce_custom_value_valid_integer():
    value, ok = coerce_custom_value("0", "integer")
    assert ok
    assert value == 0


def test_coerce_custom_value_invalid_integer():
    value, ok = coerce_custom_value("not a number", "integer")
    assert not ok
    assert value is None


def test_coerce_custom_value_string_accepts_anything():
    value, ok = coerce_custom_value("N/A but not a sentinel here", "string")
    assert ok
    assert value == "N/A but not a sentinel here"


def test_coerce_custom_value_invalid_datetime():
    value, ok = coerce_custom_value("definitely not a date", "datetime")
    assert not ok


def test_coerce_custom_value_valid_boolean():
    value, ok = coerce_custom_value("yes", "boolean")
    assert ok
    assert value == True  # noqa: E712
