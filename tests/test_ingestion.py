from pathlib import Path

import pandas as pd
import pytest

from dq_framework import ingestion
from dq_framework.ingestion import IngestionError, load_file, normalize_null_sentinels

FIXTURE = Path(__file__).parent / "fixtures" / "messy_sample.csv"


def test_load_csv_basic():
    result = load_file("messy_sample.csv", FIXTURE.read_bytes())
    assert len(result.raw_df) == 204  # 200 base rows + 4 deliberately injected duplicates
    assert "age" in result.raw_df.columns


def test_null_sentinels_normalized():
    result = load_file("messy_sample.csv", FIXTURE.read_bytes())
    assert "age" in result.sentinel_null_counts
    assert result.sentinel_null_counts["age"] > 0
    # sentinel strings must actually be gone from the column
    non_null = result.raw_df["age"].dropna().astype(str).str.lower()
    assert not non_null.isin(["n/a", "none", "-", "unknown"]).any()


def test_dtype_guess_zip_code_is_deliberately_wrong():
    """zip_code looks numeric (heuristic will guess integer/float) but
    should really stay a string — this is the fixture column designed to
    exercise Schema Confirmation's override path."""
    result = load_file("messy_sample.csv", FIXTURE.read_bytes())
    guess = result.column_guesses["zip_code"]
    assert guess.inferred_dtype in ("integer", "float")


def test_dtype_guess_reasonable_defaults():
    result = load_file("messy_sample.csv", FIXTURE.read_bytes())
    assert result.column_guesses["borough"].inferred_dtype == "categorical"
    assert result.column_guesses["signup_date"].inferred_dtype == "datetime"
    assert result.column_guesses["score"].inferred_dtype == "float"


def test_unsupported_extension_raises():
    with pytest.raises(IngestionError):
        load_file("data.txt", b"a,b\n1,2")


def test_corrupt_file_raises_friendly_error():
    with pytest.raises(IngestionError):
        load_file("data.json", b"{not valid json")


def test_encoding_fallback_latin1():
    raw = "col\nCaf\xe9".encode("latin-1")
    result = load_file("data.csv", raw)
    assert result.raw_df["col"].iloc[0] == "Café"


def test_delimiter_sniffing_semicolon():
    raw = b"a;b;c\n1;2;3\n4;5;6\n"
    result = load_file("data.csv", raw)
    assert list(result.raw_df.columns) == ["a", "b", "c"]
    assert len(result.raw_df) == 2


def test_normalize_null_sentinels_case_insensitive():
    df = pd.DataFrame({"x": ["N/A", "n/a", "None", "real value", ""]})
    normalized, counts = normalize_null_sentinels(df)
    assert counts["x"] == 4
    assert normalized["x"].isna().sum() == 4


def test_row_limit_truncates_and_warns(monkeypatch):
    monkeypatch.setattr(ingestion, "MAX_DESIGN_ROWS", 5)
    monkeypatch.setattr(ingestion, "_READ_LIMIT", 6)
    csv_bytes = ("col\n" + "\n".join(str(i) for i in range(10))).encode()
    result = ingestion.load_file("data.csv", csv_bytes)
    assert len(result.raw_df) == 5
    assert any("more than 5 rows" in w for w in result.warnings)


def test_row_limit_not_triggered_when_under_cap(monkeypatch):
    monkeypatch.setattr(ingestion, "MAX_DESIGN_ROWS", 100)
    monkeypatch.setattr(ingestion, "_READ_LIMIT", 101)
    csv_bytes = ("col\n" + "\n".join(str(i) for i in range(10))).encode()
    result = ingestion.load_file("data.csv", csv_bytes)
    assert len(result.raw_df) == 10
    assert result.warnings == []


def test_row_limit_exactly_at_cap_is_not_truncated(monkeypatch):
    """Off-by-one guard: a file with exactly MAX_DESIGN_ROWS rows must not
    be reported as truncated."""
    monkeypatch.setattr(ingestion, "MAX_DESIGN_ROWS", 10)
    monkeypatch.setattr(ingestion, "_READ_LIMIT", 11)
    csv_bytes = ("col\n" + "\n".join(str(i) for i in range(10))).encode()
    result = ingestion.load_file("data.csv", csv_bytes)
    assert len(result.raw_df) == 10
    assert result.warnings == []


def test_xml_xxe_rejected():
    malicious = b"""<?xml version="1.0"?>
<!DOCTYPE root [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>
<root><row><a>&xxe;</a></row></root>"""
    with pytest.raises(IngestionError):
        load_file("data.xml", malicious)


def test_json_with_list_and_dict_cells_loads_and_profiles():
    import json

    from dq_framework.profiling import profile_dataset

    data = json.dumps([{"id": 1, "tags": ["a", "b"]}, {"id": 2, "tags": []}, {"id": 3, "tags": None}]).encode()
    ir = load_file("x.json", data)
    assert ir.raw_df["tags"].iloc[0] == '["a", "b"]'
    profile_dataset(ir.raw_df, {c: "string" for c in ir.raw_df.columns})  # was: TypeError unhashable list
