from dq_framework.anomalies.types import AnomalyResult
from dq_framework.recommendations import build_recommendations
from dq_framework.reporting.markdown_actions import render_markdown_actions


def _fail(check_name: str, n: int, summary: str = "evidence") -> AnomalyResult:
    return AnomalyResult(check_name=check_name, passed=False, summary=summary, affected_row_count=n)


def test_passing_checks_produce_no_recommendations():
    ok = AnomalyResult("nulls:a", True, "0.0% null", 0)
    assert build_recommendations([ok], n_rows=100) == []


def test_each_failing_check_kind_maps_to_an_issue_and_column():
    results = [
        _fail("nulls:age", 11),
        _fail("duplicates:exact_rows", 8),
        _fail("duplicates:key[id]", 16),
        _fail("outliers:score", 3),
        _fail("categorical_standardization:borough", 124),
        _fail("typos:borough", 1),
        _fail("referential:borough_code->code", 38),
        _fail("consistency:close_date>=signup_date", 31),
        _fail("schema_drift", 0),
    ]
    recs = {r.issue: r for r in build_recommendations(results, n_rows=204)}

    assert recs["Missing values"].column == "age"
    assert recs["Duplicate rows"].column is None
    assert recs["Duplicate keys"].column == "id"
    assert recs["Outliers"].column == "score"
    assert recs["Inconsistent casing/whitespace"].column == "borough"
    assert recs["Possible typos"].column == "borough"
    assert recs["Referential integrity break"].column == "borough_code"
    assert recs["Cross-column inconsistency"].column == "close_date, signup_date"
    assert recs["Schema drift"].column is None
    assert all(r.action for r in recs.values())


def test_sorted_by_rows_affected_descending():
    recs = build_recommendations([_fail("nulls:a", 5), _fail("nulls:b", 50), _fail("nulls:c", 20)], n_rows=100)
    assert [r.column for r in recs] == ["b", "c", "a"]


def test_percentage_of_rows_computed():
    (rec,) = build_recommendations([_fail("nulls:a", 25)], n_rows=100)
    assert rec.pct_of_rows == 0.25


def test_evidence_carried_through_from_the_finding():
    (rec,) = build_recommendations([_fail("nulls:a", 3, summary="3.0% null (3 of 100 rows)")], n_rows=100)
    assert rec.finding == "3.0% null (3 of 100 rows)"


def test_coercion_failures_and_placeholders_become_recommendations():
    recs = build_recommendations(
        [], n_rows=100, coercion_failure_counts={"zip": 4}, sentinel_null_counts={"age": 11}
    )
    by_issue = {r.issue: r for r in recs}
    assert by_issue["Values not matching the confirmed type"].affected_rows == 4
    assert by_issue["Placeholder values treated as null"].column == "age"


def test_markdown_lists_each_recommendation():
    recs = build_recommendations([_fail("nulls:age", 11, summary="5.4% null | with a pipe")], n_rows=204)
    md = render_markdown_actions(recs, "sample")
    assert "# Recommended Actions — sample" in md
    assert "| age | Missing values | 11 (5.4%) |" in md
    assert "with a pipe" in md and "\\|" in md  # pipes escaped so the table doesn't break
    assert "Nothing in the uploaded data has been changed" in md


def test_markdown_when_nothing_found():
    assert "No data quality issues were found" in render_markdown_actions([], "sample")
