from dq_framework.anomalies.types import AnomalyResult
from dq_framework.anomaly_sections import group_anomaly_results


def _res(name: str, passed: bool, n: int = 0) -> AnomalyResult:
    return AnomalyResult(name, passed, "summary", n)


def test_results_are_grouped_by_kind_in_a_fixed_order():
    results = [
        _res("outliers:score", True),
        _res("nulls:age", False, 11),
        _res("duplicates:exact_rows", True),
        _res("nulls:name", True),
    ]
    sections = group_anomaly_results(results)
    assert [s.title for s in sections] == ["Missing values", "Duplicates", "Outliers"]
    assert [len(s.rows) for s in sections] == [2, 1, 1]


def test_failures_are_listed_first_then_by_rows_affected():
    results = [_res("nulls:a", True), _res("nulls:b", False, 3), _res("nulls:c", False, 30)]
    (section,) = group_anomaly_results(results)
    assert [r.target for r in section.rows] == ["c", "b", "a"]
    assert section.n_failing == 2


def test_targets_are_readable():
    results = [
        _res("duplicates:exact_rows", False, 2),
        _res("duplicates:key[id, email]", False, 2),
        _res("referential:code->ref_code", False, 1),
        _res("consistency:close>=open", False, 1),
        _res("schema_drift", True),
    ]
    targets = {r.target for s in group_anomaly_results(results) for r in s.rows}
    assert targets == {
        "All columns (exact duplicate rows)",
        "Key: id, email",
        "code → ref_code",
        "close ≥ open",
        "(whole table)",
    }


def test_sections_without_results_are_omitted_and_unknown_kinds_still_show():
    sections = group_anomaly_results([_res("mystery:x", False, 1)])
    assert [s.title for s in sections] == ["Mystery"]
    assert group_anomaly_results([]) == []
