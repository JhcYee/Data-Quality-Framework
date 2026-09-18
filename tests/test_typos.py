import pandas as pd

from dq_framework.anomalies.typos import detect_typos, find_typo_groups, levenshtein_distance


def test_levenshtein_distance_basic_cases():
    assert levenshtein_distance("manhattan", "manhattan") == 0
    assert levenshtein_distance("manhattan", "manhattn") == 1  # one deletion
    assert levenshtein_distance("manhattan", "manhattann") == 1  # one insertion
    assert levenshtein_distance("cat", "dog") == 3
    assert levenshtein_distance("", "abc") == 3


def test_find_typo_groups_flags_rare_misspelling():
    # "manhattn" (1 occurrence) is a likely typo of "manhattan" (10 occurrences).
    series = pd.Series(["Manhattan"] * 10 + ["Manhattn"])
    groups = find_typo_groups(series)
    assert groups == {"manhattan": ["manhattn"]}


def test_find_typo_groups_ignores_case_whitespace_variants():
    """Pure case/whitespace variants normalize to the identical string, so
    they never even reach the distance check — that's
    categorical_standardization's job, not typo detection's."""
    series = pd.Series(["Manhattan"] * 5 + ["MANHATTAN"] * 3 + ["manhattan "] * 2)
    groups = find_typo_groups(series)
    assert groups == {}


def test_find_typo_groups_does_not_flag_equally_common_values():
    # Two genuinely different categories that happen to be similar strings,
    # both common — neither should be treated as a typo of the other.
    series = pd.Series(["cat"] * 5 + ["cot"] * 5)
    groups = find_typo_groups(series)
    assert groups == {}


def test_find_typo_groups_respects_max_distance():
    series = pd.Series(["hello"] * 10 + ["xyzabc"])  # distance way above 2
    groups = find_typo_groups(series, max_distance=2)
    assert groups == {}


def test_detect_typos_only_runs_on_categorical_columns():
    df = pd.DataFrame({"free_text": ["Manhattan"] * 10 + ["Manhattn"]})
    results = detect_typos(df, {"free_text": "string"})
    assert results == {}


def test_detect_typos_flags_affected_rows_not_columns_of_other_dtypes():
    df = pd.DataFrame(
        {
            "borough": ["Manhattan"] * 10 + ["Manhattn"],
            "score": list(range(11)),
        }
    )
    results = detect_typos(df, {"borough": "categorical", "score": "float"})
    assert "borough" in results
    assert "score" not in results
    result = results["borough"]
    assert not result.passed
    assert result.affected_row_count == 1
    assert result.details is not None
    assert list(result.details.index) == [10]  # the "Manhattn" row, by original index


def test_detect_typos_respects_dismissed_variants():
    """A human-reviewed false positive (e.g. 'poor' vs 'good', both at edit
    distance 2 despite being genuinely different categories) must disappear
    entirely once dismissed — not just get a lower confidence."""
    df = pd.DataFrame({"borough": ["Manhattan"] * 10 + ["Manhattn"]})
    results = detect_typos(df, {"borough": "categorical"}, dismissed_variants={"borough": {"manhattn"}})
    assert results == {}


def test_detect_typos_dismissal_is_per_column():
    df = pd.DataFrame(
        {
            "borough": ["Manhattan"] * 10 + ["Manhattn"],
            "status": ["good"] * 7 + ["poor"] * 4,
        }
    )
    # Dismiss the "status" false positive only, not "borough" — each
    # column's review is independent.
    results = detect_typos(df, {"borough": "categorical", "status": "categorical"}, dismissed_variants={"status": {"poor"}})
    assert "borough" in results
    assert "status" not in results
