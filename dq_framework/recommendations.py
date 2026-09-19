"""Turns anomaly findings into recommended actions. This tool audits data,
it doesn't rewrite it: whether to impute a null, merge two categories, or
drop a duplicate depends on domain knowledge the tool can't have, and the
mechanical fix itself is trivial once you know what's wrong. So the value is
in the finding (what, where, how many) plus one sentence on what to do next.
"""

from __future__ import annotations

from dataclasses import dataclass

from .anomalies.types import AnomalyResult
from .expectations import ValidationOutcome


@dataclass
class Recommendation:
    column: str | None
    issue: str
    affected_rows: int
    pct_of_rows: float
    finding: str  # the evidence: why this was flagged
    action: str  # one sentence: what to do about it


def _column_from_check(kind: str, detail: str) -> str | None:
    if kind == "duplicates":
        if detail.startswith("key[") and detail.endswith("]"):
            return detail[4:-1]
        return None
    if kind == "referential":
        return detail.split("->", 1)[0]
    if kind == "consistency":
        end_col, _, start_col = detail.partition(">=")
        return f"{end_col}, {start_col}" if start_col else detail
    if kind == "schema_drift":
        return None
    return detail


def _issue_and_action(kind: str, detail: str, column: str | None) -> tuple[str, str]:
    if kind == "nulls":
        return (
            "Missing values",
            f"Decide how to handle the missing values in '{column}' (impute, fill with a default, "
            "or drop the rows) based on what a blank actually means for that column.",
        )
    if kind == "duplicates" and detail == "exact_rows":
        return (
            "Duplicate rows",
            "Remove the repeats, keeping one copy of each, after confirming they're true "
            "duplicates and not separate events that happen to look identical.",
        )
    if kind == "duplicates":
        return (
            "Duplicate keys",
            f"Decide which record is authoritative for each repeated ({column}) value, or "
            "whether that column really is a unique identifier.",
        )
    if kind == "outliers":
        return (
            "Outliers",
            f"Check these '{column}' values against the source; correct data-entry errors and "
            "keep genuine extremes.",
        )
    if kind == "categorical_standardization":
        return (
            "Inconsistent casing/whitespace",
            f"Standardize the spelling of '{column}' so each category is written exactly one way.",
        )
    if kind == "typos":
        return (
            "Possible typos",
            f"Review the flagged '{column}' values; correct the confirmed misspellings to the "
            "common spelling and leave genuine categories alone.",
        )
    if kind == "referential":
        return (
            "Referential integrity break",
            f"Add the missing entries to the reference data, or correct/remove the '{column}' "
            "values that have no match.",
        )
    if kind == "consistency":
        return (
            "Cross-column inconsistency",
            "Investigate the rows where the end value comes before the start value; one of the "
            "two is likely wrong.",
        )
    if kind == "schema_drift":
        return (
            "Schema drift",
            "Confirm the column or type changes are intentional and update anything downstream "
            "that depends on the old schema.",
        )
    return (kind.replace("_", " ").capitalize(), "Review this finding.")


def _fmt_number(x) -> str:
    return "unbounded" if x is None else f"{float(x):,.6g}"


def _fmt_range(params: dict) -> str:
    return f"[{_fmt_number(params.get('min_value'))}, {_fmt_number(params.get('max_value'))}]"


def _fmt_value_set(values, limit: int = 6) -> str:
    shown = ", ".join(repr(v) for v in list(values)[:limit])
    return shown + (", ..." if len(values) > limit else "")


def _rule_issue_finding_action(v: ValidationOutcome, n_rows: int) -> tuple[str, str, str]:
    """(issue, finding, action) for a failed validation rule. The rule type
    comes from GX's expectation name, e.g. expect_column_values_to_be_unique."""
    col, n, pct = v.column, v.unexpected_count, v.unexpected_percent
    kind = v.expectation_type.removeprefix("expect_")
    if kind == "column_values_to_not_be_null":
        return (
            "Missing values",
            f"Rule 'not null' on '{col}' failed: {n} of {n_rows} rows are empty.",
            f"Decide how to handle the missing values in '{col}' (impute, fill with a default, "
            "or drop the rows) based on what a blank actually means for that column.",
        )
    if kind == "column_values_to_be_unique":
        return (
            "Values that should be unique repeat",
            f"Rule 'unique' on '{col}' failed: {n} rows ({pct:.1%}) share their '{col}' value with another row.",
            f"Decide which record is authoritative for each repeated '{col}' value, or drop the "
            "uniqueness rule if repeats are legitimate for this column.",
        )
    if kind == "column_values_to_be_between":
        return (
            "Values outside the expected range",
            f"Rule 'between {_fmt_range(v.params)}' on '{col}' failed: {n} values ({pct:.1%}) fall outside it.",
            f"Check the out-of-range '{col}' values against the source; correct data-entry errors, "
            "or widen the rule if they're legitimate.",
        )
    if kind == "column_values_to_be_in_set":
        allowed = v.params.get("value_set") or []
        return (
            "Values outside the allowed set",
            f"Rule 'in set' on '{col}' failed: {n} values ({pct:.1%}) are not one of {_fmt_value_set(allowed)}.",
            f"Correct the unexpected '{col}' values, or add the legitimate ones to the rule's allowed list.",
        )
    if kind == "column_values_to_match_regex":
        return (
            "Values not matching the expected pattern",
            f"Rule 'regex {v.params.get('regex', '')!r}' on '{col}' failed: {n} values ({pct:.1%}) don't match.",
            f"Fix the '{col}' values that don't follow the pattern, or correct the pattern if it's too strict.",
        )
    if kind == "table_row_count_to_be_between":
        return (
            "Row count outside the expected range",
            f"Rule 'row count between {_fmt_range(v.params)}' failed: the table has {n_rows:,} rows.",
            "Check whether rows were lost or duplicated in the extract, or update the expected row count.",
        )
    return (
        v.expectation_type.replace("_", " ").capitalize(),
        f"Validation rule '{v.expectation_type}' failed" + (f" on '{col}'." if col else "."),
        "Review this rule and the values it flagged.",
    )


def _already_reported_by_anomaly_check(v: ValidationOutcome, anomaly_recs: list[Recommendation]) -> bool:
    """True when an anomaly check already reports the very same problem, so the
    rule's failure would only repeat a row that's already in the list.

    The baseline not-null, in-range and key-unique rules are built on the same
    definitions as the null, outlier and key-duplicate checks; matching on
    column (plus row count where the definitions could differ, e.g. a custom
    range rule) tells apart "same finding" from "a different rule that also
    fails".
    """
    kind = v.expectation_type.removeprefix("expect_")
    same_col = [r for r in anomaly_recs if r.column == v.column]
    if kind == "column_values_to_not_be_null":
        return any(r.issue == "Missing values" and r.affected_rows == v.unexpected_count for r in same_col)
    if kind == "column_values_to_be_between":
        return any(r.issue == "Outliers" and r.affected_rows == v.unexpected_count for r in same_col)
    if kind == "column_values_to_be_unique":
        return any(r.issue == "Duplicate keys" for r in same_col)
    return False


def build_recommendations(
    anomaly_results: list[AnomalyResult],
    n_rows: int,
    coercion_failure_counts: dict[str, int] | None = None,
    sentinel_null_counts: dict[str, int] | None = None,
    validation_outcomes: list[ValidationOutcome] | None = None,
) -> list[Recommendation]:
    recs: list[Recommendation] = []

    for res in anomaly_results:
        if res.passed:
            continue
        kind, _, detail = res.check_name.partition(":")
        column = _column_from_check(kind, detail)
        issue, action = _issue_and_action(kind, detail, column)
        recs.append(
            Recommendation(
                column=column,
                issue=issue,
                affected_rows=res.affected_row_count,
                pct_of_rows=(res.affected_row_count / n_rows) if n_rows else 0.0,
                finding=res.summary,
                action=action,
            )
        )

    for col, n in (coercion_failure_counts or {}).items():
        recs.append(
            Recommendation(
                column=col,
                issue="Values not matching the confirmed type",
                affected_rows=n,
                pct_of_rows=(n / n_rows) if n_rows else 0.0,
                finding=f"{n} values in '{col}' couldn't be converted to the confirmed type and became null.",
                action=f"Fix those source values, or reconsider whether '{col}' really is the type you confirmed.",
            )
        )

    for col, n in (sentinel_null_counts or {}).items():
        recs.append(
            Recommendation(
                column=col,
                issue="Placeholder values treated as null",
                affected_rows=n,
                pct_of_rows=(n / n_rows) if n_rows else 0.0,
                finding=f"{n} placeholder strings (e.g. 'N/A', 'None', '-') in '{col}' were read as missing.",
                action="Confirm these placeholders really mean missing, and standardize how missing "
                "values are recorded at the source.",
            )
        )

    anomaly_recs = list(recs)
    for v in validation_outcomes or []:
        if v.success or _already_reported_by_anomaly_check(v, anomaly_recs):
            continue
        issue, finding, action = _rule_issue_finding_action(v, n_rows)
        recs.append(
            Recommendation(
                column=v.column,
                issue=issue,
                affected_rows=v.unexpected_count,
                pct_of_rows=(v.unexpected_count / n_rows) if n_rows else 0.0,
                finding=finding,
                action=action,
            )
        )

    recs.sort(key=lambda r: (-r.affected_rows, r.issue, r.column or ""))
    return recs
