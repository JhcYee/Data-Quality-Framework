"""Turns anomaly findings into recommended actions. This tool audits data,
it doesn't rewrite it: whether to impute a null, merge two categories, or
drop a duplicate depends on domain knowledge the tool can't have, and the
mechanical fix itself is trivial once you know what's wrong. So the value is
in the finding (what, where, how many) plus one sentence on what to do next.
"""

from __future__ import annotations

from dataclasses import dataclass

from .anomalies.types import AnomalyResult


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


def build_recommendations(
    anomaly_results: list[AnomalyResult],
    n_rows: int,
    coercion_failure_counts: dict[str, int] | None = None,
    sentinel_null_counts: dict[str, int] | None = None,
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

    recs.sort(key=lambda r: (-r.affected_rows, r.issue, r.column or ""))
    return recs
