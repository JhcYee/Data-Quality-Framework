"""Groups the flat list of anomaly results into one section per kind of check
(missing values, duplicates, outliers, ...), so a reader can go straight to the
kind of problem they care about instead of scanning one long mixed table.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .anomalies.types import AnomalyResult
from .constants import HIGH_NULL_THRESHOLD, IQR_MULTIPLIER, MIN_DISTINCT_FOR_OUTLIER_CHECK


@dataclass
class SectionRow:
    target: str  # what was checked: a column, a key, a column pair, or the whole table
    passed: bool
    affected_rows: int
    summary: str


@dataclass
class Section:
    kind: str
    title: str
    description: str
    rows: list[SectionRow] = field(default_factory=list)

    @property
    def n_failing(self) -> int:
        return sum(1 for r in self.rows if not r.passed)


# (kind, title, what the check looks for) in the order sections are shown.
_SECTION_META: list[tuple[str, str, str]] = [
    (
        "nulls",
        "Missing values",
        "Any column with empty cells is flagged. Columns you marked 'Nulls expected?' in Confirm "
        f"Schema are only flagged above {HIGH_NULL_THRESHOLD:.0%} missing.",
    ),
    (
        "duplicates",
        "Duplicates",
        "Rows that are identical in every column, and repeated values in the key column(s) you "
        "marked in Confirm Schema.",
    ),
    (
        "outliers",
        "Outliers",
        f"Numeric values beyond {IQR_MULTIPLIER}x the interquartile range from the quartiles. "
        f"Columns with fewer than {MIN_DISTINCT_FOR_OUTLIER_CHECK} distinct values are skipped.",
    ),
    (
        "categorical_standardization",
        "Inconsistent casing / whitespace",
        "Categorical values that differ only by capitalization or stray spaces.",
    ),
    (
        "typos",
        "Possible typos",
        "Categorical values a couple of edits away from a more common value. Confirm or dismiss "
        "each one in the Rules tab.",
    ),
    (
        "schema_drift",
        "Schema drift",
        "Confirmed columns and types compared with the last saved schema for this dataset.",
    ),
    (
        "referential",
        "Referential integrity",
        "Values with no match in the reference file. Runs once a reference file and column pair are set in the Rules tab.",
    ),
    (
        "consistency",
        "Cross-column consistency",
        "Rows where an end value comes before its start value. Runs once column pairs are chosen in the Rules tab.",
    ),
]


def _target_label(kind: str, detail: str) -> str:
    if kind == "duplicates":
        if detail == "exact_rows":
            return "All columns (exact duplicate rows)"
        if detail.startswith("key[") and detail.endswith("]"):
            return f"Key: {detail[4:-1]}"
    if kind == "referential":
        return detail.replace("->", " → ")
    if kind == "consistency":
        return detail.replace(">=", " ≥ ")
    if kind == "schema_drift":
        return "(whole table)"
    return detail or "(whole table)"


def group_anomaly_results(results: list[AnomalyResult]) -> list[Section]:
    """One Section per kind of check that has results; failures listed first
    within each section, then by rows affected (most first)."""
    known = {kind: Section(kind, title, desc) for kind, title, desc in _SECTION_META}
    extra: dict[str, Section] = {}

    for res in results:
        kind, _, detail = res.check_name.partition(":")
        section = known.get(kind)
        if section is None:
            section = extra.setdefault(kind, Section(kind, kind.replace("_", " ").capitalize(), ""))
        section.rows.append(
            SectionRow(_target_label(kind, detail), res.passed, res.affected_row_count, res.summary)
        )

    ordered = [s for s in known.values() if s.rows] + list(extra.values())
    for section in ordered:
        section.rows.sort(key=lambda r: (r.passed, -r.affected_rows))  # stable: ties keep column order
    return ordered
