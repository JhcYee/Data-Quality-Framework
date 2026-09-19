"""Renders the recommended actions as a markdown table: Column | Issue |
Rows affected | Why it was flagged | Recommended action."""

from __future__ import annotations

from ..recommendations import Recommendation


def _escape(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def render_markdown_actions(
    recommendations: list[Recommendation], dataset_name: str = "dataset"
) -> str:
    lines = [f"# Recommended Actions — {dataset_name}", ""]
    if not recommendations:
        lines.append("No data quality issues were found.")
        return "\n".join(lines) + "\n"

    lines.append(
        f"{len(recommendations)} recommended actions, ordered by rows affected. "
        "Nothing in the uploaded data has been changed — these are findings for you to act on."
    )
    lines.append("")
    lines.append("| Column | Issue | Rows affected | Why it was flagged | Recommended action |")
    lines.append("|---|---|---|---|---|")
    for rec in recommendations:
        col = _escape(rec.column) if rec.column else "*(table-level)*"
        rows = f"{rec.affected_rows:,} ({rec.pct_of_rows:.1%})" if rec.affected_rows else "—"
        lines.append(
            f"| {col} | {_escape(rec.issue)} | {rows} | {_escape(rec.finding)} | {_escape(rec.action)} |"
        )

    return "\n".join(lines) + "\n"
