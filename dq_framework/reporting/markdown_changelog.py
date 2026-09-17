"""Renders the TransformationLog as a markdown table: Column | Change | Why
(one sentence, evidence-based per log entry — see cleaning.py)."""

from __future__ import annotations

from ..cleaning import TransformationLogEntry


def _escape(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def render_markdown_changelog(
    log: list[TransformationLogEntry], dataset_name: str = "dataset"
) -> str:
    lines = [f"# Data Cleaning Changelog — {dataset_name}", ""]
    if not log:
        lines.append("No changes were made — the uploaded data required no cleaning.")
        return "\n".join(lines) + "\n"

    lines.append(f"{len(log)} changes were made during cleaning.")
    lines.append("")
    lines.append("| Column | Change | Why |")
    lines.append("|---|---|---|")
    for entry in log:
        col = _escape(entry.column) if entry.column else "*(table-level)*"
        change = _escape(entry.change_type.replace("_", " "))
        reason = _escape(entry.reason)
        lines.append(f"| {col} | {change} | {reason} |")

    return "\n".join(lines) + "\n"
