"""Schema drift (pandas) — compares this upload's confirmed schema to a
previously *saved confirmed* schema for a matching dataset (see
schema_confirmation.find_matching_schema). On a dataset's first-ever
confirmed upload there's nothing to diff against, so this reports "no prior
schema on file" rather than a false negative.
"""

from __future__ import annotations

from ..schema_confirmation import ConfirmedSchema
from .types import AnomalyResult


def detect_schema_drift(
    current: ConfirmedSchema, previous: ConfirmedSchema | None
) -> AnomalyResult:
    if previous is None:
        return AnomalyResult(
            check_name="schema_drift",
            passed=True,
            summary="No prior confirmed schema on file for this dataset — nothing to compare against yet.",
            affected_row_count=0,
        )

    cur_cols = set(current.column_types)
    prev_cols = set(previous.column_types)
    added = sorted(cur_cols - prev_cols)
    removed = sorted(prev_cols - cur_cols)
    type_changes = {
        c: (previous.column_types[c], current.column_types[c])
        for c in cur_cols & prev_cols
        if previous.column_types[c] != current.column_types[c]
    }

    drifted = bool(added or removed or type_changes)
    parts = []
    if added:
        parts.append(f"added columns: {added}")
    if removed:
        parts.append(f"removed columns: {removed}")
    if type_changes:
        changes_str = ", ".join(f"{c}: {old}->{new}" for c, (old, new) in type_changes.items())
        parts.append(f"type changes: {changes_str}")
    summary = "; ".join(parts) if drifted else "No drift from the last confirmed schema for this dataset."

    return AnomalyResult(
        check_name="schema_drift",
        passed=not drifted,
        summary=summary,
        affected_row_count=0,
    )
