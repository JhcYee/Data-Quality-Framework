"""The cleaning pipeline. Always runs fresh from the post-schema-confirmation
DataFrame passed in — never incrementally chained on top of a previous
cleaning pass — so re-running with different CleaningOptions (e.g. toggling
an accept/reject checkbox off in the UI) genuinely undoes a prior choice
instead of requiring a re-upload.

Transformers run in a fixed order, since each stage's output changes the
next stage's statistics:

  1. Duplicates       4. Nulls
  2. Categorical std.  5. Schema drift coercion
  3. Outliers          6. Referential/consistency/typo flags (flag only, never edited)

Coercion-failure nulls from Schema Confirmation are already ordinary nulls
by the time this module runs (Schema Confirmation resolves coercion before
Cleaning ever starts) — there's no separate "resolve coercion failures" step
here, they just flow through step 4 like any other null.

Every transformation appends one evidence-based TransformationLogEntry — the
`reason` is built from the actual detected numbers, not a generic template,
per the brief's "one sentence explanation of why it was suspected to be a
data quality issue."
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .anomalies.categorical_standardization import find_variant_groups, suggest_canonical_forms
from .anomalies.outliers import compute_outlier_bounds, outlier_mask
from .schema_confirmation import coerce_column, coerce_custom_value

# "auto"    — default per dtype: median (numeric), mode (categorical/string/
#             boolean), flag-only (datetime, never guessed).
# "zero"    — numeric columns filled with 0 instead of the median;
#             categorical/string/datetime columns fall back to "auto" since
#             0 isn't a meaningful fill for them.
# "unknown" — categorical/string columns filled with the constant "Unknown"
#             instead of the mode; numeric/datetime columns fall back to "auto".
# "custom"  — a user-entered value (CleaningOptions.null_custom_value),
#             validated against each column's confirmed dtype via
#             coerce_custom_value; a column whose dtype the value can't be
#             coerced to falls back to "auto" for that column, and the
#             fallback is recorded in the changelog rather than done silently.
# "skip"    — leave nulls as null in every column, no imputation at all.
NULL_STRATEGIES = ["auto", "zero", "unknown", "custom", "skip"]


@dataclass
class CleaningOptions:
    key_columns: list[str] = field(default_factory=list)
    drop_exact_duplicates: bool = True
    drop_key_duplicates: bool = False  # False = flag only, True = drop
    apply_categorical_standardization: bool = True
    outlier_action: str = "flag"  # "flag" | "cap" | "drop"
    null_strategy_overrides: dict[str, str] = field(default_factory=dict)  # col -> one of NULL_STRATEGIES
    null_custom_value: str | None = None  # raw text for the "custom" strategy, validated per column


@dataclass
class TransformationLogEntry:
    column: str | None
    change_type: str
    before_summary: str
    after_summary: str
    reason: str


@dataclass
class CleaningResult:
    cleaned_df: pd.DataFrame
    log: list[TransformationLogEntry]


def clean_dataset(
    df: pd.DataFrame,
    column_types: dict[str, str],
    options: CleaningOptions,
    referential_flag_indices: dict[str, pd.Index] | None = None,
    consistency_flag_indices: dict[str, pd.Index] | None = None,
    typo_flag_indices: dict[str, pd.Index] | None = None,
    previous_schema_types: dict[str, str] | None = None,
) -> CleaningResult:
    working = df.copy()
    log: list[TransformationLogEntry] = []

    # 1. Duplicates -------------------------------------------------------
    if options.drop_exact_duplicates:
        before_n = len(working)
        dup_mask = working.duplicated(keep="first")
        n_dupes = int(dup_mask.sum())
        if n_dupes:
            working = working.loc[~dup_mask]
            log.append(
                TransformationLogEntry(
                    column=None,
                    change_type="drop_exact_duplicates",
                    before_summary=f"{before_n} rows",
                    after_summary=f"{len(working)} rows",
                    reason=f"{n_dupes} rows were exact duplicates of an earlier row and were removed.",
                )
            )

    if options.key_columns:
        key_cols = [c for c in options.key_columns if c in working.columns]
        if key_cols:
            key_dup_mask = working.duplicated(subset=key_cols, keep="first")
            n_key_dupes = int(key_dup_mask.sum())
            if n_key_dupes:
                key_desc = ", ".join(key_cols)
                if options.drop_key_duplicates:
                    before_n = len(working)
                    working = working.loc[~key_dup_mask]
                    log.append(
                        TransformationLogEntry(
                            column=key_desc,
                            change_type="drop_key_duplicates",
                            before_summary=f"{before_n} rows",
                            after_summary=f"{len(working)} rows",
                            reason=f"{n_key_dupes} rows duplicated the confirmed key ({key_desc}) and were removed.",
                        )
                    )
                else:
                    flag_col = "_duplicate_key"
                    working[flag_col] = False
                    working.loc[key_dup_mask, flag_col] = True
                    log.append(
                        TransformationLogEntry(
                            column=key_desc,
                            change_type="flag_key_duplicates",
                            before_summary=f"{n_key_dupes} unflagged duplicate-key rows",
                            after_summary=f"{n_key_dupes} rows flagged in '{flag_col}'",
                            reason=f"{n_key_dupes} rows duplicated the confirmed key ({key_desc}); flagged rather than dropped.",
                        )
                    )

    # 2. Categorical standardization ---------------------------------------
    if options.apply_categorical_standardization:
        for col, dtype in column_types.items():
            if dtype != "categorical" or col not in working.columns:
                continue
            variant_groups = find_variant_groups(working[col])
            if not variant_groups:
                continue
            mapping = suggest_canonical_forms(working[col], variant_groups)
            affected = int(working[col].astype(str).isin(mapping.keys()).sum())
            working[col] = working[col].apply(lambda v: mapping.get(str(v), v) if pd.notna(v) else v)
            example_group = next(iter(variant_groups.values()))
            log.append(
                TransformationLogEntry(
                    column=col,
                    change_type="standardize_categorical",
                    before_summary=f"{len(variant_groups)} case/whitespace variant groups",
                    after_summary=f"{affected} values rewritten to a canonical form",
                    reason=(
                        f"{len(variant_groups)} groups of values differed only by case/whitespace "
                        f"(e.g. {example_group}); standardized to the most frequent variant in each group."
                    ),
                )
            )

    # 3. Outliers -----------------------------------------------------------
    for col, dtype in column_types.items():
        if dtype not in ("integer", "float") or col not in working.columns:
            continue
        bounds = compute_outlier_bounds(working[col])
        if bounds is None:
            continue
        mask = outlier_mask(working[col], bounds)
        n = int(mask.sum())
        if not n:
            continue
        lo, hi = bounds
        if options.outlier_action == "drop":
            before_n = len(working)
            working = working.loc[~mask]
            log.append(
                TransformationLogEntry(
                    column=col,
                    change_type="drop_outliers",
                    before_summary=f"{before_n} rows",
                    after_summary=f"{len(working)} rows",
                    reason=f"{n} values fell outside the IQR-based range [{lo:.2f}, {hi:.2f}] and were dropped.",
                )
            )
        elif options.outlier_action == "cap":
            working.loc[mask & (working[col].astype("float64") < lo), col] = lo
            working.loc[mask & (working[col].astype("float64") > hi), col] = hi
            log.append(
                TransformationLogEntry(
                    column=col,
                    change_type="cap_outliers",
                    before_summary=f"{n} values outside [{lo:.2f}, {hi:.2f}]",
                    after_summary=f"{n} values capped to [{lo:.2f}, {hi:.2f}]",
                    reason=f"{n} values fell outside the IQR-based range [{lo:.2f}, {hi:.2f}] and were capped (winsorized) to the boundary.",
                )
            )
        else:  # flag
            flag_col = "_is_outlier"
            if flag_col not in working.columns:
                working[flag_col] = False
            working.loc[mask, flag_col] = True
            log.append(
                TransformationLogEntry(
                    column=col,
                    change_type="flag_outliers",
                    before_summary=f"{n} unflagged outliers",
                    after_summary=f"{n} rows flagged in '{flag_col}'",
                    reason=f"{n} values fell outside the IQR-based range [{lo:.2f}, {hi:.2f}]; flagged rather than altered.",
                )
            )

    # 4. Nulls ----------------------------------------------------------------
    for col, dtype in column_types.items():
        if col not in working.columns:
            continue
        strategy = options.null_strategy_overrides.get(col)
        null_mask = working[col].isna()
        n_null = int(null_mask.sum())
        if not n_null:
            continue
        if strategy == "skip":
            continue

        pct = n_null / len(working) if len(working) else 0.0

        if dtype == "datetime":
            # Dates get a conservative default (flag, never guessed) since
            # there's no sensible "median"/"mode" for a date — but an
            # explicit, valid custom value is the user's own choice, not a
            # guess, so it's honored if given.
            datetime_fallback_note = None
            if strategy == "custom" and options.null_custom_value is not None:
                custom_value, ok = coerce_custom_value(options.null_custom_value, dtype)
                if ok:
                    working[col] = working[col].fillna(custom_value)
                    log.append(
                        TransformationLogEntry(
                            column=col,
                            change_type="impute_nulls",
                            before_summary=f"{n_null} nulls ({pct:.0%})",
                            after_summary=f"filled with custom value {custom_value!r}",
                            reason=f"{pct:.0%} of values were null; imputed with custom value {custom_value!r}.",
                        )
                    )
                    continue
                datetime_fallback_note = (
                    f" (entered value {options.null_custom_value!r} isn't a valid date, "
                    "so nulls were flagged instead)"
                )

            flag_col = f"_{col}_was_null"
            working[flag_col] = null_mask
            log.append(
                TransformationLogEntry(
                    column=col,
                    change_type="flag_null_datetime",
                    before_summary=f"{n_null} nulls ({pct:.0%})",
                    after_summary=f"{n_null} rows flagged in '{flag_col}', values left as null",
                    reason=f"{pct:.0%} of values were null; dates aren't safely imputable, so nulls were flagged instead of guessed."
                    + (datetime_fallback_note or ""),
                )
            )
            continue

        if dtype in ("integer", "float"):
            median = working[col].median()
            fill_value = int(round(median)) if dtype == "integer" and pd.notna(median) else median
            method = "median"
        elif dtype == "boolean":
            mode = working[col].mode(dropna=True)
            fill_value = mode.iloc[0] if len(mode) else False
            method = "mode"
        else:  # string, categorical
            mode = working[col].mode(dropna=True)
            fill_value = mode.iloc[0] if len(mode) else "Unknown"
            method = "mode" if len(mode) else "constant 'Unknown'"

        custom_fallback_note = None
        if strategy == "unknown":
            fill_value = "Unknown" if dtype in ("string", "categorical") else fill_value
            method = "constant 'Unknown'" if dtype in ("string", "categorical") else method
        elif strategy == "zero":
            fill_value = 0 if dtype in ("integer", "float") else fill_value
            method = "constant 0" if dtype in ("integer", "float") else method
        elif strategy == "custom" and options.null_custom_value is not None:
            custom_value, ok = coerce_custom_value(options.null_custom_value, dtype)
            if ok:
                fill_value = custom_value
                method = "custom value"  # the value itself is appended once, below
            else:
                custom_fallback_note = (
                    f" (entered value {options.null_custom_value!r} isn't valid for dtype "
                    f"'{dtype}', so this column fell back to auto)"
                )

        if pd.isna(fill_value):
            # Every value in the column was null — nothing to impute from.
            log.append(
                TransformationLogEntry(
                    column=col,
                    change_type="null_unresolved",
                    before_summary=f"{n_null} nulls ({pct:.0%})",
                    after_summary="left null — no non-null values to derive a fill from",
                    reason=f"{pct:.0%} of values were null and the entire column was empty, so no {method} could be computed."
                    + (custom_fallback_note or ""),
                )
            )
            continue

        working[col] = working[col].fillna(fill_value)
        log.append(
            TransformationLogEntry(
                column=col,
                change_type="impute_nulls",
                before_summary=f"{n_null} nulls ({pct:.0%})",
                after_summary=f"filled with {method} ({fill_value!r})",
                reason=f"{pct:.0%} of values were null; imputed with column {method}."
                + (custom_fallback_note or ""),
            )
        )

    # 5. Schema drift coercion ------------------------------------------------
    if previous_schema_types:
        for col, prev_dtype in previous_schema_types.items():
            cur_dtype = column_types.get(col)
            if col not in working.columns or cur_dtype is None or cur_dtype == prev_dtype:
                continue
            coerced, n_failures = coerce_column(working[col], prev_dtype)
            if n_failures == 0:
                working[col] = coerced
                log.append(
                    TransformationLogEntry(
                        column=col,
                        change_type="coerce_schema_drift",
                        before_summary=f"confirmed as '{cur_dtype}' this upload",
                        after_summary=f"coerced back to '{prev_dtype}' (previously confirmed type)",
                        reason=f"Column type drifted from the previously confirmed '{prev_dtype}' to '{cur_dtype}'; coerced back since every value converted cleanly.",
                    )
                )
            else:
                log.append(
                    TransformationLogEntry(
                        column=col,
                        change_type="schema_drift_unresolved",
                        before_summary=f"confirmed as '{cur_dtype}' this upload",
                        after_summary=f"left as '{cur_dtype}' — {n_failures} values wouldn't convert to '{prev_dtype}'",
                        reason=f"Column type drifted from the previously confirmed '{prev_dtype}' to '{cur_dtype}'; left unresolved because {n_failures} values wouldn't convert back safely.",
                    )
                )

    # 6. Referential / consistency / typo flags — flag only, never drop/edit -
    for check_name, idx in (referential_flag_indices or {}).items():
        idx = idx.intersection(working.index)
        if len(idx) == 0:
            continue
        flag_col = f"_referential_issue"
        if flag_col not in working.columns:
            working[flag_col] = False
        working.loc[idx, flag_col] = True
        log.append(
            TransformationLogEntry(
                column=None,
                change_type="flag_referential_break",
                before_summary=f"{len(idx)} unflagged orphan rows ({check_name})",
                after_summary=f"{len(idx)} rows flagged in '{flag_col}'",
                reason=f"{len(idx)} rows failed the referential-integrity check '{check_name}'; flagged, not dropped.",
            )
        )

    for check_name, idx in (consistency_flag_indices or {}).items():
        idx = idx.intersection(working.index)
        if len(idx) == 0:
            continue
        flag_col = "_consistency_issue"
        if flag_col not in working.columns:
            working[flag_col] = False
        working.loc[idx, flag_col] = True
        log.append(
            TransformationLogEntry(
                column=None,
                change_type="flag_consistency_violation",
                before_summary=f"{len(idx)} unflagged rows ({check_name})",
                after_summary=f"{len(idx)} rows flagged in '{flag_col}'",
                reason=f"{len(idx)} rows failed the cross-column consistency check '{check_name}'; flagged, not dropped.",
            )
        )

    for check_name, idx in (typo_flag_indices or {}).items():
        idx = idx.intersection(working.index)
        if len(idx) == 0:
            continue
        flag_col = "_typo_suspected"
        if flag_col not in working.columns:
            working[flag_col] = False
        working.loc[idx, flag_col] = True
        log.append(
            TransformationLogEntry(
                column=check_name.split(":", 1)[-1] if ":" in check_name else None,
                change_type="flag_typo_suspected",
                before_summary=f"{len(idx)} unflagged rows ({check_name})",
                after_summary=f"{len(idx)} rows flagged in '{flag_col}'",
                reason=(
                    f"{len(idx)} rows contain a value suspected to be a typo of a more common "
                    f"value in the same column ('{check_name}'); flagged, not changed — fuzzy "
                    "matches are too uncertain to auto-correct."
                ),
            )
        )

    return CleaningResult(cleaned_df=working, log=log)
