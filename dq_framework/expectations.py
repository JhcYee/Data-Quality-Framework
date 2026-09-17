"""Great Expectations integration. Runs entirely in an ephemeral, in-memory
context — nothing is written to disk, nothing persists between uploads.

Import cost: `import great_expectations` is genuinely slow. This module
exposes `get_gx()` as a plain function so the caller (app.py) can wrap *just
the import* in `st.cache_resource` — that's the only thing safe to cache at
that shared, cross-session scope. The actual ExpectationSuite, its
expectations, and validation results are built fresh per upload and must
live in the caller's own session state, never in a resource cache, or one
user's data/rules could leak into another concurrent user's session on
shared hosting.

Auto-generated baseline expectations use deliberate tolerance (the same
IQR_MULTIPLIER as anomalies/outliers.py, and a rare-category cutoff for
in_set) rather than the raw observed min/max/value-set — a baseline fit
exactly to the data it validates would trivially pass 100% of its own rules
and the report would show nothing.

Cross-column consistency and referential-integrity checks are *not* folded
into the GX suite — they're SQL-executed (anomalies/consistency.py,
anomalies/referential.py) since GX's multi-column/cross-table expectation
API is one of the areas most likely to have shifted across versions. The
Rules UI presents them as a separate section from the GX suite for exactly
that reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .anomalies.outliers import compute_outlier_bounds
from .constants import HIGH_NULL_THRESHOLD, RARE_CATEGORY_FREQUENCY_THRESHOLD
from .profiling import DatasetProfile

EXPECTATION_KINDS = ["not_null", "unique", "between", "in_set", "regex", "row_count_between"]


def get_gx():
    """The expensive part — import once per process, cache at the caller."""
    import great_expectations as gx

    return gx


def new_ephemeral_context(gx_module=None):
    gx_module = gx_module or get_gx()
    return gx_module.get_context(mode="ephemeral")


@dataclass
class ExpectationSpec:
    kind: str
    column: str | None
    params: dict = field(default_factory=dict)


@dataclass
class ValidationOutcome:
    expectation_type: str
    column: str | None
    success: bool
    unexpected_count: int
    unexpected_percent: float
    summary: str


def build_baseline_suite(
    context,
    df: pd.DataFrame,
    profile: DatasetProfile,
    column_types: dict[str, str],
    key_columns: list[str] | None = None,
    suite_name: str = "baseline_suite",
    gx_module=None,
):
    gx_module = gx_module or get_gx()
    suite = context.suites.add(gx_module.ExpectationSuite(name=suite_name))
    specs: list[ExpectationSpec] = []
    key_columns = key_columns or []

    for col, dtype in column_types.items():
        cp = profile.columns[col]

        if cp.null_pct < HIGH_NULL_THRESHOLD:
            suite.add_expectation(gx_module.expectations.ExpectColumnValuesToNotBeNull(column=col))
            specs.append(ExpectationSpec("not_null", col))

        if dtype in ("integer", "float"):
            bounds = compute_outlier_bounds(df[col])
            if bounds is not None:
                lo, hi = bounds
                suite.add_expectation(
                    gx_module.expectations.ExpectColumnValuesToBeBetween(
                        column=col, min_value=lo, max_value=hi
                    )
                )
                specs.append(ExpectationSpec("between", col, {"min_value": lo, "max_value": hi}))

        if dtype == "categorical":
            non_null = df[col].dropna().astype(str)
            if len(non_null):
                freq = non_null.value_counts(normalize=True)
                allowed = freq[freq >= RARE_CATEGORY_FREQUENCY_THRESHOLD].index.tolist()
                if allowed and len(allowed) < non_null.nunique():
                    suite.add_expectation(
                        gx_module.expectations.ExpectColumnValuesToBeInSet(column=col, value_set=allowed)
                    )
                    specs.append(ExpectationSpec("in_set", col, {"value_set": allowed}))

    # Single-column key uniqueness. Composite-key uniqueness is already
    # covered by the SQL-based duplicate detector, so it isn't duplicated
    # here (GX's single-column ExpectColumnValuesToBeUnique can't express it).
    if len(key_columns) == 1:
        col = key_columns[0]
        suite.add_expectation(gx_module.expectations.ExpectColumnValuesToBeUnique(column=col))
        specs.append(ExpectationSpec("unique", col))

    return suite, specs


def add_manual_expectation(suite, kind: str, column: str | None, params: dict, gx_module=None):
    gx_module = gx_module or get_gx()
    exp = gx_module.expectations
    if kind == "not_null":
        expectation = exp.ExpectColumnValuesToNotBeNull(column=column)
    elif kind == "unique":
        expectation = exp.ExpectColumnValuesToBeUnique(column=column)
    elif kind == "between":
        expectation = exp.ExpectColumnValuesToBeBetween(
            column=column, min_value=params.get("min_value"), max_value=params.get("max_value")
        )
    elif kind == "in_set":
        expectation = exp.ExpectColumnValuesToBeInSet(column=column, value_set=params.get("value_set", []))
    elif kind == "regex":
        expectation = exp.ExpectColumnValuesToMatchRegex(column=column, regex=params.get("regex", ""))
    elif kind == "row_count_between":
        expectation = exp.ExpectTableRowCountToBeBetween(
            min_value=params.get("min_value"), max_value=params.get("max_value")
        )
    else:
        raise ValueError(f"Unknown expectation kind '{kind}', expected one of {EXPECTATION_KINDS}")
    suite.add_expectation(expectation)
    return ExpectationSpec(kind, column, params)


def validate_suite(context, df: pd.DataFrame, suite, gx_module=None) -> "object":
    gx_module = gx_module or get_gx()
    data_source = context.data_sources.add_pandas(name="uploaded")
    data_asset = data_source.add_dataframe_asset(name="main")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("main_batch")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})
    return batch.validate(suite)


def _get_expectation_type(cfg) -> str:
    return getattr(cfg, "type", None) or getattr(cfg, "expectation_type", None) or cfg.__class__.__name__


def _get_kwargs(cfg) -> dict:
    kwargs = getattr(cfg, "kwargs", None)
    if kwargs:
        return dict(kwargs)
    return {k: v for k, v in vars(cfg).items() if not k.startswith("_")}


def summarize_validation(result) -> list[ValidationOutcome]:
    """Defensively parses GX's result object — its internal schema has
    shifted across versions historically, so a per-result parse failure
    degrades to an "unknown" entry instead of crashing the whole report.
    """
    outcomes = []
    for r in result.results:
        try:
            cfg = r.expectation_config
            exp_type = _get_expectation_type(cfg)
            column = _get_kwargs(cfg).get("column")
            res_dict = dict(r.result) if r.result else {}
            unexpected_count = int(res_dict.get("unexpected_count") or 0)
            element_count = int(res_dict.get("element_count") or 0)
            unexpected_pct = (unexpected_count / element_count) if element_count else 0.0
            success = bool(r.success)
        except Exception:  # noqa: BLE001 - defensive parse, never crash the report
            exp_type, column, unexpected_count, unexpected_pct, success = "unknown", None, 0, 0.0, False

        status = "PASS" if success else f"FAIL ({unexpected_count} unexpected, {unexpected_pct:.1%})"
        summary = exp_type + (f" on '{column}'" if column else "") + f": {status}"
        outcomes.append(
            ValidationOutcome(exp_type, column, success, unexpected_count, unexpected_pct, summary)
        )
    return outcomes
