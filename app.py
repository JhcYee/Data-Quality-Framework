"""Streamlit entrypoint. Six steps, state held in st.session_state so
Streamlit's rerun-on-interaction model doesn't recompute expensive steps
needlessly. User upload is the sole way data gets into the tool — there is
no bundled sample dataset and no runtime network call of any kind.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

# Best-effort: disable GX's anonymous usage telemetry before it's imported.
os.environ.setdefault("GX_ANALYTICS_ENABLED", "False")
os.environ.setdefault("GE_USAGE_STATISTICS_ENABLED", "False")

import pandas as pd
import streamlit as st

from dq_framework import expectations, pipeline
from dq_framework.anomalies import consistency
from dq_framework.anomaly_sections import group_anomaly_results
from dq_framework.anomalies.typos import find_typo_groups
from dq_framework.constants import DTYPE_CHOICES
from dq_framework.expectations import EXPECTATION_KINDS, ExpectationSpec
from dq_framework.ingestion import IngestionError, get_extension, list_excel_sheets, load_file
from dq_framework.profiling import profile_dataset
from dq_framework.recommendations import build_recommendations
from dq_framework.schema_confirmation import (
    ConfirmedSchema,
    confirm_schema,
    find_matching_schema,
    save_confirmed_schema,
)
from dq_framework.sql_engine import ENGINE_CHOICES, create_engine, select_engine_kind

SCHEMAS_DIR = Path(__file__).parent / "schemas"

st.set_page_config(page_title="Data Quality Audit Framework", layout="wide")


@st.cache_resource(show_spinner="Loading validation engine...")
def _gx_module():
    return expectations.get_gx()


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


@st.cache_data(show_spinner=False)
def _cached_profile(df: pd.DataFrame, column_types: dict) -> "object":
    return profile_dataset(df, column_types)


# Same red/green as the styled HTML report, so a result reads the same way
# whether you're looking at it live in the app or in the downloaded report.
_PASS_COLOR = "#4ade80"
_FAIL_COLOR = "#f87171"


def _style_result_column(df: pd.DataFrame, column: str = "Result"):
    def _color(val):
        if val in ("FAIL", "ERROR"):
            return f"color: {_FAIL_COLOR}; font-weight: 600"
        if val == "PASS":
            return f"color: {_PASS_COLOR}; font-weight: 600"
        return ""

    return df.style.map(_color, subset=[column])


def _init_state():
    defaults = {
        "ingestion_result": None,
        "filename": None,
        "reference_ingestion_result": None,
        "reference_filename": None,
        "schema_result": None,
        "profile_before": None,
        "anomaly_outcome": None,
        "custom_expectations": [],
        "consistency_pairs_selected": [],
        "referential_pairs_selected": [],
        "dismissed_typo_variants": {},
        "sql_engine_override": "auto",
        "validation_outcomes": None,
        "reports": None,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


_init_state()

st.title("Data Quality & Validation Audit Framework")
st.caption(
    "Upload a messy dataset → confirm its schema → review anomalies → tune validation rules → "
    "review the recommended actions → download a full audit trail. Runs entirely locally, no data leaves your machine."
)

tabs = st.tabs(["1. Upload", "2. Confirm Schema", "3. Profile & Anomalies", "4. Rules", "5. Recommended Actions", "6. Report"])

# ---------------------------------------------------------------------------
# Tab 1 — Upload
# ---------------------------------------------------------------------------
with tabs[0]:
    st.header("Upload a dataset")
    uploaded = st.file_uploader("Dataset (CSV, XLSX, XLS, JSON, or XML)", type=["csv", "xlsx", "xls", "json", "xml"])

    if uploaded is not None:
        file_bytes = uploaded.getvalue()
        ext = get_extension(uploaded.name)
        sheet_name = None
        if ext in ("xlsx", "xls", "xlsm"):
            try:
                sheets = list_excel_sheets(file_bytes, ext)
            except IngestionError as e:
                st.error(str(e))
                sheets = []
            if len(sheets) > 1:
                sheet_name = st.selectbox("This workbook has multiple sheets — pick one", sheets)
            elif sheets:
                sheet_name = sheets[0]

        if st.button("Load file", type="primary"):
            try:
                result = load_file(uploaded.name, file_bytes, sheet_name=sheet_name)
            except IngestionError as e:
                st.error(f"Could not load file: {e}")
            else:
                st.session_state.ingestion_result = result
                st.session_state.filename = uploaded.name
                # Uploading a new main file invalidates everything downstream.
                for k in ("schema_result", "anomaly_outcome", "validation_outcomes", "reports"):
                    st.session_state[k] = None
                st.session_state.dismissed_typo_variants = {}
                # Custom rules name columns of the previous file — they don't carry over.
                st.session_state.custom_expectations = []
                st.success(f"Loaded {uploaded.name}: {len(result.raw_df):,} rows × {len(result.raw_df.columns)} columns")

    st.divider()
    st.subheader("Optional: reference/lookup file for referential-integrity checks")
    ref_uploaded = st.file_uploader(
        "Reference file (CSV, XLSX, XLS, JSON, or XML)", type=["csv", "xlsx", "xls", "json", "xml"], key="ref_uploader"
    )
    if ref_uploaded is not None and st.button("Load reference file"):
        try:
            ref_result = load_file(ref_uploaded.name, ref_uploaded.getvalue())
        except IngestionError as e:
            st.error(f"Could not load reference file: {e}")
        else:
            st.session_state.reference_ingestion_result = ref_result
            st.session_state.reference_filename = ref_uploaded.name
            st.success(f"Loaded reference file {ref_uploaded.name}: {len(ref_result.raw_df):,} rows")

    if st.session_state.ingestion_result is not None:
        ir = st.session_state.ingestion_result
        for warning in ir.warnings:
            st.warning(warning)
        st.subheader("Preview (first 10 rows)")
        st.dataframe(ir.raw_df.head(10))
        if ir.sentinel_null_counts:
            st.caption(
                "Null-sentinel strings normalized to true null: "
                + ", ".join(f"{c} ({n})" for c, n in ir.sentinel_null_counts.items())
            )

# ---------------------------------------------------------------------------
# Tab 2 — Confirm Schema
# ---------------------------------------------------------------------------
with tabs[1]:
    st.header("Confirm schema")
    ir = st.session_state.ingestion_result
    if ir is None:
        st.info("Upload a file in the Upload tab first.")
    else:
        dataset_name = Path(st.session_state.filename).stem
        saved_schema = find_matching_schema(list(ir.raw_df.columns), SCHEMAS_DIR)
        if saved_schema:
            st.caption(f"A previously confirmed schema for a similar dataset was found and pre-filled below.")

        rows = []
        for col in ir.raw_df.columns:
            guess = ir.column_guesses[col]
            default_type = (saved_schema.column_types.get(col) if saved_schema else None) or guess.inferred_dtype
            default_key = col in (saved_schema.key_columns if saved_schema else [])
            default_nulls_expected = col in (saved_schema.nulls_expected_columns if saved_schema else [])
            rows.append(
                {
                    "Column": col,
                    "Inferred type": guess.inferred_dtype,
                    "Confidence": round(guess.confidence, 2),
                    "Samples": ", ".join(guess.samples),
                    "Confirmed type": default_type,
                    "Key column?": default_key,
                    "Nulls expected?": default_nulls_expected,
                }
            )
        edit_df = pd.DataFrame(rows)

        st.caption(
            "By default, **any** null in a column is flagged — check 'Nulls expected?' for columns "
            "where some missing data is normal (e.g. an optional field), which raises the bar to "
            "only flagging if more than 20% of the column is null."
        )
        edited = st.data_editor(
            edit_df,
            column_config={
                "Confirmed type": st.column_config.SelectboxColumn(options=DTYPE_CHOICES, required=True),
                "Key column?": st.column_config.CheckboxColumn(),
                "Nulls expected?": st.column_config.CheckboxColumn(),
            },
            disabled=["Column", "Inferred type", "Confidence", "Samples"],
            hide_index=True,
            use_container_width=True,
            key="schema_editor",
        )

        save_for_next_time = st.checkbox("Save this schema for next time", value=True)

        if st.button("Confirm Schema", type="primary"):
            column_types = dict(zip(edited["Column"], edited["Confirmed type"]))
            key_columns = edited.loc[edited["Key column?"], "Column"].tolist()
            nulls_expected_columns = edited.loc[edited["Nulls expected?"], "Column"].tolist()
            result = confirm_schema(
                ir.raw_df,
                column_types,
                key_columns,
                dataset_name=dataset_name,
                nulls_expected_columns=nulls_expected_columns,
            )
            st.session_state.schema_result = result
            st.session_state.profile_before = _cached_profile(result.confirmed_df, column_types)
            for k in ("anomaly_outcome", "validation_outcomes", "reports"):
                st.session_state[k] = None
            st.session_state.dismissed_typo_variants = {}
            if save_for_next_time:
                save_confirmed_schema(result.schema, SCHEMAS_DIR)
            st.success("Schema confirmed.")
            if result.coercion_failure_counts:
                st.warning(
                    "Some values didn't convert to the confirmed type and became null: "
                    + ", ".join(f"{c} ({n})" for c, n in result.coercion_failure_counts.items())
                )

    if st.session_state.schema_result is not None:
        st.subheader("Confirmed data (first 10 rows)")
        st.dataframe(st.session_state.schema_result.confirmed_df.head(10))

# ---------------------------------------------------------------------------
# Tab 3 — Profile & Anomalies
# ---------------------------------------------------------------------------
with tabs[2]:
    st.header("Profile & anomalies")
    sr = st.session_state.schema_result
    if sr is None:
        st.info("Confirm the schema in the previous tab first.")
    else:
        profile = st.session_state.profile_before
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{profile.n_rows:,}")
        c2.metric("Columns", profile.n_columns)
        c3.metric("Key column(s)", ", ".join(sr.schema.key_columns) or "none set")

        size_bytes = sr.confirmed_df.memory_usage(deep=True).sum()
        engine_kind = select_engine_kind(profile.n_rows, size_bytes, st.session_state.sql_engine_override)
        st.caption(f"SQL engine: **{engine_kind}** (override in the Rules tab)")

        if st.button("Run anomaly detection", type="primary"):
            engine = create_engine(engine_kind)
            try:
                previous_schema = find_matching_schema(list(sr.confirmed_df.columns), SCHEMAS_DIR)
                if previous_schema and previous_schema.column_types == sr.schema.column_types:
                    previous_schema = None  # identical — nothing drifted
                outcome = pipeline.run_anomaly_detection(
                    sr.confirmed_df,
                    sr.schema,
                    profile,
                    engine,
                    dismissed_typo_variants=st.session_state.dismissed_typo_variants,
                    previous_schema=previous_schema,
                )
            finally:
                engine.close()
            st.session_state.anomaly_outcome = outcome
            st.session_state.reports = None

        outcome = st.session_state.anomaly_outcome
        if outcome is not None:
            st.caption(
                "One section per kind of check. Sections with a failure open automatically; "
                "failing rows are listed first."
            )
            for section in group_anomaly_results(outcome.results):
                n_failing = section.n_failing
                status = f":red[{n_failing} failing]" if n_failing else ":green[all passing]"
                with st.expander(
                    f"**{section.title}** — {status} · {len(section.rows)} checked", expanded=n_failing > 0
                ):
                    if section.description:
                        st.caption(section.description)
                    st.dataframe(
                        _style_result_column(
                            pd.DataFrame(
                                [
                                    {
                                        "Checked": r.target,
                                        "Result": "PASS" if r.passed else "FAIL",
                                        "Rows affected": r.affected_rows,
                                        "Summary": r.summary,
                                    }
                                    for r in section.rows
                                ]
                            )
                        ),
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Checked": st.column_config.Column(width=260),
                            "Result": st.column_config.Column(width=80),
                            "Rows affected": st.column_config.Column(width=120),
                            "Summary": st.column_config.Column(width=900),
                        },
                    )
            st.caption(
                "Referential integrity and cross-column consistency checks run from the Rules tab, "
                "once a reference file / column pairs are configured."
            )

# ---------------------------------------------------------------------------
# Tab 4 — Rules
# ---------------------------------------------------------------------------
with tabs[3]:
    st.header("Validation rules")
    sr = st.session_state.schema_result
    if sr is None or st.session_state.anomaly_outcome is None:
        st.info("Run anomaly detection in the Profile & Anomalies tab first.")
    else:
        profile = st.session_state.profile_before

        st.session_state.sql_engine_override = st.selectbox(
            "SQL engine", ENGINE_CHOICES, index=ENGINE_CHOICES.index(st.session_state.sql_engine_override)
        )

        st.subheader("Great Expectations rules")
        st.caption(
            "A baseline suite is auto-generated from the data with deliberate tolerance "
            "(IQR-based ranges, high-frequency categories only) — it's meant to still catch outliers, "
            "not just replay what's already in the file. Add your own rules below."
        )
        with st.form("add_expectation_form"):
            cols = list(sr.confirmed_df.columns)
            kind = st.selectbox("Rule type", EXPECTATION_KINDS)
            column = st.selectbox("Column", cols) if kind != "row_count_between" else None
            min_value = st.text_input("Min value (for between / row_count_between)", "")
            max_value = st.text_input("Max value (for between / row_count_between)", "")
            value_set = st.text_input("Allowed values, comma-separated (for in_set)", "")
            regex = st.text_input("Regex (for regex)", "")
            if st.form_submit_button("Add rule"):
                params, problem = {}, None
                try:
                    if min_value.strip():
                        params["min_value"] = float(min_value)
                    if max_value.strip():
                        params["max_value"] = float(max_value)
                except ValueError:
                    problem = "Min and max must be numbers."
                if value_set.strip():
                    params["value_set"] = [v.strip() for v in value_set.split(",")]
                if regex.strip():
                    params["regex"] = regex
                if problem is None:
                    if kind == "between" and not ({"min_value", "max_value"} & params.keys()):
                        problem = "A 'between' rule needs a min value, a max value, or both."
                    elif kind == "row_count_between" and not ({"min_value", "max_value"} & params.keys()):
                        problem = "A 'row_count_between' rule needs a min value, a max value, or both."
                    elif kind == "in_set" and not params.get("value_set"):
                        problem = "An 'in_set' rule needs at least one allowed value."
                    elif kind == "regex":
                        try:
                            re.compile(params.get("regex", ""))
                            if not params.get("regex"):
                                problem = "A 'regex' rule needs a pattern."
                        except re.error as e:
                            problem = f"That regex isn't valid: {e}"
                if problem:
                    st.error(problem)
                else:
                    st.session_state.custom_expectations.append(ExpectationSpec(kind, column, params))

        if st.session_state.custom_expectations:
            st.write("Custom rules added:")
            for i, spec in enumerate(st.session_state.custom_expectations):
                rule_col, remove_col = st.columns([6, 1])
                rule_col.write(f"- `{spec.kind}` on `{spec.column}` {spec.params}")
                if remove_col.button("Remove", key=f"remove_custom_rule_{i}"):
                    st.session_state.custom_expectations.pop(i)
                    # Results from a run that included this rule are now stale.
                    st.session_state.validation_outcomes = None
                    st.session_state.reports = None
                    st.rerun()

        st.subheader("Cross-column consistency checks (SQL)")
        suggested = consistency.suggest_pairs(sr.schema.column_types)
        pair_labels = [f"{a} <= {b}" for a, b in suggested]
        chosen = st.multiselect("Suggested pairs to check", pair_labels, default=pair_labels)
        st.session_state.consistency_pairs_selected = [suggested[pair_labels.index(c)] for c in chosen]

        st.subheader("Referential integrity")
        ref_ir = st.session_state.reference_ingestion_result
        if ref_ir is None:
            st.caption("Upload a reference/lookup file in the Upload tab to enable this check.")
        else:
            main_col = st.selectbox("Main file column", list(sr.confirmed_df.columns), key="ref_main_col")
            ref_col = st.selectbox("Reference file column", list(ref_ir.raw_df.columns), key="ref_lookup_col")
            if st.button("Add referential-integrity pair"):
                st.session_state.referential_pairs_selected.append((main_col, ref_col))
            if st.session_state.referential_pairs_selected:
                st.write("Pairs checked: " + ", ".join(f"{m} → {r}" for m, r in st.session_state.referential_pairs_selected))

        st.subheader("Review possible typos")
        st.caption(
            "Fuzzy matching flags short similar-looking values as possible typos, but some pairs "
            "are genuinely different categories that just happen to be a couple of edits apart "
            "(e.g. 'poor' vs 'good', 'active' vs 'inactive'). Uncheck anything that's intentional — "
            "it won't be flagged, and the value itself is never changed either way."
        )
        any_typo_groups = False
        for col, dt in sr.schema.column_types.items():
            if dt != "categorical":
                continue
            groups = find_typo_groups(sr.confirmed_df[col])
            if not groups:
                continue
            counts = sr.confirmed_df[col].dropna().astype(str).str.strip().str.lower().value_counts()
            dismissed_for_col = st.session_state.dismissed_typo_variants.setdefault(col, set())
            for canonical, variants in groups.items():
                for variant in variants:
                    any_typo_groups = True
                    is_typo = st.checkbox(
                        f"**{col}**: {variant!r} ({counts.get(variant, 0)} occurrence(s)) looks like a "
                        f"typo of {canonical!r} ({counts.get(canonical, 0)} occurrence(s))",
                        value=variant not in dismissed_for_col,
                        key=f"typo_review_{col}_{canonical}_{variant}",
                    )
                    if is_typo:
                        dismissed_for_col.discard(variant)
                    else:
                        dismissed_for_col.add(variant)
        if not any_typo_groups:
            st.caption("No possible typos detected in any categorical column.")

        if st.button("Run validation & consistency checks", type="primary"):
            engine_kind = select_engine_kind(
                profile.n_rows, sr.confirmed_df.memory_usage(deep=True).sum(), st.session_state.sql_engine_override
            )
            engine = create_engine(engine_kind)
            try:
                reference_df = None
                if ref_ir is not None and st.session_state.referential_pairs_selected:
                    reference_df = ref_ir.raw_df
                    # Coerce the reference join column(s) to the main file's confirmed type.
                    from dq_framework.schema_confirmation import coerce_column

                    reference_df = reference_df.copy()
                    for main_col, ref_col in st.session_state.referential_pairs_selected:
                        dtype = sr.schema.column_types.get(main_col, "string")
                        coerced, _ = coerce_column(reference_df[ref_col], dtype)
                        reference_df[ref_col] = coerced

                previous_schema = find_matching_schema(list(sr.confirmed_df.columns), SCHEMAS_DIR)
                if previous_schema and previous_schema.column_types == sr.schema.column_types:
                    previous_schema = None
                outcome = pipeline.run_anomaly_detection(
                    sr.confirmed_df,
                    sr.schema,
                    profile,
                    engine,
                    reference_df=reference_df,
                    referential_pairs=st.session_state.referential_pairs_selected,
                    consistency_pairs=st.session_state.consistency_pairs_selected,
                    dismissed_typo_variants=st.session_state.dismissed_typo_variants,
                    previous_schema=previous_schema,
                )
            finally:
                engine.close()
            st.session_state.anomaly_outcome = outcome
            st.session_state.reports = None

            gx_mod = _gx_module()
            st.session_state.validation_outcomes = pipeline.run_validation(
                sr.confirmed_df, sr.schema, profile, st.session_state.custom_expectations, gx_module=gx_mod
            )
            # Force a fresh rerun so the Profile & Anomalies tab (rendered
            # earlier in script order, on the same pass that just updated
            # anomaly_outcome) picks up the merged referential/consistency
            # results immediately, instead of only on the next interaction.
            st.rerun()

        if st.session_state.validation_outcomes is not None:
            rows = [
                {"Expectation": v.expectation_type, "Column": v.column or "", "Source": v.source, "Result": v.status, "Rows failing": "—" if v.error else f"{v.unexpected_count:,} ({v.unexpected_percent:.1%})"}
                for v in st.session_state.validation_outcomes
            ]
            st.caption(
                "Rows failing = rows that break the rule. For a 'unique' rule that means every row "
                "whose value also appears on another row, so a value repeated twice counts both rows."
            )
            st.dataframe(_style_result_column(pd.DataFrame(rows)), use_container_width=True)
            for v in st.session_state.validation_outcomes:
                if v.error:
                    st.warning(
                        f"Rule `{v.expectation_type.removeprefix('expect_')}`"
                        + (f" on `{v.column}`" if v.column else "")
                        + f" could not run: {v.error}"
                    )

# ---------------------------------------------------------------------------
# Tab 5 — Recommended Actions
# ---------------------------------------------------------------------------
def _current_recommendations():
    sr = st.session_state.schema_result
    outcome = st.session_state.anomaly_outcome
    ir = st.session_state.ingestion_result
    return build_recommendations(
        outcome.results,
        sr.confirmed_df.shape[0],
        coercion_failure_counts=sr.coercion_failure_counts,
        sentinel_null_counts=ir.sentinel_null_counts if ir else None,
        validation_outcomes=st.session_state.validation_outcomes,
    )


with tabs[4]:
    st.header("Recommended actions")
    sr = st.session_state.schema_result
    outcome = st.session_state.anomaly_outcome
    if sr is None or outcome is None:
        st.info("Run anomaly detection first.")
    else:
        recs = _current_recommendations()
        st.caption(
            "This tool audits your data — it never changes it. Each finding below says what was "
            "found, how many rows it touches, and what to do about it; the fix itself is up to you, "
            "since it usually depends on what the data means. Ordered by rows affected."
        )
        if st.session_state.validation_outcomes is None:
            st.info(
                "Validation rules haven't been run yet. Run them in the Rules tab and any failed "
                "rules will be added to this list."
            )
        if not recs:
            st.success("No data quality issues were found.")
        else:
            st.metric("Recommended actions", len(recs))
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Column": r.column or "(table-level)",
                            "Issue": r.issue,
                            "Rows affected": r.affected_rows,
                            "% of rows": f"{r.pct_of_rows:.1%}",
                            "Why it was flagged": r.finding,
                            "Recommended action": r.action,
                        }
                        for r in recs
                    ]
                ),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Column": st.column_config.Column(width=160),
                    "Issue": st.column_config.Column(width=300),
                    "Why it was flagged": st.column_config.Column(width=700),
                    "Recommended action": st.column_config.Column(width=900),
                },
            )

# ---------------------------------------------------------------------------
# Tab 6 — Report
# ---------------------------------------------------------------------------
with tabs[5]:
    st.header("Report & downloads")
    if st.session_state.schema_result is None or st.session_state.anomaly_outcome is None:
        st.info("Run anomaly detection first.")
    else:
        if st.button("Generate reports", type="primary") or st.session_state.reports is not None:
            if st.session_state.reports is None:
                dataset_name = Path(st.session_state.filename).stem
                st.session_state.reports = pipeline.generate_reports(
                    dataset_name,
                    st.session_state.anomaly_outcome.results,
                    st.session_state.validation_outcomes or [],
                    _current_recommendations(),
                    st.session_state.profile_before,
                )

            reports = st.session_state.reports
            c1, c2, c3 = st.columns(3)
            c1.download_button("recommended_actions.md", reports["recommended_actions.md"], "recommended_actions.md", "text/markdown")
            c2.download_button("dq_report.html", reports["dq_report.html"], "dq_report.html", "text/html")
            c3.download_button(
                "dq_scorecard.xlsx",
                reports["dq_scorecard.xlsx"],
                "dq_scorecard.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            st.subheader("Report preview")
            st.components.v1.html(reports["dq_report.html"].decode("utf-8"), height=800, scrolling=True)
