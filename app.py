"""Streamlit entrypoint. Six steps, state held in st.session_state so
Streamlit's rerun-on-interaction model doesn't recompute expensive steps
needlessly. User upload is the sole way data gets into the tool — there is
no bundled sample dataset and no runtime network call of any kind.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

# Best-effort: disable GX's anonymous usage telemetry before it's imported.
os.environ.setdefault("GX_ANALYTICS_ENABLED", "False")
os.environ.setdefault("GE_USAGE_STATISTICS_ENABLED", "False")

import pandas as pd
import streamlit as st

from dq_framework import expectations, pipeline
from dq_framework.anomalies import consistency
from dq_framework.cleaning import NULL_STRATEGIES, CleaningOptions
from dq_framework.constants import DTYPE_CHOICES
from dq_framework.expectations import EXPECTATION_KINDS, ExpectationSpec
from dq_framework.ingestion import IngestionError, get_extension, list_excel_sheets, load_file
from dq_framework.profiling import profile_dataset
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
        "sql_engine_override": "auto",
        "validation_outcomes": None,
        "cleaning_result": None,
        "profile_after": None,
        "reports": None,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


_init_state()

st.title("Data Quality & Validation Audit Framework")
st.caption(
    "Upload a messy dataset → confirm its schema → review anomalies → tune validation rules → "
    "clean it → download a full audit trail. Runs entirely locally, no data leaves your machine."
)

tabs = st.tabs(["1. Upload", "2. Confirm Schema", "3. Profile & Anomalies", "4. Rules", "5. Clean", "6. Report"])

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
                for k in ("schema_result", "anomaly_outcome", "validation_outcomes", "cleaning_result", "profile_after", "reports"):
                    st.session_state[k] = None
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
            rows.append(
                {
                    "Column": col,
                    "Inferred type": guess.inferred_dtype,
                    "Confidence": round(guess.confidence, 2),
                    "Samples": ", ".join(guess.samples),
                    "Confirmed type": default_type,
                    "Key column?": default_key,
                }
            )
        edit_df = pd.DataFrame(rows)

        edited = st.data_editor(
            edit_df,
            column_config={
                "Confirmed type": st.column_config.SelectboxColumn(options=DTYPE_CHOICES, required=True),
                "Key column?": st.column_config.CheckboxColumn(),
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
            result = confirm_schema(ir.raw_df, column_types, key_columns, dataset_name=dataset_name)
            st.session_state.schema_result = result
            st.session_state.profile_before = _cached_profile(result.confirmed_df, column_types)
            for k in ("anomaly_outcome", "validation_outcomes", "cleaning_result", "profile_after", "reports"):
                st.session_state[k] = None
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
                    previous_schema=previous_schema,
                )
            finally:
                engine.close()
            st.session_state.anomaly_outcome = outcome

        outcome = st.session_state.anomaly_outcome
        if outcome is not None:
            rows = [
                {"Check": r.check_name, "Result": "PASS" if r.passed else "FAIL", "Rows affected": r.affected_row_count, "Summary": r.summary}
                for r in outcome.results
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
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
                params = {}
                if min_value:
                    params["min_value"] = float(min_value)
                if max_value:
                    params["max_value"] = float(max_value)
                if value_set:
                    params["value_set"] = [v.strip() for v in value_set.split(",")]
                if regex:
                    params["regex"] = regex
                st.session_state.custom_expectations.append(ExpectationSpec(kind, column, params))

        if st.session_state.custom_expectations:
            st.write("Custom rules added:")
            for i, spec in enumerate(st.session_state.custom_expectations):
                st.write(f"- `{spec.kind}` on `{spec.column}` {spec.params}")

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
                    previous_schema=previous_schema,
                )
            finally:
                engine.close()
            st.session_state.anomaly_outcome = outcome

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
                {"Expectation": v.expectation_type, "Column": v.column or "", "Result": "PASS" if v.success else "FAIL", "Unexpected %": f"{v.unexpected_percent:.1%}"}
                for v in st.session_state.validation_outcomes
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 5 — Clean
# ---------------------------------------------------------------------------
with tabs[4]:
    st.header("Clean")
    sr = st.session_state.schema_result
    outcome = st.session_state.anomaly_outcome
    if sr is None or outcome is None:
        st.info("Run anomaly detection first.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            drop_exact_duplicates = st.checkbox("Drop exact duplicate rows", value=True)
            drop_key_duplicates = st.checkbox("Drop (rather than flag) duplicate-key rows", value=False)
            apply_categorical_standardization = st.checkbox("Standardize categorical case/whitespace variants", value=True)
        with c2:
            outlier_action = st.radio("Outlier handling", ["flag", "cap", "drop"], horizontal=True)
            null_strategy = st.radio(
                "Null handling",
                NULL_STRATEGIES,
                horizontal=True,
                captions=[
                    "median / mode per column",
                    "numeric columns only",
                    "text columns only",
                    "leave null, no imputation",
                ],
            )

        options = CleaningOptions(
            key_columns=sr.schema.key_columns,
            drop_exact_duplicates=drop_exact_duplicates,
            drop_key_duplicates=drop_key_duplicates,
            apply_categorical_standardization=apply_categorical_standardization,
            outlier_action=outlier_action,
            null_strategy_overrides=(
                {col: null_strategy for col in sr.confirmed_df.columns} if null_strategy != "auto" else {}
            ),
        )

        if st.button("Apply cleaning", type="primary"):
            previous_schema = find_matching_schema(list(sr.confirmed_df.columns), SCHEMAS_DIR)
            if previous_schema and previous_schema.column_types == sr.schema.column_types:
                previous_schema = None
            result = pipeline.run_cleaning(
                sr.confirmed_df,
                sr.schema,
                options,
                referential_flag_indices=outcome.referential_flag_indices,
                consistency_flag_indices=outcome.consistency_flag_indices,
                previous_schema=previous_schema,
            )
            st.session_state.cleaning_result = result
            st.session_state.profile_after = _cached_profile(result.cleaned_df, sr.schema.column_types)
            st.session_state.reports = None
            st.success(f"Cleaning applied: {len(result.log)} transformations.")

        cr = st.session_state.cleaning_result
        if cr is not None:
            before_rows = st.session_state.profile_before.n_rows
            after_rows = len(cr.cleaned_df)
            c1, c2 = st.columns(2)
            c1.metric("Rows before", f"{before_rows:,}")
            c2.metric("Rows after", f"{after_rows:,}", delta=after_rows - before_rows)

            st.subheader("Transformation log")
            log_rows = [
                {"Column": e.column or "(table-level)", "Change": e.change_type, "Before": e.before_summary, "After": e.after_summary, "Why": e.reason}
                for e in cr.log
            ]
            st.dataframe(pd.DataFrame(log_rows), use_container_width=True)

            st.subheader("Cleaned data (first 10 rows)")
            st.dataframe(cr.cleaned_df.head(10))

# ---------------------------------------------------------------------------
# Tab 6 — Report
# ---------------------------------------------------------------------------
with tabs[5]:
    st.header("Report & downloads")
    cr = st.session_state.cleaning_result
    if cr is None:
        st.info("Apply cleaning first.")
    else:
        if st.button("Generate reports", type="primary") or st.session_state.reports is not None:
            if st.session_state.reports is None:
                dataset_name = Path(st.session_state.filename).stem
                st.session_state.reports = pipeline.generate_reports(
                    dataset_name,
                    st.session_state.anomaly_outcome.results,
                    st.session_state.validation_outcomes or [],
                    cr.log,
                    st.session_state.profile_before,
                    cr.cleaned_df,
                    after_profile=st.session_state.profile_after,
                )

            reports = st.session_state.reports
            c1, c2, c3, c4 = st.columns(4)
            c1.download_button("cleaned_dataset.csv", reports["cleaned_dataset.csv"], "cleaned_dataset.csv", "text/csv")
            c2.download_button("changelog.md", reports["changelog.md"], "changelog.md", "text/markdown")
            c3.download_button("dq_report.html", reports["dq_report.html"], "dq_report.html", "text/html")
            c4.download_button(
                "dq_scorecard.xlsx",
                reports["dq_scorecard.xlsx"],
                "dq_scorecard.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            st.subheader("Report preview")
            st.components.v1.html(reports["dq_report.html"].decode("utf-8"), height=800, scrolling=True)
