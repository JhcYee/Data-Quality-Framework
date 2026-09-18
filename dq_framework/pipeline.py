"""Orchestrates the full run: ingestion -> schema confirmation -> profiling
-> anomaly detection -> validation rules -> cleaning -> reporting. Used by
both app.py (step by step, driven by the UI) and the tests (end to end).

A stable `_dq_row_id` column (the DataFrame's own index) is added before any
table is loaded into the SQL engine, so a SQL query's result rows can be
mapped back to the original pandas index afterward — SQL results don't
otherwise preserve it. `duplicates.py`'s GROUP BY queries explicitly list
their own columns and never select `_dq_row_id`, so its presence doesn't
affect duplicate-row grouping.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import expectations
from .anomalies import categorical_standardization, consistency, duplicates, nulls, outliers, referential, schema_drift, typos
from .anomalies.types import AnomalyResult
from .cleaning import CleaningOptions, CleaningResult, TransformationLogEntry, clean_dataset
from .expectations import ExpectationSpec, ValidationOutcome
from .profiling import DatasetProfile, profile_dataset
from .reporting.excel_scorecard import build_excel_scorecard
from .reporting.html_report import render_html_report
from .reporting.markdown_changelog import render_markdown_changelog
from .schema_confirmation import ConfirmedSchema

ROW_ID_COL = "_dq_row_id"


@dataclass
class AnomalyDetectionOutcome:
    results: list[AnomalyResult] = field(default_factory=list)
    referential_flag_indices: dict[str, pd.Index] = field(default_factory=dict)
    consistency_flag_indices: dict[str, pd.Index] = field(default_factory=dict)
    typo_flag_indices: dict[str, pd.Index] = field(default_factory=dict)


def _load_with_row_id(engine, df: pd.DataFrame, table_name: str) -> None:
    tagged = df.copy()
    tagged[ROW_ID_COL] = tagged.index
    engine.load(tagged, table_name)


def run_anomaly_detection(
    df: pd.DataFrame,
    schema: ConfirmedSchema,
    profile: DatasetProfile,
    engine,
    table_name: str = "main",
    reference_df: pd.DataFrame | None = None,
    reference_table_name: str = "reference",
    referential_pairs: list[tuple[str, str]] | None = None,
    consistency_pairs: list[tuple[str, str]] | None = None,
    previous_schema: ConfirmedSchema | None = None,
) -> AnomalyDetectionOutcome:
    outcome = AnomalyDetectionOutcome()
    _load_with_row_id(engine, df, table_name)

    outcome.results.extend(nulls.detect_nulls(profile))

    outcome.results.append(duplicates.detect_exact_duplicates(engine, table_name, list(df.columns)))
    key_dupe = duplicates.detect_key_duplicates(engine, table_name, schema.key_columns)
    if key_dupe is not None:
        outcome.results.append(key_dupe)

    outcome.results.extend(outliers.detect_outliers(df, schema.column_types).values())
    outcome.results.append(schema_drift.detect_schema_drift(schema, previous_schema))
    outcome.results.extend(
        categorical_standardization.detect_categorical_variants(df, schema.column_types).values()
    )

    for res in typos.detect_typos(df, schema.column_types).values():
        outcome.results.append(res)
        if res.details is not None and not res.details.empty:
            # Pure pandas, index-preserving — no SQL round-trip here, so
            # unlike the referential/consistency checks below there's no
            # need for the _dq_row_id indirection to recover row identity.
            outcome.typo_flag_indices[res.check_name] = res.details.index

    if reference_df is not None and referential_pairs:
        engine.load(reference_df, reference_table_name)
        for main_col, ref_col in referential_pairs:
            res = referential.detect_referential_integrity(
                engine, table_name, main_col, reference_table_name, ref_col
            )
            outcome.results.append(res)
            if res.details is not None and not res.details.empty and ROW_ID_COL in res.details.columns:
                outcome.referential_flag_indices[res.check_name] = pd.Index(res.details[ROW_ID_COL])

    for start_col, end_col in consistency_pairs or []:
        res = consistency.detect_consistency(engine, table_name, start_col, end_col)
        outcome.results.append(res)
        if res.details is not None and not res.details.empty and ROW_ID_COL in res.details.columns:
            outcome.consistency_flag_indices[res.check_name] = pd.Index(res.details[ROW_ID_COL])

    return outcome


def run_validation(
    df: pd.DataFrame,
    schema: ConfirmedSchema,
    profile: DatasetProfile,
    custom_expectations: list[ExpectationSpec] | None = None,
    gx_module=None,
) -> list[ValidationOutcome]:
    gx_module = gx_module or expectations.get_gx()
    context = expectations.new_ephemeral_context(gx_module)
    suite, _ = expectations.build_baseline_suite(
        context, df, profile, schema.column_types, schema.key_columns, gx_module=gx_module
    )
    for spec in custom_expectations or []:
        expectations.add_manual_expectation(suite, spec.kind, spec.column, spec.params, gx_module=gx_module)
    result = expectations.validate_suite(context, df, suite, gx_module=gx_module)
    return expectations.summarize_validation(result)


def run_cleaning(
    df: pd.DataFrame,
    schema: ConfirmedSchema,
    options: CleaningOptions,
    referential_flag_indices: dict[str, pd.Index] | None = None,
    consistency_flag_indices: dict[str, pd.Index] | None = None,
    typo_flag_indices: dict[str, pd.Index] | None = None,
    previous_schema: ConfirmedSchema | None = None,
) -> CleaningResult:
    return clean_dataset(
        df,
        schema.column_types,
        options,
        referential_flag_indices=referential_flag_indices,
        consistency_flag_indices=consistency_flag_indices,
        typo_flag_indices=typo_flag_indices,
        previous_schema_types=(previous_schema.column_types if previous_schema else None),
    )


def generate_reports(
    dataset_name: str,
    anomaly_results: list[AnomalyResult],
    validation_outcomes: list[ValidationOutcome],
    transformation_log: list[TransformationLogEntry],
    before_profile: DatasetProfile,
    cleaned_df: pd.DataFrame,
    after_profile: DatasetProfile | None = None,
) -> dict[str, bytes]:
    after_profile = after_profile or profile_dataset(
        cleaned_df, {c: before_profile.columns[c].dtype for c in cleaned_df.columns if c in before_profile.columns}
    )

    csv_buf = io.StringIO()
    cleaned_df.to_csv(csv_buf, index=False)

    changelog = render_markdown_changelog(transformation_log, dataset_name)

    html_report = render_html_report(
        dataset_name,
        anomaly_results,
        validation_outcomes,
        transformation_log,
        before_profile,
        after_profile,
    )

    excel_bytes = build_excel_scorecard(anomaly_results, validation_outcomes, before_profile, after_profile)

    return {
        "cleaned_dataset.csv": csv_buf.getvalue().encode("utf-8"),
        "changelog.md": changelog.encode("utf-8"),
        "dq_report.html": html_report.encode("utf-8"),
        "dq_scorecard.xlsx": excel_bytes,
    }


def write_reports_to_dir(reports: dict[str, bytes], out_dir: Path) -> None:
    """Local CLI/test use only — the Streamlit app serves these as in-memory
    download buttons instead, never writing to shared disk (see plan)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, data in reports.items():
        (out_dir / filename).write_bytes(data)
