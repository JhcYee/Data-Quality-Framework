"""Orchestrates the full run: ingestion -> schema confirmation -> profiling
-> anomaly detection -> validation rules -> recommended actions -> reporting.
Used by both app.py (step by step, driven by the UI) and the tests (end to
end). The tool audits; it never modifies the uploaded data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import expectations
from .anomalies import categorical_standardization, consistency, duplicates, nulls, outliers, referential, schema_drift, typos
from .anomalies.types import AnomalyResult
from .expectations import ExpectationSpec, ValidationOutcome
from .profiling import DatasetProfile
from .recommendations import Recommendation
from .reporting.excel_scorecard import build_excel_scorecard
from .reporting.html_report import render_html_report
from .reporting.markdown_actions import render_markdown_actions
from .schema_confirmation import ConfirmedSchema


@dataclass
class AnomalyDetectionOutcome:
    results: list[AnomalyResult] = field(default_factory=list)


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
    dismissed_typo_variants: dict[str, set[str]] | None = None,
    previous_schema: ConfirmedSchema | None = None,
) -> AnomalyDetectionOutcome:
    outcome = AnomalyDetectionOutcome()
    engine.load(df, table_name)

    outcome.results.extend(
        nulls.detect_nulls(profile, nulls_expected_columns=set(schema.nulls_expected_columns))
    )

    outcome.results.append(duplicates.detect_exact_duplicates(engine, table_name, list(df.columns)))
    key_dupe = duplicates.detect_key_duplicates(engine, table_name, schema.key_columns)
    if key_dupe is not None:
        outcome.results.append(key_dupe)

    outcome.results.extend(outliers.detect_outliers(df, schema.column_types).values())
    outcome.results.append(schema_drift.detect_schema_drift(schema, previous_schema))
    outcome.results.extend(
        categorical_standardization.detect_categorical_variants(df, schema.column_types).values()
    )
    outcome.results.extend(
        typos.detect_typos(df, schema.column_types, dismissed_variants=dismissed_typo_variants).values()
    )

    if reference_df is not None and referential_pairs:
        engine.load(reference_df, reference_table_name)
        for main_col, ref_col in referential_pairs:
            outcome.results.append(
                referential.detect_referential_integrity(
                    engine, table_name, main_col, reference_table_name, ref_col
                )
            )

    for start_col, end_col in consistency_pairs or []:
        outcome.results.append(consistency.detect_consistency(engine, table_name, start_col, end_col))

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
        context,
        df,
        profile,
        schema.column_types,
        schema.key_columns,
        nulls_expected_columns=set(schema.nulls_expected_columns),
        gx_module=gx_module,
    )
    for spec in custom_expectations or []:
        expectations.add_manual_expectation(suite, spec.kind, spec.column, spec.params, gx_module=gx_module)
    result = expectations.validate_suite(context, df, suite, gx_module=gx_module)
    return expectations.summarize_validation(result)


def generate_reports(
    dataset_name: str,
    anomaly_results: list[AnomalyResult],
    validation_outcomes: list[ValidationOutcome],
    recommendations: list[Recommendation],
    profile: DatasetProfile,
) -> dict[str, bytes]:
    html_report = render_html_report(
        dataset_name, anomaly_results, validation_outcomes, recommendations, profile
    )
    return {
        "recommended_actions.md": render_markdown_actions(recommendations, dataset_name).encode("utf-8"),
        "dq_report.html": html_report.encode("utf-8"),
        "dq_scorecard.xlsx": build_excel_scorecard(
            anomaly_results, validation_outcomes, recommendations, profile
        ),
    }


def write_reports_to_dir(reports: dict[str, bytes], out_dir: Path) -> None:
    """Local CLI/test use only — the Streamlit app serves these as in-memory
    download buttons instead, never writing to shared disk."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, data in reports.items():
        (out_dir / filename).write_bytes(data)
