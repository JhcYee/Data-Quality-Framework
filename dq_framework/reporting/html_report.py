"""Generates a single, self-contained HTML data-quality report — the GX
validation results are embedded as data into our own styled page, rather
than linking GX's own multi-file Data Docs site (which isn't a single
portable file and is a poor fit for an ephemeral, per-upload context).
"""

from __future__ import annotations

import html
from datetime import datetime, timezone

from ..anomalies.types import AnomalyResult
from ..cleaning import TransformationLogEntry
from ..expectations import ValidationOutcome
from ..profiling import DatasetProfile

CSS = """
:root { color-scheme: light dark; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       margin: 0; padding: 2rem; background: #0b0e14; color: #e6e6e6; }
h1, h2 { font-weight: 600; }
h1 { font-size: 1.6rem; margin-bottom: 0.25rem; }
.subtitle { color: #9aa4b2; margin-bottom: 2rem; }
.grid { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 2rem; }
.card { background: #151a24; border: 1px solid #262d3a; border-radius: 10px;
        padding: 1rem 1.25rem; min-width: 160px; }
.card .label { color: #9aa4b2; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.04em; }
.card .value { font-size: 1.6rem; font-weight: 700; margin-top: 0.25rem; }
table { border-collapse: collapse; width: 100%; margin-bottom: 2rem; font-size: 0.9rem; }
th, td { text-align: left; padding: 0.5rem 0.75rem; border-bottom: 1px solid #262d3a; }
th { color: #9aa4b2; text-transform: uppercase; font-size: 0.72rem; letter-spacing: 0.04em; }
.pass { color: #4ade80; font-weight: 600; }
.fail { color: #f87171; font-weight: 600; }
section { margin-bottom: 2.5rem; }
"""


def _row_count(results, passed: bool) -> int:
    return sum(1 for r in results if r.passed == passed)


def _status_span(passed: bool) -> str:
    return '<span class="pass">PASS</span>' if passed else '<span class="fail">FAIL</span>'


def render_html_report(
    dataset_name: str,
    anomaly_results: list[AnomalyResult],
    validation_outcomes: list[ValidationOutcome],
    transformation_log: list[TransformationLogEntry],
    before_profile: DatasetProfile,
    after_profile: DatasetProfile | None = None,
) -> str:
    esc = html.escape
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    n_pass = _row_count(anomaly_results, True)
    n_fail = _row_count(anomaly_results, False)
    n_exp_pass = sum(1 for v in validation_outcomes if v.success)
    n_exp_fail = sum(1 for v in validation_outcomes if not v.success)

    cards = f"""
    <div class="grid">
      <div class="card"><div class="label">Rows (before)</div><div class="value">{before_profile.n_rows:,}</div></div>
      <div class="card"><div class="label">Rows (after)</div><div class="value">{(after_profile.n_rows if after_profile else before_profile.n_rows):,}</div></div>
      <div class="card"><div class="label">Anomaly checks passed</div><div class="value">{n_pass}/{n_pass + n_fail}</div></div>
      <div class="card"><div class="label">Validation rules passed</div><div class="value">{n_exp_pass}/{n_exp_pass + n_exp_fail}</div></div>
      <div class="card"><div class="label">Changes applied</div><div class="value">{len(transformation_log)}</div></div>
    </div>
    """

    anomaly_rows = "\n".join(
        f"<tr><td>{esc(r.check_name)}</td><td>{_status_span(r.passed)}</td>"
        f"<td>{r.affected_row_count:,}</td><td>{esc(r.summary)}</td></tr>"
        for r in anomaly_results
    )

    validation_rows = "\n".join(
        f"<tr><td>{esc(v.expectation_type)}</td><td>{esc(v.column or '')}</td>"
        f"<td>{_status_span(v.success)}</td><td>{v.unexpected_count:,}</td>"
        f"<td>{v.unexpected_percent:.1%}</td></tr>"
        for v in validation_outcomes
    )

    changelog_rows = "\n".join(
        f"<tr><td>{esc(e.column or '(table-level)')}</td>"
        f"<td>{esc(e.change_type.replace('_', ' '))}</td><td>{esc(e.reason)}</td></tr>"
        for e in transformation_log
    )

    profile_rows = "\n".join(
        f"<tr><td>{esc(col)}</td><td>{esc(b.dtype)}</td>"
        f"<td>{b.null_pct:.1%}</td>"
        f"<td>{(after_profile.columns[col].null_pct if after_profile and col in after_profile.columns else b.null_pct):.1%}</td>"
        f"<td>{b.unique_pct:.1%}</td></tr>"
        for col, b in before_profile.columns.items()
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Data Quality Report — {esc(dataset_name)}</title>
<style>{CSS}</style>
</head>
<body>
  <h1>Data Quality Report — {esc(dataset_name)}</h1>
  <div class="subtitle">Generated {generated_at}</div>

  {cards}

  <section>
    <h2>Anomaly Detection</h2>
    <table>
      <thead><tr><th>Check</th><th>Result</th><th>Rows affected</th><th>Summary</th></tr></thead>
      <tbody>{anomaly_rows or '<tr><td colspan="4">No checks ran.</td></tr>'}</tbody>
    </table>
  </section>

  <section>
    <h2>Validation Rules (Great Expectations)</h2>
    <table>
      <thead><tr><th>Expectation</th><th>Column</th><th>Result</th><th>Unexpected count</th><th>Unexpected %</th></tr></thead>
      <tbody>{validation_rows or '<tr><td colspan="5">No expectations were run.</td></tr>'}</tbody>
    </table>
  </section>

  <section>
    <h2>Cleaning Changelog</h2>
    <table>
      <thead><tr><th>Column</th><th>Change</th><th>Why</th></tr></thead>
      <tbody>{changelog_rows or '<tr><td colspan="3">No changes were made.</td></tr>'}</tbody>
    </table>
  </section>

  <section>
    <h2>Column Profile (before → after null %)</h2>
    <table>
      <thead><tr><th>Column</th><th>Dtype</th><th>Null % (before)</th><th>Null % (after)</th><th>Unique %</th></tr></thead>
      <tbody>{profile_rows}</tbody>
    </table>
  </section>
</body>
</html>
"""
