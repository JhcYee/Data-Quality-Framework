"""Generates a single, self-contained HTML data-quality report — the GX
validation results are embedded as data into our own styled page, rather
than linking GX's own multi-file Data Docs site (which isn't a single
portable file and is a poor fit for an ephemeral, per-upload context).
"""

from __future__ import annotations

import html
from datetime import datetime, timezone

from ..anomalies.types import AnomalyResult
from ..anomaly_sections import group_anomaly_results
from ..expectations import ValidationOutcome, describe_params
from ..profiling import DatasetProfile
from ..recommendations import Recommendation

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
h3 { font-size: 1.05rem; margin: 1.5rem 0 0.25rem; }
.section-note { color: #9aa4b2; font-size: 0.85rem; margin: 0 0 0.5rem; }
"""


def _row_count(results, passed: bool) -> int:
    return sum(1 for r in results if r.passed == passed)


def _status_span(passed: bool, label: str | None = None) -> str:
    label = label or ("PASS" if passed else "FAIL")
    return f'<span class="{"pass" if passed else "fail"}">{label}</span>'


def render_html_report(
    dataset_name: str,
    anomaly_results: list[AnomalyResult],
    validation_outcomes: list[ValidationOutcome],
    recommendations: list[Recommendation],
    profile: DatasetProfile,
) -> str:
    esc = html.escape
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    n_pass = _row_count(anomaly_results, True)
    n_fail = _row_count(anomaly_results, False)
    n_exp_pass = sum(1 for v in validation_outcomes if v.success)
    n_exp_fail = sum(1 for v in validation_outcomes if not v.success)

    cards = f"""
    <div class="grid">
      <div class="card"><div class="label">Rows</div><div class="value">{profile.n_rows:,}</div></div>
      <div class="card"><div class="label">Columns</div><div class="value">{profile.n_columns}</div></div>
      <div class="card"><div class="label">Anomaly checks passed</div><div class="value">{n_pass}/{n_pass + n_fail}</div></div>
      <div class="card"><div class="label">Validation rules passed</div><div class="value">{n_exp_pass}/{n_exp_pass + n_exp_fail}</div></div>
      <div class="card"><div class="label">Recommended actions</div><div class="value">{len(recommendations)}</div></div>
    </div>
    """

    anomaly_sections = []
    for sec in group_anomaly_results(anomaly_results):
        status = (
            f'<span class="fail">{sec.n_failing} failing</span>'
            if sec.n_failing
            else '<span class="pass">all passing</span>'
        )
        body = "\n".join(
            f"<tr><td>{esc(r.target)}</td><td>{_status_span(r.passed)}</td>"
            f"<td>{r.affected_rows:,}</td><td>{esc(r.summary)}</td></tr>"
            for r in sec.rows
        )
        anomaly_sections.append(
            f"<h3>{esc(sec.title)} — {status} · {len(sec.rows)} checked</h3>"
            f'<p class="section-note">{esc(sec.description)}</p>'
            "<table><thead><tr><th>Checked</th><th>Result</th><th>Rows affected</th><th>Summary</th></tr></thead>"
            f"<tbody>{body}</tbody></table>"
        )
    anomaly_html = "\n".join(anomaly_sections) or "<p>No checks ran.</p>"

    validation_rows = "\n".join(
        f"<tr><td>{esc(v.expectation_type)}</td><td>{esc(v.column or '')}</td>"
        f"<td>{esc(describe_params(v.params))}</td><td>{esc(v.source)}</td>"
        f"<td>{_status_span(v.success, v.status)}</td><td>{v.unexpected_count:,}</td>"
        f"<td>{v.unexpected_percent:.1%}</td></tr>"
        for v in validation_outcomes
    )

    action_rows = "\n".join(
        f"<tr><td>{esc(r.column or '(table-level)')}</td><td>{esc(r.issue)}</td>"
        f"<td>{(f'{r.affected_rows:,} ({r.pct_of_rows:.1%})' if r.affected_rows else '—')}</td>"
        f"<td>{esc(r.finding)}</td><td>{esc(r.action)}</td></tr>"
        for r in recommendations
    )

    profile_rows = "\n".join(
        f"<tr><td>{esc(col)}</td><td>{esc(b.dtype)}</td>"
        f"<td>{b.null_pct:.1%}</td><td>{b.unique_pct:.1%}</td></tr>"
        for col, b in profile.columns.items()
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
    {anomaly_html}
  </section>

  <section>
    <h2>Validation Rules (Great Expectations)</h2>
    <table>
      <thead><tr><th>Expectation</th><th>Column</th><th>Rule</th><th>Source</th><th>Result</th><th>Unexpected count</th><th>Unexpected %</th></tr></thead>
      <tbody>{validation_rows or '<tr><td colspan="7">No validation rules were run.</td></tr>'}</tbody>
    </table>
  </section>

  <section>
    <h2>Recommended Actions</h2>
    <p class="subtitle">Ordered by rows affected. Nothing in the uploaded data has been changed.</p>
    <table>
      <thead><tr><th>Column</th><th>Issue</th><th>Rows affected</th><th>Why it was flagged</th><th>Recommended action</th></tr></thead>
      <tbody>{action_rows or '<tr><td colspan="5">No data quality issues were found.</td></tr>'}</tbody>
    </table>
  </section>

  <section>
    <h2>Column Profile</h2>
    <table>
      <thead><tr><th>Column</th><th>Dtype</th><th>Null %</th><th>Unique %</th></tr></thead>
      <tbody>{profile_rows}</tbody>
    </table>
  </section>
</body>
</html>
"""
