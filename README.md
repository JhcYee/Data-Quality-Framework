# Data Quality & Validation Audit Framework

Upload a messy real-world dataset (CSV / XLSX / XLS / JSON / XML) and get back:

- a full anomaly profile (nulls, duplicates, outliers, schema drift, referential integrity breaks, cross-column consistency violations, categorical inconsistencies)
- an editable set of validation rules (Great Expectations), auto-suggested from the data and tunable by hand
- a cleaned dataset, produced by a fixed, logged cleaning pipeline
- a full audit trail: a markdown changelog (what changed, in which column, and why), a styled HTML data-quality report, and an Excel scorecard

Everything runs locally. No paid services, no hosting cost, no outbound network calls at runtime — user upload is the only way data gets in.

## Run it

```bash
pip install -e ".[dev]"
streamlit run app.py
```

## Test it

```bash
pytest
```

## Layout

See `dq_framework/` for the pipeline modules (ingestion → schema confirmation → profiling → anomaly detection → validation rules → cleaning → reporting), orchestrated by `dq_framework/pipeline.py` and driven by the Streamlit UI in `app.py`.
