# Data Quality & Validation Audit Framework

An initial data quality check for a messy real-world dataset (CSV / XLSX / XLS / JSON / XML). It finds what's wrong, where, and how much, then tells you what to do about it. It never modifies your data: whether to impute a null or merge two categories depends on what the data means, and the fix itself is trivial once you know what to fix.

Upload a file and get back:

- a full anomaly profile (nulls, duplicates, outliers, schema drift, referential integrity breaks, cross-column consistency violations, categorical inconsistencies)
- an editable set of validation rules (Great Expectations), auto-suggested from the data and tunable by hand
- a prioritized list of recommended actions: for each issue, the column, how many rows it touches, why it was flagged, and one sentence on what to do
- downloadable reports: recommended actions (markdown), a styled HTML data-quality report, and an Excel scorecard

Everything runs locally. No paid services, no hosting cost, no outbound network calls at runtime — user upload is the only way data gets in.

## Run it

`.venv/` is gitignored (as usual — virtual environments don't belong in version control), but on the machine this was built on it already exists with everything installed. If it's there, just activate it:

```bash
cd data-quality-framework
source .venv/bin/activate
streamlit run app.py
```

Opens at `http://localhost:8501`.

### Fresh setup (a clone on another machine, or if `.venv` doesn't exist here)

```bash
cd data-quality-framework
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
streamlit run app.py
```

## Test it

```bash
source .venv/bin/activate
pytest              # fast suite
pytest -m slow      # includes the ~500k-row scale test
```

## Layout

See `dq_framework/` for the pipeline modules (ingestion → schema confirmation → profiling → anomaly detection → validation rules → recommended actions → reporting), orchestrated by `dq_framework/pipeline.py` and driven by the Streamlit UI in `app.py`.
