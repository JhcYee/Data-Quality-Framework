"""Shared tuning constants. Kept in one place so detectors that need to agree
with each other (outlier flags vs. the GX baseline suite's range tolerance)
can't silently drift apart.
"""

# Multiplier used for IQR-based fences: [Q1 - k*IQR, Q3 + k*IQR].
# Shared by anomalies/outliers.py and expectations.py's auto-generated
# baseline suite, so "flagged as an outlier" and "fails the baseline range
# expectation" agree on the same rows.
IQR_MULTIPLIER = 1.5

# Numeric columns with fewer distinct values than this are treated as
# discrete/coded (ratings, flags) rather than continuous measurements, and
# are skipped by IQR-based outlier detection.
MIN_DISTINCT_FOR_OUTLIER_CHECK = 10

# Null-percentage threshold above which a column is flagged as high-null.
HIGH_NULL_THRESHOLD = 0.20

# Categorical values with frequency below this share of non-null rows are
# left out of the auto-generated in_set baseline expectation (so they fail
# and surface as findings, instead of being baked in as "expected").
RARE_CATEGORY_FREQUENCY_THRESHOLD = 0.01

# Row-count / file-size thresholds for auto-selecting DuckDB over SQLite.
SQL_ENGINE_ROW_THRESHOLD = 100_000
SQL_ENGINE_SIZE_THRESHOLD_BYTES = 50 * 1024 * 1024

# Design/tested scale ceiling for this tool.
MAX_DESIGN_ROWS = 500_000

# Case/whitespace-insensitive strings treated as null before dtype guessing.
NULL_SENTINELS = {
    "", "na", "n/a", "n.a.", "none", "null", "nul", "nil", "-", "--",
    "unknown", "unk", "missing", "not available", "not applicable", "?",
    "nan", "nat",
}

DTYPE_CHOICES = ["string", "integer", "float", "boolean", "datetime", "categorical"]
