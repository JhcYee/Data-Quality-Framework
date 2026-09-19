"""Format-dispatch ingestion: bytes in, a raw (object-dtype) DataFrame plus
per-column dtype guesses out. Nothing here commits to a final dtype — that's
Schema Confirmation's job (see schema_confirmation.py). This module only:

  1. Parses the file safely (encoding/delimiter fallback for CSV, sheet
     listing for Excel, XXE-safe parsing for XML).
  2. Normalizes common null-sentinel strings ("N/A", "None", ...) to real NaN
     *before* anything downstream sees the column — otherwise a column full
     of "N/A" would report 0% nulls later, which breaks null detection
     outright rather than just missing an edge case.
  3. Guesses a dtype per column as a suggestion, with a confidence score and
     sample values, for the user to confirm or override.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field

import pandas as pd

from .constants import MAX_DESIGN_ROWS, NULL_SENTINELS

# Ask each format-specific loader for one row more than the design cap, so
# load_file() can tell "exactly at the cap" apart from "more rows exist
# beyond it" without a separate full-file scan just to count rows.
_READ_LIMIT = MAX_DESIGN_ROWS + 1

BOOLEAN_TRUE_SETS = [
    {"true", "false"},
    {"yes", "no"},
    {"y", "n"},
    {"t", "f"},
]

SUPPORTED_EXTENSIONS = {"csv", "xlsx", "xls", "xlsm", "json", "xml"}


class IngestionError(Exception):
    """Raised for a file that can't be parsed at all — surfaced to the UI as
    a friendly error, never an unhandled traceback."""


@dataclass
class ColumnGuess:
    inferred_dtype: str  # one of DTYPE_CHOICES
    confidence: float  # 0..1, coercion-success rate for the guessed dtype
    samples: list[str] = field(default_factory=list)


@dataclass
class IngestionResult:
    raw_df: pd.DataFrame  # object dtype throughout, sentinel-normalized to NaN
    column_guesses: dict[str, ColumnGuess]
    sentinel_null_counts: dict[str, int]
    warnings: list[str] = field(default_factory=list)


def get_extension(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in SUPPORTED_EXTENSIONS:
        raise IngestionError(
            f"Unsupported file type '.{ext}'. Supported: "
            f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}."
        )
    return ext


def list_excel_sheets(file_bytes: bytes, ext: str) -> list[str]:
    engine = "xlrd" if ext == "xls" else "openpyxl"
    try:
        xl = pd.ExcelFile(io.BytesIO(file_bytes), engine=engine)
    except Exception as e:  # noqa: BLE001 - surfaced to the UI, not re-raised as-is
        raise IngestionError(f"Could not open Excel workbook: {e}") from e
    return xl.sheet_names


# --------------------------------------------------------------------------
# Format-specific loaders. Each returns a DataFrame with every cell as a
# plain string (or NaN for cells already empty on disk) — dtype guessing and
# sentinel normalization happen uniformly afterward, regardless of format.
# --------------------------------------------------------------------------


def _decode_bytes(file_bytes: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8"):
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    # Latin-1 maps every byte 0-255, so this never raises — genuinely
    # messy real-world exports are very often Windows-1252/Latin-1, not UTF-8.
    return file_bytes.decode("latin-1")


def _sniff_delimiter(sample: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        return ","


def _load_csv(file_bytes: bytes) -> pd.DataFrame:
    text = _decode_bytes(file_bytes)
    delimiter = _sniff_delimiter(text[:8192])
    try:
        return pd.read_csv(
            io.StringIO(text),
            sep=delimiter,
            dtype=str,
            keep_default_na=False,
            nrows=_READ_LIMIT,
        )
    except Exception as e:  # noqa: BLE001
        raise IngestionError(f"Could not parse CSV: {e}") from e


def _load_excel(file_bytes: bytes, ext: str, sheet_name: str | None) -> pd.DataFrame:
    engine = "xlrd" if ext == "xls" else "openpyxl"
    try:
        return pd.read_excel(
            io.BytesIO(file_bytes),
            sheet_name=sheet_name or 0,
            dtype=str,
            keep_default_na=False,
            engine=engine,
            nrows=_READ_LIMIT,
        )
    except Exception as e:  # noqa: BLE001
        raise IngestionError(f"Could not parse Excel file: {e}") from e


def _stringify_nested(df: pd.DataFrame) -> pd.DataFrame:
    """Lists/dicts left inside cells by json_normalize (arrays, or objects
    under a list) aren't hashable, which breaks profiling and the SQL layer.
    Keep them as JSON text so the value is still visible in the audit."""
    for col in df.columns:
        if df[col].dtype == object and df[col].map(lambda v: isinstance(v, (list, dict))).any():
            df[col] = df[col].map(
                lambda v: json.dumps(v, sort_keys=True, ensure_ascii=False) if isinstance(v, (list, dict)) else v
            )
    return df


def _load_json(file_bytes: bytes) -> pd.DataFrame:
    text = _decode_bytes(file_bytes)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Fall back to JSON Lines (one record per line). Stop reading lines
        # past _READ_LIMIT rather than parsing (and discarding) the rest.
        try:
            records = []
            for line in text.splitlines():
                if not line.strip():
                    continue
                records.append(json.loads(line))
                if len(records) >= _READ_LIMIT:
                    break
        except json.JSONDecodeError as e:
            raise IngestionError(f"Could not parse JSON: {e}") from e
        if not records:
            raise IngestionError("JSON file contained no records.")
        return _stringify_nested(pd.json_normalize(records))

    if isinstance(data, list):
        if not data:
            raise IngestionError("JSON file contained an empty list.")
        return _stringify_nested(pd.json_normalize(data[:_READ_LIMIT]))
    if isinstance(data, dict):
        # Common real-world shape: {"meta": {...}, "records": [...]}.
        for value in data.values():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                return _stringify_nested(pd.json_normalize(value[:_READ_LIMIT]))
        return _stringify_nested(pd.json_normalize([data]))
    raise IngestionError("Unsupported JSON structure — expected a list of records.")


def _validate_xml_safe(file_bytes: bytes) -> None:
    """Pre-flight check for XXE / entity-expansion attacks. We parse-and-
    discard with defusedxml first; if it raises, the file is rejected before
    pandas' own (feature-complete but not XXE-hardened) lxml-backed reader
    ever sees it.
    """
    from defusedxml.common import DefusedXmlException
    from defusedxml.ElementTree import fromstring as safe_fromstring

    try:
        safe_fromstring(file_bytes)
    except DefusedXmlException as e:
        raise IngestionError(f"XML file rejected for security reasons: {e}") from e
    except Exception:
        # Not a security issue — a genuine parse error, which pd.read_xml
        # below will raise again with its own (more specific) message.
        pass


def _load_xml(file_bytes: bytes) -> pd.DataFrame:
    _validate_xml_safe(file_bytes)
    try:
        df = pd.read_xml(io.BytesIO(file_bytes), dtype=str)
    except Exception as e:  # noqa: BLE001
        raise IngestionError(f"Could not parse XML file: {e}") from e
    # pandas' XML reader has no nrows param — the whole file gets parsed
    # regardless, but we still cap what flows into the rest of the pipeline.
    return df.head(_READ_LIMIT)


# --------------------------------------------------------------------------
# Null-sentinel normalization
# --------------------------------------------------------------------------


def normalize_null_sentinels(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    df = df.copy()
    counts: dict[str, int] = {}
    for col in df.columns:
        series = df[col]
        is_sentinel = series.apply(
            lambda v: isinstance(v, str) and v.strip().lower() in NULL_SENTINELS
        )
        n = int(is_sentinel.sum())
        if n:
            df.loc[is_sentinel, col] = pd.NA
            counts[col] = n
    return df, counts


# --------------------------------------------------------------------------
# Dtype guessing
# --------------------------------------------------------------------------


def _guess_column(series: pd.Series) -> ColumnGuess:
    non_null = series.dropna()
    samples = [str(v) for v in non_null.unique()[:5]]

    if non_null.empty:
        return ColumnGuess("string", 0.0, samples)

    stripped_lower = non_null.astype(str).str.strip().str.lower()

    # Boolean: only for explicit textual pairs, never bare 1/0 (too easily a
    # real integer/flag column instead).
    unique_vals = set(stripped_lower.unique())
    for pair in BOOLEAN_TRUE_SETS:
        if unique_vals and unique_vals.issubset(pair):
            return ColumnGuess("boolean", 1.0, samples)

    # Numeric
    numeric = pd.to_numeric(non_null.astype(str).str.strip(), errors="coerce")
    numeric_confidence = float(numeric.notna().mean())
    if numeric_confidence >= 0.95:
        non_null_numeric = numeric.dropna().astype("float64")
        is_integral = bool((non_null_numeric % 1 == 0).all())
        return ColumnGuess(
            "integer" if is_integral else "float", numeric_confidence, samples
        )

    # Datetime
    datetime_parsed = pd.to_datetime(non_null.astype(str).str.strip(), errors="coerce", format="mixed")
    datetime_confidence = float(datetime_parsed.notna().mean())
    if datetime_confidence >= 0.95:
        return ColumnGuess("datetime", datetime_confidence, samples)

    # Otherwise: categorical if low-cardinality relative to row count, else
    # free-text string. Either way this is just a starting suggestion.
    n_unique = non_null.nunique()
    n_total = len(non_null)
    if n_unique <= 200 and (n_unique / n_total) <= 0.5:
        return ColumnGuess("categorical", 1.0, samples)

    return ColumnGuess("string", 1.0, samples)


def guess_all_columns(df: pd.DataFrame) -> dict[str, ColumnGuess]:
    return {col: _guess_column(df[col]) for col in df.columns}


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def load_file(
    filename: str, file_bytes: bytes, sheet_name: str | None = None
) -> IngestionResult:
    ext = get_extension(filename)
    warnings: list[str] = []

    if ext == "csv":
        df = _load_csv(file_bytes)
    elif ext in ("xlsx", "xls", "xlsm"):
        df = _load_excel(file_bytes, ext, sheet_name)
    elif ext == "json":
        df = _load_json(file_bytes)
    elif ext == "xml":
        df = _load_xml(file_bytes)
    else:  # pragma: no cover - get_extension already validated this
        raise IngestionError(f"Unsupported file type '.{ext}'.")

    if df.empty or len(df.columns) == 0:
        raise IngestionError("The file parsed but contained no data.")

    if len(df) > MAX_DESIGN_ROWS:
        df = df.head(MAX_DESIGN_ROWS)
        warnings.append(
            f"This file has more than {MAX_DESIGN_ROWS:,} rows — only the first "
            f"{MAX_DESIGN_ROWS:,} were kept. This tool is designed and tested up "
            "to that scale; the rest were discarded rather than silently slowing "
            "everything down or exhausting memory."
        )

    df.columns = [str(c).strip() for c in df.columns]
    df = df.astype(object).where(df.notna(), pd.NA)

    df, sentinel_counts = normalize_null_sentinels(df)
    guesses = guess_all_columns(df)

    return IngestionResult(
        raw_df=df,
        column_guesses=guesses,
        sentinel_null_counts=sentinel_counts,
        warnings=warnings,
    )
