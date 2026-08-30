"""
================================================================================
NEEV PROJECT — PROBLEM STATEMENT 1: SLEEP HEALTH ANALYTICS
Role      : Person 1 — Data Engineer
Input     : Sleep_health_and_lifestyle_dataset.csv  (raw dataset)
Output    : cleaned_data.csv
Purpose   : Take the raw Sleep Health & Lifestyle dataset and turn it into a
            validated, analysis-ready CSV that Person 2 (Feature Engineer)
            can consume directly to build the Sleep_Health_Tier column.

PIPELINE POSITION
------------------------------------------------------------------------------
    RAW DATA  --[THIS SCRIPT]-->  cleaned_data.csv  --[Person 2]-->
    processed_data.csv  --[Person 3]--> charts  --[Person 4]--> dashboard
    --[Person 5]--> executive summary

DOWNSTREAM COMPATIBILITY REQUIREMENT (CRITICAL)
------------------------------------------------------------------------------
Person 2's feature_engineering.py hard-codes:

    REQUIRED_COLUMNS = ["Sleep Duration", "Quality of Sleep", "Stress Level"]

and performs numerical comparisons such as `row["Sleep Duration"] < 6.0`.
Therefore this script GUARANTEES that:
  1. These three columns exist under EXACTLY these names.
  2. These three columns are numeric dtypes (no strings, no NaN silently
     breaking comparisons).
This script does NOT create `Sleep_Health_Tier` or any tier labels — that is
explicitly Person 2's responsibility.

DATASET REALITY (discovered by inspecting the actual uploaded file)
------------------------------------------------------------------------------
- 374 rows, 13 columns. Columns already match the problem statement's naming
  exactly (no renaming required).
- No missing values in ANY column except `Sleep Disorder` (219 / 374 rows,
  ~58.6%), which the problem statement explicitly says to label "None".
- Zero fully-duplicated rows (Person ID is a unique row identifier).
  242 rows are duplicated once `Person ID` is excluded — this simply reflects
  that many individuals in this dataset share identical lifestyle profiles.
  Per the problem statement and instructions, these are NOT removed: they are
  legitimate repeated observations, not data-entry duplication artifacts.
- All numeric columns (Age, Sleep Duration, Quality of Sleep, Physical
  Activity Level, Stress Level, Heart Rate, Daily Steps) are already valid
  numeric dtypes with plausible ranges (e.g. Sleep Duration 5.8-8.5 hrs,
  Quality of Sleep 4-9, Stress Level 3-8). No negative, zero, or
  out-of-range values were found.
- No leading/trailing whitespace found in any text column.
- ONE genuine categorical inconsistency was found: `BMI Category` contains
  both "Normal" (195 rows) and "Normal Weight" (21 rows) — two spellings of
  the identical clinical category. This is standardized to "Normal" (see
  clean_text_columns() for the documented justification).

Dependencies: pandas, numpy (stdlib pathlib). No ML/visualization libraries.
================================================================================
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

# ------------------------------------------------------------------------
# CONFIGURATION — no hardcoded paths scattered through the program
# ------------------------------------------------------------------------
INPUT_FILE = Path("Sleep_health_and_lifestyle_dataset.csv")
OUTPUT_FILE = Path("cleaned_data.csv")

REQUIRED_COLUMNS = ["Sleep Duration", "Quality of Sleep", "Stress Level"]

# Columns whose numeric integrity is critical for downstream tier logic.
CRITICAL_NUMERIC_COLUMNS = ["Sleep Duration", "Quality of Sleep", "Stress Level"]

# Other numeric columns present in the dataset that are useful for
# downstream visualization/dashboard work (Person 3 / Person 4) and are
# therefore also validated, though they are not required by Person 2.
OTHER_NUMERIC_COLUMNS = ["Age", "Physical Activity Level", "Heart Rate", "Daily Steps"]

TEXT_COLUMNS = ["Gender", "Occupation", "BMI Category", "Blood Pressure", "Sleep Disorder"]


def _section(title: str) -> None:
    print("\n" + "-" * 60)
    print(title)
    print("-" * 60)


# ------------------------------------------------------------------------
# [1] LOAD
# ------------------------------------------------------------------------
def load_dataset(path: Path) -> pd.DataFrame:
    """Load the raw CSV with clear, specific error handling."""
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found.\n"
            f"  Expected filename : {path.name}\n"
            f"  Looked in         : {path.resolve().parent}\n"
            f"  Full expected path: {path.resolve()}\n"
            f"Place the raw dataset next to this script, or update INPUT_FILE."
        )

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Input file '{path}' exists but contains no data.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Input file '{path}' could not be parsed as CSV: {exc}") from exc

    if df.empty:
        raise ValueError(f"Input file '{path}' was loaded but the resulting DataFrame is empty.")

    print(f"[OK] Loaded '{path.name}' -> {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ------------------------------------------------------------------------
# [2] INITIAL PROFILING
# ------------------------------------------------------------------------
def profile_dataset(df: pd.DataFrame, label: str = "RAW DATASET PROFILE") -> None:
    """Print a readable, non-flooding summary of the dataset's current state."""
    _section(label)

    print(f"Dimensions: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Columns   : {list(df.columns)}")

    print("\nData types:")
    print(df.dtypes.to_string())

    print("\nMissing values (count / %):")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_report = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
    missing_report = missing_report[missing_report["missing_count"] > 0]
    if missing_report.empty:
        print("  None")
    else:
        print(missing_report.to_string())

    dup_count = df.duplicated().sum()
    print(f"\nExact duplicate rows: {dup_count}")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        print("\nNumerical summary:")
        print(df[numeric_cols].describe().T[["min", "mean", "max"]].round(2).to_string())

    print("\nCategorical inspection (value counts):")
    for col in TEXT_COLUMNS:
        if col in df.columns:
            uniques = df[col].nunique(dropna=True)
            print(f"  {col}: {uniques} unique values -> {df[col].value_counts(dropna=False).to_dict()}")


# ------------------------------------------------------------------------
# [3] SCHEMA VALIDATION
# ------------------------------------------------------------------------
def validate_required_columns(df: pd.DataFrame, required: list) -> None:
    """Stop execution with a descriptive error if any required column is missing."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Schema validation failed — required column(s) missing.\n"
            f"  Missing columns  : {missing}\n"
            f"  Available columns: {list(df.columns)}\n"
            "This dataset cannot be handed off to Person 2 without these columns."
        )
    print(f"[OK] All required columns present: {required}")


# ------------------------------------------------------------------------
# [4] COLUMN NAME CLEANING
# ------------------------------------------------------------------------
def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip incidental whitespace from column headers only. Does NOT rename
    legitimate columns — the raw dataset's headers already match the exact
    names Person 2 requires ('Sleep Duration', 'Quality of Sleep',
    'Stress Level'), so no substantive renaming is performed.
    """
    df = df.copy()
    original_columns = list(df.columns)
    df.columns = [str(c).strip() for c in df.columns]

    changed = [(o, n) for o, n in zip(original_columns, df.columns) if o != n]
    if changed:
        print(f"[OK] Stripped whitespace from {len(changed)} column name(s): {changed}")
    else:
        print("[OK] Column names already clean — no changes needed.")

    # Guard against accidental duplicate column names after stripping.
    dupes = df.columns[df.columns.duplicated()].tolist()
    if dupes:
        raise ValueError(f"Duplicate column names detected after cleaning: {dupes}")

    return df


# ------------------------------------------------------------------------
# [5] TEXT / CATEGORICAL CLEANING
# ------------------------------------------------------------------------
def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Whitespace-strip all text columns, and fix the ONE confirmed formatting
    inconsistency found in this dataset: BMI Category contains both
    "Normal" and "Normal Weight" as two spellings of the same clinical
    category (verified via value_counts: 195 vs 21 rows). This is a
    formatting inconsistency, not a legitimate distinct category, so it is
    safe to standardize. No other categorical values are altered — Gender,
    Occupation, Blood Pressure, and Sleep Disorder values were inspected and
    found to already be clean and consistent.
    """
    df = df.copy()

    for col in TEXT_COLUMNS:
        if col in df.columns and df[col].dtype == object or (col in df.columns and str(df[col].dtype).lower() in ("object", "str", "string")):
            before = df[col].copy()
            df[col] = df[col].apply(lambda v: v.strip() if isinstance(v, str) else v)
            n_changed = (before.astype(str) != df[col].astype(str)).sum()
            if n_changed:
                print(f"[OK] Stripped whitespace in '{col}' for {n_changed} value(s).")

    if "BMI Category" in df.columns:
        before_counts = df["BMI Category"].value_counts(dropna=False).to_dict()
        n_fixed = (df["BMI Category"] == "Normal Weight").sum()
        df["BMI Category"] = df["BMI Category"].replace({"Normal Weight": "Normal"})
        if n_fixed:
            print(
                f"[OK] Standardized 'BMI Category': merged {n_fixed} 'Normal Weight' "
                f"value(s) into 'Normal' (same clinical category, inconsistent label). "
                f"Before: {before_counts} -> After: {df['BMI Category'].value_counts(dropna=False).to_dict()}"
            )

    return df


# ------------------------------------------------------------------------
# [6] SLEEP DISORDER MISSING-VALUE RULE (problem-statement mandated)
# ------------------------------------------------------------------------
def handle_sleep_disorder_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Problem statement: "Handle missing entries in Sleep Disorder by
    labeling unrecorded values as 'None'." Applied literally: NaN -> "None"
    (string), representing "no sleep disorder recorded" rather than leaving
    the column with an ambiguous NaN that downstream grouping/plotting
    would silently drop.
    """
    df = df.copy()
    if "Sleep Disorder" not in df.columns:
        print("[WARNING] 'Sleep Disorder' column not found — skipping rule.")
        return df

    n_missing = df["Sleep Disorder"].isnull().sum()
    df["Sleep Disorder"] = df["Sleep Disorder"].fillna("None")
    print(f"[OK] Sleep Disorder: replaced {n_missing} missing value(s) with 'None' (per problem statement).")

    remaining_na = df["Sleep Disorder"].isnull().sum()
    if remaining_na:
        raise ValueError(f"Sleep Disorder still has {remaining_na} NaN value(s) after fill — investigate.")

    return df


# ------------------------------------------------------------------------
# [7] NUMERIC CLEANING
# ------------------------------------------------------------------------
def clean_numeric_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Force each column to a numeric dtype using pd.to_numeric(errors="coerce").
    Any value that fails conversion becomes NaN — and is explicitly reported
    (count + column), never silently hidden. This is required so that
    Person 2's comparisons like `row["Sleep Duration"] < 6.0` never raise a
    TypeError against a string column.
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        before_dtype = df[col].dtype
        before_na = df[col].isnull().sum()

        numeric_col = pd.to_numeric(df[col], errors="coerce")
        newly_created_na = numeric_col.isnull().sum() - before_na

        if newly_created_na > 0:
            bad_mask = numeric_col.isnull() & df[col].notnull()
            bad_values = df.loc[bad_mask, col].unique().tolist()
            print(
                f"[WARNING] '{col}': {newly_created_na} value(s) could not be converted to numeric "
                f"and became NaN. Offending raw values: {bad_values}"
            )
        else:
            print(f"[OK] '{col}': dtype {before_dtype} -> {numeric_col.dtype}, 0 conversion failures.")

        df[col] = numeric_col

    return df


# ------------------------------------------------------------------------
# [8] MISSING NUMERICAL VALUES
# ------------------------------------------------------------------------
def handle_missing_numeric_values(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Policy: this dataset's numeric columns contain zero missing values in
    their raw form (verified during profiling), and neither the problem
    statement nor the dataset gives any instruction for imputing sleep,
    quality, or stress metrics. Per instructions, we do NOT fabricate
    values (no mean/median/constant fill) without justification.

    If any NaN is present here (either originally, or introduced by the
    numeric-coercion step above because a value was non-numeric garbage),
    the affected rows are dropped — because a fabricated Sleep Duration,
    Quality of Sleep, or Stress Level value would silently corrupt Person
    2's tier classification, which is a worse outcome than losing a row.
    This is reported explicitly, never silent.
    """
    df = df.copy()
    total_before = len(df)

    na_counts = df[columns].isnull().sum()
    cols_with_na = na_counts[na_counts > 0]

    if cols_with_na.empty:
        print(f"[OK] No missing values found in {columns} — no rows dropped.")
        return df

    print(f"[WARNING] Missing values found in critical numeric columns:\n{cols_with_na.to_string()}")
    rows_to_drop = df[columns].isnull().any(axis=1).sum()
    df = df.dropna(subset=columns)
    print(
        f"[OK] Dropped {rows_to_drop} row(s) with missing critical numeric data "
        f"(no defensible imputation available). Rows: {total_before} -> {len(df)}."
    )
    return df


# ------------------------------------------------------------------------
# [9] DUPLICATE HANDLING
# ------------------------------------------------------------------------
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Only exact, fully-identical rows are treated as duplicates. Rows that
    merely share the same profession/sleep duration/stress level (but have
    a different Person ID) are legitimate distinct observations and are
    preserved, per explicit instruction.
    """
    df = df.copy()
    before = df.duplicated().sum()
    print(f"Duplicates before cleaning: {before}")

    if before > 0:
        df = df.drop_duplicates()
        print(f"Duplicates removed: {before}")
    else:
        print("Duplicates removed: 0 (none found)")

    after = df.duplicated().sum()
    print(f"Duplicates after cleaning: {after}")
    if after != 0:
        raise ValueError("Duplicate removal failed — duplicates still present.")

    return df


# ------------------------------------------------------------------------
# [10] DOMAIN / RANGE VALIDATION
# ------------------------------------------------------------------------
def validate_domain_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check plausibility of the three critical numeric columns using ranges
    grounded in the actual data and common sense (no invented bounds):
      - Sleep Duration : must be > 0 hours (a value of 0 or negative is
        physically impossible for a 24h self-reported average).
      - Quality of Sleep and Stress Level : this dataset self-reports both
        on a 1-10 scale (observed data range 4-9 and 3-8 respectively), so
        values outside 1-10 are flagged as implausible.
    No values were found to violate these bounds in the raw data at the
    time of writing; this function exists so that any future/different
    upload of this same problem statement's dataset is still protected
    against implausible values reaching Person 2. Detected violations are
    REPORTED and the affected rows REMOVED (not clipped, since clipping
    would silently fabricate a different value).
    """
    df = df.copy()
    violations = pd.Series(False, index=df.index)

    if "Sleep Duration" in df.columns:
        bad = df["Sleep Duration"] <= 0
        if bad.any():
            print(f"[WARNING] Sleep Duration: {bad.sum()} value(s) <= 0 (implausible). Rows will be removed.")
        violations |= bad

    for col in ["Quality of Sleep", "Stress Level"]:
        if col in df.columns:
            bad = ~df[col].between(1, 10)
            if bad.any():
                print(f"[WARNING] {col}: {bad.sum()} value(s) outside plausible 1-10 scale. Rows will be removed.")
            violations |= bad

    n_violations = violations.sum()
    if n_violations:
        df = df.loc[~violations].copy()
        print(f"[OK] Removed {n_violations} row(s) failing domain validation.")
    else:
        print("[OK] All critical numeric columns fall within plausible domain ranges — no rows removed.")

    return df


# ------------------------------------------------------------------------
# [11] FINAL VALIDATION
# ------------------------------------------------------------------------
def validate_final_dataset(df: pd.DataFrame, rows_before: int) -> None:
    _section("FINAL VALIDATION")

    validate_required_columns(df, REQUIRED_COLUMNS)

    for col in REQUIRED_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"Required column '{col}' is not numeric (dtype={df[col].dtype}).")
    print(f"[OK] Required columns are numeric: {REQUIRED_COLUMNS}")

    dup_count = df.duplicated().sum()
    if dup_count != 0:
        raise ValueError(f"Final dataset still contains {dup_count} exact duplicate row(s).")
    print("[OK] No unintended exact duplicates remain.")

    if "Sleep Disorder" in df.columns:
        remaining_na = df["Sleep Disorder"].isnull().sum()
        if remaining_na:
            raise ValueError(f"'Sleep Disorder' still has {remaining_na} missing value(s), expected 0.")
        print("[OK] 'Sleep Disorder' missing values fully represented as 'None'.")

    print("\nFinal missing-value report:")
    final_missing = df.isnull().sum()
    final_missing = final_missing[final_missing > 0]
    print(final_missing.to_string() if not final_missing.empty else "  None")

    print(f"\nRows before cleaning: {rows_before}")
    print(f"Rows after cleaning : {len(df)}")
    print(f"Rows removed        : {rows_before - len(df)}")
    print(f"\nFinal columns ({len(df.columns)}): {list(df.columns)}")

    print("\nSample of final data (first 5 rows):")
    print(df.head(5).to_string())


# ------------------------------------------------------------------------
# [12] PERSON 2 COMPATIBILITY TEST
# ------------------------------------------------------------------------
def validate_person2_compatibility(df: pd.DataFrame) -> None:
    """
    Simulates the exact numerical comparisons Person 2's tier-classification
    logic performs, WITHOUT creating Sleep_Health_Tier or any tier labels.
    If any comparison raises a TypeError, we fail loudly here rather than
    handing Person 2 a broken file.
    """
    _section("PERSON 2 COMPATIBILITY CHECK")
    try:
        cond_a = df["Sleep Duration"] < 6.0
        cond_b = df["Sleep Duration"] < 6.5
        cond_c = df["Quality of Sleep"] <= 5
        cond_d = df["Stress Level"] >= 6
        cond_e = df["Sleep Duration"] < 7.0

        # Exercise them together the way Person 2's boolean logic would.
        _tier1_like = cond_a | (cond_b & cond_c)
        _tier2_like = (~_tier1_like) & (cond_e & cond_d)

        assert cond_a.dtype == bool and cond_e.dtype == bool
    except (TypeError, KeyError) as exc:
        raise RuntimeError(
            "Person 2 compatibility check FAILED — output would break "
            f"feature_engineering.py. Error: {exc}"
        ) from exc

    print("[OK] All Person 2 comparison operations executed without type errors:")
    print('     df["Sleep Duration"] < 6.0')
    print('     df["Sleep Duration"] < 6.5')
    print('     df["Quality of Sleep"] <= 5')
    print('     df["Stress Level"] >= 6')
    print('     df["Sleep Duration"] < 7.0')
    print("[OK] 'Sleep_Health_Tier' was NOT created — that remains Person 2's responsibility.")


# ------------------------------------------------------------------------
# [13] SAVE
# ------------------------------------------------------------------------
def save_dataset(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_csv(path, index=False)
    except OSError as exc:
        raise OSError(f"Failed to write output file '{path}': {exc}") from exc
    print(f"[OK] Cleaned dataset saved to: {path.resolve()}")


def verify_csv_round_trip(path: Path) -> None:
    """
    KNOWN PANDAS GOTCHA — verified, not assumed:
    pandas.read_csv() treats the literal text 'None' (along with 'NaN',
    'NA', 'null', etc.) as a missing value BY DEFAULT. That means even
    though this script correctly writes the string "None" into
    'Sleep Disorder', a downstream script that opens cleaned_data.csv with
    plain `pd.read_csv(...)` will silently turn every one of those "None"
    entries back into NaN — quietly undoing the problem statement's
    explicit requirement.

    This function proves the issue is real (not theoretical) by reading
    the just-saved file back with pandas' default settings and checking
    whether 'Sleep Disorder' has NaNs again. If so, it prints an
    unmissable warning with the exact fix Person 2 (or anyone else) needs
    to use when loading this file.
    """
    reread = pd.read_csv(path)
    if "Sleep Disorder" not in reread.columns:
        return

    reread_na = reread["Sleep Disorder"].isnull().sum()
    if reread_na > 0:
        print(
            "\n[WARNING] CSV ROUND-TRIP GOTCHA DETECTED:\n"
            f"  Reading '{path.name}' back with plain pd.read_csv() turns "
            f"{reread_na} 'None' string(s) in 'Sleep Disorder' back into NaN.\n"
            "  This is pandas' default NA-value handling, not a bug in this script.\n"
            "  ACTION FOR PERSON 2 (and anyone else loading this file):\n"
            "    df = pd.read_csv('cleaned_data.csv', keep_default_na=False, na_values=[])\n"
            "  or, narrower:\n"
            "    df = pd.read_csv('cleaned_data.csv')\n"
            "    df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None')\n"
        )
    else:
        print("[OK] Round-trip check: 'Sleep Disorder' values survive a plain pd.read_csv() re-load.")


# ------------------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("SLEEP HEALTH ANALYTICS — DATA CLEANING PIPELINE")
    print("Person 1 — Data Engineer")
    print("=" * 60)

    print("\n[1/8] Loading dataset...")
    df = load_dataset(INPUT_FILE)
    rows_before = len(df)

    print("\n[2/8] Profiling dataset...")
    profile_dataset(df, label="RAW DATASET PROFILE")

    print("\n[3/8] Validating schema...")
    validate_required_columns(df, REQUIRED_COLUMNS)

    print("\n[4/8] Cleaning values (column names, text, numeric types)...")
    df = clean_column_names(df)
    validate_required_columns(df, REQUIRED_COLUMNS)  # re-check post-strip
    df = clean_text_columns(df)
    df = clean_numeric_columns(df, CRITICAL_NUMERIC_COLUMNS + OTHER_NUMERIC_COLUMNS)

    print("\n[5/8] Handling missing values...")
    df = handle_sleep_disorder_missing(df)
    df = handle_missing_numeric_values(df, CRITICAL_NUMERIC_COLUMNS)

    print("\n[6/8] Removing duplicates...")
    df = remove_duplicates(df)

    print("\n[6b/8] Validating domain ranges...")
    df = validate_domain_ranges(df)

    print("\n[7/8] Running final validation...")
    validate_final_dataset(df, rows_before)
    validate_person2_compatibility(df)

    print("\n[8/8] Saving cleaned dataset...")
    save_dataset(df, OUTPUT_FILE)
    verify_csv_round_trip(OUTPUT_FILE)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # top-level: surface the real error, then exit non-zero
        print(f"\n[ERROR] Pipeline failed: {exc}", file=sys.stderr)
        sys.exit(1)