"""Merge mineral logs with WAMEX Au assays by CollarId and depth boundaries.

Log rows and assays match when both interval boundaries agree within
DEPTH_ATOL_M. The dataset holds measured, nonnegative Au results only: negative
Au (typically below-detection-limit codes), missing or nonfinite Au, invalid
depth intervals, log rows without a matching assay and ambiguous matches are
excluded and counted per hole. Numeric inputs are converted once at read time;
malformed text and infinities become NaN and are reported.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_DIR = (
    PROJECT_ROOT
    / "data/INTERIM_HyLogger_Dataset_Package_GDA2020_DB/DATABASES/CSV"
)
MINERAL_LOGS_DIR = PROJECT_ROOT / "data/extracted_data/mineral_logs"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data/extracted_data/mineral_au_dataset.csv"

DEPTH_FROM = "Depth_from_(m)"
DEPTH_TO = "Depth_to_(m)"
TARGET = "Au_PPM"

# Assumed recording tolerance (m) for each boundary; there is no relative
# tolerance and depths are never rounded. On the current cohort every retained
# log/assay pair differs by exactly 0 and a 0.01 m tolerance retains the same
# rows, so a micrometre-level value is used. Realised differences are reported
# in the notebook.
DEPTH_ATOL_M = 1e-6

# Intended cohort: the 22 holes retained for this analysis (logs follow the
# automated SWIR/TIR schema).
COHORT_DATASET_IDS = (
    "16CADD001",
    "16CADD002",
    "18LCDD0001",
    "18MMDD0434",
    "BSDD011",
    "BSDD012",
    "BSDD013",
    "BSDD014",
    "BSDD015A",
    "BSDD016",
    "BSDD017",
    "BSDD018",
    "BSDD019",
    "RRLCRDD002",
    "RRLCRDD004",
    "RRLCRDD005",
    "RRLCRDD006",
    "RRLCRDD007",
    "WRD08965-001",
    "WRD11720-003",
    "WRD13125-001",
    "WRD13400-001",
)

METADATA_COLUMNS = ["CollarId", "Project", "AssayFlatDataId", "CompanySampleId"]
SAMPLE_KEY_COLUMNS = ["Dataset_ID", DEPTH_FROM, DEPTH_TO]
LEADING_COLUMNS = [
    "Dataset_ID",
    *METADATA_COLUMNS,
    DEPTH_FROM,
    DEPTH_TO,
    "Assay_FromDepth",
    "Assay_ToDepth",
    TARGET,
]

# Outcome of each mineral-log row; "retained" rows form the dataset.
STATUSES = (
    "retained",
    "invalid_log_interval",
    "no_assay_match",
    "missing_or_nonfinite_au",
    "conflicting_assays",
    "negative_au",
)
ISSUE_COLUMNS = ["Source", "Dataset_ID", "column", "malformed", "infinite"]


def to_numeric_checked(
    frame: pd.DataFrame, columns: Sequence[str]
) -> tuple[pd.DataFrame, list[dict]]:
    """Convert columns to float; malformed text and infinities become NaN.

    Legitimate missing values stay NaN. Returns the converted frame and one
    issue record per column that had malformed or infinite values.
    """
    out = frame.copy()
    issues = []
    for column in columns:
        raw = out[column]
        number = pd.to_numeric(raw, errors="coerce").astype("float64")
        malformed = int((raw.notna() & number.isna()).sum())
        infinite = int(np.isinf(number).sum())
        out[column] = number.mask(np.isinf(number))
        if malformed or infinite:
            issues.append(
                {"column": column, "malformed": malformed, "infinite": infinite}
            )
    return out, issues


def require_numeric(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    """Raise TypeError unless every column has a numeric dtype."""
    bad = [
        c
        for c in columns
        if not pd.api.types.is_numeric_dtype(frame[c])
        or pd.api.types.is_bool_dtype(frame[c])
    ]
    if bad:
        raise TypeError(f"Non-numeric columns (convert at input): {bad}")


def validate_descriptors(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    """Require numeric descriptor columns without infinities; NaN is allowed."""
    require_numeric(frame, columns)
    infinite = [c for c in columns if np.isinf(frame[c]).any()]
    if infinite:
        raise ValueError(f"Infinite values in descriptor columns: {infinite}")


def read_mineral_log(path: Path, expected_id: str | None = None) -> pd.DataFrame:
    """Read one SM_*.csv: metadata block on top, data table below the depth header."""
    with path.open(encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    dataset_id = next(
        line.split(",")[1] for line in lines if line.startswith("Dataset,")
    )
    if expected_id is not None and dataset_id != expected_id:
        raise ValueError(
            f"{path.name} declares Dataset {dataset_id!r}, expected {expected_id!r}"
        )
    header_row = next(
        i for i, line in enumerate(lines) if line.startswith(DEPTH_FROM)
    )
    table = pd.read_csv(path, skiprows=header_row, na_values=["N.A."], dtype=str)
    table, issues = to_numeric_checked(table, list(table.columns))
    table.insert(0, "Dataset_ID", dataset_id)
    table.attrs["numeric_issues"] = issues
    return table


def load_dataset_metadata(index_path: Path, dataset_ids: Sequence[str]) -> pd.DataFrame:
    """CollarId and Project per Dataset_ID, taken from the HyLogger dataset index."""
    index = pd.read_csv(
        index_path,
        usecols=["Data_name", "WAMEX_CollarId", "PROJECT"],
        low_memory=False,
    )
    index = index[index["Data_name"].isin(dataset_ids)]
    duplicated = index.loc[index["Data_name"].duplicated(), "Data_name"].unique()
    if len(duplicated):
        raise ValueError(f"Duplicate HyLogger index entries for: {list(duplicated)}")
    index = index.set_index("Data_name").reindex(list(dataset_ids))
    missing = index.index[index["WAMEX_CollarId"].isna()].tolist()
    if missing:
        raise ValueError(f"No CollarId in HyLogger index for: {missing}")
    index["CollarId"] = index["WAMEX_CollarId"].astype("int64")
    return index.rename(columns={"PROJECT": "Project"})[["CollarId", "Project"]]


def valid_interval(start: pd.Series, end: pd.Series) -> pd.Series:
    """True where both depths are finite, nonnegative and end > start."""
    return np.isfinite(start) & np.isfinite(end) & (start >= 0) & (end > start)


def match_assays(
    logs: pd.DataFrame, assays: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match log rows to assays on CollarId and both interval boundaries.

    Each boundary must agree within DEPTH_ATOL_M. Returns the retained rows
    (one assay per log row) and a per-log-row outcome frame with a ``status``
    from STATUSES and the number of duplicate assays removed.

    All matching assays count, including those with missing or nonfinite Au.
    A log row keeps one assay only if every candidate is an identical record
    (same boundaries, Au and CompanySampleId); then the lowest AssayFlatDataId is
    kept and the rest are counted as duplicates. Candidates that differ in any of
    these fields make the row ``conflicting_assays`` and exclude it.
    """
    require_numeric(logs, [DEPTH_FROM, DEPTH_TO])
    require_numeric(assays, ["Assay_FromDepth", "Assay_ToDepth", TARGET])
    logs = logs.reset_index(drop=True).assign(_row=lambda d: np.arange(len(d)))
    outcome = pd.DataFrame(
        {
            "Dataset_ID": logs["Dataset_ID"],
            "status": "retained",
            "duplicates_removed": 0,
        }
    )

    valid_log = valid_interval(logs[DEPTH_FROM], logs[DEPTH_TO])
    outcome.loc[~valid_log, "status"] = "invalid_log_interval"

    assays = assays[valid_interval(assays["Assay_FromDepth"], assays["Assay_ToDepth"])]
    candidates = logs[valid_log].merge(assays, on="CollarId")
    aligned = (
        (candidates[DEPTH_FROM] - candidates["Assay_FromDepth"]).abs() <= DEPTH_ATOL_M
    ) & ((candidates[DEPTH_TO] - candidates["Assay_ToDepth"]).abs() <= DEPTH_ATOL_M)
    candidates = candidates[aligned]

    matched_rows = pd.Index(candidates["_row"].unique())
    outcome.loc[outcome.index[valid_log].difference(matched_rows), "status"] = (
        "no_assay_match"
    )

    record = ["Assay_FromDepth", "Assay_ToDepth", TARGET, "CompanySampleId"]
    n_records = candidates.groupby("_row").size()
    n_distinct = candidates.drop_duplicates(["_row", *record]).groupby("_row").size()
    conflict_rows = n_distinct.index[n_distinct > 1]
    outcome.loc[conflict_rows, "status"] = "conflicting_assays"

    resolved = (
        candidates[~candidates["_row"].isin(conflict_rows)]
        .sort_values(["_row", "AssayFlatDataId"])
        .drop_duplicates("_row")
    )
    rows = resolved["_row"].to_numpy()
    outcome.loc[rows, "duplicates_removed"] = n_records.loc[rows].to_numpy() - 1

    unmeasured = resolved[~np.isfinite(resolved[TARGET])]
    outcome.loc[unmeasured["_row"].to_numpy(), "status"] = "missing_or_nonfinite_au"
    measured = resolved[np.isfinite(resolved[TARGET])]
    negative = measured[measured[TARGET] < 0]
    outcome.loc[negative["_row"].to_numpy(), "status"] = "negative_au"
    return measured[measured[TARGET] >= 0].drop(columns="_row"), outcome


def summarise_exclusions(
    outcome: pd.DataFrame,
    metadata: pd.DataFrame,
    assay_invalid_intervals: pd.Series,
) -> pd.DataFrame:
    """Per-hole counts of log rows by outcome, indexed by Dataset_ID."""
    report = pd.crosstab(outcome["Dataset_ID"], outcome["status"]).reindex(
        index=metadata.index, columns=list(STATUSES), fill_value=0
    )
    report.insert(0, "log_rows", report.sum(axis=1))
    report.insert(0, "Project", metadata["Project"])
    report["duplicates_removed"] = (
        outcome.groupby("Dataset_ID")["duplicates_removed"]
        .sum()
        .reindex(report.index, fill_value=0)
    )
    report["assay_invalid_interval"] = assay_invalid_intervals.reindex(
        report.index, fill_value=0
    )
    report.index.name = "Dataset_ID"
    return report


def common_cohort_mask(
    frame: pd.DataFrame, swir_features: Sequence[str], tir_features: Sequence[str]
) -> pd.Series:
    """Rows with at least one usable (finite) SWIR and one usable TIR descriptor.

    Partially missing descriptors are kept; NaN just does not count as usable.
    """
    validate_descriptors(frame, [*swir_features, *tir_features])
    has_swir = np.isfinite(frame[list(swir_features)]).any(axis=1)
    has_tir = np.isfinite(frame[list(tir_features)]).any(axis=1)
    return has_swir & has_tir


def assert_aligned(
    frames: Sequence[pd.DataFrame],
    key_columns: Sequence[str] = SAMPLE_KEY_COLUMNS,
    target: str = TARGET,
) -> None:
    """Require identical sample keys, row order and targets across frames."""
    first = frames[0].reset_index(drop=True)
    for number, other in enumerate(frames[1:], start=1):
        other = other.reset_index(drop=True)
        if not other[list(key_columns)].equals(first[list(key_columns)]):
            raise AssertionError(f"Frame {number} differs in sample keys or row order")
        if not other[target].equals(first[target]):
            raise AssertionError(f"Frame {number} differs in {target} targets")


def build_dataset(
    logs_dir: Path = MINERAL_LOGS_DIR,
    csv_dir: Path = CSV_DIR,
    dataset_ids: Sequence[str] = COHORT_DATASET_IDS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the log + Au dataset.

    Returns ``(dataset, exclusion_report, input_issues)``: per-hole log-row counts
    by outcome, and the malformed/infinite numeric inputs that were set to NaN.
    """
    dataset_ids = tuple(dataset_ids)
    if len(set(dataset_ids)) != len(dataset_ids):
        raise ValueError("Duplicate Dataset IDs in cohort")
    index_path = csv_dir / "HyLogger_Dataset_Index.csv"
    metadata = load_dataset_metadata(index_path, dataset_ids)
    if not metadata["CollarId"].is_unique:
        raise ValueError("Several cohort datasets share one CollarId")

    tables = [read_mineral_log(logs_dir / f"SM_{d}.csv", d) for d in dataset_ids]
    issues = [
        {"Source": "mineral_log", "Dataset_ID": d, **issue}
        for d, table in zip(dataset_ids, tables, strict=True)
        for issue in table.attrs["numeric_issues"]
    ]
    logs = pd.concat(tables, ignore_index=True)
    logs.attrs.clear()
    logs["CollarId"] = logs["Dataset_ID"].map(metadata["CollarId"])
    logs["Project"] = logs["Dataset_ID"].map(metadata["Project"])
    if logs.duplicated(SAMPLE_KEY_COLUMNS).any():
        raise ValueError("Duplicate depth intervals within a mineral log")

    numeric_assay_columns = ["FromDepth", "ToDepth", TARGET]
    assays = pd.read_csv(
        csv_dir / "WAMEX_AssayFlatData.csv",
        usecols=["AssayFlatDataId", "CollarId", "CompanySampleId"]
        + numeric_assay_columns,
        dtype={
            "CompanySampleId": "string",
            **dict.fromkeys(numeric_assay_columns, str),
        },
        low_memory=False,
    )
    assays = assays[assays["CollarId"].isin(metadata["CollarId"])]
    assays, assay_issues = to_numeric_checked(assays, numeric_assay_columns)
    issues += [
        {"Source": "assay", "Dataset_ID": "(all cohort assays)", **issue}
        for issue in assay_issues
    ]
    assays = assays.rename(
        columns={"FromDepth": "Assay_FromDepth", "ToDepth": "Assay_ToDepth"}
    )

    matched, outcome = match_assays(logs, assays)
    invalid = ~valid_interval(assays["Assay_FromDepth"], assays["Assay_ToDepth"])
    dataset_by_collar = pd.Series(metadata.index, index=metadata["CollarId"])
    assay_invalid = (
        assays.loc[invalid, "CollarId"].map(dataset_by_collar).value_counts()
    )

    features = [c for c in matched.columns if c not in LEADING_COLUMNS]
    dataset = matched[LEADING_COLUMNS + features].sort_values(
        ["Dataset_ID", DEPTH_FROM], ignore_index=True
    )
    report = summarise_exclusions(outcome, metadata, assay_invalid)
    log_counts = logs["Dataset_ID"].value_counts().reindex(report.index)
    if (report["log_rows"] != log_counts).any() or report["retained"].sum() != len(
        dataset
    ):
        raise AssertionError("Exclusion counts do not reconcile with the dataset")
    return dataset, report, pd.DataFrame(issues, columns=ISSUE_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    dataset, report, issues = build_dataset()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output, index=False)
    print(report.to_string())
    if not issues.empty:
        print(issues.to_string())
    print(f"{len(dataset)} rows, {dataset['Dataset_ID'].nunique()} datasets")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
