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
import csv
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
SPECTRAL_FEATURES_SWIR = [
    "900D_Fe",
    "900W_Fe",
    "1900D_Water",
    "2200D_AlOH",
    "2200W_AlOH",
    "2250D_FeOH",
    "2250W_FeOH",
]
SPECTRAL_FEATURES_TIR = ["8625D_Quartz", "11300D_Carbonate", "11300W_Carbonate"]

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
HOLE_AUDIT_COLUMNS = [
    "Dataset_ID",
    "CollarId",
    "Project",
    "status",
    "error",
    "log_rows",
    "retained_match_rows",
    "common_cohort_rows",
    "anomalies_ge_1ppm",
    "no_usable_swir",
    "no_usable_tir",
    *STATUSES[1:],
    "duplicates_removed",
    "assay_invalid_interval",
]


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
    dataset_ids = []
    header = None
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, strict=True)
            for header_row, row in enumerate(reader):
                if len(row) > 1 and row[0] == "Dataset":
                    dataset_ids.append(row[1].strip())
                if row and row[0] == DEPTH_FROM:
                    header = row
                    break
    except csv.Error as exc:
        raise ValueError(f"{path.name} has malformed CSV metadata: {exc}") from exc
    if len(dataset_ids) != 1 or not dataset_ids[0]:
        raise ValueError(
            f"{path.name} requires one declared Dataset, found {dataset_ids!r}"
        )
    dataset_id = dataset_ids[0]
    if expected_id is not None and dataset_id != expected_id:
        raise ValueError(
            f"{path.name} declares Dataset {dataset_id!r}, expected {expected_id!r}"
        )
    if header is None:
        raise ValueError(
            f"{path.name} has no depth header starting with {DEPTH_FROM!r}"
        )
    required = [DEPTH_FROM, DEPTH_TO, *SPECTRAL_FEATURES_SWIR, *SPECTRAL_FEATURES_TIR]
    duplicates = [column for column in required if header.count(column) > 1]
    if duplicates:
        raise ValueError(
            f"{path.name} has duplicate required column headers: {duplicates}"
        )
    table = pd.read_csv(
        path, skiprows=header_row, na_values=["N.A."], dtype=str, encoding="utf-8-sig"
    )
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

    Candidates use sorted start-depth windows per collar, not a full crossproduct.
    Both boundaries are then checked with the original absolute comparisons.

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
    assays = assays.reset_index(drop=True).assign(
        _assay_row=lambda frame: np.arange(len(frame))
    )
    assay_groups = {
        None if pd.isna(collar) else collar: group.sort_values(
            "Assay_FromDepth", kind="stable"
        )
        for collar, group in assays.groupby("CollarId", sort=False, dropna=False)
    }
    candidate_tables = []
    # Widen the tolerance for subtraction rounding, then round search bounds
    # outward for addition rounding. Only the original comparisons below decide
    # membership; these conservative windows must not discard boundary matches.
    search_atol = np.nextafter(DEPTH_ATOL_M, np.inf)
    for collar, hole_logs in logs[valid_log].groupby(
        "CollarId", sort=False, dropna=False
    ):
        hole_assays = assay_groups.get(None if pd.isna(collar) else collar)
        if hole_assays is None:
            continue
        starts = hole_logs[DEPTH_FROM].to_numpy()
        assay_starts = hole_assays["Assay_FromDepth"].to_numpy()
        left = np.searchsorted(
            assay_starts, np.nextafter(starts - search_atol, -np.inf), side="left"
        )
        right = np.searchsorted(
            assay_starts, np.nextafter(starts + search_atol, np.inf), side="right"
        )
        counts = right - left
        pair_count = int(counts.sum())
        if not pair_count:
            continue
        offsets = np.cumsum(counts) - counts
        assay_positions = (
            np.repeat(left, counts) + np.arange(pair_count) - np.repeat(offsets, counts)
        )
        selected_logs = hole_logs.iloc[
            np.repeat(np.arange(len(hole_logs)), counts)
        ].copy()
        selected_logs["_assay_row"] = hole_assays["_assay_row"].to_numpy()[
            assay_positions
        ]
        candidate_tables.append(
            selected_logs.merge(hole_assays, on=["CollarId", "_assay_row"])
        )
    candidates = (
        pd.concat(candidate_tables, ignore_index=True)
        if candidate_tables
        else logs.iloc[:0].merge(assays.iloc[:0], on="CollarId")
    ).drop(columns="_assay_row")
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


def build_expanded_dataset(
    sources: pd.DataFrame, logs_dir: Path, csv_dir: Path = CSV_DIR
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the common SWIR/TIR cohort from an unambiguous source manifest.

    ``local_path`` is authoritative, absolute or relative to PROJECT_ROOT, not
    inferred from Dataset_ID or the download directory ``logs_dir``. Project
    must be explicit (including the caller's chosen unknown-project label).
    Returns dataset, one audit row per source, and ISSUE_COLUMNS numeric issues.
    Coverage exclusions count assay-retained rows; anomalies count final rows.
    Unavailable audit counts remain missing for unreadable/incompatible logs.
    """
    required = ["Dataset_ID", "CollarId", "Project", "local_path"]
    missing = [column for column in required if column not in sources.columns]
    if missing:
        raise ValueError(f"Missing source columns: {missing}")
    if sources[required].isna().any().any():
        raise ValueError("Source mapping fields must not be missing")
    for column in ["Dataset_ID", "CollarId"]:
        if sources[column].duplicated().any():
            raise ValueError(f"Duplicate {column} in source mapping")
    for column in ["Dataset_ID", "Project"]:
        if (
            not sources[column]
            .map(lambda value: isinstance(value, str) and bool(value.strip()))
            .all()
        ):
            raise ValueError(f"Source {column} must contain explicit nonempty strings")

    features = SPECTRAL_FEATURES_SWIR + SPECTRAL_FEATURES_TIR
    output_columns = LEADING_COLUMNS + features
    if sources.empty:
        return (
            pd.DataFrame(columns=output_columns),
            pd.DataFrame(columns=HOLE_AUDIT_COLUMNS),
            pd.DataFrame(columns=ISSUE_COLUMNS),
        )

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
    assays = assays[assays["CollarId"].isin(sources["CollarId"])]
    assays, assay_issues = to_numeric_checked(assays, numeric_assay_columns)
    issues = [
        {"Source": "assay", "Dataset_ID": "(all cohort assays)", **issue}
        for issue in assay_issues
    ]
    assays = assays.rename(
        columns={"FromDepth": "Assay_FromDepth", "ToDepth": "Assay_ToDepth"}
    )
    assays_by_collar = dict(tuple(assays.groupby("CollarId", sort=False)))
    empty_assays = assays.iloc[:0]

    tables = []
    audit_rows = []
    for source in sources[required].itertuples(index=False):
        dataset_id, collar_id, project, local_path = source
        audit = dict.fromkeys(HOLE_AUDIT_COLUMNS, pd.NA)
        audit.update(
            Dataset_ID=dataset_id,
            CollarId=collar_id,
            Project=project,
            status="invalid_log",
            error="",
        )
        audit_rows.append(audit)
        hole_assays = assays_by_collar.get(collar_id, empty_assays)
        audit["assay_invalid_interval"] = int(
            (
                ~valid_interval(
                    hole_assays["Assay_FromDepth"], hole_assays["Assay_ToDepth"]
                )
            ).sum()
        )
        try:
            path = Path(local_path)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            logs = read_mineral_log(path, expected_id=dataset_id)
        except (
            OSError,
            UnicodeError,
            ValueError,
            TypeError,
            StopIteration,
            IndexError,
        ) as exc:
            audit["error"] = f"{type(exc).__name__}: {exc}"
            continue

        audit["log_rows"] = len(logs)
        issues.extend(
            {"Source": "mineral_log", "Dataset_ID": dataset_id, **issue}
            for issue in logs.attrs["numeric_issues"]
        )
        missing = [c for c in [DEPTH_FROM, DEPTH_TO, *features] if c not in logs]
        if missing:
            audit.update(
                status="incompatible_schema",
                error=f"Missing required columns: {missing}",
            )
            continue
        if logs.duplicated(SAMPLE_KEY_COLUMNS).any():
            audit["error"] = "Duplicate depth intervals within a mineral log"
            continue

        logs = logs[SAMPLE_KEY_COLUMNS + features].copy()
        logs.attrs.clear()
        logs["CollarId"] = collar_id
        logs["Project"] = project
        matched, outcome = match_assays(logs, hole_assays)
        mask = common_cohort_mask(
            matched, SPECTRAL_FEATURES_SWIR, SPECTRAL_FEATURES_TIR
        )
        common = matched.loc[mask, output_columns]
        tables.append(common)
        counts = outcome["status"].value_counts()
        audit.update({status: int(counts.get(status, 0)) for status in STATUSES[1:]})
        audit.update(
            status="ready" if len(common) else "no_retained_intervals",
            retained_match_rows=len(matched),
            common_cohort_rows=len(common),
            anomalies_ge_1ppm=int((common[TARGET] >= 1).sum()),
            no_usable_swir=int(
                (~np.isfinite(matched[SPECTRAL_FEATURES_SWIR]).any(axis=1)).sum()
            ),
            no_usable_tir=int(
                (~np.isfinite(matched[SPECTRAL_FEATURES_TIR]).any(axis=1)).sum()
            ),
            duplicates_removed=int(outcome["duplicates_removed"].sum()),
        )

    dataset = (
        pd.concat(tables, ignore_index=True)
        if tables
        else pd.DataFrame(columns=output_columns)
    )
    dataset = dataset.sort_values(["Dataset_ID", DEPTH_FROM], ignore_index=True)
    return (
        dataset,
        pd.DataFrame(audit_rows, columns=HOLE_AUDIT_COLUMNS),
        pd.DataFrame(issues, columns=ISSUE_COLUMNS),
    )


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
