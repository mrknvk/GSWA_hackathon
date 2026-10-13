"""Merge mineral logs with WAMEX Au assays by Dataset_ID and depth."""

from __future__ import annotations

import argparse
from pathlib import Path

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
LEADING_COLUMNS = [
    "Dataset_ID",
    "CollarId",
    DEPTH_FROM,
    DEPTH_TO,
    "Assay_FromDepth",
    "Assay_ToDepth",
    "Au_PPM",
]


def read_mineral_log(path: Path) -> pd.DataFrame:
    """Read one SM_*.csv: metadata block on top, data table below the depth header."""
    with path.open(encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    dataset_id = next(
        line.split(",")[1] for line in lines if line.startswith("Dataset,")
    )
    header_row = next(
        i for i, line in enumerate(lines) if line.startswith(DEPTH_FROM)
    )
    table = pd.read_csv(path, skiprows=header_row, na_values=["N.A."])
    table.insert(0, "Dataset_ID", dataset_id)
    return table


def load_dataset_to_collar(index_path: Path) -> pd.Series:
    """Dataset_ID -> WAMEX CollarId, taken from the HyLogger dataset index."""
    index = pd.read_csv(
        index_path,
        usecols=["Data_name", "WAMEX_CollarId"],
        low_memory=False,
    )
    index = index.dropna(subset=["WAMEX_CollarId"])
    index["CollarId"] = index["WAMEX_CollarId"].astype("int64")
    return index.drop_duplicates("Data_name").set_index("Data_name")["CollarId"]


def match_assays(logs: pd.DataFrame, assays: pd.DataFrame) -> pd.DataFrame:
    """Attach the Au assay whose interval contains the midpoint of each log row."""
    logs = logs.copy()
    logs["_mid"] = (logs[DEPTH_FROM] + logs[DEPTH_TO]) / 2
    logs["_row"] = range(len(logs))
    merged = logs.merge(assays, on="CollarId", how="inner")
    inside = (merged["_mid"] >= merged["Assay_FromDepth"]) & (
        merged["_mid"] < merged["Assay_ToDepth"]
    )
    merged = merged[inside].copy()
    # Overlapping assay intervals: keep the narrowest one per log row.
    merged["_width"] = merged["Assay_ToDepth"] - merged["Assay_FromDepth"]
    merged = merged.sort_values(["_row", "_width"]).drop_duplicates("_row")
    return merged.drop(columns=["_mid", "_row", "_width"])


def build_dataset(
    logs_dir: Path = MINERAL_LOGS_DIR, csv_dir: Path = CSV_DIR
) -> pd.DataFrame:
    logs = pd.concat(
        [read_mineral_log(p) for p in sorted(logs_dir.glob("SM_*.csv"))],
        ignore_index=True,
    )
    collar_by_dataset = load_dataset_to_collar(csv_dir / "HyLogger_Dataset_Index.csv")
    logs["CollarId"] = logs["Dataset_ID"].map(collar_by_dataset)
    missing = logs.loc[logs["CollarId"].isna(), "Dataset_ID"].unique()
    if len(missing):
        raise ValueError(f"No CollarId in HyLogger index for: {list(missing)}")
    logs["CollarId"] = logs["CollarId"].astype("int64")

    assays = pd.read_csv(
        csv_dir / "WAMEX_AssayFlatData.csv",
        usecols=["CollarId", "FromDepth", "ToDepth", "Au_PPM"],
        low_memory=False,
    )
    assays = assays[
        assays["CollarId"].isin(logs["CollarId"].unique()) & assays["Au_PPM"].notna()
    ].rename(columns={"FromDepth": "Assay_FromDepth", "ToDepth": "Assay_ToDepth"})

    dataset = match_assays(logs, assays)
    features = [c for c in dataset.columns if c not in LEADING_COLUMNS]
    return dataset[LEADING_COLUMNS + features].sort_values(
        ["Dataset_ID", DEPTH_FROM], ignore_index=True
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    dataset = build_dataset()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output, index=False)
    print(f"{len(dataset)} rows, {dataset['Dataset_ID'].nunique()} datasets")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
