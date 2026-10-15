"""Inventory measured Yilgarn Au assays and unambiguous HyLogger sources."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gswa_hackathon.build_dataset import (
    COHORT_DATASET_IDS,
    CSV_DIR,
    to_numeric_checked,
    valid_interval,
)

ANOMALY_AU_PPM = 1.0
SOURCE_COLUMNS = ["Dataset_ID", "CollarId", "Project", "source_url"]


def build_inventory(csv_dir: Path = CSV_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return one row per measured-Au collar and safe sources, without writing.

    Counts describe source assay records before spectral matching, not unique
    intervals or mineral resources. Negative/censored Au is not substituted.
    Multiple names, URLs or project mappings are audited rather than resolved
    by arbitrarily selecting an index record.
    """
    collars = pd.read_csv(
        csv_dir / "WAMEX_Collar.csv",
        usecols=[
            "CollarId",
            "CompanyHoleId",
            "TectonicRegion",
            "Longitude",
            "Latitude",
        ],
    )
    collars = collars.loc[collars["TectonicRegion"] == "Yilgarn Craton"].copy()
    if not collars["CollarId"].is_unique:
        raise ValueError("Duplicate WAMEX collar keys")

    assays = pd.read_csv(
        csv_dir / "WAMEX_AssayFlatData.csv",
        usecols=["CollarId", "FromDepth", "ToDepth", "Au_PPM"],
    )
    assays = assays.loc[assays["CollarId"].isin(collars["CollarId"])]
    assays, _ = to_numeric_checked(assays, ["FromDepth", "ToDepth", "Au_PPM"])
    assays["negative_au"] = assays["Au_PPM"] < 0
    assays["missing_au"] = ~np.isfinite(assays["Au_PPM"])
    assay_counts = assays.groupby("CollarId").agg(
        assay_records=("CollarId", "size"),
        negative_au_records=("negative_au", "sum"),
        missing_au_records=("missing_au", "sum"),
    )
    is_measured = (
        np.isfinite(assays["Au_PPM"])
        & assays["Au_PPM"].ge(0)
        & valid_interval(assays["FromDepth"], assays["ToDepth"])
    )
    measured = assays.loc[is_measured].copy()
    measured["is_anomaly"] = measured["Au_PPM"] >= ANOMALY_AU_PPM
    gold_summary = measured.groupby("CollarId").agg(
        measured_records=("Au_PPM", "size"),
        median_au_ppm=("Au_PPM", "median"),
        max_au_ppm=("Au_PPM", "max"),
        anomalies_ge_1ppm=("is_anomaly", "sum"),
    )
    inventory = collars.merge(gold_summary, on="CollarId", validate="one_to_one")
    inventory = inventory.merge(assay_counts, on="CollarId", validate="one_to_one")
    inventory["anomaly_fraction"] = (
        inventory["anomalies_ge_1ppm"] / inventory["measured_records"]
    )

    index = pd.read_csv(
        csv_dir / "HyLogger_Dataset_Index.csv",
        usecols=["Data_name", "WAMEX_CollarId", "PROJECT", "MINERALLOG"],
    ).rename(
        columns={
            "Data_name": "Dataset_ID",
            "WAMEX_CollarId": "CollarId",
            "PROJECT": "Project",
            "MINERALLOG": "source_url",
        }
    )
    index["CollarId"] = pd.to_numeric(index["CollarId"], errors="coerce")
    for column in ["Dataset_ID", "Project", "source_url"]:
        index[column] = index[column].fillna("").str.strip()
    index["Project"] = index["Project"].replace("", "Unknown")
    index = index.drop_duplicates(SOURCE_COLUMNS)
    pilot_collars = set(
        index.loc[index["Dataset_ID"].isin(COHORT_DATASET_IDS), "CollarId"]
    )
    name_collar_counts = index.groupby("Dataset_ID")["CollarId"].nunique()
    repeated_names = set(name_collar_counts.index[name_collar_counts > 1])
    url_collar_counts = (
        index.loc[index["source_url"].ne("")]
        .groupby("source_url")["CollarId"]
        .nunique()
    )
    repeated_urls = set(url_collar_counts.index[url_collar_counts > 1])
    # Identity conflicts outside Yilgarn or without valid Au must still count.
    index = index.loc[index["CollarId"].isin(inventory["CollarId"])].copy()

    mappings = []
    source_rows = []
    for collar_id, entries in index.groupby("CollarId", sort=True):
        linked = entries.loc[entries["source_url"].ne("")]
        mapping = {
            "CollarId": collar_id,
            "Project": " | ".join(sorted(entries["Project"].unique())),
            "Dataset_IDs": " | ".join(sorted(entries["Dataset_ID"].unique())),
            "has_hylogger": True,
            "has_mineral_log": not linked.empty,
            "mapping_status": "no_mineral_log",
        }
        if not linked.empty:
            names_are_unique = not linked["Dataset_ID"].isin(repeated_names).any()
            urls_are_unique = not linked["source_url"].isin(repeated_urls).any()
            has_consistent_metadata = (
                entries["Dataset_ID"].nunique() == 1
                and entries["Project"].nunique() == 1
            )
            is_unambiguous = (
                len(linked) == 1
                and names_are_unique
                and urls_are_unique
                and has_consistent_metadata
                and linked["Dataset_ID"].iloc[0] != ""
            )
            mapping["mapping_status"] = (
                "unambiguous" if is_unambiguous else "ambiguous_mapping"
            )
            if is_unambiguous:
                source_rows.append(linked.iloc[0][SOURCE_COLUMNS].to_dict())
        mappings.append(mapping)

    mapping_columns = [
        "CollarId",
        "Project",
        "Dataset_IDs",
        "has_hylogger",
        "has_mineral_log",
        "mapping_status",
    ]
    inventory = inventory.merge(
        pd.DataFrame(mappings, columns=mapping_columns),
        on="CollarId",
        how="left",
        validate="one_to_one",
    )
    inventory["Project"] = inventory["Project"].fillna("Unknown")
    inventory["Dataset_IDs"] = inventory["Dataset_IDs"].fillna("")
    for column in ["has_hylogger", "has_mineral_log"]:
        inventory[column] = inventory[column].eq(True)
    inventory["mapping_status"] = inventory["mapping_status"].fillna("no_hylogger")
    inventory["is_pilot"] = inventory["CollarId"].isin(pilot_collars)
    inventory["Coordinate_CRS"] = "GDA2020 / EPSG:7844"
    inventory = inventory.sort_values("CollarId", ignore_index=True)
    sources = pd.DataFrame(source_rows, columns=SOURCE_COLUMNS)
    sources["CollarId"] = sources["CollarId"].astype("int64")
    return inventory, sources


def plot_inventory(inventory: pd.DataFrame, output_path: Path) -> None:
    """Save a GDA2020 collar-location map, without tiles or coordinate conversion."""
    valid_coordinates = inventory["Longitude"].between(-180, 180) & inventory[
        "Latitude"
    ].between(-90, 90)
    if not valid_coordinates.all():
        raise ValueError("Missing or out-of-range GDA2020 coordinates")
    fig, axes = plt.subplots(1, 3, figsize=(17, 7), layout="constrained")
    panels = [
        ("median_au_ppm", "Median measured Au", True),
        ("max_au_ppm", "Maximum measured Au", True),
        ("anomaly_fraction", "Fraction of assays >= 1 ppm", False),
    ]
    linked = inventory.loc[inventory["has_mineral_log"]]
    pilot = inventory.loc[inventory["is_pilot"]]
    aspect = 1 / np.cos(np.deg2rad(inventory["Latitude"].mean()))
    for axis, (column, title, use_log) in zip(axes, panels, strict=True):
        colours = np.log1p(inventory[column]) if use_log else inventory[column]
        points = axis.scatter(
            inventory["Longitude"],
            inventory["Latitude"],
            c=colours,
            cmap="viridis",
            s=24,
            alpha=0.8,
            vmin=0,
            vmax=None if use_log else 1,
        )
        axis.scatter(
            linked["Longitude"],
            linked["Latitude"],
            s=37,
            facecolors="none",
            edgecolors="#333333",
            linewidths=0.5,
            label=f"Mineral-log link (n={len(linked)})",
        )
        axis.scatter(
            pilot["Longitude"],
            pilot["Latitude"],
            s=75,
            marker="*",
            facecolors="none",
            edgecolors="#d33c29",
            linewidths=1,
            label=f"Original pilot (n={len(pilot)})",
        )
        axis.set(
            xlabel="Longitude (GDA2020 degrees)",
            ylabel="Latitude (GDA2020 degrees)",
            title=title,
            aspect=aspect,
        )
        axis.grid(alpha=0.2)
        axis.legend(loc="lower left", fontsize=7)
        colorbar = fig.colorbar(points, ax=axis, shrink=0.7)
        colorbar.set_label("log1p(Au ppm)" if use_log else "Assay-record fraction")
    fig.suptitle(
        f"Yilgarn collar inventory: {len(inventory)} holes with measured Au\n"
        "Before spectral matching; sampled collars, not resource estimates "
        "or a craton boundary"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
