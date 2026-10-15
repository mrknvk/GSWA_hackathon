"""Inventory, acquire and prepare an expanded Yilgarn cohort, without training."""

import argparse
from pathlib import Path

from gswa_hackathon.acquire_mineral_logs import acquire_logs
from gswa_hackathon.build_dataset import (
    CSV_DIR,
    MINERAL_LOGS_DIR,
    PROJECT_ROOT,
    build_expanded_dataset,
)
from gswa_hackathon.yilgarn_inventory import build_inventory, plot_inventory

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/extracted_data/yilgarn"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir", type=Path, default=CSV_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--logs-dir", type=Path, default=MINERAL_LOGS_DIR)
    parser.add_argument(
        "--download", action="store_true", help="Acquire logs and build"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=30)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    inventory, sources = build_inventory(args.csv_dir)
    inventory.to_csv(args.output_dir / "hole_inventory.csv", index=False)
    sources.to_csv(args.output_dir / "candidate_sources.csv", index=False)
    plot_inventory(inventory, args.output_dir / "yilgarn_gold_map.png")
    print(f"Measured Au: {len(inventory)} collars")
    print(f"Mineral-log links: {inventory['has_mineral_log'].sum()} collars")
    print(f"Unambiguous acquisition candidates: {len(sources)} collars")
    if not args.download:
        print(
            "Inventory/map ready. Use --download to acquire logs and build the cohort."
        )
        return

    manifest = acquire_logs(
        sources,
        args.logs_dir,
        max_workers=args.workers,
        timeout_seconds=args.timeout_seconds,
    )
    manifest.to_csv(args.output_dir / "mineral_log_manifest.csv", index=False)
    print(manifest["status"].value_counts().to_string())
    downloaded = manifest.loc[
        manifest["status"].isin(["downloaded", "already_present"])
    ]
    dataset, build_audit, issues = build_expanded_dataset(
        downloaded, args.logs_dir, args.csv_dir
    )
    dataset.to_csv(args.output_dir / "ml_dataset_expanded.csv", index=False)
    build_audit.to_csv(args.output_dir / "build_audit.csv", index=False)
    issues.to_csv(args.output_dir / "numeric_issues.csv", index=False)

    acquisition_status = manifest[["CollarId", "status", "error"]].rename(
        columns={
            "status": "download_status",
            "error": "download_error",
        }
    )
    build_status = build_audit.drop(columns=["Dataset_ID", "Project"]).rename(
        columns={"anomalies_ge_1ppm": "model_anomalies_ge_1ppm"}
    )
    audit = inventory.merge(
        acquisition_status, on="CollarId", how="left", validate="one_to_one"
    ).merge(build_status, on="CollarId", how="left", validate="one_to_one")
    audit["download_status"] = audit["download_status"].fillna("not_requested")
    audit["status"] = audit["status"].fillna(audit["mapping_status"])
    failed_download = audit["download_status"].eq("failed")
    audit.loc[failed_download, "status"] = "download_failed"
    audit.to_csv(args.output_dir / "hole_audit.csv", index=False)

    project_summary = (
        audit.assign(ready=audit["status"].eq("ready"))
        .groupby("Project", dropna=False)
        .agg(
            inventory_holes=("CollarId", "size"),
            ready_holes=("ready", "sum"),
            measured_assay_records=("measured_records", "sum"),
            source_anomalies=("anomalies_ge_1ppm", "sum"),
            model_intervals=("common_cohort_rows", "sum"),
            model_anomalies=("model_anomalies_ge_1ppm", "sum"),
        )
    )
    project_summary.to_csv(args.output_dir / "project_summary.csv")
    print("\nAcquisition/build outcomes across all measured-Au collars:")
    print(audit["status"].value_counts().to_string())
    print(f"\nExpanded: {len(dataset)} rows from {dataset['CollarId'].nunique()} holes")
    print(f"Au >= 1 ppm: {dataset['Au_PPM'].ge(1).sum()} intervals")
    if "is_pilot" in audit:
        additional_ready = audit["status"].eq("ready") & ~audit["is_pilot"]
        print(f"Additional ready holes beyond pilot: {additional_ready.sum()}")
        if not additional_ready.any():
            print("WARNING: this run did not add usable holes beyond the pilot.")
    print(f"Local results: {args.output_dir}")
    print("Pilot dataset unchanged. No models trained.")


if __name__ == "__main__":
    main()
