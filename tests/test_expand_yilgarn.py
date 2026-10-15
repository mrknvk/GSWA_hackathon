import sys
from unittest.mock import Mock

import pandas as pd
import pytest

from gswa_hackathon import acquire_mineral_logs as acquisition
from gswa_hackathon import expand_yilgarn as expansion
from gswa_hackathon.build_dataset import HOLE_AUDIT_COLUMNS, ISSUE_COLUMNS


@pytest.fixture(autouse=True)
def prevent_real_acquisition_and_build(monkeypatch):
    def unexpected_call(*args, **kwargs):
        pytest.fail("Real acquisition, builder, or network must not run")

    monkeypatch.setattr(acquisition, "urlopen", unexpected_call)
    monkeypatch.setattr(expansion, "acquire_logs", unexpected_call)
    monkeypatch.setattr(expansion, "build_expanded_dataset", unexpected_call)


@pytest.fixture
def cli_inventory(tmp_path, monkeypatch):
    inventory = pd.DataFrame(
        {
            "CollarId": [1, 2, 3, 4, 5, 6],
            "Project": ["P", "P", "Unknown", "P", "Unknown", "P"],
            "mapping_status": [
                "unambiguous",
                "unambiguous",
                "no_hylogger",
                "ambiguous_mapping",
                "unambiguous",
                "no_mineral_log",
            ],
            "has_mineral_log": [True, True, False, True, True, False],
            "measured_records": [2, 3, 4, 5, 6, 7],
            "anomalies_ge_1ppm": [1, 2, 3, 4, 5, 6],
        }
    )
    sources = pd.DataFrame(
        [
            ("NEW", 1, "P", "https://example.invalid/new.csv"),
            ("CACHED", 2, "P", "https://example.invalid/cached.csv"),
            ("FAILED", 5, "Unknown", "https://example.invalid/failed.csv"),
        ],
        columns=["Dataset_ID", "CollarId", "Project", "source_url"],
    )
    inventory_builder = Mock(return_value=(inventory, sources))
    plot = Mock()
    monkeypatch.setattr(expansion, "build_inventory", inventory_builder)
    monkeypatch.setattr(expansion, "plot_inventory", plot)
    csv_dir = tmp_path / "synthetic_csv"
    output = tmp_path / "outputs"
    logs = tmp_path / "synthetic_logs"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "expand_yilgarn",
            "--csv-dir",
            str(csv_dir),
            "--output-dir",
            str(output),
            "--logs-dir",
            str(logs),
        ],
    )
    return inventory, sources, inventory_builder, plot, csv_dir, output, logs


def test_inventory_only_writes_inventory_and_sources_without_acquisition(
    cli_inventory, capsys
):
    inventory, sources, inventory_builder, plot, csv_dir, output, logs = cli_inventory

    expansion.main()

    inventory_builder.assert_called_once_with(csv_dir)
    assert plot.call_count == 1
    assert plot.call_args.args[0] is inventory
    assert plot.call_args.args[1] == output / "yilgarn_gold_map.png"
    pd.testing.assert_frame_equal(pd.read_csv(output / "hole_inventory.csv"), inventory)
    pd.testing.assert_frame_equal(
        pd.read_csv(output / "candidate_sources.csv"), sources
    )
    assert {path.name for path in output.iterdir()} == {
        "hole_inventory.csv",
        "candidate_sources.csv",
    }
    assert not logs.exists()
    printed = capsys.readouterr().out
    assert "Measured Au: 6 collars" in printed
    assert "Mineral-log links: 4 collars" in printed
    assert "Unambiguous acquisition candidates: 3 collars" in printed
    assert "Use --download" in printed


@pytest.mark.parametrize("all_failed", [False, True], ids=["mixed", "all-failed"])
def test_download_routes_eligible_manifest_and_preserves_full_hole_audit(
    cli_inventory, monkeypatch, all_failed, capsys
):
    inventory, sources, _, _, csv_dir, output, logs = cli_inventory
    monkeypatch.setattr(
        sys,
        "argv",
        [*sys.argv, "--download", "--workers", "2", "--timeout-seconds", "7.5"],
    )
    output.mkdir()
    pilot = output / "ml_dataset.csv"
    pilot.write_bytes(b"original pilot must remain unchanged\n")
    manifest = sources.copy()
    manifest["local_path"] = [str(logs / f"remote_{hole}.csv") for hole in (1, 2, 5)]
    manifest["status"] = (
        ["failed"] * 3 if all_failed else ["downloaded", "already_present", "failed"]
    )
    manifest["error"] = [
        "synthetic timeout" if status == "failed" else "" for status in manifest.status
    ]
    acquire = Mock(return_value=manifest)
    monkeypatch.setattr(expansion, "acquire_logs", acquire)
    eligible = manifest.loc[manifest.status.isin(["downloaded", "already_present"])]
    dataset = pd.DataFrame(
        [] if all_failed else [("NEW", 1, "P", 1.0), ("CACHED", 2, "P", 0.5)],
        columns=["Dataset_ID", "CollarId", "Project", "Au_PPM"],
    )
    audit_rows = [
        {
            "Dataset_ID": row.Dataset_ID,
            "CollarId": row.CollarId,
            "Project": row.Project,
            "status": "ready",
            "error": "",
            "common_cohort_rows": 1,
            "anomalies_ge_1ppm": int(row.CollarId == 1),
        }
        for row in eligible.itertuples()
    ]
    build_audit = pd.DataFrame(audit_rows, columns=HOLE_AUDIT_COLUMNS)
    issues = pd.DataFrame(
        [] if all_failed else [("assay", "NEW", "Au_PPM", 1, 0)],
        columns=ISSUE_COLUMNS,
    )
    builder = Mock(return_value=(dataset, build_audit, issues))
    monkeypatch.setattr(expansion, "build_expanded_dataset", builder)

    expansion.main()

    assert acquire.call_count == 1
    pd.testing.assert_frame_equal(acquire.call_args.args[0], sources)
    assert acquire.call_args.args[1] == logs
    assert acquire.call_args.kwargs == {"max_workers": 2, "timeout_seconds": 7.5}
    assert builder.call_count == 1
    pd.testing.assert_frame_equal(builder.call_args.args[0], eligible)
    assert builder.call_args.args[1:] == (logs, csv_dir)
    assert {"Dataset_ID", "CollarId", "Project", "local_path"} <= set(eligible.columns)
    pd.testing.assert_frame_equal(
        pd.read_csv(output / "ml_dataset_expanded.csv"), dataset, check_dtype=False
    )
    pd.testing.assert_frame_equal(
        pd.read_csv(output / "numeric_issues.csv"), issues, check_dtype=False
    )
    saved_manifest = pd.read_csv(
        output / "mineral_log_manifest.csv", keep_default_na=False
    )
    pd.testing.assert_frame_equal(saved_manifest, manifest)
    saved_build = pd.read_csv(output / "build_audit.csv")
    assert saved_build.columns.tolist() == HOLE_AUDIT_COLUMNS
    assert saved_build.CollarId.tolist() == eligible.CollarId.tolist()
    audit = pd.read_csv(output / "hole_audit.csv").set_index("CollarId")
    assert audit.index.tolist() == inventory.CollarId.tolist()
    assert audit.index.is_unique
    assert audit.status.tolist() == [
        "download_failed" if all_failed else "ready",
        "download_failed" if all_failed else "ready",
        "no_hylogger",
        "ambiguous_mapping",
        "download_failed",
        "no_mineral_log",
    ]
    assert audit.loc[[3, 4, 6], "download_status"].eq("not_requested").all()
    assert audit.loc[5, "download_error"] == "synthetic timeout"
    assert audit.loc[[3, 4, 5, 6], "common_cohort_rows"].isna().all()
    summary = pd.read_csv(output / "project_summary.csv").set_index("Project")
    assert summary.inventory_holes.to_dict() == {"P": 4, "Unknown": 2}
    assert summary.measured_assay_records.to_dict() == {"P": 17, "Unknown": 10}
    assert summary.source_anomalies.to_dict() == {"P": 13, "Unknown": 8}
    assert summary.ready_holes.sum() == (0 if all_failed else 2)
    assert summary.model_intervals.sum() == len(dataset)
    assert summary.model_anomalies.sum() == (0 if all_failed else 1)
    assert pilot.read_bytes() == b"original pilot must remain unchanged\n"
    assert not csv_dir.exists()
    assert not logs.exists()
    assert "Pilot dataset unchanged. No models trained." in capsys.readouterr().out
