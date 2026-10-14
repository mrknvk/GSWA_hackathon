from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gswa_hackathon.build_dataset import (
    COHORT_DATASET_IDS,
    DEPTH_ATOL_M,
    STATUSES,
    assert_aligned,
    build_dataset,
    common_cohort_mask,
    load_dataset_metadata,
    match_assays,
    read_mineral_log,
    to_numeric_checked,
    valid_interval,
    validate_descriptors,
)


def make_logs(*intervals: tuple[float, float], collar: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Dataset_ID": "H1",
            "CollarId": collar,
            "Depth_from_(m)": [a for a, _ in intervals],
            "Depth_to_(m)": [b for _, b in intervals],
            "900D_Fe": 0.1,
        }
    )


def make_assays(*rows: tuple) -> pd.DataFrame:
    """Rows: (AssayFlatDataId, CollarId, from, to, au, CompanySampleId)."""
    return pd.DataFrame(
        rows,
        columns=[
            "AssayFlatDataId",
            "CollarId",
            "Assay_FromDepth",
            "Assay_ToDepth",
            "Au_PPM",
            "CompanySampleId",
        ],
    )


def statuses(outcome: pd.DataFrame) -> list[str]:
    return outcome["status"].tolist()


def test_cohort_is_the_22_unique_holes():
    assert len(COHORT_DATASET_IDS) == 22
    assert len(set(COHORT_DATASET_IDS)) == 22


def test_exact_boundaries_match_and_keep_unrounded_assay_depths():
    logs = make_logs((10.0, 11.0))
    off = DEPTH_ATOL_M / 2
    assays = make_assays((7, 1, 10.0 + off, 11.0 - off, 0.5, "S1"))
    retained, outcome = match_assays(logs, assays)
    assert statuses(outcome) == ["retained"]
    assert retained["Assay_FromDepth"].iloc[0] == 10.0 + off
    assert retained["Assay_ToDepth"].iloc[0] == 11.0 - off
    assert retained["AssayFlatDataId"].iloc[0] == 7


def test_midpoint_inside_a_wider_assay_is_not_a_match():
    logs = make_logs((10.0, 11.0))
    assays = make_assays((1, 1, 10.0, 12.0, 0.5, "S1"))
    retained, outcome = match_assays(logs, assays)
    assert retained.empty
    assert statuses(outcome) == ["no_assay_match"]


def test_each_boundary_is_checked_with_absolute_tolerance_only():
    logs = make_logs((10.0, 11.0), (20.0, 21.0), (30.0, 31.0))
    too_far = DEPTH_ATOL_M * 2
    assays = make_assays(
        (1, 1, 10.0 + too_far, 11.0, 0.5, "A"),
        (2, 1, 20.0, 21.0 + too_far, 0.5, "B"),
        (3, 1, 30.0, 31.0, 0.5, "C"),
    )
    _, outcome = match_assays(logs, assays)
    assert statuses(outcome) == ["no_assay_match", "no_assay_match", "retained"]


def test_other_collar_is_not_matched():
    logs = make_logs((10.0, 11.0), collar=1)
    assays = make_assays((1, 2, 10.0, 11.0, 0.5, "S1"))
    _, outcome = match_assays(logs, assays)
    assert statuses(outcome) == ["no_assay_match"]


def test_genuine_duplicate_is_collapsed_to_lowest_assay_id():
    logs = make_logs((10.0, 11.0))
    assays = make_assays(
        (9, 1, 10.0, 11.0, 0.5, "S1"),
        (4, 1, 10.0, 11.0, 0.5, "S1"),
    )
    retained, outcome = match_assays(logs, assays)
    assert len(retained) == 1
    assert retained["AssayFlatDataId"].iloc[0] == 4
    assert outcome["duplicates_removed"].tolist() == [1]


@pytest.mark.parametrize(
    "second",
    [(2, 1, 10.0, 11.0, 0.7, "S1"), (2, 1, 10.0, 11.0, 0.5, "S2")],
    ids=["different_au", "different_sample_id"],
)
def test_conflicting_assays_are_excluded_not_chosen(second):
    logs = make_logs((10.0, 11.0))
    assays = make_assays((1, 1, 10.0, 11.0, 0.5, "S1"), second)
    retained, outcome = match_assays(logs, assays)
    assert retained.empty
    assert statuses(outcome) == ["conflicting_assays"]


def test_conflict_between_negative_and_positive_is_not_resolved_silently():
    logs = make_logs((10.0, 11.0))
    assays = make_assays(
        (1, 1, 10.0, 11.0, -0.001, "S1"),
        (2, 1, 10.0, 11.0, 0.5, "S1"),
    )
    retained, outcome = match_assays(logs, assays)
    assert retained.empty
    assert statuses(outcome) == ["conflicting_assays"]


def test_invalid_au_and_intervals_are_excluded_zero_is_kept():
    logs = make_logs(
        (1.0, 2.0),  # NaN Au
        (2.0, 3.0),  # infinite Au
        (3.0, 4.0),  # negative Au
        (4.0, 5.0),  # zero Au: valid measurement
        (5.0, 5.0),  # empty log interval
        (np.nan, 7.0),  # missing log depth
        (7.0, 8.0),  # assay interval inverted
    )
    assays = make_assays(
        (1, 1, 1.0, 2.0, np.nan, "A"),
        (2, 1, 2.0, 3.0, np.inf, "B"),
        (3, 1, 3.0, 4.0, -0.001, "C"),
        (4, 1, 4.0, 5.0, 0.0, "D"),
        (5, 1, 5.0, 5.0, 0.1, "E"),
        (6, 1, 7.0, 8.0, 0.1, "F"),
    )
    assays.loc[5, ["Assay_FromDepth", "Assay_ToDepth"]] = [8.0, 7.0]
    retained, outcome = match_assays(logs, assays)
    assert statuses(outcome) == [
        "missing_or_nonfinite_au",
        "missing_or_nonfinite_au",
        "negative_au",
        "retained",
        "invalid_log_interval",
        "invalid_log_interval",
        "no_assay_match",
    ]
    assert retained["Au_PPM"].tolist() == [0.0]


def test_valid_interval_rejects_nonfinite_negative_and_empty():
    start = pd.Series([0.0, -1.0, 1.0, np.nan, 1.0, 2.0])
    end = pd.Series([1.0, 2.0, 1.0, 2.0, np.inf, 1.0])
    expected = [True, False, False, False, False, False]
    assert valid_interval(start, end).tolist() == expected


def test_assert_aligned_accepts_identical_samples_and_rejects_differences():
    base = pd.DataFrame(
        {
            "Dataset_ID": ["A", "A", "B"],
            "Depth_from_(m)": [1.0, 2.0, 1.0],
            "Depth_to_(m)": [2.0, 3.0, 2.0],
            "Au_PPM": [0.1, 0.2, 0.3],
        }
    )
    assert_aligned([base, base.assign(extra=1), base.copy()])

    with pytest.raises(AssertionError, match="keys or row order"):
        assert_aligned([base, base.iloc[[1, 0, 2]]])
    with pytest.raises(AssertionError, match="keys or row order"):
        assert_aligned([base, base.iloc[:2]])
    with pytest.raises(AssertionError, match="Au_PPM"):
        assert_aligned([base, base.assign(Au_PPM=[0.1, 0.2, 0.4])])


def write_log(path, dataset_id: str) -> None:
    path.write_text(
        f"GSWA TSG Summary Mineral Log,,\nDataset,{dataset_id},\n"
        "Depth_from_(m),Depth_to_(m),900D_Fe\n0.0,1.0,N.A.\n",
        encoding="utf-8",
    )


def test_read_mineral_log_parses_table_and_checks_declared_id(tmp_path):
    path = tmp_path / "SM_H1.csv"
    write_log(path, "H1")
    table = read_mineral_log(path, expected_id="H1")
    assert table["Dataset_ID"].tolist() == ["H1"]
    assert table["900D_Fe"].isna().all()
    with pytest.raises(ValueError, match="expected 'H2'"):
        read_mineral_log(path, expected_id="H2")


def test_load_dataset_metadata_enforces_cohort_membership(tmp_path):
    index = tmp_path / "index.csv"
    index.write_text(
        "Data_name,WAMEX_CollarId,PROJECT\nH1,11.0,P\nH2,22.0,P\nOTHER,33.0,Q\n",
        encoding="utf-8",
    )
    metadata = load_dataset_metadata(index, ["H1", "H2"])
    assert metadata["CollarId"].tolist() == [11, 22]
    assert metadata["Project"].tolist() == ["P", "P"]
    with pytest.raises(ValueError, match="H3"):
        load_dataset_metadata(index, ["H1", "H3"])


def test_different_recorded_boundaries_are_ambiguous_even_with_same_au_and_sample():
    logs = make_logs((10.0, 11.0))
    off = DEPTH_ATOL_M / 2
    assays = make_assays(
        (1, 1, 10.000, 11.000, 0.5, "S1"),
        (2, 1, 10.0 + off, 11.0 + off, 0.5, "S1"),
    )
    retained, outcome = match_assays(logs, assays)
    assert retained.empty
    assert statuses(outcome) == ["conflicting_assays"]


def test_finite_candidate_next_to_missing_au_candidate_is_ambiguous():
    logs = make_logs((10.0, 11.0))
    assays = make_assays(
        (1, 1, 10.0, 11.0, 0.5, "S1"),
        (2, 1, 10.0, 11.0, np.nan, "S1"),
    )
    retained, outcome = match_assays(logs, assays)
    assert retained.empty
    assert statuses(outcome) == ["conflicting_assays"]


def test_match_assays_rejects_text_depths_instead_of_converting_locally():
    logs = make_logs((10.0, 11.0)).astype({"Depth_from_(m)": str})
    assays = make_assays((1, 1, 10.0, 11.0, 0.5, "S1"))
    with pytest.raises(TypeError, match="Non-numeric"):
        match_assays(logs, assays)
    with pytest.raises(TypeError, match="Non-numeric"):
        match_assays(make_logs((10.0, 11.0)), assays.astype({"Assay_ToDepth": str}))


def test_to_numeric_checked_reports_malformed_and_infinite_but_keeps_missing():
    frame = pd.DataFrame(
        {"a": ["1.5", "abc", "inf", None, "-inf"], "b": ["1", "2", "3", "4", "5"]}
    )
    converted, issues = to_numeric_checked(frame, ["a", "b"])
    assert converted["a"].isna().tolist() == [False, True, True, True, True]
    assert converted["b"].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert issues == [{"column": "a", "malformed": 1, "infinite": 2}]


def test_validate_descriptors_allows_nan_but_rejects_text_and_infinity():
    ok = pd.DataFrame({"x": [1.0, np.nan]})
    validate_descriptors(ok, ["x"])
    with pytest.raises(TypeError):
        validate_descriptors(pd.DataFrame({"x": ["1", "2"]}), ["x"])
    with pytest.raises(ValueError, match="Infinite"):
        validate_descriptors(pd.DataFrame({"x": [1.0, np.inf]}), ["x"])


def test_common_cohort_keeps_partial_rows_and_drops_rows_without_a_group():
    frame = pd.DataFrame(
        {
            "Dataset_ID": ["A", "A", "A", "A"],
            "Depth_from_(m)": [0.0, 1.0, 2.0, 3.0],
            "Depth_to_(m)": [1.0, 2.0, 3.0, 4.0],
            "Au_PPM": [0.1, 0.2, 0.3, 0.4],
            "s1": [1.0, np.nan, np.nan, 4.0],
            "s2": [np.nan, np.nan, np.nan, 5.0],
            "t1": [1.0, 2.0, np.nan, np.nan],
            "t2": [np.nan, np.nan, 3.0, 4.0],
        }
    )
    swir, tir = ["s1", "s2"], ["t1", "t2"]
    mask = common_cohort_mask(frame, swir, tir)
    # Row 1 has no SWIR, row 2 has no SWIR; rows 0 and 3 have both groups.
    assert mask.tolist() == [True, False, False, True]

    base = ["Dataset_ID", "Depth_from_(m)", "Depth_to_(m)", "Au_PPM"]
    df_swir = frame.loc[mask, base + swir].reset_index(drop=True)
    df_tir = frame.loc[mask, base + tir].reset_index(drop=True)
    df_combined = frame.loc[mask, base + swir + tir].reset_index(drop=True)
    assert_aligned([df_swir, df_tir, df_combined])
    assert df_swir["s2"].isna().tolist() == [True, False]  # partial NaN is kept


def test_common_cohort_rejects_text_and_infinite_descriptors():
    frame = pd.DataFrame({"s": [1.0, np.inf], "t": [1.0, 1.0]})
    with pytest.raises(ValueError, match="Infinite"):
        common_cohort_mask(frame, ["s"], ["t"])
    with pytest.raises(TypeError):
        common_cohort_mask(frame.astype({"t": str}), ["s"], ["t"])


def write_synthetic_inputs(root):
    csv_dir = root / "csv"
    logs_dir = root / "logs"
    csv_dir.mkdir()
    logs_dir.mkdir()
    (csv_dir / "HyLogger_Dataset_Index.csv").write_text(
        "Data_name,WAMEX_CollarId,PROJECT\nH1,11,P1\nH2,22,P2\nOTHER,33,P3\n",
        encoding="utf-8",
    )
    (csv_dir / "WAMEX_AssayFlatData.csv").write_text(
        "AssayFlatDataId,CollarId,FromDepth,ToDepth,Au_PPM,CompanySampleId\n"
        "1,11,0,1,0.5,S1\n"
        "2,11,1,2,-0.1,S2\n"
        "3,11,2,3,x,S3\n"
        "5,11,3,4,0.7,S4\n"
        "4,11,3,4,0.7,S4\n"
        "6,22,0,1,0.0,S6\n"
        "7,33,0,1,9.0,S7\n",
        encoding="utf-8",
    )
    h1_rows = "0,1,0.5\n1,2,abc\n2,3,inf\n3,4,N.A.\n4,5,0.3\n"
    write_log_rows(logs_dir / "SM_H1.csv", "H1", h1_rows)
    write_log_rows(logs_dir / "SM_H2.csv", "H2", "0,1,0.2\n")
    write_log_rows(logs_dir / "SM_OTHER.csv", "OTHER", "0,1,0.9\n")
    return logs_dir, csv_dir


def write_log_rows(path, dataset_id: str, rows: str) -> None:
    path.write_text(
        f"GSWA TSG Summary Mineral Log,,\nDataset,{dataset_id},\n"
        f"Depth_from_(m),Depth_to_(m),900D_Fe\n{rows}",
        encoding="utf-8",
    )


def test_build_dataset_end_to_end_counts_reconcile(tmp_path):
    logs_dir, csv_dir = write_synthetic_inputs(tmp_path)
    df, report, input_issues = build_dataset(
        logs_dir, csv_dir, dataset_ids=("H1", "H2")
    )

    assert df.columns[0] == "Dataset_ID"
    key = ["Dataset_ID", "Depth_from_(m)", "AssayFlatDataId", "Au_PPM"]
    assert df[key].values.tolist() == [
        ["H1", 0.0, 1, 0.5],
        ["H1", 3.0, 4, 0.7],
        ["H2", 0.0, 6, 0.0],
    ]
    assert df["Project"].tolist() == ["P1", "P1", "P2"]
    assert df["CompanySampleId"].tolist() == ["S1", "S4", "S6"]

    h1 = report.loc["H1"]
    assert h1[["log_rows", "retained", "negative_au"]].tolist() == [5, 2, 1]
    assert h1[["missing_or_nonfinite_au", "no_assay_match"]].tolist() == [1, 1]
    assert h1["duplicates_removed"] == 1
    assert report.loc["H2", ["log_rows", "retained"]].tolist() == [1, 1]
    assert report["retained"].sum() == len(df)
    outcomes = report[list(STATUSES)].sum(axis=1)
    assert (outcomes == report["log_rows"]).all()

    issues = input_issues.set_index(["Source", "column"])
    fe = issues.loc[("mineral_log", "900D_Fe")]
    assert fe[["malformed", "infinite"]].tolist() == [1, 1]
    assert issues.loc[("assay", "Au_PPM"), "malformed"] == 1
    assert df["900D_Fe"].notna().tolist() == [True, False, True]


def test_build_dataset_enforces_cohort_files(tmp_path):
    logs_dir, csv_dir = write_synthetic_inputs(tmp_path)
    (logs_dir / "SM_H2.csv").unlink()
    with pytest.raises(FileNotFoundError):
        build_dataset(logs_dir, csv_dir, dataset_ids=("H1", "H2"))
    with pytest.raises(ValueError, match="Duplicate"):
        build_dataset(logs_dir, csv_dir, dataset_ids=("H1", "H1"))
