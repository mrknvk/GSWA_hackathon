from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import gswa_hackathon.build_dataset as dataset_module
from gswa_hackathon.build_dataset import (
    COHORT_DATASET_IDS,
    DEPTH_ATOL_M,
    HOLE_AUDIT_COLUMNS,
    ISSUE_COLUMNS,
    LEADING_COLUMNS,
    SAMPLE_KEY_COLUMNS,
    SPECTRAL_FEATURES_SWIR,
    SPECTRAL_FEATURES_TIR,
    STATUSES,
    assert_aligned,
    build_dataset,
    build_expanded_dataset,
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


def brute_force_match_assays(logs, assays):
    """Original crossproduct matcher, retained as an independent parity oracle."""
    logs = logs.reset_index(drop=True).assign(_row=lambda d: np.arange(len(d)))
    outcome = pd.DataFrame(
        {
            "Dataset_ID": logs["Dataset_ID"],
            "status": "retained",
            "duplicates_removed": 0,
        }
    )
    valid_log = valid_interval(logs["Depth_from_(m)"], logs["Depth_to_(m)"])
    outcome.loc[~valid_log, "status"] = "invalid_log_interval"
    assays = assays[valid_interval(assays["Assay_FromDepth"], assays["Assay_ToDepth"])]
    candidates = logs[valid_log].merge(assays, on="CollarId")
    aligned = (
        (candidates["Depth_from_(m)"] - candidates["Assay_FromDepth"]).abs()
        <= DEPTH_ATOL_M
    ) & (
        (candidates["Depth_to_(m)"] - candidates["Assay_ToDepth"]).abs() <= DEPTH_ATOL_M
    )
    candidates = candidates[aligned]
    matched_rows = pd.Index(candidates["_row"].unique())
    outcome.loc[outcome.index[valid_log].difference(matched_rows), "status"] = (
        "no_assay_match"
    )
    record = ["Assay_FromDepth", "Assay_ToDepth", "Au_PPM", "CompanySampleId"]
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
    unmeasured = resolved[~np.isfinite(resolved["Au_PPM"])]
    outcome.loc[unmeasured["_row"].to_numpy(), "status"] = "missing_or_nonfinite_au"
    measured = resolved[np.isfinite(resolved["Au_PPM"])]
    negative = measured[measured["Au_PPM"] < 0]
    outcome.loc[negative["_row"].to_numpy(), "status"] = "negative_au"
    return measured[measured["Au_PPM"] >= 0].drop(columns="_row"), outcome


@pytest.mark.parametrize("seed", range(20))
def test_searchsorted_matcher_matches_brute_force_randomized(seed):
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, 20, 60).astype(float)
    logs = make_logs(*zip(starts, starts + rng.choice([0.5, 1.0, 2.0], 60)))
    logs["CollarId"] = rng.choice([1.0, 2.0, 3.0, np.nan], len(logs))
    logs["Dataset_ID"] = logs["CollarId"].astype(str)
    assays = pd.DataFrame(
        {
            "AssayFlatDataId": np.arange(len(logs)),
            "CollarId": logs["CollarId"],
            "Assay_FromDepth": logs["Depth_from_(m)"]
            + rng.choice([-2, -0.5, 0, 0.5, 2], len(logs)) * DEPTH_ATOL_M,
            "Assay_ToDepth": logs["Depth_to_(m)"]
            + rng.choice([-2, -0.5, 0, 0.5, 2], len(logs)) * DEPTH_ATOL_M,
            "Au_PPM": rng.choice([0.0, 0.5, -0.01, np.nan, np.inf], len(logs)),
            "CompanySampleId": rng.choice(["A", "B", None], len(logs)),
        }
    )
    duplicates = assays.sample(n=20, random_state=seed).copy()
    duplicates["AssayFlatDataId"] += 100
    conflicts = assays.sample(n=20, random_state=seed + 1).copy()
    conflicts["AssayFlatDataId"] += 200
    conflicts["CompanySampleId"] = "conflict"
    assays = pd.concat([assays, duplicates, conflicts], ignore_index=True)
    logs.loc[:5, "Depth_from_(m)"] = [np.nan, np.inf, -1, 1, 5, 0]
    logs.loc[:5, "Depth_to_(m)"] = [1, 2, 0, 1, 4, np.inf]
    assays.loc[:5, "Assay_FromDepth"] = [np.nan, np.inf, -1, 1, 5, 0]
    assays.loc[:5, "Assay_ToDepth"] = [1, 2, 0, 1, 4, np.inf]
    logs = logs.sample(frac=1, random_state=seed)
    assays = assays.sample(frac=1, random_state=seed)
    before_logs, before_assays = logs.copy(deep=True), assays.copy(deep=True)

    actual, actual_outcome = match_assays(logs, assays)
    expected, expected_outcome = brute_force_match_assays(logs, assays)

    pd.testing.assert_frame_equal(
        actual.reset_index(drop=True), expected.reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(actual_outcome, expected_outcome)
    pd.testing.assert_frame_equal(logs, before_logs)
    pd.testing.assert_frame_equal(assays, before_assays)


def test_searchsorted_matcher_matches_brute_force_at_nextafter_boundaries():
    intervals, rows = [], []
    # Include near-zero cancellation as well as large depths whose ULP exceeds
    # the tolerance. Each pair has a separate collar to avoid masking omissions
    # behind a conflict with another candidate.
    for start in [
        0.0,
        DEPTH_ATOL_M,
        np.nextafter(DEPTH_ATOL_M, np.inf),
        1.0,
        10.0,
        1e12,
    ]:
        for sign in [-1, 1]:
            boundary = start + sign * DEPTH_ATOL_M
            for assay_start in [
                np.nextafter(boundary, -np.inf),
                boundary,
                np.nextafter(boundary, np.inf),
            ]:
                for end_offset in [-2 * DEPTH_ATOL_M, 0, DEPTH_ATOL_M]:
                    collar = len(intervals)
                    intervals.append((start, start + 1))
                    rows.append(
                        (collar, collar, assay_start, start + 1 + end_offset, 0.5, "S")
                    )
    logs = make_logs(*intervals)
    logs["CollarId"] = np.arange(len(logs))
    assays = make_assays(*rows)

    actual, actual_outcome = match_assays(logs, assays)
    expected, expected_outcome = brute_force_match_assays(logs, assays)

    pd.testing.assert_frame_equal(
        actual.reset_index(drop=True), expected.reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(actual_outcome, expected_outcome)


@pytest.mark.parametrize("empty_side", ["logs", "assays", "invalid_assays"])
def test_searchsorted_matcher_matches_brute_force_without_candidates(empty_side):
    logs = make_logs((0.0, 1.0))
    assays = make_assays((1, 1, 0.0, 1.0, 0.5, "S"))
    if empty_side == "logs":
        logs = logs.iloc[:0]
    elif empty_side == "assays":
        assays = assays.iloc[:0]
    else:
        assays["Assay_FromDepth"] = np.nan
    actual, actual_outcome = match_assays(logs, assays)
    expected, expected_outcome = brute_force_match_assays(logs, assays)
    pd.testing.assert_frame_equal(
        actual.reset_index(drop=True), expected.reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(actual_outcome, expected_outcome)


def test_searchsorted_matcher_handles_long_hole_without_crossproduct(monkeypatch):
    size = 20_000
    starts = np.arange(size, dtype=float)
    logs = make_logs(*zip(starts, starts + 1))
    assays = make_assays(
        *[(i, 1, start, start + 1, 0.5, str(i)) for i, start in enumerate(starts)]
    )
    original_merge = pd.DataFrame.merge

    def reject_collar_crossproduct(frame, other, *args, **kwargs):
        assert kwargs.get("on") != "CollarId", "Unbounded collar crossproduct"
        merged = original_merge(frame, other, *args, **kwargs)
        assert len(merged) <= size
        return merged

    monkeypatch.setattr(pd.DataFrame, "merge", reject_collar_crossproduct)
    retained, outcome = match_assays(logs, assays)
    assert len(retained) == size
    assert outcome["status"].eq("retained").all()
    assert outcome["duplicates_removed"].eq(0).all()


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


@pytest.mark.parametrize("declared_id", ['"H1"', " H1 ", '" H1 "'])
@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig"])
def test_read_mineral_log_accepts_csv_metadata_and_bom(tmp_path, declared_id, encoding):
    path = tmp_path / "download.csv"
    path.write_text(
        f"Dataset,{declared_id},\n"
        '"Depth_from_(m)",Depth_to_(m),900D_Fe\n0,1,0.5\n',
        encoding=encoding,
    )
    table = read_mineral_log(path, expected_id="H1")
    assert table["Dataset_ID"].tolist() == ["H1"]
    assert table["Depth_from_(m)"].tolist() == [0.0]
    assert table["900D_Fe"].tolist() == [0.5]
    with pytest.raises(ValueError, match="expected 'H2'"):
        read_mineral_log(path, expected_id="H2")


def test_read_mineral_log_tracks_csv_header_index_after_multiline_metadata(tmp_path):
    path = tmp_path / "download.csv"
    path.write_text(
        'Notes,"first line\nsecond line",\nDataset,"H1",\n\n'
        '"Depth_from_(m)",Depth_to_(m),900D_Fe\n0,1,0.5\n',
        encoding="utf-8",
    )
    table = read_mineral_log(path, expected_id="H1")
    assert table["900D_Fe"].tolist() == [0.5]


@pytest.mark.parametrize(
    "column",
    ["Depth_from_(m)", "Depth_to_(m)", *SPECTRAL_FEATURES_SWIR, *SPECTRAL_FEATURES_TIR],
)
def test_read_mineral_log_rejects_duplicate_required_headers_before_pandas(
    tmp_path, monkeypatch, column
):
    path = tmp_path / "download.csv"
    path.write_text(
        f'Dataset,H1,\nDepth_from_(m),Depth_to_(m),"{column}","{column}"\n'
        "0,1,0.5,0.9\n",
        encoding="utf-8",
    )

    def unexpected_read(*args, **kwargs):
        pytest.fail("Duplicate required headers must be rejected before pandas reads")

    monkeypatch.setattr(pd, "read_csv", unexpected_read)
    with pytest.raises(ValueError, match="duplicate required column headers") as error:
        read_mineral_log(path, expected_id="H1")
    assert column in str(error.value)
    assert path.name in str(error.value)


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


def write_expanded_log(path, dataset_id, starts=(0.0,), drop_columns=(), **overrides):
    features = SPECTRAL_FEATURES_SWIR + SPECTRAL_FEATURES_TIR
    frame = (
        pd.DataFrame(
            {
                "Depth_from_(m)": starts,
                "Depth_to_(m)": [start + 1 for start in starts],
                **dict.fromkeys(features, 0.5),
                "Mineral_abundance": 99.0,
            }
        )
        .assign(**overrides)
        .drop(columns=list(drop_columns))
    )
    path.write_text(
        f"Dataset,{dataset_id},\n" + frame.to_csv(index=False), encoding="utf-8"
    )


def write_expanded_assays(root, rows):
    assays = make_assays(*rows).rename(
        columns={"Assay_FromDepth": "FromDepth", "Assay_ToDepth": "ToDepth"}
    )
    assays.to_csv(root / "WAMEX_AssayFlatData.csv", index=False)


def expanded_sources(root, *dataset_ids):
    return pd.DataFrame(
        {
            "Dataset_ID": dataset_ids,
            "CollarId": range(1, len(dataset_ids) + 1),
            "Project": "Unknown (explicit)",
            "local_path": [root / f"download_{d}.csv" for d in dataset_ids],
        }
    )


def test_expanded_build_uses_manifest_paths_reads_assays_once_and_keeps_nans(
    tmp_path, monkeypatch
):
    sources = expanded_sources(tmp_path, "H1", "H2")
    write_expanded_log(sources.loc[0, "local_path"], "H1", **{"900W_Fe": ["bad"]})
    write_expanded_log(sources.loc[1, "local_path"], "H2")
    write_expanded_assays(
        tmp_path,
        [(8, 1, 0, 1, 1.0, "001"), (9, 2, 0, 1, 0.0, "002")],
    )
    monkeypatch.setattr(dataset_module, "PROJECT_ROOT", tmp_path)
    sources.loc[0, "local_path"] = "download_H1.csv"
    before = sources.copy(deep=True)
    read_csv = pd.read_csv
    assay_reads = []

    def count_assay_reads(path, *args, **kwargs):
        if path.name == "WAMEX_AssayFlatData.csv":
            assay_reads.append(path)
        return read_csv(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", count_assay_reads)
    dataset, audit, issues = build_expanded_dataset(sources, tmp_path, tmp_path)

    pd.testing.assert_frame_equal(sources, before)
    assert len(assay_reads) == 1
    assert dataset.columns.tolist() == (
        LEADING_COLUMNS + SPECTRAL_FEATURES_SWIR + SPECTRAL_FEATURES_TIR
    )
    assert dataset["Dataset_ID"].tolist() == ["H1", "H2"]
    assert dataset["CompanySampleId"].tolist() == ["001", "002"]
    assert dataset["Au_PPM"].tolist() == [1.0, 0.0]
    assert dataset["Project"].tolist() == ["Unknown (explicit)"] * 2
    assert np.isnan(dataset.loc[0, "900W_Fe"])
    assert audit["status"].tolist() == ["ready", "ready"]
    assert audit["anomalies_ge_1ppm"].tolist() == [1, 0]
    assert audit["common_cohort_rows"].sum() == len(dataset)
    assert not dataset.duplicated(SAMPLE_KEY_COLUMNS).any()
    assert issues.to_dict("records") == [
        {
            "Source": "mineral_log",
            "Dataset_ID": "H1",
            "column": "900W_Fe",
            "malformed": 1,
            "infinite": 0,
        }
    ]


def test_expanded_build_audits_mixed_schema_invalid_logs_and_zero_matches(tmp_path):
    ids = ["GOOD", "SWIR_ONLY", "MALFORMED", "UNMATCHED", "DUPLICATE", "WRONG_ID"]
    sources = expanded_sources(tmp_path, *ids)
    for source in sources.itertuples():
        write_expanded_log(source.local_path, source.Dataset_ID)
    write_expanded_log(
        sources.loc[1, "local_path"], "SWIR_ONLY", drop_columns=SPECTRAL_FEATURES_TIR
    )
    sources.loc[2, "local_path"].write_text("not a mineral log\n", encoding="utf-8")
    write_expanded_log(sources.loc[4, "local_path"], "DUPLICATE", starts=(0.0, 0.0))
    write_expanded_log(sources.loc[5, "local_path"], "OTHER")
    write_expanded_assays(tmp_path, [(1, 1, 0, 1, 2, "A")])

    dataset, audit, _ = build_expanded_dataset(sources, tmp_path, tmp_path)

    assert dataset["Dataset_ID"].tolist() == ["GOOD"]
    assert audit["Dataset_ID"].tolist() == ids
    assert audit["status"].tolist() == [
        "ready",
        "incompatible_schema",
        "invalid_log",
        "no_retained_intervals",
        "invalid_log",
        "invalid_log",
    ]
    audit = audit.set_index("Dataset_ID")
    assert "8625D_Quartz" in audit.loc["SWIR_ONLY", "error"]
    assert audit.loc["SWIR_ONLY", "log_rows"] == 1
    assert pd.isna(audit.loc["SWIR_ONLY", "retained_match_rows"])
    assert audit.loc["MALFORMED", "error"]
    assert audit.loc["UNMATCHED", "no_assay_match"] == 1
    assert audit.loc["UNMATCHED", "common_cohort_rows"] == 0
    assert "Duplicate depth" in audit.loc["DUPLICATE", "error"]
    assert "expected 'WRONG_ID'" in audit.loc["WRONG_ID", "error"]


def test_expanded_build_audits_duplicate_headers_without_aborting_other_holes(tmp_path):
    sources = expanded_sources(tmp_path, "BAD", "GOOD")
    sources.loc[0, "local_path"].write_text(
        "Dataset,BAD,\nDepth_from_(m),Depth_to_(m),900D_Fe,900D_Fe\n0,1,0.5,0.9\n",
        encoding="utf-8",
    )
    write_expanded_log(sources.loc[1, "local_path"], "GOOD")
    write_expanded_assays(tmp_path, [(1, 2, 0, 1, 0.5, "S")])

    dataset, audit, _ = build_expanded_dataset(sources, tmp_path, tmp_path)

    assert dataset["Dataset_ID"].tolist() == ["GOOD"]
    assert audit["status"].tolist() == ["invalid_log", "ready"]
    assert "duplicate required column headers" in audit.loc[0, "error"]
    assert "900D_Fe" in audit.loc[0, "error"]


def test_expanded_build_applies_coverage_after_matching_and_reports_numeric_issues(
    tmp_path,
):
    sources = expanded_sources(tmp_path, "H1")
    overrides = {
        feature: [0.5, np.nan, np.nan, 0.5] for feature in SPECTRAL_FEATURES_SWIR
    }
    overrides.update(
        {feature: [0.5, 0.5, np.nan, np.nan] for feature in SPECTRAL_FEATURES_TIR}
    )
    overrides["900W_Fe"][0] = "inf"
    write_expanded_log(
        sources.loc[0, "local_path"], "H1", starts=(0, 1, 2, 3), **overrides
    )
    write_expanded_assays(
        tmp_path, [(i + 1, 1, i, i + 1, 2, f"S{i}") for i in range(4)]
    )

    dataset, audit, issues = build_expanded_dataset(sources, tmp_path, tmp_path)

    assert len(dataset) == 1
    assert dataset["900W_Fe"].isna().all()
    assert audit.loc[0, "retained_match_rows"] == 4
    assert audit.loc[0, "common_cohort_rows"] == 1
    assert audit.loc[0, "no_usable_swir"] == 2
    assert audit.loc[0, "no_usable_tir"] == 2
    assert audit.loc[0, "anomalies_ge_1ppm"] == 1
    assert issues["infinite"].tolist() == [1]


def test_expanded_build_preserves_matching_conflicts_censoring_and_exact_depths(
    tmp_path,
):
    sources = expanded_sources(tmp_path, "H1")
    write_expanded_log(sources.loc[0, "local_path"], "H1", starts=tuple(range(7)))
    off = DEPTH_ATOL_M / 2
    write_expanded_assays(
        tmp_path,
        [
            (9, 1, off, 1 - off, 0.5, "A"),
            (2, 1, off, 1 - off, 0.5, "A"),
            (3, 1, 1, 2, -0.01, "B"),
            (4, 1, 2, 3, "bad", "C"),
            (5, 1, 3, 4, 0.5, "D"),
            (6, 1, 3, 4, np.nan, "D"),
            (7, 1, 4, 6, 0.5, "E"),
            (8, 1, 5 + 2 * DEPTH_ATOL_M, 6, 0.5, "F"),
            (10, 1, 7, 6, 0.5, "G"),
        ],
    )

    dataset, audit, issues = build_expanded_dataset(sources, tmp_path, tmp_path)

    assert dataset["AssayFlatDataId"].tolist() == [2]
    assert dataset["Assay_FromDepth"].tolist() == [off]
    assert dataset["Assay_ToDepth"].tolist() == [1 - off]
    row = audit.iloc[0]
    assert row["duplicates_removed"] == 1
    assert row["negative_au"] == 1
    assert row["missing_or_nonfinite_au"] == 1
    assert row["conflicting_assays"] == 1
    assert row["no_assay_match"] == 3
    assert row["assay_invalid_interval"] == 1
    assert row["retained_match_rows"] + row[list(STATUSES[1:])].sum() == row["log_rows"]
    assert issues.loc[issues["Source"] == "assay", "malformed"].tolist() == [1]


@pytest.mark.parametrize("duplicate_column", ["Dataset_ID", "CollarId"])
def test_expanded_build_rejects_ambiguous_mapping_before_reading_files(
    tmp_path, duplicate_column
):
    sources = expanded_sources(tmp_path, "H1", "H2")
    sources.loc[1, duplicate_column] = sources.loc[0, duplicate_column]
    with pytest.raises(ValueError, match=f"Duplicate {duplicate_column}"):
        build_expanded_dataset(sources, tmp_path, tmp_path)


@pytest.mark.parametrize(
    "case", ["empty", "incompatible", "no_assays", "no_coverage", "missing_file"]
)
def test_expanded_build_empty_results_have_stable_columns(tmp_path, case):
    sources = expanded_sources(tmp_path, "H1")
    if case == "empty":
        sources = sources.iloc[:0]
    else:
        write_expanded_assays(tmp_path, [(1, 1, 0, 1, 0.5, "A")])
    if case == "incompatible":
        write_log(sources.loc[0, "local_path"], "H1")
    if case in ("no_assays", "no_coverage"):
        write_expanded_log(
            sources.loc[0, "local_path"],
            "H1",
            starts=(10,) if case == "no_assays" else (0,),
            **dict.fromkeys(SPECTRAL_FEATURES_TIR, np.nan),
        )

    dataset, audit, issues = build_expanded_dataset(sources, tmp_path, tmp_path)

    assert dataset.empty
    assert dataset.columns.tolist() == (
        LEADING_COLUMNS + SPECTRAL_FEATURES_SWIR + SPECTRAL_FEATURES_TIR
    )
    assert audit.columns.tolist() == HOLE_AUDIT_COLUMNS
    assert issues.columns.tolist() == ISSUE_COLUMNS
    assert len(audit) == len(sources)
    if case == "empty":
        assert audit.empty
    elif case in ("no_assays", "no_coverage"):
        assert audit.loc[0, "status"] == "no_retained_intervals"
        assert audit.loc[0, "common_cohort_rows"] == 0
    else:
        assert audit.loc[0, "error"]


def test_expanded_build_requires_explicit_project_and_source_fields(tmp_path):
    sources = expanded_sources(tmp_path, "H1")
    with pytest.raises(ValueError, match="Missing source columns"):
        build_expanded_dataset(sources.drop(columns="local_path"), tmp_path, tmp_path)
    sources.loc[0, "Project"] = None
    with pytest.raises(ValueError, match="must not be missing"):
        build_expanded_dataset(sources, tmp_path, tmp_path)
    sources.loc[0, "Project"] = "  "
    with pytest.raises(ValueError, match="explicit nonempty strings"):
        build_expanded_dataset(sources, tmp_path, tmp_path)
