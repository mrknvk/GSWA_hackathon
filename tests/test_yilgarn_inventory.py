import matplotlib
import numpy as np
import pandas as pd
import pytest

from gswa_hackathon import yilgarn_inventory as inventory_module
from gswa_hackathon.build_dataset import COHORT_DATASET_IDS

matplotlib.use("Agg")


@pytest.fixture
def write_inventory(tmp_path):
    def write(collars, assays, index=()):
        pd.DataFrame(
            collars,
            columns=[
                "CollarId",
                "CompanyHoleId",
                "TectonicRegion",
                "Longitude",
                "Latitude",
            ],
        ).to_csv(tmp_path / "WAMEX_Collar.csv", index=False)
        pd.DataFrame(
            assays, columns=["CollarId", "FromDepth", "ToDepth", "Au_PPM"]
        ).to_csv(tmp_path / "WAMEX_AssayFlatData.csv", index=False)
        pd.DataFrame(
            index, columns=["Data_name", "WAMEX_CollarId", "PROJECT", "MINERALLOG"]
        ).to_csv(tmp_path / "HyLogger_Dataset_Index.csv", index=False)
        return tmp_path

    return write


def collar(collar_id, region="Yilgarn Craton", name=None):
    return (collar_id, name or f"HOLE_{collar_id}", region, 120.0, -30.0)


def test_measured_au_requires_finite_nonnegative_au_and_valid_depths(write_inventory):
    assays = [
        (1, 0, 1, 0),
        (1, 1, 2, 0.999),
        (1, 2, 3, 1),
        (1, 3, 4, 2),
        (1, 4, 5, -0.01),
        (1, 5, 6, "bad"),
        (1, 6, 7, np.nan),
        (1, 7, 8, np.inf),
        (1, 8, 9, -np.inf),
        (1, -1, 0, 100),
        (1, 2, 2, 100),
        (1, 3, 2, 100),
        (1, np.nan, 2, 100),
        (1, 0, np.inf, 100),
        (1, "bad", 2, 100),
        (2, 0, 1, -1),
        (3, 0, 1, 100),
    ]
    csv_dir = write_inventory(
        [collar(1), collar(2), collar(3, "Other"), collar(4)], assays
    )

    inventory, sources = inventory_module.build_inventory(csv_dir)

    assert inventory.CollarId.tolist() == [1]
    row = inventory.iloc[0]
    assert row.assay_records == 15
    assert row.measured_records == 4
    assert row.negative_au_records == 1
    assert row.missing_au_records == 4
    assert row.median_au_ppm == pytest.approx(0.9995)
    assert row.max_au_ppm == 2
    assert row.anomalies_ge_1ppm == 2
    assert row.anomaly_fraction == 0.5
    assert row.mapping_status == "no_hylogger"
    assert sources.empty
    assert sources.columns.tolist() == [
        "Dataset_ID",
        "CollarId",
        "Project",
        "source_url",
    ]
    assert sources.CollarId.dtype == np.dtype("int64")


def test_inventory_counts_collars_not_index_rows_and_marks_pilot(write_inventory):
    pilot = COHORT_DATASET_IDS[0]
    csv_dir = write_inventory(
        [collar(3), collar(1), collar(2), collar(4)],
        [(hole, 0, 1, 1) for hole in range(1, 5)],
        [
            (pilot, 1, " Pilot project ", " https://example.invalid/pilot.csv "),
            (pilot, 1, " Pilot project ", " https://example.invalid/pilot.csv "),
            ("NEW", 2, None, "https://example.invalid/new.csv"),
            ("NO_LOG", 4, " ", None),
        ],
    )

    inventory, sources = inventory_module.build_inventory(csv_dir)

    assert inventory.CollarId.tolist() == [1, 2, 3, 4]
    assert inventory.CollarId.is_unique
    assert inventory.has_hylogger.sum() == 3
    assert inventory.has_mineral_log.sum() == 2
    assert inventory.is_pilot.tolist() == [True, False, False, False]
    assert inventory.mapping_status.tolist() == [
        "unambiguous",
        "unambiguous",
        "no_hylogger",
        "no_mineral_log",
    ]
    assert inventory.Project.tolist() == [
        "Pilot project",
        "Unknown",
        "Unknown",
        "Unknown",
    ]
    assert inventory.Coordinate_CRS.eq("GDA2020 / EPSG:7844").all()
    pd.testing.assert_frame_equal(
        sources,
        pd.DataFrame(
            [
                (pilot, 1, "Pilot project", "https://example.invalid/pilot.csv"),
                ("NEW", 2, "Unknown", "https://example.invalid/new.csv"),
            ],
            columns=["Dataset_ID", "CollarId", "Project", "source_url"],
        ),
    )


def test_index_joins_only_on_primary_wamex_collar_id(write_inventory):
    csv_dir = write_inventory(
        [collar(1, name="MATCH"), collar(2, name="OTHER")],
        [(1, 0, 1, 1), (2, 0, 1, 1)],
        [
            ("MATCH", 2, "Primary", "https://example.invalid/primary.csv"),
            ("OTHER", None, "Wrong", "https://example.invalid/missing.csv"),
            ("FOREIGN", 999, "Wrong", "https://example.invalid/foreign.csv"),
        ],
    )

    inventory, sources = inventory_module.build_inventory(csv_dir)

    assert inventory.mapping_status.tolist() == ["no_hylogger", "unambiguous"]
    assert sources.CollarId.tolist() == [2]
    assert sources.Dataset_ID.tolist() == ["MATCH"]
    assert sources.Project.tolist() == ["Primary"]


@pytest.mark.parametrize(
    "entries",
    [
        [("A", 1, "P", "url-a"), ("B", 1, "P", "url-b")],
        [("A", 1, "P", "url-a"), ("A", 1, "P", "url-b")],
        [("A", 1, "P", "url-a"), ("A", 1, "Other", "url-a")],
        [("A", 1, "P", "url-a"), ("A", 2, "P", "url-a")],
        [("A", 1, "P", "url-a"), ("A", 2, "P", "url-b")],
        [(" ", 1, "P", "url-a")],
        [("A", 1, "P", "shared-url"), ("B", 2, "P", "shared-url")],
    ],
    ids=[
        "multiple-names",
        "multiple-urls",
        "multiple-projects",
        "repeated-name-url",
        "repeated-name-different-url",
        "missing-name",
        "shared-url-different-names",
    ],
)
def test_ambiguous_mappings_are_audited_without_arbitrary_source_choice(
    write_inventory, entries
):
    csv_dir = write_inventory(
        [collar(1), collar(2), collar(3)],
        [(hole, 0, 1, 1) for hole in (1, 2, 3)],
        [
            *[
                (name, hole, project, f"https://example.invalid/{url}.csv")
                for name, hole, project, url in entries
            ],
            ("SAFE", 3, "Safe project", "https://example.invalid/safe.csv"),
        ],
    )

    inventory, sources = inventory_module.build_inventory(csv_dir)

    ambiguous_collars = {entry[1] for entry in entries}
    audited = inventory.loc[inventory.CollarId.isin(ambiguous_collars)]
    assert audited.mapping_status.eq("ambiguous_mapping").all()
    assert audited.has_hylogger.all()
    assert audited.has_mineral_log.all()
    assert sources.CollarId.tolist() == [3]
    assert sources.Dataset_ID.tolist() == ["SAFE"]


def test_duplicate_primary_collar_keys_are_rejected(write_inventory):
    csv_dir = write_inventory([collar(1), collar(1)], [(1, 0, 1, 1)])
    with pytest.raises(ValueError, match="Duplicate WAMEX collar keys"):
        inventory_module.build_inventory(csv_dir)


@pytest.mark.parametrize("shared_field", ["name", "url"])
def test_identity_conflicts_outside_eligible_population_are_not_ignored(
    write_inventory, shared_field
):
    csv_dir = write_inventory(
        [collar(1), collar(2, "Other")],
        [(1, 0, 1, 1)],
        [
            ("A", 1, "P", "https://example.invalid/a.csv"),
            (
                "A" if shared_field == "name" else "B",
                2,
                "Q",
                "https://example.invalid/a.csv"
                if shared_field == "url"
                else "https://example.invalid/b.csv",
            ),
        ],
    )
    inventory, sources = inventory_module.build_inventory(csv_dir)
    assert inventory.mapping_status.tolist() == ["ambiguous_mapping"]
    assert sources.empty


def test_unlinked_index_rows_cannot_hide_conflicting_project_metadata(write_inventory):
    csv_dir = write_inventory(
        [collar(1)],
        [(1, 0, 1, 1)],
        [("A", 1, "P", "https://example.invalid/a.csv"), ("A", 1, "Q", "")],
    )
    inventory, sources = inventory_module.build_inventory(csv_dir)
    assert inventory.mapping_status.tolist() == ["ambiguous_mapping"]
    assert sources.empty


def test_empty_measured_inventory_preserves_schemas(write_inventory):
    csv_dir = write_inventory([collar(1)], [(1, 0, 1, -1)])
    inventory, sources = inventory_module.build_inventory(csv_dir)
    assert inventory.empty
    assert sources.empty
    assert {"mapping_status", "is_pilot", "Coordinate_CRS"} <= set(inventory.columns)


@pytest.fixture
def map_inventory():
    return pd.DataFrame(
        {
            "Longitude": [120.0, 121.0],
            "Latitude": [-30.0, -31.0],
            "median_au_ppm": [0.0, 1.0],
            "max_au_ppm": [1.0, 2.0],
            "anomaly_fraction": [0.0, 1.0],
            "has_mineral_log": [False, True],
            "is_pilot": [True, False],
        }
    )


def test_map_generation_uses_agg_and_closes_figure(map_inventory, tmp_path):
    output = tmp_path / "maps" / "inventory.png"
    before = inventory_module.plt.get_fignums()

    inventory_module.plot_inventory(map_inventory, output)

    assert matplotlib.get_backend().lower() == "agg"
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert inventory_module.plt.get_fignums() == before


@pytest.mark.parametrize(
    ("column", "coordinate"),
    [
        ("Longitude", np.nan),
        ("Longitude", np.inf),
        ("Longitude", -181),
        ("Longitude", 181),
        ("Latitude", np.nan),
        ("Latitude", -np.inf),
        ("Latitude", -91),
        ("Latitude", 91),
    ],
)
def test_map_rejects_missing_nonfinite_or_out_of_range_coordinates(
    map_inventory, tmp_path, column, coordinate
):
    map_inventory.loc[0, column] = coordinate
    output = tmp_path / "invalid.png"
    with pytest.raises(ValueError, match="coordinates"):
        inventory_module.plot_inventory(map_inventory, output)
    assert not output.exists()
