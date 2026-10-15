import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def load_notebook_code():
    notebook_path = Path(__file__).resolve().parents[1] / "ml_pipeline/ml.ipynb"
    notebook = json.loads(notebook_path.read_text())
    return {
        cell["id"]: "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    }


def test_notebook_split_preserves_groups_labels_and_reproducibility(tmp_path):
    code_cells = load_notebook_code()
    au_values = [0.0, 0.5, 0.999, 1.0, 1.001, 10.0]
    samples = pd.DataFrame(
        {
            "CollarId": [hole for hole in range(10) for _ in au_values],
            "Au_PPM": au_values * 10,
            "Project": "Synthetic",
        }
    )
    dataset_path = tmp_path / "ml_dataset.csv"
    samples.to_csv(dataset_path, index=False)

    namespace = {}
    exec(code_cells["split-setup"], namespace)
    namespace["DATASET_PATH"] = dataset_path
    exec(code_cells["grouped-split"], namespace)
    exec(code_cells["split-summary"], namespace)
    exec(code_cells["feature-selection"], namespace)

    assert namespace["df"]["Au_Class"].iloc[:6].tolist() == [
        "Normal",
        "Normal",
        "Normal",
        "Anomaly",
        "Anomaly",
        "Anomaly",
    ]
    train = namespace["df_train"]
    test = namespace["df_test"]
    assert set(train["CollarId"]).isdisjoint(test["CollarId"])
    assert len(train) + len(test) == len(samples)
    assert set(train.index) | set(test.index) == set(samples.index)
    assert "Project" in train.columns
    for features in namespace["feature_sets"].values():
        assert not {"Au_PPM", "Au_Class", "CollarId", "Project"}.intersection(features)

    exec(code_cells["grouped-split"], namespace)
    pd.testing.assert_frame_equal(train, namespace["df_train"])
    pd.testing.assert_frame_equal(test, namespace["df_test"])


def test_notebook_models_share_split_and_fit_imputation_on_training_only(tmp_path):
    code_cells = load_notebook_code()
    au_values = [0.0, 0.5, 0.999, 1.0, 1.001, 10.0]
    samples = pd.DataFrame(
        {
            "CollarId": np.repeat(np.arange(10), len(au_values)),
            "Au_PPM": au_values * 10,
            "Project": "Synthetic",
        }
    )
    dataset_path = tmp_path / "ml_dataset.csv"
    samples.to_csv(dataset_path, index=False)

    namespace = {}
    exec(code_cells["split-setup"], namespace)
    namespace["DATASET_PATH"] = dataset_path
    exec(code_cells["grouped-split"], namespace)
    exec(code_cells["feature-selection"], namespace)

    train = namespace["df_train"]
    test = namespace["df_test"]
    feature_names = namespace["feature_sets"]["combined"]
    for feature in feature_names:
        train[feature] = train.index.to_numpy(dtype=float)
        train.loc[train.index[::4], feature] = np.nan
        test[feature] = 10000.0
    empty_training_feature = feature_names[0]
    train[empty_training_feature] = np.nan

    exec(code_cells["model-setup"], namespace)
    # Small ensembles verify the actual notebook workflow without expensive tests.
    namespace["classifiers"]["Random Forest"].set_params(n_estimators=5, n_jobs=1)
    namespace["classifiers"]["XGBoost"].set_params(n_estimators=5, n_jobs=1)
    exec(code_cells["model-training"], namespace)
    exec(code_cells["model-evaluation"], namespace)
    exec(code_cells["feature-importances"], namespace)

    assert len(namespace["fitted_models"]) == 9
    assert len(namespace["feature_importances"]) == 6
    assert np.isfinite(namespace["metrics"].to_numpy()).all()
    assert namespace["CLASS_TO_ID"] == {"Normal": 0, "Anomaly": 1}
    xgboost_parameters = namespace["classifiers"]["XGBoost"].get_params()
    assert xgboost_parameters["objective"] == "binary:logistic"
    assert xgboost_parameters["eval_metric"] == "logloss"
    for experiment, pipeline in namespace["fitted_models"].items():
        feature_set_name, model_name = experiment
        selected_features = namespace["feature_sets"][feature_set_name]
        imputer = pipeline.named_steps["imputer"]
        expected_medians = train[selected_features].median().fillna(0).to_numpy()
        np.testing.assert_allclose(imputer.statistics_, expected_medians)

        predictions = namespace["prediction_tables"][experiment]
        pd.testing.assert_index_equal(predictions.index, test.index)
        assert predictions["CollarId"].equals(test["CollarId"])
        assert predictions["Project"].equals(test["Project"])
        probability_columns = [f"score_{label}" for label in namespace["CLASS_NAMES"]]
        np.testing.assert_allclose(predictions[probability_columns].sum(axis=1), 1)
        assert pipeline.classes_.tolist() == namespace["CLASS_IDS"]
        confusion = namespace["confusion_matrices"][experiment]
        assert confusion.shape == (2, 2)
        assert confusion.index.tolist() == ["Normal", "Anomaly"]
        assert confusion.columns.tolist() == ["Normal", "Anomaly"]
        assert confusion.to_numpy().sum() == len(test)
        report = namespace["classification_reports"][experiment]
        metrics = namespace["metrics"].loc[experiment]
        assert metrics["anomaly_precision"] == report.loc["Anomaly", "precision"]
        assert metrics["anomaly_recall"] == report.loc["Anomaly", "recall"]
        assert metrics["anomaly_f1"] == report.loc["Anomaly", "f1-score"]

        if model_name != "Dummy":
            importance = namespace["feature_importances"][experiment]
            assert set(importance.index) == set(imputer.get_feature_names_out())
            assert importance.index.str.startswith("missingindicator_").any()


@pytest.mark.parametrize("only_class", ["Normal", "Anomaly"])
def test_notebook_stops_when_training_is_missing_a_class(only_class):
    code_cells = load_notebook_code()
    namespace = {}
    exec(code_cells["split-setup"], namespace)
    namespace["y_train"] = pd.Series([only_class, only_class])
    namespace["y_test"] = pd.Series(["Normal", "Anomaly"])
    with pytest.raises(ValueError, match="both Au classes"):
        exec(code_cells["model-setup"], namespace)


def test_spectral_plots_use_only_observed_training_values(tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    code_cells = load_notebook_code()
    train = pd.DataFrame(
        {
            "Au_Class": ["Normal", "Anomaly", "Normal", "Anomaly"],
            "Au_PPM": [0.0, 1.0, 0.5, 3.0],
            "Project": ["P1", "P2", "P1", "P2"],
            "2250D_FeOH": [0.1, 0.2, np.nan, 0.4],
            "2250W_FeOH": [2250.0, np.nan, 2252.0, 2253.0],
            "2200D_AlOH": [0.1, 0.2, np.nan, 0.3],
            "11300D_Carbonate": [np.nan, 0.1, 0.2, 0.3],
        },
        index=[10, 20, 30, 40],
    )
    original_train = train.copy(deep=True)
    namespace = {
        "ROOT": tmp_path,
        "np": np,
        "pd": pd,
        "df_train": train,
        "CLASS_NAMES": ["Normal", "Anomaly"],
        "RANDOM_STATE": 67,
        "ANOMALY_AU_PPM": 1.0,
    }
    for cell_id in [
        "spectral-plot-setup",
        "feature-class-distributions",
        "feoh-pair-scatter",
        "feoh-au-scatter",
    ]:
        exec(code_cells[cell_id], namespace)

    assert namespace["distribution_counts"] == {
        "2250D_FeOH": [1, 2],
        "2250W_FeOH": [2, 1],
        "2200D_AlOH": [1, 2],
        "11300D_Carbonate": [1, 2],
    }
    assert namespace["feoh_pairs"].index.tolist() == [10, 40]
    au_plot_data = namespace["au_plot_data"]
    assert au_plot_data.index.tolist() == [10, 20, 40]
    np.testing.assert_allclose(au_plot_data["log1p_Au_PPM"], np.log1p([0.0, 1.0, 3.0]))
    pd.testing.assert_frame_equal(train, original_train)
    plot_dir = tmp_path / "data" / "extracted_data" / "plots"
    for filename in [
        "spectral_by_class.png",
        "feoh_position_vs_depth.png",
        "feoh_depth_vs_au_by_project.png",
    ]:
        assert (plot_dir / filename).stat().st_size > 0


@pytest.mark.parametrize("normal_x", [[0.1, 0.2], [0.1, 0.1], [np.nan, 0.2]])
def test_feoh_aloh_regressions_use_measured_class_pairs(
    tmp_path, monkeypatch, normal_x
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    code_cells = load_notebook_code()
    train = pd.DataFrame(
        {
            "Au_Class": ["Normal", "Normal", "Anomaly", "Anomaly", "Anomaly"],
            "2250D_FeOH": normal_x + [0.3, 0.4, 0.5],
            "2200D_AlOH": [1.2, 1.4, 0.9, 1.2, np.nan],
        }
    )
    original_train = train.copy(deep=True)
    namespace = {
        "ROOT": tmp_path,
        "np": np,
        "pd": pd,
        "df_train": train,
        "CLASS_NAMES": ["Normal", "Anomaly"],
        "RANDOM_STATE": 67,
        "ANOMALY_AU_PPM": 1.0,
    }
    exec(code_cells["spectral-plot-setup"], namespace)
    exec(code_cells["feoh-aloh-scatter"], namespace)

    fits = namespace["regression_fits"]
    assert fits["Anomaly"]["slope"] == pytest.approx(3.0)
    assert fits["Anomaly"]["intercept"] == pytest.approx(0.0, abs=1e-12)
    has_two_normal_x = pd.Series(normal_x).nunique() >= 2
    assert ("Normal" in fits) == has_two_normal_x
    if has_two_normal_x:
        assert fits["Normal"]["slope"] == pytest.approx(2.0)
        assert fits["Normal"]["intercept"] == pytest.approx(1.0)
    assert namespace["feoh_aloh_pairs"].index.tolist() == train.dropna().index.tolist()
    pd.testing.assert_frame_equal(train, original_train)
    plot_path = namespace["PLOT_DIR"] / "feoh_vs_aloh_depth_by_class.png"
    assert plot_path.stat().st_size > 0


@pytest.mark.parametrize("has_missing_coordinate", [False, True])
def test_collar_coordinates_cover_full_cohort_and_preserve_crs(
    tmp_path, has_missing_coordinate
):
    code_cells = load_notebook_code()
    csv_dir = (
        tmp_path
        / "data"
        / "INTERIM_HyLogger_Dataset_Package_GDA2020_DB"
        / "DATABASES"
        / "CSV"
    )
    csv_dir.mkdir(parents=True)
    (tmp_path / "data" / "extracted_data").mkdir()
    pd.DataFrame(
        {
            "CollarId": [22, 11],
            "Longitude": [120.0, 121.0],
            "Latitude": [-30.0, np.nan if has_missing_coordinate else -31.0],
        }
    ).to_csv(csv_dir / "WAMEX_Collar.csv", index=False)
    pd.DataFrame(
        {
            "Data_name": ["H2", "H1", "OTHER", "OTHER"],
            "Shapefile_longitude": [120.001, 121.001, 120.0, 120.0],
            "Shapefile_latitude": [-30.001, -31.001, -31.0, -31.0],
        }
    ).to_csv(csv_dir / "HyLogger_Dataset_Index.csv", index=False)
    namespace = {
        "ROOT": tmp_path,
        "np": np,
        "pd": pd,
        "df": pd.DataFrame(
            {
                "Dataset_ID": ["H1", "H1", "H2"],
                "CollarId": [11, 11, 22],
                "Project": ["P1", "P1", "P2"],
            }
        ),
    }
    if has_missing_coordinate:
        with pytest.raises(ValueError, match="Missing or nonfinite"):
            exec(code_cells["collar-coordinates"], namespace)
        assert not (tmp_path / "data/extracted_data/drillhole_coordinates.csv").exists()
        return

    exec(code_cells["collar-coordinates"], namespace)
    coordinates = pd.read_csv(namespace["COORDINATE_PATH"])
    assert coordinates["Dataset_ID"].tolist() == ["H1", "H2"]
    assert coordinates["Longitude"].tolist() == [121.0, 120.0]
    assert coordinates["Latitude"].tolist() == [-31.0, -30.0]
    assert coordinates["Shapefile_longitude"].tolist() == [121.001, 120.001]
    assert coordinates["Coordinate_CRS"].eq("GDA2020 / EPSG:7844").all()
    assert coordinates["Coordinate_Source"].eq("WAMEX_Collar.csv").all()
