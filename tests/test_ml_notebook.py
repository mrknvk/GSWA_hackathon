import json
from pathlib import Path

import pandas as pd


def test_notebook_split_preserves_groups_labels_and_reproducibility(tmp_path):
    notebook_path = Path(__file__).resolve().parents[1] / "ml_pipeline/ml.ipynb"
    notebook = json.loads(notebook_path.read_text())
    code_cells = {
        cell["id"]: "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    }
    au_values = [0.0, 0.099, 0.1, 0.5, 0.501, 1.0]
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
        "Background",
        "Background",
        "Au-anomalous",
        "Au-anomalous",
        "High-priority",
        "High-priority",
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
