import json
import subprocess
from pathlib import Path

from gswa_hackathon.prepare_public_notebooks import clear_execution_outputs

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_DOCUMENTS = {
    "data/INTERIM_HyLogger_Dataset_Package_GDA2020_DB/DOCUMENTS/Licence_CCBY4_intrim.pdf",
    "data/INTERIM_HyLogger_Dataset_Package_GDA2020_DB/DOCUMENTS/METADATA/"
    "DATADICTIONARY_INTERIM_Hylogger_GDA2020.pdf",
    "data/INTERIM_HyLogger_Dataset_Package_GDA2020_DB/DOCUMENTS/METADATA/"
    "METADATA_INTERIM_Hylogger_GDA2020.pdf",
}


def test_clear_execution_outputs_preserves_code_and_markdown():
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "source": ["df.head()"],
                "execution_count": 7,
                "outputs": [{"output_type": "stream", "text": "sample data"}],
                "metadata": {},
            },
            {"cell_type": "markdown", "source": ["Analysis"], "metadata": {}},
        ],
        "metadata": {"widgets": {"state": "sample data"}, "language_info": {}},
    }
    assert clear_execution_outputs(notebook)
    assert notebook["cells"][0]["source"] == ["df.head()"]
    assert notebook["cells"][0]["outputs"] == []
    assert notebook["cells"][0]["execution_count"] is None
    assert notebook["cells"][1]["source"] == ["Analysis"]
    assert notebook["metadata"] == {"language_info": {}}
    assert not clear_execution_outputs(notebook)


def test_public_notebooks_have_no_stored_execution_outputs():
    for path in (PROJECT_ROOT / "ml_pipeline").glob("*.ipynb"):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert not clear_execution_outputs(notebook), (
            f"{path.name} contains execution data; run "
            "PYTHONPATH=src uv run python -m "
            "gswa_hackathon.prepare_public_notebooks --clear"
        )


def test_tracked_data_contains_only_official_documents():
    tracked = subprocess.run(
        ["git", "ls-files", "--", "data/"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    unexpected = set(tracked).difference(OFFICIAL_DOCUMENTS)
    assert not unexpected, f"Source/derived data tracked in Git: {sorted(unexpected)}"


def test_gitignore_excludes_data_but_allows_code_and_reviewed_reports():
    excluded = [
        "data/new_package/spectra.csv",
        "data/extracted_data/ml_dataset.csv",
        "data/extracted_data/drillhole_coordinates.csv",
        "data/extracted_data/plots/spectral_by_class.png",
        "SM_TEST.csv",
        "WAMEX_AssayFlatData.csv",
        "HyLogger_Dataset_Index.csv",
        "source.sqlite",
        "package.zip",
        "raw/spectra.txt",
        "raw_models/model.bin",
        "finetuned_models/model.json",
    ]
    allowed = [
        *sorted(OFFICIAL_DOCUMENTS),
        "src/gswa_hackathon/build_dataset.py",
        "ml_pipeline/ml.ipynb",
        "reports/cohort_summary.md",
        "reports/reviewed_aggregate.csv",
    ]
    ignored = subprocess.run(
        ["git", "check-ignore", "--no-index", "--", *excluded, *allowed],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert set(ignored) == set(excluded)
