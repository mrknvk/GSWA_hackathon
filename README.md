# Spectral Indicators of Gold Mineralisation

**GSWA HyLogger Hackathon 2026**

**Marko Novak and Ziyi Zhu**

## Project Hypothesis

Fine grained gold in drill cores does not produce a sufficiently strong or
spatially consistent spectral response to be classified by a HyLogger.
However, gold mineralisation may be associated with alteration assemblages and
subtle combinations of spectral features. Although individual Spectral features may be
too weak to appear in an abstract spectral interpretation with the HyLoggers Mineral Library, their combined multivariate pattern may provide a useful indicator of gold mineralisation.

The intended result is a screening tool that prioritises intervals for further
investigation. It is NOT a direct gold detector or a replacement for laboratory
assays and geological assessment.


## Source Data

The source HyLogger data is not redistributed in this repository. The
**HyLogger Drillholes Package** can be obtained directly from the [GSWA Data and Software
Centre](https://dasc.dmirs.wa.gov.au), which ensures that the Geological Survey of Western Australia remains the authoritative source and receives the required
attribution.

The package integrates HyLogger records with WAMEX drillhole, geology, and
geochemical assay data. WAMEX assays are laboratory measurements linked to
drillhole depth intervals. In this project, `Au_PPM` is the gold reference
value/target for classification.

Place the extracted package so that its SQLite database is available at:

```text
data/INTERIM_HyLogger_Dataset_Package_GDA2020_DB/DATABASES/HyLogger_Dataset_Package.sqlite
```

Obtain the required mineral logs through the official `MINERALLOG` references
in the package index and store them locally under:

```text
data/extracted_data/mineral_logs/
```

Both the source database and extracted mineral logs are excluded from this Repo.

## Cohort and Gold Distribution

The current pilot uses 22 drillholes across BADGEBUP, BGM Exploration 2007-2008,
MtMorgan, and Welcome Well. It is a fixed, deliberately shortlisted cohort,
not a representative sample of the Yilgarn Craton. See
[the cohort summary](reports/cohort_summary.md) for the documented selection
criteria, Au distributions before and after interval matching, and limitations.

The binary target is `Au_PPM >= 1.0`. In the matched pilot cohort all positive
intervals come from BADGEBUP, although other selected projects contain positive
assay records before matching. Project-related confounding must therefore be
investigated before claiming a transferable Au signature.

## Expanding Beyond the Pilot

The initial 22 holes are not all eligible Yilgarn holes. Inventory the local
package and generate a GDA2020 collar map without downloading anything:

```bash
PYTHONPATH=src uv run python -m gswa_hackathon.expand_yilgarn
```

Acquire all unambiguously mapped candidate logs and apply the existing matching
rules (four concurrent requests, validated caches, HTTPS and checksums):

```bash
PYTHONPATH=src uv run python -m gswa_hackathon.expand_yilgarn --download
```

Inspect `ml_pipeline/yilgarn_inventory.ipynb`. Local results under
`data/extracted_data/yilgarn/` include the map, per-collar inventory, acquisition
manifest, full eligibility audit, numeric issues, and project-level summary.
The new `ml_dataset_expanded.csv` contains the same ten spectral predictors and
metadata as the pilot. Both positive and negative holes are considered; no
grade-dependent selection is made after inventory eligibility.

The pilot files remain unchanged. `ml_pipeline/ml.ipynb` currently selects
`DATASET_VARIANT = "expanded"`; set it to `"pilot"` to reproduce the original
22-hole experiment. Changing the dataset establishes a new cohort and split;
do not compare its scores as though the original holdout were unchanged.
Inspect unknown project labels and positive-project coverage first.

See [the expansion design](docs/architecture/yilgarn-expansion.md) for boundaries
and scientific safeguards. A source link does not guarantee a readable,
schema-compatible or interval-matched training example. Failures are recorded,
not silently removed or replaced with zero Au.

The first completed run is documented in
[the aggregated expansion report](reports/yilgarn_expansion_summary.md): 457
usable holes and 46,336 intervals, including 1,067 intervals at or above 1 ppm.
The report separates preparation results from any future model evaluation.

## Public Repository Policy

- Keep code, notebook source, synthetic tests, attribution, and official metadata.
- Keep source data, extracted tables, sample-level predictions, coordinates,
  model artifacts, and local figures outside Git. A CSV is not automatically
  safe to publish just because it is a table.
- Put reviewed, aggregated result tables and reports under `reports/`.
  Any published figures must be reviewed separately for source-data disclosure.
- Do not commit notebook execution outputs or widget state: they can embed
  sample-level tables and images even when the source CSVs are ignored.

Before preparing a public commit, clear notebook outputs:

```bash
PYTHONPATH=src uv run python -m gswa_hackathon.prepare_public_notebooks --clear
```

Check without changing files:

```bash
PYTHONPATH=src uv run python -m gswa_hackathon.prepare_public_notebooks
uv run pytest
```

These safeguards apply to the current working tree and future commits. They do
not remove data from old Git history; history rewriting is intentionally out
of scope. Ignoring a file also does not untrack an already tracked file.

## Installation

Install [`uv`](https://docs.astral.sh/uv/), clone this repository, and create
the locked Python 3.12 environment:

```bash
uv sync
```

This installs NumPy, pandas, Matplotlib, SciPy, scikit-learn, XGBoost, Pytest, and Ruff
inside `.venv`.

Commands can run inside the managed environment without manual activation:

```bash
uv run ruff check .
```

To activate the environment on macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Leave the environment with:

```bash
deactivate
```
s
## Authors and Acknowledgement

Developed by **Marko Novak and Ziyi Zhu** for the **GSWA HyLogger Hackathon 2026**.

HyLogger and WAMEX data is provided by the Geological Survey of Western
Australia and remains subject to their original terms, licensing, and
attribution requirements.
