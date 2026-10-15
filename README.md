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
