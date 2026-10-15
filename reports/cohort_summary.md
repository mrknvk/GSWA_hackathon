# Cohort and Au Distribution

This is an aggregated descriptive summary of the 22 selected drillholes, not
an estimate of mineral resources, reserves, contained gold, or economic value.
Au is reported in ppm (numerically equivalent to g/t for mass concentrations).
Statistics are unweighted per record/interval; interval lengths, assay methods,
censoring, and sampling density can differ. A maximum does not describe a whole
deposit, and a median below 1 ppm does not mean a project is unmineralised.

## Highest Measured Concentration

BADGEBUP has the highest observed Au value in both populations below:
**438 ppm**. In the matched dataset the maximum is in **BSDD012**.
The high outlier strongly affects the mean; the median is reported separately.

## Before Matching

All assay records associated with the selected collars, requiring finite,
nonnegative Au and valid nonnegative depth intervals. Negative and missing Au
are excluded, not replaced. These are source assay records, not necessarily
unique samples: no duplicate/conflict resolution or spectral matching has been
applied at this stage. Rows need not overlap scanned core.

| Project | Holes | Measured assay records | Median Au (ppm) | Mean Au (ppm) | Maximum Au (ppm) | Records >= 1 ppm |
|---|---:|---:|---:|---:|---:|---:|
| BADGEBUP | 9 | 1,514 | 0.040 | 1.340198 | 438.00 | 366 |
| BGM Exploration 2007-2008 | 4 | 1,353 | 0.150 | 0.381992 | 43.55 | 117 |
| MtMorgan | 4 | 2,033 | 0.005 | 0.034716 | 19.20 | 8 |
| Welcome Well | 5 | 587 | 0.003 | 0.004571 | 0.11 | 0 |

Excluded Au records within these selected collars, before spectral matching:

| Project | All assay records | Negative Au | Missing/nonfinite Au |
|---|---:|---:|---:|
| BADGEBUP | 1,870 | 9 | 347 |
| BGM Exploration 2007-2008 | 1,354 | 0 | 1 |
| MtMorgan | 3,290 | 1,257 | 0 |
| Welcome Well | 677 | 90 | 0 |

Negative results may encode below-detection-limit measurements. No universal
detection limit is assumed. Their exclusion means these summaries do not
represent a censoring-aware estimate of the underlying Au distribution.

## Matched Model Cohort

The prepared SWIR/TIR common cohort requires an unambiguous assay whose two
interval boundaries match within the documented tolerance, measured
nonnegative Au, and at least one usable descriptor in each spectral group.

| Project | Holes | Model intervals | Median Au (ppm) | Mean Au (ppm) | Maximum Au (ppm) | Intervals >= 1 ppm |
|---|---:|---:|---:|---:|---:|---:|---:|
| BADGEBUP | 9 | 791 | 0.110 | 2.338062 | 438.000 | 339 |
| BGM Exploration 2007-2008 | 4 | 39 | 0.010 | 0.068205 | 0.510 | 0 |
| MtMorgan | 4 | 1,177 | 0.005 | 0.011127 | 0.600 | 0 |
| Welcome Well | 5 | 321 | 0.003 | 0.004882 | 0.066 | 0 |

There are **339 Anomaly intervals and 1,989 Normal intervals** in the 2,328-row
model cohort. All Anomaly intervals are from BADGEBUP. BADGEBUP therefore accounts
for all positives, but not all BADGEBUP intervals are positive: 452 are Normal.
Most positives are concentrated in BSDD011-014 (336 of 339).

The initial 9,379 mineral-log rows yield 2,328 retained intervals (24.8%).
Unmatched interval boundaries and missing/censored results substantially change
the project and grade mix. The absence of BGM/MtMorgan positives after matching
does not establish that those projects lack gold; they contain positive source
assay records. Scanning coverage and interval support must be investigated
before changing the matching policy. Do not replace exact matching with a
containing-interval join without assessing the scientific consequences.

## Why These Holes?

The acquisition query explicitly requires Yilgarn Craton collars, a project
shortlist, a mineral-log reference, and at least one non-null Au assay with an
ordered depth interval. The analysis then fixes 22 holes whose logs follow the
automated SWIR/TIR schema; see `COHORT_DATASET_IDS` in
`src/gswa_hackathon/build_dataset.py`.

The local acquisition manifest contains five additional MacPhersons holes that
are not in this analysis. Their logs have 22 feature columns versus 49 for the
included holes. This is consistent with a schema-compatibility concern, but an
explicit exclusion decision and detailed schema comparison are not documented.
Do not describe them as missing assays or failed downloads.

Geological variety was the team's intended motivation for selecting several
projects. The repository does not yet establish formation-level coverage or
document a systematic geological-diversity selection procedure. The defensible
description is **a deliberately selected, technically compatible pilot cohort**,
not a balanced or representative sample of Yilgarn geology. The exact reason
for choosing every individual hole remains incompletely documented.

## Interpretation and Provenance

Strong performance could partly reflect project, host-rock, measurement, or
sampling differences rather than a transferable Au-associated alteration
signature. Drillhole grouping prevents downhole train/test overlap but does
not remove project confounding. Training-only grouped validation and additional
positive projects are needed before claiming transfer.

Sources, retained locally: GSWA `HyLogger_Dataset_Index.csv`,
`WAMEX_AssayFlatData.csv`, the acquisition manifest, and the prepared
`data/extracted_data/ml_dataset.csv`. This report concerns the selected 22
collars, **not every assay or project in the GSWA package**. Only aggregated
statistics are reproduced here; source rows remain excluded from Git.
