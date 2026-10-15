# Yilgarn Expansion: First Completed Audit

Snapshot of the local GSWA package processed with the existing 1 ppm Au target,
ten descriptors, common SWIR/TIR coverage, and absolute boundary tolerance of
1e-6 m. This is an availability and preparation audit, not a new model result.
The original pilot exports and model settings were not replaced.

## Inventory Before Spectral Matching

| Population | Collars | Measured source assay records | Records >= 1 ppm |
|---|---:|---:|---:|
| All measured-Au Yilgarn collars in the package | 665 | 132,145 | 4,505 |
| Primary HyLogger match and mineral-log link | 563 | 105,698 | 3,197 |
| Original 22-hole pilot before matching | 22 | 5,487 | 491 |

Counts require finite nonnegative Au, finite nonnegative start depth, and end
depth greater than start. Source assay records are not necessarily unique
physical samples. Negative/censored values are excluded, not substituted.
These are local-package counts, not an inventory of every Yilgarn assay ever
collected. All 665 measured-Au collars have GDA2020 geographic coordinates.

## Complete Outcome Accounting

| Final outcome | Collars | Meaning |
|---|---:|---|
| Ready | 457 | At least one usable common-cohort interval |
| No retained intervals | 91 | Compatible log, no interval survives matching/coverage |
| Incompatible schema | 10 | Required descriptor columns absent |
| Ambiguous source mapping | 3 | Multiple log identities share a collar; not chosen arbitrarily |
| Acquisition validation failed | 2 | Declared Dataset differs from the index identity |
| No primary HyLogger mapping | 102 | Gold assays exist, but no usable primary index join |
| **Total** | **665** | Every measured-Au collar accounted for |

There are 560 unambiguous acquisition candidates. Of these, 558 validated logs
are available locally: the 22 original logs and 536 newly acquired files. The
two remaining identity conflicts must be resolved against source metadata,
not by assuming the index or file is correct.

All five MacPhersons logs and four Austin logs lack the required TIR descriptors;
one other log lacks the water descriptor. Those bands were not fabricated.
These holes could support a separately defined SWIR-only experiment, but do
not belong in the current aligned three-feature-set comparison.

The first network attempt was rejected with HTTP 403. Using the historical
client headers and allowing the exact official DASC Azure mineral-log redirect
resolved access. Ordinary TLS verification and a strict host/path allowlist
remain enabled. A transient DNS failure succeeded on the subsequent cached run.

## Resulting Model-Ready Cohort

| Measure | Pilot | Expanded |
|---|---:|---:|
| Retained collars | 22 | 457 |
| Common-cohort intervals | 2,328 | 46,336 |
| Intervals >= 1 ppm | 339 | 1,067 |
| Intervals < 1 ppm | 1,989 | 45,269 |
| Collars containing retained Au >= 1 ppm | 7 | 142 |

All 2,328 original pilot rows are preserved identically in the expanded table
when selecting the same predictors and metadata. There are 435 additional
usable holes. No imputation, scaling, splitting or training was performed as
part of this expansion.

Selected groups with retained positive examples:

| Project label | Holes | Model intervals | Intervals >= 1 ppm |
|---|---:|---:|---:|
| Unknown | 313 | 33,653 | 482 |
| BADGEBUP | 9 | 791 | 339 |
| SS | 3 | 782 | 105 |
| WALLABYRES | 1 | 357 | 38 |
| NORTH | 2 | 186 | 29 |
| ECLIPSE | 2 | 105 | 14 |
| KCGM Fimiston CHUD022 | 1 | 65 | 8 |
| Manindi | 3 | 117 | 6 |
| LEFROY | 1 | 197 | 5 |
| KATANNING | 3 | 72 | 5 |

There are 72 distinct named labels in the retained cohort, 26 of which contain
positive examples. Unknown is a missing-project bucket, not one project,
deposit, geological formation, or independent validation group. Name variation
may also prevent these counts from equalling the number of geological projects.

## Scientific Limits and Next Evaluation

- Anomalies are approximately 2.3% of expanded intervals, versus 14.6% in the
  pilot. More data does not guarantee better metrics; the screening population
  and imbalance have changed substantially.
- Interval matching, spectral coverage and schema requirements still remove
  source positives. The local per-hole audit records each exclusion stage.
- Positive examples now extend beyond BADGEBUP, but this does not eliminate
  project or host-rock confounding. Unknown-project provenance needs attention.
- Re-establish a grouped evaluation protocol before training on the expanded
  table. Changing cohorts and test holes makes old and new scores incomparable
  as an isolated model improvement. Do not select a split by model performance.

## Local Artifacts and Reproduction

Run `PYTHONPATH=src uv run python -m gswa_hackathon.expand_yilgarn --download`.
Valid cached downloads are reused; source failures remain explicit. The output
directory is `data/extracted_data/yilgarn/` and remains Git-ignored.

It contains `hole_inventory.csv`, `candidate_sources.csv`,
`mineral_log_manifest.csv`, `build_audit.csv`, `numeric_issues.csv`,
`hole_audit.csv`, `project_summary.csv`, `ml_dataset_expanded.csv`, and the
three-panel `yilgarn_gold_map.png`. The map shows median/max Au and source-assay
positive fraction, with the original pilot and linked holes marked separately.
It depicts collar positions without background tiles or a craton boundary;
it is not a subsurface mineralisation or resource map.

Open `ml_pipeline/yilgarn_inventory.ipynb` to inspect the map and audit. This
report publishes aggregated counts only, not source or sample-level records.
