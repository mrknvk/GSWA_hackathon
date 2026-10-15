# Yilgarn Cohort Expansion

## Decision

Keep the original 22-hole pilot fixed. Build the expanded cohort as a separate,
auditable artifact, selected by data availability rather than positive grades.
No new training or test-based tuning is part of expansion.

## Responsibilities

- `yilgarn_inventory.py`: one row per Yilgarn collar with finite nonnegative Au
  and valid assay intervals; map positions in GDA2020; resolve primary HyLogger
  mappings and explicitly exclude ambiguous mappings from acquisition.
- `acquire_mineral_logs.py`: trusted DASC requests and its exact official Azure
  mineral-log redirect; bounded responses/concurrency, ordinary TLS validation,
  validated caches, atomic no-overwrite installs, checksummed local manifest.
- `build_dataset.py`: scientific numeric validation and interval matching;
  expanded preparation reuses pilot rules and reports outcomes per hole.
- `expand_yilgarn.py`: orchestrates local exports, complete eligibility audit,
  and project support tables. Does not train models or overwrite pilot exports.
- `yilgarn_inventory.ipynb`: reads local artifacts for interpretation.

Dependencies flow from orchestration to inventory/acquisition/preparation.
Inventory and preparation do not initiate network access. Network requests
require the explicit `--download` CLI option. No map tiles or new external
mapping dependency are needed; the static map is not a craton boundary map.

## Constraints

Primary `WAMEX_CollarId` is the join key. Multiple candidate names, URLs or
project mappings are not resolved by arbitrary first-row selection. Unknown
projects remain explicitly unknown; this bucket is not a geological domain.

Measured Au availability is assessed before spectral matching and is not proof
of usable training coverage. Logs must contain all ten existing predictor
columns; no missing band is fabricated. Individual missing values remain NaN.
Both depth boundaries must satisfy the original absolute tolerance; conflicts,
invalid intervals, negative/censored Au, and missing Au follow unchanged rules.
Sorted start-depth windows bound candidate matching memory without changing
the final absolute-boundary comparison or conflict resolution.

Do not merge every source assay with every log interval simply to enlarge the
sample count. Broader overlap/aggregation matching would be a separate
scientific decision, requiring support and sensitivity analysis.

## Outputs and Evaluation

All source data, collar coordinates, model-ready rows and manifests remain
under ignored `data/`. Only reviewed aggregated reports may be published.
Notebook outputs must be cleared before committing. The CLI can be rerun to
reuse valid downloads and reattempt failures; every hole remains in the audit.

Changing the cohort changes the split. Expanded-cohort model results are not
an independent confirmation of the repeatedly inspected pilot test. Future
evaluation needs grouped training validation, a frozen holdout, and assessment
of project-level confounding and sampling/censoring effects.
