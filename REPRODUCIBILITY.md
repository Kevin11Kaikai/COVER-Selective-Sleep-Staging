# Reproduction contract

## Main evaluation

`artifacts/frozen/outputs/analysis_inputs/frozen_analysis_inputs.npz` contains public dataset identities, primary-error indicators, risk scores, and fixed acceptance masks. `artifacts/frozen/New_Objective_Idea/outputs/{decisions,anchored/decisions}/outer-*.npz` contains the corresponding Linear-BCE, SELE, FixedScale, and BCEAnchor outputs. All NumPy files are loaded with `allow_pickle=False`.

The reproduction aligns every epoch by `(dataset, subject_id, record_id, epoch_index)` and requires complete nonoverlapping OUTER coverage. It recomputes own-rule results and matched per-subject quotas using the saved anchor mask and a common deterministic identity ordering for ties. The historical original-COVER reference remains a diagnostic and is not silently substituted for the final manuscript's Linear-BCE anchor.

For subject `s`, let `n_s` be all epochs, `a_s` accepted epochs, and `k_s` accepted errors. Coverage is `mean(a_s/n_s)`. Subject-equal accepted risk is `mean(k_s/n_s) / mean(a_s/n_s)`. Pooled risk is `sum(k_s) / sum(a_s)`. Zero acceptance does not imply zero risk; an undefined ratio stays undefined.

## Development and nested analysis

The input `per_subject_records.csv` preserves counts for each subject/context, method, and evaluation rule. The replay validates the stored coverage and error mass against those integer counts, then averages each person's repeated appearances before computing the ratio. Twenty FIT contexts each contain 15 subjects; each person's number of appearances need not equal 15. Completeness is checked against the frozen nested-BCE memberships. The context's OUTER subject must not occur among its FIT records.

The replay reconstructs `subject_means.csv`, `summary.csv`, `paired_differences.csv`, and `delete_one_subject.csv`. Saved ranking oracles remain diagnostics. Neither bootstrap resampling nor reaggregation generates a new independent cohort or accounts for the entire preceding research selection process.

## Algorithm and training sources

The core scorer code consumes features derived from primary/auxiliary probabilities and binary primary-error targets. Primary and auxiliary predictions used to fit a risk scorer must respect the subject split; in-sample component predictions are not an equivalent training input. Calibration and selection must also remain within their assigned FIT roles. Synthetic tests check gradients, calibration boundaries, nested held-out-label invariance, deterministic ties, and checkpoint incompatibility.

The original `New_Objective_Idea/run*.py` drivers expect fingerprinted historical development caches under `models/stage1/development/risk/`, historical fitted models and receipts, and the relevant data manifests. Those private/local dependencies are not reconstructed from the saved OUTER outcomes. The public entry point never launches those drivers.

The historical component sources under `archive/source_code/` preserve Conformer, random-forest feature/training, and DuoF implementation details. This archive is for inspection and adaptation, not a claim that its original campaign launchers run in a fresh public clone. The DuoF launcher exposes historical roots through `COVER_STAGE2_ROOT`, `COVER_STAGE3_ROOT`, and `COVER_STAGE4_ROOT` instead of a developer's drive paths. No raw EEG or model weights are included. The unavailable original EDF conversion is documented in the manuscript.

## Provenance and validation

`provenance/import_manifest.json` records each imported file's hash, source-relative origin, and any changes made for this release. Numeric source files come from the same submitted supplement used for the arXiv preparation. The public wrapper checks this manifest before computing results. It records package versions, elapsed time, table sizes and comparison results in `outputs/reproduction/verification.json`.

The default output directory is ignored by Git. A failed check writes `status: FAIL` and exits unsuccessfully. There is no fallback to copying the reference table into the output. Tests create only temporary synthetic state under ignored output directories. The observed historical environment is retained in `requirements/environment_observed.json`; the CPU release uses `requirements.txt` and does not require its historical CUDA dependencies.
