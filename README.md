# COVER: Learning to Accept More in Selective Sleep Staging

Research code and frozen-result reproduction for **COVER**.

**Yukai Song, Yangfan Deng, Jijun Yin, Zhi-Hong Mao, Jingtong Hu**

Yukai Song, Jijun Yin, Zhi-Hong Mao, and Jingtong Hu: Department of Electrical and Computer Engineering, University of Pittsburgh. Yangfan Deng: Department of Electrical and Computer Engineering, University of Maryland, College Park.

COVER keeps a primary sleep classifier's predicted label and learns which epochs to accept. A 37-dimensional risk vector combines primary probabilities, auxiliary random-forest probabilities, and their disagreement. Linear-BCE learns primary error; FixedScale refines that scorer toward greater soft coverage under a soft risk constraint. A separate, empirical hard-rule selection step determines acceptance.

This release accompanies the **20-page manuscript prepared for arXiv in September 2026**. The public paper identifier will be added after arXiv announcement. This repository does not imply conference acceptance.

## Run the reproduction

Use Python 3.10 or 3.11. A CPU is sufficient; no EEG download or model weights are needed for the commands below.

```bash
git clone https://github.com/Kevin11Kaikai/COVER-Selective-Sleep-Staging.git
cd COVER-Selective-Sleep-Staging
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python Scripts/reproduce.py
python -m unittest discover -s New_Objective_Idea/tests -v
```

The reproduction checks imported-file hashes, recomputes the main comparisons from 42,308 saved epoch scores and acceptance masks, reconstructs the matched per-subject acceptance budgets, and reaggregates 28,800 saved development records. It also recomputes descriptive paired subject bootstrap intervals (10,000 draws; seed 20260901). Eight output tables are checked against the submitted reference files with absolute numeric tolerance `1e-12`. A mismatch gives a nonzero exit code.

Outputs appear in `outputs/reproduction/`: `verification.json`, the `main/` and `development/` tables, and a plot of the recomputed operating points in `figures/`. `--output PATH` selects a separate output directory; `--no-figure` skips plotting. The numerical figure is redrawn from recomputed data; it does not reproduce the paper's layout or conceptual diagram pixel for pixel.

## Reproduction scope

| Included and executable | Scope |
|---|---|
| Main comparison and matched acceptance budgets | Recomputed from saved scores, primary-error indicators, and fixed masks; does not refit or reselect the models |
| Development/nested tables and descriptive intervals | Reaggregated from saved subject/context counts; repeated appearances are averaged per person before taking the risk ratio |
| Linear-BCE, SELE, soft-risk and fixed-scale solvers | Original implementation, analytic gradients, and synthetic numerical tests |
| Calibration, deterministic ties, hard-rule selection, nested selection | Original code and protocol tests, including held-out-label invariance |
| Historical Conformer, RF and DuoF sources | Reference implementation retained under `archive/source_code/` |

**This is not a complete raw-EDF-to-paper training package.** Raw EEG, component weights, trusted development caches, and some decision-replay inputs are not included. The original EDF-to-NPZ conversion script was not recovered, so exact end-to-end preprocessing reproduction remains unavailable. Historical experiment drivers are included for inspection, but require their original inputs; `Scripts/reproduce.py` is the supported self-contained reproduction entry point. No new component training is performed by that command.

The stored `CERT` name denotes the empirical calibration/rule-selection split. It does **not** mean that the reported risks have a population guarantee. The nested analysis uses the same historical cohort and is development evidence, not independent external validation. Reported uncertainty is descriptive.

## Selected results

Main evaluation uses Sleep-EDF-20: 20 subjects, 39 recordings, and 42,308 epochs. `C` is mean subject coverage; `R_SE` is the ratio of mean accepted-error mass to mean accepted mass; `R_pool` is the pooled accepted error rate. These denominators are different.

| Scorer | C (%) | R_SE (%) | R_pool (%) | Accepted epochs |
|---|---:|---:|---:|---:|
| MSP | 31.45 | 3.94 | 4.15 | 13,625 |
| Forward-KL | 30.56 | 5.17 | 5.17 | 12,935 |
| Primary-only | 36.24 | 3.59 | 3.67 | 15,603 |
| DuoF | 43.62 | 4.29 | 4.36 | 18,598 |
| Linear-BCE | 48.79 | 4.15 | 4.20 | 20,639 |
| FixedScale | 50.40 | 4.53 | 4.58 | 21,455 |
| SELE | 49.35 | 4.31 | 4.35 | 20,875 |
| BCEAnchor | 49.22 | 4.06 | 4.11 | 20,805 |

At Linear-BCE's exact per-subject acceptance budgets (20,639 total), accepted errors are 867 for Linear-BCE, 890 for FixedScale, 868 for SELE, and 857 for BCEAnchor. The own-rule coverage gain is therefore not a uniform matched-budget error improvement. The nested locked-rule comparison is FixedScale-1 `C=50.21%, R_SE=4.45%` versus Linear-BCE `C=48.56%, R_SE=4.84%`.

## Code map

| Path | Purpose |
|---|---|
| `Scripts/reproduce.py` | Portable entry point and reference checks |
| `Scripts/recompute_main.py` | Final manuscript's Linear-BCE-anchored analysis |
| `Scripts/reproduce_nested.py` | Reaggregation of saved development records |
| `src/stage1_analysis/risk_features.py` | 37-dimensional primary/auxiliary risk features |
| `src/stage1_analysis/acceptance.py` | Subject weighting, deterministic ties, hard-rule selection |
| `New_Objective_Idea/src/new_objective/` | Losses, solvers, calibration, main and nested experiment orchestration |
| `New_Objective_Idea/*protocol.json` | Frozen experimental specifications |
| `AutoML_Claude/automl/grids.py` | Exact threshold-quantile grid utility used by the scorer experiments |
| `archive/source_code/` | Historical primary, RF, and DuoF implementation references |
| `artifacts/frozen/` | Numeric inputs copied from the submitted supplement |
| `artifacts/expected/main/` | Final manuscript comparison tables used as reference outputs |
| `provenance/import_manifest.json` | Imported source/input hashes and explicit portability changes |

The historical directory name `AutoML_Claude` is retained to preserve imports; only its threshold-grid utility is included. Later backbone and cross-setting experiments are outside this release.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the data contract and evidence boundaries, and [NOTICE.md](NOTICE.md) for source attribution. Code is distributed under the inherited [GNU GPL v3 license](LICENSE). Citation metadata is in [CITATION.cff](CITATION.cff).
