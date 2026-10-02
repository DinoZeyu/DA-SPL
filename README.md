# GlaBoost on GRAPE: fixed-detector longitudinal validation

The study tests whether simple integration of independently generated GlaBoost visit
scores improves glaucoma progression assessment over the latest visit alone, following
[Dr. Zhang's email](Mail%20-%20Han,%20Zeyu%20-%20Outlook.pdf).

The default workflow trains an image-only diagnosis detector on the retained HF source
dataset, then freezes it before applying it to GRAPE. It stays fixed across GRAPE visits,
folds and progression endpoints. The A/B logistic mappings are fitted and patient-cross-
validated within GRAPE. This externally applies a fixed diagnosis detector while
internally validating the progression mappings; it is not independent external validation
of a complete pretrained progression pipeline.

## Current status

**The HF-training workflow is ready to run.** The automatically updated results block
below and [report index](result/INDEX.md) show completed analyses. Run the command below
to train the diagnosis models, apply each fixed model to GRAPE, and generate separate
source and longitudinal evaluation reports. Formal experiments are run by the user.

The preserved HF diagnosis data are at
`/scratch/users/zeyuhan/DA-SPL/archive/glaucoma_diagnosis_json_analysis/`.
The released files contain **589 training records and 100 test records**. Their original
labels are **0 = glaucoma, 1 = normal**; training explicitly converts them to this
project's **0 = normal, 1 = glaucoma**, so the exported positive-class score means
glaucoma. Only the original images are predictors. Descriptions, annotation-derived
features and other unavailable GRAPE modalities are not used.

Released counts differ from the retained analysis counts. Within each source split,
the first occurrence of each identical decoded image is retained. When the same image
appears in both splits, its training occurrence is preserved and matching test rows
are excluded before fitting or evaluation. Conflicting labels stop the run, as does
any exact image overlap between HF and GRAPE. The source report records retained counts
and exclusions; raw files are unchanged.

The read-only audit of the retained release found:

| Source split | Released records | Retained images | Retained glaucoma | Retained normal |
|---|---:|---:|---:|---:|
| Training | 589 | 562 | 352 | 210 |
| Test | 100 | 92 | 45 | 47 |

The audit excluded **27 duplicate training records**, **1 duplicate within the test
split**, and **7 test records matching training images**. No conflicting labels or
exact decoded-image matches with any of the **631 original GRAPE CFPs** were found.
These are data-audit counts, not model results; zero exact-image overlap does not
establish patient-level independence.

This produces a **GlaBoost-compatible reconstruction trained on a different source
dataset**, not the authors' fitted model or a reproduction of their reported 99%+
performance.

## Latest completed primary analysis

This block displays the **predeclared primary model** from the latest completed batch,
never the model with the largest observed GRAPE metric.

<!-- glaboost-results:start -->

Latest completed run: `grape_validation/paper100_depth6` (updated on completion, never selected by performance).
**Cohort:** 58 patients, 105 eyes, and 382 eligible CFP visits.
Median visits per eye: 3.0 (IQR 3.0–4.0); median CFP follow-up: 25.6 months (IQR 19.9–38.4).
[Full report](result/grape_validation/paper100_depth6/report.html) · [Markdown](result/grape_validation/paper100_depth6/report.md) · [All completed reports](result/INDEX.md)

**Primary comparison: balanced accuracy (BA).** A uses the latest visit; B integrates longitudinal scores.
BA and its 95% CI are percentages; the paired B − A difference and its CI are percentage points (pp).

| Endpoint | Eyes | Progressors, n (%) | A: latest BA, % [95% CI] | B: longitudinal BA, % [95% CI] | Paired B − A, pp [95% CI] |
|---|---:|---:|---:|---:|---:|
| PLR2 | 105 | 13 (12.4%) | 49.8 [35.8, 63.1] | 50.3 [36.0, 64.0] | +0.5 [-16.5, +17.2] |
| PLR3 | 105 | 5 (4.8%) | 44.5 [31.0, 67.5] | 51.5 [29.5, 79.0] | +7.0 [-8.3, +28.5] |
| MD slope | 105 | 9 (8.6%) | 29.9 [16.8, 46.3] | 45.1 [27.8, 62.6] | +15.3 [-7.4, +36.1] |

[Primary comparison figure (PNG)](result/grape_validation/paper100_depth6/figures/primary_comparison.png) · [PDF](result/grape_validation/paper100_depth6/figures/primary_comparison.pdf) · [Primary results CSV](result/grape_validation/paper100_depth6/primary_results.csv)

**Secondary metrics:** point estimates on the 0–1 scale. AUPRC uses average precision (AP).
The [supplementary CSV](result/grape_validation/paper100_depth6/supplementary_metrics.csv) and full report include all 95% CIs and reasons for non-estimable results.

| Endpoint | Method | AUROC | AUPRC (AP) | Sensitivity | Specificity | F1 |
|---|---|---:|---:|---:|---:|---:|
| PLR2 | A: latest | 0.470 | 0.134 | 0.615 | 0.380 | 0.205 |
| PLR2 | B: longitudinal | 0.483 | 0.141 | 0.538 | 0.467 | 0.203 |
| PLR3 | A: latest | 0.378 | 0.047 | 0.200 | 0.690 | 0.054 |
| PLR3 | B: longitudinal | 0.544 | 0.061 | 0.400 | 0.630 | 0.091 |
| MD slope | A: latest | 0.334 | 0.070 | 0.222 | 0.375 | 0.056 |
| MD slope | B: longitudinal | 0.493 | 0.096 | 0.444 | 0.458 | 0.123 |

![Latest versus longitudinal balanced accuracy and paired differences, with 95% patient-bootstrap CIs](result/grape_validation/paper100_depth6/figures/primary_comparison.png)

The visit model uses a frozen ImageNet ResNet152 encoder with 2,048 output features and an XGBoost binary logistic classifier: 100 trees, maximum depth 6, learning rate 0.05, row subsampling 1.0, column subsampling 1.0, L2 1.0, and L1 0.0. Image preprocessing: color RGB; resize [224, 224]; interpolation bilinear; scale divide_by_255; ImageNet mean/std normalization.

The fixed visit detector is declared independent of GRAPE in the supplied provenance. The progression mappings are nevertheless fitted and patient-cross-validated within GRAPE; the complete progression pipeline has not undergone an independent external validation.

Confidence intervals use paired patient-level bootstrap conditional on fixed out-of-fold predictions; they exclude uncertainty from refitting the training pipeline. This is retrospective progression assessment, not forecasting future progression. A positive point estimate for one endpoint does not establish a consistent benefit across all three endpoints.

<!-- glaboost-results:end -->

## Run

From this project directory on your **already allocated GPU node**:

```bash
bash run_grape.sh
```

The script uses the existing `da-spl-repro` Conda environment and visible GPUs. It
trains the source diagnosis models, reports held-out source metrics, then generates
GRAPE visit scores, patient-cross-validated progression metrics and cohort reports.
It requests no resources, installs no packages, creates no venv and has no CPU fallback.
Offline is the default; `--allow-download` permits missing pretrained encoder weights
to be downloaded to scratch. Raw datasets are never downloaded or modified.

Use `--run-name NAME` for a new output name; existing outputs are never overwritten.
To evaluate the saved detectors without repeating source training, provide their plan:

```bash
bash run_grape.sh --plan result/hf_training/external_models.json --run-name grape_recheck
```

See `bash run_grape.sh --help` for runtime options. Formal experiments are run by the user.

## Fixed analysis plan

[configs/hf_training.json](configs/hf_training.json) defines the source and model grid.
All configurations use frozen ResNet152 features, learning rate 0.05, full row/column
sampling and seed 42. The source test set and GRAPE never select a detector configuration.

| Configuration | Trees | Maximum depth | Role |
|---|---:|---:|---|
| `paper100_depth6` | 100 | 6 | Predeclared primary |
| `paper100_depth3` | 100 | 3 | Supplementary |
| `paper500_depth3` | 500 | 3 | Supplementary |
| `paper500_depth6` | 500 | 6 | Supplementary |

A uses the latest visit score. B uses the latest score, last-minus-first change,
OLS slope per year, mean and fraction of scores above 0.5. Both use training-fold
standardization and balanced L2 logistic regression with C = 1 and decision threshold
0.5. Requested CV is 3 patient-grouped folds; both eyes and all visits stay together.
A/B share the same eligible eyes, labels, folds and fixed detector outputs.

Eyes need at least three original CFP visits. Missing images are omitted; VF values,
baseline-only OCT, text and clinician ratings are not predictors. PLR2, PLR3 and MD
slope are evaluated separately. Balanced accuracy and paired B − A are primary;
AUROC, average precision, sensitivity, specificity and F1 are also reported.

Intervals use 2,000 paired patient-bootstrap draws conditional on fixed out-of-fold
predictions; they exclude refitting uncertainty and are not adjusted for multiple
comparisons. Supplementary configurations are exploratory, with no automatic winner.
Source patient IDs are unavailable: duplicate checks cannot establish patient separation
or exclude near duplicates. Diagnosis-test and progression metrics answer different
questions. Scores are not calibrated clinical risks; assessment is retrospective,
and the released VF label window can extend beyond the final available CFP.

## Reports and storage

Current reports: [HF diagnosis training](result/hf_training/report.html) ·
[GRAPE comparison](result/grape_validation/report.html) · [Report index](result/INDEX.md).
The automatic results block above presents the predeclared primary analysis.

| Location for a new run | Contents |
|---|---|
| `result/<run>_source/` | Source counts, exclusions, diagnosis metrics and training provenance |
| `result/<run>/` | All-configuration comparison, per-model cohort reports, figures, CSV/JSON audit records and code snapshot |
| `artifacts/<run>_source/` | Trained diagnosis models and frozen image features |
| `artifacts/<run>/` | GRAPE visit scores and fitted progression mappings |

Reports stay in home. `artifacts` and `.cache` point to `/scratch/users/zeyuhan/DA-SPL/`.
Current report folders have shorter display names; their recorded run IDs and scratch
artifact paths remain unchanged. Raw data, PDFs, Git history and Conda are retained.

## Environment

Conda provides Python 3.9.23; uv locks dependencies without creating a venv:

```bash
conda activate da-spl-repro
UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX" UV_CACHE_DIR="$PWD/.cache/uv" uv sync --locked --inexact --python "$CONDA_PREFIX/bin/python"
```

The wrapper uses the installed environment and does not synchronize it. The pinned
XGBoost 1.7.6 wheel has an upstream Python-tag warning; uv may reinstall it during sync.
Reusable source training, fixed scoring, longitudinal evaluation and reporting code
lives in `src/glaboost/`. Software tests use temporary synthetic data.
