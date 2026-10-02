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

The previous GRAPE-trained progression experiments have been retired from this
workflow. Their results are not evidence of external validation.

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

[All completed validation batches](result/INDEX.md)

## Run source training and GRAPE evaluation

From this project directory on your **already allocated GPU node**:

```bash
bash run_grape.sh
```

The default [HF training plan](configs/hf_training.json) fixes the source data,
ResNet152 encoder and model configurations before examining GRAPE results. The command:

1. Checks the retained source files and image overlap, then extracts frozen ResNet152 features.
2. Fits XGBoost diagnosis models using the source training split only and saves their parameters.
3. Reports diagnosis metrics on the retained source test rows, without using those metrics to select models.
4. Applies each saved, fixed detector independently to eligible GRAPE visits.
5. Fits the patient-cross-validated A/B progression mappings and generates the comparison reports.

| Role | Trees | Maximum depth | Learning rate | Row / column subsampling |
|---|---:|---:|---:|---:|
| Primary: `paper100_depth6` | 100 | 6 | 0.05 | 1.0 / 1.0 |
| Supplementary | 100 | 3 | 0.05 | 1.0 / 1.0 |
| Supplementary | 500 | 3 | 0.05 | 1.0 / 1.0 |
| Supplementary | 500 | 6 | 0.05 | 1.0 / 1.0 |

All configurations use the same frozen ResNet152 features and seed 42. Each configuration
produces one diagnosis detector shared across the three GRAPE progression outcomes.
The primary designation is fixed in advance; no configuration is selected automatically
from source-test or GRAPE performance. Supplementary comparisons are exploratory.

Optional naming:

```bash
bash run_grape.sh --run-name hf_grape_seed42
```

The script uses the existing `da-spl-repro` Conda environment and visible GPUs. It does
not request Slurm resources, create a venv, install packages or fall back to CPU. Offline
is the default; `--allow-download` explicitly permits missing pretrained encoder weights
to be fetched into the scratch cache. Original datasets are never downloaded or modified.
Names are generated automatically when omitted, and existing directories are never
overwritten. See `bash run_grape.sh --help` for runtime options.

The source dataset has no usable patient identifiers for verifying patient separation
between its released training and test splits. Exact decoded-image duplicate checks
and exclusions provide a limited overlap audit: they cannot establish patient-level independence or
exclude near-duplicate images. Source-test diagnosis performance and GRAPE progression
performance answer different questions and are reported separately.

## Use an existing independently trained detector instead

To skip source training, populate [the external model plan](configs/external_models.json)
and run:

```bash
bash run_grape.sh --plan configs/external_models.json
```

This optional plan starts empty. Each detector bundle contains
native `model.json` and `metadata.json` files from `glaboost.GlaBoost.save()`. The runner
checks classifier/encoder metadata, checksums, image-only frozen ResNet152 compatibility,
feature schema and diagnosis label meaning.

Each entry in `models` must contain:

| Field | Required content |
|---|---|
| `name` | Unique name identifying the fitted configuration |
| `model_directory` | Bundle path; relative paths resolve from the plan file |
| `training_data.description` | Actual diagnostic training and model-selection data |
| `training_data.reference` | Dataset revision, training record or other traceable source |
| `training_data.grape_overlap` | `none`, supported by the supplied training history |
| `training_data.independence_evidence` | Why training, preprocessing and selection exclude GRAPE |

Set `primary_model` to a listed name before inspecting its GRAPE results. Other models
are supplementary comparisons. For example, 100/500 trees and depth 3/6 require four
separately fitted external models. Prediction options cannot change learned trees.
Do not select the best GRAPE result and describe it as unaffected by that selection.

The runner records supplied independence evidence; checksums establish integrity,
not patient-level independence. Unknown training provenance cannot enter the declared
external-validation workflow. Senior bundles with different schemas, label polarity
or formats need an explicit verified adapter; renaming metadata is insufficient.

## Analysis specification

| Component | Specification |
|---|---|
| Cohort | Eyes with at least 3 original CFP visits; actual elapsed time retained |
| Visit evidence | Source-trained frozen ResNet152 + XGBoost; continuous glaucoma-class output, not a calibrated clinical risk |
| A: latest | L2 logistic mapping of the latest visit score |
| B: longitudinal | Latest score, last-minus-first change, OLS slope/year, mean, persistence |
| Persistence | Fraction of scores strictly above 0.5, fixed in advance |
| Reference outcomes | PLR2, PLR3 and MD slope evaluated separately |
| Splitting | Patient-grouped stratified CV; both eyes and all visits stay together |
| Logistic mappings | Training-fold scaling, balanced class weights, L2, C = 1 |
| Decisions | Threshold 0.5; class-weighted outputs are not calibrated clinical risks |
| Primary metric | Balanced accuracy and paired B minus A difference |
| Other metrics | AUROC, AUPRC (average precision), sensitivity, specificity, F1 |
| Uncertainty | 2,000 paired patient-bootstrap draws, conditional on fixed OOF predictions |
| Reproducibility | Seed 42, requested 3 folds, actual memberships and environment saved |

A/B use the same eligible eyes, labels, folds and fixed visit scores. Only the logistic
mappings are fitted on GRAPE. No GRAPE labels fit or select the diagnosis detector.
Missing CFP visits are omitted consistently. Baseline OCT, unavailable text and human
ratings are not synthesized or copied to later visits; VF measurements are not predictors.

This is retrospective assessment. The last CFP can precede the end of the original
GRAPE label window, which reports disclose. It does not establish a future prediction
horizon or persistent longitudinal clinical reasoning. Positive, negative and inconclusive
results are reported alike. CIs exclude model-refitting uncertainty and multiplicity adjustment.

## Reports and storage

| Location | Contents |
|---|---|
| `result/<run>_source/` | Source-training specification, diagnosis-test metrics and overlap/provenance summary |
| `result/<run>/report.html` and `report.md` | Primary designation and all-configuration comparison |
| `result/<run>/<model-name>/report.html` | Complete cohort report for each fixed detector |
| Configuration report directory | Cohort/prevalence, mapping, model/CV specification, metric tables, three-endpoint figure, exclusions, provenance |
| `result/<run>/` | Immutable plan, status, comparability audit, code snapshot and numeric tables |
| `artifacts/<run>_source/` | Source-trained model parameters and frozen image features on scratch |
| `artifacts/<run>/` | GRAPE visit scores, fitted progression mappings and numerical artifacts on scratch |
| Supplied external model directories | Optional existing model parameters; not refitted or overwritten during validation |

Reports remain in home. `artifacts` and `.cache` point to `/scratch/users/zeyuhan/DA-SPL/`;
store all model parameters there as well. `data/raw/grape` points to the preserved raw
cohort. HF raw data, PDFs, Git history and the Conda environment are protected during cleanup.

## Environment and reusable code

Use Conda for Python and uv for locked packages, without creating a venv:

```bash
conda activate da-spl-repro
UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX" UV_CACHE_DIR="$PWD/.cache/uv" uv sync --locked --inexact --python "$CONDA_PREFIX/bin/python"
```

Python 3.9.23 and package versions are pinned in `pyproject.toml` and `uv.lock`.
The existing XGBoost 1.7.6 wheel has an upstream Python-tag packaging warning; repeated
uv sync may reinstall it. The wrapper does not run environment synchronization.

`src/glaboost/` contains source diagnosis training, fixed-model scoring, raw-data mapping,
longitudinal evaluation, GPU numerical routines, provenance and reporting. `external.py`
runs the fixed-model evaluation plan. Development tests use temporary synthetic data only.
`Glaboost_CH.py` remains the senior-code reference; its original ResNet18 multimodal
interface is not the GRAPE validation entry point. Obsolete GRAPE-label XGBoost training
and internal tree-comparison entry points have been removed.
