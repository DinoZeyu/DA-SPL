# GlaBoost on GRAPE: longitudinal progression assessment

Compare **the latest encounter alone** with **simple longitudinal integration** of
fundus-image evidence for PLR2, PLR3, and MD-slope progression. The image pipeline
uses a **frozen ImageNet ResNet-152 encoder and XGBoost**.

**Study design: preliminary retrospective internal validation on GRAPE, using
nested patient-level cross-validation.** This is a progression-task adaptation of
the paper's architecture. Visit models are trained on GRAPE training patients;
this run does not establish external validation of the authors' fixed diagnostic model.
Only the retained local GRAPE dataset is used. Formal experiments are run by the user.

[Metric evaluation](#metric-evaluation) · [PLR3 example](#presentation-example-plr3) ·
[Run the pipeline](#run-the-pipeline) · [Methods](#model-and-validation) ·
[Environment](#environment-and-reproducibility)

## Metric evaluation

<!-- glaboost-results:start -->

Latest completed run: `primary_seed42_fixed` (updated on completion, never selected by performance).
**Cohort:** 58 patients, 105 eyes, and 382 eligible CFP visits.
Median visits per eye: 3.0 (IQR 3.0–4.0); median CFP follow-up: 25.6 months (IQR 19.9–38.4).
[Full report](result/primary_seed42_fixed/report.html) · [Markdown](result/primary_seed42_fixed/report.md) · [All completed reports](result/INDEX.md)

**Primary comparison: balanced accuracy (BA).** A uses the latest visit; B integrates longitudinal scores.
BA and its 95% CI are percentages; the paired B − A difference and its CI are percentage points (pp).

| Endpoint | Eyes | Progressors, n (%) | A: latest BA, % [95% CI] | B: longitudinal BA, % [95% CI] | Paired B − A, pp [95% CI] |
|---|---:|---:|---:|---:|---:|
| PLR2 | 105 | 13 (12.4%) | 53.0 [38.5, 67.7] | 46.9 [34.1, 61.1] | -6.1 [-16.9, +3.5] |
| PLR3 | 105 | 5 (4.8%) | 45.0 [22.1, 73.8] | 61.0 [31.7, 82.6] | +16.0 [+1.5, +37.8] |
| MD slope | 105 | 9 (8.6%) | 64.8 [51.5, 74.2] | 62.3 [45.7, 75.8] | -2.4 [-15.8, +6.4] |

[Primary comparison figure (PNG)](result/primary_seed42_fixed/figures/primary_comparison.png) · [PDF](result/primary_seed42_fixed/figures/primary_comparison.pdf) · [Primary results CSV](result/primary_seed42_fixed/primary_results.csv)

**Secondary metrics:** point estimates on the 0–1 scale. AUPRC uses average precision (AP).
The [supplementary CSV](result/primary_seed42_fixed/supplementary_metrics.csv) and full report include all 95% CIs and reasons for non-estimable results.

| Endpoint | Method | AUROC | AUPRC (AP) | Sensitivity | Specificity | F1 |
|---|---|---:|---:|---:|---:|---:|
| PLR2 | A: latest | 0.548 | 0.136 | 0.462 | 0.598 | 0.214 |
| PLR2 | B: longitudinal | 0.447 | 0.124 | 0.308 | 0.630 | 0.157 |
| PLR3 | A: latest | 0.486 | 0.068 | 0.400 | 0.500 | 0.070 |
| PLR3 | B: longitudinal | 0.578 | 0.105 | 0.600 | 0.620 | 0.130 |
| MD slope | A: latest | 0.600 | 0.126 | 0.889 | 0.406 | 0.216 |
| MD slope | B: longitudinal | 0.699 | 0.194 | 0.778 | 0.469 | 0.209 |

![Latest versus longitudinal balanced accuracy and paired differences, with 95% patient-bootstrap CIs](result/primary_seed42_fixed/figures/primary_comparison.png)

This is **retrospective internal validation on GRAPE**: ResNet-152 features and XGBoost are adapted to three separate progression endpoints. Inner patient-held-out scores train A/B mappings; outer held-out patients supply the final estimates. This is not external validation of a pretrained diagnosis model.

Confidence intervals use paired patient-level bootstrap conditional on fixed out-of-fold predictions; they exclude uncertainty from refitting the training pipeline. This is retrospective progression assessment, not forecasting future progression. A positive point estimate for one endpoint does not establish a consistent benefit across all three endpoints.

<!-- glaboost-results:end -->

## Presentation example: PLR3

**Illustrative run: `primary_seed42_fixed`.** This example remains tied to that run
even when the latest-run summary above is refreshed. PLR3 is highlighted because it
had the largest positive paired balanced-accuracy difference in this run. All three
prespecified outcomes remain in the evaluation tables above.

For PLR3, longitudinal balanced accuracy was **61.0%**, compared with **45.0%** for
the latest encounter: **+16.0 percentage points (95% CI +1.5 to +37.8)**.
There were **5 progression-positive eyes from 5 patients**, among 105 eligible eyes.
This is an exploratory signal in a small positive subgroup. PLR2 and MD slope did
not show the same improvement, and overall predictive performance remains limited.

The PLR3 decisions at the prespecified output threshold of 0.5 were:

| Actual PLR3 outcome | A: latest encounter | B: longitudinal integration |
|---|---:|---:|
| Progression detected correctly (true positives) | 2 / 5 | 3 / 5 |
| Progression missed (false negatives) | 3 / 5 | 2 / 5 |
| Non-progression identified correctly (true negatives) | 50 / 100 | 62 / 100 |
| Non-progression classified as progression (false positives) | 50 / 100 | 38 / 100 |

### A held-out eye: `58_OS`

This eye had four eligible CFP visits over **3.64 years (43.7 months)** and a released
PLR3 progression label of **1**. Patient `58`, including both eyes and all visits,
was held out of the corresponding outer-fold model and integration-layer training.
The same outer-fold visit model independently scored each image:

| Original visit number | Years since baseline | Visit-model score |
|---|---:|---:|
| 1 | 0.000 | 0.0644 |
| 2 | 0.805 | 0.0170 |
| 4 | 2.366 | 0.0118 |
| 5 | 3.642 | 0.0097 |

Original visit numbers are retained; only visits with an available original CFP are
included, and slopes use elapsed years rather than visit numbers.

| Assessment | Information used | Final model output | Decision at 0.5 | Matches PLR3 label? |
|---|---|---:|---|---|
| A: latest encounter | Last visit-model score | 0.488 | Non-progression | No |
| B: longitudinal integration | Last score, change, time slope, mean, persistence | 0.906 | Progression | Yes |

The visit scores and final integration outputs are different quantities. The final
outputs are **not calibrated clinical risk probabilities**. The declining visit-score
sequence does not demonstrate clinical improvement; the integration layer uses learned
associations across all five features. This example illustrates a corrected decision,
without establishing an anatomical explanation or the time progression began.

**Selection disclosure:** this successful case was selected after inspecting the
held-out predictions, solely to explain the computation. It is not a representative
sample or independent evidence of benefit. Cohort-level conclusions use every eligible
eye, including errors. The report's separate trajectory figure retains its original
ID-based selection rule.

Source records:
[eye predictions](result/primary_seed42_fixed/predictions.csv),
[visit predictions](result/primary_seed42_fixed/visit_predictions.csv),
[temporal features](result/primary_seed42_fixed/temporal_features.csv), and
[fold assignments](result/primary_seed42_fixed/evaluation.json).

## Run the pipeline

From this project directory on an allocated GPU compute node:

```bash
bash run_grape.sh
```

The script uses the existing **`da-spl-repro` Conda environment** and all CUDA GPUs
visible to the job. It does not install packages or create a virtual environment.
The default CUDA path fails explicitly if CUDA is unavailable. It performs:

1. Local cohort inspection and verification of original input files.
2. Frozen 2,048-dimensional ResNet-152 feature extraction for each eligible image.
3. XGBoost fitting within nested patient partitions.
4. Latest/longitudinal logistic fitting, metrics, and patient bootstrap intervals.
5. Report, figure, and model output; refresh of the report index and README results.

To specify a new run name or require cached weights:

```bash
bash run_grape.sh --run-name my_grape_run --offline
```

Without a name, the script uses a UTC timestamp plus process ID. Existing runs are
never overwritten. Relative paths resolve from the script's project directory.
A rerun with a new name repeats feature extraction and fitting rather than resuming
partial models. Failures during training/report generation save
`artifacts/<run>/error_traceback.txt`; `status.json` records the failed stage and log path.

The first run may download the official approximately 230 MiB ImageNet V1 ResNet-152
checkpoint to `.cache/glaboost/torch/`; subsequent runs reuse it. GRAPE originals are
read without modification or additional downloads. A local checkpoint can also be used:

```bash
bash run_grape.sh --offline --image-weights /path/to/resnet152-394f9c45.pth
```

The official SHA256 prefix `394f9c45` is checked. A GRAPE-fine-tuned encoder cannot
replace the outcome-independent encoder before patient splitting.
See `bash run_grape.sh --help` for all options.

### GPU execution and progress

| Stage | Execution |
|---|---|
| Frozen image features | PyTorch CUDA; `DataParallel` across visible GPUs |
| Visit-model training/prediction | XGBoost 1.7.6 `gpu_hist` / `gpu_predictor`, explicit `gpu_id` |
| Temporal summaries, scaling, logistic heads | PyTorch CUDA float64 |
| Metrics, patient bootstrap, percentile intervals | PyTorch CUDA float64 |
| Input/output, patient partition indices, report rendering | Host processing |

The default global image batch is **64 × visible GPU count: 128 on two A100s**.
Use `--image-batch-size N` to override it or `--device cuda:0` to select one logical GPU.
Device selection respects the scheduler's `CUDA_VISIBLE_DEVICES`. Image encoding
uses float32; mixed precision is not enabled.

With two GPUs, GPU 0 processes PLR2 followed by MD slope, while GPU 1 processes PLR3.
Each device handles one endpoint at a time; a single GPU runs all endpoints serially.
XGBoost configuration is checked after fitting to reject CPU fallback.
`--xgb-threads` controls host helper threads, defaulting to 1.

CUDA linear algebra is initialized sequentially using constant tiny matrices before
endpoint workers start, avoiding PyTorch 2.0.1's
[concurrent first-use loading issue](https://github.com/pytorch/pytorch/issues/90613).
This initialization uses no study observations or random draws.

`tqdm` displays input verification, downloads when needed, feature extraction,
endpoint/fold/model progress, and valid/skipped bootstrap counts.
Set `TQDM_DISABLE=1` to suppress progress bars. GPU and CPU implementations are not
assumed to produce bitwise-identical results or bootstrap samples at the same seed.

## Reports and storage

Code and reports live under `/users/zeyuhan/charlie_codebase/DA-SPL`.
Each report has a real directory at `result/<run>/` in home storage.
Large data, models, features, and caches live under `/scratch/users/zeyuhan/DA-SPL`.

| Project entry | Storage location |
|---|---|
| `result/<run>/` | Home: project `result/<run>/` |
| `data/raw/grape/` | Scratch: `data/raw/grape/` |
| `artifacts/` | Scratch: `artifacts/` |
| `.cache/` | Scratch: `cache/` |
| `.git` pointer file | Scratch: `.git/`, preserving repository history |

Local data/cache links are not included in Git. Migration verified originals using
SHA256; its manifest remains in the scratch project root. The earlier non-GRAPE raw
dataset is retained in the scratch archive and is not used. The existing Conda
environment remains in place. Scratch storage addresses home disk quota, not RAM or
GPU memory capacity. Project package/model/plot caches are directed to `.cache/`.

| Output in `result/<run>/` | Purpose |
|---|---|
| `report.html` | English cohort report with embedded figures; share as one file or print to PDF |
| `report.md` | Editable cohort-level summary and methods |
| `primary_results.csv` | Paired balanced accuracy, differences, intervals, and counts |
| `supplementary_metrics.csv` | AUROC, average precision, sensitivity, specificity, F1, and intervals |
| `figures/` | Paired comparisons and descriptive trajectories, in PNG/PDF/SVG |
| `predictions.csv`, `temporal_features.csv` | Held-out eye predictions and temporal inputs |
| `visit_predictions.csv` | Endpoint/fold-specific inner and outer held-out visit scores |
| `evaluation.json`, `provenance.json` | Splits, coefficients, parameters, source hashes, and environment |
| `cohort.json`, `exclusions.csv` | Cohort statistics, observation windows, and eye-level exclusions |
| `status.json` | Run completion status |

`artifacts/<run>/` stores frozen features, their manifest, and fold-specific models.
`models/<endpoint>/outer_<fold>/` contains inner/outer XGBoost models and metadata;
`heads.json` contains both logistic heads and training-only scaling parameters.
No extra full-cohort model is used to report training-set performance.

Completed runs are listed in [the report index](result/INDEX.md). Generated run folders
are excluded from Git by default: retain/share the matching report directory for
README figure/report links to work on another machine. A complete audit also needs
its artifact directory, code snapshot, and environment lock. Synthetic software-test
outputs stay in temporary directories; incomplete runs are not listed as completed.

## Model and validation

| Component | Primary configuration |
|---|---|
| Image encoder | Frozen ImageNet V1 ResNet-152; 2,048 features before the classification layer |
| Preprocessing | RGB, 224 × 224 bilinear resize, divide by 255, ImageNet normalization, no augmentation |
| Visit classifier | XGBoost, `binary:logistic`, `logloss` |
| Trees | 100 trees, maximum depth 6, learning rate 0.05, L2 = 1, L1 = 0 |
| Sampling / tuning | All rows and columns; seed 42; no search or early stopping |
| Visit weights | Inverse eligible visit count per eye, normalized to mean 1 per training fit |
| A: latest head | Last eligible visit score only |
| B: longitudinal head | Last score, last minus first, OLS slope/year, mean, fraction of scores > 0.5 |
| Both heads | Training-only standardization; balanced L2 logistic regression, C = 1; output threshold 0.5 |
| Validation | Outer 3-fold / inner 2-fold patient-grouped stratified CV; seed 42 |
| Primary metric | Balanced accuracy; paired difference = B minus A |
| Uncertainty | 2,000 paired patient bootstrap draws; percentile 95% CIs |

GRAPE contains glaucoma eyes. This analysis learns **progression versus
non-progression**, separately for the released PLR2, PLR3, and MD-slope definitions.
It does not learn healthy-versus-glaucoma diagnosis. A training visit inherits its
eye's whole-follow-up label: weak supervision, not a known disease state at that visit.
The three outcomes are never combined into one ground truth.

Let `z_it` be the frozen 2,048-dimensional feature vector for visit image `image_it`.
For endpoint `e` and outer fold `k`, the held-out visit score is
`S_it^(e,k) = XGB_(e,k).predict_proba(z_it.reshape(1, -1))[0, 1]`, with XGBoost fitted
on that fold's training patients only. Each held-out eye's whole trajectory uses the same
visit model. A and B share scores, eligible eyes, labels, and partitions.

Within each outer training partition:

1. Inner XGBoost models generate scores for training patients they did not fit on.
2. Those inner out-of-fold scores are summarized per eye; only outer-training patients
   are used to fit the scaler and A/B heads.
3. A fresh XGBoost model fits all outer-training patients and independently scores each
   visit from untouched outer-test patients.
4. Both heads assess those held-out scores; final metrics pool outer-held-out predictions.

Both eyes and all visits from a patient remain together. Patients are stratified by
whether any included eye has the endpoint. Fold counts may decrease only for class
feasibility; a non-estimable endpoint is reported explicitly. No seed or split is chosen
by predictive performance. Frozen ImageNet features can be computed once because the
encoder learns nothing from GRAPE images or outcomes.

The GPU logistic solver uses float64 damped Newton steps and Armijo line search,
at most 2,000 iterations, and absolute gradient infinity norm <= 0.0001. It includes
the intercept in the L2 penalty, matching binary liblinear with `intercept_scaling=1`;
the solver and stopping rule differ. Scaling follows StandardScaler's population
variance and near-constant feature rule.

Bootstrap resamples patients, retains both eyes, and pairs A/B within each draw.
Single-class draws are skipped and counted; fewer than 20 valid draws makes intervals
non-estimable. Torch device RNG draws differ from NumPy PCG64 even at the same seed.
**Intervals condition on fitted out-of-fold predictions; they exclude the full
uncertainty of refitting models or changing CV partitions.**

## Data mapping and interpretation

| GRAPE source / original modality | Use in this analysis |
|---|---|
| `Follow-up / Corresponding CFP` → `extracted/CFPs/` | Original photographs: the sole visit-model input; annotation overlays excluded |
| `Subject Number`, `Laterality` | Patient groups and eye trajectories; not predictors |
| `Visit Number`, `Interval Years` | Visit identity and actual elapsed time; years determine temporal slopes |
| `Baseline / Progression Status / PLR2, PLR3, MD` | Separate eye-level training/reference labels; never image or temporal input features |
| VF measurements | Used by the dataset's reference definitions; excluded from predictor inputs |
| `Follow-up / IOP` | Available for all 382 included visits in the illustrated run; not used |
| Baseline OCT/RNFL measurements | Baseline-only; not copied into later visits or used as predictors |
| Explicit C/D, rim-thinning, and other structured fundus features | Not extracted or supplied as a separate model channel |
| Longitudinal clinical text | Unavailable as mapped longitudinal input; not synthesized |
| Clinician risk/confidence inputs | Unavailable as mapped longitudinal inputs; not synthesized |

The follow-up sheet already includes baseline visits. Missing CFP entries (`/`) are
omitted without fabricating images; an eye needs at least three eligible original CFPs.
In `primary_seed42_fixed`, **158 eyes were excluded for fewer than three CFP visits**.
VF values are not used to reconstruct the reference labels as predictor features.
Later held-out visits never enter an earlier visit's image score.

Follow-up duration is the first-to-last eligible CFP span. In **13 of 105** eyes in the
illustrated run, the last CFP preceded the last recorded visit: the released label
window can extend beyond the image window. Original labels are retained. This is
retrospective assessment, not prediction over a uniform future horizon.

Scores and slopes are not clinical diagnoses, calibrated future risks, severity
measurements, or anatomical explanations. PLR3 has only five positive patients in the
illustrated run; some inner fits have just one. Feasible CV does not establish adequate
precision or clinical readiness. The report's descriptive trajectories select the
first two held-out eyes per endpoint by patient/eye ID without using outcomes.

## Environment and reproducibility

Conda provides **Python 3.9.23**; `uv` locks packages in that existing environment.
To synchronize dependencies when needed:

```bash
conda activate da-spl-repro
export UV_CACHE_DIR="$PWD/.cache/uv"
UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX" uv sync --locked --inexact --python "$CONDA_PREFIX/bin/python"
```

This targets Conda directly and does not create `.venv`. `--inexact` preserves unrelated
existing packages. The image-only analysis does not need optional text dependencies.
PyTorch uses CUDA 11.7 wheels; the compute node needs a compatible driver.

The pinned XGBoost 1.7.6 upstream wheel has a platform-tag inconsistency: `pip/uv check`
may warn and repeated synchronization may reinstall it. Small software tests run in
the existing environment; a warning-free environment check is not claimed.
Pins are in [pyproject.toml](pyproject.toml) and [uv.lock](uv.lock). Each run records
installed versions and source/lock hashes in `provenance.json`.

Run software tests without a formal GRAPE experiment:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=src python -m unittest discover -s tests -v
```

The main reusable entry point is `glaboost.training.train_grape_report()`;
`glaboost.progression.train_progression()` accepts frozen feature arrays.
CLI help: `python -m glaboost train-grape --help`.
The earlier diagnostic-model class and `score-grape` / `evaluate-grape` interfaces
remain for a future compatible fixed detector; `run_grape.sh` uses the current
progression adaptation.

## References

- [Local GlaBoost paper](GlaBoost.pdf) and [professor's study request](Mail%20-%20Han,%20Zeyu%20-%20Outlook.pdf).
- [GRAPE dataset paper](https://www.nature.com/articles/s41597-023-02424-4).
- [scikit-learn cross-fitting / stacking](https://scikit-learn.org/1.3/modules/generated/sklearn.ensemble.StackingClassifier.html).
- [XGBoost 1.7.6 GPU support](https://xgboost.readthedocs.io/en/release_1.7.0/gpu/index.html).
- [Logistic regression class weights and intercept penalty](https://scikit-learn.org/1.3/modules/generated/sklearn.linear_model.LogisticRegression.html).
