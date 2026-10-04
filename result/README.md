# GRAPE fixed-detector analysis reports

These are cohort-level reports for Dr. Zhang's longitudinal A/B study. The diagnostic
detector is fixed independently of GRAPE; progression mappings are trained and
patient-cross-validated within GRAPE.

The completed analysis uses two short report-directory names:

- [hf_training](hf_training/report.html): source diagnosis training and test results.
- [grape_validation](grape_validation/report.html): longitudinal A/B comparisons on GRAPE.

The GRAPE overview opens with the professor's cohort-level template filled from the
prespecified primary analysis, followed by all configuration comparisons and detailed reports.

The saved provenance retains the original run identifiers and execution paths;
the corresponding model artifacts keep those identifiers on scratch.

Run `bash run_grape.sh` inside the already allocated GPU node to train the source
diagnosis models and then evaluate their fixed outputs on GRAPE. Source training uses
the retained HF training split; retained test rows supply separate diagnosis
metrics that are never used to select a model. Existing outputs are never overwritten.

The source release has 589 training and 100 test records before exclusions. Identical
decoded images within a split keep their first occurrence; train/test matches retain
the training image and exclude matching test rows. Conflicting labels or any exact
HF/GRAPE image overlap stop the run. Source reports show actual retained counts and
exclusions, rather than treating all 100 released test rows as independent observations.

The workflow is ready to run. [INDEX.md](INDEX.md) and the root README's automatic
results block show which analyses have completed.
The primary model is `paper100_depth6` (100 trees, depth 6); the other predeclared
tree/depth configurations are supplementary. These are source-trained reconstructions,
not the authors' model weights or a claim to their reported accuracy.

| File or directory | Purpose |
|---|---|
| `<run>_source/` | Source-training configuration, held-out diagnosis metrics and overlap/provenance summary |
| `<run>/report.html` | Standalone overview: all configurations, embedded detailed reports, figures, metric tables and audit records |
| `<run>/report.md` | Markdown overview with links to companion files |
| `<run>/<model-name>/report.html` | Standalone cohort report with embedded figures |
| Configuration CSV/JSON files | Metrics, scores, OOF predictions, cohort, exclusions, provenance and folds |
| Configuration `figures/` | Three-endpoint paired comparison and descriptive score histories |
| Batch plan/status/audit files | Exact inputs, intended configurations, completion and comparability |
| Batch `code/` | Source and environment snapshot for reproduction |

Completed batches appear in [INDEX.md](INDEX.md). The root README shows the predeclared
primary analysis from the latest complete batch, not the best observed configuration.
Failed batches preserve their status and completed children. Synthetic tests stay in
temporary directories and are not research results.

Reports remain in home. Source model parameters/features are saved at
`artifacts/<run>_source/` and GRAPE scores/progression mappings at `artifacts/<run>/`;
the `artifacts` link points into `/scratch/users/zeyuhan/DA-SPL/`.

The source split lacks patient IDs, and exact-image duplicate checks cannot establish
patient independence. The fixed detector is trained outside GRAPE, but the progression
mappings are trained and patient-cross-validated on GRAPE; this does not establish
external validation of the complete progression pipeline. Old GRAPE-trained progression
runs are retired and must not be presented as evidence for the new workflow.

To reuse the saved diagnostic models without retraining, run
`bash run_grape.sh --plan result/hf_training/external_models.json --run-name grape_recheck`.
