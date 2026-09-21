# IEEE DA-SPL Reconstruction

Independent code, without imports from the retired GitHub model or repairs.
The local published IEEE paper is the authority, DOI 10.1109/BIBE66822.2025.00121.

Results, evidence, limitations and subsequent decisions are tracked in the
[experiment log](../../docs/experiment_log.md), including a dated snapshot of
the longer LEM-off run. The log is separate from operational instructions below.

**Read [the method contract](../../docs/paper_method_contract.md).** It maps the
paper equations to code and identifies every necessary interpretation. The
current scope is the image-input DAM + PLN + LEM core, not a claimed recovery
of the underspecified image/corpus/factor implementation or original folds.

## Run

```bash
# Synthetic tests only, no real-image model inference or optimizer steps
bash experiments/paper_method/test.sh

# Data/import/cache verification; no model construction
bash experiments/paper_method/run.sh --check

# On an allocated GPU, after reviewing the reconstruction choices
bash experiments/paper_method/run.sh --accept-reconstruction
```

Default: seed 123, batch 16, five epochs, AMP, fresh initialization using the
cached ImageNet ConViT backbone. The backbone stays frozen in eval mode, while
the new projection and all decoder/LEM parameters train. No automatic download.

Output: `artifacts/paper_method/image_core_seed123/`. Use a new `--run-dir` for
another experiment; nothing is overwritten. This is a debugging run on the
retained 444/111/100 split, not the paper's 200-epoch/ten-fold experiment.
Use `--train-only` to omit TEST generation. For CPU training, explicitly pass
`--device cpu --no-amp`; full-size CPU training is not part of the test suite.

Evaluate an existing checkpoint from this reconstruction only:

```bash
bash experiments/paper_method/run.sh \
  --evaluate-run artifacts/paper_method/image_core_seed123 \
  --accept-reconstruction
```

Evaluation verifies the data, vocabulary, configuration and archived core
sources. It writes a new sibling directory with `_evaluation` appended. Old
GitHub/repair checkpoints cannot be reused as this model's weights.

## Outputs

- `best.pt`, `last.pt`: whole-model states, format/version, vocabularies,
  selected epoch, configuration and data fingerprints. Last also stores optimizer
  and scaler state; resuming interrupted training is not yet implemented.
- `history.json`: separate primary CE, secondary CE, label BCE and EOS CE for
  both report heads, in addition to the total training/validation objective.
- `predictions.json`: exact generated token IDs, text, log probability and EOS flag.
- `report_checks.json`: repeated/missing/incomplete/conflicting fields, identical
  report groups, and exact per-field reference agreement. No text rewriting.
- `metrics.json`: BLEU, ROUGE-L, CIDEr in raw and x100 scales, plus structural counts.
- `config.json`, `data_fingerprints.json`, `manifest.json`, `source.tar.gz`,
  `status.json`: provenance and explicit completion/failure state.

Body CE and total epoch losses are token-weighted batch means; EOS CE is a
per-report average at the real terminal target. Loss scales differ from the
retired models and should not be compared as if they were the same objective.

No EOS upweighting, lagged branch fusion, n-gram blocking, forced END, minimum
length, confidence substitution or output editing is carried over. Unit tests
establish implementation invariants, not correct medical reports. Assess the
actual user-run outputs for repetition, truncation, completeness and content
agreement before deciding whether the previous failure persists.

## Combined Pipeline Audit

After the completed LEM-off greedy/beam comparison, run ONE command:

```bash
bash experiments/paper_method/audit_pipeline.sh
```

This reuses `lem_ablation_seed123/lem_off/last.pt` and the completed
`lem_ablation_seed123_lem_off_decoding_comparison`. It writes a new sibling
`lem_ablation_seed123_lem_off_pipeline_audit/`; existing outputs are never
overwritten. Use `--run`, `--comparison`, and a new `--output` to specify sources.
No training, weight updates, core-code changes, downloads or TEST inference.

The combined checks cover:

- Raw parquet row identity, RGB/bilinear-resized pixels versus actual HDF5
  arrays, all 14 source fields, caption/tag encoding, unknown words and actual
  TRAIN/VAL pixel duplicates. Source consistency is not clinical label validity.
- Initial/final module parameter differences and existing learning curves.
  A delta includes weight decay, not necessarily useful task gradients.
- TRAIN-only majority baselines, per-class recall, and fixed cosine 5-NN using
  frozen-backbone versus projected CLS/mean-patch features. VAL never supplies
  neighbor labels. TRAIN excludes its own neighbor but is not an independent
  evaluation of the trained projection. No probe classifier is trained.
- All TRAIN/VAL reference-token scores for primary, secondary and mixture:
  field values versus template words, punctuation and EOS. Field-first-token
  and teacher-forced value-token agreement are distinct from free generation.
- Identical reference prefixes with own-image versus mean TRAIN visual memory.
  This tests image dependence; mean memory can be out of distribution and is
  not a fitted text-only baseline. Both branches and all 14 fields are scored.
- VAL singleton versus saved-order/shuffled batches using identical cached
  features, plus zero-head/context statistics from unmodified DAM calls.
- Free greedy TRAIN reports and free primary-only/secondary-only VAL reports.
  Compare with existing mixture-greedy/beam-5 VAL reports after exact score
  replay. Single-head routes retain the original same-step recurrent states;
  no reference prefixes, field rewriting or forced termination enter generation.

Images are encoded once (444 TRAIN + 111 VAL), and cached features serve all
probes. There are 444 new TRAIN and 222 new VAL greedy generations, rather than
another training sweep or repeated beam-5 runs. Status and progress identify the
active stage. `static_audit.json`, `feature_knn.json`, `summary.json`, per-sample
teacher scores and exact generated tokens preserve the evidence separately.

`--check` performs only raw-data/checkpoint/encoding/parameter checks without
constructing a model, running inference or creating an output directory. This
audit narrows several hypotheses together; it does not promise a unique cause
or change the paper-method defaults. Leave follow-up training decisions until
the combined results are inspected.

## Longer LEM-Off Training

After the five-epoch ablation and pipeline audit, test training duration without
changing the architecture, frozen-backbone policy, optimizer, loss or data:

```bash
bash experiments/paper_method/train_lem_off.sh --accept-reconstruction
```

Default: **30 total epochs**, only LEM off, seed 123, same 444 TRAIN / 111 VAL
samples, new output `artifacts/paper_method/lem_off_30epochs_seed123/`.
No TEST inference. Existing runs and core sources remain untouched.

The old ablation checkpoint lacks Adam/scaler state. Therefore this run starts
from its archived **initial.pt**, with fresh Adam/scaler, not from the fifth-epoch
weights. Initialization, vocabularies, data, archived training code and the first
five training orders are verified. Only `epochs` changes in the saved config.
Each epoch uses the original seeded order/RNG-reset protocol. At epoch five,
`reference_epoch_comparison.json` records whether model tensors exactly reproduce
the source final state; cross-device bitwise equality is not assumed.

- Every epoch: original training/VAL losses and full VAL **beam=5 plus greedy**,
  all 14 field matches, per-class recalls (including rare risk classes), risk
  macro recall, report diversity, largest identical group and structural failures.
- Every epoch: VAL teacher-forced field/token scores for both heads and their
  mixture, with value/template/punctuation/EOS losses separated. These scores
  must not be presented as free-generation accuracy.
- Epochs 5, 10, 15, 20, 25, 30: full TRAIN greedy reports and teacher-forced scores
  to distinguish insufficient TRAIN fitting from a widening TRAIN/VAL gap.
  A nondefault final epoch and the source reference epoch are always included.
- Each monitored image is encoded once for its free-generation/teacher-forced
  comparisons. Monitoring uses eval FP32, and the next epoch resets RNG state.

`history.json` keeps the complete curves; `epoch_XXX/val/` and scheduled
`epoch_XXX/train/` contain exact prediction tokens, per-sample teacher scores and
structural checks. `majority_baselines.json` uses TRAIN counts only.

`best.pt` retains the original lowest-VAL-report-CE selection. An additional
`best_fields.pt` selects highest **VAL greedy 12-field agreement**, excluding
confidence and additional observations; it is an exploratory content-oriented
selection, not a changed training objective. Ties keep the earliest epoch.
`selection.json` records both choices; `last.pt` saves final model, Adam and AMP
scaler state. `reference_epoch.pt` preserves the five-epoch model separately.
No scheduler, early stopping, sample weighting or output rewriting is introduced.
The legacy evaluation/diagnostic entrypoints do not accept this new checkpoint
format; this runner writes its own per-epoch evaluations. Resume is not exposed
by this entrypoint; an existing output directory is never overwritten.

Use `--check` for read-only preflight with no model construction or training.
Optional `--source-run`, `--output` and `--epochs` specify another completed
paired LEM-off source, a NEW output directory and a larger total epoch count.
GPU execution is required when the source used AMP; it is not silently disabled.

## Risk Conditioning Audit

This diagnostic reads the existing best checkpoint; it never trains or runs
TEST inference. Core model/evaluation source files and saved results are unchanged.
Run on your GPU node:

```bash
bash experiments/paper_method/diagnose_risk.sh \
  --run artifacts/paper_method/image_core_seed123
```

Add `--check` to verify the checkpoint, data and VAL pairings without constructing
a model. The default new output directory is
`artifacts/paper_method/image_core_seed123_risk_diagnostic/`. For another audit,
provide a new `--output`; existing directories are never overwritten.

For each VAL image, select a seeded, opposite-reference-group VAL donor. Groups
are healthy/very healthy versus moderate/high risk. Donors may be reused; their
IDs and reuse counts are recorded. Compare:

- Own image + own reference report prefix.
- Donor image + the identical reference prefix (image intervention).
- Own image + donor reference prefix (whole-text intervention).
- Own image + own freely generated report prefix.
- Donor image + that identical generated prefix.

Prefixes stop immediately after the first `glaucoma risk assessment :` marker,
before the risk value and subsequent confidence field. Each condition starts
from a fresh recurrent state and uses one image, eval mode and FP32, matching
normal generation. No reference label or confidence is provided to the model.
Free generation uses the existing unmodified beam search. Missing generated
markers are recorded and excluded from paired contrasts, never filled from gold
text. Malformed risk values, including internal repetition, are counted invalid.

`cases.json` records exact prefixes, image/donor IDs, generated text, visual-memory
differences and primary/secondary/mixture scores. `summary.json` reports free
generation risk matches/confusion and paired probability/choice changes. The
first risk words are distinct: healthy, very, moderate, high. Their probabilities
are taken from the full vocabulary, with their total mass and unrestricted top
five words also saved. Four-label rankings are diagnostic restrictions, NOT
free-generation accuracy or calibrated clinical confidence. Whole-risk candidate
scores also include the period; these are summed log probabilities without length
normalization, so use first-word results alongside them.

Larger effects from swapping text than swapping images would support strong text
conditioning for these tested prefixes, not prove the encoder ignores every
image. Reference-versus-generated-prefix degradation can expose error propagation.
Cross-group image/text combinations can be out of distribution, and full-prefix
swaps change both wording and length. This is checkpoint-selected VAL debugging,
not an independent clinical validation or proof of a unique failure cause.

## Trace Risk Reversals

After the risk-conditioning audit, trace only the VAL cases whose local first
word AND complete risk phrase favor `very healthy`, but whose final generated
risk is `high risk`. This selects 12 cases in the initial seed-123 audit without
using their reference labels as a selection criterion.

```bash
bash experiments/paper_method/trace_risk_beam.sh \
  --audit artifacts/paper_method/image_core_seed123_risk_diagnostic
```

Add `--check` for checkpoint/data/selection checks without model construction.
Default output: `artifacts/paper_method/image_core_seed123_risk_beam_trace/`.
Use a new `--output` to repeat the audit. No training, TEST inference, model
changes, confidence replacement or forced EOS is performed.

An opaque-state observer calls the ORIGINAL `beam_search` and returns unmodified
logits. It records actual expansions, per-parent top-k admission, global ranking,
EOS probabilities and the first elimination of the exact healthy-phrase path.
Reconstructed rankings must match the observed calls, and generated tokens and
scores must reproduce the preceding audit. Otherwise tracing stops with an error.

Separately, a COUNTERFACTUAL continuation fixes the selected generated prefix
and `very healthy .`, then lets the same beam search choose everything afterward.
The forced prefix retains its true model probability. This uses the original
TOTAL token budget; no extra decode steps, forced confidence or EOS are granted.
Primary/secondary/mixture token scores are replayed for both the original path
and the explored healthy continuation, including the real ending probability.

`cases.json` contains actual-search beam snapshots, the counterfactual report,
token-level scores and a cumulative-score comparison. `summary.json` lists the
first pruning event, first later score reversal on the explored continuation,
and whether a completed valid healthy-risk path scores above the original report.
Positions are predicted-token positions excluding BOS. Completed path scores are
carried forward when comparing unequal lengths, as in the production search.

Original-search pruning and counterfactual score reversals are distinct evidence:
the explored continuation may never have survived the original beam. A better
completed valid healthy-risk path witnesses a missed candidate, but does not
prove its other fields are correct. A worse explored path does not prove all
healthy continuations are worse: the continuation search still has finite width.

## Ending Learning Audit

After the beam trace, inspect why punctuation/EOS scores are weak:

```bash
bash experiments/paper_method/diagnose_endings.sh \
  --run artifacts/paper_method/image_core_seed123
```

The existing `_risk_beam_trace` sibling directory supplies the focused VAL cases.
Use `--trace` to specify a different completed trace from the same checkpoint.
Add `--check` to inspect actual target endings and validate provenance without
building a model. Output is a NEW
`artifacts/paper_method/image_core_seed123_ending_diagnostic/` directory; use
`--output` to repeat without overwriting. No training or TEST inference occurs.

The audit scores the 444 TRAIN and 111 VAL reference reports using teacher forcing,
separately at the risk-ending period, confidence number, terminal period and EOS.
It saves target probabilities/NLL, unrestricted top-five competitors and colon
probabilities for the primary, secondary and mixture distributions. Confidence
numbers are text targets, never loss weights or image inputs.

Three reference conditions reuse the SAME frozen-encoder features: single-image,
saved-order batches of the configured size, and seeded-shuffle batches. This
tests the batch-averaged DAM term without also changing image encoding, dropout,
precision, weights or text. Batches are probes, NOT historical training batches.
Active sample IDs at each scored position are saved because finished reports
are excluded from the DAM average. All probes use eval mode and FP32; they do
not reproduce historical AMP training losses.

For the selected VAL failures, a further paired comparison changes only the
report body before the risk field. Both bodies use the exact same already-traced
healthy-risk/confidence/period/EOS suffix, even if the reference used a different
risk label or confidence number. The generated-body scores must replay the saved
trace. This controls the suffix when testing whether earlier generated text
affects termination. It is a teacher-forced counterfactual, not a new prediction
or an accuracy evaluation; no new beam search is performed.

`data_audit.json` reports exact target suffixes, risk/confidence counts and EOS
frequency in the CE targets. `train_cases.json`, `val_cases.json` and `summary.json`
provide scores and paired changes by reference risk and confidence. Interpret
poor TRAIN-reference scores as evidence of incomplete fitting; batch changes as
evidence of composition dependence; and controlled-body changes as sensitivity
to the preceding text. These checks can narrow hypotheses but do not prove a
unique training cause or justify changing model outputs by hand.

## Ending Loss Conflict Audit

To test whether the weighted LEM signal opposes CE at the ending positions:

```bash
bash experiments/paper_method/diagnose_loss_conflict.sh
```

Uses the existing `image_core_seed123` checkpoint by default (`--run` overrides).
Output is a new sibling directory ending in `_loss_conflict`; use `--output`
for another run. `--check` validates data and checkpoint without building a model.
No training, parameter updates, free generation, VAL or TEST inference occurs.

The probe uses all TRAIN references in saved-order batches with the original
batch size and loss weights. Encoder/decoder run in eval FP32 without gradients.
Their detached logits become independent gradient probes, passed through the
unchanged probability mixture, expected word embeddings and LEM. LEM uses native
LSTM operations because cuDNN eval-mode LSTM does not support backward. All model
parameters remain frozen. Saved-order batches are NOT historical training replay.

`token_audit.json` separates punctuation frequency from EOS frequency. `cases.json`
and `summary.json` record weighted CE/LEM gradient norms, their cosine, and the
target-minus-colon gradient at terminal period/EOS for both heads. Positive margin
gradient means gradient descent in logit space would favor the colon over the
correct target. `lem_opposes_ce_margin` counts opposing LEM signals;
`total_opposes_ce_margin` counts cases where the total reverses the CE direction.
`batches.json` preserves batch membership and loss reductions for interpretation.

This measures local pressure at report logits, not parameter-space gradients,
actual Adam steps or the historical cause of a learned error. Positive evidence
motivates a controlled training ablation; negative evidence does not exclude
shared-parameter interference. Neither token frequency nor scalar loss magnitude
alone proves which loss dominates learning.

## Paired LEM Training Ablation

Run both fresh-start arms sequentially on one GPU:

```bash
bash experiments/paper_method/ablate_lem.sh --accept-reconstruction
```

Default: seed 123, five epochs EACH, the existing configuration and dataset.
`--check` validates data/configuration/cache without building a model or training.
`--epochs N` changes both arms together; `--config` chooses another positive-LEM
baseline configuration. Output defaults to
`artifacts/paper_method/lem_ablation_seed123/`; use a NEW `--output` to repeat.
Existing runs/checkpoints/data are never overwritten. Only the user launches
real training; the tests use synthetic models and mocked experiment workflows.

- `lem_on`: original CE1 + 0.5 CE2 + 5 LEM (or the supplied baseline weights).
- `lem_off`: same architecture and computation, but the LEM coefficient is zero.
  There is no supervised LEM gradient; ordinary optimizer weight decay may still
  affect its unused parameters. CE weights and probability fusion do not change.

One new initialization is saved as `initial.pt` and loaded into each arm, with a
tensor-content SHA-256 check. Adam/scaler state starts fresh for each arm. Shared
epoch permutations are written to `training_order.json`; delivered training IDs
are checked against them and saved per epoch. Both arms reset RNGs to the same
epoch seed before training. These controls define a new matched experiment,
not an exact replay or continuation of `image_core_seed123`.

Every epoch evaluates VAL losses in FP32, generates all VAL reports with the
unchanged beam search, and scores reference ending positions for both branches
and their mixture. `epoch_NNN/` contains predictions, ending cases and summaries.
These are distinct: teacher-forced ending scores are NOT free-generation or
clinical accuracy. Risk agreement uses the strict parser, counting malformed
values invalid. Repetition counts and structural field matches remain visible.
No TEST images are used for training, generation, checkpoint selection or audits.

Each arm has `history.json`, `best.pt` and `last.pt`. The common best-checkpoint
criterion is VAL CE1 + lambda CE2, excluding LEM; total loss is not comparable
between arms. `comparison.json` compares matching epochs, with the final matched
epoch designated the primary comparison, not separately selected best epochs.
Initial/last/best weights are saved, not one large checkpoint per epoch. There is
no resume interface. Checkpoints use a separate ablation format and are not
silently accepted by the original run/diagnostic commands. The parent archive
captures the code for BOTH arms. The ordinary paper runner still rejects a zero
LEM weight; disabling it is confined to this explicit ablation.

## Field Image Sensitivity

After the paired ablation, inspect the final `lem_off` checkpoint without training:

```bash
bash experiments/paper_method/diagnose_fields.sh
```

Default source: `artifacts/paper_method/lem_ablation_seed123/lem_off/last.pt`.
Use `--run` for another completed off arm. It must match the parent configuration,
data fingerprints, checkpoint format and archived core code. The final epoch's
saved VAL predictions are verified against the tokenized references and replayed
against the checkpoint's log probabilities; no new beam search is needed.
`--check` performs metadata/data/prefix checks only, not model construction or
score replay. Output is a NEW sibling of the paired run named
`lem_ablation_seed123_lem_off_field_diagnostic/`; `--output` must not overwrite or
nest within an existing run. No model weights or data are changed.

The probe covers optic disc size, cup-to-disc ratio and rim color. Candidate
values are the TRAIN single-token values only. Longer descriptions remain in the
dataset and are explicitly listed as uncovered candidates, not normalized or
relabelled. First-token probabilities keep full-vocabulary normalization, with
unrestricted top-five alternatives and candidate mass. Separate value-plus-period
scores distinguish a local value preference from punctuation/continuation effects.

For every VAL image/field, compare own image versus a different-value VAL donor
image under the IDENTICAL reference prefix. Repeat under the saved generated
prefix. Prefixes stop at the first field colon, before the target value or any
later findings, risk or confidence. Missing generated markers remain missing.
Same-image reference/generated-prefix contrasts expose text sensitivity; the
first field's prefixes are normally identical and provide a useful control.

Donors preferentially have the SAME exact reference risk label but a different
target-field value, probing detail within a risk category. When unavailable,
cross-risk donors are used and reported separately. Pairings are seeded, try to
balance reuse, and record IDs, labels and image-memory differences. Labels guide
pair selection only and are never supplied after the prefix boundary. TRAIN is
read only for text candidate counts; image inference uses VAL alone, never TEST.

`cases.json` saves each head's candidate probabilities, continuations, prefixes,
donors and original generated values. `summary.json` reports actual saved field
agreement separately from restricted diagnostic rankings, and paired effects by
donor stratum. `prediction_replay.json` verifies the saved path scores. A ranking
change is not a newly generated report, and a local/generation disagreement does
not by itself prove a beam-search bug. Image swaps alter multiple visual traits;
weak sensitivity cannot uniquely distinguish frozen visual features, decoder
learning and text conditioning. This is targeted checkpoint debugging, not a
clinical accuracy study or proof of one root cause.

## Greedy Versus Beam Five

When local field scores disagree with the final report, compare the complete
generation workflow with only the search width changed:

```bash
bash experiments/paper_method/compare_decoding.sh
```

Uses the final `lem_off/last.pt` from the paired seed-123 run by default. `--run`
selects another completed off arm with baseline beam width five. The baseline
is its saved final-epoch VAL predictions; every saved path's probability must
replay against the unchanged checkpoint. Greedy uses the existing `beam_search`
with `width=1`, beginning at BOS and receiving only image features and its own
previous tokens. Mixture weights, EOS behavior, the 161-token default cap and
FP32 evaluation stay unchanged. No reference prefix, field candidate restriction,
random sampling, repetition constraint or output repair is introduced. The
standard paper runner and its beam width remain untouched.

`--check` performs metadata/data/baseline checks without model construction or
generation. Default output is a NEW sibling directory named
`lem_ablation_seed123_lem_off_decoding_comparison/`; use `--output` to repeat.
No training, parameter updates or TEST inference occurs.

`summary.json` compares risk agreement, all 14 field matches, full text matches,
distinct reports, the largest identical group, missing/repeated fields and
termination. `paired_cases.json` records which fields greedy fixes or worsens
for each sample; original and greedy texts are saved separately. A higher unique
report count alone is not success. This ablation can reveal decoding-dependent
errors but cannot repair missing visual detail or prove greedy is generally
better. Different early greedy choices also change later prefixes, so this
compares full generation policies, not one isolated field intervention.
