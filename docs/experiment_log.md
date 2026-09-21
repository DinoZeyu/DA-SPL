# Experiment Results and Decisions

Snapshot: **2026-09-20 22:44 America/Chicago (2026-09-21 03:44 UTC)**.
This is a manually maintained evidence log, not a live dashboard. Artifact JSON
files remain the source of truth. Results below are separated into completed
experiments and an explicitly incomplete training snapshot.

## Scope and Metric Definitions

- Active project: independently reconstructed ConViT + DAM + PLN + LEM
  **image-input core**, subject to the [method contract](paper_method_contract.md).
  This is not the full multimodal implementation or the published ten-fold result.
- Dataset: retained `data/processed/glaucoma_rerun_v1/`, with 444 TRAIN, 111 VAL
  and 100 TEST images. Reports, vocabularies and splits were reconstructed, not
  recovered from the authors' original preprocessing.
- Legacy GitHub/repair implementations were retired. Their artifacts remain
  historical records; this log starts with the independent reconstruction.
- **14-field agreement:** exact reference-value matches divided by `14 * N`.
  Includes confidence and additional observations; not full-report accuracy.
- **12-field content agreement:** the same calculation excluding confidence
  and additional observations. Introduced for the longer run's extra checkpoint
  selection; it does not change training targets or loss weights.
- **Risk agreement:** exact agreement with the reference's four risk categories.
  **Risk macro recall:** unweighted mean of the four per-class recalls when all
  four reference categories are present, as in the current TRAIN/VAL splits.
- **Full-report match:** exact text equality, not expert judgment of equivalence.
  **Unique reports / largest group:** diversity and concentration of exact texts.
  Repetition alone is not wrong when references legitimately agree.
- **Teacher forcing:** the model receives earlier reference tokens. Token/field
  scores in this condition are not autonomous report-generation accuracy.
- None of these metrics establishes clinical correctness. Source annotations
  have not been independently adjudicated. VAL has been inspected repeatedly;
  TEST also has historical exposure and is not a fresh independent holdout.
- Confidence values such as `0.9` and `0.95` are target text, not image inputs,
  loss weights, or calibrated medical confidence estimates.

## Completed Experiments

### E01: Initial Independent Reconstruction

Source: [initial run](../artifacts/paper_method/image_core_seed123/status.json),
[structural checks](../artifacts/paper_method/image_core_seed123/report_checks.json),
[text metrics](../artifacts/paper_method/image_core_seed123/metrics.json).
Five epochs, selected epoch 5, original positive LEM weight, beam=5; **TEST=100**.

| Metric | Result |
|---|---:|
| Risk-field agreement | 52/100 |
| 14-field agreement | 990/1400 = 70.71% |
| Unique reports / largest identical group | 16 / 35 |
| Unfinished reports | 0 |
| All fields exactly once and complete | 97/100 |
| Reports with repeated confidence field | 3 |

Raw BLEU-4 was 0.7841 and ROUGE-L 0.8786, despite the weak risk result. This is a
concrete warning that high template-overlap scores do not establish useful
clinical content. The reconstruction removed the earlier universal truncation
pattern, but did not solve content errors or repeated reports.

### E02: Risk Conditioning and Beam Trace

Sources: [risk audit](../artifacts/paper_method/image_core_seed123_risk_diagnostic/summary.json)
and [beam trace](../artifacts/paper_method/image_core_seed123_risk_beam_trace/summary.json).
Same initial reconstruction checkpoint, **VAL=111**; no training.

- Holding the reference prefix fixed and swapping to an opposite-risk image
  changed the local mixture risk-first-word choice in 89/111 cases.
- Holding the image fixed and replacing the reference prefix changed that
  choice in 6/111 cases. These whole-image/whole-prefix interventions can be
  out of distribution; they are not isolated clinical-variable interventions.
- Traced 12 cases locally favoring `very healthy` but finally generating
  `high risk`. All 12 explored completed healthy alternatives had lower or
  equal whole-sequence scores; later punctuation/EOS often reversed the lead.

Meaning: images influence the decoder. A locally favored risk word need not
win full-sequence decoding. These results do not demonstrate a broken beam
implementation, nor prove that every alternative healthy report scores worse.

### E03: Ending Learning and Local Loss Conflict

Sources: [ending audit](../artifacts/paper_method/image_core_seed123_ending_diagnostic/summary.json)
and [gradient audit](../artifacts/paper_method/image_core_seed123_loss_conflict/summary.json).

- On 164 TRAIN `very healthy` references, EOS was the top choice for the
  primary head in 0/164, secondary in 164/164, and mixture in 1/164.
- At the primary EOS logit, the weighted LEM contribution opposed the CE
  target-versus-colon direction in 127/164 cases, but the combined gradient
  reversed that direction in only 2/164. Secondary reversals were 39/164.
- EOS was included in supervision; padding was excluded. The diagnostic did
  not find a missing-EOS-target bug.

Meaning: an ending-learning defect and some CE/LEM conflict were observed.
The gradient probe is local to logits, not a replay of Adam updates or a proof
of the entire historical cause. Decision: test LEM with matched training arms.

### E04: Matched LEM-On/Off Training

Source: [paired comparison](../artifacts/paper_method/lem_ablation_seed123/comparison.json).
Fresh identical initial tensors, identical epoch orders and five epochs per
arm. Only the LEM loss coefficient changes from 5 to 0. **VAL=111, beam=5**.

| Final epoch-5 metric | LEM on | LEM off |
|---|---:|---:|
| Risk agreement | 62/111 (55.86%) | 86/111 (77.48%) |
| Healthy-group references generated as high risk | 40/42 | 16/42 |
| Reference-prefix mixture EOS top-1 | 70/111 | 111/111 |
| 14-field agreement | 70.40% | 69.05% |
| Unique reports / largest group | 13 / 38 | 7 / 59 |
| Exact full-report matches | 6/111 | 6/111 |
| Unfinished reports | 0 | 0 |

Meaning: switching LEM off improved endings and risk agreement in this matched
short run, but worsened diversity and overall field agreement. LEM is therefore
not the whole explanation for repeated reports. This is evidence about this
reconstruction/seed/duration, not a refutation of the published method.

Decision: keep LEM off as the controlled diagnostic branch while investigating
content learning. The earlier E01 run used a different training-order protocol
and reported TEST results; do not compare it as a paired LEM-on control.

### E05: Field/Image Sensitivity

Source: [field audit](../artifacts/paper_method/lem_ablation_seed123_lem_off_field_diagnostic/summary.json).
Final epoch-5 LEM-off checkpoint, **111 VAL images / 333 field cases**.

| Field | Local choice with reference prefix | Local choice with beam-generated prefix | Saved beam report match |
|---|---:|---:|---:|
| Optic disc size | 81/111 | 81/111 | 72/111 |
| Cup-to-disc ratio | 73/111 | 73/111 | 66/111 |
| Rim color | 91/111 | 76/111 | 76/111 |

Candidates came from single-token TRAIN field values; local rankings are not
free-generation metrics. For these fields the reported first-word choices also
matched unrestricted vocabulary top-1 choices. Saved beam scores replayed exactly.

Meaning: both whole-sequence selection and earlier generated text can reduce
field agreement. This motivated a decoding comparison without retraining.

### E06: Greedy Versus Beam=5

Source: [decoding comparison](../artifacts/paper_method/lem_ablation_seed123_lem_off_decoding_comparison/summary.json).
Same epoch-5 LEM-off weights, **VAL=111**, FP32 and token budget 161.

| Metric | Beam=5 | Greedy |
|---|---:|---:|
| 14-field agreement | 1073/1554 (69.05%) | 1107/1554 (71.24%) |
| Risk agreement | 86/111 | 86/111 |
| Unique reports / largest group | 7 / 59 | 13 / 45 |
| Exact full-report matches | 6/111 | 6/111 |
| Unfinished / missing-field / repeated-field reports | 0 / 0 / 0 | 0 / 0 / 0 |

Greedy changed 61 reports, fixed 54 field values and regressed 20, a net gain of
34. Every sample's risk value stayed unchanged. The main field gains were optic
disc size (+9), cup-to-disc ratio (+7), and ISNT rule (+19).

Meaning: decoding contributes, but is insufficient to solve collapse. More
distinct reports are not automatically better. Keep both decoding policies as
separate measurements; do not silently redefine the paper's beam=5 setting.

### E07: Combined Pipeline Audit

Sources: [summary](../artifacts/paper_method/lem_ablation_seed123_lem_off_pipeline_audit/summary.json),
[data/parameter checks](../artifacts/paper_method/lem_ablation_seed123_lem_off_pipeline_audit/static_audit.json),
[feature probes](../artifacts/paper_method/lem_ablation_seed123_lem_off_pipeline_audit/feature_knn.json).
Same final epoch-5 LEM-off checkpoint; **TRAIN=444 and VAL=111**.

Checkpoint SHA256:
`ed517be3ec821dd3d7b2027cb17c44f40f856a8e81526deaeca1e603f2cbb752`.
Both saved beam and greedy scores replayed exactly on all 111 VAL images.

#### Data and Parameters

All 555 resized RGB images matched their raw source rows pixel-for-pixel. All
7770 normalized, vocabulary-encoded fields matched; caption/tag alignment had
no reported issues. There were no exact processed-pixel duplicates in TRAIN/VAL.
VAL contained one unknown target token. This checks source consistency, not
clinical annotation validity or patient-level independence.

Projection, attention and both report branches changed from initialization.
Backbone tensors stayed frozen as configured. Parameter change includes weight
decay and alone does not prove useful task gradients.

#### Fitting and Branches

| Split / decoding | 14-field agreement | Risk agreement | Unique reports | Largest group | Exact report matches |
|---|---:|---:|---:|---:|---:|
| TRAIN, mixture greedy | 73.10% | 350/444 | 19 | 158 | 34/444 |
| VAL, mixture greedy | 71.24% | 86/111 | 13 | 45 | 6/111 |
| VAL, primary-only greedy | 72.65% | 88/111 | 17 | 43 | 6/111 |
| VAL, secondary-only greedy | 70.14% | 85/111 | 12 | 47 | 6/111 |

All these outputs terminated with complete, nonrepeated fields. TRAIN references
contained 248 unique reports, VAL references 79. Thus repeated generation is
already present on training examples, not just unseen examples. With correct
prefixes, TRAIN cup-to-disc ratio still matched only 282/444 (63.51%).

Primary-only decoding helped slightly, but did not fix repetition. It remains
an ablation, not an adopted architecture change.

#### Image Information and Generated-Prefix Degradation

Fixed cosine 5-NN using TRAIN neighbors gave VAL risk agreement of 92/111
(raw CLS), 95/111 (raw mean-patch), and 94/111 (either projected view). These are
82.88%-85.59%, with no probe classifier trained and no VAL neighbor labels used.
This supports available visual label information; it does not certify fine-grained
recognition. Poor pooled-feature kNN results would not prove absence of information.

At the risk-first-word position, replacing each image's memory with mean TRAIN
memory reduced correct choices from 95/111 to 61/111, holding reference text
fixed. This supports image dependence, although mean memory can be out of distribution.

| VAL field | Correct-reference-prefix first-word match | Free greedy field match |
|---|---:|---:|
| ISNT rule | 111/111 | 88/111 |
| Rim color | 91/111 | 75/111 |
| Glaucoma risk | 95/111 | 86/111 |

The first two fields have single-token values in VAL; the four risk labels have
distinct first words. The paired counts support degradation from self-generated
history, not a claim that teacher-forced accuracy equals full-report accuracy.
The evidence supports insufficient content fitting plus error propagation;
it does not establish a unique cause or guarantee that longer training fixes both.

#### Token Loss, Attention and Imbalance

- On VAL reference prefixes, mixture top-1 accuracy was 100% on template words,
  98.86% on punctuation, 100% on EOS, and 80.67% on value tokens.
- Value tokens accounted for 85.45% of summed mixture NLL on VAL (85.33% TRAIN).
  This is a diagnostic NLL decomposition, not the exact two-head training-loss
  decomposition. Current residual error is not predominantly punctuation/EOS.
- No all-heads-zero event occurred in 78,882 TRAIN or 19,660 VAL attention calls.
  This does not establish that attention learned clinically appropriate regions.
- Compared with singleton decoding, saved-order batches changed 1/1554 VAL
  field-first-word choices; shuffled batches changed 2/1554. These tested eval
  batch effects are too small to explain most observed failures.
- At epoch five, free reports used only `high risk` and `very healthy`. VAL
  `healthy` (2 references) and `moderate risk` (8) both had zero recall. Overall
  risk agreement 77.48% concealed **40.63% macro recall** (rounded).
- A TRAIN-majority-per-field baseline scored 58.11% on VAL across all 14 fields.
  Always choosing TRAIN-majority disc size `normal` alone scored 81/111, equal
  to the model's disc-size match count, although their individual choices differ.

Decision after E07: test training duration with the same LEM-off setup before
changing architecture, visual freezing, sampling or loss weights.

## E08: Thirty-Epoch Run, In Progress

Sources: [manifest](../artifacts/paper_method/lem_off_30epochs_seed123/manifest.json),
[live status](../artifacts/paper_method/lem_off_30epochs_seed123/status.json),
[history](../artifacts/paper_method/lem_off_30epochs_seed123/history.json),
[epoch-five replay](../artifacts/paper_method/lem_off_30epochs_seed123/reference_epoch_comparison.json).

At this snapshot, history contained **11 completed epochs**, and status reported
**epoch 12, VAL monitoring**. The observations below are not a completed 30-epoch
result. Do not infer completion from an existing `last.pt`.

The source ablation lacked optimizer state, so this run starts from the SAME
archived initial tensors with fresh Adam/scaler, not epoch-five weights plus
25 fresh-optimizer epochs. The only config change is `epochs: 5 -> 30`.
The first five training orders match, and **epoch-five model tensors exactly
reproduced the original LEM-off final state** in this actual run.

Every epoch measures VAL beam=5 and greedy, field/class agreement, repetition,
structure, and teacher-forced token categories. Full TRAIN greedy/teacher-forced
audits run at epochs 5, 10, 15, 20, 25 and 30. No TEST inference or early stopping.

### Interim VAL Curve

All generation columns below are **mixture greedy, VAL=111**. Report CE is the
batch-evaluated `primary_ce + 0.5 * secondary_ce`, not the mixture token NLL.

| Epoch | Report CE | 14-field agreement | Risk matches | Risk macro recall | Unique reports | Largest group | Unfinished |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.11870 | 0.97% | 0/111 | 0.00% | 87 | 6 | 111 |
| 2 | 0.92343 | 67.05% | 61/111 | 25.00% | 13 | 31 | 0 |
| 3 | 0.56832 | 74.07% | 89/111 | 42.93% | 9 | 51 | 0 |
| 4 | 0.42920 | 76.51% | 92/111 | 45.67% | 11 | 45 | 0 |
| 5 | 0.37822 | 71.24% | 86/111 | 40.63% | 13 | 45 | 0 |
| 6 | 0.32396 | 78.64% | 93/111 | 46.29% | 10 | 32 | 0 |
| 7 | 0.30008 | 74.71% | 83/111 | 42.19% | 22 | 26 | 0 |
| 8 | 0.27710 | 78.70% | 94/111 | 46.70% | 14 | 41 | 0 |
| 9 | 0.25763 | 79.67% | 95/111 | 49.18% | 24 | 23 | 0 |
| 10 | 0.25396 | 79.21% | 97/111 | 55.86% | 14 | 38 | 0 |
| 11 | 0.24667 | 78.12% | 93/111 | 48.36% | 10 | 40 | 0 |

Interim meaning, **limited to completed epochs 1-11**:

- More training is already improving content relative to epoch five. TRAIN
  14-field agreement rose from 73.10% at epoch 5 to 85.10% at epoch 10; TRAIN
  risk agreement rose from 350/444 to 403/444.
- VAL epoch 10 risk agreement reached 97/111 (87.39%); moderate-risk recall was
  3/8 instead of 0/8. `healthy` recall remained 0/2. Healthy-group-to-high-risk
  errors fell from 16/42 at epoch 5 to 2/42 at epoch 10.
- Among these completed epochs, epoch 9 had the highest 12-field content score
  (77.78%). Lowest report CE was at epoch 11. These are different objectives.
- Repetition and content scores still fluctuate. Epoch 9 had 24 unique reports;
  epoch 11 had 10 despite lower CE. Epoch 1's 87 unique reports were all
  unfinished, illustrating why diversity alone cannot select a good model.
- These observations strengthen the insufficient-training explanation for part
  of the five-epoch failure, but do not establish that 30 epochs will solve
  repeated reports, rare classes, or generated-prefix error propagation.

### Selection and Final Review

`best.pt` retains lowest VAL report CE. `best_fields.pt` separately selects the
highest VAL-greedy 12-field content agreement; ties keep the earliest epoch.
Neither selection uses TEST. `reference_epoch.pt` preserves epoch 5; `last.pt`
stores model, optimizer and scaler. See the [run instructions](../experiments/paper_method/README.md#longer-lem-off-training).

After `status.json` reports `complete`, record epoch 30 and both selected epochs,
not just the single most favorable metric. Compare fixed epochs 5/10/20/30 for
TRAIN/VAL content agreement, per-class risk recalls, healthy-to-high errors,
structural failures, exact-report matches and repetition against the references.
Keep teacher-forced scores separate from free generation. Inspect whether the
TRAIN/VAL gap widens or both remain poorly fitted before choosing another change.
No new architecture/loss/data experiment is authorized by this log alone.

## Update Protocol

1. Add a dated entry with hypothesis, source checkpoint, split, sample count,
   changed variables, and links to manifest/status/summary or history files.
2. Label the entry `complete`, `in progress`, or `failed`. Interim entries must
   state exactly which completed epochs their numbers cover.
3. Keep observations, interpretation, limitations and the resulting decision
   distinct. Do not overwrite an earlier conclusion without noting new evidence.
4. Report regressions and minority-class behavior alongside improvements. Do
   not replace reference agreement with the phrase "clinical accuracy".
5. Append the final E08 review when available; preserve this interim snapshot
   so the reason for the next decision remains traceable. Do not edit artifacts.
