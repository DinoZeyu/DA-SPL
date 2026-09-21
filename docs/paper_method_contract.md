# DA-SPL Reconstruction Contract

Authority: the local published IEEE PDF, DOI 10.1109/BIBE66822.2025.00121,
printed pages 698-705. The arXiv version and retired GitHub code are not the
implementation specification. This document distinguishes stated methods from
choices required to make an executable model.

## Scope and Claims

This stage reconstructs the **image-input core**: ConViT, DAM, PLN and LEM.
It is not an exact recovery of the authors' program, historical folds, Optuna
settings, or best three-modality result. The corpus/factor fusion in Table VIII
is underspecified and is NOT fabricated here. The professor's later project is
separate and unimplemented.

The existing prepared dataset is retained as a fixed input contract so a new
model can be investigated without also changing reports/splits. Its 444/111/100
TRAIN/VAL/TEST split and deterministic JSON verbalization are reconstructed,
not recovered from the paper. Raw data and all old results remain untouched.
Five epochs are a debugging run, not the paper's 200-epoch/ten-fold protocol.
Repeated inspection of this test set precludes a new independent validation claim.

## Module Mapping

| Paper | Independent implementation | Interpretation or unresolved detail |
|---|---|---|
| III-A, IV-B: ImageNet ConViT, 512 features | `model.VisualEncoder`: cached ConViT-base, classifier removed, trainable 768-to-512 projection | Variant, token retention and freezing are not fully specified. Retain CLS plus spatial tokens; freeze backbone in eval mode for this first run; train projection. |
| Eq. 1-2: scaled dot-product multi-head attention | `model.DualAttention`: hidden-state query, visual token keys/values, head concatenation/projection | Query wiring and exact tensor layout are implementation choices. Eight DAM heads follow Table I; ConViT keeps its pretrained head count. |
| Eq. 3-4: normalized learned head weights | Functional `heads * softmax(head_logits)`, uniform at initialization | Optimizer updates logits. No in-forward Parameter overwrite; a literal recursive softmax update is not specified as a valid gradient algorithm. |
| Eq. 5-6: similarity to important head, batch average | Cosines to the head selected by highest learned weight; mean over active reports | N is used inconsistently for heads and batch size. Follow the prose's batch mean. The iteration lag/state persistence is unspecified; use current pre-weight head outputs, with no cross-batch cache. |
| Eq. 7-9: geometric balance and rectification | `beta = exp(mean(log(clamp(abs(mean_cos), eps))))`; `relu(beta - mean_cos)`; multiply learned weights and head contexts | Combining the two weights multiplicatively is an explicit choice. No residual `+1` or normalization is silently added. |
| Eq. 9: feature LSTM | `ParallelDecoder.visual_cell`, input CLS representation plus weighted visual context | T1 is mapped to projected CLS. Cell state and concatenation convention are not fully defined in the paper. |
| Eq. 10-11: primary language LSTM/MLP | Primary cell takes CLS, current visual hidden state and previous word; keeps recurrent h/c | Both report heads predict the SAME next token after the supplied word. |
| Eq. 12-13: secondary LSTM/MLP | Secondary cell takes second attention context, current primary hidden state and same previous word, initialized from current primary h/c | Three separate LSTMCells; the two language cells have the same architecture, not shared parameters. Eq. 13 repeats the first-head symbol, treated as a notation error. No previous-step logit fusion or lookahead target. |
| III-C, Eq. 14: report-based label LSTM | `LabelEnhancement`: expected generated word embeddings, category LSTM, linear tag logits | Soft embeddings preserve gradients to both report heads. Pool final valid hidden state rather than an unspecified fixed-length flattening. No label/report input during inference. |
| Eq. 15-16: two CE terms and multi-label loss | `losses.objective`: independent next-token CE1 + 0.5 CE2 + 5 BCE-with-logits | Eq. 14 says softmax, while Eq. 16/SoftMarginLoss require independent sigmoid logits. Follow multi-label loss semantics, not sigmoid(softmax(logits)); this is a disclosed resolution, not an exact literal reproduction of both inconsistent equations. |
| IV-B: beam width 5 | `decoding.beam_search`, per-image log probabilities | Paper does not specify head fusion: use same-step probability mixture `(p1 + lambda*p2)/(1+lambda)`. Never combine unnormalized logits from different timesteps. |

## Important Limitations

The literal rectified cosine term can zero every head when head contexts are
identical. It also suppresses the selected base head when beta < 1. Those are
properties of the printed equations, not silently repaired with a new formula.
Tests expose these cases. Their compatibility with the paper's verbal claims
requires clarification from the authors.

Batch-averaged cosine weights make DAM depend on training batch composition.
Finished/padded reports are excluded from that average. Generation evaluates
one image and one beam hypothesis per step call, avoiding unrelated images or
other beam hypotheses in the cosine average. Equality is tested between training
forward and stepwise decoding for the same prefix and batch, not falsely claimed
across different batches.

The paper describes LEM as report enhancement but gives no executable word
replacement or reranking algorithm. Here LEM supervises generated report features
during training. It does not rewrite a generated report after decoding.

The text templates, tag vocabulary, modality encoding/fusion, true folds,
patient-level provenance, exact decay, dropout and stopping settings need author
confirmation before claiming historical reproduction. Existing reference risk
and confidence are targets, never clinical input features or generation controls.

## Fixed First-Run Choices

- 512 embedding/hidden dimensions, eight DAM heads, Adam LR 0.0004,
  lambda 0.5, tag coefficient 5, and beam width 5.
- AMP enabled on CUDA as stated by the paper. Weight decay 0.0001,
  gradient-norm clip 5, no added dropout, frozen pretrained backbone with a
  trainable projection: explicitly chosen where the paper lacks a complete setup.
- Seed 123, batch 16, five epochs, max decode steps 161; lowest validation
  objective selects the checkpoint. No EOS upweighting or repetition constraint.
- START and PAD cannot be generated as report words. END is learned normally;
  no minimum length, forced ending, confidence rewrite or field repair.
- Historical checkpoints cannot be loaded into this model. Runs require a new
  output directory and an explicit `--accept-reconstruction` acknowledgement.

## Acceptance Tests

Synthetic tests cover head weights/formulas and gradients, spatial attention,
batch permutation, no parameter mutation during forward, padding exclusion,
both heads' next-token/EOS alignment, LEM gradients, train/step equivalence,
beam termination and lack of state sharing across images, strict data and
checkpoint contracts, source archiving and no-overwrite safeguards.

These tests prove wiring and implementation invariants, not that actual fundus
reports are now correct. Only a user-run training/evaluation can assess repeated
reports, truncation, structural completeness and agreement with reference fields.
