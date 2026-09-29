"""Torch metrics and paired patient bootstrap, evaluated on the requested device.

CUDA is never replaced by CPU. Explicit ``device='cpu'`` is provided for small
reference tests. Bootstrap draws use a seeded Torch device generator, so CUDA
draws are not bitwise identical to the existing NumPy PCG64 implementation.
"""

import numpy as np
import torch
from tqdm.auto import tqdm

from .longitudinal import METRIC_NAMES, MIN_VALID_BOOTSTRAPS


def _weighted_metrics_tensor(y, probabilities, weights):
    """Return float64 metric vectors for nonnegative ``weights[draw, eye]``.

    Both classes must have positive weight in every row. Grouping tied scores
    before curve integration matches sklearn AUROC and average precision,
    including integer weights representing duplicated bootstrap observations.
    """
    if not all(isinstance(value, torch.Tensor) for value in (y, probabilities, weights)):
        raise TypeError("y, probabilities and weights must be tensors")
    if probabilities.device != y.device or weights.device != y.device:
        raise ValueError("All metric tensors must be on the same device")
    if (y.ndim != 1 or probabilities.shape != y.shape or y.numel() == 0
            or weights.ndim != 2 or weights.shape[1] != y.numel()):
        raise ValueError("Expected y/probabilities[N] and weights[B,N]")
    y = y.to(torch.float64)
    probabilities = probabilities.to(torch.float64)
    weights = weights.to(torch.float64)
    if (not bool(torch.all((y == 0) | (y == 1)))
            or not bool(torch.all(torch.isfinite(probabilities)))
            or not bool(torch.all((probabilities >= 0) & (probabilities <= 1)))
            or not bool(torch.all(torch.isfinite(weights) & (weights >= 0)))):
        raise ValueError("Require binary targets, finite probabilities in [0,1], and nonnegative weights")
    positive = weights * y
    negative = weights * (1 - y)
    positives, negatives = positive.sum(dim=1), negative.sum(dim=1)
    if not bool(torch.all((positives > 0) & (negatives > 0))):
        raise ValueError("Each weighted metric row requires both reference classes")
    predicted = (probabilities >= 0.5).to(torch.float64)
    tp = (positive * predicted).sum(dim=1)
    fp = (negative * predicted).sum(dim=1)
    sensitivity = tp / positives
    specificity = (negatives - fp) / negatives

    order = torch.argsort(probabilities, descending=True, stable=True)
    sorted_scores = probabilities[order]
    ends = torch.cat((sorted_scores[:-1] != sorted_scores[1:],
                      torch.ones(1, dtype=torch.bool, device=y.device)))
    curve_tp = torch.cumsum(positive[:, order], dim=1)[:, ends]
    curve_fp = torch.cumsum(negative[:, order], dim=1)[:, ends]
    origin = torch.zeros((weights.shape[0], 1), dtype=torch.float64, device=y.device)
    previous_tp = torch.cat((origin, curve_tp[:, :-1]), dim=1)
    previous_fp = torch.cat((origin, curve_fp[:, :-1]), dim=1)
    auroc = ((curve_fp - previous_fp) * (curve_tp + previous_tp)).sum(dim=1)
    auroc = auroc / (2 * positives * negatives)
    # Zero-weight leading thresholds have zero recall increment and precision 0.
    precision = curve_tp / (curve_tp + curve_fp).clamp_min(torch.finfo(torch.float64).tiny)
    auprc = ((curve_tp - previous_tp) * precision).sum(dim=1) / positives
    return {
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "auroc": auroc,
        "auprc": auprc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": 2 * tp / (positives + tp + fp),
    }


def paired_bootstrap_gpu(y_true, latest, longitudinal, groups, config, *, device,
                         progress_desc="Paired patient bootstrap"):
    """Compute paired cluster-bootstrap metrics/percentile CIs on ``device``.

    Each draw samples as many patients as the original cohort, with replacement.
    Its patient multiplicities weight all corresponding eyes together. All score
    arithmetic, confusion counts, curve integrals and percentiles stay on device;
    only patient-ID encoding and final result serialization run on the host.
    """
    device = torch.device(device)
    if device.type not in ("cuda", "cpu"):
        raise ValueError("Bootstrap supports CUDA, or explicit CPU reference testing")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA metrics were requested but CUDA is unavailable; no CPU fallback")
    y = torch.as_tensor(y_true, dtype=torch.float64, device=device)
    predictions = {
        "latest": torch.as_tensor(latest, dtype=torch.float64, device=device),
        "longitudinal": torch.as_tensor(longitudinal, dtype=torch.float64, device=device),
    }
    groups = np.asarray(groups)
    if y.ndim != 1 or groups.ndim != 1 or groups.shape[0] != y.numel():
        raise ValueError("Patient groups must align with a one-dimensional target array")
    patients, inverse = np.unique(groups, return_inverse=True)
    eye_patients = torch.as_tensor(inverse, dtype=torch.int64, device=device)
    point_weights = torch.ones((1, y.numel()), dtype=torch.float64, device=device)
    points = {method: _weighted_metrics_tensor(y, probabilities, point_weights)
              for method, probabilities in predictions.items()}
    generator = torch.Generator(device=device).manual_seed(config.seed)
    # Bound temporary tensors to approximately 64 MiB and at most 256 draws.
    chunk_size = min(256, max(1, (64 * 1024 ** 2) // max(1, 8 * (16 * y.numel() + 3 * len(patients)))))
    samples = {method: {name: [] for name in METRIC_NAMES} for method in predictions}
    deltas, skipped, valid_count = [], 0, 0
    with torch.no_grad(), tqdm(total=config.bootstrap_replicates, desc=progress_desc,
                              unit="draw", dynamic_ncols=True) as progress:
        progress.set_postfix(valid=0, skipped=0, refresh=False)
        for offset in range(0, config.bootstrap_replicates, chunk_size):
            size = min(chunk_size, config.bootstrap_replicates - offset)
            draws = torch.randint(len(patients), (size, len(patients)),
                                  generator=generator, device=device)
            counts = torch.zeros((size, len(patients)), dtype=torch.int64, device=device)
            counts.scatter_add_(1, draws, torch.ones_like(draws))
            weights = counts[:, eye_patients].to(torch.float64)
            positive_counts = (weights * y).sum(dim=1)
            negative_counts = (weights * (1 - y)).sum(dim=1)
            valid = (positive_counts > 0) & (negative_counts > 0)
            n_valid = int(valid.sum().item())
            skipped += size - n_valid
            valid_count += n_valid
            if n_valid:
                pair = {method: _weighted_metrics_tensor(y, probabilities, weights[valid])
                        for method, probabilities in predictions.items()}
                for method in samples:
                    for metric in METRIC_NAMES:
                        samples[method][metric].append(pair[method][metric])
                deltas.append(pair["longitudinal"]["balanced_accuracy"] - pair["latest"]["balanced_accuracy"])
            progress.set_postfix(valid=valid_count, skipped=skipped, refresh=False)
            progress.update(size)

    quantiles = torch.tensor([0.025, 0.975], dtype=torch.float64, device=device)

    def interval(chunks):
        if valid_count < MIN_VALID_BOOTSTRAPS:
            return {"ci_low": None, "ci_high": None}
        bounds = torch.quantile(torch.cat(chunks), quantiles, interpolation="linear")
        return {"ci_low": float(bounds[0].item()), "ci_high": float(bounds[1].item())}

    output = {
        method: {metric: {"estimate": float(points[method][metric][0].item()),
                          **interval(samples[method][metric])} for metric in METRIC_NAMES}
        for method in predictions
    }
    delta = points["longitudinal"]["balanced_accuracy"] - points["latest"]["balanced_accuracy"]
    metadata = {
        "requested": config.bootstrap_replicates, "valid": valid_count,
        "skipped_single_class": skipped, "seed": config.seed,
        "unit": "patient", "paired": True, "confidence_level": 0.95,
        "method": "percentile", "minimum_valid_replicates": MIN_VALID_BOOTSTRAPS,
        "conditional_on": "fixed_out_of_fold_predictions; no model refitting",
        "ci_status": "ok" if valid_count >= MIN_VALID_BOOTSTRAPS else "not_estimable",
        "backend": "torch", "device": str(device), "dtype": "float64",
        "rng": f"torch.Generator(device={device}) + torch.randint",
        "rng_note": "Device-specific Torch draws differ from NumPy PCG64; identical seeds do not imply identical bootstrap samples.",
        "bootstrap_chunk_size": chunk_size, "percentile_interpolation": "linear",
    }
    return output, {"estimate": float(delta[0].item()), **interval(deltas)}, metadata
