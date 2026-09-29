"""Explicit-device float64 temporal summaries and balanced logistic heads.

The production caller supplies a CUDA device. Explicit ``device="cpu"`` exists
for small numerical reference tests; CUDA failures never trigger a CPU fallback.
The Newton solver optimizes the same L2 logistic objective as binary liblinear
with balanced class weights and intercept_scaling=1, including the intercept
penalty. Its stopping rule and numerical trajectory differ from liblinear, so
the implementation does not promise bitwise-identical fitted coefficients.
"""

import numpy as np


_MAX_ITER = 2000
_TOL = 1e-4
_MAX_LINE_SEARCH = 60
_ARMIJO = 1e-4


def _torch_device(device):
    import torch

    try:
        resolved = torch.device(device)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError("device must explicitly identify a CPU or CUDA device") from error
    if resolved.type not in ("cpu", "cuda"):
        raise ValueError("Only explicit CPU reference or CUDA execution is supported")
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; no CPU fallback is performed")
        index = torch.cuda.current_device() if resolved.index is None else resolved.index
        if index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device {index} is unavailable; no CPU fallback is performed")
        resolved = torch.device("cuda", index)
    return torch, resolved


def initialize_cuda_linalg(gpu_ids):
    """Load CUDA solve backends serially before launching endpoint threads.

    PyTorch 2.0.1's lazy CUDA linalg wrapper is unsafe on concurrent first use
    (pytorch/pytorch#90613). Exercise the float64 2x2 and 6x6 systems used by
    the latest/longitudinal heads on each selected device, then synchronize.
    Constant tensors do not consume random draws or use any study data.
    The caller must finish this initialization before creating its workers.
    """
    import torch

    with torch.no_grad():
        for gpu_id in gpu_ids:
            device = torch.device("cuda", gpu_id)
            with torch.cuda.device(device):
                for size in (2, 6):
                    matrix = torch.eye(size, dtype=torch.float64, device=device)
                    rhs = torch.ones(size, dtype=torch.float64, device=device)
                    torch.linalg.solve(matrix, rhs)
                torch.cuda.synchronize(device)


def _feature_arrays(train_features, test_features, labels):
    train, test, labels = map(np.asarray, (train_features, test_features, labels))
    for name, values in (("train_features", train), ("test_features", test)):
        if (values.ndim != 2 or not values.shape[0] or not values.shape[1]
                or values.dtype.kind not in "fiu" or not np.all(np.isfinite(values))):
            raise ValueError(f"{name} must be a nonempty finite numeric matrix")
    if train.shape[1] != test.shape[1]:
        raise ValueError("Training and test feature widths must match")
    if (labels.ndim != 1 or len(labels) != len(train) or labels.dtype.kind not in "fiu"
            or not np.all(np.isin(labels, (0, 1))) or np.unique(labels).size != 2):
        raise ValueError("Training labels must contain both binary classes and align with training rows")
    return train.astype(np.float64), test.astype(np.float64), labels.astype(np.float64)


def fit_fold_gpu(train_features, test_features, labels, config, *, device):
    """Fit a training-only scaler and L2 logistic head on the explicit device.

    The objective is ``0.5 * ||theta||^2 + C * sum(w_i * logistic_loss_i)``.
    ``theta`` includes the intercept; ``w_i = n / (2 * n_class_i)``. A full
    Newton direction uses the positive-definite Hessian and Armijo backtracking.
    Convergence requires the absolute infinity norm of the gradient <= 1e-4;
    failure within 2,000 Newton steps raises instead of reporting predictions.
    """
    from .longitudinal import _number

    train, test, labels = _feature_arrays(train_features, test_features, labels)
    c = _number(config.logistic_c, "logistic_c")
    if c <= 0:
        raise ValueError("logistic_c must be positive")
    torch, resolved = _torch_device(device)
    with torch.no_grad():
        x = torch.as_tensor(train, dtype=torch.float64, device=resolved)
        test_x = torch.as_tensor(test, dtype=torch.float64, device=resolved)
        y = torch.as_tensor(labels, dtype=torch.float64, device=resolved)
        n, p = x.shape

        # Corrected two-pass population variance and the same numerical-constant
        # criterion used by sklearn StandardScaler. Test rows never participate.
        mean = x.sum(dim=0) / n
        centered = x - mean
        variance = ((centered.square().sum(dim=0) - centered.sum(dim=0).square() / n) / n).clamp_min(0)
        eps = torch.finfo(torch.float64).eps
        constant_bound = n * eps * variance + (n * mean * eps).square()
        scale = torch.where(variance <= constant_bound, torch.ones_like(variance), variance.sqrt())
        standardized = centered / scale
        standardized_test = (test_x - mean) / scale
        if not (bool(torch.isfinite(standardized).all()) and bool(torch.isfinite(standardized_test).all())
                and bool(torch.isfinite(mean).all()) and bool(torch.isfinite(variance).all())):
            raise ValueError("Feature standardization produced nonfinite values")
        design = torch.cat((standardized, torch.ones((n, 1), dtype=torch.float64, device=resolved)), dim=1)
        positive = y == 1
        positive_count = int(positive.sum().item())
        class_weights = (n / (2 * (n - positive_count)), n / (2 * positive_count))
        weights = torch.where(positive, torch.full_like(y, class_weights[1]), torch.full_like(y, class_weights[0]))
        identity = torch.eye(p + 1, dtype=torch.float64, device=resolved)
        theta = torch.zeros(p + 1, dtype=torch.float64, device=resolved)

        def objective(parameters):
            margin = (1 - 2 * y) * (design @ parameters)
            return .5 * parameters.square().sum() + c * (
                weights * torch.nn.functional.softplus(margin)).sum()

        total_backtracks = 0
        initial_gradient = None
        for iteration in range(_MAX_ITER + 1):
            logits = design @ theta
            probability = torch.sigmoid(logits)
            residual = torch.where(positive, -torch.sigmoid(-logits), probability)
            gradient = theta + c * (design.T @ (weights * residual))
            gradient_norm = float(gradient.abs().max().item())
            loss = objective(theta)
            if not np.isfinite(gradient_norm) or not bool(torch.isfinite(loss)):
                raise RuntimeError("The logistic Newton solver produced nonfinite objective or gradient")
            if initial_gradient is None:
                initial_gradient = gradient_norm
            if gradient_norm <= _TOL:
                break
            if iteration == _MAX_ITER:
                raise RuntimeError("The explicit-device logistic regression did not converge; results are not reportable")
            curvature = weights * probability * torch.sigmoid(-logits)
            hessian = identity + c * (design.T @ (curvature[:, None] * design))
            direction = torch.linalg.solve(hessian, gradient)
            decrement = torch.dot(gradient, direction)
            if not bool(torch.isfinite(direction).all()) or float(decrement.item()) <= 0:
                raise RuntimeError("The logistic Newton solver could not obtain a finite descent direction")
            step = 1.0
            for backtrack in range(_MAX_LINE_SEARCH):
                candidate = theta - step * direction
                candidate_loss = objective(candidate)
                if bool(torch.isfinite(candidate_loss)) and bool(
                        candidate_loss <= loss - _ARMIJO * step * decrement):
                    theta = candidate
                    total_backtracks += backtrack
                    break
                step *= .5
            else:
                raise RuntimeError("The logistic Newton line search failed; results are not reportable")

        probabilities = torch.sigmoid(standardized_test @ theta[:-1] + theta[-1])
        if not bool(torch.isfinite(probabilities).all()):
            raise RuntimeError("The fitted logistic head produced nonfinite predictions")
        metadata = {
            "scaler_mean": mean.cpu().tolist(), "scaler_scale": scale.cpu().tolist(),
            "scaler_variance": variance.cpu().tolist(), "coefficient_standardized": theta[:-1].cpu().tolist(),
            "intercept": float(theta[-1].item()), "iterations": iteration,
            "backend": "torch", "device": str(resolved), "dtype": "float64",
            "solver": "damped_newton_armijo", "max_iter": _MAX_ITER, "tol": _TOL,
            "objective": "0.5 * (sum(coefficients^2) + intercept^2) + C * sum(balanced_weight * logistic_loss)",
            "intercept_regularized": True, "intercept_scaling": 1.0,
            "class_weight": "balanced", "class_weights": {"0": class_weights[0], "1": class_weights[1]},
            "logistic_c": c, "gradient_inf_norm": gradient_norm,
            "initial_gradient_inf_norm": initial_gradient,
            "convergence_rule": "absolute gradient infinity norm <= tol",
            "objective_value": float(loss.item()), "line_search_backtracks": total_backtracks,
            "numerical_equivalence": "same regularized objective as binary liblinear; different solver and stopping rule",
        }
        return probabilities.cpu().numpy(), metadata


def temporal_features_gpu(records, threshold=0.5, *, device):
    """Compute [last, delta, OLS slope/year, mean, persistence > threshold].

    Ragged visit histories are padded and masked on the requested device. Only
    scores and actual elapsed times enter the arithmetic; reference labels are
    validated for schema compatibility but never enter feature construction.
    """
    from .longitudinal import FEATURE_NAMES, _number, _validate_eye_records

    records = list(records)
    threshold = _number(threshold, "persistence_threshold")
    if not 0 <= threshold <= 1:
        raise ValueError("persistence_threshold must lie in [0, 1]")
    _validate_eye_records(records)
    torch, resolved = _torch_device(device)
    if not records:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float64)
    lengths = np.asarray([len(row["times"]) for row in records], dtype=np.int64)
    times = np.zeros((len(records), int(lengths.max())), dtype=np.float64)
    scores = np.zeros_like(times)
    for index, record in enumerate(records):
        times[index, :lengths[index]] = record["times"]
        scores[index, :lengths[index]] = record["scores"]
    with torch.no_grad():
        t = torch.as_tensor(times, dtype=torch.float64, device=resolved)
        s = torch.as_tensor(scores, dtype=torch.float64, device=resolved)
        counts = torch.as_tensor(lengths, dtype=torch.int64, device=resolved)
        mask = torch.arange(times.shape[1], device=resolved)[None, :] < counts[:, None]
        mean_time = t.sum(dim=1) / counts
        mean_score = s.sum(dim=1) / counts
        centered_time = (t - mean_time[:, None]) * mask
        centered_score = (s - mean_score[:, None]) * mask
        slope = (centered_time * centered_score).sum(dim=1) / centered_time.square().sum(dim=1)
        last = s[torch.arange(len(records), device=resolved), counts - 1]
        persistence = ((s > threshold) & mask).sum(dim=1).to(torch.float64) / counts
        result = torch.stack((last, last - s[:, 0], slope, mean_score, persistence), dim=1)
        if not bool(torch.isfinite(result).all()):
            raise ValueError("Time/score arithmetic produced nonfinite temporal features")
        return result.cpu().numpy()
