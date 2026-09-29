"""Prespecified retrospective, paired analysis of frozen visit-level scores.

Only scores observed by the last included visit enter these features. This is
retrospective progression assessment, not forecasting after an earlier landmark.
Neither reference labels nor patient IDs enter the feature matrix. The upstream
diagnostic model must already be fixed independently of this validation cohort.
"""

from dataclasses import asdict, dataclass
import re
from numbers import Integral, Real
from typing import Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm


ENDPOINTS = ("plr2", "plr3", "md_slope")
FEATURE_NAMES = ("last", "delta", "slope", "mean", "persistence")
METRIC_NAMES = ("balanced_accuracy", "auroc", "auprc", "sensitivity", "specificity", "f1")
MIN_VALID_BOOTSTRAPS = 20


@dataclass(frozen=True)
class EvaluationConfig:
    """Choices fixed before examining outcome-specific performance.

    ``persistence`` is the fraction of visit scores > persistence_threshold.
    The progression decision threshold is always 0.5 for both comparators.
    Bootstrap intervals condition on the fitted cross-validation predictions;
    they do not include uncertainty from refitting the models or feature choices.
    """

    n_splits: int = 3
    seed: int = 42
    bootstrap_replicates: int = 2000
    persistence_threshold: float = 0.5
    logistic_c: float = 1.0
    compute_device: str = "cpu"

    def __post_init__(self):
        if (not isinstance(self.compute_device, str)
                or not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", self.compute_device)):
            raise ValueError("compute_device must be cpu, cuda, or cuda:N")
        for name, minimum in (("n_splits", 2), ("bootstrap_replicates", MIN_VALID_BOOTSTRAPS)):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if (isinstance(self.seed, (bool, np.bool_)) or not isinstance(self.seed, Integral)
                or not 0 <= self.seed < 2 ** 32):
            raise ValueError("seed must be an integer in [0, 2**32)")
        threshold = _number(self.persistence_threshold, "persistence_threshold")
        if not 0 <= threshold <= 1:
            raise ValueError("persistence_threshold must lie in [0, 1]")
        if _number(self.logistic_c, "logistic_c") <= 0:
            raise ValueError("logistic_c must be positive")
        # Normalize NumPy scalar arguments so asdict() is JSON serializable.
        for name in ("n_splits", "seed", "bootstrap_replicates"):
            object.__setattr__(self, name, int(getattr(self, name)))
        for name in ("persistence_threshold", "logistic_c"):
            object.__setattr__(self, name, float(getattr(self, name)))


def _number(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _identifier(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _labels(value, eye_id):
    if not isinstance(value, Mapping):
        raise ValueError(f"{eye_id}: progression labels must be a mapping")
    result = {}
    for endpoint in ENDPOINTS:
        label = value.get(endpoint)
        if isinstance(label, (bool, np.bool_)) or not isinstance(label, Integral) or label not in (0, 1):
            raise ValueError(f"{eye_id}: {endpoint} label must be an integer 0 or 1")
        result[endpoint] = int(label)
    return result


def prepare_eye_records(visits: Sequence[Mapping], progression_labels: Mapping, min_visits=3):
    """Group independently scored visits without using outcomes for eligibility.

    Input order is irrelevant: visits are sorted by actual elapsed years within
    each eye. Duplicate sample IDs, duplicate times and conflicting patient IDs
    fail rather than silently collapsing data. Eyes below the minimum number of
    scored visits are excluded. Every included eye requires all three labels.
    """
    if isinstance(min_visits, (bool, np.bool_)) or not isinstance(min_visits, Integral) or min_visits < 3:
        raise ValueError("min_visits must be an integer >= 3")
    if not isinstance(progression_labels, Mapping):
        raise ValueError("progression_labels must be a mapping")
    seen, grouped = set(), {}
    for row in visits:
        if not isinstance(row, Mapping):
            raise ValueError("each scored visit must be a mapping")
        sample_id = _identifier(row.get("sample_id"), "sample_id")
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id {sample_id!r}")
        seen.add(sample_id)
        patient = _identifier(row.get("patient_id"), "patient_id")
        eye = _identifier(row.get("eye_id"), "eye_id")
        time = _number(row.get("time_years"), "time_years")
        if time < 0:
            raise ValueError("time_years must be nonnegative elapsed time")
        score = _number(row.get("glaucoma_score"), "glaucoma_score")
        if not 0 <= score <= 1:
            raise ValueError("glaucoma_score must lie in [0, 1]")
        if eye not in grouped:
            grouped[eye] = {"eye_id": eye, "patient_id": patient, "visits": []}
        if grouped[eye]["patient_id"] != patient:
            raise ValueError(f"{eye}: an eye must belong to exactly one patient")
        grouped[eye]["visits"].append((time, score))
    records = []
    for group in sorted(grouped.values(), key=lambda g: (g["patient_id"], g["eye_id"])):
        ordered = sorted(group["visits"])
        times, scores = zip(*ordered)
        if np.any(np.diff(times) <= 0):
            raise ValueError(f"{group['eye_id']}: visit times must strictly increase")
        if len(times) < min_visits:
            continue
        eye = group["eye_id"]
        labels = _labels(progression_labels.get(eye), eye)
        records.append({
            "eye_id": eye, "patient_id": group["patient_id"],
            "times": list(times), "scores": list(scores), "labels": labels,
            "n_visits": len(times), "followup_years": times[-1] - times[0],
        })
    return records


def _validate_eye_records(eye_records):
    seen = set()
    for record in eye_records:
        if not isinstance(record, Mapping):
            raise ValueError("each eye record must be a mapping")
        eye = _identifier(record.get("eye_id"), "eye_id")
        if eye in seen:
            raise ValueError(f"duplicate eye_id {eye!r}")
        seen.add(eye)
        _identifier(record.get("patient_id"), "patient_id")
        _labels(record.get("labels"), eye)
        times, scores = record.get("times"), record.get("scores")
        if not isinstance(times, (list, tuple, np.ndarray)) or not isinstance(scores, (list, tuple, np.ndarray)):
            raise ValueError(f"{eye}: times and scores must be sequences")
        if len(times) < 3 or len(times) != len(scores):
            raise ValueError(f"{eye}: expected at least three matching times and scores")
        times = np.array([_number(t, "time_years") for t in times])
        scores = np.array([_number(s, "glaucoma_score") for s in scores])
        if np.any(times < 0) or np.any(np.diff(times) <= 0):
            raise ValueError(f"{eye}: nonnegative visit times must strictly increase")
        if np.any((scores < 0) | (scores > 1)):
            raise ValueError(f"{eye}: glaucoma scores must lie in [0, 1]")
        if record.get("n_visits", len(times)) != len(times):
            raise ValueError(f"{eye}: n_visits does not match the score sequence")
        if "followup_years" in record and not np.isclose(
            _number(record["followup_years"], "followup_years"), times[-1] - times[0]
        ):
            raise ValueError(f"{eye}: followup_years does not match the included time window")


def temporal_features(eye_records, persistence_threshold=0.5, *, device="cpu"):
    """Return [last, last-first, OLS slope/year, mean, persistence] per eye."""
    if device != "cpu":
        from .gpu_heads import temporal_features_gpu
        return temporal_features_gpu(eye_records, persistence_threshold, device=device)
    eye_records = list(eye_records)
    threshold = _number(persistence_threshold, "persistence_threshold")
    if not 0 <= threshold <= 1:
        raise ValueError("persistence_threshold must lie in [0, 1]")
    _validate_eye_records(eye_records)
    result = []
    for record in eye_records:
        times = np.asarray(record["times"], dtype=np.float64)
        scores = np.asarray(record["scores"], dtype=np.float64)
        centered = times - np.mean(times)
        slope = np.dot(centered, scores - np.mean(scores)) / np.dot(centered, centered)
        result.append([
            scores[-1], scores[-1] - scores[0], slope, np.mean(scores),
            np.mean(scores > threshold),
        ])
    output = np.asarray(result, dtype=np.float64).reshape((-1, len(FEATURE_NAMES)))
    if not np.all(np.isfinite(output)):
        raise ValueError("Time/score arithmetic produced nonfinite temporal features")
    return output


def _metrics(y_true, probabilities):
    """Metrics require both reference classes; AP is the AUPRC convention."""
    predicted = probabilities >= 0.5
    positive = y_true == 1
    tp = int(np.sum(predicted & positive))
    tn = int(np.sum(~predicted & ~positive))
    fp = int(np.sum(predicted & ~positive))
    fn = int(np.sum(~predicted & positive))
    sensitivity, specificity = tp / (tp + fn), tn / (tn + fp)
    f1_denominator = 2 * tp + fp + fn
    return {
        "balanced_accuracy": float((sensitivity + specificity) / 2),
        "auroc": float(roc_auc_score(y_true, probabilities)),
        "auprc": float(average_precision_score(y_true, probabilities)),
        "sensitivity": float(sensitivity), "specificity": float(specificity),
        "f1": float(2 * tp / f1_denominator) if f1_denominator else 0.0,
    }


def _patient_bootstrap_indices(groups, rng):
    """Sample patients with replacement, retaining every eye and multiplicity."""
    patients = np.unique(groups)
    sampled = rng.choice(patients, size=len(patients), replace=True)
    return np.concatenate([np.flatnonzero(groups == patient) for patient in sampled])


def _paired_bootstrap(y_true, latest, longitudinal, groups, config, *,
                      progress_desc="Paired patient bootstrap"):
    if config.compute_device != "cpu":
        from .gpu_metrics import paired_bootstrap_gpu
        return paired_bootstrap_gpu(y_true, latest, longitudinal, groups, config,
                                    device=config.compute_device, progress_desc=progress_desc)
    rng = np.random.default_rng(config.seed)
    samples = {method: {metric: [] for metric in METRIC_NAMES} for method in ("latest", "longitudinal")}
    deltas, skipped = [], 0
    bootstrap_progress = tqdm(range(config.bootstrap_replicates), desc=progress_desc,
                              unit="draw", dynamic_ncols=True)
    bootstrap_progress.set_postfix(valid=0, skipped=0, refresh=False)
    for _ in bootstrap_progress:
        indices = _patient_bootstrap_indices(groups, rng)
        labels = y_true[indices]
        if np.unique(labels).size < 2:
            skipped += 1
            bootstrap_progress.set_postfix(valid=len(deltas), skipped=skipped, refresh=False)
            continue
        pair = {
            "latest": _metrics(labels, latest[indices]),
            "longitudinal": _metrics(labels, longitudinal[indices]),
        }
        for method in samples:
            for metric in METRIC_NAMES:
                samples[method][metric].append(pair[method][metric])
        deltas.append(pair["longitudinal"]["balanced_accuracy"] - pair["latest"]["balanced_accuracy"])
        bootstrap_progress.set_postfix(valid=len(deltas), skipped=skipped, refresh=False)

    def interval(values):
        if len(values) < MIN_VALID_BOOTSTRAPS:
            return {"ci_low": None, "ci_high": None}
        low, high = np.percentile(values, [2.5, 97.5])
        return {"ci_low": float(low), "ci_high": float(high)}

    output = {}
    for method, probabilities in (("latest", latest), ("longitudinal", longitudinal)):
        point = _metrics(y_true, probabilities)
        output[method] = {
            metric: {"estimate": point[metric], **interval(samples[method][metric])}
            for metric in METRIC_NAMES
        }
    delta = output["longitudinal"]["balanced_accuracy"]["estimate"] - output["latest"]["balanced_accuracy"]["estimate"]
    metadata = {
        "requested": config.bootstrap_replicates, "valid": len(deltas),
        "skipped_single_class": skipped, "seed": config.seed,
        "unit": "patient", "paired": True, "confidence_level": 0.95,
        "method": "percentile", "minimum_valid_replicates": MIN_VALID_BOOTSTRAPS,
        "conditional_on": "fixed_out_of_fold_predictions; no model refitting",
        "ci_status": "ok" if len(deltas) >= MIN_VALID_BOOTSTRAPS else "not_estimable",
    }
    return output, {"estimate": float(delta), **interval(deltas)}, metadata


def _class_counts(labels):
    return {"negative": int(np.sum(labels == 0)), "positive": int(np.sum(labels == 1))}


def _make_folds(labels, groups, config):
    positive_patients = np.unique(groups[labels == 1]).size
    negative_patients = np.unique(groups[labels == 0]).size
    maximum = min(config.n_splits, positive_patients, negative_patients, np.unique(groups).size)
    if maximum < 2:
        return None, "At least two distinct patients containing each reference class are required."
    # Reduce only for class feasibility. Never retry seeds or choose folds by
    # predictive performance. Both A and B reuse this exact endpoint split.
    for n_splits in range(maximum, 1, -1):
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=config.seed)
        candidate = list(splitter.split(np.zeros((len(labels), 1)), labels, groups))
        if all(np.unique(labels[train]).size == 2 and np.unique(labels[test]).size == 2
               for train, test in candidate):
            return candidate, None
    return None, "No group-stratified split with both classes in every training and test fold was feasible at the prespecified seed."


def _fit_fold(train_features, test_features, labels, config):
    if config.compute_device != "cpu":
        from .gpu_heads import fit_fold_gpu
        return fit_fold_gpu(train_features, test_features, labels, config,
                            device=config.compute_device)
    scaler = StandardScaler().fit(train_features)
    model = LogisticRegression(
        penalty="l2", C=config.logistic_c, solver="liblinear", class_weight="balanced",
        max_iter=2000, tol=1e-4, random_state=config.seed,
    )
    model.fit(scaler.transform(train_features), labels)
    if int(np.max(model.n_iter_)) >= model.max_iter:
        raise RuntimeError("The prespecified logistic regression did not converge; results are not reportable.")
    probabilities = model.predict_proba(scaler.transform(test_features))[:, 1]
    provenance = {
        "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
        "scaler_variance": scaler.var_.tolist(), "coefficient_standardized": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]), "iterations": int(model.n_iter_[0]),
    }
    return probabilities, provenance


def evaluate_longitudinal(eye_records, config=None):
    """Patient-grouped OOF comparison and paired patient bootstrap for each label.

    Endpoint-specific folds are allowed; A and B always share folds, eyes and
    labels within an endpoint. A single full-cohort fit is never used to report
    discrimination. Too few independent positive/negative patients, or an
    infeasible fixed-seed split, produce an explicit ``not_estimable`` result.
    """
    config = EvaluationConfig() if config is None else config
    if not isinstance(config, EvaluationConfig):
        raise TypeError("config must be an EvaluationConfig")
    eye_records = list(eye_records)
    _validate_eye_records(eye_records)
    records = sorted(eye_records, key=lambda r: (r["patient_id"], r["eye_id"]))
    features = temporal_features(records, config.persistence_threshold, device=config.compute_device)
    groups = np.asarray([record["patient_id"] for record in records], dtype=str)
    result = {
        "config": {
            **asdict(config), "decision_threshold": 0.5,
            "feature_names": list(FEATURE_NAMES), "persistence_rule": "fraction of visit scores > threshold",
            "latest_features": ["last"], "logistic_penalty": "l2",
            "logistic_solver": ("liblinear" if config.compute_device == "cpu" else "torch_newton"),
            "logistic_class_weight": "balanced", "logistic_max_iter": 2000, "logistic_tol": 1e-4,
            "standardization": "training fold only", "fold_method": "StratifiedGroupKFold",
            "analysis": "retrospective progression assessment at last included visit",
            "auprc_definition": "average_precision",
            "method_specs": {
                "latest": "L2 logistic regression using the last visit score",
                "longitudinal": "L2 logistic regression using last, delta, OLS slope/year, mean, persistence",
                "persistence": f"fraction of visit scores strictly > {config.persistence_threshold}",
                "preprocessing": "StandardScaler fitted on training patients only",
                "classification": "predicted progression score >= 0.5",
                "probability_interpretation": "class-weighted logistic scores; not clinically calibrated risk",
                "fold_selection": "prespecified seed; reduce fold count for both-class feasibility only",
            },
        },
        "endpoints": {}, "predictions": [],
        "features": [
            {"eye_id": record["eye_id"], "patient_id": record["patient_id"],
             "n_visits": len(record["times"]),
             "followup_years": float(record["times"][-1] - record["times"][0]),
             **dict(zip(FEATURE_NAMES, values.tolist()))}
            for record, values in zip(records, features)
        ],
    }
    endpoint_progress = tqdm(ENDPOINTS, desc="Progression endpoints", unit="endpoint", dynamic_ncols=True)
    for endpoint in endpoint_progress:
        endpoint_progress.set_postfix_str(endpoint, refresh=True)
        labels = np.asarray([record["labels"][endpoint] for record in records], dtype=int)
        summary = {
            "n_eyes": len(records), "n_patients": int(np.unique(groups).size),
            "n_positive_eyes": int(np.sum(labels == 1)), "n_negative_eyes": int(np.sum(labels == 0)),
            "n_positive_patients": int(np.unique(groups[labels == 1]).size),
            "n_negative_patients": int(np.unique(groups[labels == 0]).size),
            "n_splits": 0, "folds": [],
        }
        folds, reason = _make_folds(labels, groups, config)
        if folds is None:
            summary.update(status="not_estimable", reason=reason)
            result["endpoints"][endpoint] = summary
            endpoint_progress.set_postfix_str(f"{endpoint}: not estimable", refresh=True)
            continue
        summary.update(status="ok", n_splits=len(folds))
        latest, longitudinal = np.full(len(records), np.nan), np.full(len(records), np.nan)
        fold_ids = np.full(len(records), -1, dtype=int)
        fold_progress = tqdm(enumerate(folds), total=len(folds), desc=f"{endpoint}: patient folds",
                             unit="fold", dynamic_ncols=True, leave=False)
        for fold_id, (train, test) in fold_progress:
            if set(groups[train]) & set(groups[test]):
                raise RuntimeError("Patient leakage detected between training and test folds")
            if np.any(fold_ids[test] != -1):
                raise RuntimeError("A held-out eye appears in more than one fold")
            fold_progress.set_postfix_str(f"fold {fold_id + 1}: latest head", refresh=True)
            latest[test], latest_model = _fit_fold(features[train, :1], features[test, :1], labels[train], config)
            fold_progress.set_postfix_str(f"fold {fold_id + 1}: longitudinal head", refresh=True)
            longitudinal[test], longitudinal_model = _fit_fold(features[train], features[test], labels[train], config)
            fold_ids[test] = fold_id
            summary["folds"].append({
                "fold": fold_id, "train_patient_ids": np.unique(groups[train]).tolist(),
                "test_patient_ids": np.unique(groups[test]).tolist(),
                "train_eye_ids": [records[i]["eye_id"] for i in train],
                "test_eye_ids": [records[i]["eye_id"] for i in test],
                "train_class_counts": _class_counts(labels[train]), "test_class_counts": _class_counts(labels[test]),
                "latest": {"feature_names": ["last"], **latest_model},
                "longitudinal": {"feature_names": list(FEATURE_NAMES), **longitudinal_model},
            })
        if np.any(fold_ids < 0) or not np.all(np.isfinite(latest)) or not np.all(np.isfinite(longitudinal)):
            raise RuntimeError("Every included eye must have finite paired out-of-fold probabilities")
        summary["metrics"], summary["delta_balanced_accuracy"], summary["bootstrap"] = _paired_bootstrap(
            labels, latest, longitudinal, groups, config,
            progress_desc=f"{endpoint}: paired patient bootstrap",
        )
        result["endpoints"][endpoint] = summary
        result["predictions"].extend([
            {"endpoint": endpoint, "eye_id": record["eye_id"], "patient_id": record["patient_id"],
             "fold": int(fold_ids[i]), "y_true": int(labels[i]),
             "latest_probability": float(latest[i]), "longitudinal_probability": float(longitudinal[i])}
            for i, record in enumerate(records)
        ])
    return result
