"""GRAPE-only adaptation: nested patient-held-out progression assessment.

Frozen ImageNet image features are inputs; no fitting on GRAPE happens before
the outer split. Each endpoint has its own XGBoost visit model. The whole-eye
progression label is a weak label repeated across its visits, not a contemporaneous
clinical state. Inner out-of-fold visit scores train the two logistic heads;
outer-test patients never enter either level of fitting.
"""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from numbers import Integral
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm

from .config import GlaBoostConfig
from .data import VisitInput
from .encoders import resolve_image_devices
from .gpu_heads import initialize_cuda_linalg
from .longitudinal import (
    ENDPOINTS, FEATURE_NAMES, EvaluationConfig, _class_counts, _fit_fold,
    _identifier, _labels, _number, _paired_bootstrap, temporal_features,
)
from .model import assert_xgb_backend, make_xgb_classifier


def _inputs(visits, features, progression_labels):
    visits = list(visits)
    if not visits or not all(isinstance(visit, VisitInput) for visit in visits):
        raise ValueError("Provide a nonempty sequence of VisitInput records")
    if not hasattr(progression_labels, "get"):
        raise ValueError("progression_labels must be a mapping")
    features = np.asarray(features)
    if (features.ndim != 2 or features.shape[0] != len(visits) or features.shape[1] < 1
            or features.dtype.kind not in "fiu" or not np.all(np.isfinite(features))):
        raise ValueError("features must be a finite numeric matrix aligned with visits")
    features = features.astype(np.float32)
    if not np.all(np.isfinite(features)):
        raise ValueError("features overflow float32")
    seen, grouped = set(), {}
    for i, visit in enumerate(visits):
        sample = _identifier(visit.sample_id, "sample_id")
        patient = _identifier(visit.patient_id, "patient_id")
        eye = _identifier(visit.eye_id, "eye_id")
        time = _number(visit.time_years, "time_years")
        if sample in seen:
            raise ValueError(f"duplicate sample_id {sample!r}")
        seen.add(sample)
        if time < 0:
            raise ValueError("time_years must be nonnegative")
        if eye not in grouped:
            grouped[eye] = {"eye_id": eye, "patient_id": patient, "indices": []}
        if grouped[eye]["patient_id"] != patient:
            raise ValueError(f"{eye}: an eye must belong to exactly one patient")
        grouped[eye]["indices"].append(i)
    records = []
    for record in sorted(grouped.values(), key=lambda row: (row["patient_id"], row["eye_id"])):
        indices = sorted(record["indices"], key=lambda i: visits[i].time_years)
        times = [float(visits[i].time_years) for i in indices]
        if len(indices) < 3:
            raise ValueError(f"{record['eye_id']}: at least three eligible image visits are required")
        if np.any(np.diff(times) <= 0):
            raise ValueError(f"{record['eye_id']}: visit times must strictly increase")
        records.append({
            **record, "indices": indices, "times": times, "n_visits": len(times),
            "followup_years": times[-1] - times[0],
            "labels": _labels(progression_labels.get(record["eye_id"]), record["eye_id"]),
        })
    return visits, features, records


def _patient_folds(labels, groups, requested, seed, *, require_test_classes):
    """Stratify unique patients by any positive eye, retaining both eyes.

    Fold count is reduced only for reference-class feasibility, never by scores.
    In an inner split only training needs both classes: no inner validation metric
    is selected or reported.
    """
    patients = np.unique(groups)
    strata = np.asarray([int(np.any(labels[groups == patient] == 1)) for patient in patients])
    counts = np.bincount(strata, minlength=2)
    maximum = min(requested, int(np.min(counts)))
    if maximum < 2:
        return None, "At least two patients with a positive eye and two patients with only negative eyes are required for patient stratification."
    for n_splits in range(maximum, 1, -1):
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        candidate = []
        for train_patients, test_patients in splitter.split(patients, strata):
            train = np.flatnonzero(np.isin(groups, patients[train_patients]))
            test = np.flatnonzero(np.isin(groups, patients[test_patients]))
            candidate.append((train, test))
        if all(np.unique(labels[train]).size == 2 and
               (not require_test_classes or np.unique(labels[test]).size == 2)
               for train, test in candidate):
            return candidate, None
    return None, "No patient-stratified split with both training classes was feasible at the prespecified seed."


def _visit_indices(records, eyes):
    return np.asarray([i for eye in eyes for i in records[eye]["indices"]], dtype=int)


def _eye_weighted_labels(records, eyes, endpoint):
    labels = np.asarray([records[eye]["labels"][endpoint]
                         for eye in eyes for _ in records[eye]["indices"]], dtype=int)
    weights = np.asarray([1.0 / len(records[eye]["indices"])
                          for eye in eyes for _ in records[eye]["indices"]])
    weights /= np.mean(weights)
    return labels, weights


def _score_records(records, eyes, visit_scores):
    result = []
    for eye in eyes:
        record = records[eye]
        result.append({key: record[key] for key in
                       ("eye_id", "patient_id", "times", "n_visits", "followup_years", "labels")})
        result[-1]["scores"] = [float(visit_scores[i]) for i in record["indices"]]
    return result


def _save_base(model, directory, metadata):
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "model.json"
    model.save_model(str(path))
    metadata = dict(metadata, model_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    with (directory / "metadata.json").open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _base_scores(records, visits, features, endpoint, train, test, config, *,
                 outer_fold, inner_fold, model_dir):
    train_indices = _visit_indices(records, train)
    test_indices = _visit_indices(records, test)
    train_patients = sorted({records[i]["patient_id"] for i in train})
    test_patients = sorted({records[i]["patient_id"] for i in test})
    if set(train_patients) & set(test_patients):
        raise RuntimeError("Patient leakage detected in the base model")
    labels, weights = _eye_weighted_labels(records, train, endpoint)
    if np.unique(labels).size != 2:
        raise RuntimeError("The base model requires both progression classes")
    model = make_xgb_classifier(config)
    model.fit(features[train_indices], labels, sample_weight=weights)
    if config.tree_method == "gpu_hist":
        assert_xgb_backend(model, config)
    probabilities = np.asarray(model.predict_proba(features[test_indices]))
    if (probabilities.shape != (len(test_indices), 2) or not np.all(np.isfinite(probabilities))
            or np.any((probabilities < 0) | (probabilities > 1))
            or not np.allclose(probabilities.sum(axis=1), 1)):
        raise RuntimeError("Base model returned invalid progression probabilities")
    role = "outer_test" if inner_fold is None else "inner_oof"
    metadata = {
        "endpoint": endpoint, "outer_fold": outer_fold, "inner_fold": inner_fold,
        "role": role, "train_patient_ids": train_patients, "test_patient_ids": test_patients,
        "train_eye_ids": [records[i]["eye_id"] for i in train],
        "test_eye_ids": [records[i]["eye_id"] for i in test],
        "train_sample_ids": [visits[i].sample_id for i in train_indices],
        "test_sample_ids": [visits[i].sample_id for i in test_indices],
        "target": "whole-eye retrospective progression label repeated across visits (weak supervision)",
        "target_mapping": {"0": "nonprogression", "1": "progression"},
        "sample_weight": "inverse eligible visit count per training eye, normalized to mean 1",
        "train_visit_class_counts": _class_counts(labels),
        "train_eye_class_counts": _class_counts(np.array([records[i]["labels"][endpoint] for i in train])),
        "n_features": features.shape[1], "features": "frozen pretrained image embeddings only",
        "hyperparameters": {
            "objective": "binary:logistic", "eval_metric": "logloss", "learning_rate": config.learning_rate,
            "max_depth": config.max_depth, "n_estimators": config.n_estimators,
            "random_state": config.random_state, "n_jobs": config.n_jobs, "tree_method": config.tree_method,
            "subsample": 1.0, "colsample_bytree": 1.0, "reg_alpha": 0.0, "reg_lambda": 1.0,
            "scale_pos_weight": 1.0,
            **({"gpu_id": config.gpu_id, "predictor": "gpu_predictor"}
               if config.tree_method == "gpu_hist" else {}),
        },
    }
    if model_dir is not None:
        name = "outer_base" if inner_fold is None else f"inner_{inner_fold}_base"
        relative = Path(endpoint) / f"outer_{outer_fold}" / name
        _save_base(model, model_dir / relative, metadata)
        metadata["artifact"] = str(relative)
    rows = [{
        "endpoint": endpoint, "outer_fold": outer_fold, "inner_fold": inner_fold,
        "role": role, "sample_id": visits[i].sample_id, "patient_id": visits[i].patient_id,
        "eye_id": visits[i].eye_id, "time_years": float(visits[i].time_years),
        "progression_score": float(probabilities[j, 1]),
    } for j, i in enumerate(test_indices)]
    return test_indices, probabilities[:, 1], metadata, rows


def _train_endpoint(endpoint, records, visits, features, groups, config, evaluation, *,
                    model_dir, inner_splits):
    """Train one isolated endpoint; all mutable results belong to this call."""
    result = {"endpoints": {}, "predictions": [], "features": [],
              "visit_predictions": [], "eye_records": []}
    labels = np.asarray([record["labels"][endpoint] for record in records], dtype=int)
    summary = {
        "compute_device": evaluation.compute_device,
        "n_eyes": len(records), "n_patients": int(np.unique(groups).size),
        "n_positive_eyes": int(np.sum(labels == 1)), "n_negative_eyes": int(np.sum(labels == 0)),
        "n_positive_patients": int(np.unique(groups[labels == 1]).size),
        "n_negative_patients": int(np.unique(groups[labels == 0]).size), "n_splits": 0, "folds": [],
    }
    outer, reason = _patient_folds(labels, groups, evaluation.n_splits, evaluation.seed,
                                   require_test_classes=True)
    inner_plans = []
    if outer is not None:
        for outer_id, (train, _) in enumerate(outer):
            inner, inner_reason = _patient_folds(labels[train], groups[train], int(inner_splits),
                                                evaluation.seed, require_test_classes=False)
            if inner is None:
                reason = f"Outer fold {outer_id} cannot support patient-held-out inner training: {inner_reason}"
                outer = None
                break
            inner_plans.append([(train[inner_train], train[inner_test]) for inner_train, inner_test in inner])
    if outer is None:
        result["endpoints"][endpoint] = dict(summary, status="not_estimable", reason=reason)
        return result
    latest, longitudinal = np.full(len(records), np.nan), np.full(len(records), np.nan)
    fold_ids = np.full(len(records), -1, dtype=int)
    summary.update(status="ok", n_splits=len(outer))
    progress_label = (endpoint if evaluation.compute_device == "cpu"
                      else f"{endpoint} [{evaluation.compute_device}]")
    fold_progress = tqdm(enumerate(zip(outer, inner_plans)), total=len(outer),
                         desc=f"{progress_label}: outer patient folds", unit="fold",
                         dynamic_ncols=True, leave=False)
    for outer_id, ((train, test), inner) in fold_progress:
        train_scores = np.full(len(visits), np.nan)
        test_scores = np.full(len(visits), np.nan)
        base_metadata = []
        for inner_id, (inner_train, inner_test) in enumerate(inner):
            fold_progress.set_postfix_str(
                f"fold {outer_id + 1}: inner model {inner_id + 1}/{len(inner)}", refresh=True)
            if (set(groups[inner_train]) | set(groups[inner_test])) & set(groups[test]):
                raise RuntimeError("Outer-test patients leaked into inner fitting")
            indices, scores, metadata, rows = _base_scores(
                records, visits, features, endpoint, inner_train, inner_test, config,
                outer_fold=outer_id, inner_fold=inner_id, model_dir=model_dir)
            if np.any(np.isfinite(train_scores[indices])):
                raise RuntimeError("An inner held-out visit was predicted more than once")
            train_scores[indices] = scores
            base_metadata.append(metadata)
            result["visit_predictions"].extend(rows)
        if not np.all(np.isfinite(train_scores[_visit_indices(records, train)])):
            raise RuntimeError("Every outer-training visit must have an inner-held-out score")
        fold_progress.set_postfix_str(f"fold {outer_id + 1}: outer base model", refresh=True)
        indices, scores, outer_metadata, rows = _base_scores(
            records, visits, features, endpoint, train, test, config,
            outer_fold=outer_id, inner_fold=None, model_dir=model_dir)
        test_scores[indices] = scores
        result["visit_predictions"].extend(rows)
        train_records = _score_records(records, train, train_scores)
        test_records = _score_records(records, test, test_scores)
        train_x = temporal_features(train_records, evaluation.persistence_threshold,
                                    device=evaluation.compute_device)
        test_x = temporal_features(test_records, evaluation.persistence_threshold,
                                   device=evaluation.compute_device)
        fold_progress.set_postfix_str(f"fold {outer_id + 1}: latest head", refresh=True)
        latest[test], latest_head = _fit_fold(train_x[:, :1], test_x[:, :1], labels[train], evaluation)
        fold_progress.set_postfix_str(f"fold {outer_id + 1}: longitudinal head", refresh=True)
        longitudinal[test], long_head = _fit_fold(train_x, test_x, labels[train], evaluation)
        if np.any(fold_ids[test] >= 0):
            raise RuntimeError("An outer held-out eye was predicted more than once")
        fold_ids[test] = outer_id
        fold = {
            "fold": outer_id, "train_patient_ids": np.unique(groups[train]).tolist(),
            "test_patient_ids": np.unique(groups[test]).tolist(),
            "train_eye_ids": [records[i]["eye_id"] for i in train],
            "test_eye_ids": [records[i]["eye_id"] for i in test],
            "train_class_counts": _class_counts(labels[train]), "test_class_counts": _class_counts(labels[test]),
            "inner_splits": len(inner), "inner_models": base_metadata, "outer_base": outer_metadata,
            "head_training_scores": "inner patient-held-out predictions only",
            "latest": {"feature_names": ["last"], **latest_head},
            "longitudinal": {"feature_names": list(FEATURE_NAMES), **long_head},
        }
        summary["folds"].append(fold)
        if model_dir is not None:
            with (model_dir / endpoint / f"outer_{outer_id}" / "heads.json").open("x", encoding="utf-8") as handle:
                json.dump(fold, handle, indent=2, allow_nan=False)
                handle.write("\n")
        result["eye_records"].extend(dict(record, endpoint=endpoint, fold=outer_id) for record in test_records)
        result["features"].extend({
            "endpoint": endpoint, "fold": outer_id, "eye_id": record["eye_id"],
            "patient_id": record["patient_id"], "n_visits": record["n_visits"],
            "followup_years": record["followup_years"], **dict(zip(FEATURE_NAMES, values.tolist())),
        } for record, values in zip(test_records, test_x))
    if (np.any(fold_ids < 0) or not np.all(np.isfinite(latest))
            or not np.all(np.isfinite(longitudinal))):
        raise RuntimeError("Every eye requires paired finite outer-held-out progression scores")
    summary["metrics"], summary["delta_balanced_accuracy"], summary["bootstrap"] = _paired_bootstrap(
        labels, latest, longitudinal, groups, evaluation,
        progress_desc=f"{progress_label}: paired patient bootstrap")
    result["endpoints"][endpoint] = summary
    result["predictions"].extend({
        "endpoint": endpoint, "eye_id": record["eye_id"], "patient_id": record["patient_id"],
        "fold": int(fold_ids[i]), "y_true": int(labels[i]),
        "latest_probability": float(latest[i]), "longitudinal_probability": float(longitudinal[i]),
    } for i, record in enumerate(records))
    return result


def train_progression(visits, features, progression_labels, *, model_config=None,
                      evaluation_config=None, model_dir=None, inner_splits=2):
    """Fit and assess the paper's image/XGBoost architecture on GRAPE endpoints.

    Returns the longitudinal report schema plus per-visit nested predictions and
    endpoint-specific held-out eye trajectories. It never fits a full-cohort model
    or reports in-sample metrics. The caller supplies fixed, outcome-independent
    pretrained image embeddings in exactly the input visit order.
    """
    config = GlaBoostConfig() if model_config is None else model_config
    evaluation = EvaluationConfig() if evaluation_config is None else evaluation_config
    if not isinstance(config, GlaBoostConfig) or not isinstance(evaluation, EvaluationConfig):
        raise TypeError("Expected GlaBoostConfig and EvaluationConfig")
    if not config.use_image or any((config.use_text, config.use_structured,
                                   config.use_human_risk, config.use_human_confidence)):
        raise ValueError("This GRAPE progression adaptation requires image-only frozen features")
    if isinstance(inner_splits, (bool, np.bool_)) or not isinstance(inner_splits, Integral) or inner_splits < 2:
        raise ValueError("inner_splits must be an integer >= 2")
    gpu_ids = ()
    if config.tree_method == "gpu_hist":
        _, gpu_ids = resolve_image_devices(config.device)
        if not gpu_ids:
            raise ValueError("gpu_hist requires at least one selected CUDA device")
        evaluation = replace(evaluation, compute_device=f"cuda:{gpu_ids[0]}")
    elif evaluation.compute_device != "cpu":
        raise ValueError("CPU base models require compute_device='cpu'; use gpu_hist for GPU evaluation")
    visits, features, records = _inputs(visits, features, progression_labels)
    groups = np.asarray([record["patient_id"] for record in records])
    if model_dir is not None:
        model_dir = Path(model_dir)
        if model_dir.exists():
            raise FileExistsError("Choose a new model_dir; existing models are never overwritten")
        if any(model_dir.resolve().parts[i:i + 2] == ("data", "raw")
               for i in range(len(model_dir.resolve().parts) - 1)):
            raise ValueError("Model artifacts must not be written into data/raw")
        model_dir.mkdir(parents=True, exist_ok=False)
    result = {
        "config": {
            **asdict(evaluation), "inner_splits": int(inner_splits),
            "study_design": "retrospective internal validation on GRAPE",
            "visit_score_definition": "endpoint-specific, weakly supervised progression evidence; not visit-level diagnosis or calibrated clinical risk",
            "decision_threshold": 0.5, "feature_names": list(FEATURE_NAMES), "latest_features": ["last"],
            "persistence_rule": "fraction of visit scores > threshold", "logistic_penalty": "l2",
            "logistic_solver": "torch_newton" if gpu_ids else "liblinear",
            "gpu_device_ids": list(gpu_ids),
            "endpoint_compute_devices": {endpoint: (f"cuda:{gpu_ids[i % len(gpu_ids)]}"
                                                   if gpu_ids else "cpu")
                                         for i, endpoint in enumerate(ENDPOINTS)},
            "logistic_class_weight": "balanced", "logistic_max_iter": 2000,
            "logistic_tol": 1e-4, "standardization": "outer-training inner-OOF eye features only",
            "fold_method": "StratifiedKFold on unique patients stratified by any positive eye",
            "analysis": "retrospective progression assessment at last included visit",
            "auprc_definition": "average_precision", "base_model_config": asdict(config),
            "method_specs": {
                "latest": "L2 logistic regression using the last endpoint-specific visit score",
                "longitudinal": "L2 logistic regression using last, delta, OLS slope/year, mean, persistence",
                "persistence": f"fraction of visit scores strictly > {evaluation.persistence_threshold}",
                "preprocessing": "Frozen outcome-independent image features; head StandardScaler fitted on outer-training inner-OOF scores only",
                "classification": "predicted progression score >= 0.5",
                "probability_interpretation": "weakly supervised visit scores and class-weighted logistic outputs; not clinically calibrated risk",
                "fold_selection": "fixed-seed patient stratification; reduce count for class feasibility only; all inner plans validated before fitting",
                "cross_fitting": "Inner patient-held-out base scores train both heads; fresh outer-training base model scores untouched outer-test patients",
                "base_sample_weight": "inverse eligible visit count per eye, normalized to mean 1; no outcome class weighting",
                "target_adaptation": "Separate whole-eye progression labels repeated across visits; no known per-visit progression state",
            },
        }, "endpoints": {}, "predictions": [], "features": [], "visit_predictions": [], "eye_records": [],
    }
    completed = {}
    if gpu_ids:
        # One lane owns a CUDA device for its entire lifetime. An endpoint runs
        # all base fits, heads, and bootstrap on that device before the next
        # endpoint starts, so two endpoint jobs never overlap on one GPU.
        import torch

        # The process-wide lazy linalg loader must finish before any worker can
        # reach its first logistic Newton solve; CUDA contexts alone do not
        # initialize this backend. Keep subsequent endpoint work concurrent.
        tqdm.write("Initializing CUDA linear algebra before parallel endpoint training")
        initialize_cuda_linalg(gpu_ids)

        def run_lane(gpu_id, endpoints):
            lane_config = replace(config, gpu_id=gpu_id, device=f"cuda:{gpu_id}")
            lane_evaluation = replace(evaluation, compute_device=f"cuda:{gpu_id}")
            lane_results = {}
            with torch.cuda.device(gpu_id):
                for endpoint in endpoints:
                    lane_results[endpoint] = _train_endpoint(
                        endpoint, records, visits, features, groups, lane_config, lane_evaluation,
                        model_dir=model_dir, inner_splits=inner_splits)
                    with endpoint_progress.get_lock():
                        endpoint_progress.update(1)
            return lane_results

        features.setflags(write=False)
        with tqdm(total=len(ENDPOINTS), desc="Progression endpoints", unit="endpoint",
                  dynamic_ncols=True) as endpoint_progress:
            with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
                futures = [executor.submit(run_lane, gpu_id, ENDPOINTS[lane::len(gpu_ids)])
                           for lane, gpu_id in enumerate(gpu_ids) if lane < len(ENDPOINTS)]
                for future in as_completed(futures):
                    lane_results = future.result()
                    completed.update(lane_results)
    else:
        endpoint_progress = tqdm(ENDPOINTS, desc="Progression endpoints", unit="endpoint", dynamic_ncols=True)
        for endpoint in endpoint_progress:
            endpoint_progress.set_postfix_str(endpoint, refresh=True)
            completed[endpoint] = _train_endpoint(
                endpoint, records, visits, features, groups, config, evaluation,
                model_dir=model_dir, inner_splits=inner_splits)
            if completed[endpoint]["endpoints"][endpoint]["status"] != "ok":
                endpoint_progress.set_postfix_str(f"{endpoint}: not estimable", refresh=True)

    # Merge only in the prespecified endpoint order, independent of GPU timing.
    for endpoint in ENDPOINTS:
        partial = completed[endpoint]
        result["endpoints"][endpoint] = partial["endpoints"][endpoint]
        for collection in ("predictions", "features", "visit_predictions", "eye_records"):
            result[collection].extend(partial[collection])
    return result
