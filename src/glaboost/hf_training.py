"""Train source-only diagnostic detectors, freeze them, then assess GRAPE.

Retained rows preserve source split membership after fixed duplicate exclusions.
Neither source test outcomes nor
GRAPE outcomes select the prespecified detector settings. Frozen image features
are computed once and shared by the tree fits; no neural network is fine-tuned.
"""

import csv
import gc
import hashlib
import html
import json
import multiprocessing
import re
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from .config import GlaBoostConfig
from .data import VisitInput
from .encoders import ResNet152Encoder, resolve_image_devices
from .external import _snapshot_code, run_external_validation
from .hf_data import load_hf_diagnosis
from .longitudinal import EvaluationConfig, _metrics
from .model import GlaBoost
from .reporting import embed_report_links
from .study import (_environment_info, _json_write, _read_json, _refresh_result_index,
                    ensure_outside_raw, sha256_file)


_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")
_LIMITATIONS = (
    "This is an image-only GlaBoost-style reconstruction, not the authors' fitted model "
    "or a reproduction of their reported accuracy. Source labels are mapped from 0=glaucoma, "
    "1=normal to the model's 1=glaucoma, 0=normal. Only source training images and diagnosis "
    "labels fit the detector. Before fitting, same-label identical RGB images within each split "
    "are reduced to the first released row; test images matching training pixels are excluded "
    "from testing while training membership is retained. Conflicting duplicate labels or any "
    "source/GRAPE pixel overlap stop the run. Source test data are used only for fixed-threshold diagnosis "
    "metrics; there is no early stopping, setting selection, calibration, or refitting on test data. "
    "Source patient IDs are unavailable, so subject separation within the released splits "
    "and across datasets cannot be verified. The duplicate audit detects identical decoded "
    "RGB images, not near duplicates or different images of the same person. "
    "GRAPE images and outcomes never train or select these diagnostic detectors. Their "
    "fixed outputs are subsequently used by progression mappings trained and patient-cross-validated "
    "within GRAPE. This does not externally validate a frozen complete progression pipeline."
)


def _load_training_plan(path):
    path = Path(path).expanduser().resolve()
    plan = _read_json(path)
    allowed = {"format_version", "source", "primary_model", "models", "min_visits", "evaluation"}
    if set(plan) - allowed or plan.get("format_version") != 1:
        raise ValueError("Unknown training-plan fields or unsupported format_version.")
    source = plan.get("source", {})
    if (not isinstance(source, dict) or set(source) != {"repo_id", "revision"}
            or not isinstance(source["repo_id"], str) or not source["repo_id"].strip()
            or not isinstance(source["revision"], str)
            or not re.fullmatch(r"[0-9a-f]{40}", source["revision"])):
        raise ValueError("Pin the source repo_id and exact 40-character revision in the training plan.")
    models, names = [], set()
    if not isinstance(plan.get("models"), list) or not plan["models"]:
        raise ValueError("Prespecify at least one diagnostic model.")
    parameters = {"n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "random_state"}
    for entry in plan["models"]:
        if (not isinstance(entry, dict) or set(entry) - (parameters | {"name"})
                or not isinstance(entry.get("name"), str) or not _NAME.fullmatch(entry["name"])
                or entry["name"] in names):
            raise ValueError("Each source configuration needs a unique safe name and only tree parameters.")
        names.add(entry["name"])
        config = GlaBoostConfig.for_image_method(**{k: v for k, v in entry.items() if k != "name"})
        models.append({"name": entry["name"], **{k: getattr(config, k) for k in sorted(parameters)}})
    if plan.get("primary_model") not in names:
        raise ValueError("primary_model must name a prespecified source configuration.")
    minimum = plan.get("min_visits", 3)
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 3:
        raise ValueError("min_visits must be an integer >= 3.")
    evaluation = plan.get("evaluation", {})
    allowed_evaluation = {"n_splits", "seed", "bootstrap_replicates", "persistence_threshold", "logistic_c"}
    if not isinstance(evaluation, dict) or set(evaluation) - allowed_evaluation:
        raise ValueError("Invalid prespecified progression evaluation settings.")
    evaluated = asdict(EvaluationConfig(**evaluation))
    evaluated.pop("compute_device")
    return path, {**plan, "models": models, "min_visits": minimum, "evaluation": evaluated}


def _protect_source(path, hf_root, grape_root):
    """The HF archive is raw data even though it lives outside data/raw."""
    path = ensure_outside_raw(path, grape_root)
    if path == hf_root or hf_root in path.parents or path in hf_root.parents:
        raise ValueError("Outputs and cache must not overlap the retained raw HF archive.")
    return path


class _FeatureTableEncoder:
    """Internal exact row lookup into this run's frozen image feature table.

    Integer indices are never predictors. Saved bundles retain the real image
    encoder identity, so ordinary GlaBoost.load restores image-based inference.
    """

    output_dim = 2048

    def __init__(self, features, spec):
        self.features = features
        self._spec = deepcopy(spec)
        if (features.ndim != 2 or features.shape[1] != self.output_dim
                or not np.isfinite(features).all() or spec.get("output_dim") != self.output_dim
                or spec.get("encoder") != "resnet152" or spec.get("frozen") is not True):
            raise ValueError("Expected verified, finite frozen ResNet152 features.")

    def transform(self, inputs):
        rows = np.asarray(inputs)
        if (rows.ndim != 1 or rows.dtype.kind not in "iu" or np.any(rows < 0)
                or np.any(rows >= len(self.features))):
            raise ValueError("Invalid frozen-feature row indices.")
        return self.features[rows]

    def spec(self):
        return deepcopy(self._spec)


def _fit_device_group(payload):
    """Spawn-safe worker: one process owns one GPU and fits its assigned trees."""
    features = np.load(payload["feature_path"], mmap_mode="r", allow_pickle=False)
    if sha256_file(payload["feature_path"]) != payload["feature_sha256"]:
        raise ValueError("Frozen source features changed after extraction.")
    encoder = _FeatureTableEncoder(features, payload["encoder_spec"])
    split = len(payload["train_ids"])
    train = [VisitInput(sample_id=name, image=i) for i, name in enumerate(payload["train_ids"])]
    test = [VisitInput(sample_id=name, image=i + split) for i, name in enumerate(payload["test_ids"])]
    if len(features) != len(train) + len(test):
        raise ValueError("Frozen feature rows differ from the audited source split.")
    results = []
    for item in payload["models"]:
        config = GlaBoostConfig.from_dict(item["config"])
        model = GlaBoost(config, image_encoder=encoder).fit(train, payload["train_labels"])
        model.training_summary_.update({
            "source_repo_id": payload["source"]["repo_id"], "source_revision": payload["source"]["revision"],
            "source_split": "train", "source_audit_sha256": payload["audit_sha256"],
            "feature_table_sha256": payload["feature_sha256"],
            "source_label_mapping": {"0": 1, "1": 0},
            "test_used_for_selection": False, "grape_used_for_training_or_selection": False,
        })
        # Freeze/save before test scoring. No test label is passed to this worker.
        destination = Path(payload["artifact_path"]) / item["name"]
        model.save(destination)
        probabilities = np.asarray(model.predict_score(test), dtype=float)
        if (probabilities.shape != (len(test),) or not np.isfinite(probabilities).all()
                or np.any((probabilities < 0) | (probabilities > 1))):
            raise ValueError("Invalid held-out source diagnosis probabilities.")
        results.append({"name": item["name"], "model_directory": str(destination),
                        "probabilities": probabilities.tolist()})
    return results


def _diagnostic_metrics(labels, probabilities, device):
    if device == "cpu":
        return _metrics(np.asarray(labels), np.asarray(probabilities))
    import torch
    from .gpu_metrics import _weighted_metrics_tensor
    y = torch.as_tensor(labels, dtype=torch.float64, device=device)
    p = torch.as_tensor(probabilities, dtype=torch.float64, device=device)
    with torch.no_grad():
        metrics = _weighted_metrics_tensor(y, p, torch.ones((1, len(labels)), device=device))
    return {key: float(value[0].item()) for key, value in metrics.items()}


def _write_source_report(report, plan, audit, results, test_ids, test_labels, device, synthetic):
    rows = []
    for model in plan["models"]:
        result = results[model["name"]]
        probabilities = result["probabilities"]
        metrics = _diagnostic_metrics(test_labels, probabilities, device)
        rows.append({"model": model["name"], "role": "Primary" if model["name"] == plan["primary_model"] else "Exploratory",
                     **{k: model[k] for k in ("n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree")},
                     "n_test": len(test_labels), "diagnosis_threshold": .5, **metrics, "synthetic": synthetic})
        with (report / (model["name"] + "_test_predictions.csv")).open("x", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample_id", "diagnosis_label_1_glaucoma", "glaucoma_score", "synthetic"])
            writer.writerows((sample, int(label), score, synthetic)
                             for sample, label, score in zip(test_ids, test_labels, probabilities))
    with (report / "diagnosis_metrics.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    title = ("SYNTHETIC SOFTWARE TEST — " if synthetic else "") + "Source diagnosis training and held-out assessment"
    keys = ("model", "role", "n_estimators", "max_depth", "balanced_accuracy", "auroc", "auprc", "sensitivity", "specificity", "f1")
    headings = ("Model", "Role", "Trees", "Depth", "BA", "AUROC", "AUPRC", "Sensitivity", "Specificity", "F1")
    cells = [[f"{row[k]:.3f}" if k in keys[4:] else str(row[k]) for k in keys] for row in rows]
    opening = (f"Source: {audit['repo_id']} at {audit['revision']}. Prespecified primary: {plan['primary_model']}. "
               "Metrics below concern glaucoma diagnosis in the released source test split, not GRAPE progression. "
               "All metrics use a 0–1 scale; the fixed diagnosis threshold is 0.5. No source patient-level confidence intervals "
               "are claimed because patient IDs are unavailable. GRAPE progression reports provide paired patient bootstrap intervals.")
    accounting = []
    for split in ("train", "test"):
        counts = audit["splits"][split]
        accounting.append(f"{split}: {counts['released_rows']} released, {counts['retained_rows']} retained, "
                          f"{counts['within_split_duplicates_removed']} within-split duplicates removed, "
                          f"{counts['train_overlap_duplicates_removed']} training-overlap test images excluded")
    accounting = "Source cohort after the prespecified duplicate audit — " + "; ".join(accounting) + "."
    links = ("[Source and duplicate audit](source_audit.json) · [Training plan](training_plan.json) · "
             "[Fixed-model plan](external_models.json) · [Numeric diagnosis metrics](diagnosis_metrics.csv) · "
             f"[GRAPE progression comparison](../{plan['run_name']}/report.html)")
    markdown = [f"# {title}", "", opening, "", accounting, "", _LIMITATIONS, "", "| " + " | ".join(headings) + " |",
                "|" + "---|" * len(headings), *["| " + " | ".join(row) + " |" for row in cells], "", links, ""]
    (report / "report.md").write_text("\n".join(markdown), encoding="utf-8")
    esc = html.escape
    table = "<table><tr>" + "".join(f"<th>{esc(c)}</th>" for c in headings) + "</tr>"
    table += "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in row) + "</tr>" for row in cells) + "</table>"
    page = (f"<!doctype html><html lang='en'><meta charset='utf-8'><title>{esc(title)}</title>"
            "<style>body{font:16px/1.5 system-ui;max-width:1200px;margin:32px auto;padding:0 20px}"
            "table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccc}th{background:#edf2f7}</style>"
            f"<body><h1>{esc(title)}</h1><p>{esc(opening)}</p><p>{esc(accounting)}</p><p>{esc(_LIMITATIONS)}</p>{table}"
            "<p><a href='source_audit.json'>Source and duplicate audit</a> · "
            "<a href='training_plan.json'>Training plan</a> · <a href='external_models.json'>Fixed-model plan</a> · "
            "<a href='diagnosis_metrics.csv'>Numeric diagnosis metrics</a> · "
            f"<a href='../{esc(plan['run_name'])}/report.html'>GRAPE progression comparison</a></p></body></html>\n")
    (report / "report.html").write_text(page, encoding="utf-8")


def run_hf_grape(*, training_plan_path, run_name,
                 hf_root="/scratch/users/zeyuhan/DA-SPL/archive/glaucoma_diagnosis_json_analysis",
                 grape_root="data/raw/grape", result_dir="result", artifact_dir="artifacts",
                 device="cuda", cache_dir=".cache/glaboost", image_weights=None,
                 image_batch_size=None, allow_download=False, synthetic=False):
    """Complete source training then fixed-detector GRAPE assessment; never overwrite."""
    if (not isinstance(run_name, str) or not _NAME.fullmatch(run_name) or len(run_name) > 73
            or not isinstance(synthetic, bool)):
        raise ValueError("run_name must be 1–73 safe characters; synthetic must be boolean.")
    submitted_path, plan = _load_training_plan(training_plan_path)
    source_root, grape = (Path(p).expanduser().resolve() for p in (hf_root, grape_root))
    output, artifacts, cache = (_protect_source(p, source_root, grape)
                                for p in (result_dir, artifact_dir, cache_dir))
    for left, right in ((output, artifacts), (output, cache), (artifacts, cache)):
        if left == right or left in right.parents or right in left.parents:
            raise ValueError("Report, artifact and cache roots must be separate, non-overlapping directories.")
    if (output / "INDEX.md").is_symlink():
        raise ValueError("The report index must not be a symbolic link.")
    for root in (output, artifacts):
        for name in (run_name, run_name + "_source"):
            if (root / name).exists() or (root / name).is_symlink():
                raise FileExistsError(f"Run already exists; choose a new name: {root / name}")
    checkpoint = (Path(image_weights).expanduser().resolve() if image_weights is not None else
                  cache / "torch" / ResNet152Encoder.checkpoint_filename)
    _protect_source(cache / "torch" / ResNet152Encoder.checkpoint_filename, source_root, grape)
    if not checkpoint.is_file() and (image_weights is not None or not allow_download):
        raise FileNotFoundError(f"Frozen ResNet152 checkpoint is missing: {checkpoint}")
    if (checkpoint.is_file() and not synthetic
            and not sha256_file(checkpoint).startswith(ResNet152Encoder.checkpoint_sha256_prefix)):
        raise ValueError("Source training requires the official ResNet152 ImageNet V1 checkpoint SHA256 prefix; "
                         "an unverified encoder cannot support this independent-source protocol.")
    print("[1/4] Verify retained source data, labels and exact image overlap (read-only)", flush=True)
    dataset = load_hf_diagnosis(source_root, grape_root=grape)
    if any(dataset.audit.get(key) != value for key, value in plan["source"].items()):
        raise ValueError("HF dataset identity/revision differs from the prespecified source plan.")
    resolved, gpu_ids = resolve_image_devices(device)
    if image_batch_size is None:
        image_batch_size = 64 * len(gpu_ids) if gpu_ids else 16
    if isinstance(image_batch_size, bool) or not isinstance(image_batch_size, int) or image_batch_size < 1:
        raise ValueError("image_batch_size must be a positive integer.")
    source_report, source_artifacts = output / (run_name + "_source"), artifacts / (run_name + "_source")
    source_artifacts.mkdir(parents=True, exist_ok=False)
    source_report.mkdir(parents=True, exist_ok=False)
    created = datetime.now(timezone.utc).isoformat()
    status = {"status": "running", "stage": "source_feature_extraction", "created_at": created,
              "run_name": run_name + "_source", "synthetic": synthetic,
              "validation_design": "independent_source_diagnosis_training", "completed_models": []}

    def save_status():
        for directory in (source_report, source_artifacts):
            (directory / "status.json").write_text(json.dumps(status, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    try:
        plan.update(run_name=run_name, synthetic=synthetic, created_at=created,
                    submitted_plan_sha256=sha256_file(submitted_path),
                    runtime={"requested_device": device, "resolved_device": resolved, "gpu_ids": list(gpu_ids),
                             "image_batch_size": image_batch_size, "feature_extraction": "once for all source configurations",
                             "tree_workers": min(len(plan["models"]), max(1, len(gpu_ids)))},
                    interpretation=_LIMITATIONS)
        audit = dataset.audit
        for directory in (source_report, source_artifacts):
            _json_write(directory / "training_plan.json", plan)
            _json_write(directory / "source_audit.json", audit)
        # Retain a directly runnable submitted plan, not only the enriched audit.
        submitted_bytes = submitted_path.read_bytes()
        if hashlib.sha256(submitted_bytes).hexdigest() != plan["submitted_plan_sha256"]:
            raise ValueError("Submitted training plan changed during preflight.")
        with (source_report / "submitted_training_plan.json").open("xb") as handle:
            handle.write(submitted_bytes)
        _json_write(source_report / "environment.json", _environment_info())
        _snapshot_code(source_report / "code")
        save_status()
        print("[2/4] Frozen ResNet152 source features (shared across all tree configurations)", flush=True)
        encoder = ResNet152Encoder(weights_path=image_weights, cache_dir=str(cache), device=device,
                                   batch_size=image_batch_size, allow_download=allow_download)
        train_ids, test_ids = ([v.sample_id for v in rows] for rows in (dataset.train_visits, dataset.test_visits))
        train_labels, test_labels = np.array(dataset.train_labels), np.array(dataset.test_labels)
        visits = dataset.train_visits + dataset.test_visits
        blocks = []
        with tqdm(total=len(visits), desc="ResNet152 source CFP features", unit="image", dynamic_ncols=True) as progress:
            for start in range(0, len(visits), image_batch_size):
                batch = visits[start:start + image_batch_size]
                blocks.append(encoder.transform([v.image for v in batch]))
                progress.update(len(batch))
        features, spec = np.concatenate(blocks), encoder.spec()
        _FeatureTableEncoder(features, spec)  # Validate before any tree fit.
        if not synthetic and not spec["source"]["sha256"].startswith(ResNet152Encoder.checkpoint_sha256_prefix):
            raise ValueError("Frozen source encoder does not match the required official ImageNet checkpoint.")
        feature_path = source_artifacts / "resnet152_features.npy"
        with feature_path.open("xb") as handle:
            np.save(handle, features, allow_pickle=False)
        feature_hash = sha256_file(feature_path)
        _json_write(source_artifacts / "feature_index.json", {"train_ids": train_ids, "test_ids": test_ids,
                    "train_labels": train_labels.tolist(), "test_labels": test_labels.tolist(),
                    "feature_sha256": feature_hash, "encoder_spec": spec})
        del encoder, dataset, visits, blocks, features, batch
        gc.collect()
        if gpu_ids:
            import torch
            torch.cuda.empty_cache()
        print("[3/4] Source-only XGBoost training and held-out diagnostic assessment", flush=True)
        status["stage"] = "source_tree_training"
        save_status()
        workers = min(len(plan["models"]), max(1, len(gpu_ids)))
        groups = [[] for _ in range(workers)]
        for i, item in enumerate(plan["models"]):
            gpu = gpu_ids[i % workers] if gpu_ids else None
            config = GlaBoostConfig.for_image_method(
                **{k: v for k, v in item.items() if k != "name"}, device=f"cuda:{gpu}" if gpu is not None else "cpu",
                gpu_id=gpu, tree_method="gpu_hist" if gpu is not None else "hist", cache_dir=str(cache),
                image_weights_path=str(checkpoint), image_batch_size=image_batch_size)
            groups[i % workers].append({"name": item["name"], "config": config.to_dict()})
        common = {"feature_path": str(feature_path), "feature_sha256": feature_hash, "encoder_spec": spec,
                  "train_ids": train_ids, "test_ids": test_ids, "train_labels": train_labels.tolist(),
                  "source": plan["source"], "audit_sha256": sha256_file(source_report / "source_audit.json"),
                  "artifact_path": str(source_artifacts)}
        results = {}
        with tqdm(total=len(plan["models"]), desc="Source diagnosis configurations", unit="model", dynamic_ncols=True) as progress:
            def collect(items):
                for item in items:
                    results[item["name"]] = item
                    status["completed_models"].append(item["name"])
                save_status()
                progress.update(len(items))
            if workers == 1:
                for item in groups[0]:
                    collect(_fit_device_group({**common, "models": [item]}))
            else:
                # Spawn avoids inheriting CUDA contexts and first-use library state.
                with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                    futures = [pool.submit(_fit_device_group, {**common, "models": group}) for group in groups]
                    for future in as_completed(futures):
                        collect(future.result())
        training_source = {
            "description": f"Image-only diagnosis training on the official train split of {audit['repo_id']}",
            "reference": f"HF revision {audit['revision']}; source audit SHA256 {common['audit_sha256']}",
            "grape_overlap": "none",
            "independence_evidence": (
                "Detector fitting uses only verified source-train Parquet image/diagnosis rows. "
                "No source test or GRAPE outcomes select settings; no GRAPE images enter fitting. "
                "After prespecified exclusion of test images matching training pixels, the retained source "
                "splits have no identical decoded RGB images; no source/GRAPE CFP pixel match was found. "
                "Patient overlap and near duplicates cannot be excluded because source patient IDs are unavailable. "
                f"Audit: {source_report / 'source_audit.json'}; SHA256 {common['audit_sha256']}"),
        }
        fixed_plan = {"format_version": 1, "primary_model": plan["primary_model"], "min_visits": plan["min_visits"],
                      "evaluation": plan["evaluation"],
                      "models": [{"name": item["name"], "model_directory": results[item["name"]]["model_directory"],
                                  "training_data": training_source} for item in plan["models"]]}
        for directory in (source_report, source_artifacts):
            _json_write(directory / "external_models.json", fixed_plan)
        _write_source_report(source_report, plan, audit, results, test_ids, test_labels, resolved, synthetic)
        status.update(stage="fixed_grape_validation", source_training_complete=True)
        save_status()
        print("[4/4] Fixed detectors on GRAPE; patient-separated longitudinal A/B reports", flush=True)
        report = run_external_validation(plan_path=source_report / "external_models.json", run_name=run_name,
                    grape_root=grape, result_dir=output, artifact_dir=artifacts, device=device, cache_dir=cache,
                    image_weights=str(checkpoint), image_batch_size=image_batch_size,
                    allow_download=allow_download, synthetic=synthetic)
        # Include source evidence in the portable overview while keeping
        # diagnostic and progression metrics in separate sections.
        source_link = f"../{source_report.name}/report.html"
        page_path = report / "report.html"
        page = page_path.read_text(encoding="utf-8")
        insertion = '<section class="report-attachment"' if '<section class="report-attachment"' in page else "</body>"
        page_path.write_text(page.replace(insertion,
            f"<p><a href='{html.escape(source_link, quote=True)}'>Source diagnosis training, "
            "held-out metrics and duplicate exclusions</a></p>" + insertion, 1), encoding="utf-8")
        embed_report_links(page_path)
        with (report / "report.md").open("a", encoding="utf-8") as handle:
            handle.write(f"\n[Source diagnosis training, held-out metrics and duplicate exclusions]({source_link})\n")
        status.update(status="complete", stage="complete", grape_report=str(report))
        save_status()
        _refresh_result_index(output)
        return report
    except Exception as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        try:
            (source_artifacts / "error_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        except Exception as diagnostic_error:
            status["traceback_write_error"] = str(diagnostic_error)
        try:
            save_status()
        except Exception:
            pass
        raise
