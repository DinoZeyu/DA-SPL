"""Apply prespecified independent, fixed visit detectors to the GRAPE cohort.

Detector training is deliberately absent. A supplied plan declares the detector
sources and one primary configuration before any GRAPE progression evaluation.
Only the two progression mappings are fitted within patient-separated GRAPE folds.
"""

import hashlib
import html
import json
import re
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from .config import GlaBoostConfig
from .data import load_grape
from .encoders import image_encoder_class, resolve_image_devices
from .longitudinal import ENDPOINTS, EvaluationConfig
from .model import GlaBoost
from .study import (
    PROJECT_ROOT, _json_write, _read_json, _refresh_project_readme, _refresh_result_index,
    create_study_report, ensure_outside_raw, score_metadata_path, sha256_file, write_visit_scores,
)


_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")
_INTERPRETATION = (
    "Each fixed glaucoma diagnosis detector is applied independently to every eligible visit. "
    "Its independence from GRAPE is documented by the supplied declaration and evidence; "
    "this software cannot independently establish the truth of that declaration. "
    "The progression mappings are fitted and patient-cross-validated within GRAPE. "
    "This is external application of the visit detector with internally validated progression mappings, "
    "not independent external validation of a frozen complete progression pipeline or future forecasting. "
    "A uses the latest visit score; B uses last, change, slope/year, mean and persistence. "
    "The primary detector is specified in the plan. Additional configurations are exploratory and "
    "are not selected or ranked by these GRAPE results. Paired patient bootstrap intervals condition "
    "on fixed out-of-fold predictions, exclude uncertainty from refitting, and are not adjusted "
    "for multiple endpoints/configurations. Within-configuration intervals do not compare detectors."
)


def _declaration(value, name):
    if (not isinstance(value, str) or len(value.strip()) < (5 if name == "reference" else 12)
            or value.strip().lower() in {"not documented", "not available", "to be supplied", "unknown"}
            or value.strip().lower().startswith(("todo", "tbd", "<", "placeholder"))):
        raise ValueError(f"Document substantive detector training_data.{name}; placeholders cannot support validation.")
    return value.strip()


def _load_plan(path):
    path = Path(path).expanduser().resolve()
    plan = _read_json(path)
    if plan.get("format_version") != 1:
        raise ValueError("The external-validation plan requires format_version: 1.")
    if not isinstance(plan.get("models"), list) or not plan["models"]:
        raise ValueError("The plan has no fixed diagnostic models. Supply independent trained model artifacts and provenance first.")
    names, models = set(), []
    for item in plan["models"]:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not _NAME.fullmatch(item["name"]):
            raise ValueError("Every model needs a safe, unique name of 1–80 characters.")
        if item["name"] in names:
            raise ValueError("Model names must be unique.")
        names.add(item["name"])
        source = item.get("training_data")
        if not isinstance(source, dict) or source.get("grape_overlap") != "none":
            raise ValueError("Every fixed detector requires an explicit declaration of no GRAPE training/selection overlap.")
        source = {key: _declaration(source.get(key), key)
                  for key in ("description", "reference", "independence_evidence")}
        source["grape_overlap"] = "none"
        directory = item.get("model_directory")
        if not isinstance(directory, str) or not directory.strip():
            raise ValueError("Every planned model requires model_directory.")
        directory = Path(directory).expanduser()
        if not directory.is_absolute():
            directory = path.parent / directory
        models.append({"name": item["name"], "model_directory": str(directory.resolve()), "training_data": source})
    if plan.get("primary_model") not in names:
        raise ValueError("primary_model must explicitly name one of the supplied fixed models.")
    minimum = plan.get("min_visits", 3)
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 3:
        raise ValueError("min_visits must be an integer >= 3.")
    evaluation = plan.get("evaluation", {})
    allowed = {"n_splits", "seed", "bootstrap_replicates", "persistence_threshold", "logistic_c"}
    if not isinstance(evaluation, dict) or set(evaluation) - allowed:
        raise ValueError("The plan evaluation must contain only folds, seed, bootstrap and prespecified mapping settings.")
    # Check all numerical choices before inspecting devices or creating outputs.
    defaults = asdict(EvaluationConfig(**evaluation))
    defaults.pop("compute_device")
    return path, {"format_version": 1, "primary_model": plan["primary_model"],
                  "models": models, "min_visits": minimum, "evaluation": defaults}


def _preflight_model(entry, cache_dir, image_weights, allow_download):
    from xgboost import Booster

    directory = Path(entry["model_directory"])
    metadata_path, model_path = directory / "metadata.json", directory / "model.json"
    metadata = _read_json(metadata_path)
    if (metadata.get("format_version") != 1
            or metadata.get("target") != {"0": "normal", "1": "glaucoma"}):
        raise ValueError(f"{entry['name']}: requires a saved normal/glaucoma diagnostic detector, not a GRAPE progression model.")
    config = GlaBoostConfig.from_dict(metadata.get("config", {}))
    if (not config.use_image or config.image_encoder != "resnet152"
            or any((config.use_text, config.use_structured, config.use_human_risk, config.use_human_confidence))):
        raise ValueError(f"{entry['name']}: this protocol requires an image-only ResNet152 detector.")
    if metadata.get("structured_state") is not None or metadata.get("human_state") is not None:
        raise ValueError("Image-only detectors cannot contain fitted structured or human-input preprocessing.")
    specs = metadata.get("encoder_specs", {})
    spec = specs.get("image", {})
    digest = (spec.get("source") or {}).get("sha256")
    if (set(specs) != {"image"} or spec.get("encoder") != "resnet152" or spec.get("frozen") is not True
            or spec.get("output_dim") != 2048 or not isinstance(spec.get("preprocessing"), dict)
            or not isinstance(digest, str) or re.fullmatch(r"[a-f0-9]{64}", digest) is None
            or spec.get("fingerprint") != "sha256:" + digest):
        raise ValueError(f"{entry['name']}: missing verified frozen ResNet152 encoder specification/checksum.")
    if metadata.get("feature_names") != [f"image_{i}" for i in range(2048)]:
        raise ValueError("Saved diagnostic feature schema must contain exactly 2048 ordered image features.")
    if metadata.get("training_summary", {}).get("n_features") != 2048:
        raise ValueError("Saved diagnostic training summary has an incompatible feature dimension.")
    if sha256_file(model_path) != metadata.get("model_sha256"):
        raise ValueError("Diagnostic model checksum does not match its metadata.")
    # Loading native JSON validates the classifier format; it performs no fitting
    # or prediction and does not instantiate an image encoder.
    booster = Booster(params={"nthread": 1}, model_file=str(model_path))
    learner = json.loads(booster.save_config())["learner"]
    if (booster.num_features() != 2048 or learner["objective"]["name"] != "binary:logistic"
            or booster.num_boosted_rounds() != config.n_estimators):
        raise ValueError("Native diagnostic classifier target, feature count or tree count disagrees with metadata.")
    saved = booster.attr("scikit_learn")
    if saved:
        parameters = json.loads(saved)
        for field in ("n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree"):
            if parameters.get(field) != getattr(config, field):
                raise ValueError(f"Saved diagnostic classifier {field} disagrees with metadata.")
    del booster
    explicit_weights = image_weights if image_weights is not None else config.image_weights_path
    checkpoint = (Path(explicit_weights).expanduser().resolve() if explicit_weights is not None else
                  cache_dir / "torch" / image_encoder_class("resnet152").checkpoint_filename)
    if checkpoint.is_file():
        if sha256_file(checkpoint) != digest:
            raise ValueError(f"{entry['name']}: available image checkpoint differs from the saved detector encoder.")
    elif explicit_weights is not None or not allow_download or not digest.startswith(image_encoder_class("resnet152").checkpoint_sha256_prefix):
        raise FileNotFoundError(f"Matching frozen image checkpoint is missing: {checkpoint}")
    return {**entry, "model_sha256": metadata["model_sha256"],
            "model_metadata_sha256": sha256_file(metadata_path), "metadata": metadata}


def _snapshot_code(directory):
    manifest = {}
    files = list((PROJECT_ROOT / "src" / "glaboost").glob("*.py"))
    files += [PROJECT_ROOT / name for name in ("run_grape.sh", "pyproject.toml", "uv.lock")]
    files += list((PROJECT_ROOT / "configs").glob("*.json"))
    for source in files:
        if not source.is_file():
            continue
        relative = source.relative_to(PROJECT_ROOT)
        destination = directory / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = source.read_bytes()
        with destination.open("xb") as handle:
            handle.write(data)
        manifest[str(relative)] = hashlib.sha256(data).hexdigest()
    _json_write(directory / "manifest.json", {"sha256": manifest,
                                             "note": "Exact source and environment files captured before evaluation; no model weights or raw data."})


def _analysis_signature(evaluation, cohort, source, records):
    folds = {}
    for endpoint in ENDPOINTS:
        summary = evaluation["endpoints"][endpoint]
        if summary["status"] not in ("ok", "not_estimable"):
            raise ValueError("Unexpected progression endpoint status.")
        item = {key: value for key, value in summary.items()
                if key not in ("folds", "metrics", "bootstrap", "delta_balanced_accuracy")}
        item["folds"] = []
        for fold in summary["folds"]:
            if set(fold["train_patient_ids"]) & set(fold["test_patient_ids"]):
                raise ValueError("Patient overlap in saved progression folds.")
            fixed = {key: value for key, value in fold.items() if key not in ("latest", "longitudinal")}
            fixed["head_features"] = {name: fold[name]["feature_names"] for name in ("latest", "longitudinal")}
            item["folds"].append(fixed)
        folds[endpoint] = item
    return {"evaluation_config": evaluation["config"], "cohort": cohort, "source": source,
            "ordered_visits": records, "folds": folds,
            "held_out_labels": [{key: row[key] for key in ("endpoint", "patient_id", "eye_id", "fold", "y_true")}
                                for row in evaluation["predictions"]]}


def _metric(row, prefix, signed=False):
    estimate, low, high = (row.get(prefix + key) for key in ("estimate", "ci_low", "ci_high"))
    if estimate is None:
        return "Not estimable"
    fmt = "+.1f" if signed else ".1f"
    value = format(100 * estimate, fmt)
    if low is None or high is None:
        return value + " [CI not estimable]"
    return f"{value} [{format(100 * low, fmt)}, {format(100 * high, fmt)}]"


def _write_summary(report, plan, rows, supplementary, cohort):
    from .reporting import _write_csv

    _write_csv(report / "comparison.csv", rows, ("model", "role", "endpoint"))
    _write_csv(report / "supplementary_metrics.csv", supplementary, ("model", "role", "endpoint", "method", "metric"))
    title = ("SYNTHETIC SOFTWARE TEST — " if plan["synthetic"] else "") + "GRAPE fixed-detector longitudinal assessment"
    opening = (f"Prespecified primary detector: {plan['primary_model']}. "
               f"Cohort: {cohort['n_patients']} patients, {cohort['n_eyes']} eyes, {cohort['n_visits']} eligible CFP visits. "
               "BA is balanced accuracy. BA/CI entries are percentages; B − A entries are percentage points. "
               "CSV values use the 0–1 scale. Every configured model and all three outcomes are retained.")
    headings = ("Detector", "Role", "Outcome", "Positive / total eyes", "A: latest BA [95% CI]",
                "B: longitudinal BA [95% CI]", "B − A [95% CI]", "Status")
    labels = {"plr2": "PLR2", "plr3": "PLR3", "md_slope": "MD slope"}
    cells = [[row["model"], row["role"], labels[row["endpoint"]],
              f"{row['n_positive_eyes']} / {row['n_eyes']}", _metric(row, "latest_"),
              _metric(row, "longitudinal_"), _metric(row, "delta_", True),
              row["status"] + (": " + row["reason"] if row.get("reason") else "")] for row in rows]
    links = [(entry["name"], f"{entry['name']}/report.html") for entry in plan["models"]]
    setting_headings = ("Detector", "Role", "Trees", "Maximum depth", "Learning rate", "Subsample", "Column sample")
    setting_cells = [[entry["name"], "Primary" if entry["name"] == plan["primary_model"] else "Exploratory",
                      *[entry["model_config"][key] for key in
                        ("n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree")]]
                     for entry in plan["models"]]
    notes = ("All detectors use the same cohort, visit chronology, endpoint labels, patient folds and mapping settings. "
             "Only supplied detector artifacts may differ. Missing estimates remain not estimable. "
             "AUROC, AUPRC, sensitivity, specificity and F1 are included in supplementary_metrics.csv. "
             "Each detector report includes cohort accounting, paired metrics, confidence intervals, exclusions, "
             "score trajectories and the cohort-level narrative.")
    markdown = [f"# {title}", "", opening, "", _INTERPRETATION, "", "## Fixed detector settings", "",
                "| " + " | ".join(setting_headings) + " |", "|" + "---|" * len(setting_headings)]
    markdown.extend("| " + " | ".join(map(str, row)) + " |" for row in setting_cells)
    markdown += ["", "These are the settings recorded in each supplied, already trained model artifact; "
                 "the detector is not refitted on GRAPE.", "", "## Paired progression results", "",
                 "| " + " | ".join(headings) + " |", "|" + "---|" * len(headings)]
    markdown.extend("| " + " | ".join(str(cell).replace("|", "\\|").replace("\n", " ") for cell in row) + " |" for row in cells)
    markdown += ["", notes, "", "[Numeric comparison](comparison.csv) · [All metrics](supplementary_metrics.csv) · "
                 "[Plan and model provenance](plan.json) · [Comparability audit](comparison_audit.json) · "
                 "[Code/environment checksums](code/manifest.json)", "", "Full detector reports:", ""]
    markdown.extend(f"- [{name}]({path})" for name, path in links)
    esc = lambda value: html.escape(str(value), quote=True)
    table = "<table><thead><tr>" + "".join(f"<th>{esc(c)}</th>" for c in headings) + "</tr></thead><tbody>"
    table += "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in row) + "</tr>" for row in cells) + "</tbody></table>"
    setting_table = "<table><thead><tr>" + "".join(f"<th>{esc(c)}</th>" for c in setting_headings) + "</tr></thead><tbody>"
    setting_table += "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in row) + "</tr>" for row in setting_cells) + "</tbody></table>"
    page = ("<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>{esc(title)}</title><style>body{{font:16px/1.5 system-ui,sans-serif;max-width:1400px;margin:32px auto;padding:0 20px}}"
            "table{border-collapse:collapse;font-size:14px;width:100%}th,td{border:1px solid #ccd5df;padding:8px;text-align:left}"
            "th{background:#edf2f7}.table{overflow:auto}</style><body>"
            f"<h1>{esc(title)}</h1><p>{esc(opening)}</p><p>{esc(_INTERPRETATION)}</p>"
            f"<h2>Fixed detector settings</h2><div class='table'>{setting_table}</div>"
            "<p>Settings come from the supplied, already trained artifacts; detectors are not refitted on GRAPE.</p>"
            f"<h2>Paired progression results</h2><div class='table'>{table}</div><p>{esc(notes)}</p>"
            "<p><a href='comparison.csv'>Numeric comparison</a> · <a href='supplementary_metrics.csv'>All metrics</a> · "
            "<a href='plan.json'>Plan and model provenance</a> · <a href='comparison_audit.json'>Comparability audit</a> · "
            "<a href='code/manifest.json'>Code/environment checksums</a></p><ul>"
            + "".join(f"<li><a href='{esc(path)}'>{esc(name)}</a></li>" for name, path in links) + "</ul></body></html>\n")
    for filename, content in (("report.md", "\n".join(markdown) + "\n"), ("report.html", page)):
        with (report / filename).open("x", encoding="utf-8") as handle:
            handle.write(content)


def run_external_validation(*, plan_path, run_name, grape_root="data/raw/grape", result_dir="result",
                            artifact_dir="artifacts", device="cuda", cache_dir=".cache/glaboost",
                            image_weights=None, image_batch_size=None, allow_download=False, synthetic=False):
    """Validate all declared fixed detectors, then publish their paired A/B reports."""
    from .reporting import _primary_rows, _supplementary_rows

    if not isinstance(run_name, str) or _NAME.fullmatch(run_name) is None:
        raise ValueError("run_name must be 1–80 safe filename characters, starting alphanumeric.")
    if not isinstance(synthetic, bool):
        raise ValueError("synthetic must be a boolean.")
    submitted_path, submitted = _load_plan(plan_path)
    root = Path(grape_root).expanduser().resolve()
    output, artifacts_root = (ensure_outside_raw(path, root) for path in (result_dir, artifact_dir))
    if output == artifacts_root or output in artifacts_root.parents or artifacts_root in output.parents:
        raise ValueError("Reports and model artifacts require separate, non-overlapping output roots.")
    if (output / "INDEX.md").is_symlink():
        raise ValueError("The report index must not be a symbolic link.")
    for path in (output / run_name, artifacts_root / run_name):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Run already exists; choose a new run name: {path}")
    report, artifacts = output / run_name, artifacts_root / run_name
    cache = ensure_outside_raw(cache_dir, root)
    ensure_outside_raw(cache / "torch" / image_encoder_class("resnet152").checkpoint_filename, root)
    prepared = [_preflight_model(entry, cache, image_weights, allow_download) for entry in submitted["models"]]
    for entry in prepared:
        directory = Path(entry["model_directory"])
        if any(path == directory or directory in path.parents or path in directory.parents
               for path in (report, artifacts, cache)):
            raise ValueError("Outputs/cache must not overlap any supplied fixed model directory.")
    dataset = load_grape(root)
    visits = dataset.image_visits(min_visits=submitted["min_visits"])
    if not visits:
        raise ValueError("No eyes meet the prespecified CFP visit eligibility criteria.")
    records = [{"sample_id": v.sample_id, "patient_id": v.patient_id, "eye_id": v.eye_id,
                "time_years": v.time_years, "labels": dataset.progression_labels[v.eye_id]} for v in visits]
    source = {"workbook_sha256": sha256_file(root / "files" / "VF and clinical information.xlsx"),
              "image_sha256": {v.sample_id: sha256_file(v.image) for v in visits}}
    resolved, gpu_ids = resolve_image_devices(device)
    if image_batch_size is None:
        image_batch_size = 64 * len(gpu_ids) if gpu_ids else 16
    if isinstance(image_batch_size, bool) or not isinstance(image_batch_size, int) or image_batch_size < 1:
        raise ValueError("image_batch_size must be a positive integer.")
    evaluation_config = EvaluationConfig(**submitted["evaluation"], compute_device=resolved)
    created = datetime.now(timezone.utc).isoformat()
    plan = {**submitted, "run_name": run_name, "created_at": created, "synthetic": synthetic,
            "submitted_plan_sha256": sha256_file(submitted_path), "source": source,
            "models": [{**{key: value for key, value in entry.items() if key != "metadata"},
                        "model_config": entry["metadata"]["config"],
                        "encoder_specs": entry["metadata"]["encoder_specs"]} for entry in prepared],
            "runtime": {"requested_device": device, "resolved_device": resolved, "gpu_ids": list(gpu_ids),
                        "image_batch_size": image_batch_size, "cache_dir": str(cache)},
            "interpretation": _INTERPRETATION}
    status = {"status": "running", "stage": "fixed_detector_scoring", "run_name": run_name,
              "created_at": created, "validation_design": "external_fixed_detector_comparison",
              "synthetic": synthetic, "primary_model": plan["primary_model"], "completed_models": []}
    artifacts.mkdir(parents=True, exist_ok=False)
    report.mkdir(parents=True, exist_ok=False)

    def save_status():
        for directory in (report, artifacts):
            (directory / "status.json").write_text(json.dumps(status, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    try:
        for directory in (report, artifacts):
            _json_write(directory / "plan.json", plan)
        save_status()
        _snapshot_code(report / "code")
        reference, rows, supplementary, primary = None, [], [], None
        for entry in tqdm(prepared, desc="Fixed diagnostic detectors", unit="model", dynamic_ncols=True):
            name, directory = entry["name"], Path(entry["model_directory"])
            status.update(stage="fixed_detector_scoring", current_model=name)
            save_status()
            if (sha256_file(directory / "model.json") != entry["model_sha256"]
                    or sha256_file(directory / "metadata.json") != entry["model_metadata_sha256"]):
                raise ValueError("A fixed detector changed after plan preflight.")
            child_artifacts = artifacts / name
            child_artifacts.mkdir(exist_ok=False)
            _json_write(child_artifacts / "model_metadata.json", entry["metadata"])
            model = GlaBoost.load(directory, device=device, cache_dir=str(cache),
                                  image_weights_path=image_weights, image_batch_size=image_batch_size,
                                  allow_download=allow_download)
            blocks = []
            with tqdm(total=len(visits), desc=f"{name}: fixed visit scores", unit="visit", dynamic_ncols=True) as progress:
                for start in range(0, len(visits), image_batch_size):
                    batch = visits[start:start + image_batch_size]
                    block = np.asarray(model.predict_score(batch), dtype=float)
                    if block.shape != (len(batch),):
                        raise ValueError("Fixed detector returned an invalid visit score shape.")
                    blocks.append(block)
                    progress.update(len(batch))
            _json_write(child_artifacts / "prediction_runtime.json", getattr(model, "prediction_runtime_", {}))
            del model
            if gpu_ids:
                import torch
                torch.cuda.empty_cache()
            scores_path = child_artifacts / "visit_scores.csv"
            training = entry["training_data"]
            write_visit_scores(scores_path, visits, np.concatenate(blocks), model_directory=directory,
                               grape_root=root, min_visits=plan["min_visits"],
                               training_data_description=training["description"], training_data_reference=training["reference"],
                               grape_overlap=training["grape_overlap"], independence_evidence=training["independence_evidence"])
            score_metadata = _read_json(score_metadata_path(scores_path))
            if (score_metadata["source"] != source or score_metadata["model_metadata_sha256"] != entry["model_metadata_sha256"]
                    or score_metadata["model_info"]["model_sha256"] != entry["model_sha256"]):
                raise ValueError("Scoring inputs or fixed model changed after plan preflight.")
            status["stage"] = "patient_cross_validated_mappings"
            save_status()
            child = create_study_report(scores_path, run_name=name, grape_root=root, result_dir=report,
                                        config=evaluation_config, synthetic=synthetic)
            if Path(child).resolve() != report / name or _read_json(child / "status.json")["status"] != "complete":
                raise ValueError("A detector report did not finish at its planned output path.")
            evaluation = _read_json(child / "evaluation.json")
            cohort = _read_json(child / "cohort.json")
            provenance = _read_json(child / "provenance.json")
            if (evaluation.get("synthetic") != synthetic or provenance.get("synthetic") != synthetic
                    or evaluation["cohort"] != cohort or provenance["source"] != source
                    or provenance["model_metadata_sha256"] != entry["model_metadata_sha256"]):
                raise ValueError("Saved detector evaluation, cohort or provenance records disagree.")
            for key, value in asdict(evaluation_config).items():
                if evaluation["config"].get(key) != value:
                    raise ValueError(f"Saved progression setting differs from the plan: {key}")
            signature = _analysis_signature(evaluation, cohort, source, records)
            if reference is None:
                reference = signature
            elif signature != reference:
                raise ValueError("Detector configurations have different cohorts, labels, folds or progression settings.")
            role = "Primary" if name == plan["primary_model"] else "Exploratory"
            settings = {key: entry["metadata"]["config"][key] for key in
                        ("n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree")}
            rows.extend({"model": name, "role": role, **settings, **row, "synthetic": synthetic, "report": f"{name}/report.html"}
                        for row in _primary_rows(evaluation))
            supplementary.extend({"model": name, "role": role, **settings, **row, "synthetic": synthetic}
                                 for row in _supplementary_rows(evaluation))
            _json_write(child_artifacts / "progression_heads.json", {
                "synthetic": synthetic, "config": evaluation["config"],
                "endpoints": {endpoint: result["folds"] for endpoint, result in evaluation["endpoints"].items()}})
            if role == "Primary":
                primary = evaluation, cohort, provenance
            status["completed_models"].append(name)
            save_status()
        status["stage"] = "summary"
        save_status()
        audit = {"comparability_verified": True, "synthetic": synthetic, "checked_fields": list(reference),
                 "model_names": status["completed_models"], "primary_model": plan["primary_model"],
                 "analysis_signature_sha256": hashlib.sha256(json.dumps(reference, sort_keys=True, allow_nan=False).encode()).hexdigest(),
                 "detector_independence": "Declared with supplied evidence; not independently established by this software."}
        for directory in (report, artifacts):
            _json_write(directory / "comparison_audit.json", audit)
        _write_summary(report, plan, rows, supplementary, reference["cohort"])
        status.update(status="complete", stage="complete", comparability_verified=True)
        status.pop("current_model", None)
        save_status()
        _refresh_result_index(output)
        if not synthetic:
            _refresh_project_readme(output, f"{run_name}/{plan['primary_model']}", *primary)
        print(f"Completed fixed-detector assessment: {report / 'report.html'}", flush=True)
        return report
    except Exception as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        try:
            diagnostic = artifacts / "error_traceback.txt"
            with diagnostic.open("x", encoding="utf-8") as handle:
                handle.write(traceback.format_exc())
            status["traceback_path"] = str(diagnostic)
        except Exception as diagnostic_error:
            status["traceback_write_error"] = str(diagnostic_error)
        try:
            save_status()
        except Exception:
            pass
        raise
