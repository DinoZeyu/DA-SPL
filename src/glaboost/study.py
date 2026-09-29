"""GRAPE study orchestration, score provenance, and separate report runs.

No function here trains the diagnostic detector. Evaluation fits only the two
prespecified progression integrators inside patient-disjoint training folds.
"""

import csv
import hashlib
import json
import math
import platform
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from .data import load_grape


SCORE_COLUMNS = ("sample_id", "patient_id", "eye_id", "time_years", "glaucoma_score")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENDPOINTS = ("plr2", "plr3", "md_slope")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_write(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _read_json(path):
    def invalid(value):
        raise ValueError(f"Nonfinite JSON value: {value}")
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=invalid)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def score_metadata_path(scores_path):
    return Path(scores_path).with_suffix(".metadata.json")


def ensure_outside_raw(path, grape_root):
    """Protect both the retained project data and an explicitly supplied root."""
    path = Path(path).expanduser().resolve()
    protected = (PROJECT_ROOT / "data" / "raw", Path.cwd() / "data" / "raw",
                 Path(grape_root).expanduser().resolve())
    if any(path == base.resolve() or base.resolve() in path.parents for base in protected):
        raise ValueError("Outputs must be outside raw data directories.")
    return path


def _check_model_mapping(metadata):
    if metadata.get("target") != {"0": "normal", "1": "glaucoma"}:
        raise ValueError("Visit scores must come from a normal/glaucoma diagnosis model.")
    config = metadata.get("config", {})
    if not config.get("use_image"):
        raise ValueError("This study requires a CFP-based fixed detector.")
    if any(config.get(k) for k in ("use_text", "use_human_risk", "use_human_confidence")):
        raise ValueError("GRAPE does not provide the selected text/human modalities.")
    if config.get("use_structured") and (
            set(config.get("numeric_features", [])) - {"iop"}
            or config.get("categorical_features")):
        raise ValueError("Only contemporaneous IOP is supported as a GRAPE structured predictor.")


def write_visit_scores(output, visits, scores, *, model_directory, grape_root,
                       min_visits=3, training_data_description="",
                       training_data_reference="", grape_overlap="unknown"):
    """Save scores and a checksum-linked provenance sidecar, without overwrite.

    Training independence is an operator declaration, not inferred from a high
    accuracy, an architecture name, or a checkpoint hash. Unknown provenance is
    retained explicitly and cannot support an external-validation claim.
    """
    if grape_overlap not in ("none", "unknown", "present"):
        raise ValueError("grape_overlap must be none, unknown, or present.")
    if grape_overlap == "present":
        raise ValueError("A detector trained/selected using GRAPE cannot enter this fixed-detector study.")
    if grape_overlap == "none" and not training_data_description.strip():
        raise ValueError("Document the diagnostic training data when declaring no GRAPE overlap.")
    if isinstance(min_visits, bool) or not isinstance(min_visits, int) or min_visits < 3:
        raise ValueError("The longitudinal study requires at least three CFP visits per eye.")
    output = ensure_outside_raw(output, grape_root)
    if output.suffix.lower() != ".csv":
        raise ValueError("Visit scores must use a .csv output filename.")
    sidecar = score_metadata_path(output)
    if output.exists() or sidecar.exists():
        raise FileExistsError(f"Score output or provenance sidecar already exists: {output}")
    visits = list(visits)
    scores = np.asarray(scores, dtype=float)
    if (not visits or scores.shape != (len(visits),) or not np.isfinite(scores).all()
            or np.any((scores < 0) | (scores > 1))):
        raise ValueError("Each visit needs one finite glaucoma score in [0, 1].")
    model_directory = Path(model_directory).resolve()
    model_metadata = _read_json(model_directory / "metadata.json")
    _check_model_mapping(model_metadata)
    if sha256_file(model_directory / "model.json") != model_metadata.get("model_sha256"):
        raise ValueError("Diagnostic model checksum does not match its metadata.")
    root = Path(grape_root).resolve()
    source = {
        "workbook_sha256": sha256_file(root / "files" / "VF and clinical information.xlsx"),
        "image_sha256": {v.sample_id: sha256_file(v.image) for v in visits},
    }
    manifest = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "score_definition": "P(glaucoma), independently computed from each encounter",
        "scoring_protocol": "fixed_detector_independent_visits_v1",
        "min_visits": min_visits,
        "n_visits": len(visits),
        "source": source,
        "model_info": {key: model_metadata.get(key) for key in (
            "implementation", "target", "config", "encoder_specs", "training_summary",
            "model_sha256", "versions", "python")},
        "model_metadata_sha256": sha256_file(model_directory / "metadata.json"),
        "detector_training_data": {
            "description": training_data_description.strip() or "Not documented",
            "reference": training_data_reference.strip() or None,
            "grape_overlap": grape_overlap,
            "independence_evidence": "Operator declaration; not independently verified",
            "scope": "Detector training, preprocessing, feature selection and model selection",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(SCORE_COLUMNS)
        for visit, probability in zip(visits, scores):
            writer.writerow([visit.sample_id, visit.patient_id, visit.eye_id,
                             visit.time_years, float(probability)])
    manifest["scores_sha256"] = sha256_file(output)
    _json_write(sidecar, manifest)
    return output, sidecar


def load_verified_scores(scores_path, dataset, grape_root):
    """Require a complete, unchanged eligible cohort and fixed-model provenance."""
    scores_path = Path(scores_path).resolve()
    metadata = _read_json(score_metadata_path(scores_path))
    if (metadata.get("format_version") != 1
            or metadata.get("scoring_protocol") != "fixed_detector_independent_visits_v1"):
        raise ValueError("Unsupported visit-score provenance; regenerate with score-grape.")
    if sha256_file(scores_path) != metadata.get("scores_sha256"):
        raise ValueError("Visit-score CSV checksum does not match its provenance sidecar.")
    _check_model_mapping(metadata.get("model_info", {}))
    training = metadata.get("detector_training_data", {})
    overlap = training.get("grape_overlap")
    if overlap not in ("none", "unknown"):
        raise ValueError("Detector training/selection must not include GRAPE.")
    if overlap == "none" and (not training.get("description")
                               or training["description"] == "Not documented"):
        raise ValueError("A no-overlap declaration needs a training-data description.")
    minimum = metadata.get("min_visits")
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 3:
        raise ValueError("Score cohort must require at least three CFP visits per eye.")
    expected = {v.sample_id: v for v in dataset.image_visits(min_visits=minimum)}
    rows = []
    seen = set()
    with scores_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(SCORE_COLUMNS):
            raise ValueError(f"Score CSV columns must be exactly {SCORE_COLUMNS}.")
        for row in reader:
            if set(row) != set(SCORE_COLUMNS) or any(value is None for value in row.values()):
                raise ValueError("Each CSV row must have exactly the documented score columns.")
            sample_id = row["sample_id"]
            if sample_id in seen or sample_id not in expected:
                raise ValueError(f"Duplicate or ineligible scored visit: {sample_id}")
            seen.add(sample_id)
            visit = expected[sample_id]
            try:
                elapsed = float(row["time_years"])
                probability = float(row["glaucoma_score"])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid numeric score record: {sample_id}") from exc
            if (not math.isfinite(probability) or not 0 <= probability <= 1
                    or not math.isfinite(elapsed)):
                raise ValueError(f"Nonfinite or out-of-range score record: {sample_id}")
            if (row["patient_id"] != visit.patient_id or row["eye_id"] != visit.eye_id
                    or not math.isclose(elapsed, visit.time_years, rel_tol=0, abs_tol=1e-9)):
                raise ValueError(f"Scored patient/eye/time disagrees with raw GRAPE: {sample_id}")
            rows.append({**row, "time_years": elapsed, "glaucoma_score": probability})
    if not expected or seen != set(expected) or metadata.get("n_visits") != len(rows):
        raise ValueError("Scores must cover every eligible CFP visit exactly once; no selective omissions.")
    source = metadata.get("source", {})
    workbook = Path(grape_root) / "files" / "VF and clinical information.xlsx"
    if source.get("workbook_sha256") != sha256_file(workbook):
        raise ValueError("Raw workbook differs from the scored cohort.")
    image_hashes = source.get("image_sha256", {})
    if set(image_hashes) != set(expected):
        raise ValueError("Score provenance must identify every eligible CFP.")
    for sample_id, visit in expected.items():
        if image_hashes[sample_id] != sha256_file(visit.image):
            raise ValueError(f"Original CFP changed after scoring: {sample_id}")
    return rows, metadata


def _distribution(values):
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    return {"median": float(median), "q1": float(q1), "q3": float(q3)}


def describe_cohort(dataset, eye_records, model_info):
    """Descriptive eligibility and timing information, without fitting a model."""
    eligible = {eye["eye_id"] for eye in eye_records}
    all_by_eye = defaultdict(list)
    for visit in dataset.visits:
        all_by_eye[visit.eye_id].append(visit)
    counts = {eye: sum(v.image is not None for v in visits) for eye, visits in all_by_eye.items()}
    exclusions = [{"eye_id": eye, "patient_id": visits[0].patient_id,
                   "available_cfp_visits": counts[eye],
                   "reason": "Fewer than the prespecified minimum available original CFP visits"}
                  for eye, visits in sorted(all_by_eye.items()) if eye not in eligible]
    gaps = [12 * (max(v.time_years for v in all_by_eye[eye["eye_id"]])
                  - max(eye["times"])) for eye in eye_records]
    selected = [v for v in dataset.visits if v.eye_id in eligible and v.image is not None]
    iop_available = sum(v.structured.get("iop") is not None for v in selected)
    return {
        "n_patients": len({eye["patient_id"] for eye in eye_records}),
        "n_eyes": len(eye_records), "n_visits": len(selected),
        "visits_per_eye": _distribution([eye["n_visits"] for eye in eye_records]),
        "followup_months": _distribution([12 * eye["followup_years"] for eye in eye_records]),
        "progression": {ep: {
            "positive_eyes": sum(eye["labels"][ep] for eye in eye_records),
            "total_eyes": len(eye_records),
            "prevalence": sum(eye["labels"][ep] for eye in eye_records) / len(eye_records),
        } for ep in ENDPOINTS},
        "source": {"n_patients": len({v.patient_id for v in dataset.visits}),
                   "n_eyes": len(dataset.progression_labels), "n_visits": len(dataset.visits),
                   "n_visits_with_cfp": sum(counts.values())},
        "exclusions": exclusions,
        "modalities": {
            "original_cfp": {"selected": True, "available_visits": len(selected)},
            "contemporaneous_iop": {
                "selected": bool(model_info["config"].get("use_structured")),
                "available_visits": iop_available, "missing_visits": len(selected) - iop_available},
            "text_and_human_risk": {"selected": False, "available_visits": 0},
            "vf": {"selected": False, "role": "Reference outcomes only"},
            "baseline_oct": {"selected": False, "role": "Not repeated across follow-up"},
        },
        "observation_window": {
            "eyes_last_cfp_before_last_recorded_visit": sum(gap > 1e-8 for gap in gaps),
            "cfp_to_last_recorded_visit_months": _distribution(gaps),
            "note": "CFP observation span is not necessarily the full VF label ascertainment window; "
                    "no prospectively defined prediction horizon. Released outcome labels are not recomputed.",
        },
    }


def _environment_info():
    packages = {}
    for name in ("numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "xgboost", "openpyxl", "torch", "torchvision"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": packages, "git_commit": commit,
            "source_sha256": {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
            "uv_lock_sha256": sha256_file(PROJECT_ROOT / "uv.lock")
            if (PROJECT_ROOT / "uv.lock").is_file() else None}


def _refresh_result_index(result_dir):
    """Index complete runs only; never select the run with the best performance."""
    if (result_dir / "INDEX.md").is_symlink():
        raise ValueError("The report index must not be a symbolic link.")
    lines = ["# GRAPE analysis reports", "", "Separate runs are retained; no run is selected by performance.",
             "Synthetic smoke tests are not stored here.", "",
             "| Run | Created (UTC) | Status | Report |", "|---|---|---|---|"]
    for directory in sorted(result_dir.iterdir()):
        status_path = directory / "status.json"
        if not directory.is_dir() or not status_path.is_file():
            continue
        status = _read_json(status_path)
        if status.get("status") == "complete":
            name = directory.name
            lines.append(f"| {name} | {status['created_at']} | {status['validation_design']} | "
                         f"[HTML]({name}/report.html) / [Markdown]({name}/report.md) |")
    (result_dir / "INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _refresh_project_readme(result_dir, run_name, evaluation, cohort, provenance):
    """Update only the designated results block for the default project output.

    Custom output locations (including synthetic tests in /tmp) never modify
    the project's README. The latest completed run is shown, not the best run.
    """
    if provenance.get("synthetic") or evaluation.get("synthetic"):
        return
    if result_dir.resolve() != (PROJECT_ROOT / "result").resolve():
        return
    readme = PROJECT_ROOT / "README.md"
    if not readme.is_file():
        return
    if readme.is_symlink():
        raise ValueError("The root README must not be a symbolic link when updating results.")
    start, end = "<!-- glaboost-results:start -->", "<!-- glaboost-results:end -->"
    text = readme.read_text(encoding="utf-8")
    if text.count(start) != 1 or text.count(end) != 1:
        return
    def finite(number):
        try:
            return number is not None and math.isfinite(float(number))
        except (TypeError, ValueError):
            return False

    def value(number, *, digits=3, scale=1, signed=False):
        if not finite(number):
            return "Not estimable"
        return format(float(number) * scale, ("+" if signed else "") + f".{digits}f")

    def estimate_interval(metric, *, signed=False):
        metric = metric or {}
        point = value(metric.get("estimate"), digits=1, scale=100, signed=signed)
        if not finite(metric.get("estimate")):
            return point
        if not all(finite(metric.get(key)) for key in ("ci_low", "ci_high")):
            return f"{point} [CI not estimable]"
        bounds = [value(metric[key], digits=1, scale=100, signed=signed) for key in ("ci_low", "ci_high")]
        return f"{point} [{bounds[0]}, {bounds[1]}]"

    def count(number):
        return str(number) if finite(number) else "Not recorded"

    names = {"plr2": "PLR2", "plr3": "PLR3", "md_slope": "MD slope"}
    lines = [start, "", f"Latest completed run: `{run_name}` (updated on completion, never selected by performance).",
             f"**Cohort:** {count(cohort.get('n_patients'))} patients, {count(cohort.get('n_eyes'))} eyes, "
             f"and {count(cohort.get('n_visits'))} eligible CFP visits."]
    medians = []
    for key, label, unit in (("visits_per_eye", "visits per eye", ""),
                             ("followup_months", "CFP follow-up", " months")):
        distribution = cohort.get(key, {})
        if finite(distribution.get("median")):
            summary = f"median {label}: {value(distribution['median'], digits=1)}{unit}"
            if all(finite(distribution.get(q)) for q in ("q1", "q3")):
                summary += f" (IQR {value(distribution['q1'], digits=1)}–{value(distribution['q3'], digits=1)})"
            medians.append(summary)
    if medians:
        summary = "; ".join(medians)
        lines.append(summary[0].upper() + summary[1:] + ".")
    lines += [f"[Full report](result/{run_name}/report.html) · [Markdown](result/{run_name}/report.md) · "
              "[All completed reports](result/INDEX.md)",
              "", "**Primary comparison: balanced accuracy (BA).** A uses the latest visit; B integrates longitudinal scores.",
              "BA and its 95% CI are percentages; the paired B − A difference and its CI are percentage points (pp).",
              "", "| Endpoint | Eyes | Progressors, n (%) | A: latest BA, % [95% CI] | B: longitudinal BA, % [95% CI] | Paired B − A, pp [95% CI] |",
              "|---|---:|---:|---:|---:|---:|"]
    for endpoint in ENDPOINTS:
        result = evaluation.get("endpoints", {}).get(endpoint, {})
        progression = cohort.get("progression", {}).get(endpoint, {})
        eyes = result.get("n_eyes", progression.get("total_eyes", cohort.get("n_eyes")))
        positive = result.get("n_positive_eyes", progression.get("positive_eyes"))
        prevalence = (float(positive) / float(eyes) if finite(positive) and finite(eyes) and float(eyes) > 0
                      else progression.get("prevalence"))
        progressors = count(positive)
        if finite(prevalence):
            progressors += f" ({value(prevalence, digits=1, scale=100)}%)"
        entries = ["Not estimable"] * 3
        if result.get("status") == "ok":
            metrics = result.get("metrics", {})
            entries = [estimate_interval(metrics.get(method, {}).get("balanced_accuracy"))
                       for method in ("latest", "longitudinal")]
            entries.append(estimate_interval(result.get("delta_balanced_accuracy"), signed=True))
        lines.append(f"| {names[endpoint]} | {count(eyes)} | {progressors} | " + " | ".join(entries) + " |")
    lines += ["", f"[Primary comparison figure (PNG)](result/{run_name}/figures/primary_comparison.png) · "
              f"[PDF](result/{run_name}/figures/primary_comparison.pdf) · "
              f"[Primary results CSV](result/{run_name}/primary_results.csv)",
              "", "**Secondary metrics:** point estimates on the 0–1 scale. AUPRC uses average precision (AP).",
              f"The [supplementary CSV](result/{run_name}/supplementary_metrics.csv) and full report include all 95% CIs "
              "and reasons for non-estimable results.",
              "", "| Endpoint | Method | AUROC | AUPRC (AP) | Sensitivity | Specificity | F1 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for endpoint in ENDPOINTS:
        result = evaluation.get("endpoints", {}).get(endpoint, {})
        for method, label in (("latest", "A: latest"), ("longitudinal", "B: longitudinal")):
            metrics = result.get("metrics", {}).get(method, {}) if result.get("status") == "ok" else {}
            estimates = [value(metrics.get(metric, {}).get("estimate"))
                         for metric in ("auroc", "auprc", "sensitivity", "specificity", "f1")]
            lines.append(f"| {names[endpoint]} | {label} | " + " | ".join(estimates) + " |")
    lines += ["", f"![Latest versus longitudinal balanced accuracy and paired differences, with 95% patient-bootstrap CIs](result/{run_name}/figures/primary_comparison.png)"]
    design = provenance.get("validation_design")
    if design == "internal_nested_patient_cv":
        lines += ["", "This is **retrospective internal validation on GRAPE**: ResNet-152 features and XGBoost are "
                  "adapted to three separate progression endpoints. Inner patient-held-out scores train A/B mappings; "
                  "outer held-out patients supply the final estimates. This is not external validation of a pretrained diagnosis model."]
    elif design == "external_fixed_detector":
        lines += ["", "The fixed visit detector is declared independent of GRAPE in the supplied provenance. "
                  "The progression mappings are nevertheless fitted and patient-cross-validated within GRAPE; "
                  "the complete progression pipeline has not undergone an independent external validation."]
    else:
        lines += ["", "The visit detector's independence from GRAPE has not been established; these results "
                  "cannot establish external validation."]
    lines += ["", "Confidence intervals use paired patient-level bootstrap conditional on fixed out-of-fold predictions; "
              "they exclude uncertainty from refitting the training pipeline. This is retrospective progression "
              "assessment, not forecasting future progression. A positive point estimate for one endpoint does not "
              "establish a consistent benefit across all three endpoints.", "", end]
    before, tail = text.split(start, 1)
    _, after = tail.split(end, 1)
    readme.write_text(before + "\n".join(lines) + after, encoding="utf-8")


def create_study_report(scores_path, *, run_name, grape_root="data/raw/grape",
                        result_dir="result", config=None):
    """Run the prespecified A/B evaluation and create one immutable report folder.

    This is an explicit experiment entry point for the user to execute. The
    implementation is tested using small synthetic fixtures only.
    """
    from .longitudinal import EvaluationConfig, evaluate_longitudinal, prepare_eye_records
    from .reporting import write_report

    if not isinstance(run_name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", run_name):
        raise ValueError("run_name must be 1-80 letters/digits/dots/underscores/hyphens, starting alphanumeric.")
    result_dir = ensure_outside_raw(result_dir, grape_root)
    if (result_dir / "INDEX.md").is_symlink():
        raise ValueError("The report index must not be a symbolic link.")
    run_dir = ensure_outside_raw(result_dir / run_name, grape_root)
    if run_dir.exists():
        raise FileExistsError(f"Report run already exists; use a new run name: {run_dir}")
    dataset = load_grape(grape_root)
    rows, metadata = load_verified_scores(scores_path, dataset, grape_root)
    eyes = prepare_eye_records(rows, dataset.progression_labels, min_visits=metadata["min_visits"])
    cohort = describe_cohort(dataset, eyes, metadata["model_info"])
    evaluation = evaluate_longitudinal(eyes, config or EvaluationConfig())
    design = ("external_fixed_detector" if metadata["detector_training_data"]["grape_overlap"] == "none"
              else "retrospective_fixed_detector_unverified_external")
    provenance = {
        "run_name": run_name, "created_at": datetime.now(timezone.utc).isoformat(),
        "validation_design": design, "model_origin": "paper_reimplementation",
        "model_info": metadata["model_info"],
        "model_metadata_sha256": metadata.get("model_metadata_sha256"),
        "detector_training_data": metadata["detector_training_data"],
        "score_definition": metadata["score_definition"],
        "minimum_cfp_visits_per_eye": metadata["min_visits"],
        "scores_sha256": metadata["scores_sha256"],
        "score_metadata_sha256": sha256_file(score_metadata_path(scores_path)),
        "source": metadata["source"], "environment": _environment_info(),
        "analysis_scope": "Retrospective progression detection; temporal integrators are internally "
                          "cross-validated on GRAPE. Not external validation of a frozen complete progression model.",
    }
    run_dir.mkdir(parents=True, exist_ok=False)
    status = {"status": "incomplete", "run_name": run_name,
              "created_at": provenance["created_at"], "validation_design": design}
    _json_write(run_dir / "status.json", status)
    try:
        write_report(run_dir, evaluation, cohort, provenance, eyes)
        _json_write(run_dir / "cohort.json", cohort)
        # Preserve exact bytes: the archived CSV and sidecar retain the checksums
        # already recorded in provenance, including original numeric formatting.
        for source, name, expected_hash in (
            (scores_path, "visit_scores.csv", metadata["scores_sha256"]),
            (score_metadata_path(scores_path), "visit_scores.metadata.json", provenance["score_metadata_sha256"]),
        ):
            original = Path(source).read_bytes()
            if hashlib.sha256(original).hexdigest() != expected_hash:
                raise ValueError("Score inputs changed while creating the report.")
            with (run_dir / name).open("xb") as handle:
                handle.write(original)
        with (run_dir / "exclusions.csv").open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("eye_id", "patient_id", "available_cfp_visits", "reason"))
            writer.writeheader()
            writer.writerows(cohort["exclusions"])
    except Exception as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        (run_dir / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        raise
    status["status"] = "complete"
    (run_dir / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    _refresh_result_index(result_dir)
    _refresh_project_readme(result_dir, run_name, evaluation, cohort, provenance)
    return run_dir
