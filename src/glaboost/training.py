"""One-run GRAPE progression training, nested evaluation, and report output.

This explicitly adapts GlaBoost's frozen ResNet152 + XGBoost architecture to
retrospective eye-level progression. It does not train a glaucoma diagnosis
classifier or claim external validation of a pre-existing GlaBoost model.
"""

import csv
import json
import re
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from .config import GlaBoostConfig
from .data import load_grape
from .encoders import ResNet152Encoder
from .longitudinal import EvaluationConfig
from .study import (
    _environment_info, _json_write, _refresh_project_readme, _refresh_result_index,
    describe_cohort, ensure_outside_raw, sha256_file,
)


def _cohort_records(visits, labels):
    """Build eligibility metadata without inventing scores for the cohort."""
    grouped = defaultdict(list)
    for visit in visits:
        grouped[visit.eye_id].append(visit)
    output = []
    for eye_id, sequence in sorted(grouped.items()):
        sequence = sorted(sequence, key=lambda v: v.time_years)
        times = [v.time_years for v in sequence]
        output.append({"eye_id": eye_id, "patient_id": sequence[0].patient_id,
                       "times": times, "labels": labels[eye_id], "n_visits": len(sequence),
                       "followup_years": times[-1] - times[0]})
    return output


def extract_frozen_features(visits, model_config, *, encoder=None, allow_download=False):
    """Encode each visit once without fitting or reading any progression labels.

    ImageNet normalization and frozen encoder parameters are independent of
    GRAPE. Only this fixed transformation may precede the patient splits.
    """
    visits = list(visits)
    if not visits or any(visit.image is None for visit in visits):
        raise ValueError("Every selected visit requires an original CFP.")
    official_encoder = encoder is None
    if official_encoder:
        if model_config.image_weights_path is not None:
            path = Path(model_config.image_weights_path).expanduser()
            if not sha256_file(path).startswith("394f9c45"):
                raise ValueError("Use the official ImageNet V1 ResNet152 checkpoint (SHA256 prefix 394f9c45).")
        import torch
        torch.set_num_threads(model_config.n_jobs)
        torch.manual_seed(model_config.random_state)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        encoder = ResNet152Encoder(
            weights_path=model_config.image_weights_path, cache_dir=model_config.cache_dir,
            device=model_config.device, batch_size=model_config.image_batch_size,
            allow_download=allow_download,
        )
    blocks = []
    with tqdm(total=len(visits), desc="ResNet152 CFP features", unit="image", dynamic_ncols=True) as progress:
        for start in range(0, len(visits), model_config.image_batch_size):
            batch = visits[start:start + model_config.image_batch_size]
            block = np.asarray(encoder.transform([v.image for v in batch]), dtype=np.float32)
            if block.shape != (len(batch), encoder.output_dim) or not np.isfinite(block).all():
                raise ValueError("Frozen image encoder returned invalid features.")
            blocks.append(block)
            progress.update(len(batch))
    specification = encoder.spec()
    if (specification.get("frozen") is not True or not specification.get("fingerprint")
            or specification.get("output_dim") != encoder.output_dim
            or not isinstance(specification.get("preprocessing"), dict)):
        raise ValueError("The feature encoder must document frozen weights, preprocessing, and fingerprint.")
    if official_encoder and not specification.get("source", {}).get("sha256", "").startswith("394f9c45"):
        raise ValueError("This internal-validation protocol requires the official ImageNet V1 ResNet152 checkpoint; "
                         "a GRAPE-finetuned or unverified encoder cannot be cached before patient splitting.")
    return np.concatenate(blocks), specification


def _write_csv(path, rows, fieldnames):
    with Path(path).open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def train_grape_report(*, run_name, grape_root="data/raw/grape", result_dir="result",
                       artifact_dir="artifacts", model_config=None, evaluation_config=None,
                       inner_splits=2, min_visits=3, allow_download=False, encoder=None,
                       synthetic=False):
    """Train and assess the adapted method, then publish a separate report run.

    ``encoder`` and ``synthetic`` support small offline software tests; the
    command-line entry point always uses the real pretrained image encoder.
    Formal execution is left to the user. No existing result is overwritten.
    """
    from .progression import train_progression
    from .reporting import write_report

    if encoder is not None and not synthetic:
        raise ValueError("Injected feature encoders are reserved for explicitly marked synthetic tests.")
    if not isinstance(run_name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", run_name):
        raise ValueError("run_name must start alphanumeric and contain at most 80 letters/digits/dots/_/-.")
    if isinstance(min_visits, bool) or not isinstance(min_visits, int) or min_visits < 3:
        raise ValueError("The progression study requires at least three CFP visits per eye.")
    if isinstance(inner_splits, bool) or not isinstance(inner_splits, int) or inner_splits < 2:
        raise ValueError("inner_splits must be an integer >= 2.")
    config = model_config or GlaBoostConfig()
    evaluation_config = evaluation_config or EvaluationConfig()
    if (not config.use_image or config.use_structured or config.use_text
            or config.use_human_risk or config.use_human_confidence):
        raise ValueError("The prespecified GRAPE progression training uses original CFP images only.")
    root = Path(grape_root).expanduser().resolve()
    output = ensure_outside_raw(result_dir, root)
    report = ensure_outside_raw(output / run_name, root)
    artifacts = ensure_outside_raw(Path(artifact_dir) / run_name, root)
    cache = ensure_outside_raw(config.cache_dir, root)
    ensure_outside_raw(cache / "torch" / "resnet152-394f9c45.pth", root)
    if report == artifacts or report in artifacts.parents or artifacts in report.parents:
        raise ValueError("Model artifacts and report outputs must be separate directories.")
    if (output / "INDEX.md").is_symlink():
        raise ValueError("The report index must not be a symbolic link.")
    for path in (report, artifacts):
        if path.exists():
            raise FileExistsError(f"Run directory already exists; choose a new run name: {path}")
    dataset = load_grape(root)
    visits = dataset.image_visits(min_visits=min_visits)
    if not visits:
        raise ValueError("No eyes meet the minimum number of eligible CFP visits.")
    records = _cohort_records(visits, dataset.progression_labels)
    cohort = describe_cohort(dataset, records, {"config": config.to_dict()})
    created = datetime.now(timezone.utc).isoformat()
    status = {"status": "running", "stage": "features", "run_name": run_name,
              "created_at": created, "validation_design": "internal_nested_patient_cv",
              "synthetic": synthetic}
    artifacts.mkdir(parents=True, exist_ok=False)
    status_path = artifacts / "status.json"
    def save_status():
        status_path.write_text(json.dumps(status, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    save_status()
    try:
        print(f"Training cohort: {cohort['n_patients']} patients, {cohort['n_eyes']} eyes, {len(visits)} visits", flush=True)
        print("[1/4] Verifying input files", flush=True)
        source = {
            "workbook_sha256": sha256_file(root / "files" / "VF and clinical information.xlsx"),
            "image_sha256": {v.sample_id: sha256_file(v.image) for v in
                             tqdm(visits, desc="Input SHA256", unit="image", dynamic_ncols=True)},
        }
        print("[2/4] Loading frozen ImageNet encoder and extracting image features", flush=True)
        features, encoder_spec = extract_frozen_features(
            visits, config, encoder=encoder, allow_download=allow_download)
        if config.tree_method == "gpu_hist":
            import torch
            # The local encoder has been released; leave its unused CUDA cache
            # available to XGBoost and the endpoint-specific numerical kernels.
            torch.cuda.empty_cache()
        with (artifacts / "image_features.npy").open("xb") as handle:
            np.save(handle, features, allow_pickle=False)
        manifest = {
            "format_version": 1, "synthetic": synthetic, "encoder_spec": encoder_spec,
            "feature_sha256": sha256_file(artifacts / "image_features.npy"),
            "shape": list(features.shape), "source": source,
            "rows": [{"sample_id": v.sample_id, "patient_id": v.patient_id, "eye_id": v.eye_id,
                      "time_years": v.time_years} for v in visits],
            "note": "Fixed ImageNet features only; no GRAPE normalization, target or fitted preprocessing in this matrix.",
        }
        _json_write(artifacts / "feature_manifest.json", manifest)
        status["stage"] = "nested_training"
        save_status()
        print("[3/4] Endpoint-specific XGBoost training, A/B evaluation and bootstrap", flush=True)
        evaluation = train_progression(
            visits, features, dataset.progression_labels, model_config=config,
            evaluation_config=evaluation_config, model_dir=artifacts / "models", inner_splits=inner_splits)
        eye_records = evaluation.pop("eye_records")
        visit_scores = evaluation.pop("visit_predictions")
        provenance = {
            "run_name": run_name, "created_at": created, "validation_design": "internal_nested_patient_cv",
            "model_origin": "GlaBoost architecture adapted to GRAPE progression",
            "score_definition": "Endpoint-specific visit evidence learned from eye-level retrospective progression labels; "
                                "not glaucoma diagnosis or prospectively calibrated risk",
            "minimum_cfp_visits_per_eye": min_visits, "artifacts_directory": str(artifacts),
            "model_info": {
                "config": config.to_dict(), "encoder_spec": encoder_spec,
                "target": {"0": "non-progressing eye", "1": "progressing eye", "separate_endpoints": list(dataset.progression_labels[visits[0].eye_id])},
                "sample_weight": "Inverse number of visits per eye, normalized to mean one within each fit",
                "supervision": "Eye-level full-follow-up outcome repeated over that eye's training visits; weak visit supervision",
            },
            "source": source, "environment": _environment_info(),
            "feature_manifest_sha256": sha256_file(artifacts / "feature_manifest.json"),
            "analysis_scope": "Retrospective internal patient-level validation. Each outer fold holds out entire patients "
                              "from all XGBoost and logistic fits. Inner patient cross-fitting generates training "
                              "scores for both temporal mappings. No future forecasting or external validation claim.",
            "synthetic": synthetic,
        }
        status["stage"] = "report"
        save_status()
        print("[4/4] Generating cohort report, figures and audit tables", flush=True)
        report.mkdir(parents=True, exist_ok=False)
        _json_write(report / "status.json", {**status, "status": "incomplete"})
        write_report(report, evaluation, cohort, provenance, eye_records, synthetic=synthetic)
        _json_write(report / "cohort.json", cohort)
        _write_csv(report / "exclusions.csv", cohort["exclusions"],
                   ("eye_id", "patient_id", "available_cfp_visits", "reason"))
        _write_csv(report / "visit_predictions.csv", visit_scores,
                   ("endpoint", "outer_fold", "inner_fold", "role", "sample_id", "patient_id", "eye_id", "time_years", "progression_score"))
        # Complete training audit stays with the saved fold models as well.
        _json_write(artifacts / "evaluation.json", evaluation)
        _json_write(artifacts / "provenance.json", provenance)
        status.update(status="complete", stage="complete")
        save_status()
        (report / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        _refresh_result_index(output)
        if not synthetic:
            _refresh_project_readme(output, run_name, evaluation, cohort, provenance)
        print(f"Completed report: {report / 'report.html'}", flush=True)
        return report
    except Exception as exc:
        failure_traceback = traceback.format_exc()
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        traceback_path = artifacts / "error_traceback.txt"
        diagnostics = []
        try:
            with traceback_path.open("x", encoding="utf-8") as handle:
                handle.write(failure_traceback)
            status["traceback_path"] = str(traceback_path)
            diagnostics.append(f"Full error traceback: {traceback_path}")
        except Exception as diagnostic_error:
            status["traceback_write_error"] = f"{type(diagnostic_error).__name__}: {diagnostic_error}"
            diagnostics.append(f"Could not save error traceback at {traceback_path}: {diagnostic_error}")
        try:
            save_status()
            if (report / "status.json").is_file():
                (report / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        except Exception as diagnostic_error:
            diagnostics.append(f"Could not update failure status: {diagnostic_error}")
        try:
            print("\n".join(diagnostics), file=sys.stderr, flush=True)
        except Exception:
            # An unavailable log stream must not replace the training failure.
            pass
        raise
