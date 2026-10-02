"""External-data GlaBoost training, fixed scoring, and longitudinal GRAPE reports."""

import argparse
import json
import re
from collections import Counter

from .data import load_grape
from .model import GlaBoost
from .study import ensure_outside_raw, score_metadata_path, write_visit_scores


DEFAULT_HF_ROOT = "/scratch/users/zeyuhan/DA-SPL/archive/glaucoma_diagnosis_json_analysis"


def _execution_settings(device, image_batch_size=None):
    from .encoders import resolve_image_devices
    if image_batch_size is not None and (
            isinstance(image_batch_size, bool) or not isinstance(image_batch_size, int) or image_batch_size < 1):
        raise ValueError("image_batch_size must be a positive integer")
    resolved, gpu_ids = resolve_image_devices(device)
    batch = image_batch_size if image_batch_size is not None else (64 * len(gpu_ids) if gpu_ids else 16)
    return resolved, gpu_ids, batch


def cohort_summary(dataset, min_visits=3):
    selected = dataset.image_visits(min_visits=min_visits)
    eyes = {v.eye_id for v in selected}
    return {
        "all_patients": len({v.patient_id for v in dataset.visits}),
        "all_eyes": len(dataset.progression_labels),
        "all_visits_including_baseline": len(dataset.visits),
        "visits_with_original_cfp": sum(v.image is not None for v in dataset.visits),
        "minimum_cfp_visits_per_eye": min_visits,
        "eligible_patients": len({v.patient_id for v in selected}),
        "eligible_eyes": len(eyes), "eligible_cfp_visits": len(selected),
        "progression_positive_eyes_in_selected_cohort": {
            endpoint: sum(dataset.progression_labels[eye][endpoint] for eye in eyes)
            for endpoint in ("plr2", "plr3", "md_slope")},
        "eligible_eyes_by_number_of_cfps": dict(sorted(Counter(
            Counter(v.eye_id for v in selected).values()).items())),
        "diagnosis_training_available": False,
        "note": "GRAPE contains glaucoma eyes; progression labels are not diagnosis labels.",
    }


def _output_path(value, root):
    output = ensure_outside_raw(value, root)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if output.suffix.lower() != ".csv":
        raise ValueError("Visit scores must use a .csv filename.")
    if score_metadata_path(output).exists():
        raise FileExistsError(f"Score provenance already exists: {score_metadata_path(output)}")
    return output


def _device_value(value):
    if value != "cpu" and re.fullmatch(r"cuda(?::[0-9]+)?", value) is None:
        raise argparse.ArgumentTypeError("Use cuda, cuda:N, or explicit cpu; automatic CPU fallback is disabled")
    return value


def _image_options(parser):
    parser.add_argument("--device", type=_device_value, default="cuda", help="cuda: all visible GPUs; cuda:N: one GPU; cpu: explicit CPU")
    parser.add_argument("--cache-dir", default=".cache/glaboost")
    parser.add_argument("--image-weights", help="Relocated image checkpoint matching the saved detector fingerprint")
    parser.add_argument("--image-batch-size", type=int, help="Global image batch (default: 64 per selected GPU)")
    parser.add_argument("--allow-download", action="store_true", help="Explicitly permit pretrained encoder downloads; default offline")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    pipeline = subparsers.add_parser("run-hf-grape", help="Train diagnostic detectors on retained HF data, freeze them, then evaluate longitudinal GRAPE evidence")
    pipeline.add_argument("--training-plan", default="configs/hf_training.json", help="Prespecified external-source training configurations and GRAPE analysis settings")
    pipeline.add_argument("--hf-root", default=DEFAULT_HF_ROOT, help="Existing read-only HF archive; never downloaded")
    pipeline.add_argument("--root", default="data/raw/grape", help="Existing read-only GRAPE root")
    pipeline.add_argument("--run-name", required=True)
    pipeline.add_argument("--result-dir", default="result")
    pipeline.add_argument("--artifact-dir", default="artifacts")
    _image_options(pipeline)
    validate = subparsers.add_parser("validate-grape", help="Apply the predefined fixed external detector plan and generate reports")
    validate.add_argument("--plan", default="configs/external_models.json", help="Model bundles, training provenance and fixed analysis settings")
    validate.add_argument("--run-name", required=True, help="New unique name for this analysis package")
    validate.add_argument("--root", default="data/raw/grape")
    validate.add_argument("--result-dir", default="result")
    validate.add_argument("--artifact-dir", default="artifacts")
    _image_options(validate)
    inspect = subparsers.add_parser("inspect-grape", help="Inspect the existing cohort without loading a model")
    score = subparsers.add_parser("score-grape", help="Apply one fitted diagnosis detector independently to visits")
    for sub in (inspect, score):
        sub.add_argument("--root", default="data/raw/grape")
        sub.add_argument("--min-visits", type=int, default=3, help="Minimum original CFP visits per eye")
    score.add_argument("--model", required=True, help="Directory with compatible model.json and metadata.json")
    score.add_argument("--output", required=True, help="New score CSV path outside raw data")
    _image_options(score)
    score.add_argument("--training-data-description", default="", help="Actual diagnostic training and selection data")
    score.add_argument("--training-data-reference", default="", help="Training provenance reference, if available")
    score.add_argument("--independence-evidence", default="Operator declaration; not independently verified",
                       help="Evidence for independence from GRAPE; hashes alone cannot verify independence")
    score.add_argument("--grape-training-overlap", choices=("unknown", "none", "present"), default="unknown",
                       help="Training/preprocessing/selection overlap; unknown cannot establish external validation")
    evaluate = subparsers.add_parser("evaluate-grape", help="Evaluate verified scores using patient-grouped A/B validation")
    evaluate.add_argument("--root", default="data/raw/grape")
    evaluate.add_argument("--scores", required=True, help="score-grape CSV and matching .metadata.json sidecar")
    evaluate.add_argument("--run-name", required=True)
    evaluate.add_argument("--result-dir", default="result")
    evaluate.add_argument("--device", type=_device_value, default="cuda", help="GPU for logistic fitting and evaluation; cpu is explicit")
    evaluate.add_argument("--folds", type=int, default=3, help="Patient-grouped CV folds; reduced only for class feasibility")
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument("--bootstrap", type=int, default=2000)
    evaluate.add_argument("--persistence-threshold", type=float, default=0.5)
    evaluate.add_argument("--logistic-c", type=float, default=1.0)
    args = parser.parse_args(argv)
    try:
        if args.command == "run-hf-grape":
            from .hf_training import run_hf_grape
            run = run_hf_grape(
                training_plan_path=args.training_plan, run_name=args.run_name,
                hf_root=args.hf_root, grape_root=args.root, result_dir=args.result_dir,
                artifact_dir=args.artifact_dir, device=args.device, cache_dir=args.cache_dir,
                image_weights=args.image_weights, image_batch_size=args.image_batch_size,
                allow_download=args.allow_download)
            print(f"Saved HF-trained detector / GRAPE longitudinal analysis: {run / 'report.html'}")
            return 0
        if args.command == "validate-grape":
            from .external import run_external_validation
            # Plan/model validation comes before CUDA resolution and output creation.
            run = run_external_validation(
                plan_path=args.plan, run_name=args.run_name, grape_root=args.root,
                result_dir=args.result_dir, artifact_dir=args.artifact_dir,
                device=args.device, cache_dir=args.cache_dir, image_weights=args.image_weights,
                image_batch_size=args.image_batch_size, allow_download=args.allow_download)
            print(f"Saved external-detector longitudinal analysis: {run / 'report.html'}")
            return 0
        if args.command == "evaluate-grape":
            from .longitudinal import EvaluationConfig
            from .study import create_study_report
            resolved, _, _ = _execution_settings(args.device)
            configuration = EvaluationConfig(n_splits=args.folds, seed=args.seed,
                                             bootstrap_replicates=args.bootstrap,
                                             persistence_threshold=args.persistence_threshold,
                                             logistic_c=args.logistic_c, compute_device=resolved)
            run = create_study_report(args.scores, run_name=args.run_name, grape_root=args.root,
                                      result_dir=args.result_dir, config=configuration)
            print(f"Saved cohort report: {run / 'report.html'}")
            return 0
        if args.min_visits < 1:
            raise ValueError("--min-visits must be positive")
        if args.command == "inspect-grape":
            print(json.dumps(cohort_summary(load_grape(args.root), args.min_visits), indent=2))
            return 0
        if args.min_visits < 3:
            raise ValueError("The longitudinal study requires --min-visits >= 3")
        if args.grape_training_overlap == "present":
            raise ValueError("A detector trained or selected using GRAPE cannot enter this fixed-detector study")
        if args.grape_training_overlap == "none" and not args.training_data_description.strip():
            raise ValueError("--training-data-description is required when declaring no GRAPE overlap")
        if not args.independence_evidence.strip():
            raise ValueError("--independence-evidence must describe the actual evidence or uncertainty")
        output = _output_path(args.output, args.root)
        cache_dir = ensure_outside_raw(args.cache_dir, args.root)
        resolved, gpu_ids, batch = _execution_settings(args.device, args.image_batch_size)
        print(f"Scoring: requested={args.device}, primary={resolved}, GPUs={list(gpu_ids)}, global image batch={batch}", flush=True)
        model = GlaBoost.load(args.model, device=args.device, cache_dir=str(cache_dir),
                             image_weights_path=args.image_weights, image_batch_size=batch,
                             allow_download=args.allow_download)
        c = model.config
        from .encoders import image_encoder_class
        ensure_outside_raw(cache_dir / "torch" / image_encoder_class(c.image_encoder).checkpoint_filename, args.root)
        if c.use_text or c.use_human_risk or c.use_human_confidence:
            raise ValueError("GRAPE does not supply the detector's text or human risk/confidence inputs")
        if c.use_structured and (set(c.numeric_features) - {"iop"} or c.categorical_features):
            raise ValueError("This GRAPE adapter exposes only visit-level IOP as a structured predictor")
        if not c.use_image:
            raise ValueError("score-grape requires a CFP-based detector and CFP-eligible cohort")
        dataset = load_grape(args.root)
        visits = dataset.image_visits(min_visits=args.min_visits)
        if not visits:
            raise ValueError("No eyes have the requested number of eligible CFP visits")
        scores = model.predict_score(visits)
        _, metadata_path = write_visit_scores(
            output, visits, scores, model_directory=args.model, grape_root=args.root,
            min_visits=args.min_visits, training_data_description=args.training_data_description,
            training_data_reference=args.training_data_reference, grape_overlap=args.grape_training_overlap,
            independence_evidence=args.independence_evidence)
        print(f"Saved {len(visits)} independent visit scores to {output}")
        print(f"Saved score provenance to {metadata_path}")
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"glaboost: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
