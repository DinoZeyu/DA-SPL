"""GRAPE training, patient-level progression validation, and cohort reports."""

import argparse
import json
from collections import Counter

from .data import load_grape
from .config import GlaBoostConfig
from .model import GlaBoost
from .study import ensure_outside_raw, score_metadata_path, write_visit_scores


def _execution_settings(device, image_batch_size=None, xgb_threads=None):
    from .encoders import resolve_image_devices
    resolved, gpu_ids = resolve_image_devices(device)
    for name, value in (("image_batch_size", image_batch_size), ("xgb_threads", xgb_threads)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            raise ValueError(f"{name} must be a positive integer.")
    batch = image_batch_size if image_batch_size is not None else (64 * len(gpu_ids) if gpu_ids else 16)
    threads = xgb_threads if xgb_threads is not None else 1
    return resolved, gpu_ids, batch, threads


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
        "eligible_eyes": len(eyes),
        "eligible_cfp_visits": len(selected),
        "progression_positive_eyes_in_selected_cohort": {
            endpoint: sum(dataset.progression_labels[eye][endpoint] for eye in eyes)
            for endpoint in ("plr2", "plr3", "md_slope")
        },
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect = subparsers.add_parser("inspect-grape", help="Check cohort mapping; no model or training")
    score = subparsers.add_parser("score-grape", help="Apply a previously fitted diagnosis model to visits")
    evaluate = subparsers.add_parser("evaluate-grape", help="Run patient-grouped A/B evaluation and create a report")
    train = subparsers.add_parser("train-grape", help="Train the paper architecture on GRAPE progression and generate an internal-validation report")
    for sub in (inspect, score):
        sub.add_argument("--root", default="data/raw/grape", help="Existing GRAPE raw directory")
        sub.add_argument("--min-visits", type=int, default=3, help="Minimum original CFP visits per eye")
    score.add_argument("--model", required=True, help="Directory containing model.json and metadata.json")
    score.add_argument("--output", required=True, help="New CSV path, outside data/raw")
    score.add_argument("--device", default="cpu")
    score.add_argument("--cache-dir", default=".cache/glaboost")
    score.add_argument("--image-weights", help="Optional relocated ResNet152 state_dict")
    score.add_argument("--allow-download", action="store_true", help="Explicitly allow pretrained encoder downloads")
    score.add_argument("--training-data-description", default="", help="Actual diagnostic detector training/selection data")
    score.add_argument("--training-data-reference", default="", help="Training provenance reference, if available")
    score.add_argument("--grape-training-overlap", choices=("unknown", "none", "present"), default="unknown",
                       help="Declare GRAPE overlap in detector training/preprocessing/selection; unknown cannot establish external validation")
    evaluate.add_argument("--root", default="data/raw/grape")
    evaluate.add_argument("--scores", required=True, help="score-grape CSV with matching .metadata.json sidecar")
    evaluate.add_argument("--run-name", required=True, help="New unique subdirectory name under result/")
    evaluate.add_argument("--result-dir", default="result")
    evaluate.add_argument("--folds", type=int, default=3, help="Requested patient-grouped CV folds; reduced only for class feasibility")
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument("--bootstrap", type=int, default=2000, help="Patient-cluster bootstrap replicates")
    evaluate.add_argument("--persistence-threshold", type=float, default=0.5,
                          help="Prespecify before observing outcomes; fraction of visit scores above this value")
    evaluate.add_argument("--logistic-c", type=float, default=1.0, help="Prespecified inverse L2 regularization strength")
    train.add_argument("--root", default="data/raw/grape")
    train.add_argument("--run-name", required=True)
    train.add_argument("--result-dir", default="result")
    train.add_argument("--artifact-dir", default="artifacts")
    train.add_argument("--min-visits", type=int, default=3)
    train.add_argument("--folds", type=int, default=3)
    train.add_argument("--inner-folds", type=int, default=2)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--bootstrap", type=int, default=2000)
    train.add_argument("--persistence-threshold", type=float, default=0.5)
    train.add_argument("--logistic-c", type=float, default=1.0)
    train.add_argument("--device", default="cuda", help="cuda (default): all visible GPUs, no CPU fallback; cuda:N: one GPU; cpu: explicit CPU execution")
    train.add_argument("--image-batch-size", type=int, help="Global image batch (default: 64 per used GPU; preset for A100 80GB)")
    train.add_argument("--xgb-threads", type=int, help="Host helper threads per XGBoost fit (default: 1; tree learning uses GPU)")
    train.add_argument("--cache-dir", default=".cache/glaboost")
    train.add_argument("--image-weights", help="Optional local copy of official ImageNet V1 ResNet152 weights")
    train.add_argument("--allow-download", action="store_true", help="Allow downloading pretrained ImageNet encoder weights")
    args = parser.parse_args(argv)
    try:
        if args.command == "train-grape":
            from .longitudinal import EvaluationConfig
            from .training import train_grape_report
            resolved, gpu_ids, batch_size, threads = _execution_settings(
                args.device, args.image_batch_size, args.xgb_threads)
            print(f"Execution: requested={args.device}, image_device={resolved}, "
                  f"GPU IDs={list(gpu_ids)}, GPU count={len(gpu_ids)}, "
                  f"global image batch={batch_size}, host helper threads={threads}, "
                  f"numeric backend={'CUDA' if gpu_ids else 'explicit CPU'}", flush=True)
            model_config = GlaBoostConfig(
                image_weights_path=args.image_weights, cache_dir=args.cache_dir, device=args.device,
                image_batch_size=batch_size, random_state=args.seed, n_jobs=threads,
                tree_method="gpu_hist" if gpu_ids else "hist", gpu_id=gpu_ids[0] if gpu_ids else None)
            evaluation = EvaluationConfig(n_splits=args.folds, seed=args.seed,
                                          bootstrap_replicates=args.bootstrap,
                                          persistence_threshold=args.persistence_threshold,
                                          logistic_c=args.logistic_c, compute_device=resolved)
            run = train_grape_report(run_name=args.run_name, grape_root=args.root,
                                     result_dir=args.result_dir, artifact_dir=args.artifact_dir,
                                     model_config=model_config, evaluation_config=evaluation,
                                     inner_splits=args.inner_folds, min_visits=args.min_visits,
                                     allow_download=args.allow_download)
            print(f"Saved internal-validation report: {run / 'report.html'}")
            return 0
        if args.command == "evaluate-grape":
            from .longitudinal import EvaluationConfig
            from .study import create_study_report
            configuration = EvaluationConfig(n_splits=args.folds, seed=args.seed,
                                             bootstrap_replicates=args.bootstrap,
                                             persistence_threshold=args.persistence_threshold,
                                             logistic_c=args.logistic_c)
            run = create_study_report(args.scores, run_name=args.run_name, grape_root=args.root,
                                      result_dir=args.result_dir, config=configuration)
            print(f"Saved cohort report: {run / 'report.html'}")
            return 0
        if args.min_visits < 1:
            raise ValueError("--min-visits must be positive.")
        dataset = load_grape(args.root)
        if args.command == "inspect-grape":
            print(json.dumps(cohort_summary(dataset, args.min_visits), indent=2))
            return 0
        if args.min_visits < 3:
            raise ValueError("The longitudinal study requires --min-visits >= 3.")
        if args.grape_training_overlap == "present":
            raise ValueError("This fixed-detector study cannot use a model trained or selected on GRAPE.")
        if args.grape_training_overlap == "none" and not args.training_data_description.strip():
            raise ValueError("--training-data-description is required when declaring no GRAPE overlap.")
        output = _output_path(args.output, args.root)
        cache_dir = ensure_outside_raw(args.cache_dir, args.root)
        ensure_outside_raw(cache_dir / "torch" / "resnet152-394f9c45.pth", args.root)
        model = GlaBoost.load(args.model, device=args.device, cache_dir=str(cache_dir),
                             image_weights_path=args.image_weights,
                             allow_download=args.allow_download)
        c = model.config
        if c.use_text or c.use_human_risk or c.use_human_confidence:
            raise ValueError("GRAPE does not supply the paper's visit-level text or human risk/confidence inputs.")
        if c.use_structured and (set(c.numeric_features) - {"iop"} or c.categorical_features):
            raise ValueError("This GRAPE adapter exposes only visit-level IOP as a structured predictor.")
        if not c.use_image:
            raise ValueError("score-grape currently requires a CFP-based model and CFP-eligible cohort.")
        visits = dataset.image_visits(min_visits=args.min_visits)
        if not visits:
            raise ValueError("No eyes have the requested number of eligible CFP visits.")
        scores = model.predict_score(visits)
        _, metadata_path = write_visit_scores(
            output, visits, scores, model_directory=args.model, grape_root=args.root,
            min_visits=args.min_visits, training_data_description=args.training_data_description,
            training_data_reference=args.training_data_reference,
            grape_overlap=args.grape_training_overlap)
        print(f"Saved {len(visits)} independent visit scores to {output}")
        print(f"Saved score provenance to {metadata_path}")
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"glaboost: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
