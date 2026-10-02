"""External-data GlaBoost training, fixed scoring, and longitudinal GRAPE reports."""

import argparse
import re


DEFAULT_HF_ROOT = "/scratch/users/zeyuhan/DA-SPL/archive/glaucoma_diagnosis_json_analysis"


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
    validate.add_argument("--plan", required=True, help="An existing run's external_models.json with model bundles and training provenance")
    validate.add_argument("--run-name", required=True, help="New unique name for this analysis package")
    validate.add_argument("--root", default="data/raw/grape")
    validate.add_argument("--result-dir", default="result")
    validate.add_argument("--artifact-dir", default="artifacts")
    _image_options(validate)
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
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"glaboost: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
