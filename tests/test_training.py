"""Synthetic tests for the one-run training/report path; no real GRAPE fits."""

import csv
import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch
import unittest

import numpy as np

import test_study
from glaboost.config import GlaBoostConfig
from glaboost.longitudinal import EvaluationConfig
from glaboost.training import extract_frozen_features, train_grape_report


class SyntheticEncoder:
    output_dim = 4

    def __init__(self):
        self.seen = []

    def transform(self, images):
        self.seen.extend(images)
        rows = []
        for image in images:
            patient, side, visit = Path(image).stem.split("_")
            rows.append([int(patient) / 8, int(visit) / 3,
                         np.sin(int(patient) + int(visit)), float(side == "OS")])
        return np.asarray(rows, dtype=np.float32)

    def spec(self):
        return {"encoder": "synthetic-test-encoder", "frozen": True,
                "fingerprint": "synthetic-test-only", "output_dim": 4,
                "preprocessing": {"purpose": "SYNTHETIC SOFTWARE TEST ONLY"}}


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_study.StudyTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.directory = self.fixture.directory
        self.dataset = self.fixture.dataset

    def test_frozen_encoding_keeps_visit_order_and_encodes_each_visit_once(self):
        encoder = SyntheticEncoder()
        with redirect_stdout(io.StringIO()):
            features, spec = extract_frozen_features(
                self.fixture.visits, GlaBoostConfig(image_batch_size=5), encoder=encoder)
        self.assertEqual(features.shape, (48, 4))
        self.assertEqual(encoder.seen, [v.image for v in self.fixture.visits])
        self.assertTrue(spec["frozen"])
        np.testing.assert_array_equal(features[:3, 1], np.array([1/3, 2/3, 1], dtype=np.float32))

    def test_injected_encoder_requires_explicit_synthetic_marking(self):
        with self.assertRaisesRegex(ValueError, "synthetic"):
            train_grape_report(run_name="blocked", encoder=SyntheticEncoder(), grape_root=self.root)

    def test_unverified_pretrained_checkpoint_is_rejected_before_encoding(self):
        weights = self.directory / "not_imagenet.pth"
        weights.write_bytes(b"Not ImageNet, not allowed before patient splitting")
        with self.assertRaisesRegex(ValueError, "official ImageNet"):
            extract_frozen_features(self.fixture.visits, GlaBoostConfig(image_weights_path=str(weights)))

    def test_cache_torch_symlink_cannot_write_into_raw_data(self):
        cache = self.directory / "encoder_cache"
        cache.mkdir()
        (cache / "torch").symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "outside raw"):
            train_grape_report(run_name="blocked", grape_root=self.root,
                               result_dir=self.directory / "result", artifact_dir=self.directory / "artifacts",
                               model_config=GlaBoostConfig(cache_dir=str(cache)))

    def test_missing_local_weights_marks_failed_and_never_publishes_report(self):
        artifacts = self.directory / "artifacts"
        results = self.directory / "result"
        config = GlaBoostConfig(image_weights_path=str(self.directory / "missing.pth"))
        stderr = io.StringIO()
        with patch("glaboost.training.load_grape", return_value=self.dataset), \
                redirect_stdout(io.StringIO()), redirect_stderr(stderr):
            with self.assertRaises(FileNotFoundError):
                train_grape_report(run_name="missing", grape_root=self.root, result_dir=results,
                                   artifact_dir=artifacts, model_config=config)
        status = json.loads((artifacts / "missing/status.json").read_text())
        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["stage"], "features")
        traceback_path = artifacts / "missing/error_traceback.txt"
        self.assertEqual(status["traceback_path"], str(traceback_path))
        trace = traceback_path.read_text()
        self.assertIn("Traceback (most recent call last)", trace)
        self.assertIn("extract_frozen_features", trace)
        self.assertIn("FileNotFoundError", trace)
        self.assertIn("missing.pth", trace)
        self.assertIn(str(traceback_path), stderr.getvalue())
        self.assertFalse((results / "missing/report.html").exists())

    def test_nested_training_failure_preserves_exception_chain_and_does_not_publish(self):
        artifacts, results = self.directory / "artifacts", self.directory / "result"
        failure = RuntimeError("lazy wrapper should be called at most once")

        def fail_training(*args, **kwargs):
            try:
                raise ValueError("synthetic numerical backend initialization failure")
            except ValueError as cause:
                raise failure from cause

        stderr = io.StringIO()
        with patch("glaboost.training.load_grape", return_value=self.dataset), \
                patch("glaboost.progression.train_progression", side_effect=fail_training), \
                patch("glaboost.reporting.write_report") as write_report, \
                redirect_stdout(io.StringIO()), redirect_stderr(stderr):
            with self.assertRaises(RuntimeError) as raised:
                train_grape_report(run_name="nested-failure", grape_root=self.root, result_dir=results,
                                   artifact_dir=artifacts, model_config=GlaBoostConfig(tree_method="hist", device="cpu"),
                                   encoder=SyntheticEncoder(), synthetic=True)
        self.assertIs(raised.exception, failure)
        write_report.assert_not_called()
        status = json.loads((artifacts / "nested-failure/status.json").read_text())
        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["stage"], "nested_training")
        self.assertEqual(status["error"], "RuntimeError: lazy wrapper should be called at most once")
        traceback_path = artifacts / "nested-failure/error_traceback.txt"
        self.assertEqual(status["traceback_path"], str(traceback_path))
        trace = traceback_path.read_text()
        self.assertIn("train_grape_report", trace)
        self.assertIn("fail_training", trace)
        self.assertIn("ValueError: synthetic numerical backend initialization failure", trace)
        self.assertIn("direct cause", trace)
        self.assertIn("RuntimeError: lazy wrapper should be called at most once", trace)
        self.assertIn(str(traceback_path), stderr.getvalue())
        self.assertFalse((results / "nested-failure").exists())
        self.assertFalse((results / "INDEX.md").exists())

    def test_failure_diagnostic_write_error_cannot_mask_original_exception(self):
        artifacts, results = self.directory / "artifacts", self.directory / "result"
        failure = RuntimeError("synthetic original training failure")
        path_open = Path.open

        def fail_traceback_write(path, *args, **kwargs):
            if path.name == "error_traceback.txt":
                raise OSError("synthetic unavailable diagnostic file")
            return path_open(path, *args, **kwargs)

        stderr = io.StringIO()
        with patch("glaboost.training.load_grape", return_value=self.dataset), \
                patch("glaboost.progression.train_progression", side_effect=failure), \
                patch.object(Path, "open", fail_traceback_write), \
                redirect_stdout(io.StringIO()), redirect_stderr(stderr):
            with self.assertRaises(RuntimeError) as raised:
                train_grape_report(run_name="diagnostic-failure", grape_root=self.root, result_dir=results,
                                   artifact_dir=artifacts, model_config=GlaBoostConfig(tree_method="hist", device="cpu"),
                                   encoder=SyntheticEncoder(), synthetic=True)
        self.assertIs(raised.exception, failure)
        status = json.loads((artifacts / "diagnostic-failure/status.json").read_text())
        self.assertEqual(status["status"], "failed")
        self.assertNotIn("traceback_path", status)
        self.assertIn("synthetic unavailable diagnostic file", status["traceback_write_error"])
        self.assertIn("Could not save error traceback", stderr.getvalue())
        self.assertFalse((results / "diagnostic-failure").exists())

    def test_cli_training_dispatch_no_diagnosis_model_required(self):
        from glaboost.cli import main
        with patch("glaboost.training.train_grape_report", return_value=self.directory) as train, \
                patch("glaboost.cli.GlaBoost.load", side_effect=AssertionError("Must not load diagnosis weights")), \
                patch("glaboost.encoders.resolve_image_devices", return_value=("cuda:0", (0, 1))), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(main(["train-grape", "--run-name", "primary", "--allow-download"]), 0)
        args = train.call_args.kwargs
        self.assertTrue(args["allow_download"])
        self.assertEqual(args["inner_splits"], 2)
        self.assertEqual(args["model_config"].n_estimators, 100)
        self.assertEqual(args["model_config"].max_depth, 6)
        self.assertEqual(args["model_config"].learning_rate, 0.05)
        self.assertEqual(args["model_config"].device, "cuda")
        self.assertEqual(args["model_config"].image_batch_size, 128)
        self.assertEqual(args["model_config"].n_jobs, 1)
        self.assertEqual(args["model_config"].tree_method, "gpu_hist")
        self.assertEqual(args["model_config"].gpu_id, 0)
        self.assertEqual(args["evaluation_config"].compute_device, "cuda:0")

    def test_cli_resource_defaults_scale_global_batch_with_selected_gpus(self):
        from glaboost.cli import _execution_settings
        for ids in ((0,), (0, 1, 2, 3), (2,)):
            with patch("glaboost.encoders.resolve_image_devices", return_value=(f"cuda:{ids[0]}", ids)):
                result = _execution_settings("cuda")
                self.assertEqual(result[2:], (64 * len(ids), 1))
                self.assertEqual(_execution_settings("cuda", 7, 3)[2:], (7, 3))
                with self.assertRaises(ValueError):
                    _execution_settings("cuda", 0)

    def test_small_native_training_to_report_end_to_end(self):
        encoder = SyntheticEncoder()
        results, artifacts = self.directory / "result", self.directory / "artifacts"
        raw_before = (self.root / "files/VF and clinical information.xlsx").read_bytes()
        with patch("glaboost.training.load_grape", return_value=self.dataset), redirect_stdout(io.StringIO()):
            report = train_grape_report(
                run_name="synthetic-training", grape_root=self.root, result_dir=results,
                artifact_dir=artifacts, model_config=GlaBoostConfig(n_estimators=2, max_depth=2),
                evaluation_config=EvaluationConfig(bootstrap_replicates=20),
                encoder=encoder, synthetic=True)
        self.assertEqual(len(encoder.seen), 48)
        self.assertEqual(raw_before, (self.root / "files/VF and clinical information.xlsx").read_bytes())
        self.assertTrue((report / "report.html").is_file())
        self.assertIn("SYNTHETIC", (report / "report.md").read_text())
        provenance = json.loads((report / "provenance.json").read_text())
        self.assertEqual(provenance["validation_design"], "internal_nested_patient_cv")
        self.assertEqual(json.loads((report / "cohort.json").read_text())["n_eyes"], 16)
        evaluation = json.loads((report / "evaluation.json").read_text())
        self.assertEqual(set(evaluation["endpoints"]), {"plr2", "plr3", "md_slope"})
        self.assertTrue(all(r["status"] == "ok" for r in evaluation["endpoints"].values()))
        self.assertEqual(json.loads((report / "status.json").read_text())["status"], "complete")
        native = list((artifacts / "synthetic-training/models").glob("*/outer_*/outer_base/model.json"))
        self.assertEqual(len(native), 9)
        with (report / "visit_predictions.csv").open(newline="") as handle:
            reader = csv.DictReader(handle)
            self.assertIn("progression_score", reader.fieldnames)
            self.assertNotIn("glaucoma_score", reader.fieldnames)
            rows = list(reader)
        outer = [row for row in rows if row["role"] == "outer_test"]
        self.assertEqual(len(outer), 48 * 3)
        self.assertEqual(len({(r["endpoint"], r["sample_id"]) for r in outer}), 48 * 3)
        self.assertTrue(all(row["outer_fold"] != "" for row in outer))
        with self.assertRaises(FileExistsError):
            train_grape_report(run_name="synthetic-training", grape_root=self.root, result_dir=results,
                               artifact_dir=artifacts, encoder=SyntheticEncoder(), synthetic=True)


if __name__ == "__main__":
    unittest.main()
