"""Offline fixed-detector orchestration tests using tiny synthetic models."""

import csv
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import test_study
from glaboost.config import GlaBoostConfig
from glaboost.external import _analysis_signature, _write_summary, run_external_validation
from glaboost.longitudinal import ENDPOINTS
from glaboost.model import GlaBoost
from glaboost.study import sha256_file


class SyntheticImageEncoder:
    output_dim = 2048

    def __init__(self, checkpoint):
        self.checkpoint = checkpoint

    def transform(self, images):
        matrix = np.zeros((len(images), self.output_dim), dtype=np.float32)
        for index, image in enumerate(images):
            patient, side, visit = Path(image).stem.split("_")
            matrix[index, :4] = [int(patient) / 8, int(visit) / 3, side == "OS", np.sin(int(patient) + int(visit))]
        return matrix

    def spec(self):
        digest = sha256_file(self.checkpoint)
        return {"encoder": "resnet152", "output_dim": self.output_dim, "frozen": True,
                "fingerprint": "sha256:" + digest, "source": {"sha256": digest},
                "preprocessing": {"purpose": "SYNTHETIC SOFTWARE TEST ONLY; no image encoding"}}


class ExternalValidationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_study.StudyTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.directory, self.root = self.fixture.directory, self.fixture.root
        self.plan_path = self.directory / "plan.json"
        self.results, self.artifacts = self.directory / "result", self.directory / "artifacts"
        self.checkpoint = self.directory / "synthetic_frozen_encoder.pth"
        self.checkpoint.write_bytes(b"SYNTHETIC encoder fingerprint fixture; not neural weights")
        self.encoder = SyntheticImageEncoder(self.checkpoint)
        self.models = []
        for name, trees in (("primary", 2), ("alternate", 3)):
            destination = self.directory / ("diagnostic_" + name)
            config = GlaBoostConfig(image_encoder="resnet152", n_estimators=trees, max_depth=2,
                                   image_weights_path=str(self.checkpoint), cache_dir=str(self.directory / "cache"))
            model = GlaBoost(config, image_encoder=self.encoder)
            visits = self.fixture.visits[:12]
            model.fit(visits, [int(v.patient_id) % 2 for v in visits])
            model.save(destination)
            self.models.append(destination)
        self.plan = {"format_version": 1, "primary_model": "primary", "min_visits": 3,
                     "evaluation": {"n_splits": 3, "seed": 42, "bootstrap_replicates": 20,
                                    "persistence_threshold": .5, "logistic_c": 1.0},
                     "models": [{"name": name, "model_directory": path.name,
                                 "training_data": {"description": "SYNTHETIC unrelated diagnostic fixture",
                                                   "reference": "SYNTHETIC software test fixture",
                                                   "grape_overlap": "none",
                                                   "independence_evidence": "SYNTHETIC declaration only; no clinical claim"}}
                                for name, path in zip(("primary", "alternate"), self.models)]}
        self.original_load = GlaBoost.load

    def write_plan(self):
        self.plan_path.write_text(json.dumps(self.plan), encoding="utf-8")

    def run_study(self, **overrides):
        self.write_plan()
        kwargs = dict(plan_path=self.plan_path, run_name="synthetic_external", grape_root=self.root,
                      result_dir=self.results, artifact_dir=self.artifacts, device="cpu",
                      cache_dir=self.directory / "cache", image_batch_size=9, synthetic=True)
        kwargs.update(overrides)
        with ExitStack() as stack:
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(redirect_stderr(io.StringIO()))
            stack.enter_context(patch("glaboost.external.load_grape", return_value=self.fixture.dataset))
            stack.enter_context(patch("glaboost.study.load_grape", return_value=self.fixture.dataset))
            stack.enter_context(patch("glaboost.external.GlaBoost.load", side_effect=lambda *args, **kw:
                                      self.original_load(*args, image_encoder=self.encoder, **kw)))
            return run_external_validation(**kwargs)

    def assert_no_outputs(self):
        self.assertFalse(self.results.exists())
        self.assertFalse(self.artifacts.exists())

    def test_empty_or_undocumented_plans_fail_before_devices_or_output_creation(self):
        valid = deepcopy(self.plan)
        mutations = [
            lambda p: p.update(models=[]),
            lambda p: p.update(primary_model="missing"),
            lambda p: p["models"][0]["training_data"].update(grape_overlap="unknown"),
            lambda p: p["models"][0]["training_data"].update(grape_overlap="present"),
            lambda p: p["models"][0]["training_data"].update(independence_evidence="TODO"),
            lambda p: p["models"][0]["training_data"].pop("reference"),
            lambda p: p["models"][1].update(name="primary"),
        ]
        with patch("glaboost.external.resolve_image_devices") as devices:
            for mutate in mutations:
                self.plan = deepcopy(valid)
                mutate(self.plan)
                with self.subTest(plan=self.plan), self.assertRaises(ValueError):
                    self.run_study()
                self.assert_no_outputs()
        devices.assert_not_called()

    def test_all_models_preflight_before_scoring_or_outputs(self):
        metadata_path = self.models[1] / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["target"] = {"0": "nonprogression", "1": "progression"}
        metadata_path.write_text(json.dumps(metadata))
        with patch("glaboost.external.resolve_image_devices") as devices, self.assertRaisesRegex(ValueError, "diagnostic"):
            self.run_study()
        devices.assert_not_called()
        self.assert_no_outputs()

    def test_corrupted_classifier_or_encoder_checkpoint_blocks_preflight(self):
        model_path = self.models[1] / "model.json"
        original = model_path.read_bytes()
        model_path.write_bytes(original + b" ")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.run_study()
        self.assert_no_outputs()
        model_path.write_bytes(original)
        self.checkpoint.write_bytes(b"CHANGED SYNTHETIC encoder fixture")
        with self.assertRaisesRegex(ValueError, "checkpoint differs"):
            self.run_study()
        self.assert_no_outputs()

    def test_nonfrozen_or_multimodal_or_wrong_schema_detectors_are_rejected(self):
        path = self.models[0] / "metadata.json"
        original = json.loads(path.read_text())
        mutations = [lambda m: m["encoder_specs"]["image"].update(frozen=False),
                     lambda m: m["config"].update(use_text=True),
                     lambda m: m["config"].update(image_encoder="resnet18"),
                     lambda m: m.update(feature_names=["wrong_feature"])]
        for mutate in mutations:
            metadata = deepcopy(original)
            mutate(metadata)
            path.write_text(json.dumps(metadata))
            with self.subTest(metadata=metadata["config"]), self.assertRaises(ValueError):
                self.run_study()
            self.assert_no_outputs()

    def test_complete_batch_preserves_primary_scores_models_audits_and_code_snapshot(self):
        before = {(path, name): sha256_file(path / name) for path in self.models for name in ("metadata.json", "model.json")}
        with patch("glaboost.external._refresh_project_readme") as readme:
            report = self.run_study()
        readme.assert_not_called()
        status = json.loads((report / "status.json").read_text())
        self.assertEqual(status["status"], "complete")
        self.assertEqual(status["completed_models"], ["primary", "alternate"])
        self.assertEqual(status["primary_model"], "primary")
        self.assertTrue(status["synthetic"])
        with (report / "comparison.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 6)
        self.assertEqual({row["role"] for row in rows if row["model"] == "primary"}, {"Primary"})
        self.assertEqual({row["role"] for row in rows if row["model"] == "alternate"}, {"Exploratory"})
        self.assertEqual({row["n_estimators"] for row in rows if row["model"] == "primary"}, {"2"})
        self.assertEqual({row["n_estimators"] for row in rows if row["model"] == "alternate"}, {"3"})
        self.assertTrue(all(row["synthetic"] == "True" for row in rows))
        self.assertIn("SYNTHETIC SOFTWARE TEST", (report / "report.md").read_text())
        self.assertIn("| primary | Primary | 2 | 2 | 0.05 | 1.0 | 1.0 |", (report / "report.md").read_text())
        saved_plan = json.loads((report / "plan.json").read_text())
        self.assertEqual(saved_plan["models"][0]["model_config"]["n_estimators"], 2)
        self.assertEqual(saved_plan["models"][1]["model_config"]["n_estimators"], 3)
        self.assertIn("not independent external validation", (report / "report.md").read_text())
        self.assertIn("primary/report.html", (report / "report.html").read_text())
        audit = json.loads((report / "comparison_audit.json").read_text())
        self.assertTrue(audit["comparability_verified"])
        manifest = json.loads((report / "code/manifest.json").read_text())["sha256"]
        self.assertIn("src/glaboost/external.py", manifest)
        for relative, digest in manifest.items():
            self.assertEqual(sha256_file(report / "code" / relative), digest)
        for name in ("primary", "alternate"):
            artifacts = self.artifacts / report.name / name
            self.assertTrue((artifacts / "progression_heads.json").exists())
            sidecar = json.loads((artifacts / "visit_scores.metadata.json").read_text())
            self.assertEqual(sidecar["detector_training_data"]["independence_evidence"],
                             self.plan["models"][0]["training_data"]["independence_evidence"])
            self.assertEqual(sha256_file(artifacts / "visit_scores.csv"), sidecar["scores_sha256"])
        for (path, name), digest in before.items():
            self.assertEqual(sha256_file(path / name), digest)
        with self.assertRaises(FileExistsError):
            self.run_study()

    def test_failed_scoring_marks_batch_failed_without_summary_or_next_model(self):
        error = RuntimeError("SYNTHETIC scoring failure")
        with patch.object(GlaBoost, "predict_score", side_effect=error) as score:
            with self.assertRaises(RuntimeError) as raised:
                self.run_study()
        self.assertIs(raised.exception, error)
        score.assert_called_once()
        report = self.results / "synthetic_external"
        status = json.loads((report / "status.json").read_text())
        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["completed_models"], [])
        self.assertFalse((report / "report.html").exists())
        self.assertFalse((self.artifacts / report.name / "alternate").exists())
        self.assertIn("SYNTHETIC scoring failure", Path(status["traceback_path"]).read_text())


class ExternalAuditTests(unittest.TestCase):
    def test_nonestimable_outcomes_are_retained_without_zero_imputation(self):
        from glaboost.reporting import _primary_rows, _supplementary_rows

        evaluation = {"config": {}, "predictions": [], "endpoints": {
            endpoint: {"status": "not_estimable", "reason": "SYNTHETIC insufficient positive patients",
                       "n_eyes": 4, "n_patients": 2, "n_positive_eyes": 0, "n_splits": 0, "folds": []}
            for endpoint in ENDPOINTS}}
        cohort = {"n_eyes": 4, "n_patients": 2, "n_visits": 12}
        signature = _analysis_signature(evaluation, cohort, {}, [])
        self.assertEqual(set(signature["folds"]), set(ENDPOINTS))
        rows = [{"model": "fixed", "role": "Primary", **row} for row in _primary_rows(evaluation)]
        supplementary = [{"model": "fixed", "role": "Primary", **row}
                         for row in _supplementary_rows(evaluation)]
        plan = {"synthetic": True, "primary_model": "fixed", "models": [
            {"name": "fixed", "model_config": GlaBoostConfig().to_dict()}]}
        with tempfile.TemporaryDirectory(prefix="synthetic_external_summary_") as directory:
            report = Path(directory)
            _write_summary(report, plan, rows, supplementary, cohort)
            with (report / "comparison.csv").open() as handle:
                saved = list(csv.DictReader(handle))
            self.assertEqual(len(saved), 3)
            self.assertTrue(all(row["latest_estimate"] == row["delta_estimate"] == "" for row in saved))
            self.assertIn("Not estimable", (report / "report.html").read_text())

    def test_signature_ignores_fitted_coefficients_but_detects_different_folds(self):
        fold = {"fold": 0, "train_patient_ids": ["train"], "test_patient_ids": ["test"],
                "latest": {"feature_names": ["last"], "coefficients": [1]},
                "longitudinal": {"feature_names": ["last", "delta"], "coefficients": [1, 2]}}
        evaluation = {"config": {}, "predictions": [], "endpoints": {
            endpoint: {"status": "ok", "folds": [deepcopy(fold)]} for endpoint in ENDPOINTS}}
        reference = _analysis_signature(evaluation, {}, {}, [])
        evaluation["endpoints"]["plr2"]["folds"][0]["latest"]["coefficients"] = [999]
        self.assertEqual(reference, _analysis_signature(evaluation, {}, {}, []))
        evaluation["endpoints"]["plr2"]["folds"][0]["test_patient_ids"] = ["another_test"]
        self.assertNotEqual(reference, _analysis_signature(evaluation, {}, {}, []))
        evaluation["endpoints"]["plr2"]["folds"][0]["test_patient_ids"] = ["train"]
        with self.assertRaisesRegex(ValueError, "Patient overlap"):
            _analysis_signature(evaluation, {}, {}, [])


if __name__ == "__main__":
    unittest.main()
