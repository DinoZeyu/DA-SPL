"""Source-to-GRAPE integration on synthetic features only, entirely in /tmp."""

import csv
import io
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import test_study
from test_external import SyntheticImageEncoder
from glaboost.data import VisitInput
from glaboost.config import GlaBoostConfig
from glaboost.hf_training import _FeatureTableEncoder, _fit_device_group, _load_training_plan, run_hf_grape
from glaboost.model import GlaBoost
from glaboost.study import sha256_file


class HFTrainingTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_study.StudyTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.directory, self.grape = self.fixture.directory, self.fixture.root
        self.source = self.directory / "source_raw"
        self.source.mkdir()
        self.results, self.artifacts = self.directory / "results", self.directory / "artifacts"
        self.checkpoint = self.directory / "synthetic_encoder.pth"
        self.checkpoint.write_bytes(b"SYNTHETIC encoder; not pretrained weights")
        self.encoder = SyntheticImageEncoder(self.checkpoint)
        self.train = tuple(VisitInput(sample_id=f"source_train_{i}", image=f"{i}_OD_1.png") for i in range(12))
        self.test = tuple(VisitInput(sample_id=f"source_test_{i}", image=f"{i}_OS_2.png") for i in range(12, 20))
        self.plan_path = self.directory / "training.json"
        self.plan = {"format_version": 1, "source": {"repo_id": "synthetic/fixture", "revision": "a" * 40},
                     "primary_model": "primary", "models": [{"name": "primary", "n_estimators": 2, "max_depth": 2},
                                                               {"name": "alternate", "n_estimators": 3, "max_depth": 2}],
                     "evaluation": {"n_splits": 3, "seed": 42, "bootstrap_replicates": 20}}
        self.dataset = SimpleNamespace(train_visits=self.train, test_visits=self.test,
                    train_labels=np.arange(12) % 2, test_labels=np.arange(12, 20) % 2,
                    audit={**self.plan["source"], "synthetic": True,
                           "splits": {name: {"released_rows": n, "retained_rows": n,
                                            "within_split_duplicates_removed": 0,
                                            "train_overlap_duplicates_removed": 0}
                                      for name, n in (("train", 12), ("test", 8))}})
        self.original_load, self.original_fit = GlaBoost.load, GlaBoost.fit

    def run_pipeline(self, **overrides):
        self.plan_path.write_text(json.dumps(self.plan), encoding="utf-8")
        options = dict(training_plan_path=self.plan_path, run_name="synthetic_hf", hf_root=self.source,
                       grape_root=self.grape, result_dir=self.results, artifact_dir=self.artifacts,
                       cache_dir=self.directory / "cache", image_weights=str(self.checkpoint),
                       device="cpu", image_batch_size=7, synthetic=True)
        options.update(overrides)
        with ExitStack() as stack:
            stack.enter_context(redirect_stderr(io.StringIO()))
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(patch("glaboost.hf_training.load_hf_diagnosis", return_value=self.dataset))
            stack.enter_context(patch("glaboost.hf_training.ResNet152Encoder", return_value=self.encoder))
            stack.enter_context(patch("glaboost.external.load_grape", return_value=self.fixture.dataset))
            stack.enter_context(patch("glaboost.study.load_grape", return_value=self.fixture.dataset))
            stack.enter_context(patch("glaboost.external.GlaBoost.load", side_effect=lambda *args, **kw:
                                      self.original_load(*args, image_encoder=self.encoder, **kw)))
            return run_hf_grape(**options)

    def test_complete_source_training_to_fixed_grape_reports(self):
        with patch.object(GlaBoost, "fit", autospec=True, side_effect=self.original_fit) as fit, \
                patch("glaboost.external._refresh_project_readme") as readme:
            report = self.run_pipeline()
        readme.assert_not_called()
        self.assertEqual(fit.call_count, 2)
        for call in fit.call_args_list:
            self.assertEqual([v.sample_id for v in call.args[1]], [v.sample_id for v in self.train])
            np.testing.assert_array_equal(call.args[2], self.dataset.train_labels)
            self.assertTrue(all(v.patient_id is None and v.text is None and not v.human and not v.structured
                                for v in call.args[1]))
        source_report = self.results / "synthetic_hf_source"
        source_artifacts = self.artifacts / "synthetic_hf_source"
        source_status = json.loads((source_report / "status.json").read_text())
        self.assertEqual(source_status["status"], "complete")
        self.assertTrue(source_status["source_training_complete"])
        self.assertEqual(source_status["grape_report"], str(report))
        self.assertEqual(json.loads((report / "status.json").read_text())["status"], "complete")
        plan = json.loads((source_report / "external_models.json").read_text())
        self.assertEqual(plan["primary_model"], "primary")
        for name in ("primary", "alternate"):
            metadata = json.loads((source_artifacts / name / "metadata.json").read_text())
            summary = metadata["training_summary"]
            self.assertEqual(summary["n_visits"], 12)
            self.assertEqual(summary["n_glaucoma"], 6)
            self.assertEqual(summary["source_split"], "train")
            self.assertFalse(summary["test_used_for_selection"])
            self.assertFalse(summary["grape_used_for_training_or_selection"])
            self.assertEqual(summary["source_label_mapping"], {"0": 1, "1": 0})
            self.assertEqual(metadata["encoder_specs"]["image"], self.encoder.spec())
            self.assertEqual(metadata["target"], {"0": "normal", "1": "glaucoma"})
            self.assertEqual(summary["feature_table_sha256"], sha256_file(source_artifacts / "resnet152_features.npy"))
            with (source_report / f"{name}_test_predictions.csv").open() as handle:
                predictions = list(csv.DictReader(handle))
            self.assertEqual([r["sample_id"] for r in predictions], [v.sample_id for v in self.test])
            self.assertTrue((report / name / "report.html").is_file())
        self.assertEqual(np.load(source_artifacts / "resnet152_features.npy").shape, (20, 2048))
        self.assertFalse((source_report / "resnet152_features.npy").exists())
        with (source_report / "diagnosis_metrics.csv").open() as handle:
            metrics = list(csv.DictReader(handle))
        self.assertEqual([r["role"] for r in metrics], ["Primary", "Exploratory"])
        self.assertTrue(all(row["n_test"] == "8" and row["synthetic"] == "True" for row in metrics))
        self.assertIn("not GRAPE progression", (source_report / "report.md").read_text())
        self.assertIn("12 released, 12 retained", (source_report / "report.md").read_text())
        self.assertIn("../synthetic_hf_source/report.html", (report / "report.html").read_text())
        self.assertEqual((source_report / "submitted_training_plan.json").read_bytes(), self.plan_path.read_bytes())
        _, replayable = _load_training_plan(source_report / "submitted_training_plan.json")
        self.assertEqual(replayable["primary_model"], "primary")
        self.assertIn("synthetic_hf_source/report.html", (self.results / "INDEX.md").read_text())
        with self.assertRaises(FileExistsError):
            self.run_pipeline()

    def test_source_revision_mismatch_precedes_devices_outputs_and_training(self):
        self.dataset.audit["revision"] = "b" * 40
        with patch("glaboost.hf_training.resolve_image_devices") as devices, patch.object(GlaBoost, "fit") as fit:
            with self.assertRaisesRegex(ValueError, "identity/revision"):
                self.run_pipeline()
        devices.assert_not_called()
        fit.assert_not_called()
        self.assertFalse(self.results.exists())
        self.assertFalse(self.artifacts.exists())

    def test_source_and_grape_raw_output_paths_are_rejected(self):
        alias = self.directory / "source_alias"
        alias.symlink_to(self.source, target_is_directory=True)
        for options in ({"result_dir": self.source / "report"}, {"artifact_dir": alias / "models"},
                        {"cache_dir": self.source}, {"artifact_dir": self.grape / "models"}):
            with self.subTest(options=options), patch.object(GlaBoost, "fit") as fit, self.assertRaises(ValueError):
                self.run_pipeline(**options)
            fit.assert_not_called()
        self.assertEqual(list(self.source.iterdir()), [])

    def test_failed_training_is_recorded_and_does_not_start_grape(self):
        with patch("glaboost.hf_training._fit_device_group", side_effect=RuntimeError("SYNTHETIC fit failure")), \
                patch("glaboost.hf_training.run_external_validation") as external:
            with self.assertRaisesRegex(RuntimeError, "SYNTHETIC fit failure"):
                self.run_pipeline()
        external.assert_not_called()
        status = json.loads((self.results / "synthetic_hf_source/status.json").read_text())
        self.assertEqual((status["status"], status["stage"]), ("failed", "source_tree_training"))
        self.assertFalse((self.results / "synthetic_hf").exists())
        self.assertTrue((self.artifacts / "synthetic_hf_source/error_traceback.txt").is_file())

    def test_source_plan_rejects_unapproved_modalities_selection_and_bad_primary(self):
        original = deepcopy(self.plan)
        mutations = [lambda p: p["models"][0].update(use_text=True), lambda p: p.update(primary_model="best"),
                     lambda p: p.update(select_best=True), lambda p: p["source"].update(revision="main"),
                     lambda p: p["evaluation"].update(compute_device="cpu"),
                     lambda p: p["models"][0].update(n_estimators=0)]
        for mutate in mutations:
            plan = deepcopy(original)
            mutate(plan)
            self.plan_path.write_text(json.dumps(plan))
            with self.subTest(plan=plan), self.assertRaises(ValueError):
                _load_training_plan(self.plan_path)

    def test_feature_indices_never_become_predictors(self):
        features = self.encoder.transform([v.image for v in self.train])
        lookup = _FeatureTableEncoder(features, self.encoder.spec())
        np.testing.assert_array_equal(lookup.transform([11, 1, 3]), features[[11, 1, 3]])
        for rows in ([-1], [len(features)], [1.5], [True], ["1"]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                lookup.transform(rows)

    def test_real_spawn_workers_can_save_native_synthetic_models(self):
        """Exercise serialization/imports across real processes, without CUDA/data."""
        features = self.encoder.transform([v.image for v in self.train + self.test])
        feature_path = self.directory / "synthetic_features.npy"
        np.save(feature_path, features, allow_pickle=False)
        payload = {"feature_path": str(feature_path), "feature_sha256": sha256_file(feature_path),
                   "encoder_spec": self.encoder.spec(), "train_ids": [v.sample_id for v in self.train],
                   "test_ids": [v.sample_id for v in self.test], "train_labels": self.dataset.train_labels.tolist(),
                   "source": self.plan["source"], "audit_sha256": "a" * 64,
                   "artifact_path": str(self.artifacts)}
        futures = []
        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
            for name in ("worker_a", "worker_b"):
                config = GlaBoostConfig.for_image_method(n_estimators=2, max_depth=2,
                                image_weights_path=str(self.checkpoint), device="cpu", tree_method="hist")
                futures.append(pool.submit(_fit_device_group, {**payload, "models": [{"name": name,
                                                                                      "config": config.to_dict()}]}))
            results = [future.result(timeout=60)[0] for future in futures]
        for result in results:
            directory = Path(result["model_directory"])
            restored = self.original_load(directory, device="cpu", image_encoder=self.encoder)
            np.testing.assert_array_equal(restored.predict_score(self.test), result["probabilities"])
        self.assertEqual(results[0]["probabilities"], results[1]["probabilities"])


if __name__ == "__main__":
    unittest.main()
