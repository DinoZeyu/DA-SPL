"""Nested leakage checks on synthetic image-feature matrices; no GRAPE fitting."""

import copy
from contextlib import contextmanager, nullcontext
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
from tqdm import tqdm

from glaboost.config import GlaBoostConfig
from glaboost.data import VisitInput
from glaboost.longitudinal import ENDPOINTS, EvaluationConfig, temporal_features
from glaboost.longitudinal import _fit_fold, _paired_bootstrap
from glaboost.model import make_xgb_classifier
from glaboost.progression import train_progression


def cohort(n_patients=18):
    visits, features, labels = [], [], {}
    for patient in range(n_patients):
        for side in range(2):
            eye = f"p{patient:02d}_{side}"
            labels[eye] = {"plr2": patient % 2, "plr3": (patient // 2) % 2,
                           "md_slope": (patient // 3) % 2}
            times = (0.0, 0.5, 2.0, 3.5) if side == 1 and patient % 3 == 0 else (0.0, 0.5, 2.0)
            for visit, time in enumerate(times):
                visits.append(VisitInput(f"{eye}_{visit}", image="unused.jpg",
                                         patient_id=f"p{patient:02d}", eye_id=eye, time_years=time))
                # First synthetic feature identifies the patient solely so the
                # spy can detect leakage; actual inputs are frozen image vectors.
                features.append([patient, (patient % 2) * 0.5 + side * 0.1, time])
    return visits, np.array(features, dtype=np.float32), labels


class LeakageSpyClassifier:
    fitted = []

    def fit(self, x, y, sample_weight):
        self.x, self.y, self.weights = x.copy(), y.copy(), sample_weight.copy()
        self.patients = set(x[:, 0])
        self.fitted.append(self)
        return self

    def predict_proba(self, x):
        if self.patients & set(x[:, 0]):
            raise AssertionError("An in-sample patient's visit was used for assessment/head fitting")
        self.test_x = x.copy()
        p = np.clip(0.2 + x[:, 1] * 0.7 + x[:, 2] * 0.025, 0.01, 0.99)
        return np.column_stack((1 - p, p))

    def save_model(self, path):
        Path(path).write_text('{"synthetic_spy": true}', encoding="utf-8")


class ProgressionTests(unittest.TestCase):
    def setUp(self):
        LeakageSpyClassifier.fitted = []
        self.config = EvaluationConfig(n_splits=3, bootstrap_replicates=20)

    def run_spy(self, data=None, **kwargs):
        visits, features, labels = cohort() if data is None else data
        with patch("glaboost.progression.make_xgb_classifier", side_effect=lambda config: LeakageSpyClassifier()):
            return train_progression(visits, features, labels, evaluation_config=self.config, **kwargs)

    def test_nested_patient_exclusion_and_complete_paired_predictions(self):
        visits, features, labels = cohort()
        result = self.run_spy((visits, features, labels))
        json.dumps(result, allow_nan=False)
        self.assertEqual(len(LeakageSpyClassifier.fitted), 27)
        self.assertEqual(len(result["predictions"]), 3 * len(labels))
        for endpoint, summary in result["endpoints"].items():
            self.assertEqual(summary["status"], "ok")
            self.assertEqual(summary["n_splits"], 3)
            all_heldout = []
            for fold in summary["folds"]:
                outer_train, outer_test = set(fold["train_patient_ids"]), set(fold["test_patient_ids"])
                self.assertFalse(outer_train & outer_test)
                all_heldout.extend(fold["test_eye_ids"])
                self.assertEqual(set(fold["outer_base"]["train_patient_ids"]), outer_train)
                self.assertEqual(set(fold["outer_base"]["test_patient_ids"]), outer_test)
                inner_heldout = []
                for inner in fold["inner_models"]:
                    inner_train, inner_test = set(inner["train_patient_ids"]), set(inner["test_patient_ids"])
                    self.assertFalse(inner_train & inner_test)
                    self.assertFalse((inner_train | inner_test) & outer_test)
                    self.assertEqual(inner_train | inner_test, outer_train)
                    inner_heldout.extend(inner["test_eye_ids"])
                self.assertCountEqual(inner_heldout, fold["train_eye_ids"])
                self.assertEqual(fold["latest"]["feature_names"], ["last"])
                self.assertEqual(len(fold["longitudinal"]["feature_names"]), 5)
            self.assertCountEqual(all_heldout, labels)
            rows = [row for row in result["visit_predictions"] if row["endpoint"] == endpoint and row["role"] == "outer_test"]
            self.assertCountEqual([row["sample_id"] for row in rows], [visit.sample_id for visit in visits])
        self.assertNotIn("glaucoma_score", json.dumps(result))

    def test_training_progress_identifies_endpoint_fold_and_model_role(self):
        output = io.StringIO()
        with patch("glaboost.progression.tqdm", side_effect=lambda *args, **kwargs:
                   tqdm(*args, **kwargs, file=output, disable=False, mininterval=0)):
            result = self.run_spy()
        text = output.getvalue()
        self.assertIn("plr2: outer patient folds", text)
        self.assertIn("fold 1: inner model 1/2", text)
        self.assertIn("fold 1: outer base model", text)
        self.assertIn("fold 1: latest head", text)
        self.assertIn("fold 1: longitudinal head", text)
        self.assertEqual(len(LeakageSpyClassifier.fitted), 27)
        self.assertTrue(all(row["status"] == "ok" for row in result["endpoints"].values()))

    def test_heads_are_scaled_on_inner_oof_features_only(self):
        result = self.run_spy()
        for endpoint, summary in result["endpoints"].items():
            for fold in summary["folds"]:
                rows = [row for row in result["visit_predictions"]
                        if row["endpoint"] == endpoint and row["outer_fold"] == fold["fold"]
                        and row["role"] == "inner_oof"]
                grouped = {}
                for row in rows:
                    grouped.setdefault(row["eye_id"], []).append(row)
                records = []
                for eye, visits in sorted(grouped.items()):
                    visits.sort(key=lambda row: row["time_years"])
                    records.append({"eye_id": eye, "patient_id": visits[0]["patient_id"],
                                    "times": [row["time_years"] for row in visits],
                                    "scores": [row["progression_score"] for row in visits],
                                    "labels": dict.fromkeys(ENDPOINTS, 0)})
                expected = temporal_features(records, self.config.persistence_threshold).mean(axis=0)
                np.testing.assert_allclose(fold["latest"]["scaler_mean"], expected[:1])
                np.testing.assert_allclose(fold["longitudinal"]["scaler_mean"], expected)

    def test_inverse_visit_weights_give_equal_total_weight_per_eye(self):
        visits, features, labels = cohort()
        result = self.run_spy((visits, features, labels))
        index = {visit.sample_id: visit for visit in visits}
        # Models fit in inner-then-outer order, with the same order in metadata.
        models = iter(LeakageSpyClassifier.fitted)
        for summary in result["endpoints"].values():
            for fold in summary["folds"]:
                for metadata in fold["inner_models"] + [fold["outer_base"]]:
                    model = next(models)
                    contributions = {}
                    for sample, weight in zip(metadata["train_sample_ids"], model.weights):
                        eye = index[sample].eye_id
                        contributions[eye] = contributions.get(eye, 0) + weight
                    np.testing.assert_allclose(list(contributions.values()), list(contributions.values())[0])
                    self.assertAlmostEqual(float(model.weights.mean()), 1)

    def test_endpoint_specific_weak_labels_not_metadata_as_model_inputs(self):
        visits, features, labels = cohort()
        for visit in visits:
            visit.structured = {"plr2": labels[visit.eye_id]["plr2"], "future_md": 1000}
            visit.text = "future information must never enter image-only fitting"
        result = self.run_spy((visits, features, labels))
        visit_lookup = {visit.sample_id: visit for visit in visits}
        models = iter(LeakageSpyClassifier.fitted)
        for endpoint, summary in result["endpoints"].items():
            for fold in summary["folds"]:
                for metadata in fold["inner_models"] + [fold["outer_base"]]:
                    model = next(models)
                    self.assertEqual(model.x.shape[1], features.shape[1])
                    expected = [labels[visit_lookup[sample].eye_id][endpoint]
                                for sample in metadata["train_sample_ids"]]
                    np.testing.assert_array_equal(model.y, expected)

    def test_unestimable_nested_endpoint_never_fits_partial_models(self):
        visits, features, labels = cohort(12)
        for visit in visits:
            labels[visit.eye_id]["plr3"] = int(visit.patient_id in {"p00", "p01"})
        result = self.run_spy((visits, features, labels))
        summary = result["endpoints"]["plr3"]
        self.assertEqual(summary["status"], "not_estimable")
        self.assertIn("inner", summary["reason"])
        self.assertEqual(summary["folds"], [])
        for collection in ("predictions", "features", "visit_predictions", "eye_records"):
            self.assertFalse(any(row["endpoint"] == "plr3" for row in result[collection]))
        self.assertNotIn("metrics", summary)

    def test_heldout_features_and_trajectories_use_real_elapsed_time(self):
        result = self.run_spy()
        for record in result["eye_records"]:
            feature = next(row for row in result["features"]
                           if row["endpoint"] == record["endpoint"] and row["eye_id"] == record["eye_id"])
            expected = temporal_features([record])[0]
            actual = [feature[key] for key in ("last", "delta", "slope", "mean", "persistence")]
            np.testing.assert_allclose(actual, expected)
            self.assertEqual(record["times"][:3], [0.0, 0.5, 2.0])

    def test_artifacts_record_progression_provenance_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            model_dir = Path(temporary) / "models"
            result = self.run_spy(model_dir=model_dir)
            self.assertEqual(len(list(model_dir.rglob("model.json"))), 27)
            self.assertEqual(len(list(model_dir.rglob("heads.json"))), 9)
            metadata = json.loads(next(model_dir.rglob("metadata.json")).read_text())
            self.assertEqual(metadata["target_mapping"]["1"], "progression")
            self.assertEqual(len(metadata["model_sha256"]), 64)
            self.assertIn("whole-eye", metadata["target"])
            with self.assertRaises(FileExistsError):
                self.run_spy(model_dir=model_dir)
            self.assertIn("internal", result["config"]["study_design"])

    def test_paper_base_defaults_and_small_native_training(self):
        model = make_xgb_classifier(GlaBoostConfig())
        params = model.get_params()
        for name, expected in (("learning_rate", 0.05), ("max_depth", 6), ("n_estimators", 100),
                               ("objective", "binary:logistic"), ("eval_metric", "logloss")):
            self.assertEqual(params[name], expected)
        visits, features, labels = cohort(12)
        for row in labels.values():
            row["plr3"] = row["md_slope"] = 0
        result = train_progression(visits, features, labels, model_config=GlaBoostConfig(n_estimators=3),
                                   evaluation_config=self.config)
        self.assertEqual(result["endpoints"]["plr2"]["status"], "ok")
        self.assertEqual(result["endpoints"]["plr3"]["status"], "not_estimable")
        self.assertTrue(all(0 <= row["latest_probability"] <= 1 for row in result["predictions"]))

    def test_gpu_endpoint_lanes_isolate_devices_and_merge_in_canonical_order(self):
        """Exercise real thread scheduling with mocked GPU work, never CUDA fitting."""
        visits, features, labels = cohort()
        config = GlaBoostConfig(device="cuda", tree_method="gpu_hist", gpu_id=0)
        local = threading.local()
        lock = threading.Lock()
        first_jobs = threading.Barrier(2)
        second_gpu_finished = threading.Event()
        initialized = threading.Event()
        caller_thread = threading.get_ident()
        active, maximum, calls, finished = {}, {}, [], []

        def initialize(gpu_ids):
            self.assertEqual(threading.get_ident(), caller_thread)
            self.assertEqual(gpu_ids, (0, 1))
            self.assertEqual(calls, [])
            initialized.set()

        @contextmanager
        def device_context(gpu_id):
            local.gpu_id = gpu_id
            yield
            del local.gpu_id

        def endpoint_job(endpoint, records, actual_visits, matrix, groups,
                         model_config, evaluation, *, model_dir, inner_splits):
            self.assertTrue(initialized.is_set(), "CUDA lazy loader must finish before workers run")
            gpu_id = model_config.gpu_id
            self.assertEqual(local.gpu_id, gpu_id)
            self.assertEqual(model_config.device, f"cuda:{gpu_id}")
            self.assertEqual(evaluation.compute_device, f"cuda:{gpu_id}")
            self.assertEqual(evaluation.seed, self.config.seed)
            self.assertFalse(matrix.flags.writeable)
            np.testing.assert_array_equal(matrix, features)
            with lock:
                active[gpu_id] = active.get(gpu_id, 0) + 1
                maximum[gpu_id] = max(maximum.get(gpu_id, 0), active[gpu_id])
                calls.append((endpoint, gpu_id, threading.get_ident()))
            if endpoint in ENDPOINTS[:2]:
                first_jobs.wait(timeout=10)
            if endpoint == ENDPOINTS[0]:
                self.assertTrue(second_gpu_finished.wait(timeout=10))
            directory = model_dir / endpoint
            directory.mkdir()
            (directory / "device.txt").write_text(str(gpu_id))
            partial = {"endpoints": {endpoint: {"status": "ok", "compute_device": f"cuda:{gpu_id}"}}}
            partial.update({name: [{"endpoint": endpoint}] for name in
                            ("predictions", "features", "visit_predictions", "eye_records")})
            with lock:
                active[gpu_id] -= 1
                finished.append(endpoint)
            if endpoint == ENDPOINTS[1]:
                second_gpu_finished.set()
            return partial

        with tempfile.TemporaryDirectory() as temporary:
            model_dir = Path(temporary) / "models"
            with patch("glaboost.progression.resolve_image_devices", return_value=("cuda:0", (0, 1))), \
                    patch("glaboost.progression.initialize_cuda_linalg", side_effect=initialize) as warmup, \
                    patch("torch.cuda.device", side_effect=device_context), \
                    patch("glaboost.progression._train_endpoint", side_effect=endpoint_job):
                result = train_progression(visits, features, labels, model_config=config,
                                           evaluation_config=self.config, model_dir=model_dir)
            warmup.assert_called_once_with((0, 1))
            self.assertEqual({p.name for p in model_dir.iterdir()}, set(ENDPOINTS))
        self.assertEqual(finished[0], ENDPOINTS[1])
        self.assertEqual(maximum, {0: 1, 1: 1})
        assignment = {endpoint: gpu_id for endpoint, gpu_id, _ in calls}
        self.assertEqual(assignment, {ENDPOINTS[0]: 0, ENDPOINTS[1]: 1, ENDPOINTS[2]: 0})
        gpu_threads = {gpu_id: {thread for _, device, thread in calls if device == gpu_id}
                       for gpu_id in (0, 1)}
        self.assertTrue(gpu_threads[0].isdisjoint(gpu_threads[1]))
        self.assertEqual(list(result["endpoints"]), list(ENDPOINTS))
        for name in ("predictions", "features", "visit_predictions", "eye_records"):
            self.assertEqual([row["endpoint"] for row in result[name]], list(ENDPOINTS))
        self.assertEqual(result["config"]["gpu_device_ids"], [0, 1])
        self.assertEqual(result["config"]["logistic_solver"], "torch_newton")
        self.assertEqual(config.device, "cuda")
        self.assertEqual(config.gpu_id, 0)
        self.assertEqual(self.config.compute_device, "cpu")
        self.assertTrue(features.flags.writeable)

    def test_cuda_initialization_failure_stops_before_starting_workers(self):
        with patch("glaboost.progression.resolve_image_devices", return_value=("cuda:0", (0, 1))), \
                patch("glaboost.progression.initialize_cuda_linalg", side_effect=RuntimeError("CUDA init failed")), \
                patch("glaboost.progression.ThreadPoolExecutor") as executor, \
                patch("glaboost.progression._train_endpoint") as train:
            with self.assertRaisesRegex(RuntimeError, "CUDA init failed"):
                self.run_spy(model_config=GlaBoostConfig(device="cuda", tree_method="gpu_hist", gpu_id=0))
        executor.assert_not_called()
        train.assert_not_called()

    def test_single_gpu_routes_all_numeric_steps_and_records_actual_backend(self):
        visits, features, labels = cohort()
        devices = {"temporal": [], "heads": [], "bootstrap": []}

        def temporal(records, threshold, *, device):
            devices["temporal"].append(device)
            return temporal_features(records, threshold)

        def heads(train, test, y, config):
            devices["heads"].append(config.compute_device)
            return _fit_fold(train, test, y, replace(config, compute_device="cpu"))

        def bootstrap(y, latest, longitudinal, groups, config, **kwargs):
            devices["bootstrap"].append(config.compute_device)
            return _paired_bootstrap(y, latest, longitudinal, groups,
                                     replace(config, compute_device="cpu"), **kwargs)

        def classifier(config):
            self.assertEqual(config.gpu_id, 1)
            self.assertEqual(config.device, "cuda:1")
            return LeakageSpyClassifier()

        with patch("glaboost.progression.resolve_image_devices", return_value=("cuda:1", (1,))) as resolver, \
                patch("glaboost.progression.initialize_cuda_linalg") as warmup, \
                patch("torch.cuda.device", side_effect=lambda _: nullcontext()), \
                patch("glaboost.progression.make_xgb_classifier", side_effect=classifier), \
                patch("glaboost.progression.assert_xgb_backend") as backend_check, \
                patch("glaboost.progression.temporal_features", side_effect=temporal), \
                patch("glaboost.progression._fit_fold", side_effect=heads), \
                patch("glaboost.progression._paired_bootstrap", side_effect=bootstrap):
            result = train_progression(visits, features, labels,
                                       model_config=GlaBoostConfig(device="cuda:1", tree_method="gpu_hist", gpu_id=1),
                                       evaluation_config=self.config)
        resolver.assert_called_once_with("cuda:1")
        warmup.assert_called_once_with((1,))
        self.assertEqual(backend_check.call_count, 27)
        self.assertEqual(len(devices["heads"]), 18)
        self.assertEqual(len(devices["temporal"]), 18)
        self.assertEqual(len(devices["bootstrap"]), 3)
        for values in devices.values():
            self.assertEqual(set(values), {"cuda:1"})
        for endpoint in ENDPOINTS:
            summary = result["endpoints"][endpoint]
            self.assertEqual(summary["compute_device"], "cuda:1")
            for fold in summary["folds"]:
                for metadata in fold["inner_models"] + [fold["outer_base"]]:
                    self.assertEqual(metadata["hyperparameters"]["gpu_id"], 1)
                    self.assertEqual(metadata["hyperparameters"]["predictor"], "gpu_predictor")
                    self.assertEqual(metadata["hyperparameters"]["tree_method"], "gpu_hist")

    def test_gpu_fallback_is_rejected_before_scoring(self):
        with patch("glaboost.progression.resolve_image_devices", return_value=("cuda:0", (0,))), \
                patch("glaboost.progression.initialize_cuda_linalg"), \
                patch("torch.cuda.device", side_effect=lambda _: nullcontext()), \
                patch("glaboost.progression.assert_xgb_backend", side_effect=RuntimeError("CPU fallback")):
            with self.assertRaisesRegex(RuntimeError, "CPU fallback"):
                self.run_spy(model_config=GlaBoostConfig(device="cuda", tree_method="gpu_hist", gpu_id=0))
        self.assertEqual(len(LeakageSpyClassifier.fitted), 1)
        self.assertFalse(hasattr(LeakageSpyClassifier.fitted[0], "test_x"))

    def test_gpu_hist_requires_selected_cuda_devices(self):
        with patch("glaboost.progression.resolve_image_devices", return_value=("cpu", ())):
            with self.assertRaisesRegex(ValueError, "selected CUDA"):
                self.run_spy(model_config=GlaBoostConfig(device="auto", tree_method="gpu_hist", gpu_id=0))
        self.assertEqual(LeakageSpyClassifier.fitted, [])

    def test_invalid_alignment_identifiers_times_and_targets_fail(self):
        visits, features, labels = cohort()
        bad_cases = []
        bad_cases.append((visits, features[:-1], labels))
        bad = features.copy()
        bad[0, 0] = np.nan
        bad_cases.append((visits, bad, labels))
        bad_visits = copy.deepcopy(visits)
        bad_visits[1].sample_id = bad_visits[0].sample_id
        bad_cases.append((bad_visits, features, labels))
        bad_visits = copy.deepcopy(visits)
        bad_visits[1].time_years = bad_visits[0].time_years
        bad_cases.append((bad_visits, features, labels))
        bad_labels = copy.deepcopy(labels)
        bad_labels[visits[0].eye_id]["plr2"] = True
        bad_cases.append((visits, features, bad_labels))
        for data in bad_cases:
            with self.subTest(case=len(data[0])):
                with self.assertRaises(ValueError):
                    self.run_spy(data)
        for inner in (True, 1, 2.5):
            with self.assertRaises(ValueError):
                self.run_spy(inner_splits=inner)
        with self.assertRaises(ValueError):
            self.run_spy(model_config=GlaBoostConfig(use_structured=True))


if __name__ == "__main__":
    unittest.main()
