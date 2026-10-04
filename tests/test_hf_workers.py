"""Source worker isolation and GPU dispatch using only synthetic arrays and mocks."""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np

from glaboost.config import GlaBoostConfig
from glaboost.data import VisitInput
from glaboost.hf_training import _FeatureTableEncoder, _fit_device_group, run_hf_grape
from glaboost.study import sha256_file


def encoder_spec():
    return {"encoder": "resnet152", "frozen": True, "output_dim": 2048,
            "fingerprint": "synthetic-source-features", "preprocessing": {"synthetic": True},
            "source": {"sha256": "394f9c45" + "0" * 56}}


class MockSourceEncoder:
    checkpoint_filename = "resnet152-394f9c45.pth"
    checkpoint_sha256_prefix = "394f9c45"

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def transform(self, images):
        return np.asarray([np.full(2048, image[0], dtype=np.float32) for image in images])

    def spec(self):
        return encoder_spec()


class SourceWorkerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="synthetic_hf_worker_", dir="/tmp")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.features = np.arange(4 * 2048, dtype=np.float32).reshape(4, 2048)
        self.feature_path = self.root / "features.npy"
        np.save(self.feature_path, self.features, allow_pickle=False)
        self.config = GlaBoostConfig.for_image_method(n_estimators=2, max_depth=1)
        self.payload = {
            "feature_path": str(self.feature_path), "feature_sha256": sha256_file(self.feature_path),
            "encoder_spec": encoder_spec(), "train_ids": ["train-a", "train-b"],
            "test_ids": ["test-a", "test-b"], "train_labels": [0, 1],
            "source": {"repo_id": "synthetic-source", "revision": "0" * 40},
            "audit_sha256": "1" * 64, "artifact_path": str(self.root / "models"),
            "models": [{"name": "tiny", "config": self.config.to_dict()}],
        }

    def test_worker_fits_only_train_rows_then_saves_before_test_scoring(self):
        events = []
        test = self

        class FitSpy:
            def __init__(self, config, *, image_encoder):
                self.config, self.encoder = config, image_encoder
                self.training_summary_ = {}

            def fit(self, visits, labels):
                events.append("fit")
                test.assertEqual([visit.sample_id for visit in visits], ["train-a", "train-b"])
                test.assertEqual(list(labels), [0, 1])
                np.testing.assert_array_equal(self.encoder.transform([v.image for v in visits]), test.features[:2])
                test.assertTrue(all(v.structured == {} and v.text is None and v.human == {} for v in visits))
                return self

            def save(self, destination):
                events.append("save")
                test.assertFalse(self.training_summary_["test_used_for_selection"])
                test.assertFalse(self.training_summary_["grape_used_for_training_or_selection"])
                test.assertEqual(self.training_summary_["feature_table_sha256"], test.payload["feature_sha256"])

            def predict_score(self, visits):
                events.append("predict_test")
                test.assertEqual([v.sample_id for v in visits], ["test-a", "test-b"])
                np.testing.assert_array_equal(self.encoder.transform([v.image for v in visits]), test.features[2:])
                return np.asarray([.2, .8])

        with patch("glaboost.hf_training.GlaBoost", FitSpy):
            results = _fit_device_group(self.payload)
        self.assertEqual(events, ["fit", "save", "predict_test"])
        self.assertEqual(results[0]["probabilities"], [.2, .8])
        self.assertNotIn("test_labels", self.payload)
        self.assertEqual(results[0]["name"], "tiny")

    def test_corrupted_feature_file_and_invalid_index_rows_cannot_reach_fitting(self):
        np.save(self.feature_path, self.features + 1, allow_pickle=False)
        with patch("glaboost.hf_training.GlaBoost") as model:
            with self.assertRaisesRegex(ValueError, "changed after extraction"):
                _fit_device_group(self.payload)
            model.assert_not_called()
        encoder = _FeatureTableEncoder(self.features, encoder_spec())
        for rows in ([-1], [4], [1.0], [True], [[0]]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                encoder.transform(rows)
        np.testing.assert_array_equal(encoder.transform([3, 0]), self.features[[3, 0]])


class SourceGPUDispatchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="synthetic_hf_dispatch_", dir="/tmp")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.weights = self.root / "synthetic_weights.pth"
        self.weights.write_bytes(b"synthetic checkpoint; never decoded")
        self.plan_path = self.root / "training.json"
        self.model_names = ["trees1_depth1", "trees2_depth1", "trees3_depth1", "trees4_depth1"]
        self.plan = {"format_version": 1, "source": {"repo_id": "synthetic-source", "revision": "0" * 40},
                     "primary_model": self.model_names[0],
                     "models": [{"name": name, "n_estimators": i + 1, "max_depth": 1}
                                for i, name in enumerate(self.model_names)],
                     "evaluation": {"bootstrap_replicates": 20}, "min_visits": 3}
        self.plan_path.write_text(json.dumps(self.plan))
        self.dataset = SimpleNamespace(
            train_visits=tuple(VisitInput(f"train-{i}", image=bytes([i + 1])) for i in range(2)),
            test_visits=tuple(VisitInput(f"test-{i}", image=bytes([i + 3])) for i in range(2)),
            train_labels=np.asarray([0, 1]), test_labels=np.asarray([1, 0]),
            audit={"repo_id": "synthetic-source", "revision": "0" * 40, "synthetic": True,
                   "splits": {split: {"released_rows": 2, "retained_rows": 2,
                                      "within_split_duplicates_removed": 0,
                                      "train_overlap_duplicates_removed": 0}
                              for split in ("train", "test")}})

    def run_mocked(self, *, failure=None):
        payloads = []
        pool = MagicMock()

        def submit(function, payload):
            self.assertIs(function, _fit_device_group)
            payloads.append(deepcopy(payload))
            future = Mock()
            if failure is not None and len(payloads) == 2:
                future.result.side_effect = failure
            else:
                future.result.return_value = [{"name": model["name"],
                                               "model_directory": str(Path(payload["artifact_path"]) / model["name"]),
                                               "probabilities": [.2, .8]} for model in payload["models"]]
            return future

        pool.submit.side_effect = submit
        constructor = Mock(return_value=MagicMock(__enter__=Mock(return_value=pool), __exit__=Mock(return_value=False)))
        report = self.root / "result" / "gpu_mock"

        def external_report(**kwargs):
            report.mkdir(parents=True)
            (report / "report.html").write_text("<!doctype html><html><body>SYNTHETIC stub</body></html>\n")
            (report / "report.md").write_text("# SYNTHETIC GRAPE stub\n")
            return report

        with ExitStack() as stack:
            stack.enter_context(patch("glaboost.hf_training.load_hf_diagnosis", return_value=self.dataset))
            stack.enter_context(patch("glaboost.hf_training.ResNet152Encoder", MockSourceEncoder))
            stack.enter_context(patch("glaboost.hf_training.resolve_image_devices", return_value=("cuda:0", (0, 1))))
            context = stack.enter_context(patch("glaboost.hf_training.multiprocessing.get_context", return_value="mock-spawn"))
            stack.enter_context(patch("glaboost.hf_training.ProcessPoolExecutor", constructor))
            stack.enter_context(patch("glaboost.hf_training.as_completed", side_effect=lambda futures: list(futures)))
            stack.enter_context(patch("glaboost.hf_training._snapshot_code"))
            stack.enter_context(patch("glaboost.hf_training._environment_info", return_value={"synthetic": True}))
            stack.enter_context(patch("torch.cuda.empty_cache"))
            def source_stub(report, *args):
                (report / "report.html").write_text("<html><body>SYNTHETIC source stub</body></html>")
            source_report = stack.enter_context(patch("glaboost.hf_training._write_source_report", side_effect=source_stub))
            external = stack.enter_context(patch("glaboost.hf_training.run_external_validation", side_effect=external_report))
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(redirect_stderr(io.StringIO()))
            if failure is None:
                actual = run_hf_grape(training_plan_path=self.plan_path, run_name="gpu_mock",
                    hf_root=self.root / "hf", grape_root=self.root / "grape", result_dir=self.root / "result",
                    artifact_dir=self.root / "artifacts", cache_dir=self.root / "cache", device="cuda",
                    image_weights=self.weights, image_batch_size=2, synthetic=True)
                self.assertEqual(actual, report)
            else:
                with self.assertRaises(RuntimeError) as raised:
                    run_hf_grape(training_plan_path=self.plan_path, run_name="gpu_mock",
                        hf_root=self.root / "hf", grape_root=self.root / "grape", result_dir=self.root / "result",
                        artifact_dir=self.root / "artifacts", cache_dir=self.root / "cache", device="cuda",
                        image_weights=self.weights, image_batch_size=2, synthetic=True)
                self.assertIs(raised.exception, failure)
            constructor.assert_called_once_with(max_workers=2, mp_context="mock-spawn")
            context.assert_called_once_with("spawn")
        return payloads, source_report, external

    def test_spawned_gpu_groups_receive_training_labels_only_and_fixed_feature_indices(self):
        payloads, source_report, external = self.run_mocked()
        self.assertEqual(len(payloads), 2)
        for gpu, payload in enumerate(payloads):
            self.assertEqual([item["name"] for item in payload["models"]], self.model_names[gpu::2])
            self.assertNotIn("test_labels", payload)
            self.assertNotIn("grape_labels", payload)
            self.assertEqual(payload["train_labels"], [0, 1])
            self.assertEqual(payload["train_ids"], ["train-0", "train-1"])
            self.assertEqual(payload["test_ids"], ["test-0", "test-1"])
            for model in payload["models"]:
                self.assertEqual(model["config"]["device"], f"cuda:{gpu}")
                self.assertEqual(model["config"]["gpu_id"], gpu)
                self.assertEqual(model["config"]["tree_method"], "gpu_hist")
        self.assertEqual(payloads[0]["feature_path"], payloads[1]["feature_path"])
        self.assertEqual(payloads[0]["feature_sha256"], payloads[1]["feature_sha256"])
        features = np.load(payloads[0]["feature_path"], allow_pickle=False)
        np.testing.assert_array_equal(features[:, 0], [1, 2, 3, 4])
        np.testing.assert_array_equal(source_report.call_args.args[5], [1, 0])
        plan = json.loads(Path(external.call_args.kwargs["plan_path"]).read_text())
        self.assertEqual([model["name"] for model in plan["models"]], self.model_names)
        self.assertEqual(plan["primary_model"], self.model_names[0])
        self.assertEqual(external.call_args.kwargs["run_name"], "gpu_mock")
        for filename in ("report.html", "report.md"):
            self.assertIn("../gpu_mock_source/report.html", (self.root / "result/gpu_mock" / filename).read_text())

    def test_worker_failure_keeps_partial_source_results_and_never_starts_grape(self):
        failure = RuntimeError("synthetic source worker failed")
        _, source_report, external = self.run_mocked(failure=failure)
        source_report.assert_not_called()
        external.assert_not_called()
        report, artifacts = self.root / "result/gpu_mock_source", self.root / "artifacts/gpu_mock_source"
        status = json.loads((report / "status.json").read_text())
        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["stage"], "source_tree_training")
        self.assertEqual(status["completed_models"], self.model_names[::2])
        self.assertEqual(status, json.loads((artifacts / "status.json").read_text()))
        self.assertIn("synthetic source worker failed", (artifacts / "error_traceback.txt").read_text())
        self.assertTrue((artifacts / "resnet152_features.npy").is_file())
        self.assertFalse((self.root / "result/gpu_mock").exists())


if __name__ == "__main__":
    unittest.main()
