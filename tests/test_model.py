"""Tiny synthetic checks, never a GRAPE training run or performance estimate."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from importlib.metadata import PackageNotFoundError
from unittest.mock import Mock, patch

import numpy as np

from glaboost import GlaBoost, GlaBoostConfig, VisitInput


class TinyEncoder:
    output_dim = 2

    def __init__(self, text=False):
        self.text = text
        self.fingerprint = "tiny-test-fixture"

    def transform(self, inputs):
        if self.text:
            return np.array([[len(v), len(v) / 2] for v in inputs], dtype=np.float32)
        return np.asarray(inputs, dtype=np.float32)

    def spec(self):
        return {"fingerprint": self.fingerprint, "resolved_revision": None,
                "encoder": "tiny", "output_dim": self.output_dim,
                "preprocessing": {"fixture": True}}


class GlaBoostTests(unittest.TestCase):
    def test_gpu_classifier_pins_training_prediction_and_device(self):
        from glaboost.model import make_xgb_classifier
        config = GlaBoostConfig(device="cuda:1", tree_method="gpu_hist", gpu_id=1)
        params = make_xgb_classifier(config).get_params()
        self.assertEqual(params["tree_method"], "gpu_hist")
        self.assertEqual(params["predictor"], "gpu_predictor")
        self.assertEqual(params["gpu_id"], 1)
        with self.assertRaises(ValueError):
            GlaBoostConfig(tree_method="gpu_hist")

    def test_gpu_backend_guard_rejects_fallback_or_wrong_gpu(self):
        from glaboost.model import assert_xgb_backend
        config = GlaBoostConfig(device="cuda:1", tree_method="gpu_hist", gpu_id=1)
        state = {"learner": {"generic_param": {"gpu_id": "1"}, "gradient_booster": {
            "gbtree_train_param": {"tree_method": "gpu_hist", "predictor": "gpu_predictor"}}}}
        model = Mock()
        model.get_booster.return_value.save_config.side_effect = lambda: json.dumps(state)
        assert_xgb_backend(model, config)
        for field, bad in (("tree_method", "hist"), ("predictor", "cpu_predictor")):
            params = state["learner"]["gradient_booster"]["gbtree_train_param"]
            original = params[field]
            params[field] = bad
            with self.assertRaises(RuntimeError):
                assert_xgb_backend(model, config)
            params[field] = original
        state["learner"]["generic_param"]["gpu_id"] = "0"
        with self.assertRaises(RuntimeError):
            assert_xgb_backend(model, config)

    def setUp(self):
        self.visits = [VisitInput(
            sample_id=f"v{i}", patient_id=f"p{i // 2}", eye_id=f"e{i}",
            time_years=float(i), image=[i / 10, i % 2], text="rim " + "thin" * (i % 3 + 1),
            structured={"iop": 10.0 + i, "status": "a" if i % 2 else "b"},
            human={"glaucoma_risk_assessment": "high" if i % 2 else "low", "confidence_level": .8},
        ) for i in range(16)]
        self.y = np.asarray([i % 2 for i in range(16)])
        self.config = GlaBoostConfig(
            use_image=False, use_structured=True,
            numeric_features=("iop",), categorical_features=("status",),
            n_estimators=3, max_depth=2,
        )

    def test_paper_parameters_and_grape_default(self):
        c = GlaBoostConfig()
        self.assertTrue(c.use_image)
        self.assertFalse(c.use_text or c.use_structured or c.use_human_risk or c.use_human_confidence)
        self.assertEqual((c.learning_rate, c.max_depth, c.n_estimators, c.text_max_length), (.05, 6, 100, 128))

    def test_inference_is_visit_independent_and_does_not_refit(self):
        model = GlaBoost(self.config).fit(self.visits, self.y)
        before = model._structured.to_dict()
        scores = model.predict_score(self.visits)
        singles = [model.predict_score([visit])[0] for visit in self.visits]
        np.testing.assert_allclose(scores, singles)
        modified = replace(self.visits[0], structured={"iop": 1000., "status": "new"})
        model.predict_score([modified])
        self.assertEqual(before, model._structured.to_dict())
        probabilities = model.predict_proba(self.visits)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.)
        np.testing.assert_array_equal(model.predict(self.visits), probabilities[:, 1] >= .5)

    def test_disabled_human_text_and_metadata_are_not_features(self):
        model = GlaBoost(self.config).fit(self.visits, self.y)
        changed = [replace(v, text="diagnosis label text", human={"target": 999},
                           patient_id="another", time_years=999.) for v in self.visits]
        np.testing.assert_array_equal(model.transform(self.visits), model.transform(changed))
        self.assertTrue(all(name.startswith("structured::") for name in model.feature_names_))

    def test_native_json_round_trip(self):
        model = GlaBoost(self.config).fit(self.visits, self.y)
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "model"
            model.save(output)
            restored = GlaBoost.load(output)
            self.assertEqual(restored.classifier_.get_params()["predictor"], "cpu_predictor")
            self.assertEqual(restored.classifier_.get_params()["gpu_id"], -1)
            np.testing.assert_allclose(model.predict_proba(self.visits), restored.predict_proba(self.visits))
            self.assertEqual(model.feature_names_, restored.feature_names_)
            self.assertEqual(set(model.feature_importance()), set(model.feature_names_))
            metadata = json.loads((output / "metadata.json").read_text())
            self.assertEqual(metadata["target"], {"0": "normal", "1": "glaucoma"})
            with self.assertRaises(FileExistsError):
                model.save(output)
            with (output / "model.json").open("a") as stream:
                stream.write(" ")
            with self.assertRaisesRegex(ValueError, "checksum"):
                GlaBoost.load(output)

    def test_loaded_classifier_uses_runtime_device_without_changing_learned_settings(self):
        original = GlaBoost(self.config).fit(self.visits, self.y)
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "model"
            original.save(output)
            metadata_path = output / "metadata.json"
            base_metadata = json.loads(metadata_path.read_text())
            cases = ((self.config, "cuda:1", "cuda:1", (1,), "gpu_predictor", 1),
                     (replace(self.config, device="cuda:1", tree_method="gpu_hist", gpu_id=1),
                      "cpu", "cpu", (), "cpu_predictor", -1))
            for training_config, device, resolved, gpu_ids, predictor, gpu_id in cases:
                with self.subTest(device=device):
                    # Only mock runtime placement; these are not GPU-trained checkpoints.
                    metadata = dict(base_metadata, config=training_config.to_dict())
                    metadata_path.write_text(json.dumps(metadata))
                    before = metadata_path.read_bytes()
                    state = {"learner": {"generic_param": {"gpu_id": str(gpu_id)}, "gradient_booster": {
                        "gbtree_train_param": {"tree_method": training_config.tree_method, "predictor": predictor}}}}
                    classifier = Mock()
                    classifier.n_features_in_ = len(original.feature_names_)
                    classifier.get_booster.return_value.save_config.side_effect = lambda: json.dumps(state)
                    classifier.predict_proba.return_value = np.tile([.25, .75], (len(self.visits), 1))
                    with patch("glaboost.model.XGBClassifier", return_value=classifier), \
                            patch("glaboost.model.resolve_image_devices", return_value=(resolved, gpu_ids)):
                        restored = GlaBoost.load(output, device=device)
                    classifier.set_params.assert_called_once_with(
                        n_jobs=training_config.n_jobs, predictor=predictor, gpu_id=gpu_id)
                    self.assertEqual(restored.training_config_, training_config)
                    self.assertEqual(restored.config.n_estimators, 3)
                    self.assertEqual(restored.config.max_depth, 2)
                    self.assertEqual(restored.prediction_runtime_["resolved_device"], resolved)
                    np.testing.assert_array_equal(restored.predict_score(self.visits), np.full(len(self.visits), .75))
                    self.assertEqual(metadata_path.read_bytes(), before)
                    classifier.fit.assert_not_called()
                    # Reject backend changes before exposing predictions.
                    classifier.predict_proba.reset_mock()
                    state["learner"]["generic_param"]["gpu_id"] = "0" if gpu_id != 0 else "1"
                    with self.assertRaisesRegex(RuntimeError, "prediction device/backend"):
                        restored.predict_score(self.visits)
                    classifier.predict_proba.assert_not_called()

    def test_gpu_load_rejects_cpu_fallback_and_unavailable_device(self):
        original = GlaBoost(self.config).fit(self.visits, self.y)
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "model"
            original.save(output)
            classifier = Mock()
            classifier.n_features_in_ = len(original.feature_names_)
            classifier.get_booster.return_value.save_config.return_value = json.dumps({
                "learner": {"generic_param": {"gpu_id": "-1"}, "gradient_booster": {
                    "gbtree_train_param": {"tree_method": "hist", "predictor": "cpu_predictor"}}}})
            with patch("glaboost.model.XGBClassifier", return_value=classifier), \
                    patch("glaboost.model.resolve_image_devices", return_value=("cuda:0", (0, 1))):
                with self.assertRaisesRegex(RuntimeError, "CPU fallback"):
                    GlaBoost.load(output, device="cuda")
            with patch("glaboost.model.resolve_image_devices", side_effect=RuntimeError("CUDA unavailable")), \
                    patch("glaboost.model.XGBClassifier") as constructor:
                with self.assertRaisesRegex(RuntimeError, "CUDA unavailable"):
                    GlaBoost.load(output, device="cuda")
                constructor.assert_not_called()

    def test_runtime_batch_and_cache_overrides_preserve_resaved_training_provenance(self):
        original = GlaBoost(self.config).fit(self.visits, self.y)
        with tempfile.TemporaryDirectory() as root:
            output, resaved = Path(root) / "model", Path(root) / "resaved"
            original.save(output)
            metadata = json.loads((output / "metadata.json").read_text())
            restored = GlaBoost.load(output, device="cpu", image_batch_size=128, cache_dir=str(Path(root) / "cache"))
            self.assertEqual(restored.config.image_batch_size, 128)
            self.assertEqual(restored.training_config_, self.config)
            restored.save(resaved)
            saved = json.loads((resaved / "metadata.json").read_text())
            self.assertEqual(saved["config"], metadata["config"])
            self.assertEqual(saved["target"], {"0": "normal", "1": "glaucoma"})
            self.assertEqual(len(restored.classifier_.get_booster().get_dump()), 3)
            np.testing.assert_array_equal(original.predict_score(self.visits), restored.predict_score(self.visits))
            for invalid in (0, -1, True, 1.5):
                with self.subTest(batch=invalid), self.assertRaisesRegex(ValueError, "image_batch_size"):
                    GlaBoost.load(output, image_batch_size=invalid)

    def test_fusion_order_and_encoder_identity_round_trip(self):
        c = replace(self.config, use_image=True, use_text=True,
                    use_human_risk=True, use_human_confidence=True)
        image, text = TinyEncoder(), TinyEncoder(text=True)
        model = GlaBoost(c, image_encoder=image, text_encoder=text).fit(self.visits, self.y)
        names = model.feature_names_
        self.assertEqual(names[:2], ["text_0", "text_1"])
        self.assertTrue(names[2].startswith("structured::"))
        self.assertLess(next(i for i,n in enumerate(names) if n.startswith("structured::")),
                        next(i for i,n in enumerate(names) if n.startswith("human::")))
        self.assertEqual(names[-2:], ["image_0", "image_1"])
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "model"
            model.save(output)
            restored = GlaBoost.load(output, image_encoder=image, text_encoder=text)
            np.testing.assert_allclose(model.predict_score(self.visits), restored.predict_score(self.visits))
            image.fingerprint = "different-weights"
            with self.assertRaisesRegex(ValueError, "weights or preprocessing differ"):
                restored.predict_score(self.visits)

    def test_single_class_cannot_train_diagnosis_model(self):
        with self.assertRaisesRegex(ValueError, "both normal and glaucoma"):
            GlaBoost().fit(self.visits, np.ones(len(self.visits)))

    def test_encoder_spec_is_an_independent_training_snapshot(self):
        encoder = TinyEncoder()
        shared_spec = encoder.spec()
        encoder.spec = lambda: shared_spec
        c = replace(self.config, use_image=True, use_structured=False)
        model = GlaBoost(c, image_encoder=encoder).fit(self.visits, self.y)
        shared_spec["preprocessing"]["fixture"] = False
        with self.assertRaisesRegex(ValueError, "weights or preprocessing differ"):
            model.predict_score(self.visits)

    def test_injected_text_encoder_does_not_require_optional_transformers_package(self):
        model = GlaBoost(replace(self.config, use_text=True), text_encoder=TinyEncoder(text=True))
        model.fit(self.visits, self.y)
        with tempfile.TemporaryDirectory() as root, \
             patch("glaboost.model.version", side_effect=PackageNotFoundError):
            model.save(Path(root) / "model")
            metadata = json.loads((Path(root) / "model" / "metadata.json").read_text())
            self.assertIsNone(metadata["versions"]["transformers"])

    def test_bad_inputs_fail_before_network_loading(self):
        with self.assertRaisesRegex(ValueError, "Missing enabled image"):
            GlaBoost().fit([VisitInput("one"), VisitInput("two")], [0, 1])
        with self.assertRaisesRegex(ValueError, "unique"):
            GlaBoost(self.config).fit([self.visits[0], self.visits[0]], [0, 1])
        with self.assertRaises(ValueError):
            GlaBoost(self.config).fit(self.visits, np.arange(len(self.visits)))
        with self.assertRaises(ValueError):
            GlaBoost(self.config).fit(self.visits, self.y, sample_weight=np.full(len(self.visits), -1))
        with self.assertRaises(RuntimeError):
            GlaBoost(self.config).predict_score(self.visits)
        with self.assertRaises(ValueError):
            GlaBoostConfig(numeric_features=("plr2",))
        for malformed in ("iop", b"iop", [" "], [" iop"], [["iop"]]):
            with self.assertRaises(ValueError):
                GlaBoostConfig(numeric_features=malformed)


if __name__ == "__main__":
    unittest.main()
