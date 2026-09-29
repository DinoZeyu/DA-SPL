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
