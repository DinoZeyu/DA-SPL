"""Offline checks for reusable paper/senior image diagnosis recipes."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from glaboost import GlaBoost, GlaBoostConfig
from glaboost.encoders import ResNet18Encoder, ResNet152Encoder, image_encoder_class
from glaboost.model import make_xgb_classifier
from test_encoders import TinyDataParallel, TinyResNet, cpu_only_tensor_to


class TinyResNet18(TinyResNet):
    def forward(self, pixels):
        return super().forward(pixels)[:, :512]


class SeniorMethodTests(unittest.TestCase):
    def test_presets_and_legacy_configuration_round_trip(self):
        ch = GlaBoostConfig.for_image_method("ch")
        self.assertEqual((ch.image_encoder, ch.n_estimators, ch.max_depth,
                          ch.learning_rate, ch.subsample, ch.colsample_bytree),
                         ("resnet18", 500, 6, .05, .8, .8))
        self.assertEqual(GlaBoostConfig.from_dict(ch.to_dict()), ch)
        paper = GlaBoostConfig.for_image_method()
        self.assertEqual(paper, GlaBoostConfig())
        legacy = paper.to_dict()
        for key in ("image_encoder", "subsample", "colsample_bytree"):
            legacy.pop(key)
        self.assertEqual(GlaBoostConfig.from_dict(legacy), paper)
        self.assertEqual(GlaBoostConfig.for_image_method(n_estimators=2).n_estimators, 2)
        for invalid in ("unknown", "CH", None):
            with self.assertRaises(ValueError):
                GlaBoostConfig.for_image_method(invalid)

    def test_recipe_validation_rejects_invalid_sampling_and_encoder(self):
        for key in ("subsample", "colsample_bytree"):
            for invalid in (0, -1, 1.01, float("nan"), float("inf"), True, "0.8"):
                with self.subTest(key=key, value=invalid), self.assertRaises(ValueError):
                    GlaBoostConfig(**{key: invalid})
        with self.assertRaises(ValueError):
            GlaBoostConfig(image_encoder="resnet50")
        with self.assertRaises(ValueError):
            image_encoder_class("resnet50")

    def test_classifier_and_lazy_model_use_selected_recipe(self):
        config = GlaBoostConfig.for_image_method("ch")
        params = make_xgb_classifier(config).get_params()
        for key in ("n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree"):
            self.assertEqual(params[key], getattr(config, key))
        with patch("glaboost.encoders._resnet18_uninitialized") as builder:
            model = GlaBoost(config)
            self.assertIsInstance(model.image_encoder, ResNet18Encoder)
            self.assertEqual(model.image_encoder.transform([]).shape, (0, 512))
            builder.assert_not_called()
        self.assertIs(image_encoder_class("resnet152"), ResNet152Encoder)
        self.assertIs(image_encoder_class("resnet18"), ResNet18Encoder)

    def test_preprocessing_exactly_matches_notebook_resize_and_tensor(self):
        pixels = np.arange(13 * 9 * 3, dtype=np.uint8).reshape(13, 9, 3)
        image = Image.fromarray(pixels)
        expected = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])(image)
        actual = ResNet18Encoder._prepare(image)
        self.assertTrue(torch.equal(actual, expected))
        self.assertFalse(torch.equal(actual, ResNet152Encoder._prepare(image)))
        spec = ResNet18Encoder().spec()
        self.assertEqual((spec["encoder"], spec["output_dim"]), ("resnet18", 512))
        self.assertIsNone(spec["preprocessing"]["mean"])
        self.assertIsNone(spec["preprocessing"]["std"])
        self.assertEqual(spec["requested_weights"], "ResNet18_Weights.IMAGENET1K_V1")

    def test_resnet18_local_weights_freezing_and_ordered_multigpu_batches(self):
        images = [Image.new("RGB", (5, 3), (v, v, v)) for v in (0, 40, 100, 180, 255)]
        with tempfile.TemporaryDirectory() as directory:
            network = TinyResNet18()
            checkpoint = Path(directory) / "tiny.pth"
            torch.save(network.state_dict(), checkpoint)
            with patch("glaboost.encoders._resnet18_uninitialized", return_value=network), \
                 patch("torch.cuda.is_available", return_value=True), \
                 patch("torch.cuda.device_count", return_value=2), \
                 patch.object(network, "to", return_value=network), \
                 patch.object(torch.Tensor, "to", new=cpu_only_tensor_to), \
                 patch("torch.nn.DataParallel", side_effect=TinyDataParallel):
                encoder = ResNet18Encoder(weights_path=checkpoint, device="cuda", batch_size=3)
                result = encoder.transform(images)
            self.assertEqual(result.shape, (5, 512))
            np.testing.assert_allclose(result[:, 0], np.array([0, 40, 100, 180, 255]) / 255, atol=1e-6)
            self.assertEqual(encoder._model.global_batches, [3, 2])
            self.assertFalse(encoder._model.training)
            self.assertTrue(all(not p.requires_grad for p in encoder._model.parameters()))
            self.assertTrue(all(not mode for mode in network.grad_modes))
            self.assertEqual(encoder.spec()["active_gpu_ids"], [0, 1])
            self.assertTrue(encoder.spec()["fingerprint"].startswith("sha256:"))

    def test_resnet18_missing_or_corrupted_cache_never_uses_random_weights(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch("glaboost.encoders._resnet18_uninitialized") as builder, \
             patch("torch.hub.load_state_dict_from_url") as download:
            encoder = ResNet18Encoder(cache_dir=directory)
            with self.assertRaises(FileNotFoundError):
                encoder.transform([Image.new("RGB", (2, 2))])
            cache = Path(directory) / "torch" / encoder.checkpoint_filename
            cache.parent.mkdir()
            torch.save(TinyResNet18().state_dict(), cache)
            with self.assertRaisesRegex(ValueError, "invalid official SHA256"):
                encoder.transform([Image.new("RGB", (2, 2))])
            builder.assert_not_called()
            download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
