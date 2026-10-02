"""Small offline tests: no pretrained downloads and no model training."""

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from glaboost.encoders import ResNet152Encoder, image_encoder_class, resolve_image_devices


class TinyResNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.marker = torch.nn.Parameter(torch.ones(1))
        self.fc = torch.nn.Linear(1, 1)
        self.batches = []
        self.grad_modes = []

    def forward(self, pixels):
        self.batches.append(pixels.detach().cpu().clone())
        self.grad_modes.append(torch.is_grad_enabled())
        return pixels.mean(dim=(1, 2, 3)).unsqueeze(1).repeat(1, 2048) * self.marker


class TinyDataParallel(torch.nn.Module):
    """CPU-only ordered scatter/gather stand-in, without claiming GPU execution."""
    def __init__(self, module, device_ids, output_device):
        super().__init__()
        self.module = module
        self.device_ids = device_ids
        self.output_device = output_device
        self.global_batches = []

    def forward(self, inputs):
        self.global_batches.append(len(inputs))
        return torch.cat([self.module(chunk) for chunk in inputs.chunk(len(self.device_ids))], dim=0)


_tensor_to = torch.Tensor.to


def cpu_only_tensor_to(tensor, *args, **kwargs):
    if args and isinstance(args[0], str) and args[0].startswith("cuda"):
        return tensor
    return _tensor_to(tensor, *args, **kwargs)


class EncoderTests(unittest.TestCase):
    def test_lazy_construction_empty_input_and_json_spec(self):
        with patch("glaboost.encoders._resnet152_uninitialized") as image_loader:
            image = ResNet152Encoder()
            self.assertEqual(image.transform([]).shape, (0, 2048))
            self.assertIsNone(image.spec()["fingerprint"])
            json.dumps(image.spec())
            self.assertIs(image_encoder_class("resnet152"), ResNet152Encoder)
            for name in ("resnet18", "resnet50", None):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    image_encoder_class(name)
            image_loader.assert_not_called()

    def test_device_resolution_respects_torch_visible_count_and_explicit_selection(self):
        for requested, count, expected in (
            ("auto", 0, ("cpu", ())), ("auto", 1, ("cuda:0", (0,))),
            ("auto", 3, ("cuda:0", (0, 1, 2))), ("cuda", 2, ("cuda:0", (0, 1))),
            ("cuda:1", 3, ("cuda:1", (1,))),
        ):
            with self.subTest(requested=requested, count=count), \
                 patch("torch.cuda.is_available", return_value=count > 0), \
                 patch("torch.cuda.device_count", return_value=count):
                self.assertEqual(resolve_image_devices(requested), expected)
        with patch("torch.cuda.is_available") as available:
            self.assertEqual(resolve_image_devices("cpu"), ("cpu", ()))
            available.assert_not_called()
        with patch("torch.cuda.is_available", return_value=False):
            for requested in ("cuda", "cuda:0", "cuda:3"):
                with self.assertRaisesRegex(RuntimeError, "no CUDA"):
                    resolve_image_devices(requested)
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.device_count", return_value=2):
            with self.assertRaisesRegex(RuntimeError, "only 2"):
                resolve_image_devices("cuda:2")
        for requested in ("gpu", "cuda:-1", "mps", "cpu:1"):
            with self.assertRaises(ValueError):
                resolve_image_devices(requested)

    def test_lazy_specs_do_not_probe_gpus_or_load_networks(self):
        with patch("glaboost.encoders.resolve_image_devices") as resolve:
            encoder = ResNet152Encoder(device="auto")
            self.assertEqual(encoder.spec()["requested_device"], "auto")
            self.assertIsNone(encoder.spec()["resolved_device"])
            self.assertEqual(encoder.spec()["active_gpu_ids"], [])
            self.assertEqual(encoder.spec()["parallelism"], "unresolved")
            encoder.transform([])
            resolve.assert_not_called()

    def test_multigpu_image_scatter_preserves_order_global_batch_and_fingerprint(self):
        images = [Image.new("RGB", (5, 3), (value, value, value)) for value in (0, 40, 100, 180, 255)]
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "tiny.pth"
            torch.save(TinyResNet().state_dict(), checkpoint)
            with patch("glaboost.encoders._resnet152_uninitialized", return_value=TinyResNet()):
                cpu_encoder = ResNet152Encoder(weights_path=checkpoint, batch_size=3)
                expected = cpu_encoder.transform(images)
            network = TinyResNet()
            with patch("glaboost.encoders._resnet152_uninitialized", return_value=network), \
                 patch("torch.cuda.is_available", return_value=True), \
                 patch("torch.cuda.device_count", return_value=2), \
                 patch.object(network, "to", return_value=network) as model_to, \
                 patch.object(torch.Tensor, "to", new=cpu_only_tensor_to), \
                 patch("torch.nn.DataParallel", side_effect=TinyDataParallel) as parallel:
                encoder = ResNet152Encoder(weights_path=checkpoint, device="auto", batch_size=3)
                actual = encoder.transform(images)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(actual.dtype, np.float32)
            self.assertEqual(encoder._model.global_batches, [3, 2])
            parallel.assert_called_once_with(network, device_ids=[0, 1], output_device=0)
            model_to.assert_called_once_with("cuda:0")
            self.assertFalse(encoder._model.training)
            self.assertTrue(all(not p.requires_grad for p in encoder._model.parameters()))
            self.assertTrue(all(mode is False for mode in network.grad_modes))
            spec = encoder.spec()
            self.assertEqual(spec["active_gpu_count"], 2)
            self.assertEqual(spec["active_gpu_ids"], [0, 1])
            self.assertEqual(spec["batch_size"], 3)
            self.assertEqual(spec["batch_size_scope"], "global")
            self.assertEqual(spec["fingerprint"], cpu_encoder.spec()["fingerprint"])
            self.assertEqual(spec["preprocessing"], cpu_encoder.spec()["preprocessing"])

    def test_single_selected_gpu_does_not_wrap_data_parallel(self):
        for requested, visible, selected in (("auto", 1, 0), ("cuda:1", 3, 1)):
            with self.subTest(requested=requested), tempfile.TemporaryDirectory() as directory:
                network = TinyResNet()
                checkpoint = Path(directory) / "tiny.pth"
                torch.save(network.state_dict(), checkpoint)
                with patch("glaboost.encoders._resnet152_uninitialized", return_value=network), \
                     patch("torch.cuda.is_available", return_value=True), \
                     patch("torch.cuda.device_count", return_value=visible), \
                     patch.object(network, "to", return_value=network) as model_to, \
                     patch.object(torch.Tensor, "to", new=cpu_only_tensor_to), \
                     patch("torch.nn.DataParallel") as parallel:
                    encoder = ResNet152Encoder(weights_path=checkpoint, device=requested)
                    encoder.transform([Image.new("RGB", (2, 2))])
                parallel.assert_not_called()
                model_to.assert_called_once_with(f"cuda:{selected}")
                self.assertEqual(encoder.spec()["active_gpu_ids"], [selected])

    def test_explicit_unavailable_gpu_fails_before_weight_download(self):
        with patch("torch.cuda.is_available", return_value=False), \
             patch("torch.hub.load_state_dict_from_url") as download:
            with self.assertRaisesRegex(RuntimeError, "no CUDA"):
                ResNet152Encoder(device="cuda", allow_download=True).transform([Image.new("RGB", (2, 2))])
            download.assert_not_called()

    def test_download_enables_progress_and_hash_verification(self):
        network = TinyResNet()
        with tempfile.TemporaryDirectory() as directory:
            def download(url, model_dir, **kwargs):
                checkpoint = Path(model_dir) / url.rsplit("/", 1)[1]
                torch.save(network.state_dict(), checkpoint)
                return network.state_dict()
            with patch("glaboost.encoders._resnet152_uninitialized", return_value=network), \
                 patch("torch.hub.load_state_dict_from_url", side_effect=download) as loader, \
                 patch("glaboost.encoders._sha256", return_value="394f9c45" + "0" * 56):
                ResNet152Encoder(cache_dir=directory, allow_download=True).transform([Image.new("RGB", (2, 2))])
            self.assertTrue(loader.call_args.kwargs["progress"])
            self.assertTrue(loader.call_args.kwargs["check_hash"])

    def test_missing_resnet_weights_never_fall_back_to_random_or_download(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch("glaboost.encoders._resnet152_uninitialized") as builder, \
             patch("torch.hub.load_state_dict_from_url") as download:
            encoder = ResNet152Encoder(cache_dir=directory)
            with self.assertRaisesRegex(FileNotFoundError, "Random weights are never used"):
                encoder.transform([Image.new("RGB", (3, 5))])
            builder.assert_not_called()
            download.assert_not_called()
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_corrupted_official_cache_is_rejected_before_model_construction(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch("glaboost.encoders._resnet152_uninitialized") as builder, \
             patch("torch.hub.load_state_dict_from_url") as download:
            encoder = ResNet152Encoder(cache_dir=directory)
            checkpoint = Path(directory) / "torch" / encoder.checkpoint_filename
            checkpoint.parent.mkdir()
            torch.save(TinyResNet().state_dict(), checkpoint)
            with self.assertRaisesRegex(ValueError, "invalid official SHA256"):
                encoder.transform([Image.new("RGB", (2, 2))])
            builder.assert_not_called()
            download.assert_not_called()

    def test_resnet_preprocessing_order_freezing_and_local_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "weights.pth"
            network = TinyResNet()
            torch.save(network.state_dict(), checkpoint)
            image_path = Path(directory) / "gray.png"
            Image.new("L", (9, 2), 0).save(image_path)
            buffer = io.BytesIO()
            Image.new("RGBA", (5, 11), (127, 127, 127, 0)).save(buffer, format="PNG")
            with patch("glaboost.encoders._resnet152_uninitialized", return_value=network):
                encoder = ResNet152Encoder(weights_path=checkpoint, batch_size=2)
                result = encoder.transform([image_path, buffer.getvalue(), Image.new("RGB", (7, 3), "white")])
            self.assertEqual(result.shape, (3, 2048))
            self.assertEqual(result.dtype, np.float32)
            self.assertTrue(np.all(np.diff(result[:, 0]) > 0))
            self.assertEqual([tuple(batch.shape) for batch in network.batches], [(2, 3, 224, 224), (1, 3, 224, 224)])
            np.testing.assert_allclose(network.batches[0][0, :, 0, 0], -np.array([.485, .456, .406]) / [.229, .224, .225], rtol=1e-6)
            self.assertFalse(network.training)
            self.assertTrue(all(not parameter.requires_grad for parameter in network.parameters()))
            self.assertEqual(network.grad_modes, [False, False])
            self.assertIsInstance(network.fc, torch.nn.Identity)
            self.assertTrue(encoder.spec()["fingerprint"].startswith("sha256:"))
            json.dumps(encoder.spec())
            with self.assertRaisesRegex(ValueError, "row 1"):
                encoder.transform([image_path, None, image_path])

    def test_resnet_rejects_incomplete_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "bad.pth"
            torch.save({}, checkpoint)
            with patch("glaboost.encoders._resnet152_uninitialized", return_value=TinyResNet()):
                encoder = ResNet152Encoder(weights_path=checkpoint)
                with self.assertRaises(RuntimeError):
                    encoder.transform([Image.new("RGB", (2, 2))])
            self.assertFalse(encoder.spec()["loaded"])

    def test_invalid_parameters_and_scalar_inputs(self):
        for batch_size in (0, -1, True, 1.2):
            with self.subTest(batch_size=batch_size), self.assertRaises(ValueError):
                ResNet152Encoder(batch_size=batch_size)
        with self.assertRaises(TypeError):
            ResNet152Encoder().transform("single sample")


if __name__ == "__main__":
    unittest.main()
