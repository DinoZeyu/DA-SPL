"""CPU-only numerical references for the optional Torch GPU metrics backend."""

import io
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from tqdm import tqdm

from glaboost.gpu_metrics import _weighted_metrics_tensor, paired_bootstrap_gpu
from glaboost.longitudinal import EvaluationConfig, METRIC_NAMES, _metrics


class GPUMetricsTests(unittest.TestCase):
    def test_weighted_ties_zero_one_and_threshold_match_sklearn(self):
        y = np.array([0, 1, 0, 1, 0, 1, 1, 0])
        probabilities = np.array([0, 0, .5, .5, 1, 1, .2, .8])
        weights = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
                            [3, 0, 1, 4, 0, 2, 2, 1],
                            [0, 0, 2, 3, 0, 0, 0, 0],
                            [0, 3, 0, 0, 2, 0, 0, 0]])
        actual = _weighted_metrics_tensor(torch.tensor(y), torch.tensor(probabilities), torch.tensor(weights))
        for row, weight in enumerate(weights):
            indices = np.repeat(np.arange(len(y)), weight)
            expected = _metrics(y[indices], probabilities[indices])
            for metric in METRIC_NAMES:
                self.assertAlmostEqual(actual[metric][row].item(), expected[metric], places=14)
            self.assertAlmostEqual(actual["auroc"][row].item(),
                                   roc_auc_score(y, probabilities, sample_weight=weight), places=14)
            self.assertAlmostEqual(actual["auprc"][row].item(),
                                   average_precision_score(y, probabilities, sample_weight=weight), places=14)

    def test_all_scores_tied_and_fractional_weights(self):
        y = torch.tensor([0, 1, 0, 1])
        probabilities = torch.full((4,), .5, dtype=torch.float64)
        weights = torch.tensor([[.2, .4, .8, .1], [0., .3, .7, 0.]], dtype=torch.float64)
        actual = _weighted_metrics_tensor(y, probabilities, weights)
        np.testing.assert_allclose(actual["auroc"].numpy(), [.5, .5], atol=1e-15)
        for row in range(len(weights)):
            self.assertAlmostEqual(actual["auprc"][row].item(), average_precision_score(
                y.numpy(), probabilities.numpy(), sample_weight=weights[row].numpy()), places=14)

    def test_random_tied_metric_batches_match_explicit_duplicates(self):
        rng = np.random.default_rng(52)
        y = np.tile([0, 1], 12)
        probabilities = rng.choice([0., .1, .2, .5, .8, 1.], size=len(y))
        weights = rng.integers(0, 5, size=(12, len(y)))
        actual = _weighted_metrics_tensor(torch.tensor(y), torch.tensor(probabilities), torch.tensor(weights))
        for row, weight in enumerate(weights):
            indices = np.repeat(np.arange(len(y)), weight)
            expected = _metrics(y[indices], probabilities[indices])
            for metric in METRIC_NAMES:
                self.assertAlmostEqual(actual[metric][row].item(), expected[metric], places=14)

    def test_identical_patient_draws_match_numpy_reference_intervals(self):
        # Noncontiguous patient eyes, including a mixed-label patient, establish
        # that every duplicate patient draw retains all of that patient's eyes.
        groups = np.array(["p", "q", "r", "p", "q", "r"])
        y = np.array([0, 1, 0, 0, 1, 1])
        latest = np.array([.3, .4, .7, .5, .8, .8])
        longitudinal = np.array([.3, .8, .4, .3, .8, .5])
        draws = np.tile([[0, 0, 0], [0, 1, 2], [1, 1, 2], [2, 2, 0]], (10, 1))
        reference = {method: {metric: [] for metric in METRIC_NAMES}
                     for method in ("latest", "longitudinal")}
        differences = []
        for draw in draws:
            indices = np.concatenate([np.flatnonzero(groups == ["p", "q", "r"][patient]) for patient in draw])
            if len(np.unique(y[indices])) < 2:
                continue
            pair = {method: _metrics(y[indices], scores[indices])
                    for method, scores in (("latest", latest), ("longitudinal", longitudinal))}
            for method in reference:
                for metric in METRIC_NAMES:
                    reference[method][metric].append(pair[method][metric])
            differences.append(pair["longitudinal"]["balanced_accuracy"] - pair["latest"]["balanced_accuracy"])
        log = io.StringIO()
        with patch("glaboost.gpu_metrics.torch.randint", return_value=torch.tensor(draws)) as sample, \
             patch("glaboost.gpu_metrics.tqdm", side_effect=lambda *args, **kwargs:
                   tqdm(*args, **kwargs, file=log, disable=False, mininterval=0)):
            output, delta, metadata = paired_bootstrap_gpu(
                y, latest, longitudinal, groups, EvaluationConfig(bootstrap_replicates=40),
                device="cpu", progress_desc="plr2: paired bootstrap")
        self.assertEqual(sample.call_count, 1)
        self.assertEqual(sample.call_args.args, (3, (40, 3)))
        self.assertEqual(metadata["valid"], 30)
        self.assertEqual(metadata["skipped_single_class"], 10)
        self.assertEqual(metadata["ci_status"], "ok")
        self.assertIn("40/40", log.getvalue())
        self.assertIn("valid=30", log.getvalue())
        self.assertIn("skipped=10", log.getvalue())
        self.assertIn("NumPy PCG64", metadata["rng_note"])
        for method, scores in (("latest", latest), ("longitudinal", longitudinal)):
            points = _metrics(y, scores)
            for metric in METRIC_NAMES:
                self.assertAlmostEqual(output[method][metric]["estimate"], points[metric], places=14)
                bounds = np.percentile(reference[method][metric], [2.5, 97.5])
                np.testing.assert_allclose([output[method][metric]["ci_low"], output[method][metric]["ci_high"]],
                                           bounds, rtol=0, atol=1e-14)
        np.testing.assert_allclose([delta["ci_low"], delta["ci_high"]],
                                   np.percentile(differences, [2.5, 97.5]), rtol=0, atol=1e-14)

    def test_all_single_class_draws_produce_no_false_intervals(self):
        with patch("glaboost.gpu_metrics.torch.randint", return_value=torch.zeros((20, 2), dtype=torch.int64)):
            metrics, delta, metadata = paired_bootstrap_gpu(
                [0, 1], [.1, .9], [.9, .1], ["p", "q"],
                EvaluationConfig(bootstrap_replicates=20), device="cpu")
        self.assertEqual(metadata["valid"], 0)
        self.assertEqual(metadata["skipped_single_class"], 20)
        self.assertEqual(metadata["ci_status"], "not_estimable")
        self.assertEqual(delta, {"estimate": -1., "ci_low": None, "ci_high": None})
        self.assertTrue(all(value["ci_low"] is None for method in metrics.values() for value in method.values()))

    def test_seeded_chunked_draws_are_reproducible_and_paired(self):
        y = np.array([0, 1, 0, 1, 0, 1])
        probabilities = np.array([.1, .2, .3, .7, .8, .9])
        args = (y, probabilities, probabilities, ["p", "p", "q", "q", "r", "r"],
                EvaluationConfig(bootstrap_replicates=300))
        first = paired_bootstrap_gpu(*args, device="cpu")
        second = paired_bootstrap_gpu(*args, device="cpu")
        self.assertEqual(first, second)
        metrics, delta, metadata = first
        self.assertEqual(metrics["latest"], metrics["longitudinal"])
        self.assertEqual(delta, {"estimate": 0., "ci_low": 0., "ci_high": 0.})
        self.assertEqual(metadata["valid"], 300)
        self.assertLessEqual(metadata["bootstrap_chunk_size"], 256)

    def test_cuda_unavailable_does_not_fall_back_to_cpu(self):
        with patch("glaboost.gpu_metrics.torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "no CPU fallback"):
                paired_bootstrap_gpu([0, 1], [.1, .9], [.1, .9], ["p", "q"],
                                     EvaluationConfig(bootstrap_replicates=20), device="cuda:0")

    def test_invalid_metric_inputs_fail(self):
        y = torch.tensor([0, 1])
        scores = torch.tensor([.1, .9])
        for target, probabilities, weights in (
            (y, scores, torch.tensor([[1., 0.]])),
            (y, scores, torch.tensor([[1., -1.]])),
            (y, torch.tensor([.1, float("nan")]), torch.ones((1, 2))),
            (torch.tensor([0, 2]), scores, torch.ones((1, 2))),
            (y, torch.tensor([1.1, .2]), torch.ones((1, 2))),
        ):
            with self.assertRaises(ValueError):
                _weighted_metrics_tensor(target, probabilities, weights)


if __name__ == "__main__":
    unittest.main()
