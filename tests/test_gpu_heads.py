"""Tiny float64 CPU reference tests of the explicit-device CUDA algorithms.

No GPU, real cohort, network download, or research experiment is used here.
"""

import copy
from contextlib import contextmanager
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from glaboost.gpu_heads import fit_fold_gpu, initialize_cuda_linalg, temporal_features_gpu
from glaboost.longitudinal import ENDPOINTS, temporal_features


def reference(train, test, labels, c=1.0):
    scaler = StandardScaler().fit(train)
    model = LogisticRegression(C=c, penalty="l2", class_weight="balanced", solver="liblinear",
                               tol=1e-4, max_iter=2000, random_state=42).fit(scaler.transform(train), labels)
    return model.predict_proba(scaler.transform(test))[:, 1], scaler, model


def synthetic_records():
    return [{"eye_id": "eye_" + str(i), "patient_id": "patient_" + str(i),
             "times": times, "scores": scores, "labels": dict.fromkeys(ENDPOINTS, i % 2)}
            for i, (times, scores) in enumerate((
                ([0, 1, 3], [.2, .5, .8]),
                ([.4, .9, 1.4, 3.7, 4.2], [0, .5, .51, 1, .9]),
                ([0, .01, .05, 2], [.6, .6, .6, .6]),
            ))]


class GPUHeadTests(unittest.TestCase):
    def test_cuda_initialization_serializes_devices_without_consuming_randomness(self):
        """Simulate device placement; solve only constant tiny CPU test matrices."""
        import torch

        events, current = [], []
        cpu_eye, cpu_ones, solve = torch.eye, torch.ones, torch.linalg.solve
        rng_before = torch.random.get_rng_state().clone()

        @contextmanager
        def device_context(device):
            self.assertEqual(device.type, "cuda")
            self.assertEqual(current, [])
            current.append(device.index)
            events.append(("enter", device.index))
            try:
                yield
            finally:
                events.append(("exit", current.pop()))

        def allocate(factory, size, *, dtype, device):
            self.assertEqual(current, [device.index])
            self.assertEqual(device.type, "cuda")
            self.assertEqual(dtype, torch.float64)
            return factory(size, dtype=dtype, device="cpu")

        def checked_solve(matrix, rhs):
            self.assertFalse(torch.is_grad_enabled())
            events.append(("solve", current[0], len(rhs)))
            result = solve(matrix, rhs)
            torch.testing.assert_close(result, rhs)
            return result

        def synchronize(device):
            self.assertEqual(current, [device.index])
            events.append(("sync", device.index))

        with patch("torch.cuda.device", side_effect=device_context), \
                patch("torch.eye", side_effect=lambda *args, **kwargs: allocate(cpu_eye, *args, **kwargs)), \
                patch("torch.ones", side_effect=lambda *args, **kwargs: allocate(cpu_ones, *args, **kwargs)), \
                patch("torch.linalg.solve", side_effect=checked_solve), \
                patch("torch.cuda.synchronize", side_effect=synchronize):
            initialize_cuda_linalg((0, 2))
        self.assertEqual(events, [
            ("enter", 0), ("solve", 0, 2), ("solve", 0, 6), ("sync", 0), ("exit", 0),
            ("enter", 2), ("solve", 2, 2), ("solve", 2, 6), ("sync", 2), ("exit", 2),
        ])
        self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))

    def test_cuda_initialization_propagates_backend_errors_without_fallback(self):
        with patch("torch.cuda.device"), \
                patch("torch.eye", side_effect=RuntimeError("CUDA backend failure")) as allocate, \
                patch("torch.linalg.solve") as solve:
            with self.assertRaisesRegex(RuntimeError, "CUDA backend failure"):
                initialize_cuda_linalg((0, 1))
        self.assertEqual(allocate.call_count, 1)
        self.assertEqual(str(allocate.call_args.kwargs["device"]), "cuda:0")
        solve.assert_not_called()

    def test_single_and_five_feature_heads_match_liblinear_reference(self):
        rng = np.random.default_rng(42)
        for features, positives, c in ((1, 40, 1.0), (5, 8, 1.0), (5, 3, .1), (5, 40, 4.0)):
            with self.subTest(features=features, positives=positives, c=c):
                train, test = rng.normal(size=(80, features)), rng.normal(size=(15, features))
                labels = np.zeros(80, dtype=int)
                labels[np.argsort(train[:, 0])[-positives:]] = 1
                expected, scaler, model = reference(train, test, labels, c)
                actual, meta = fit_fold_gpu(train, test, labels, SimpleNamespace(logistic_c=c), device="cpu")
                np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=0)
                np.testing.assert_allclose(meta["scaler_mean"], scaler.mean_, atol=1e-14)
                np.testing.assert_allclose(meta["scaler_variance"], scaler.var_, atol=1e-14)
                np.testing.assert_allclose(meta["scaler_scale"], scaler.scale_, atol=1e-14)
                self.assertLessEqual(meta["gradient_inf_norm"], 1e-4)
                self.assertEqual(meta["backend"], "torch")
                self.assertEqual(meta["dtype"], "float64")
                self.assertEqual(meta["device"], "cpu")
                self.assertEqual(meta["solver"], "damped_newton_armijo")
                self.assertTrue(meta["intercept_regularized"])
                json.dumps(meta, allow_nan=False)

    def test_regularized_intercept_objective_has_stationary_gradient(self):
        rng = np.random.default_rng(8)
        train = rng.normal(size=(70, 5))
        labels = (train[:, 0] + .2 * train[:, 1] > 1).astype(int)
        c = .03
        _, meta = fit_fold_gpu(train, train[:5], labels, SimpleNamespace(logistic_c=c), device="cpu")
        transformed = (train - meta["scaler_mean"]) / meta["scaler_scale"]
        design = np.column_stack((transformed, np.ones(len(train))))
        theta = np.r_[meta["coefficient_standardized"], meta["intercept"]]
        probabilities = 1 / (1 + np.exp(-(design @ theta)))
        weights = len(labels) / (2 * np.bincount(labels)[labels])
        gradient = theta + c * design.T @ (weights * (probabilities - labels))
        np.testing.assert_allclose(gradient, np.zeros(6), atol=1e-4, rtol=0)
        self.assertGreater(abs(meta["intercept"]), .01)
        unpenalized_intercept_gradient = gradient[-1] - theta[-1]
        self.assertGreater(abs(unpenalized_intercept_gradient), .01)

    def test_constant_and_numerically_constant_scaling_matches_standard_scaler(self):
        rng = np.random.default_rng(9)
        train = np.column_stack((np.ones(40), 1e6 + np.arange(40) * 1e-10,
                                 rng.normal(size=40), rng.normal(size=40), np.zeros(40)))
        test = train[::5].copy()
        labels = np.arange(40) % 2
        expected, scaler, _ = reference(train, test, labels)
        actual, meta = fit_fold_gpu(train, test, labels, SimpleNamespace(logistic_c=1.), device="cpu")
        np.testing.assert_allclose(meta["scaler_mean"], scaler.mean_, atol=1e-8)
        np.testing.assert_allclose(meta["scaler_variance"], scaler.var_, atol=1e-14)
        np.testing.assert_allclose(meta["scaler_scale"], scaler.scale_, atol=1e-14)
        self.assertEqual(meta["scaler_scale"][0], 1.)
        self.assertEqual(meta["scaler_scale"][1], 1.)
        self.assertEqual(meta["scaler_scale"][-1], 1.)
        np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=0)

    def test_test_distribution_cannot_change_scaler_or_fit(self):
        train = np.arange(20, dtype=float).reshape(10, 2)
        labels = np.array([0, 1] * 5)
        before = train.copy()
        _, original = fit_fold_gpu(train, train[:3], labels, SimpleNamespace(logistic_c=1.), device="cpu")
        _, shifted = fit_fold_gpu(train, np.full((3, 2), 10000.), labels, SimpleNamespace(logistic_c=1.), device="cpu")
        self.assertEqual(original, shifted)
        np.testing.assert_array_equal(train, before)
        np.testing.assert_allclose(original["scaler_mean"], train.mean(axis=0))

    def test_repeated_fits_are_identical_for_explicit_cpu_reference(self):
        train = np.arange(30, dtype=float).reshape(10, 3)
        labels = np.array([0, 0, 1, 0, 0, 1, 1, 0, 1, 1])
        args = train, train[:4], labels, SimpleNamespace(logistic_c=1.)
        first, first_meta = fit_fold_gpu(*args, device="cpu")
        second, second_meta = fit_fold_gpu(*args, device="cpu")
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first_meta, second_meta)

    def test_invalid_arrays_and_single_class_labels_fail(self):
        train, test, labels = np.arange(8.).reshape(4, 2), np.ones((2, 2)), np.array([0, 1, 0, 1])
        invalid = ((train[:, 0], test, labels), (train, test[:, :1], labels),
                   (train, np.empty((0, 2)), labels), (train, test, np.ones(4)),
                   (train, test, np.array([0, 1])), (train, test, np.array([0, 2, 0, 1])),
                   (train, np.full((2, 2), np.nan), labels), (train.astype(bool), test, labels))
        for arrays in invalid:
            with self.subTest(shapes=[np.shape(value) for value in arrays]), self.assertRaises(ValueError):
                fit_fold_gpu(*arrays, SimpleNamespace(logistic_c=1.), device="cpu")
        for c in (0, -1, np.inf, True):
            with self.subTest(c=c), self.assertRaises(ValueError):
                fit_fold_gpu(train, test, labels, SimpleNamespace(logistic_c=c), device="cpu")

    def test_nonconvergence_and_line_search_failure_cannot_return_predictions(self):
        train = np.arange(10.).reshape(-1, 1)
        labels = np.r_[np.zeros(5), np.ones(5)]
        args = train, train[:2], labels, SimpleNamespace(logistic_c=1.)
        with patch("glaboost.gpu_heads._MAX_ITER", 0), self.assertRaisesRegex(RuntimeError, "did not converge"):
            fit_fold_gpu(*args, device="cpu")
        with patch("glaboost.gpu_heads._MAX_LINE_SEARCH", 0), self.assertRaisesRegex(RuntimeError, "line search failed"):
            fit_fold_gpu(*args, device="cpu")

    def test_unavailable_cuda_never_falls_back_and_other_device_types_fail(self):
        with patch("torch.cuda.is_available", return_value=False), self.assertRaisesRegex(RuntimeError, "no CPU fallback"):
            fit_fold_gpu([[0], [1]], [[.5]], [0, 1], SimpleNamespace(logistic_c=1.), device="cuda:0")
        with patch("torch.cuda.is_available", return_value=False), self.assertRaisesRegex(RuntimeError, "no CPU fallback"):
            temporal_features_gpu([], device="cuda:0")
        with self.assertRaises(ValueError):
            temporal_features_gpu(synthetic_records(), device="meta")


class GPUTemporalTests(unittest.TestCase):
    def test_ragged_real_time_features_match_existing_semantics(self):
        records = synthetic_records()
        before = copy.deepcopy(records)
        actual = temporal_features_gpu(records, .5, device="cpu")
        np.testing.assert_allclose(actual, temporal_features(records, .5), atol=1e-14, rtol=1e-14)
        self.assertEqual(actual.dtype, np.float64)
        self.assertEqual(actual.shape, (3, 5))
        np.testing.assert_allclose(actual[0], [.8, .6, 27 / 140, .5, 1 / 3], atol=1e-14, rtol=0)
        self.assertEqual(records, before)
        for threshold in (0., 1.):
            np.testing.assert_allclose(temporal_features_gpu(records, threshold, device="cpu"),
                                       temporal_features(records, threshold), atol=1e-14, rtol=1e-14)

    def test_labels_and_identifiers_do_not_enter_feature_arithmetic(self):
        records = synthetic_records()
        changed = copy.deepcopy(records)
        for i, row in enumerate(changed):
            row.update(patient_id="replacement", eye_id="replacement_" + str(i),
                       labels=dict.fromkeys(ENDPOINTS, 1 - i % 2), future_score=100)
        np.testing.assert_array_equal(temporal_features_gpu(records, device="cpu"),
                                      temporal_features_gpu(changed, device="cpu"))

    def test_empty_and_invalid_temporal_records(self):
        self.assertEqual(temporal_features_gpu([], device="cpu").shape, (0, 5))
        records = synthetic_records()
        for threshold in (True, -.1, 1.1, np.nan):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                temporal_features_gpu(records, threshold, device="cpu")
        with self.assertRaisesRegex(ValueError, "duplicate eye_id"):
            temporal_features_gpu(records + [records[0]], device="cpu")
        records[0]["times"] = [0, 0, 2]
        with self.assertRaises(ValueError):
            temporal_features_gpu(records, device="cpu")


if __name__ == "__main__":
    unittest.main()
