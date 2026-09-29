"""Statistical and leakage checks on tiny synthetic cohorts only."""

import copy
import io
import json
import unittest
from unittest.mock import patch

import numpy as np
from tqdm import tqdm

from glaboost.longitudinal import (
    ENDPOINTS, FEATURE_NAMES, EvaluationConfig, _metrics,
    _paired_bootstrap, _patient_bootstrap_indices,
    evaluate_longitudinal, prepare_eye_records, temporal_features,
)


def cohort(n_patients=16):
    visits, labels = [], {}
    for patient in range(n_patients):
        for eye_index, laterality in enumerate(("OD", "OS")):
            eye_id = f"p{patient:02d}_{laterality}"
            label = patient % 2
            latest = 0.48 + 0.025 * (patient % 5) + 0.01 * eye_index
            scores = [latest - 0.25 * label, latest - 0.2 * label, latest]
            for visit, (time, score) in enumerate(zip((0.0, 0.5, 2.0), scores)):
                visits.append({
                    "sample_id": f"{eye_id}_{visit}", "patient_id": f"p{patient:02d}",
                    "eye_id": eye_id, "time_years": time, "glaucoma_score": score,
                })
            labels[eye_id] = dict.fromkeys(ENDPOINTS, label)
    return visits, labels


class LongitudinalTests(unittest.TestCase):
    def test_actual_time_ols_and_persistence_boundary(self):
        rows = [
            {"sample_id": f"v{i}", "patient_id": "p", "eye_id": "e", "time_years": t,
             "glaucoma_score": s}
            for i, (t, s) in enumerate(((0, 0.2), (1, 0.5), (3, 0.8)))
        ]
        records = prepare_eye_records(list(reversed(rows)), {"e": dict.fromkeys(ENDPOINTS, 0)})
        np.testing.assert_allclose(temporal_features(records)[0], [0.8, 0.6, 27 / 140, 0.5, 1 / 3])
        self.assertEqual(records[0]["n_visits"], 3)
        self.assertEqual(records[0]["followup_years"], 3)
        self.assertEqual(records[0]["times"], [0, 1, 3])

    def test_equal_latest_distinct_trajectories_and_no_metadata_features(self):
        rows, labels = cohort(2)
        # Both eyes here are manually assigned the same last score.
        records = prepare_eye_records(rows, labels)
        for record in records:
            record["scores"][-1] = 0.9
            record["future_score"] = 999
            record["diagnosis_label"] = 999
        features = temporal_features(records)
        self.assertEqual(features.shape, (4, 5))
        self.assertEqual(tuple(FEATURE_NAMES), ("last", "delta", "slope", "mean", "persistence"))
        self.assertTrue(np.all(features[:, 0] == 0.9))
        self.assertNotEqual(features[0, 1], features[2, 1])
        altered = copy.deepcopy(records)
        for i, record in enumerate(altered):
            record["eye_id"], record["patient_id"] = f"unrelated_{i}", "different"
            record["labels"] = {key: 1 - val for key, val in record["labels"].items()}
        np.testing.assert_array_equal(features, temporal_features(altered))

    def test_record_preparation_eligibility_and_no_input_mutation(self):
        rows, labels = cohort(2)
        rows = [r for r in rows if r["sample_id"] != "p00_OS_2"]
        before = copy.deepcopy(rows)
        records = prepare_eye_records(rows, labels)
        self.assertEqual(len(records), 3)
        self.assertEqual(rows, before)
        self.assertNotIn("p00_OS", {r["eye_id"] for r in records})
        self.assertEqual(prepare_eye_records(rows, labels, min_visits=4), [])
        self.assertEqual(prepare_eye_records([], {}), [])

    def test_duplicate_visits_times_and_conflicting_patients_fail(self):
        rows, labels = cohort(2)
        bad_inputs = []
        bad_inputs.append(rows + [rows[0]])
        changed = copy.deepcopy(rows)
        changed[1]["time_years"] = changed[0]["time_years"]
        bad_inputs.append(changed)
        changed = copy.deepcopy(rows)
        changed[1]["patient_id"] = "other patient"
        bad_inputs.append(changed)
        for bad in bad_inputs:
            with self.subTest(rows=bad[:2]):
                with self.assertRaises(ValueError):
                    prepare_eye_records(bad, labels)

    def test_invalid_inputs_cannot_be_silently_scored(self):
        rows, labels = cohort(2)
        for column, value in (("time_years", -1), ("time_years", np.inf),
                              ("glaucoma_score", np.nan), ("glaucoma_score", 1.01),
                              ("glaucoma_score", True), ("patient_id", "")):
            bad = copy.deepcopy(rows)
            bad[0][column] = value
            with self.subTest(column=column, value=value):
                with self.assertRaises(ValueError):
                    prepare_eye_records(bad, labels)
        for value in (None, True, 0.0, 2):
            bad_labels = copy.deepcopy(labels)
            bad_labels["p00_OD"]["plr2"] = value
            with self.subTest(label=value):
                with self.assertRaises(ValueError):
                    prepare_eye_records(rows, bad_labels)
        for minimum in (True, 2, 3.5):
            with self.assertRaises(ValueError):
                prepare_eye_records(rows, labels, min_visits=minimum)

    def test_evaluation_config_is_strict_and_json_safe(self):
        for kwargs in ({"n_splits": 1}, {"bootstrap_replicates": 19}, {"seed": -1},
                       {"seed": True}, {"logistic_c": 0}, {"persistence_threshold": np.nan},
                       {"persistence_threshold": 1.1}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    EvaluationConfig(**kwargs)
        config = EvaluationConfig(n_splits=np.int64(3), bootstrap_replicates=np.int64(20))
        self.assertIs(type(config.n_splits), int)

    def test_patient_grouped_paired_oof_and_train_only_scaling(self):
        rows, labels = cohort()
        # Noncontiguous input eyes must not break patient grouping.
        records = prepare_eye_records(rows, labels)
        records = records[::2] + records[1::2]
        config = EvaluationConfig(n_splits=3, bootstrap_replicates=25)
        result = evaluate_longitudinal(records, config)
        json.dumps(result, allow_nan=False)
        features = {row["eye_id"]: row for row in result["features"]}
        for endpoint in ENDPOINTS:
            summary = result["endpoints"][endpoint]
            self.assertEqual(summary["status"], "ok")
            predictions = [p for p in result["predictions"] if p["endpoint"] == endpoint]
            self.assertEqual(len(predictions), len(records))
            self.assertEqual(len({p["eye_id"] for p in predictions}), len(records))
            patient_folds = {}
            for prediction in predictions:
                patient_folds.setdefault(prediction["patient_id"], set()).add(prediction["fold"])
            self.assertTrue(all(len(folds) == 1 for folds in patient_folds.values()))
            for fold in summary["folds"]:
                self.assertFalse(set(fold["train_patient_ids"]) & set(fold["test_patient_ids"]))
                train = np.asarray([[features[eye][name] for name in FEATURE_NAMES] for eye in fold["train_eye_ids"]])
                np.testing.assert_allclose(fold["longitudinal"]["scaler_mean"], train.mean(axis=0))
                np.testing.assert_allclose(fold["latest"]["scaler_mean"], train[:, :1].mean(axis=0))
                for method in ("latest", "longitudinal"):
                    fit = fold[method]
                    self.assertGreater(fold["train_class_counts"]["positive"], 0)
                    self.assertGreater(fold["test_class_counts"]["negative"], 0)
                    for row in predictions:
                        if row["fold"] != fold["fold"]:
                            continue
                        values = np.asarray([features[row["eye_id"]][name] for name in fit["feature_names"]])
                        scaled = (values - fit["scaler_mean"]) / fit["scaler_scale"]
                        logit = np.dot(scaled, fit["coefficient_standardized"]) + fit["intercept"]
                        self.assertAlmostEqual(row[f"{method}_probability"], 1 / (1 + np.exp(-logit)))
            reference = np.asarray([p["y_true"] for p in predictions])
            for method in ("latest", "longitudinal"):
                expected = _metrics(reference, np.asarray([p[f"{method}_probability"] for p in predictions]))
                for metric, value in expected.items():
                    self.assertEqual(summary["metrics"][method][metric]["estimate"], value)

    def test_results_invariant_to_input_order(self):
        rows, labels = cohort(12)
        records = prepare_eye_records(rows, labels)
        config = EvaluationConfig(n_splits=3, bootstrap_replicates=20)
        self.assertEqual(evaluate_longitudinal(records, config), evaluate_longitudinal(records[::-1], config))

    def test_bootstrap_preserves_both_eyes_and_duplicate_patient_multiplicity(self):
        class FakeRandom:
            def choice(self, patients, size, replace):
                self.assertion = (patients.tolist(), size, replace)
                return np.array(["p", "p", "q"])
        rng = FakeRandom()
        indices = _patient_bootstrap_indices(np.array(["p", "q", "p", "r", "q"]), rng)
        np.testing.assert_array_equal(indices, [0, 2, 0, 2, 1, 4])
        self.assertEqual(rng.assertion, (["p", "q", "r"], 3, True))

    def test_bootstrap_pairs_methods_and_reports_its_conditional_scope(self):
        y = np.array([0, 1, 0, 1])
        probabilities = np.array([0.1, 0.2, 0.3, 0.4])
        metrics, delta, metadata = _paired_bootstrap(
            y, probabilities, probabilities.copy(), np.array(["p", "p", "q", "q"]),
            EvaluationConfig(bootstrap_replicates=25),
        )
        self.assertEqual(delta, {"estimate": 0.0, "ci_low": 0.0, "ci_high": 0.0})
        self.assertEqual(metrics["latest"], metrics["longitudinal"])
        self.assertEqual(metadata["valid"], 25)
        self.assertIn("no model refitting", metadata["conditional_on"])

    def test_single_class_bootstrap_draws_skipped_not_given_false_intervals(self):
        y = np.array([0, 1])
        with patch("glaboost.longitudinal._patient_bootstrap_indices", return_value=np.array([0, 0])):
            metrics, delta, metadata = _paired_bootstrap(
                y, np.array([0.2, 0.8]), np.array([0.8, 0.2]), np.array(["p", "q"]),
                EvaluationConfig(bootstrap_replicates=20),
            )
        self.assertEqual(metadata["valid"], 0)
        self.assertEqual(metadata["skipped_single_class"], 20)
        self.assertEqual(metadata["ci_status"], "not_estimable")
        self.assertIsNone(delta["ci_low"])
        self.assertEqual(delta["estimate"], -1)
        self.assertIsNone(metrics["latest"]["auroc"]["ci_high"])

    def test_bootstrap_progress_counts_skipped_draws_without_changing_sampling(self):
        output = io.StringIO()
        groups = np.array(["p", "q"])
        draws = [np.array([0, 0]), np.array([0, 1])] * 10
        with patch("glaboost.longitudinal.tqdm", side_effect=lambda *args, **kwargs:
                   tqdm(*args, **kwargs, file=output, disable=False, mininterval=0)), \
             patch("glaboost.longitudinal._patient_bootstrap_indices", side_effect=draws) as sample:
            _, _, metadata = _paired_bootstrap(
                np.array([0, 1]), np.array([0.2, 0.8]), np.array([0.8, 0.2]), groups,
                EvaluationConfig(bootstrap_replicates=20), progress_desc="plr2: paired patient bootstrap",
            )
        self.assertEqual(sample.call_count, 20)
        self.assertEqual(metadata["valid"], 10)
        self.assertEqual(metadata["skipped_single_class"], 10)
        self.assertIn("plr2: paired patient bootstrap", output.getvalue())
        self.assertIn("20/20", output.getvalue())
        self.assertIn("skipped=10", output.getvalue())
        self.assertIn("valid=10", output.getvalue())

    def test_insufficient_positive_patients_not_estimable_even_with_two_eyes(self):
        rows, labels = cohort(4)
        for eye in labels:
            labels[eye]["plr3"] = int(eye.startswith("p01"))
        records = prepare_eye_records(rows, labels)
        result = evaluate_longitudinal(records, EvaluationConfig(n_splits=2, bootstrap_replicates=20))
        summary = result["endpoints"]["plr3"]
        self.assertEqual(summary["status"], "not_estimable")
        self.assertEqual(summary["n_positive_eyes"], 2)
        self.assertEqual(summary["n_positive_patients"], 1)
        self.assertNotIn("metrics", summary)
        self.assertFalse(any(row["endpoint"] == "plr3" for row in result["predictions"]))

    def test_empty_cohort_is_explicitly_not_estimable(self):
        result = evaluate_longitudinal([], EvaluationConfig(bootstrap_replicates=20))
        self.assertEqual(result["predictions"], [])
        self.assertTrue(all(value["status"] == "not_estimable" for value in result["endpoints"].values()))
        json.dumps(result, allow_nan=False)

    def test_direct_eye_record_validation(self):
        rows, labels = cohort(2)
        records = prepare_eye_records(rows, labels)
        for field, value in (("n_visits", 10), ("followup_years", 10), ("times", [0, 0, 2]),
                             ("scores", [0.1, 0.2]), ("scores", [0.1, 0.2, np.inf])):
            bad = copy.deepcopy(records)
            bad[0][field] = value
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValueError):
                    evaluate_longitudinal(bad, EvaluationConfig(bootstrap_replicates=20))
        with self.assertRaisesRegex(ValueError, "duplicate eye_id"):
            temporal_features(records + [records[0]])


if __name__ == "__main__":
    unittest.main()
