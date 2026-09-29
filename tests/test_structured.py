import copy
import json
import unittest

import numpy as np

from glaboost.structured import StructuredEncoder, encode_oct_status


class StructuredEncoderTests(unittest.TestCase):
    def test_training_statistics_and_categories_do_not_leak(self):
        encoder = StructuredEncoder(["value"], ["category"])
        encoder.fit([{"value": 1, "category": " A "}, {"value": 3, "category": "b"}, {}])
        before = encoder.to_dict()
        actual = encoder.transform([{"value": 100, "category": "unseen"}, {}])
        np.testing.assert_allclose(actual[0, 0], 98 / np.sqrt(2 / 3), rtol=1e-6)
        np.testing.assert_array_equal(actual[:, 1], [0, 1])
        np.testing.assert_array_equal(actual[:, 2:], [[0, 0, 0, 1], [0, 0, 1, 0]])
        self.assertEqual(encoder.to_dict(), before)
        self.assertEqual(actual.dtype, np.float32)

    def test_all_missing_training_features_remain_usable(self):
        encoder = StructuredEncoder(["value"], ["category"])
        actual = encoder.fit_transform([{}, {"value": np.nan, "category": np.float32("nan")}])
        np.testing.assert_array_equal(actual, [[0, 1, 1, 0], [0, 1, 1, 0]])
        np.testing.assert_array_equal(encoder.transform([{"value": 9, "category": "new"}]), [[9, 0, 0, 1]])

    def test_constant_values_have_unit_scale(self):
        encoder = StructuredEncoder(["value"]).fit([{"value": 0.1}] * 3)
        self.assertEqual(encoder.to_dict()["numeric_stats"]["value"]["scale"], 1.0)
        np.testing.assert_allclose(encoder.transform([{"value": 1.1}]), [[1, 0]])

    def test_no_standardization_keeps_confidence_scale(self):
        encoder = StructuredEncoder(["confidence"], standardize=False)
        actual = encoder.fit_transform([{"confidence": 0.2}, {"confidence": 0.8}, {}])
        np.testing.assert_allclose(actual, [[0.2, 0], [0.8, 0], [0.5, 1]])

    def test_boolean_and_string_category_normalization(self):
        encoder = StructuredEncoder(categorical_features=["flag"])
        actual = encoder.fit_transform([{"flag": np.bool_(True)}, {"flag": " TRUE "}, {"flag": False}])
        np.testing.assert_array_equal(actual, [[0, 1, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0]])

    def test_real_category_cannot_collide_with_missing_or_unknown(self):
        encoder = StructuredEncoder(categorical_features=["category"])
        encoder.fit([{"category": "<missing>"}, {"category": "<unknown>"}])
        actual = encoder.transform([{"category": "<missing>"}, {}, {"category": "unseen"}])
        np.testing.assert_array_equal(actual, [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.assertEqual(len(set(encoder.feature_names_)), actual.shape[1])

    def test_only_explicit_fields_are_read(self):
        encoder = StructuredEncoder(["value"])
        encoder.fit([{"value": 1, "label": {"arbitrary": "ignored"}}, {"value": 3}])
        np.testing.assert_array_equal(encoder.transform([{"value": 2, "diagnosis": object()}]), [[0, 0]])
        self.assertEqual(len(encoder.feature_names_), 2)

    def test_json_roundtrip_preserves_feature_order_and_output(self):
        encoder = StructuredEncoder(["value"], ["b", "a"]).fit([
            {"value": 1, "b": "X", "a": True}, {"value": 3, "b": None, "a": False}
        ])
        restored = StructuredEncoder.from_dict(json.loads(json.dumps(encoder.to_dict(), allow_nan=False)))
        rows = [{"value": 5, "b": "unknown", "a": True}, {}]
        np.testing.assert_array_equal(restored.transform(rows), encoder.transform(rows))
        self.assertEqual(restored.feature_names_, encoder.feature_names_)
        exported = restored.to_dict()
        exported["numeric_stats"]["value"]["median"] = 999
        exported["categories"]["a"].append("mutation")
        self.assertEqual(restored.to_dict(), encoder.to_dict())

    def test_empty_transform_has_fitted_width(self):
        encoder = StructuredEncoder(["value"], ["category"]).fit([{}])
        self.assertEqual(encoder.transform([]).shape, (0, 4))

    def test_invalid_constructor_inputs_are_rejected(self):
        for kwargs in [{}, {"numeric_features": ["x", "x"]}, {"categorical_features": ["x", "x"]},
                       {"numeric_features": ["x"], "categorical_features": ["x"]},
                       {"numeric_features": "value"}, {"numeric_features": [""]},
                       {"numeric_features": ["x"], "standardize": "yes"}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                StructuredEncoder(**kwargs)

    def test_unfitted_or_empty_fit_is_rejected(self):
        encoder = StructuredEncoder(["value"])
        for call in [lambda: encoder.fit([]), lambda: encoder.transform([{}]), encoder.to_dict,
                     lambda: encoder.feature_names_, lambda: encoder.fit(["bad row"])]:
            with self.assertRaises(ValueError):
                call()

    def test_invalid_numeric_values_are_rejected_during_fit_and_inference(self):
        encoder = StructuredEncoder(["value"]).fit([{"value": 1}, {"value": 3}])
        for value in ["abc", "", float("inf"), float("-inf"), "nan", [1], {}, 1 + 2j, np.array([1]), 10**1000]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    StructuredEncoder(["value"]).fit([{"value": value}])
                with self.assertRaises(ValueError):
                    encoder.transform([{"value": value}])
        with self.assertRaises(ValueError):
            encoder.transform([{"value": 1e100}])

    def test_invalid_serialized_state_is_rejected(self):
        state = StructuredEncoder(["value"], ["category"]).fit([{"value": 1, "category": "x"}]).to_dict()
        bad_states = [{}, {**state, "version": 999}, {**state, "categories": {"category": ["x", "x"]}}]
        bad_scale = copy.deepcopy(state)
        bad_scale["numeric_stats"]["value"]["scale"] = 0
        bad_states.append(bad_scale)
        for bad in bad_states:
            with self.subTest(state=bad), self.assertRaises(ValueError):
                StructuredEncoder.from_dict(bad)

    def test_oct_status_uses_explicit_paper_mapping(self):
        self.assertEqual(encode_oct_status("Outside Normal"), 0.0)
        self.assertEqual(encode_oct_status(" borderline "), 0.5)
        self.assertEqual(encode_oct_status("WITHIN  NORMAL"), 1.0)
        self.assertIsNone(encode_oct_status(None))
        self.assertIsNone(encode_oct_status(np.nan))
        for value in ["normal", "", 1]:
            with self.assertRaises(ValueError):
                encode_oct_status(value)


if __name__ == "__main__":
    unittest.main()
