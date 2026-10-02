"""Synthetic checks for the notebook consolidation; no real images or downloads."""

from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

import Glaboost_CH as implementation


def description(ratio=0.7, risk="high", confidence=0.8, rim="thin"):
    return json.dumps({
        "fundus_features": {"cup_to_disc_ratio": ratio, "optic_disc_size": "large",
                            "isnt_rule_followed": False, "neuroretinal_rim": rim},
        "glaucoma_risk_assessment": risk,
        "confidence_level": confidence,
    })


def png_bytes(value=64):
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), (value, value, value)).save(buffer, format="PNG")
    return buffer.getvalue()


def mocked_encoders(batch_size=2):
    """Use real tensor batching with tiny callable stand-ins for pretrained nets."""
    import torch

    encoders = object.__new__(implementation._FrozenEncoders)
    encoders.device, encoders.batch_size = "cpu", batch_size
    encoders.preprocess = lambda image: torch.from_numpy(
        np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1)
    encoders.image_model = lambda pixels: pixels.mean(dim=(1, 2, 3)).reshape(-1, 1).repeat(1, 512)

    def tokenizer(texts, **options):
        if options != {"padding": True, "truncation": True, "max_length": 32,
                       "return_tensors": "pt"}:
            raise AssertionError(f"Unexpected tokenizer settings: {options}")
        return {"input_ids": torch.tensor([[len(text)] for text in texts]),
                "attention_mask": torch.ones((len(texts), 1), dtype=torch.long)}

    encoders.tokenizer = tokenizer
    encoders.text_model = lambda input_ids, attention_mask: input_ids.float().repeat(1, 768)
    return encoders


class GlaboostCHTests(unittest.TestCase):
    def test_schema_uses_training_categories_and_preserves_missing_numbers(self):
        schema = implementation._TabularSchema()
        train = pd.DataFrame({"ratio": [0.1, None], "disc": ["small", "large"]})
        train_features = schema.transform(train, fit=True)
        state = copy.deepcopy(schema.state)
        validation = pd.DataFrame({"ratio": [None, 0.9], "disc": ["unseen", None]})
        features = schema.transform(validation)

        self.assertEqual(schema.state, state)
        self.assertEqual(features.shape, train_features.shape)
        self.assertNotIn("disc_unseen", schema.state["outputs"])
        ratio = schema.state["outputs"].index("ratio")
        missing = schema.state["outputs"].index("disc_nan")
        self.assertTrue(np.isnan(features[0, ratio]))
        self.assertEqual(features[1, ratio], np.float32(0.9))
        self.assertEqual(features[0, missing], 0)
        self.assertEqual(features[1, missing], 1)
        categorical = [i for i, name in enumerate(state["outputs"]) if name.startswith("disc_")]
        np.testing.assert_array_equal(features[0, categorical], 0)
        absent_numeric = schema.transform(pd.DataFrame({"disc": ["small"]}))
        self.assertTrue(np.isnan(absent_numeric[0, ratio]))

    def test_structured_fit_save_load_preserves_predictions_and_label_zero(self):
        labels = np.tile([0, 1], 20)
        train = pd.DataFrame({
            "description": [description(0.85 if label == 0 else 0.15,
                                        risk="shared") for label in labels],
            "label": labels,
            "annotation": ["glaucoma" if label == 0 else "normal" for label in labels],
        })
        validation = pd.DataFrame({"description": [description(0.9, "validation_only"),
                                                   description(0.1, "validation_only")],
                                   "label": [0, 1]}, index=[41, 43])
        model = implementation.GlaboostCH(mode="structured", device="cpu", glaucoma_label=0)
        with redirect_stdout(io.StringIO()):
            model.fit(train, validation)
        self.assertEqual(model.model.get_booster().num_boosted_rounds(), 100)
        self.assertFalse(any("validation_only" in name for name in model.feature_names))
        self.assertFalse(any("annotation" in name or name == "label" for name in model.feature_names))
        before = model.predict_proba(validation)
        self.assertEqual(before.shape, (2, 2))
        np.testing.assert_allclose(before.sum(axis=1), 1)
        self.assertGreater(before[0, 0], before[1, 0])

        with tempfile.TemporaryDirectory(prefix="glaboost_ch_test_", dir="/tmp") as temporary:
            bundle = Path(temporary) / "model"
            model.save(bundle)
            loaded = implementation.GlaboostCH.load(bundle, device="cpu")
            np.testing.assert_array_equal(loaded.predict_proba(validation), before)
            self.assertEqual(loaded.glaucoma_label, 0)
            self.assertEqual(loaded.feature_names, model.feature_names)
            manifest = json.loads((bundle / "manifest.json").read_text())
            self.assertEqual(manifest["probability_column_labels"], [0, 1])
            with self.assertRaises(FileExistsError):
                loaded.save(bundle)

        predictions = implementation._prediction_frame(loaded, validation, before)
        np.testing.assert_array_equal(predictions["probability_glaucoma"], before[:, 0])
        np.testing.assert_array_equal(predictions["source_row"], [41, 43])
        np.testing.assert_array_equal(predictions["predicted_label"], before.argmax(axis=1))

    def test_invalid_descriptions_report_original_row(self):
        for value in ("not json", "[]", '{"fundus_features": []}',
                      '{"fundus_features": {"nested": {}}}'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "Invalid description at row eye_9"):
                implementation._descriptions(pd.DataFrame({"description": [value]}, index=["eye_9"]))

    def test_literal_bytes_and_csv_relative_image_paths(self):
        encoded = png_bytes()
        image = implementation._open_image(repr({"bytes": encoded}))
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.getpixel((0, 0)), (64, 64, 64))
        with tempfile.TemporaryDirectory(prefix="glaboost_ch_images_", dir="/tmp") as temporary:
            root = Path(temporary)
            (root / "sample.png").write_bytes(encoded)
            frame = pd.DataFrame({"image": ["sample.png", repr({"bytes": None, "path": "sample.png"}),
                                            repr({"bytes": encoded})]})
            csv = root / "input.csv"
            frame.to_csv(csv, index=False)
            decoded = implementation._read_csv(csv)
            self.assertEqual(decoded.loc[0, "image"], str(root / "sample.png"))
            self.assertEqual(decoded.loc[1, "image"]["path"], str(root / "sample.png"))
            for value in decoded["image"]:
                self.assertEqual(implementation._open_image(value).getpixel((0, 0)), (64, 64, 64))

    def test_frozen_encoder_batches_include_final_singleton(self):
        frame = pd.DataFrame({"image": [repr({"bytes": png_bytes(value)}) for value in (10, 20, 30)]},
                             index=[3, 8, 12])
        descriptions = pd.DataFrame({"neuroretinal_rim": ["thin", None, "normal"]}, index=frame.index)
        with patch.object(implementation, "tqdm", side_effect=lambda iterable, **kwargs: iterable):
            images, texts = mocked_encoders().transform(frame, descriptions)
        self.assertEqual(images.shape, (3, 512))
        self.assertEqual(texts.shape, (3, 768))
        np.testing.assert_array_equal(images[:, 0], [10, 20, 30])
        np.testing.assert_array_equal(texts[:, 0], [4, 0, 6])

    def test_csv_group_ids_keep_the_same_type_across_files(self):
        with tempfile.TemporaryDirectory(prefix="glaboost_ch_groups_", dir="/tmp") as temporary:
            root = Path(temporary)
            train_csv, test_csv = root / "train.csv", root / "test.csv"
            train_csv.write_text("patient\n001\n12\n", encoding="utf-8")
            test_csv.write_text("patient\n001\np3\n", encoding="utf-8")
            train = implementation._read_csv(train_csv, group_column="patient")
            test = implementation._read_csv(test_csv, group_column="patient")
            self.assertEqual(train["patient"].tolist(), ["001", "12"])
            self.assertEqual(set(train["patient"]) & set(test["patient"]), {"001"})

    def test_invalid_image_reports_original_row(self):
        frame = pd.DataFrame({"image": [repr({"bytes": b"not a PNG"})]}, index=[42])
        descriptions = pd.DataFrame({"neuroretinal_rim": [""]}, index=frame.index)
        with patch.object(implementation, "tqdm", side_effect=lambda iterable, **kwargs: iterable):
            with self.assertRaisesRegex(ValueError, "Invalid image at row 42"):
                mocked_encoders().transform(frame, descriptions)

    def test_multimodal_fusion_order_singleton_and_dimension_validation(self):
        frame = pd.DataFrame({"description": [description(confidence=0.9)], "image": ["unused"]})
        model = implementation.GlaboostCH(mode="multimodal", device="cpu")
        images = np.full((1, 512), 11, dtype=np.float32)
        texts = np.full((1, 768), 22, dtype=np.float32)
        model.encoders = SimpleNamespace(transform=lambda frame, descriptions: (images, texts))
        features = model._features(frame, fit=True)
        structured_width = len(model.schema.state["outputs"])
        risk_width = len(model.risk_schema.state["outputs"])
        self.assertEqual(features.shape, (1, structured_width + 512 + 768 + risk_width + 1))
        self.assertEqual(len(model.feature_names), features.shape[1])
        self.assertEqual(model.feature_names[structured_width], "img_0")
        self.assertEqual(model.feature_names[structured_width + 512], "rim_emb_0")
        self.assertEqual(model.feature_names[-1], "confidence_level")
        np.testing.assert_array_equal(features[:, structured_width:structured_width + 512], images)
        np.testing.assert_array_equal(features[:, structured_width + 512:structured_width + 1280], texts)
        np.testing.assert_allclose(features[:, -1], [0.9])
        expected_risk = model.risk_schema.transform(implementation._descriptions(frame)[["glaucoma_risk_assessment"]])
        np.testing.assert_array_equal(features[:, -risk_width - 1:-1], expected_risk)
        model.encoders = SimpleNamespace(transform=lambda frame, descriptions: (images.reshape(512), texts))
        with self.assertRaisesRegex(ValueError, "512-dimensional"):
            model._features(frame)


if __name__ == "__main__":
    unittest.main()
