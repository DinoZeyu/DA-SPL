"""Tiny synthetic Parquet/image audits; no downloads or diagnostic training."""

from contextlib import redirect_stderr
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

from glaboost.hf_data import HF_REPO_ID, _rgb_sha256, load_hf_diagnosis
from glaboost.study import sha256_file


def image_bytes(color, *, kind="PNG", size=(3, 2)):
    stream = io.BytesIO()
    with Image.new("RGB", size, color) as image:
        image.save(stream, format=kind)
    return stream.getvalue()


class HFDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="synthetic_hf_diagnosis_")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.root, self.grape = self.directory / "hf", self.directory / "grape"
        (self.root / "data").mkdir(parents=True)
        self.cfps = self.grape / "extracted/CFPs"
        self.cfps.mkdir(parents=True)
        (self.cfps / "synthetic_grape.png").write_bytes(image_bytes((13, 14, 15)))
        self.rows = {}
        for split, colors in (("train", ((1, 2, 3), (4, 5, 6))), ("test", ((7, 8, 9), (10, 11, 12)))):
            self.rows[split] = [
                {"image": {"bytes": image_bytes(color), "path": f"unused_{index}.png"},
                 "label": index, "filename": f"synthetic_{'glaucoma' if index == 0 else 'normal'}_{split}.png",
                 "annotation": "glaucoma" if index == 0 else "normal",
                 "description": "SYNTHETIC diagnostic text; must never become a model feature"}
                for index, color in enumerate(colors)]
        self.schema = pa.schema([("image", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
                                 ("label", pa.int64()), ("filename", pa.string()),
                                 ("annotation", pa.string()), ("description", pa.string())])

    def save(self):
        checksums = {}
        for split, rows in self.rows.items():
            relative = f"data/{split}-00000-of-00001.parquet"
            pq.write_table(pa.Table.from_pylist(rows, schema=self.schema), self.root / relative)
            checksums[relative] = sha256_file(self.root / relative)
        (self.root / "provenance.json").write_text(json.dumps({
            "repo_id": HF_REPO_ID, "revision": "0" * 40, "files": checksums}))

    def load(self):
        with redirect_stderr(io.StringIO()):
            return load_hf_diagnosis(self.root, grape_root=self.grape)

    def test_released_partitions_label_inversion_immutable_bytes_and_no_text_features(self):
        self.save()
        paths = [*self.root.rglob("*.parquet"), self.root / "provenance.json", *self.cfps.iterdir()]
        before = {path: sha256_file(path) for path in paths}
        dataset = self.load()
        np.testing.assert_array_equal(dataset.train_labels, [1, 0])
        np.testing.assert_array_equal(dataset.test_labels, [1, 0])
        self.assertEqual([v.image for v in dataset.train_visits], [r["image"]["bytes"] for r in self.rows["train"]])
        for visit in (*dataset.train_visits, *dataset.test_visits):
            self.assertIsInstance(visit.image, bytes)
            self.assertEqual(visit.structured, {})
            self.assertEqual(visit.human, {})
            self.assertIsNone(visit.text)
            self.assertIsNone(visit.patient_id)
            self.assertIsNone(visit.eye_id)
        with self.assertRaises(ValueError):
            dataset.train_labels[0] = 0
        with self.assertRaises(ValueError):
            dataset.train_labels.setflags(write=True)
        audit = dataset.audit
        self.assertEqual(audit["repo_id"], HF_REPO_ID)
        self.assertEqual(audit["revision"], "0" * 40)
        self.assertEqual(audit["label_mapping"]["source_to_model"], {"0": 1, "1": 0})
        self.assertEqual(audit["overlap_audit"]["hf_grape_exact_rgb_matches"], 0)
        self.assertEqual(audit["overlap_audit"]["grape"]["n_images"], 1)
        self.assertFalse(audit["overlap_audit"]["grape"]["progression_labels_read"])
        self.assertNotIn("description", audit["samples"][0])
        self.assertNotIn("annotation", audit["samples"][0])
        json.dumps(audit, allow_nan=False)
        self.assertEqual({path: sha256_file(path) for path in paths}, before)

    def test_within_split_pixel_duplicates_keep_first_even_different_containers(self):
        duplicate = deepcopy(self.rows["train"][0])
        duplicate["filename"] = "synthetic_glaucoma_duplicate.bmp"
        duplicate["image"]["bytes"] = image_bytes((1, 2, 3), kind="BMP")
        self.rows["train"].append(duplicate)
        self.save()
        dataset = self.load()
        self.assertEqual(len(dataset.train_visits), 2)
        stats = dataset.audit["splits"]["train"]
        self.assertEqual((stats["released_rows"], stats["retained_rows"], stats["within_split_duplicates_removed"]), (3, 2, 1))
        skipped = next(row for row in dataset.audit["samples"] if not row["retained"])
        self.assertEqual(skipped["duplicate_of"], dataset.train_visits[0].sample_id)
        first = dataset.audit["samples"][0]
        self.assertEqual(first["rgb_sha256"], skipped["rgb_sha256"])
        self.assertNotEqual(first["encoded_sha256"], skipped["encoded_sha256"])

    def test_conflicting_within_split_duplicate_labels_fail(self):
        self.rows["train"][1]["image"]["bytes"] = self.rows["train"][0]["image"]["bytes"]
        self.save()
        with self.assertRaisesRegex(ValueError, "Conflicting diagnosis labels"):
            self.load()

    def test_across_split_overlap_excludes_test_row_and_preserves_training(self):
        duplicate = deepcopy(self.rows["test"][0])
        duplicate["filename"] = "synthetic_glaucoma_train_overlap.bmp"
        duplicate["image"]["bytes"] = image_bytes((1, 2, 3), kind="BMP")
        self.rows["test"].insert(0, duplicate)
        self.save()
        dataset = self.load()
        self.assertEqual((len(dataset.train_visits), len(dataset.test_visits)), (2, 2))
        self.assertEqual(dataset.train_visits[0].image, self.rows["train"][0]["image"]["bytes"])
        self.assertEqual(dataset.audit["splits"]["test"]["train_overlap_duplicates_removed"], 1)
        overlaps = dataset.audit["overlap_audit"]
        self.assertEqual(overlaps["train_test_exact_rgb_match_rows"], 1)
        self.assertEqual(overlaps["retained_train_test_exact_rgb_matches"], 0)
        self.assertEqual(overlaps["removed_test_duplicates"][0]["duplicate_of"], dataset.train_visits[0].sample_id)

    def test_conflicting_labels_across_splits_fail_instead_of_silent_exclusion(self):
        self.rows["test"][1]["image"]["bytes"] = image_bytes((1, 2, 3), kind="BMP")
        self.save()
        with self.assertRaisesRegex(ValueError, "Conflicting diagnosis labels"):
            self.load()

    def test_any_original_grape_image_overlap_blocks_training_inputs(self):
        # No workbook or eligibility information is read; all original CFPs count.
        (self.cfps / "outside_eligible_cohort.bmp").write_bytes(image_bytes((1, 2, 3), kind="BMP"))
        self.save()
        with self.assertRaisesRegex(ValueError, "overlap between HF train.*GRAPE"):
            self.load()

    def test_parquet_checksum_and_annotation_polarity_are_verified(self):
        self.save()
        path = self.root / "data/train-00000-of-00001.parquet"
        path.write_bytes(path.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "Parquet checksum mismatch"):
            self.load()
        self.rows["train"][0]["annotation"] = "normal"
        self.save()
        with self.assertRaisesRegex(ValueError, "annotation contradicts"):
            self.load()

    def test_paths_cannot_replace_embedded_image_bytes_and_corrupt_images_fail(self):
        self.rows["train"][0]["image"] = {"bytes": None, "path": str(self.cfps / "synthetic_grape.png")}
        self.save()
        with self.assertRaisesRegex(ValueError, "embedded image bytes are required"):
            self.load()
        self.rows["train"][0]["image"]["bytes"] = b"not an image"
        self.save()
        with self.assertRaisesRegex(ValueError, "Cannot decode"):
            self.load()

    def test_pixel_hash_includes_dimensions(self):
        left = _rgb_sha256(io.BytesIO(image_bytes((1, 2, 3), size=(2, 3))))
        right = _rgb_sha256(io.BytesIO(image_bytes((1, 2, 3), size=(3, 2))))
        self.assertNotEqual(left[0], right[0])


if __name__ == "__main__":
    unittest.main()
