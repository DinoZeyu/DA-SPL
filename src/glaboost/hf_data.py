"""Read the retained HF diagnosis release without modifying or extracting it.

Only immutable encoded image bytes enter VisitInput. Annotation strings verify
the source label convention; descriptions and diagnostic text are never features.
Exact decoded RGB duplicates are checked before any model fitting is possible.
"""

from collections import Counter
from dataclasses import dataclass
import hashlib
import io
from numbers import Integral
from pathlib import Path
import re
from typing import Tuple

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from .data import VisitInput
from .study import _read_json, sha256_file


HF_REPO_ID = "AswanthCManoj/glaucoma_diagnosis_json_analysis"
_SOURCE_ANNOTATIONS = {0: "glaucoma", 1: "normal"}
_PIXEL_HASH_METHOD = "SHA256 of RGB-v1 marker, big-endian width/height, and unresized row-major decoded RGB pixels"


@dataclass(frozen=True)
class HFDiagnosticDataset:
    train_visits: Tuple[VisitInput, ...]
    train_labels: np.ndarray
    test_visits: Tuple[VisitInput, ...]
    test_labels: np.ndarray
    audit: dict


def _rgb_sha256(source):
    """Hash one decoded image at a time, without normalization or resizing."""
    try:
        with Image.open(source) as opened:
            if getattr(opened, "n_frames", 1) != 1:
                raise ValueError("Expected a single-frame fundus image.")
            rgb = opened.convert("RGB")
            try:
                width, height = rgb.size
                if width < 1 or height < 1:
                    raise ValueError("Fundus image dimensions must be positive.")
                digest = hashlib.sha256(b"RGB-v1\0")
                digest.update(width.to_bytes(8, "big"))
                digest.update(height.to_bytes(8, "big"))
                # A stripe bounds the extra pixel-byte buffer independently of
                # image height; no collection of decoded images is retained.
                for top in range(0, height, 64):
                    with rgb.crop((0, top, width, min(top + 64, height))) as stripe:
                        digest.update(stripe.tobytes())
                return digest.hexdigest(), width, height
            finally:
                rgb.close()
    except (OSError, SyntaxError) as exc:
        raise ValueError("Cannot decode the supplied fundus image.") from exc


def _readonly_labels(values):
    return np.frombuffer(np.asarray(values, dtype=np.int64).tobytes(), dtype=np.int64)


def _filename_source(filename):
    for marker in ("_glaucoma_", "_normal_"):
        if marker in filename:
            return filename.split(marker, 1)[0]
    return "unidentified_filename_prefix"


def _audit_grape_images(grape_root):
    directory = (Path(grape_root).expanduser().resolve() / "extracted" / "CFPs").resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Original GRAPE CFP directory is missing: {directory}")
    extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
    paths = sorted(path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in extensions)
    if not paths:
        raise ValueError("No original GRAPE CFP images are available for the overlap audit.")
    hashes, rows = {}, []
    for path in tqdm(paths, desc="Audit original GRAPE CFPs", unit="image", dynamic_ncols=True):
        pixel_hash, width, height = _rgb_sha256(path)
        relative = str(path.relative_to(directory))
        hashes.setdefault(pixel_hash, []).append(relative)
        rows.append({"filename": relative, "encoded_sha256": sha256_file(path),
                     "rgb_sha256": pixel_hash, "width": width, "height": height})
    return hashes, {"cfp_directory": str(directory), "n_images": len(rows), "n_distinct_rgb_images": len(hashes),
                    "method": _PIXEL_HASH_METHOD, "scope": "All retained original CFP files, irrespective of progression eligibility",
                    "progression_labels_read": False, "files": rows}


def load_hf_diagnosis(hf_root, *, grape_root="data/raw/grape"):
    """Verify and return the official train/test partitions with diagnosis labels.

    HF label 0/glaucoma becomes model label 1; HF label 1/normal becomes 0.
    Same-label duplicate RGB images within a split retain only their first
    released row. Training rows take precedence: matching test images are
    excluded before fitting. Conflicting labels and overlap with any original
    GRAPE CFP fail. Exact-image checks cannot establish patient-level
    independence or rule out differently cropped/re-encoded versions of an eye.
    """
    import pyarrow.parquet as pq

    root = Path(hf_root).expanduser().resolve()
    provenance_path = root / "provenance.json"
    provenance_digest = sha256_file(provenance_path)
    provenance = _read_json(provenance_path)
    if provenance.get("repo_id") != HF_REPO_ID:
        raise ValueError(f"Expected retained diagnosis release {HF_REPO_ID}.")
    revision = provenance.get("revision")
    if not isinstance(revision, str) or re.fullmatch(r"[a-f0-9]{40}", revision) is None:
        raise ValueError("Diagnosis provenance requires the exact 40-character repository revision.")
    files = {split: f"data/{split}-00000-of-00001.parquet" for split in ("train", "test")}
    checksums = provenance.get("files")
    if not isinstance(checksums, dict) or set(checksums) != set(files.values()):
        raise ValueError("Diagnosis provenance must identify exactly the released train and test Parquet files.")
    verified = {}
    for relative in files.values():
        expected = checksums[relative]
        if not isinstance(expected, str) or re.fullmatch(r"[a-f0-9]{64}", expected) is None:
            raise ValueError("Parquet provenance requires complete SHA256 checksums.")
        actual = sha256_file(root / relative)
        if actual != expected:
            raise ValueError(f"Retained diagnosis Parquet checksum mismatch: {relative}")
        verified[relative] = actual

    grape_hashes, grape_audit = _audit_grape_images(grape_root)
    all_pixels, all_filenames = {}, {}
    visits, labels, splits, samples, removed_test_duplicates = {}, {}, {}, [], []
    required = ("image", "label", "filename", "annotation")
    for split in ("train", "test"):
        parquet = pq.ParquetFile(root / files[split])
        columns = parquet.schema_arrow.names
        if not set(required).issubset(columns):
            raise ValueError("Diagnosis Parquet requires image, label, filename and annotation columns.")
        if parquet.metadata.num_rows < 1:
            raise ValueError(f"Released diagnosis split is empty: {split}")
        split_visits, split_labels, seen_filenames = [], [], {}
        raw_counts, kept_counts, prefixes = Counter(), Counter(), Counter()
        duplicate_count, train_overlap_count = 0, 0
        # Do not read descriptions into memory. Small batches retain compressed
        # bytes for returned visits while decoded images are immediately released.
        with tqdm(total=parquet.metadata.num_rows, desc=f"Audit HF {split}", unit="image", dynamic_ncols=True) as progress:
            row_number = 0
            for batch in parquet.iter_batches(batch_size=8, columns=list(required), use_threads=False):
                for row in batch.to_pylist():
                    row_number += 1
                    source_label = row["label"]
                    if isinstance(source_label, bool) or not isinstance(source_label, Integral) or source_label not in (0, 1):
                        raise ValueError(f"{split} row {row_number}: diagnosis label must be integer 0 or 1.")
                    source_label = int(source_label)
                    annotation = row["annotation"]
                    if not isinstance(annotation, str) or annotation.strip().lower() != _SOURCE_ANNOTATIONS[source_label]:
                        raise ValueError(f"{split} row {row_number}: annotation contradicts HF label mapping (0=glaucoma, 1=normal).")
                    filename = row["filename"]
                    if (not isinstance(filename, str) or not filename.strip() or filename != filename.strip()
                            or Path(filename).name != filename or "\\" in filename):
                        raise ValueError(f"{split} row {row_number}: filename must be an unambiguous basename.")
                    embedded = row["image"]
                    if (not isinstance(embedded, dict) or not isinstance(embedded.get("bytes"), (bytes, bytearray))
                            or not embedded["bytes"]):
                        raise ValueError(f"{split} row {row_number}: original embedded image bytes are required; paths are not loaded.")
                    encoded = bytes(embedded["bytes"])
                    pixel_hash, width, height = _rgb_sha256(io.BytesIO(encoded))
                    if pixel_hash in grape_hashes:
                        raise ValueError(f"Exact decoded RGB overlap between HF {split}/{filename} and original GRAPE CFP "
                                         f"{grape_hashes[pixel_hash][0]}; detector training/evaluation is blocked.")
                    sample_id = f"hf_{split}:{filename}"
                    previous = all_pixels.get(pixel_hash)
                    overlaps_train = previous is not None and previous["split"] != split
                    if overlaps_train and (split != "test" or previous["split"] != "train"):
                        raise ValueError("Unexpected released partition order during the overlap audit.")
                    if filename in all_filenames and all_filenames[filename] != split and not overlaps_train:
                        raise ValueError(f"A filename identifies different decoded images across released splits: {filename}.")
                    if filename in seen_filenames and seen_filenames[filename] != pixel_hash:
                        raise ValueError(f"Same filename identifies different images within {split}: {filename}.")
                    if previous is not None and previous["diagnosis_label"] != 1 - source_label:
                        raise ValueError(f"Conflicting diagnosis labels for identical decoded RGB images involving {split}.")
                    record = {"sample_id": sample_id, "split": split, "released_row": row_number, "filename": filename,
                              "filename_source": _filename_source(filename), "source_label": source_label,
                              "diagnosis_label": 1 - source_label, "rgb_sha256": pixel_hash,
                              "encoded_sha256": hashlib.sha256(encoded).hexdigest(), "width": width, "height": height,
                              "retained": previous is None}
                    raw_counts[source_label] += 1
                    prefixes[record["filename_source"]] += 1
                    seen_filenames[filename] = pixel_hash
                    all_filenames[filename] = split
                    if previous is not None:
                        record["duplicate_of"] = previous["sample_id"]
                        if overlaps_train:
                            record["exclusion_reason"] = "Test image duplicates a retained training image in decoded RGB pixels"
                            train_overlap_count += 1
                            removed_test_duplicates.append(record)
                        else:
                            record["exclusion_reason"] = "Within-split duplicate; retain first released RGB-identical image"
                            duplicate_count += 1
                    else:
                        all_pixels[pixel_hash] = record
                        split_visits.append(VisitInput(sample_id=sample_id, image=encoded))
                        split_labels.append(1 - source_label)
                        kept_counts[1 - source_label] += 1
                    samples.append(record)
                    progress.update(1)
        if set(split_labels) != {0, 1}:
            raise ValueError(f"Diagnosis split {split} requires both normal and glaucoma after within-split deduplication.")
        visits[split], labels[split] = tuple(split_visits), _readonly_labels(split_labels)
        splits[split] = {"released_rows": row_number, "retained_rows": len(split_visits),
                         "within_split_duplicates_removed": duplicate_count,
                         "train_overlap_duplicates_removed": train_overlap_count,
                         "source_label_counts": {str(key): value for key, value in sorted(raw_counts.items())},
                         "diagnosis_label_counts": {str(key): value for key, value in sorted(kept_counts.items())},
                         "filename_source_counts": dict(sorted(prefixes.items())), "available_columns": columns,
                         "patient_id_available": "patient_id" in columns}
    # Detect source file replacement during the audit before returning fit inputs.
    if (sha256_file(provenance_path) != provenance_digest
            or any(sha256_file(root / name) != digest for name, digest in verified.items())):
        raise ValueError("Diagnosis inputs changed during their verification audit.")
    audit = {
        "format_version": 1, "repo_id": provenance["repo_id"], "revision": revision,
        "hf_root": str(root), "provenance_sha256": provenance_digest, "parquet_sha256": verified,
        "label_mapping": {"source": {"0": "glaucoma", "1": "normal"},
                          "model": {"0": "normal", "1": "glaucoma"}, "source_to_model": {"0": 1, "1": 0},
                          "evidence": "Every released row's annotation was checked against source 0=glaucoma and 1=normal; labels are inverted for GlaBoost."},
        "model_inputs": ["original embedded image bytes"],
        "excluded_inputs": ["filename", "annotation", "description", "human risk/confidence", "GRAPE outcomes"],
        "split_policy": "Released train/test membership and row order preserved; no rows are reassigned. Training takes precedence; RGB-identical test duplicates are excluded before fitting or performance assessment.",
        "duplicate_policy": "Within each split retain the first RGB-identical image in released order. Exclude test images matching training; require consistent labels for all duplicates. Any HF/GRAPE overlap is a hard failure.",
        "splits": splits, "samples": samples,
        "overlap_audit": {"method": _PIXEL_HASH_METHOD,
                          "train_test_exact_rgb_match_rows": len(removed_test_duplicates),
                          "train_test_exact_rgb_match_groups": len({row["rgb_sha256"] for row in removed_test_duplicates}),
                          "removed_test_duplicates": removed_test_duplicates,
                          "retained_train_test_exact_rgb_matches": 0,
                          "hf_grape_exact_rgb_matches": 0, "grape": grape_audit},
        "limitations": [
            "The released columns do not supply usable patient/eye identifiers; patient-level independence across source splits or cohorts cannot be established.",
            "Filename prefixes describe apparent source datasets and are not independently verified provenance.",
            "RGB hashes detect identical decoded pixels, including changes of lossless container/metadata; differently cropped, resized or lossy-reencoded images may evade this check.",
            "Zero exact-image overlap is evidence about these retained files, not proof that subjects or all upstream source datasets are independent.",
        ],
    }
    return HFDiagnosticDataset(visits["train"], labels["train"], visits["test"], labels["test"], audit)
