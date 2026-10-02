r"""Reusable consolidation of bibm-1.ipynb and bibm-2.ipynb (diagnosis).

Modes:
  structured: all flattened description fields -> XGBoost (100 trees).
  multimodal: ten structured fields + frozen ResNet18 (512) + frozen BERT
    CLS (768) + risk one-hot + confidence -> XGBoost (500 trees).

The multimodal notebook uses bert-base-uncased, maximum 32 tokens, and image
Resize((224, 224)) + ToTensor WITHOUT ImageNet normalization. These choices
are retained; this is the senior's notebook implementation, not the separate
paper-based GRAPE progression pipeline. Neural encoders are not fine-tuned.

Input CSV: description is a JSON object containing fundus_features and optional
glaucoma_risk_assessment/confidence_level. Training also requires label (0/1).
Multimodal input needs image: either the notebook's Python literal {'bytes':
b'...'} or an image path relative to the CSV. annotation is never a feature.
The notebooks' saved samples use 0=glaucoma, 1=normal, contrary to a source
comment. Numeric labels are preserved; explicitly set --glaucoma-label if
probabilities should additionally be named probability_glaucoma.

Corrections: fit categorical vocabularies on training rows only, reject invalid
JSON/images, and persist preprocessing plus trained and frozen model weights.
Risk/confidence and descriptive findings can encode the diagnosis; assess their
provenance before treating accuracy as independent diagnostic validation.
Default splits reproduce the notebooks' row-level fractions; use --group-column
for repeated patients. This script does not define longitudinal outcomes.

Examples (activate the existing Conda environment first):
  python Glaboost_CH.py train --train-csv glaucoma_train.csv \
    --output-dir artifacts/glaboost_ch --mode multimodal --allow-download \
    --glaucoma-label 0
  python Glaboost_CH.py predict --model-dir artifacts/glaboost_ch \
    --input-csv glaucoma_test.csv --output-csv result/glaboost_ch_predictions.csv

CUDA is the default; 'cuda' uses all visible GPUs for neural feature extraction
and the first GPU for XGBoost. 'cuda:N' selects one GPU. CPU is explicit only.
Weights are downloaded only with --allow-download. Saved bundles predict
offline, including the frozen encoders. Existing outputs are not overwritten.
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import platform
import shutil
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from tqdm.auto import tqdm


STRUCTURED_FIELDS = (
    "optic_disc_size", "cup_to_disc_ratio", "isnt_rule_followed", "rim_pallor",
    "rim_color", "bayoneting", "sharp_edge", "laminar_dot_sign", "notching",
    "rim_thinning",
)
BERT_NAME = "bert-base-uncased"
RESNET_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
DEFAULT_CACHE = Path(__file__).resolve().parent / ".cache" / "glaboost_ch"


def _descriptions(frame):
    if "description" not in frame:
        raise ValueError("Input must contain a description column with JSON objects.")
    rows = []
    for index, value in frame["description"].items():
        try:
            item = json.loads(value) if isinstance(value, str) else value
            if not isinstance(item, dict) or not isinstance(item.get("fundus_features"), dict):
                raise ValueError("expected an object containing fundus_features")
            flat = {key: item[key] for key in ("glaucoma_risk_assessment", "confidence_level")
                    if key in item}
            flat.update(item["fundus_features"])
            if any(isinstance(value, (dict, list)) for value in flat.values()):
                raise ValueError("feature values must be scalar")
            rows.append(flat)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid description at row {index}: {exc}") from exc
    return pd.DataFrame(rows, index=frame.index)


def _labels(frame):
    if "label" not in frame:
        raise ValueError("Training/evaluation requires a label column containing 0 and 1.")
    labels = pd.to_numeric(frame["label"], errors="raise")
    if not labels.isin([0, 1]).all():
        raise ValueError("Only binary labels 0 and 1 are supported; labels are never remapped.")
    return labels.to_numpy(dtype=np.int32)


class _TabularSchema:
    """Notebook-style get_dummies, with a training-only, serializable schema."""

    def __init__(self, state=None):
        self.state = state

    def transform(self, frame, *, fit=False):
        if fit:
            self.state = {"inputs": list(frame.columns), "categorical": [
                column for column in frame if not pd.api.types.is_numeric_dtype(frame[column])
            ]}
        if self.state is None:
            raise RuntimeError("Fit the feature schema before prediction.")
        selected = frame.reindex(columns=self.state["inputs"]).copy()
        for column in selected:
            if column in self.state["categorical"]:
                selected[column] = selected[column].map(
                    lambda value: np.nan if pd.isna(value) else str(value)
                ).astype(object)
            else:
                selected[column] = pd.to_numeric(selected[column], errors="raise")
        encoded = pd.get_dummies(selected, columns=self.state["categorical"],
                                 dummy_na=True, dtype=np.float32)
        if fit:
            if not encoded.columns.is_unique:
                raise ValueError("Categorical encoding produced ambiguous column names.")
            self.state["outputs"] = list(encoded.columns)
        encoded = encoded.reindex(columns=self.state["outputs"], fill_value=0)
        result = encoded.to_numpy(dtype=np.float32)
        if np.isinf(result).any():
            raise ValueError("Infinite feature values are not supported.")
        return result


def _open_image(value):
    from PIL import Image

    if isinstance(value, str):
        value = ast.literal_eval(value) if value.lstrip().startswith("{") else Path(value)
    if isinstance(value, dict):
        value = value.get("bytes") if value.get("bytes") is not None else value.get("path")
    if isinstance(value, (bytes, bytearray)):
        value = io.BytesIO(value)
    if value is None:
        raise ValueError("image has neither bytes nor a path")
    with Image.open(value) as image:
        return image.convert("RGB")


def _devices(device):
    if device == "cpu":
        return "cpu", []
    import torch

    if device != "cuda" and not (device.startswith("cuda:") and device[5:].isdigit()):
        raise ValueError("device must be cuda, cuda:N, or cpu")
    count = torch.cuda.device_count()
    ids = list(range(count)) if device == "cuda" else [int(device[5:])]
    if not ids or max(ids) >= count:
        raise RuntimeError(f"Requested {device}, but only {count} CUDA GPUs are visible.")
    return f"cuda:{ids[0]}", ids


class _FrozenEncoders:
    def __init__(self, device, batch_size, cache_dir, allow_download, bundle=None):
        import torch
        from torchvision import models, transforms
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Multimodal mode requires the project's optional 'text' dependencies.") from exc

        self.device, ids = _devices(device)
        self.batch_size = batch_size
        self.resnet = models.resnet18(weights=None)
        self.resnet.fc = torch.nn.Identity()
        if bundle is not None:
            state = torch.load(str(bundle / "resnet18.pt"), map_location="cpu", weights_only=True)
            bert_source, bert_options = str(bundle / "bert"), {"local_files_only": True}
        else:
            checkpoint = cache_dir / "torch" / RESNET_URL.rsplit("/", 1)[-1]
            if not checkpoint.is_file():
                if not allow_download:
                    raise FileNotFoundError(f"Missing {checkpoint}; use --allow-download to fetch pretrained weights.")
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.hub.load_state_dict_from_url(RESNET_URL, model_dir=str(checkpoint.parent),
                                                  map_location="cpu", check_hash=True)
            state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
            state.pop("fc.weight")
            state.pop("fc.bias")
            bert_source = BERT_NAME
            bert_options = {"cache_dir": str(cache_dir / "huggingface"),
                            "local_files_only": not allow_download}
        self.resnet.load_state_dict(state, strict=True)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_source, **bert_options)
        self.bert = AutoModel.from_pretrained(bert_source, **bert_options)
        for model in (self.resnet, self.bert):
            model.requires_grad_(False).eval().to(self.device)

        class CLS(torch.nn.Module):
            def __init__(self, bert):
                super().__init__()
                self.bert = bert

            def forward(self, **inputs):
                return self.bert(**inputs).last_hidden_state[:, 0, :]

        self.image_model, self.text_model = self.resnet, CLS(self.bert).eval()
        if len(ids) > 1:
            self.image_model = torch.nn.DataParallel(self.image_model, device_ids=ids).eval()
            self.text_model = torch.nn.DataParallel(self.text_model, device_ids=ids).eval()
        self.preprocess = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    def transform(self, frame, descriptions):
        import torch

        if "image" not in frame:
            raise ValueError("Multimodal mode requires the image column.")
        images, texts = [], []
        for start in tqdm(range(0, len(frame), self.batch_size), desc="Frozen ResNet18 + BERT", unit="batch"):
            batch = frame.iloc[start:start + self.batch_size]
            pixels = []
            for index, value in batch["image"].items():
                try:
                    pixels.append(self.preprocess(_open_image(value)))
                except Exception as exc:
                    raise ValueError(f"Invalid image at row {index}: {exc}") from exc
            rims = descriptions.iloc[start:start + self.batch_size].reindex(
                columns=["neuroretinal_rim"])["neuroretinal_rim"].fillna("").astype(str).tolist()
            tokens = self.tokenizer(rims, padding=True, truncation=True, max_length=32, return_tensors="pt")
            with torch.inference_mode():
                images.append(self.image_model(torch.stack(pixels).to(self.device)).cpu().numpy())
                texts.append(self.text_model(**{k: v.to(self.device) for k, v in tokens.items()}).cpu().numpy())
        return np.concatenate(images), np.concatenate(texts)

    def save(self, directory):
        import torch

        torch.save({key: value.detach().cpu() for key, value in self.resnet.state_dict().items()},
                   directory / "resnet18.pt")
        self.bert.save_pretrained(directory / "bert")
        self.tokenizer.save_pretrained(directory / "bert")


class GlaboostCH:
    """Fit, save, load, and predict either of the senior's notebook models."""

    def __init__(self, mode="multimodal", device="cuda", batch_size=32,
                 cache_dir=DEFAULT_CACHE, allow_download=False, seed=42, glaucoma_label=None):
        if mode not in ("structured", "multimodal"):
            raise ValueError("mode must be structured or multimodal")
        if batch_size < 1 or glaucoma_label not in (None, 0, 1):
            raise ValueError("batch_size must be positive; glaucoma_label must be 0, 1, or None")
        self.mode, self.device, self.batch_size = mode, device, batch_size
        self.cache_dir, self.allow_download = Path(cache_dir), allow_download
        self.seed, self.glaucoma_label = seed, glaucoma_label
        self.schema, self.risk_schema = _TabularSchema(), _TabularSchema()
        self.encoders = self.model = self.bundle = None
        self.feature_names = []

    def _features(self, frame, *, fit=False):
        if frame.empty:
            raise ValueError("Input contains no samples.")
        descriptions = _descriptions(frame)
        if self.mode == "structured":
            features = self.schema.transform(descriptions, fit=fit)
            self.feature_names = self.schema.state["outputs"]
            return features
        structured = self.schema.transform(descriptions.reindex(columns=STRUCTURED_FIELDS), fit=fit)
        risk = self.risk_schema.transform(descriptions.reindex(columns=["glaucoma_risk_assessment"]), fit=fit)
        confidence = pd.to_numeric(descriptions.reindex(columns=["confidence_level"])["confidence_level"],
                                   errors="raise").to_numpy(dtype=np.float32).reshape(-1, 1)
        if self.encoders is None:
            self.encoders = _FrozenEncoders(self.device, self.batch_size, self.cache_dir,
                                            self.allow_download, self.bundle)
        images, texts = self.encoders.transform(frame, descriptions)
        if images.shape != (len(frame), 512) or texts.shape != (len(frame), 768):
            raise ValueError("Expected ResNet18 512-dimensional and BERT 768-dimensional features.")
        self.feature_names = (self.schema.state["outputs"] + [f"img_{i}" for i in range(512)]
                              + [f"rim_emb_{i}" for i in range(768)]
                              + self.risk_schema.state["outputs"] + ["confidence_level"])
        features = np.hstack([structured, images, texts, risk, confidence]).astype(np.float32)
        if np.isinf(features).any():
            raise ValueError("Infinite feature values are not supported.")
        return features

    def _backend(self):
        resolved, ids = _devices(self.device)
        if int(xgb.__version__.split(".")[0]) >= 2:
            return {"tree_method": "hist", "device": resolved}
        return ({"tree_method": "gpu_hist", "predictor": "gpu_predictor", "gpu_id": ids[0]}
                if ids else {"tree_method": "hist", "predictor": "cpu_predictor"})

    def fit(self, train, validation=None):
        if self.model is not None:
            raise RuntimeError("Create a new GlaboostCH instance to fit another model.")
        labels = _labels(train)
        if set(labels) != {0, 1}:
            raise ValueError("Training split must contain both labels 0 and 1.")
        params = dict(objective="binary:logistic", eval_metric="logloss", n_jobs=1,
                      random_state=self.seed, **self._backend())
        if self.mode == "structured":
            params.update(n_estimators=100)
        else:
            params.update(n_estimators=500, max_depth=6, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, min_child_weight=1,
                          gamma=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1)
        features = self._features(train, fit=True)
        eval_set = None
        if validation is not None:
            eval_set = [(self._features(validation), _labels(validation))]
        model = xgb.XGBClassifier(**params)
        model.fit(features, labels, eval_set=eval_set, verbose=25 if eval_set else False)
        self.model = model
        return self

    def predict_proba(self, frame):
        """Columns are P(label=0), P(label=1), without inferred clinical meaning."""
        if self.model is None:
            raise RuntimeError("Fit or load a model first.")
        return self.model.predict_proba(self._features(frame))

    def predict(self, frame):
        return self.predict_proba(frame).argmax(axis=1)

    def save(self, directory):
        if self.model is None:
            raise RuntimeError("Fit a model before saving.")
        directory = _output_path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        try:
            self.model.save_model(str(directory / "xgboost.json"))
            if self.mode == "multimodal":
                if self.encoders is None:
                    # A freshly loaded bundle may be saved before first prediction.
                    shutil.copy2(self.bundle / "resnet18.pt", directory / "resnet18.pt")
                    shutil.copytree(self.bundle / "bert", directory / "bert")
                else:
                    self.encoders.save(directory)
            packages = ["numpy", "pandas", "scikit-learn", "xgboost"]
            if self.mode == "multimodal":
                packages += ["torch", "torchvision", "transformers", "pillow"]
            state = {"format_version": 1, "mode": self.mode, "seed": self.seed,
                     "glaucoma_label": self.glaucoma_label, "schema": self.schema.state,
                     "risk_schema": self.risk_schema.state, "feature_names": self.feature_names,
                     "probability_column_labels": [0, 1], "decision_rule": "argmax; ties use label 0",
                     "python_version": platform.python_version(),
                     "package_versions": {name: version(name) for name in packages},
                     "encoders": {"image": "ResNet18 ImageNet1K V1; Resize224 + ToTensor; no normalization",
                                  "text": "bert-base-uncased; CLS; max_length=32"}
                     if self.mode == "multimodal" else None}
            _write_json(directory / "manifest.json", state)
        except Exception:
            # Leave partial files for diagnosis; load requires the final manifest.
            raise

    @classmethod
    def load(cls, directory, *, device="cuda", batch_size=32):
        directory = Path(directory).expanduser().resolve()
        state = json.loads((directory / "manifest.json").read_text())
        if state["format_version"] != 1:
            raise ValueError("Unsupported saved model format.")
        instance = cls(mode=state["mode"], device=device, batch_size=batch_size,
                       seed=state["seed"], glaucoma_label=state["glaucoma_label"])
        instance.schema, instance.risk_schema = _TabularSchema(state["schema"]), _TabularSchema(state["risk_schema"])
        instance.feature_names, instance.bundle = state["feature_names"], directory
        instance.model = xgb.XGBClassifier()
        instance.model.load_model(str(directory / "xgboost.json"))
        instance.model.set_params(**instance._backend())
        return instance


def _output_path(value):
    path = Path(value).expanduser().absolute()
    for candidate in (path, path.resolve()):
        if any(candidate.parts[i:i + 2] == ("data", "raw") for i in range(len(candidate.parts) - 1)):
            raise ValueError(f"Outputs must not be written into raw data: {path}")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Output already exists: {path}")
    return path


def _read_csv(path, group_column=None):
    path = Path(path).expanduser().resolve()
    frame = pd.read_csv(path, dtype={group_column: "string"} if group_column else None)
    if "image" in frame:
        def resolve(value):
            if isinstance(value, str) and not value.lstrip().startswith("{"):
                image_path = Path(value).expanduser()
                return str(image_path if image_path.is_absolute() else path.parent / image_path)
            if isinstance(value, str) and value.lstrip().startswith("{"):
                try:
                    item = ast.literal_eval(value)
                    if isinstance(item, dict) and item.get("bytes") is None and item.get("path"):
                        image_path = Path(item["path"]).expanduser()
                        item["path"] = str(image_path if image_path.is_absolute() else path.parent / image_path)
                        return item
                except (SyntaxError, ValueError):
                    pass  # Image decoding below reports the offending row.
            return value
        frame["image"] = frame["image"].map(resolve)
    return frame


def _write_json(path, content):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(content, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _prediction_frame(model, frame, probabilities):
    output = frame[[key for key in ("filename", "label") if key in frame]].copy()
    output.insert(0, "source_row", frame.index.to_numpy())
    output["predicted_label"] = probabilities.argmax(axis=1)
    output["probability_label_0"] = probabilities[:, 0]
    output["probability_label_1"] = probabilities[:, 1]
    if model.glaucoma_label is not None:
        output["probability_glaucoma"] = probabilities[:, model.glaucoma_label]
    return output


def _evaluate(model, frame, directory, name):
    labels, probabilities = _labels(frame), model.predict_proba(frame)
    predicted = probabilities.argmax(axis=1)
    metrics = {"n_samples": len(frame), "accuracy": accuracy_score(labels, predicted),
               "balanced_accuracy": balanced_accuracy_score(labels, predicted),
               "roc_auc_positive_label_1": roc_auc_score(labels, probabilities[:, 1])
               if len(np.unique(labels)) == 2 else None,
               "confusion_matrix_label_order": [0, 1],
               "confusion_matrix": confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
               "classification_report": classification_report(labels, predicted, labels=[0, 1],
                                                              output_dict=True, zero_division=0)}
    _write_json(directory / f"{name}_metrics.json", metrics)
    _prediction_frame(model, frame, probabilities).to_csv(directory / f"{name}_predictions.csv", index=False, mode="x")
    print(f"{name}: n={len(frame)}, accuracy={metrics['accuracy']:.4f}, "
          f"balanced accuracy={metrics['balanced_accuracy']:.4f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train a notebook model and save an offline bundle.")
    train.add_argument("--train-csv", required=True, type=Path)
    train.add_argument("--test-csv", type=Path, help="Optional separate test set; never used for fitting.")
    train.add_argument("--output-dir", required=True, type=Path)
    train.add_argument("--mode", choices=["structured", "multimodal"], default="multimodal")
    train.add_argument("--validation-fraction", type=float, help="Default: structured 0.2, multimodal 0.5.")
    train.add_argument("--group-column", help="Keep each patient/group entirely in one split.")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--glaucoma-label", type=int, choices=[0, 1], help="Optional explicit diagnosis label mapping.")
    train.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    train.add_argument("--allow-download", action="store_true")
    predict = commands.add_parser("predict", help="Predict from an offline saved bundle.")
    predict.add_argument("--model-dir", required=True, type=Path)
    predict.add_argument("--input-csv", required=True, type=Path)
    predict.add_argument("--output-csv", required=True, type=Path)
    for command in (train, predict):
        command.add_argument("--device", default="cuda", help="cuda (all visible encoders), cuda:N, or explicit cpu")
        command.add_argument("--batch-size", type=int, default=32, help="Global encoder batch size across visible GPUs.")
    args = parser.parse_args(argv)
    if args.command == "predict":
        output = _output_path(args.output_csv)
        model = GlaboostCH.load(args.model_dir, device=args.device, batch_size=args.batch_size)
        frame = _read_csv(args.input_csv)
        predictions = _prediction_frame(model, frame, model.predict_proba(frame))
        output.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(output, index=False, mode="x")
        print(f"Saved predictions: {output}")
        return

    output = _output_path(args.output_dir)
    frame = _read_csv(args.train_csv, args.group_column)
    _labels(frame)
    fraction = args.validation_fraction
    if fraction is None:
        fraction = 0.2 if args.mode == "structured" else 0.5
    if not 0 < fraction < 1:
        parser.error("--validation-fraction must be strictly between 0 and 1")
    if args.group_column:
        if args.group_column not in frame or frame[args.group_column].isna().any():
            parser.error("--group-column must name a column without missing values")
        splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=args.seed)
        train_indices, val_indices = next(splitter.split(frame, groups=frame[args.group_column]))
    else:
        train_indices, val_indices = train_test_split(np.arange(len(frame)), test_size=fraction,
                                                      random_state=args.seed)
        print("Using the notebooks' random row split. Repeated patients require --group-column.")
    test = _read_csv(args.test_csv, args.group_column) if args.test_csv else None
    if test is not None:
        if args.test_csv.resolve() == args.train_csv.resolve():
            parser.error("--test-csv must differ from --train-csv")
        _labels(test)
        if "filename" in frame and "filename" in test:
            train_names = set(frame["filename"].dropna().astype(str)) - {""}
            test_names = set(test["filename"].dropna().astype(str)) - {""}
            if train_names & test_names:
                parser.error("Test and training CSVs contain overlapping filenames.")
        if args.group_column:
            if args.group_column not in test or test[args.group_column].isna().any():
                parser.error("Test data must contain nonmissing group IDs.")
            if set(frame[args.group_column]) & set(test[args.group_column]):
                parser.error("Test and training CSVs contain overlapping patient/group IDs.")
    model = GlaboostCH(mode=args.mode, device=args.device, batch_size=args.batch_size,
                       cache_dir=args.cache_dir, allow_download=args.allow_download,
                       seed=args.seed, glaucoma_label=args.glaucoma_label)
    validation = frame.iloc[val_indices]
    model.fit(frame.iloc[train_indices], validation)
    model.save(output)
    assignments = pd.DataFrame({"source_row": np.arange(len(frame)), "split": "train"})
    assignments.loc[val_indices, "split"] = "validation"
    if args.group_column:
        assignments[args.group_column] = frame[args.group_column].to_numpy()
    assignments.to_csv(output / "split_assignments.csv", index=False, mode="x")
    _write_json(output / "run.json", {"train_csv": str(args.train_csv.resolve()),
                                     "test_csv": str(args.test_csv.resolve()) if args.test_csv else None,
                                     "validation_fraction": fraction, "seed": args.seed,
                                     "group_column": args.group_column})
    _evaluate(model, validation, output, "validation")
    if test is not None:
        _evaluate(model, test, output, "test")
    pd.DataFrame({"feature": model.feature_names, "importance": model.model.feature_importances_}).sort_values(
        "importance", ascending=False).to_csv(output / "feature_importance.csv", index=False, mode="x")
    print(f"Saved model, schema, split assignments, and evaluation: {output}")


if __name__ == "__main__":
    main()
