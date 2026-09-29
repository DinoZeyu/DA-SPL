"""Single-visit GlaBoost: frozen features, train-only preprocessing, XGBoost.

This is a paper-based implementation, not the authors' code or trained model.
The caller supplies training-only visits and diagnosis labels. No progression
labels, cross-validation, or temporal integration are performed in this class.
"""

import hashlib
import json
import platform
from copy import deepcopy
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Sequence

import numpy as np
from xgboost import XGBClassifier

from .config import GlaBoostConfig
from .data import VisitInput
from .encoders import MBERTEncoder, ResNet152Encoder
from .structured import StructuredEncoder


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        # Custom injected encoders need not install the optional mBERT library.
        return None


def make_xgb_classifier(config):
    """Shared paper classifier settings for diagnosis and progression adaptation.

    The caller defines and records the target. The architecture alone does not
    turn a progression classifier into the paper's glaucoma diagnosis model.
    """
    execution = ({"gpu_id": config.gpu_id, "predictor": "gpu_predictor"}
                 if config.tree_method == "gpu_hist" else {})
    return XGBClassifier(
        objective="binary:logistic", eval_metric="logloss",
        learning_rate=config.learning_rate, max_depth=config.max_depth,
        n_estimators=config.n_estimators, random_state=config.random_state,
        n_jobs=config.n_jobs, tree_method=config.tree_method,
        subsample=1.0, colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=1.0,
        **execution,
    )


def assert_xgb_backend(model, config):
    """Reject a fitted GPU run if XGBoost changed its requested GPU backend."""
    if config.tree_method != "gpu_hist":
        return
    state = json.loads(model.get_booster().save_config())["learner"]
    generic = state.get("generic_param", {})
    tree = state.get("gradient_booster", {}).get("gbtree_train_param", {})
    if (int(generic.get("gpu_id", -1)) != config.gpu_id
            or tree.get("tree_method") != "gpu_hist"
            or tree.get("predictor") != "gpu_predictor"):
        raise RuntimeError("XGBoost did not retain the requested gpu_hist/gpu_predictor/GPU ID; "
                           "refusing a silent CPU fallback.")


class GlaBoost:
    """A reusable diagnosis classifier; probability column 1 means glaucoma.

    Image/text encoders can be injected for testing or deliberate extensions.
    Their transform(), output_dim, and spec() interfaces must match encoders.py.
    Weights are never downloaded unless allow_download=True is explicitly set.
    """

    def __init__(self, config=None, *, allow_download=False,
                 image_encoder=None, text_encoder=None):
        self.config = config or GlaBoostConfig()
        c = self.config
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        if c.use_image and self.image_encoder is None:
            self.image_encoder = ResNet152Encoder(
                weights_path=c.image_weights_path, cache_dir=c.cache_dir,
                device=c.device, batch_size=c.image_batch_size,
                allow_download=allow_download,
            )
        if c.use_text and self.text_encoder is None:
            self.text_encoder = MBERTEncoder(
                model_name=c.text_model_name, revision=c.text_revision,
                cache_dir=c.cache_dir, device=c.device,
                batch_size=c.text_batch_size, max_length=c.text_max_length,
                allow_download=allow_download,
            )
        self._fitted = False
        self._encoder_specs = {}
        self._structured = None
        self._human = None
        self.feature_names_ = []
        self.classes_ = np.array([0, 1])

    def _visits(self, visits):
        visits = list(visits)
        if not visits or not all(isinstance(v, VisitInput) for v in visits):
            raise ValueError("Provide a nonempty sequence of VisitInput records.")
        ids = [v.sample_id for v in visits]
        if any(not isinstance(i, str) or not i.strip() for i in ids) or len(set(ids)) != len(ids):
            raise ValueError("sample_id must be nonempty and unique within each call.")
        for v in visits:
            if self.config.use_image and v.image is None:
                raise ValueError(f"Missing enabled image modality: {v.sample_id}")
            if self.config.use_text and (not isinstance(v.text, str) or not v.text.strip()):
                raise ValueError(f"Missing enabled text modality: {v.sample_id}")
            if self.config.use_structured and "cup_to_disc_ratio" in self.config.numeric_features:
                self._unit_interval(v.structured.get("cup_to_disc_ratio"), "cup_to_disc_ratio")
            if self.config.use_human_confidence:
                self._unit_interval(v.human.get("confidence_level"), "confidence_level")
        return visits

    @staticmethod
    def _unit_interval(value, name):
        if value is None:
            return
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric, missing, or in [0, 1].") from exc
        if not np.isnan(value) and not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0, 1].")

    def _encode(self, encoder, inputs, modality, *, fitting):
        block = np.asarray(encoder.transform(inputs), dtype=np.float32)
        if block.shape != (len(inputs), encoder.output_dim):
            raise ValueError(f"{modality} encoder returned an unexpected shape.")
        spec = encoder.spec()
        if (not isinstance(spec, dict) or not spec.get("fingerprint")
                or not spec.get("encoder") or spec.get("output_dim") != encoder.output_dim
                or not isinstance(spec.get("preprocessing"), dict)):
            raise ValueError(f"{modality} encoder must identify its weights, dimension, and preprocessing.")
        if fitting:
            self._encoder_specs[modality] = deepcopy(spec)
        elif any(spec.get(key) != self._encoder_specs[modality].get(key)
                 for key in ("fingerprint", "encoder", "output_dim", "preprocessing")):
            raise ValueError(f"{modality} encoder weights or preprocessing differ from the fitted model.")
        return block

    def _features(self, visits, *, fitting):
        c = self.config
        blocks, names = [], []
        # The order is equation (5): text, structured, human, image.
        if c.use_text:
            block = self._encode(self.text_encoder, [v.text for v in visits], "text", fitting=fitting)
            blocks.append(block)
            names.extend(f"text_{i}" for i in range(block.shape[1]))
        if c.use_structured:
            rows = [v.structured for v in visits]
            if fitting:
                self._structured = StructuredEncoder(c.numeric_features, c.categorical_features).fit(rows)
            blocks.append(self._structured.transform(rows))
            names.extend("structured::" + name for name in self._structured.feature_names_)
        if c.use_human_risk or c.use_human_confidence:
            rows = [v.human for v in visits]
            if fitting:
                self._human = StructuredEncoder(
                    ("confidence_level",) if c.use_human_confidence else (),
                    ("glaucoma_risk_assessment",) if c.use_human_risk else (),
                    standardize=False,
                ).fit(rows)
            blocks.append(self._human.transform(rows))
            names.extend("human::" + name for name in self._human.feature_names_)
        if c.use_image:
            block = self._encode(self.image_encoder, [v.image for v in visits], "image", fitting=fitting)
            blocks.append(block)
            names.extend(f"image_{i}" for i in range(block.shape[1]))
        features = np.concatenate(blocks, axis=1).astype(np.float32, copy=False)
        if not np.isfinite(features).all():
            raise ValueError("Features contain NaN or infinity after preprocessing.")
        if fitting:
            self.feature_names_ = names
        elif names != self.feature_names_:
            raise ValueError("Feature schema differs from the fitted model.")
        return features

    def fit(self, visits: Sequence[VisitInput], y, *, sample_weight=None):
        """Fit only on caller-supplied training visits; y=1 is glaucoma diagnosis."""
        visits = self._visits(visits)
        target = np.asarray(y)
        if target.shape != (len(visits),) or not np.isin(target, [0, 1]).all():
            raise ValueError("y must have one binary diagnosis target per visit (1=glaucoma).")
        if len(np.unique(target)) != 2:
            raise ValueError("Diagnosis training requires both normal and glaucoma examples.")
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=float)
            if (sample_weight.shape != target.shape or not np.isfinite(sample_weight).all()
                    or np.any(sample_weight < 0) or sample_weight.sum() <= 0):
                raise ValueError("sample_weight must be finite, nonnegative, and have positive sum.")
        self._fitted = False
        self._encoder_specs = {}
        features = self._features(visits, fitting=True)
        self.classifier_ = make_xgb_classifier(self.config)
        self.classifier_.fit(features, target.astype(np.int64), sample_weight=sample_weight)
        assert_xgb_backend(self.classifier_, self.config)
        self.training_summary_ = {
            "n_visits": len(visits), "n_normal": int((target == 0).sum()),
            "n_glaucoma": int((target == 1).sum()),
            "all_patient_ids_available": all(v.patient_id is not None for v in visits),
            "n_features": features.shape[1],
        }
        self._fitted = True
        return self

    def _require_fitted(self):
        if not self._fitted:
            raise RuntimeError("GlaBoost is not fitted. Train or load a fitted model first.")

    def transform(self, visits):
        """Reuse fitted preprocessing; this never updates training statistics."""
        self._require_fitted()
        return self._features(self._visits(visits), fitting=False)

    def predict_proba(self, visits):
        """Return [P(normal), P(glaucoma)] for each independent visit."""
        features = self.transform(visits)
        return self.classifier_.predict_proba(features)

    def predict_score(self, visits):
        """Continuous visit-level evidence S_it; not a progression probability."""
        return self.predict_proba(visits)[:, 1]

    def predict(self, visits, *, threshold=0.5):
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be in [0, 1].")
        return (self.predict_score(visits) >= threshold).astype(np.int64)

    def feature_importance(self):
        """Normalized XGBoost gain, named by the saved fused feature schema."""
        self._require_fitted()
        return dict(zip(self.feature_names_, self.classifier_.feature_importances_.tolist()))

    def save(self, directory):
        """Save native XGBoost JSON and preprocessing JSON in a new directory."""
        self._require_fitted()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        model_path = directory / "model.json"
        self.classifier_.save_model(model_path)
        config = self.config.to_dict()
        resolved_revision = self._encoder_specs.get("text", {}).get("resolved_revision")
        if resolved_revision:
            config["text_revision"] = resolved_revision
        metadata = {
            "format_version": 1,
            "implementation": "GlaBoost paper-based reconstruction; not author weights",
            "target": {"0": "normal", "1": "glaucoma"},
            "config": config,
            "structured_state": self._structured.to_dict() if self._structured else None,
            "human_state": self._human.to_dict() if self._human else None,
            "encoder_specs": self._encoder_specs,
            "feature_names": self.feature_names_,
            "training_summary": self.training_summary_,
            "model_sha256": _sha256(model_path),
            "versions": {name: _package_version(name) for name in
                         ("numpy", "scikit-learn", "torch", "torchvision", "xgboost")
                         + (("transformers",) if self.config.use_text else ())},
            "python": platform.python_version(),
        }
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")

    @classmethod
    def load(cls, directory, *, device=None, cache_dir=None,
             image_weights_path=None, text_model_name=None, allow_download=False,
             image_encoder=None, text_encoder=None):
        """Load fixed preprocessing and classifier; encoder fingerprints must match."""
        directory = Path(directory)
        metadata = json.loads((directory / "metadata.json").read_text())
        if metadata.get("format_version") != 1 or metadata.get("target") != {"0": "normal", "1": "glaucoma"}:
            raise ValueError("Unsupported GlaBoost artifact format or target encoding.")
        if _sha256(directory / "model.json") != metadata["model_sha256"]:
            raise ValueError("Model checksum does not match metadata.json.")
        config = GlaBoostConfig.from_dict(metadata["config"])
        overrides = {key: value for key, value in {
            "device": device, "cache_dir": cache_dir,
            "image_weights_path": image_weights_path, "text_model_name": text_model_name,
        }.items() if value is not None}
        model = cls(replace(config, **overrides), allow_download=allow_download,
                    image_encoder=image_encoder, text_encoder=text_encoder)
        if metadata["structured_state"] is not None:
            model._structured = StructuredEncoder.from_dict(metadata["structured_state"])
        if metadata["human_state"] is not None:
            model._human = StructuredEncoder.from_dict(metadata["human_state"])
        model.feature_names_ = metadata["feature_names"]
        model._encoder_specs = metadata["encoder_specs"]
        model.training_summary_ = metadata["training_summary"]
        model.classifier_ = XGBClassifier()
        model.classifier_.load_model(directory / "model.json")
        model.classifier_.set_params(n_jobs=model.config.n_jobs)
        if model.classifier_.n_features_in_ != len(model.feature_names_):
            raise ValueError("Saved feature schema does not match the classifier.")
        model._fitted = True
        return model
