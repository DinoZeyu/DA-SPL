"""Single-visit GlaBoost: frozen ResNet152 image features and XGBoost.

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
from .encoders import image_encoder_class, resolve_image_devices


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
        return None


def make_xgb_classifier(config):
    """Build the configured binary glaucoma diagnosis classifier."""
    execution = ({"gpu_id": config.gpu_id, "predictor": "gpu_predictor"}
                 if config.tree_method == "gpu_hist" else {})
    return XGBClassifier(
        objective="binary:logistic", eval_metric="logloss",
        learning_rate=config.learning_rate, max_depth=config.max_depth,
        n_estimators=config.n_estimators, random_state=config.random_state,
        n_jobs=config.n_jobs, tree_method=config.tree_method,
        subsample=config.subsample, colsample_bytree=config.colsample_bytree,
        reg_alpha=0.0, reg_lambda=1.0,
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


def _assert_prediction_backend(classifier, runtime):
    """Check prediction placement independently of the original training method."""
    state = json.loads(classifier.get_booster().save_config())["learner"]
    generic = state.get("generic_param", {})
    tree = state.get("gradient_booster", {}).get("gbtree_train_param", {})
    if (int(generic.get("gpu_id", -2)) != runtime["gpu_id"]
            or tree.get("predictor") != runtime["predictor"]):
        raise RuntimeError("XGBoost did not retain the requested prediction device/backend; "
                           "refusing a silent device change or CPU fallback.")


class GlaBoost:
    """A reusable diagnosis classifier; probability column 1 means glaucoma.

    A frozen image encoder can be injected for small tests or verified cached
    features. Its transform(), output_dim, and spec() match encoders.py.
    Weights are never downloaded unless allow_download=True is explicitly set.
    """

    def __init__(self, config=None, *, allow_download=False,
                 image_encoder=None):
        self.config = config or GlaBoostConfig()
        if not isinstance(self.config, GlaBoostConfig):
            raise TypeError("config must be a GlaBoostConfig.")
        c = self.config
        self.image_encoder = image_encoder
        if self.image_encoder is None:
            self.image_encoder = image_encoder_class(c.image_encoder)(
                weights_path=c.image_weights_path, cache_dir=c.cache_dir,
                device=c.device, batch_size=c.image_batch_size,
                allow_download=allow_download,
            )
        self._fitted = False
        self._encoder_specs = {}
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
            if v.image is None:
                raise ValueError(f"Missing enabled image modality: {v.sample_id}")
        return visits

    def _features(self, visits, *, fitting):
        encoder = self.image_encoder
        features = np.asarray(encoder.transform([visit.image for visit in visits]), dtype=np.float32)
        if encoder.output_dim != 2048 or features.shape != (len(visits), 2048):
            raise ValueError("Frozen ResNet152 encoder must return 2048 features per visit.")
        spec = encoder.spec()
        if (not isinstance(spec, dict) or not spec.get("fingerprint")
                or spec.get("encoder") != "resnet152" or spec.get("frozen") is not True
                or spec.get("output_dim") != 2048
                or not isinstance(spec.get("preprocessing"), dict)):
            raise ValueError("The image encoder must document frozen ResNet152 weights, dimension and preprocessing.")
        if fitting:
            self._encoder_specs = {"image": deepcopy(spec)}
        elif any(spec.get(key) != self._encoder_specs["image"].get(key)
                 for key in ("fingerprint", "encoder", "frozen", "output_dim", "preprocessing")):
            raise ValueError("Image encoder weights or preprocessing differ from the fitted model.")
        names = [f"image_{i}" for i in range(2048)]
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
        self.training_config_ = self.config
        # A deliberate new fit replaces any previous load-time device binding.
        self.__dict__.pop("prediction_runtime_", None)
        self._fitted = True
        return self

    def _require_fitted(self):
        if not self._fitted:
            raise RuntimeError("GlaBoost is not fitted. Train or load a fitted model first.")

    def transform(self, visits):
        """Return the fixed image features without updating encoder parameters."""
        self._require_fitted()
        return self._features(self._visits(visits), fitting=False)

    def predict_proba(self, visits):
        """Return [P(normal), P(glaucoma)] for each independent visit."""
        features = self.transform(visits)
        runtime = getattr(self, "prediction_runtime_", None)
        if runtime is not None:
            _assert_prediction_backend(self.classifier_, runtime)
        probabilities = self.classifier_.predict_proba(features)
        if runtime is not None:
            _assert_prediction_backend(self.classifier_, runtime)
        return probabilities

    def predict_score(self, visits):
        """Continuous visit-level evidence S_it; not a progression probability."""
        return self.predict_proba(visits)[:, 1]

    def predict(self, visits, *, threshold=0.5):
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be in [0, 1].")
        return (self.predict_score(visits) >= threshold).astype(np.int64)

    def feature_importance(self):
        """Normalized XGBoost gain, named by the saved image feature schema."""
        self._require_fitted()
        return dict(zip(self.feature_names_, self.classifier_.feature_importances_.tolist()))

    def save(self, directory):
        """Save native XGBoost JSON and preprocessing JSON in a new directory."""
        self._require_fitted()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        model_path = directory / "model.json"
        self.classifier_.save_model(model_path)
        # Relocating inference to another GPU/CPU must not rewrite the recorded
        # training configuration or imply that the learned trees were refitted.
        config = self.training_config_.to_dict()
        metadata = {
            "format_version": 1,
            "implementation": "GlaBoost paper-based reconstruction; not author weights",
            "target": {"0": "normal", "1": "glaucoma"},
            "config": config,
            "structured_state": None, "human_state": None,
            "encoder_specs": self._encoder_specs,
            "feature_names": self.feature_names_,
            "training_summary": self.training_summary_,
            "model_sha256": _sha256(model_path),
            "versions": {name: _package_version(name) for name in
                         ("numpy", "scikit-learn", "torch", "torchvision", "xgboost")},
            "python": platform.python_version(),
        }
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")

    @classmethod
    def load(cls, directory, *, device=None, cache_dir=None,
             image_weights_path=None, image_batch_size=None, allow_download=False,
             image_encoder=None):
        """Load fixed parameters, binding both encoder and classifier inference.

        ``training_config_`` retains the saved fit configuration. Device and
        path overrides are runtime settings; learned tree parameters do not change.
        """
        directory = Path(directory)
        metadata = json.loads((directory / "metadata.json").read_text())
        if metadata.get("format_version") != 1 or metadata.get("target") != {"0": "normal", "1": "glaucoma"}:
            raise ValueError("Unsupported GlaBoost artifact format or target encoding.")
        if _sha256(directory / "model.json") != metadata["model_sha256"]:
            raise ValueError("Model checksum does not match metadata.json.")
        config = GlaBoostConfig.from_dict(metadata["config"])
        if metadata.get("structured_state") is not None or metadata.get("human_state") is not None:
            raise ValueError("Only image-only model bundles with no structured/human preprocessing are supported.")
        specs = metadata.get("encoder_specs", {})
        image_spec = specs.get("image", {})
        if (set(specs) != {"image"} or image_spec.get("encoder") != "resnet152"
                or image_spec.get("frozen") is not True or image_spec.get("output_dim") != 2048
                or metadata.get("feature_names") != [f"image_{i}" for i in range(2048)]):
            raise ValueError("Saved feature schema must be frozen ResNet152 with 2048 ordered image features.")
        overrides = {key: value for key, value in {
            "device": device, "cache_dir": cache_dir,
            "image_weights_path": image_weights_path, "image_batch_size": image_batch_size,
        }.items() if value is not None}
        model = cls(replace(config, **overrides), allow_download=allow_download,
                    image_encoder=image_encoder)
        resolved, gpu_ids = resolve_image_devices(model.config.device)
        model.prediction_runtime_ = {
            "requested_device": model.config.device, "resolved_device": resolved,
            "predictor": "gpu_predictor" if gpu_ids else "cpu_predictor",
            "gpu_id": gpu_ids[0] if gpu_ids else -1,
        }
        model.training_config_ = config
        model.feature_names_ = metadata["feature_names"]
        model._encoder_specs = metadata["encoder_specs"]
        model.training_summary_ = metadata["training_summary"]
        model.classifier_ = XGBClassifier()
        model.classifier_.load_model(directory / "model.json")
        model.classifier_.set_params(
            n_jobs=model.config.n_jobs, predictor=model.prediction_runtime_["predictor"],
            gpu_id=model.prediction_runtime_["gpu_id"])
        _assert_prediction_backend(model.classifier_, model.prediction_runtime_)
        if model.classifier_.n_features_in_ != len(model.feature_names_):
            raise ValueError("Saved feature schema does not match the classifier.")
        model._fitted = True
        return model
