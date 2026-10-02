"""Explicit paper defaults and the senior notebook's GRAPE image recipe."""

from dataclasses import asdict, dataclass
from numbers import Real
from typing import Optional, Tuple

from .data import FUNDUS_CATEGORICAL_FEATURES, FUNDUS_NUMERIC_FEATURES


@dataclass(frozen=True)
class GlaBoostConfig:
    # GRAPE consistently offers CFPs, but not the paper's text or human inputs.
    use_image: bool = True
    use_text: bool = False
    use_structured: bool = False
    use_human_risk: bool = False
    use_human_confidence: bool = False
    numeric_features: Tuple[str, ...] = FUNDUS_NUMERIC_FEATURES
    categorical_features: Tuple[str, ...] = FUNDUS_CATEGORICAL_FEATURES
    image_encoder: str = "resnet152"
    image_weights_path: Optional[str] = None
    text_model_name: str = "google-bert/bert-base-multilingual-uncased"
    text_revision: Optional[str] = None
    text_max_length: int = 128
    cache_dir: str = ".cache/glaboost"
    device: str = "cpu"
    image_batch_size: int = 16
    text_batch_size: int = 16
    # GlaBoost, III.G (pp. 4-5).
    learning_rate: float = 0.05
    max_depth: int = 6
    n_estimators: int = 100
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    # The paper does not specify these choices.
    random_state: int = 42
    n_jobs: int = 1
    tree_method: str = "hist"
    gpu_id: Optional[int] = None

    def __post_init__(self):
        if self.image_encoder not in ("resnet152", "resnet18"):
            raise ValueError("image_encoder must be resnet152 or resnet18.")
        for name in ("subsample", "colsample_bytree"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not 0 < value <= 1:
                raise ValueError(f"{name} must be a number in (0, 1].")
        for name in ("numeric_features", "categorical_features"):
            values = getattr(self, name)
            if not isinstance(values, (tuple, list)):
                raise ValueError(f"{name} must be a list or tuple of field names, not a string.")
        object.__setattr__(self, "numeric_features", tuple(self.numeric_features))
        object.__setattr__(self, "categorical_features", tuple(self.categorical_features))
        toggles = (self.use_image, self.use_text, self.use_structured,
                   self.use_human_risk, self.use_human_confidence)
        if not all(isinstance(value, bool) for value in toggles) or not any(toggles):
            raise ValueError("Enable at least one modality using boolean flags.")
        for name in ("max_depth", "n_estimators", "n_jobs", "image_batch_size",
                     "text_batch_size", "text_max_length"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not 0 < self.learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1].")
        if not isinstance(self.random_state, int) or not 0 <= self.random_state < 2**32:
            raise ValueError("random_state must be an integer in [0, 2**32).")
        if not 3 <= self.text_max_length <= 512:
            raise ValueError("mBERT text_max_length must be between 3 and 512.")
        if self.tree_method not in {"hist", "exact", "approx", "gpu_hist"}:
            raise ValueError("XGBoost tree_method must be hist, exact, approx, or gpu_hist.")
        if self.gpu_id is not None and (isinstance(self.gpu_id, bool) or
                                        not isinstance(self.gpu_id, int) or self.gpu_id < 0):
            raise ValueError("gpu_id must be a nonnegative integer or None.")
        if self.tree_method == "gpu_hist" and self.gpu_id is None:
            raise ValueError("gpu_hist requires an explicit gpu_id.")
        fields = self.numeric_features + self.categorical_features
        if any(not isinstance(f, str) or not f.strip() or f != f.strip() for f in fields):
            raise ValueError("Structured feature names must be nonempty strings without surrounding whitespace.")
        if len(set(fields)) != len(fields):
            raise ValueError("Structured feature names must be nonempty and unique.")
        if self.use_structured and not fields:
            raise ValueError("Structured modality needs an explicit feature schema.")
        forbidden = {"label", "target", "annotation", "diagnosis", "filename",
                     "sample_id", "patient_id", "eye_id", "plr2", "plr3",
                     "md", "md_slope", "progression", "total_visits",
                     "glaucoma_risk_assessment", "confidence_level"}
        if any(f.lower() in forbidden for f in fields):
            raise ValueError("Labels, identifiers, outcomes, and human assessments are not structured predictors.")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, state):
        return cls(**state)

    @classmethod
    def for_image_method(cls, method="paper", **kwargs):
        """Select a diagnosis image recipe independently of any evaluation cohort.

        ``ch`` follows the senior notebook's available image/XGBoost branch.
        ``paper`` uses the paper-based ResNet152 image configuration. Neither
        recipe fits a model or establishes independence from validation data.
        """
        if method == "ch":
            settings = dict(image_encoder="resnet18", n_estimators=500,
                            learning_rate=0.05, max_depth=6,
                            subsample=0.8, colsample_bytree=0.8)
        elif method == "paper":
            settings = dict(image_encoder="resnet152", n_estimators=100,
                            learning_rate=0.05, max_depth=6,
                            subsample=1.0, colsample_bytree=1.0)
        else:
            raise ValueError("Image method must be 'ch' or 'paper'.")
        settings.update(kwargs)
        return cls(**settings)
