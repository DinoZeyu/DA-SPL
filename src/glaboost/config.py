"""Configuration for the fixed ResNet152 image features and XGBoost classifier."""

from dataclasses import asdict, dataclass
from numbers import Real
from typing import Optional, Tuple

@dataclass(frozen=True)
class GlaBoostConfig:
    # Disabled legacy fields remain serializable so existing image-only model
    # bundles load without changing their recorded training configuration.
    use_image: bool = True
    use_text: bool = False
    use_structured: bool = False
    use_human_risk: bool = False
    use_human_confidence: bool = False
    numeric_features: Tuple[str, ...] = ("cup_to_disc_ratio",)
    categorical_features: Tuple[str, ...] = (
        "optic_disc_size", "isnt_rule_followed", "rim_pallor", "rim_color", "bayoneting",
        "sharp_edge", "laminar_dot_sign", "notching", "rim_thinning")
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
        if self.image_encoder != "resnet152":
            raise ValueError("The supported image_encoder is resnet152.")
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
        if not all(isinstance(value, bool) for value in toggles) or toggles != (True, False, False, False, False):
            raise ValueError("GlaBoost supports image-only ResNet152; text, structured and human inputs must remain disabled.")
        for name in ("max_depth", "n_estimators", "n_jobs", "image_batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not 0 < self.learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1].")
        if not isinstance(self.random_state, int) or not 0 <= self.random_state < 2**32:
            raise ValueError("random_state must be an integer in [0, 2**32).")
        if self.tree_method not in {"hist", "exact", "approx", "gpu_hist"}:
            raise ValueError("XGBoost tree_method must be hist, exact, approx, or gpu_hist.")
        if self.gpu_id is not None and (isinstance(self.gpu_id, bool) or
                                        not isinstance(self.gpu_id, int) or self.gpu_id < 0):
            raise ValueError("gpu_id must be a nonnegative integer or None.")
        if self.tree_method == "gpu_hist" and self.gpu_id is None:
            raise ValueError("gpu_hist requires an explicit gpu_id.")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, state):
        return cls(**state)

    @classmethod
    def for_image_method(cls, method="paper", **kwargs):
        """Paper image settings, with explicit tree overrides saved in metadata."""
        if method != "paper":
            raise ValueError("Only the ResNet152 paper image method is supported.")
        settings = dict(image_encoder="resnet152", n_estimators=100,
                        learning_rate=0.05, max_depth=6,
                        subsample=1.0, colsample_bytree=1.0)
        settings.update(kwargs)
        return cls(**settings)
