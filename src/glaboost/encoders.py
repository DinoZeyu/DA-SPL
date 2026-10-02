"""Frozen image/text encoders for the documented GlaBoost method variants.

Networks are loaded on the first nonempty ``transform`` call. Downloads require
an explicit opt-in; the default cache belongs to this project, not the user home.
No training, augmentation, or missing-modality imputation happens here.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
from PIL import Image


_DEFAULT_CACHE = Path(__file__).resolve().parents[2] / ".cache" / "glaboost"
_RESNET_URL = "https://download.pytorch.org/models/resnet152-394f9c45.pth"
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_PUNCTUATION = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"',
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-",
    "\u2014": "-", "\u2015": "-", "\u2212": "-", "\u2026": "...",
})


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resnet152_uninitialized():
    from torchvision.models import resnet152

    return resnet152(weights=None)


def _resnet18_uninitialized():
    from torchvision.models import resnet18

    return resnet18(weights=None)


def _transformer_classes():
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError("MBERTEncoder requires the project's transformers dependency") from exc
    return AutoModel, AutoTokenizer


def _rows(inputs: Sequence[Any]) -> list:
    if isinstance(inputs, (str, bytes, bytearray, Path, Image.Image)):
        raise TypeError("inputs must be a sequence of samples, even for a single sample")
    return list(inputs)


def resolve_image_devices(device: str):
    """Return primary device and selected CUDA IDs, honoring torch visibility.

    ``auto`` falls back to CPU and otherwise selects every visible GPU, as does
    explicit ``cuda``. ``cuda:N`` selects one visible index. No model, weights,
    or network access is needed. Explicit unavailable CUDA never silently falls
    back to CPU. GPU IDs refer to the CUDA_VISIBLE_DEVICES-remapped namespace.
    """
    import torch

    requested = str(device)
    if requested == "cpu":
        return "cpu", ()
    if requested not in ("auto", "cuda") and not re.fullmatch(r"cuda:[0-9]+", requested):
        raise ValueError("device must be 'auto', 'cpu', 'cuda', or 'cuda:N'")
    count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    if requested == "auto" and count == 0:
        return "cpu", ()
    if count == 0:
        raise RuntimeError(f"Requested {requested}, but no CUDA GPUs are available to PyTorch")
    if requested in ("auto", "cuda"):
        return "cuda:0", tuple(range(count))
    index = int(requested.split(":", 1)[1])
    if index >= count:
        raise RuntimeError(f"Requested {requested}, but PyTorch sees only {count} CUDA GPU(s)")
    return f"cuda:{index}", (index,)


def _runtime_spec(requested_device, resolved_device, gpu_ids, batch_size):
    return {
        "requested_device": requested_device,
        "resolved_device": resolved_device,
        "active_gpu_ids": list(gpu_ids),
        "active_gpu_count": len(gpu_ids),
        "batch_size": batch_size,
        "batch_size_scope": "global",
        "parallelism": ("unresolved" if resolved_device is None else
                        "data_parallel" if len(gpu_ids) > 1 else "single_device"),
    }


def normalize_rim_text(text: str) -> str:
    """Normalize Unicode, punctuation, case, and whitespace without inventing text."""
    if not isinstance(text, str):
        raise ValueError("Rim text must be a nonempty string; missing text is not imputed")
    normalized = unicodedata.normalize("NFKC", text).translate(_PUNCTUATION).lower()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        raise ValueError("Rim text must be nonempty; missing text is not imputed")
    return normalized


class ResNet152Encoder:
    """ImageNet-1K V1 ResNet152 global-average-pool embeddings (2048 values).

    ``weights_path`` accepts a trusted, complete torchvision ResNet152 state
    dictionary, including its original classifier. It is loaded strictly before
    the classifier is removed. A local checkpoint is identified by its SHA256;
    this cannot establish that a user-provided checkpoint is ImageNet pretrained.

    ``auto`` / ``cuda`` use all torch-visible GPUs; explicit ``cuda:N`` uses one.
    ``batch_size`` always describes the global batch, divided across devices.
    """

    output_dim = 2048
    encoder_name = "resnet152"
    display_name = "ResNet152"
    checkpoint_filename = "resnet152-394f9c45.pth"
    checkpoint_sha256_prefix = "394f9c45"
    weights_url = _RESNET_URL
    weights_name = "ResNet152_Weights.IMAGENET1K_V1"
    normalize_image = True

    @staticmethod
    def _build_model():
        return _resnet152_uninitialized()

    def __init__(self, *, weights_path: Optional[Union[str, Path]] = None,
                 cache_dir: Optional[Union[str, Path]] = None, device: str = "cpu",
                 batch_size: int = 16, allow_download: bool = False):
        self.weights_path = Path(weights_path).expanduser().resolve() if weights_path is not None else None
        self.cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir is not None else _DEFAULT_CACHE
        self.device = str(device)
        self.batch_size = _positive_integer(batch_size, "batch_size")
        self.allow_download = bool(allow_download)
        self._model = None
        self._source = None
        self._resolved_device = None
        self._gpu_ids = ()

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch

        self._resolved_device, self._gpu_ids = resolve_image_devices(self.device)
        checkpoint = self.weights_path or self.cache_dir / "torch" / self.checkpoint_filename
        if not checkpoint.is_file():
            if self.weights_path is not None or not self.allow_download:
                raise FileNotFoundError(
                    f"Pretrained {self.display_name} weights not found: {checkpoint}. "
                    "Supply weights_path or explicitly enable allow_download. "
                    "Random weights are never used."
                )
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            state = torch.hub.load_state_dict_from_url(
                self.weights_url, model_dir=str(checkpoint.parent), map_location="cpu",
                progress=True, check_hash=True,
            )
        else:
            state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        digest = _sha256(checkpoint)
        if self.weights_path is None and not digest.startswith(self.checkpoint_sha256_prefix):
            raise ValueError(f"Cached {self.display_name} weights have an invalid official SHA256 prefix: {checkpoint}")
        model = self._build_model()
        model.load_state_dict(state, strict=True)
        model.fc = torch.nn.Identity()
        model.requires_grad_(False)
        model.eval()
        model.to(self._resolved_device)
        if len(self._gpu_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=list(self._gpu_ids),
                                          output_device=self._gpu_ids[0])
            model.eval()
        self._source = {
            "kind": "local_checkpoint" if self.weights_path is not None else "torchvision",
            "path": str(checkpoint), "sha256": digest,
            "weights": None if self.weights_path is not None else self.weights_name,
            "url": None if self.weights_path is not None else self.weights_url,
        }
        self._model = model

    @classmethod
    def _prepare(cls, value: Any):
        import torch

        if isinstance(value, Image.Image):
            image = value.convert("RGB")
        elif isinstance(value, (str, Path)):
            if not str(value).strip():
                raise ValueError("Missing image path")
            with Image.open(value) as opened:
                image = opened.convert("RGB")
        elif isinstance(value, (bytes, bytearray)):
            with Image.open(io.BytesIO(value)) as opened:
                image = opened.convert("RGB")
        else:
            raise ValueError("Image must be a path, encoded bytes, or PIL image; missing images are not imputed")
        image = image.resize((224, 224), resample=Image.Resampling.BILINEAR)
        array = np.array(image, dtype=np.float32, copy=True) / np.float32(255.0)
        if cls.normalize_image:
            array = (array - np.asarray(_IMAGENET_MEAN, dtype=np.float32)) / np.asarray(_IMAGENET_STD, dtype=np.float32)
        return torch.from_numpy(array.transpose(2, 0, 1).copy())

    def transform(self, inputs: Sequence[Any]) -> np.ndarray:
        rows = _rows(inputs)
        if not rows:
            return np.empty((0, self.output_dim), dtype=np.float32)
        import torch

        self._load()
        self._model.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, len(rows), self.batch_size):
                batch = []
                for offset, value in enumerate(rows[start:start + self.batch_size]):
                    try:
                        batch.append(self._prepare(value))
                    except (OSError, TypeError, ValueError) as exc:
                        raise ValueError(f"Invalid image at row {start + offset}: {exc}") from exc
                encoded = self._model(torch.stack(batch).to(self._resolved_device))
                if tuple(encoded.shape) != (len(batch), self.output_dim):
                    raise ValueError(f"{self.display_name} returned unexpected shape {tuple(encoded.shape)}")
                outputs.append(encoded.detach().cpu().numpy().astype(np.float32, copy=False))
        return np.concatenate(outputs, axis=0)

    def spec(self) -> dict:
        """Return provenance without loading a network or accessing the network."""
        return {
            "encoder": self.encoder_name, "output_dim": self.output_dim, "frozen": True,
            "loaded": self._model is not None,
            "fingerprint": "sha256:" + self._source["sha256"] if self._source is not None else None,
            "source": dict(self._source) if self._source is not None else None,
            "requested_weights_path": str(self.weights_path) if self.weights_path is not None else None,
            "requested_weights": self.weights_name if self.weights_path is None else "local_state_dict",
            "cache_dir": str(self.cache_dir),
            **_runtime_spec(self.device, self._resolved_device, self._gpu_ids, self.batch_size),
            "preprocessing": {"color": "RGB", "resize": [224, 224], "interpolation": "bilinear",
                              "scale": "divide_by_255", "mean": list(_IMAGENET_MEAN) if self.normalize_image else None,
                              "std": list(_IMAGENET_STD) if self.normalize_image else None, "augmentation": False},
        }


class ResNet18Encoder(ResNet152Encoder):
    """Senior notebook image branch: frozen ImageNet V1 ResNet18, 512 values.

    The notebook uses PIL RGB images, Resize((224, 224)) and ToTensor without
    ImageNet mean/std normalization. This intentionally preserves that recipe.
    Weight validation, lazy loading and multi-GPU execution match ResNet152.
    """

    output_dim = 512
    encoder_name = "resnet18"
    display_name = "ResNet18"
    checkpoint_filename = "resnet18-f37072fd.pth"
    checkpoint_sha256_prefix = "f37072fd"
    weights_url = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
    weights_name = "ResNet18_Weights.IMAGENET1K_V1"
    normalize_image = False

    @staticmethod
    def _build_model():
        return _resnet18_uninitialized()


def image_encoder_class(name):
    """Resolve a configured image encoder without loading weights or CUDA."""
    if name == "resnet152":
        return ResNet152Encoder
    if name == "resnet18":
        return ResNet18Encoder
    raise ValueError("image encoder must be resnet152 or resnet18")


class MBERTEncoder:
    """Frozen uncased multilingual BERT with attention-mask mean pooling.

    The paper does not specify treatment of special tokens in mean pooling. This
    implementation includes [CLS] and [SEP] and excludes all padding tokens.
    ``auto`` uses the first visible CUDA GPU or CPU; this optional encoder is
    deliberately single-device even when the image encoder uses several GPUs.
    """

    output_dim = 768

    def __init__(self, *, model_name: str = "google-bert/bert-base-multilingual-uncased",
                 revision: Optional[str] = None, cache_dir: Optional[Union[str, Path]] = None,
                 device: str = "cpu", batch_size: int = 16, max_length: int = 128,
                 allow_download: bool = False):
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must identify pretrained mBERT weights or a local model directory")
        self.model_name = model_name
        self.revision = revision
        self.cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir is not None else _DEFAULT_CACHE
        self.device = str(device)
        self.batch_size = _positive_integer(batch_size, "batch_size")
        self.max_length = _positive_integer(max_length, "max_length")
        if not 3 <= self.max_length <= 512:
            raise ValueError("max_length must be between 3 and 512; the main method uses 128")
        self.allow_download = bool(allow_download)
        self._model = None
        self._tokenizer = None
        self._source = None
        self._resolved_device = None
        self._gpu_ids = ()

    def _load(self) -> None:
        if self._model is not None:
            return
        self._resolved_device, gpu_ids = resolve_image_devices(self.device)
        self._gpu_ids = gpu_ids[:1]
        AutoModel, AutoTokenizer = _transformer_classes()
        kwargs = {
            "cache_dir": str(self.cache_dir / "huggingface"),
            "revision": self.revision,
            "local_files_only": not self.allow_download,
            "trust_remote_code": False,
        }
        try:
            model, loading_info = AutoModel.from_pretrained(
                self.model_name, add_pooling_layer=False, output_loading_info=True, **kwargs,
            )
            if loading_info.get("missing_keys") or loading_info.get("mismatched_keys") or loading_info.get("error_msgs"):
                raise ValueError(f"Incomplete pretrained mBERT checkpoint; refusing random parameter initialization: {loading_info}")
            if getattr(model.config, "model_type", None) != "bert" or model.config.hidden_size != self.output_dim:
                raise ValueError("The mBERT encoder requires a BERT checkpoint with hidden_size=768")
            commit = getattr(model.config, "_commit_hash", None)
            tokenizer_kwargs = dict(kwargs)
            if commit:
                tokenizer_kwargs["revision"] = commit
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, **tokenizer_kwargs)
        except OSError as exc:
            raise FileNotFoundError(
                f"Could not load pretrained mBERT '{self.model_name}' from {self.cache_dir / 'huggingface'}. "
                f"allow_download={self.allow_download}; supply a complete local model or explicitly enable downloading."
            ) from exc
        model.requires_grad_(False)
        model.eval()
        model.to(self._resolved_device)
        local_dir = Path(self.model_name).expanduser()
        hashes = None
        if local_dir.is_dir():
            hashes = {str(path.relative_to(local_dir)): _sha256(path)
                      for path in sorted(local_dir.rglob("*")) if path.is_file()
                      and path.suffix in {".json", ".txt", ".bin", ".safetensors"}}
            fingerprint = "sha256:" + hashlib.sha256(
                json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            resolved_revision = None
        else:
            resolved_revision = commit
            if not resolved_revision and self.revision and re.fullmatch(r"[0-9a-fA-F]{40}", self.revision):
                resolved_revision = self.revision
            if not resolved_revision:
                raise ValueError("mBERT checkpoint has no resolved commit; specify an exact Hugging Face commit revision")
            fingerprint = "hf_commit:" + resolved_revision
        self._source = {"model_name": self.model_name, "requested_revision": self.revision,
                        "resolved_commit": commit, "local_file_sha256": hashes,
                        "fingerprint": fingerprint, "resolved_revision": resolved_revision,
                        "ignored_checkpoint_keys": list(loading_info.get("unexpected_keys", []))}
        self._model, self._tokenizer = model, tokenizer

    def transform(self, inputs: Sequence[str]) -> np.ndarray:
        rows = _rows(inputs)
        normalized = []
        for index, value in enumerate(rows):
            try:
                normalized.append(normalize_rim_text(value))
            except ValueError as exc:
                raise ValueError(f"Invalid rim text at row {index}: {exc}") from exc
        if not normalized:
            return np.empty((0, self.output_dim), dtype=np.float32)
        import torch

        self._load()
        self._model.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, len(normalized), self.batch_size):
                batch = normalized[start:start + self.batch_size]
                tokens = self._tokenizer(batch, padding=True, truncation=True,
                                         max_length=self.max_length, return_tensors="pt",
                                         return_attention_mask=True)
                tokens = {key: value.to(self._resolved_device) for key, value in tokens.items()}
                hidden = self._model(**tokens).last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1).to(dtype=hidden.dtype)
                counts = mask.sum(dim=1)
                if torch.any(counts == 0):
                    raise ValueError("Tokenizer returned a sample without any unmasked tokens")
                pooled = (hidden * mask).sum(dim=1) / counts
                if tuple(pooled.shape) != (len(batch), self.output_dim):
                    raise ValueError(f"mBERT returned unexpected shape {tuple(pooled.shape)}")
                outputs.append(pooled.detach().cpu().numpy().astype(np.float32, copy=False))
        return np.concatenate(outputs, axis=0)

    def spec(self) -> dict:
        return {
            "encoder": "mbert", "output_dim": self.output_dim, "frozen": True,
            "loaded": self._model is not None,
            "fingerprint": self._source["fingerprint"] if self._source is not None else None,
            "resolved_revision": self._source["resolved_revision"] if self._source is not None else None,
            "source": dict(self._source) if self._source is not None else None,
            "requested_model_name": self.model_name, "requested_revision": self.revision,
            "cache_dir": str(self.cache_dir),
            **_runtime_spec(self.device, self._resolved_device, self._gpu_ids, self.batch_size),
            "preprocessing": {"unicode": "NFKC", "punctuation": "curly_quotes_dashes_ellipsis_to_ascii",
                              "lowercase": True, "collapse_whitespace": True,
                              "max_length": self.max_length, "truncation": True,
                              "pooling": "attention_mask_mean", "include_special_tokens": True,
                              "special_token_policy_is_assumption": True},
        }
