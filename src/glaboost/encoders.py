"""Frozen ResNet152 image encoder for the GRAPE validation pipeline.

Networks are loaded on the first nonempty ``transform`` call. Downloads require
an explicit opt-in; the default cache belongs to this project, not the user home.
No training, augmentation, or missing-modality imputation happens here.
"""

from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
from PIL import Image


_DEFAULT_CACHE = Path(__file__).resolve().parents[2] / ".cache" / "glaboost"
_RESNET_URL = "https://download.pytorch.org/models/resnet152-394f9c45.pth"
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


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
                              "scale": "divide_by_255", "mean": list(_IMAGENET_MEAN),
                              "std": list(_IMAGENET_STD), "augmentation": False},
        }


def image_encoder_class(name):
    """Resolve a configured image encoder without loading weights or CUDA."""
    if name == "resnet152":
        return ResNet152Encoder
    raise ValueError("image encoder must be resnet152")
