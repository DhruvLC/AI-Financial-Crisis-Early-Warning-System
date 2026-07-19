"""Configurable augmentations for tabular financial features.

Each augmentation is a small callable ``(x, generator) -> x'`` over one
sample tensor of shape ``(n_features,)`` (or a batch ``(B, n_features)`` —
all operations broadcast), drawing randomness only from the passed
:class:`torch.Generator` so view generation is fully deterministic under
the pipeline seed.

Implemented (all suitable for standardized tabular data):

* :class:`FeatureMasking`      — zero a random fraction of features
* :class:`GaussianNoise`       — additive N(0, sigma) noise
* :class:`FeatureDropout`      — Bernoulli dropout without rescaling
* :class:`RandomCorruption`    — replace a random fraction with N(0, 1)
  draws (marginal-like corruption, VIME/SCARF-style)
* :class:`ColumnShuffle`       — swap-noise: replace a fraction of values
  with the same feature's value from another in-batch row (batch input
  only; degrades to corruption on single samples)
* :class:`Mixup`               — convex combination with a shuffled batch
  row (optional; batch input only, identity on single samples)

:func:`build_augmentations` turns the config list into a composed
:class:`AugmentationPipeline`; unknown names or invalid probabilities
raise :class:`SSLError`.
"""
from __future__ import annotations

import torch

from ingestion.logging_config import get_logger

from .base import SSLError

__all__ = ["AugmentationPipeline", "FeatureMasking", "GaussianNoise",
           "FeatureDropout", "RandomCorruption", "ColumnShuffle", "Mixup",
           "AUGMENTATION_REGISTRY", "build_augmentations"]

log = get_logger("ssl.augment")


def _check_prob(name: str, key: str, value: float) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise SSLError(f"augmentation '{name}': {key}={value} must be "
                       f"in [0, 1]")
    return value


class BaseAugmentation:
    """Template: subclasses implement :meth:`apply`."""

    name = "base"

    def __call__(self, x: torch.Tensor,
                 generator: torch.Generator) -> torch.Tensor:
        out = self.apply(x, generator)
        if out.shape != x.shape:
            raise SSLError(f"augmentation '{self.name}' changed the "
                           f"shape {tuple(x.shape)} -> {tuple(out.shape)}")
        return out

    def apply(self, x: torch.Tensor,
              generator: torch.Generator) -> torch.Tensor:
        raise NotImplementedError

    def params(self) -> dict:
        return {}

    def as_dict(self) -> dict:
        return {"name": self.name, **self.params()}


class FeatureMasking(BaseAugmentation):
    """Zero out each feature independently with probability ``ratio``."""

    name = "feature_masking"

    def __init__(self, ratio: float = 0.15) -> None:
        self.ratio = _check_prob(self.name, "ratio", ratio)

    def apply(self, x, generator):
        mask = torch.rand(x.shape, generator=generator) >= self.ratio
        return x * mask

    def params(self):
        return {"ratio": self.ratio}


class GaussianNoise(BaseAugmentation):
    """Add N(0, sigma^2) noise to every feature."""

    name = "gaussian_noise"

    def __init__(self, sigma: float = 0.1) -> None:
        self.sigma = float(sigma)
        if self.sigma < 0:
            raise SSLError(f"gaussian_noise: sigma={sigma} must be >= 0")

    def apply(self, x, generator):
        return x + torch.randn(x.shape, generator=generator) * self.sigma

    def params(self):
        return {"sigma": self.sigma}


class FeatureDropout(BaseAugmentation):
    """Bernoulli feature dropout (no inverse scaling — views should stay
    on the data scale)."""

    name = "feature_dropout"

    def __init__(self, p: float = 0.1) -> None:
        self.p = _check_prob(self.name, "p", p)

    def apply(self, x, generator):
        keep = torch.rand(x.shape, generator=generator) >= self.p
        return x * keep

    def params(self):
        return {"p": self.p}


class RandomCorruption(BaseAugmentation):
    """Replace a random fraction of features with standard-normal draws —
    a marginal-distribution stand-in for standardized features
    (VIME / SCARF style)."""

    name = "random_corruption"

    def __init__(self, ratio: float = 0.1, scale: float = 1.0) -> None:
        self.ratio = _check_prob(self.name, "ratio", ratio)
        self.scale = float(scale)

    def apply(self, x, generator):
        corrupt = torch.rand(x.shape, generator=generator) < self.ratio
        noise = torch.randn(x.shape, generator=generator) * self.scale
        return torch.where(corrupt, noise, x)

    def params(self):
        return {"ratio": self.ratio, "scale": self.scale}


class ColumnShuffle(BaseAugmentation):
    """Swap-noise: replace a fraction of values with the same column's
    value from a random other row of the batch. Only meaningful on batch
    input ``(B, F)``; on single samples it falls back to
    :class:`RandomCorruption` semantics so per-sample loaders still work.
    """

    name = "column_shuffle"

    def __init__(self, ratio: float = 0.1) -> None:
        self.ratio = _check_prob(self.name, "ratio", ratio)
        self._fallback = RandomCorruption(ratio)

    def apply(self, x, generator):
        if x.dim() < 2 or len(x) < 2:
            return self._fallback.apply(x, generator)
        perm = torch.randperm(len(x), generator=generator)
        swap = torch.rand(x.shape, generator=generator) < self.ratio
        return torch.where(swap, x[perm], x)

    def params(self):
        return {"ratio": self.ratio}


class Mixup(BaseAugmentation):
    """Convex combination with a shuffled batch row:
    ``lam * x + (1 - lam) * x[perm]`` with ``lam ~ Beta(alpha, alpha)``
    clamped to keep ``x`` dominant. Identity on single samples."""

    name = "mixup"

    def __init__(self, alpha: float = 0.2) -> None:
        self.alpha = float(alpha)
        if self.alpha <= 0:
            raise SSLError(f"mixup: alpha={alpha} must be > 0")

    def apply(self, x, generator):
        if x.dim() < 2 or len(x) < 2:
            return x
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample()
        lam = torch.clamp(lam, 0.5, 1.0)  # keep original view dominant
        perm = torch.randperm(len(x), generator=generator)
        return lam * x + (1.0 - lam) * x[perm]

    def params(self):
        return {"alpha": self.alpha}


AUGMENTATION_REGISTRY: dict[str, type[BaseAugmentation]] = {
    cls.name: cls for cls in (FeatureMasking, GaussianNoise, FeatureDropout,
                              RandomCorruption, ColumnShuffle, Mixup)
}


class AugmentationPipeline:
    """Sequentially apply the configured augmentations to one view."""

    def __init__(self, augmentations: list[BaseAugmentation]) -> None:
        if not augmentations:
            raise SSLError("augmentation pipeline is empty — contrastive "
                           "views need at least one augmentation")
        self.augmentations = list(augmentations)

    def __call__(self, x: torch.Tensor,
                 generator: torch.Generator) -> torch.Tensor:
        for aug in self.augmentations:
            x = aug(x, generator)
        return x

    def as_dict(self) -> list[dict]:
        return [a.as_dict() for a in self.augmentations]


def build_augmentations(cfg: list | None) -> AugmentationPipeline:
    """Config list → composed pipeline (raises :class:`SSLError`).

    Config shape::

        - {name: feature_masking, ratio: 0.15}
        - {name: gaussian_noise, sigma: 0.1}
    """
    cfg = cfg if cfg is not None else [{"name": "feature_masking",
                                        "ratio": 0.15},
                                       {"name": "gaussian_noise",
                                        "sigma": 0.1}]
    augs: list[BaseAugmentation] = []
    for item in cfg:
        item = dict(item)
        name = str(item.pop("name", "")).lower()
        cls = AUGMENTATION_REGISTRY.get(name)
        if cls is None:
            raise SSLError(f"unsupported augmentation '{name}' "
                           f"(supported: {sorted(AUGMENTATION_REGISTRY)})")
        try:
            augs.append(cls(**item))
        except TypeError as exc:
            raise SSLError(f"invalid parameters for augmentation "
                           f"'{name}': {exc}") from exc
    log.info("augmentation pipeline: %s",
             " -> ".join(a.name for a in augs))
    return AugmentationPipeline(augs)
