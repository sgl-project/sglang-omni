# SPDX-License-Identifier: Apache-2.0
"""Direct weight loading for the AuK DiT and VAE."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

WEIGHT_FILE_CANDIDATES = (
    "auk_base.safetensors",
    "auk_flash.safetensors",
    "auk.safetensors",
    "model.safetensors",
)

VAE_FILE_CANDIDATES = (
    "vae.safetensors",
    "vae/vae.safetensors",
    "audio_vae.safetensors",
)

_STRIPPABLE_PREFIXES = ("ema_model.", "module.", "model.", "net.")


@dataclass
class LoadReport:
    loaded: int = 0
    missing: int = 0
    unexpected: int = 0
    missing_examples: tuple[str, ...] = ()
    unexpected_examples: tuple[str, ...] = ()

    def __str__(self) -> str:
        return (
            f"loaded={self.loaded} missing={self.missing} unexpected={self.unexpected}"
        )


def _first_existing(root: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        candidate = root / name
        if candidate.is_file():
            return candidate
    return None


def resolve_weight_file(model_path: str) -> Path:
    """Locate the DiT checkpoint inside model_path."""
    root = Path(model_path)
    if root.is_file():
        return root
    found = _first_existing(root, WEIGHT_FILE_CANDIDATES)
    if found is not None:
        return found
    globbed = sorted(root.glob("*.safetensors"))
    vae_names = set(VAE_FILE_CANDIDATES)
    globbed = [p for p in globbed if p.name not in vae_names]
    if not globbed:
        raise FileNotFoundError(f"No AuK weights found under {model_path}")
    return globbed[0]


def resolve_vae_file(model_path: str) -> Path | None:
    root = Path(model_path)
    if root.is_file():
        root = root.parent
    return _first_existing(root, VAE_FILE_CANDIDATES)


def _read_safetensors(path: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    return load_file(str(path), device="cpu")


def _strip_prefix(key: str) -> str:
    for prefix in _STRIPPABLE_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def normalize_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Drop EMA/DDP wrappers so keys match the transformer module tree."""
    return {_strip_prefix(k): v for k, v in state_dict.items()}


def _load_weights(module: torch.nn.Module, path: Path) -> LoadReport:
    state_dict = normalize_state_dict(_read_safetensors(path))
    module.load_state_dict(state_dict, strict=True, assign=True)
    report = LoadReport(loaded=len(state_dict))
    logger.info("AuK: loaded weights from %s (%s)", path, report)
    return report


def load_dit_weights(
    flow: torch.nn.Module,
    model_path: str,
) -> LoadReport:
    """Load the DiT and layer-fusion parameters into AuKFlowMatching."""
    return _load_weights(flow, resolve_weight_file(model_path))


def load_vae_weights(
    vae: torch.nn.Module,
    model_path: str,
) -> LoadReport:
    """Load vae.safetensors into a BigVGANFlowVAE."""
    vae_file = resolve_vae_file(model_path)
    if vae_file is None:
        raise FileNotFoundError(f"No AuK VAE weights found under {model_path}")
    return _load_weights(vae, vae_file)
