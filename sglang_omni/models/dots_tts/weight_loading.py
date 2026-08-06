# SPDX-License-Identifier: Apache-2.0
"""Checkpoint weight routing for the dots.tts SGLang engine."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file

logger = logging.getLogger(__name__)

OWNER_AR_BACKBONE = "ar_backbone"
OWNER_TTS_TAIL = "tts_tail"

AR_BACKBONE_PREFIX = "llm."
# note (chenyang): tie_word_embeddings makes llm.lm_head.weight an alias of the
# embedding, and the SGLang backbone never materializes a text head.
LM_HEAD_SKIP_PREFIX = "llm.lm_head."

TTS_TAIL_PREFIXES = (
    "patch_encoder.",
    "velocity_field_predictor.",
    "coordinate_proj.",
    "hidden_proj.",
    "latent_proj.",
    "xvec_proj.",
    "eos_proj.",
)


@dataclass
class DotsTtsWeightReport:
    """Per-owner accounting used to fail fast on checkpoint drift."""

    loaded: dict[str, int] = field(
        default_factory=lambda: {OWNER_AR_BACKBONE: 0, OWNER_TTS_TAIL: 0}
    )
    skipped: list[str] = field(default_factory=list)
    leftovers: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    def add_loaded(self, owner: str) -> None:
        self.loaded[owner] = self.loaded[owner] + 1

    def summary(self) -> str:
        return (
            "dots.tts weight load: "
            f"ar_backbone={self.loaded[OWNER_AR_BACKBONE]} "
            f"tts_tail={self.loaded[OWNER_TTS_TAIL]} "
            f"skipped={len(self.skipped)}"
        )


def classify_dots_tts_weight(name: str) -> str | None:
    """Owner of one checkpoint tensor, or ``None`` when it must be skipped."""
    if name.startswith(LM_HEAD_SKIP_PREFIX):
        return None
    if name.startswith(AR_BACKBONE_PREFIX):
        return OWNER_AR_BACKBONE
    if name.startswith(TTS_TAIL_PREFIXES):
        return OWNER_TTS_TAIL
    return None


def assert_dots_tts_weight_coverage(report: DotsTtsWeightReport) -> None:
    if report.leftovers:
        raise RuntimeError(
            "dots.tts checkpoint contains unroutable tensors: "
            f"{sorted(report.leftovers)[:8]}"
        )
    if report.missing:
        raise RuntimeError(
            f"dots.tts model parameters were never loaded: {report.missing[:8]}"
        )
    for owner, count in report.loaded.items():
        if count == 0:
            raise RuntimeError(f"dots.tts checkpoint loaded no weights for {owner}")


def load_dots_tts_module_weights(module: nn.Module, checkpoint_path: str | Path) -> int:
    """Load one standalone safetensors artifact into ``module``, strictly."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"dots.tts artifact is missing: {path}")
    state_dict = load_file(str(path), device="cpu")
    mismatch = module.load_state_dict(state_dict, strict=False)
    if mismatch.missing_keys or mismatch.unexpected_keys:
        raise RuntimeError(
            f"dots.tts artifact {path.name} does not match the module: "
            f"missing={mismatch.missing_keys[:8]} "
            f"unexpected={mismatch.unexpected_keys[:8]}"
        )
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return len(state_dict)


def load_dots_tts_latent_stats(
    checkpoint_path: str | Path,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Global latent mean/std used to normalize acoustic latents."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"dots.tts latent stats are missing: {path}")
    stats = torch.load(path, weights_only=False)
    mean = torch.as_tensor(stats["mean"], dtype=torch.float32)
    std = torch.sqrt(torch.as_tensor(stats["var"], dtype=torch.float32))
    return mean, std


__all__ = [
    "OWNER_AR_BACKBONE",
    "OWNER_TTS_TAIL",
    "DotsTtsWeightReport",
    "assert_dots_tts_weight_coverage",
    "classify_dots_tts_weight",
    "load_dots_tts_latent_stats",
    "load_dots_tts_module_weights",
]
