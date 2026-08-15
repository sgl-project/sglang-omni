# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from sglang_omni.models.accelerator_support import (
    AcceleratorSupportStatus,
    get_model_accelerator_support,
    iter_model_accelerator_support,
)
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY


def _canonical_architectures() -> set[str]:
    return {
        str(config_cls.architecture)
        for config_cls in set(PIPELINE_CONFIG_REGISTRY.configs.values())
    }


def test_rocm_declarations_cover_every_canonical_model() -> None:
    declarations = list(iter_model_accelerator_support("rocm"))

    assert {item.architecture for item in declarations} == _canonical_architectures()
    assert len(declarations) == 19
    assert all(item.gpu_architectures == ("gfx942", "gfx950") for item in declarations)


def test_minimax_music3_is_supported_on_both_rocm_targets() -> None:
    support = get_model_accelerator_support(
        "MiniMaxMusic3ForConditionalGeneration", "ROCm"
    )

    assert support is not None
    assert support.status is AcceleratorSupportStatus.SUPPORTED
    assert support.supports_gpu_architecture("GFX942")
    assert support.supports_gpu_architecture("gfx950")
    assert "eager-generation" in support.validated_features
    with pytest.raises(FrozenInstanceError):
        support.status = AcceleratorSupportStatus.PREVIEW


def test_registered_alias_resolves_canonical_support() -> None:
    alias = get_model_accelerator_support("MossTTSDelay", "rocm")
    canonical = get_model_accelerator_support("MossTTSDelayModel", "rocm")

    assert alias == canonical
    assert alias is not None
    assert not alias.supports_gpu_architecture("gfx942:sramecc+:xnack-")


def test_unknown_model_or_accelerator_has_no_declaration() -> None:
    assert get_model_accelerator_support("Unknown", "rocm") is None
    assert (
        get_model_accelerator_support("MiniMaxMusic3ForConditionalGeneration", "cuda")
        is None
    )
