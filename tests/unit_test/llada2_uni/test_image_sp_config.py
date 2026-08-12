# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import inspect

import pytest

from sglang_omni.models.llada2_uni import config, stages


def test_llada2_exposes_image_decoder_sequence_parallel_policy() -> None:
    policy = config.LLaDA2UniOmniPipelineConfig.sequence_parallel_policy(
        stage_name=config.IMAGE_DECODE_STAGE
    )

    assert policy is not None
    assert policy.attention_heads == 30
    assert policy.requires_power_of_two is True
    assert (
        config.LLaDA2UniOmniPipelineConfig.sequence_parallel_policy(
            stage_name=config.THINKER_STAGE
        )
        is None
    )


@pytest.mark.parametrize(
    ("sp_size", "ulysses_degree", "ring_degree"),
    [(2, 2, 1), (4, 2, 2)],
)
def test_image_decoder_accepts_supported_sp_decompositions(
    sp_size: int,
    ulysses_degree: int,
    ring_degree: int,
) -> None:
    settings = config.resolve_image_decoder_runtime_settings(
        backend=None,
        attention_backend=None,
        sp_size=sp_size,
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
    )

    assert settings.backend == "sglang"
    assert settings.attention_backend == "fa"


def test_image_decoder_rejects_diffusers_backend_for_sp() -> None:
    with pytest.raises(ValueError, match="requires backend='sglang'"):
        config.resolve_image_decoder_runtime_settings(
            backend="diffusers",
            attention_backend=None,
            sp_size=2,
            ulysses_degree=2,
            ring_degree=1,
        )


def test_image_decoder_rejects_head_incompatible_ulysses_degree() -> None:
    with pytest.raises(ValueError, match="30 attention heads"):
        config.resolve_image_decoder_runtime_settings(
            backend=None,
            attention_backend=None,
            sp_size=4,
            ulysses_degree=4,
            ring_degree=1,
        )


def test_image_decoder_factory_accepts_generic_sp_runtime_metadata() -> None:
    parameters = inspect.signature(stages.create_image_decode_executor).parameters

    assert {
        "stage_role",
        "sp_rank",
        "sp_size",
        "nccl_port",
        "ulysses_degree",
        "ring_degree",
        "checkpoint_load_device",
    } <= parameters.keys()
