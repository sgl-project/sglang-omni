# SPDX-License-Identifier: Apache-2.0
"""The two channels that feed SGLang's server args have to merge per key.

A stage can reach ``server_args_overrides`` through the free-form
``factory.server_args_overrides`` block or through the typed ``engine.*`` group.
Writing both used to drop the free-form block whenever the stage carried any
engine value, silently discarding the backend the author asked for.
"""

from __future__ import annotations

from sglang_omni.config import EngineArgs, EngineStageConfig, FactoryArgs
from sglang_omni.config.runtime import resolve_stage_typed_kwargs


def stage_with(engine: EngineArgs | None, **factory_extra: object) -> EngineStageConfig:
    return EngineStageConfig(
        name="asr",
        factory_path="sglang_omni.models.qwen3_asr.stages.create_sglang_qwen3_asr_executor",
        factory=FactoryArgs(device="musa", **factory_extra),
        engine=engine,
    )


def test_factory_block_survives_a_populated_engine_group() -> None:
    factory_overrides = {"attention_backend": "torch_native"}
    kwargs = resolve_stage_typed_kwargs(
        stage_with(
            EngineArgs(max_running_requests=1, mm_attention_backend="sdpa"),
            server_args_overrides=factory_overrides,
        )
    )
    assert kwargs["server_args_overrides"] == {
        "attention_backend": "torch_native",
        "max_running_requests": 1,
        "mm_attention_backend": "sdpa",
    }
    assert factory_overrides == {"attention_backend": "torch_native"}


def test_engine_group_wins_when_both_set_the_same_key() -> None:
    kwargs = resolve_stage_typed_kwargs(
        stage_with(
            EngineArgs(mm_attention_backend="sdpa"),
            server_args_overrides={
                "mm_attention_backend": "fa3",
                "attention_backend": "torch_native",
            },
        )
    )
    assert kwargs["server_args_overrides"] == {
        "attention_backend": "torch_native",
        "mm_attention_backend": "sdpa",
    }


def test_factory_block_passes_through_without_an_engine_group() -> None:
    kwargs = resolve_stage_typed_kwargs(
        stage_with(None, server_args_overrides={"dtype": "bfloat16"})
    )
    assert kwargs["server_args_overrides"] == {"dtype": "bfloat16"}
