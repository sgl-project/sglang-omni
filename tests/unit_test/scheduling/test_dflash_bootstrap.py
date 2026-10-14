# SPDX-License-Identifier: Apache-2.0
"""Request budgets use initialized context, position and KV capacities."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, create_autospec

import pytest
import torch
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
from sglang.srt.utils.hf_transformers import get_config

from sglang_omni.models.qwen3_omni.thinker_model_runner import (
    Qwen3OmniThinkerModelRunner,
)
from sglang_omni.scheduling import dflash_bootstrap
from sglang_omni.scheduling.sglang_backend import cache


@pytest.fixture
def initialized_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[SimpleNamespace, Mock, Mock, SimpleNamespace, SimpleNamespace]:
    configuration = SimpleNamespace(
        tp_size=1,
        pp_size=1,
        enable_streaming_session=False,
        speculative_draft_window_size=None,
        enable_hierarchical_cache=False,
        enable_lmcache=False,
        enable_unified_cache_external_linker=False,
        disaggregation_mode="null",
        enable_pd_role_switch=False,
        enable_two_batch_overlap=False,
        enable_lora=False,
        speculative_accept_threshold_single=1.0,
        speculative_accept_threshold_acc=1.0,
    )
    request_pool = SimpleNamespace(
        max_context_len=32776,
        req_to_token=torch.empty((17, 32776), device="meta", dtype=torch.int32),
    )
    token_allocator = SimpleNamespace(size=84228)
    target_runner = Mock(preloaded_weights_bytes=10)
    target_runner.sliding_window_size = None
    target_runner.model_config = SimpleNamespace(context_len=32768)
    target_runner.req_to_token_pool = request_pool
    target_runner.token_to_kv_pool_allocator = token_allocator
    target_runner.token_to_kv_pool = SimpleNamespace(
        size=84228, post_capture_active=False
    )
    target_runner.model.model.layers = [
        SimpleNamespace(
            self_attn=SimpleNamespace(
                rotary_emb=SimpleNamespace(
                    position_start=0,
                    cos_sin_cache=torch.empty((32768, 128), device="meta"),
                )
            )
        )
        for _ in range(48)
    ]
    target_worker = Mock(model_runner=target_runner, nccl_port=0)
    target_worker.get_memory_pool.return_value = request_pool, token_allocator
    draft_worker = Mock(preloaded_weights_bytes=5)
    draft_runner = draft_worker.draft_model_runner
    draft_runner.model_config = SimpleNamespace(context_len=32768)
    draft_runner.req_to_token_pool = request_pool
    draft_runner.token_to_kv_pool_allocator = token_allocator
    draft_runner.token_to_kv_pool = SimpleNamespace(size=84228)
    draft_worker.draft_model.layers = [
        SimpleNamespace(
            self_attn=SimpleNamespace(
                rotary_emb=SimpleNamespace(
                    position_start=0,
                    cos_sin_cache=torch.empty((32776, 128), device="meta"),
                )
            )
        )
        for _ in range(5)
    ]
    monkeypatch.setattr(
        dflash_bootstrap,
        "get_context",
        lambda: SimpleNamespace(is_config_namespace_published=lambda namespace: False),
    )
    monkeypatch.setattr(
        dflash_bootstrap, "get_schedule", lambda: SimpleNamespace(page_size=1)
    )
    monkeypatch.setattr(
        dflash_bootstrap, "ModelWorker", Mock(return_value=target_worker)
    )
    monkeypatch.setattr(
        dflash_bootstrap, "DFlashWorkerV2", Mock(return_value=draft_worker)
    )
    monkeypatch.setattr(dflash_bootstrap, "SpeculativeTargetWorker", Mock())
    monkeypatch.setattr(dflash_bootstrap, "consume_stage_kv_cache_bytes", lambda: None)
    monkeypatch.setattr(dflash_bootstrap, "init_sglang_cuda_graphs", Mock())
    monkeypatch.setattr(
        dflash_bootstrap, "refresh_deep_gemm_layout_memory_budget", Mock()
    )
    monkeypatch.setattr(
        dflash_bootstrap,
        "create_tree_cache",
        create_autospec(cache.create_tree_cache),
    )
    return configuration, target_worker, draft_worker, request_pool, token_allocator


def construct(
    initialized_workers: tuple[
        SimpleNamespace, Mock, Mock, SimpleNamespace, SimpleNamespace
    ],
) -> dflash_bootstrap.DFlashInfrastructure:
    configuration, _, _, _, _ = initialized_workers
    return dflash_bootstrap.create_dflash_infrastructure(
        configuration,
        0,
        tp_rank=0,
        nccl_port=None,
        total_gpu_memory_fraction=None,
        enable_prefill_input_embeds=False,
    )


def test_initialized_32768_supports_original_video_budget(
    initialized_workers: tuple[
        SimpleNamespace, Mock, Mock, SimpleNamespace, SimpleNamespace
    ],
) -> None:
    infrastructure = construct(initialized_workers)
    assert infrastructure.draft_context_limit_tokens == 32768
    assert infrastructure.target_position_limit_tokens == 32768
    for prompt_tokens in (14189, 14200, 14171, 14174):
        assert prompt_tokens + 256 + 8 - 1 <= infrastructure.draft_context_limit_tokens
    assert 32761 + 1 + 8 - 1 == infrastructure.draft_context_limit_tokens + 1
    _, target_worker, draft_worker, request_pool, token_allocator = initialized_workers
    assert (
        Qwen3OmniThinkerModelRunner.create_speculative_infrastructure
        is dflash_bootstrap.create_dflash_infrastructure
    )
    draft_worker.alloc_memory_pool.assert_called_once_with(
        memory_pool_config=target_worker.model_runner.memory_pool_config,
        req_to_token_pool=request_pool,
        token_to_kv_pool_allocator=token_allocator,
    )
    target_worker.model_runner.init_attention_backends.assert_called_once()
    draft_worker.init_cuda_graphs.assert_called_once()


@pytest.mark.parametrize("sliding_window_size", [None, 4096])
def test_cache_factory_receives_initialized_runner_window(
    initialized_workers: tuple[
        SimpleNamespace, Mock, Mock, SimpleNamespace, SimpleNamespace
    ],
    monkeypatch: pytest.MonkeyPatch,
    sliding_window_size: int | None,
) -> None:
    _, target_worker, _, request_pool, token_allocator = initialized_workers
    target_worker.model_runner.sliding_window_size = sliding_window_size
    monkeypatch.setattr(dflash_bootstrap, "create_tree_cache", cache.create_tree_cache)
    monkeypatch.setattr(
        cache,
        "get_memory",
        lambda: SimpleNamespace(
            disable_radix_cache=True,
            radix_eviction_policy="lru",
            radix_eviction_policy_config=None,
        ),
    )
    monkeypatch.setattr(
        cache, "get_schedule", lambda: SimpleNamespace(chunked_prefill_size=2048)
    )
    monkeypatch.setattr(
        cache, "get_serving", lambda: SimpleNamespace(enable_streaming_session=False)
    )
    chunk_cache = create_autospec(cache.ChunkCache)
    monkeypatch.setattr(cache, "ChunkCache", chunk_cache)

    infrastructure = construct(initialized_workers)

    chunk_cache.assert_called_once()
    parameters = chunk_cache.call_args.args[0]
    assert parameters.req_to_token_pool is request_pool
    assert parameters.token_to_kv_pool_allocator is token_allocator
    assert parameters.page_size == 1
    assert parameters.sliding_window_size == sliding_window_size
    assert infrastructure.tree_cache is chunk_cache.return_value


@pytest.mark.parametrize(
    "capacity_owner",
    [
        "target_context",
        "draft_context",
        "mapping_declared",
        "mapping_actual",
        "allocator",
        "target_kv",
        "draft_kv",
        "target_rotary",
        "draft_rotary",
    ],
)
def test_budget_obeys_each_initialized_capacity(
    initialized_workers: tuple[
        SimpleNamespace, Mock, Mock, SimpleNamespace, SimpleNamespace
    ],
    capacity_owner: str,
) -> None:
    _, target_worker, draft_worker, request_pool, token_allocator = initialized_workers
    target_runner = target_worker.model_runner
    draft_runner = draft_worker.draft_model_runner
    if capacity_owner == "target_context":
        target_runner.model_config.context_len = 12000
    elif capacity_owner == "draft_context":
        draft_runner.model_config.context_len = 12000
    elif capacity_owner == "mapping_declared":
        request_pool.max_context_len = 12000
    elif capacity_owner == "mapping_actual":
        request_pool.req_to_token = torch.empty((17, 12000), device="meta")
    elif capacity_owner == "allocator":
        token_allocator.size = 12000
    elif capacity_owner == "target_kv":
        target_runner.token_to_kv_pool.size = 12000
    elif capacity_owner == "draft_kv":
        draft_runner.token_to_kv_pool.size = 12000
    elif capacity_owner == "target_rotary":
        target_runner.model.model.layers[31].self_attn.rotary_emb.cos_sin_cache = (
            torch.empty((12000, 128), device="meta")
        )
    else:
        draft_worker.draft_model.layers[4].self_attn.rotary_emb.cos_sin_cache = (
            torch.empty((12000, 128), device="meta")
        )
    assert construct(initialized_workers).draft_context_limit_tokens == 12000


@pytest.mark.parametrize("invalid_capacity", [0, -1, None, True, 32768.0])
def test_invalid_effective_context_rejected(
    initialized_workers: tuple[
        SimpleNamespace, Mock, Mock, SimpleNamespace, SimpleNamespace
    ],
    invalid_capacity: int | float | bool | None,
) -> None:
    initialized_workers[1].model_runner.model_config.context_len = invalid_capacity
    with pytest.raises(RuntimeError, match="positive initialized context"):
        construct(initialized_workers)


@pytest.mark.parametrize(
    "owner_name",
    ["target_request", "draft_request", "target_allocator", "draft_allocator"],
)
def test_separate_pool_owner_rejected(
    initialized_workers: tuple[
        SimpleNamespace, Mock, Mock, SimpleNamespace, SimpleNamespace
    ],
    owner_name: str,
) -> None:
    _, target_worker, draft_worker, _, _ = initialized_workers
    owner = (
        target_worker.model_runner
        if owner_name.startswith("target")
        else draft_worker.draft_model_runner
    )
    if owner_name.endswith("request"):
        owner.req_to_token_pool = SimpleNamespace()
    else:
        owner.token_to_kv_pool_allocator = SimpleNamespace()
    with pytest.raises(RuntimeError, match="shared request and KV"):
        construct(initialized_workers)


def test_nonzero_rotary_origin_rejected(
    initialized_workers: tuple[
        SimpleNamespace, Mock, Mock, SimpleNamespace, SimpleNamespace
    ],
) -> None:
    initialized_workers[2].draft_model.layers[2].self_attn.rotary_emb.position_start = 1
    with pytest.raises(ValueError, match="zero-based rotary"):
        construct(initialized_workers)


def test_empty_layer_collection_rejected(
    initialized_workers: tuple[
        SimpleNamespace, Mock, Mock, SimpleNamespace, SimpleNamespace
    ],
) -> None:
    initialized_workers[2].draft_model.layers = []
    with pytest.raises(RuntimeError, match="initialized attention layers"):
        construct(initialized_workers)


def test_post_capture_final_capacity_is_used(
    initialized_workers: tuple[
        SimpleNamespace, Mock, Mock, SimpleNamespace, SimpleNamespace
    ],
) -> None:
    target_runner = initialized_workers[1].model_runner
    target_runner.token_to_kv_pool.post_capture_active = True

    def resize(*, draft_runners: tuple[Mock]) -> None:
        target_runner.token_to_kv_pool.size = 16000
        draft_runners[0].token_to_kv_pool.size = 16000

    target_runner.post_capture_resize_kv_pool.side_effect = resize
    assert construct(initialized_workers).draft_context_limit_tokens == 16000


def test_actual_native_override_and_standard_rotary_capacity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    configuration_path = tmp_path / "config.json"
    configuration_path.write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "architectures": ["DFlashDraftModel"],
                "max_position_embeddings": 8192,
                "rope_theta": 1000000.0,
                "rope_scaling": None,
            }
        )
    )
    configuration = get_config(str(tmp_path), trust_remote_code=False)
    owner = SimpleNamespace(
        is_draft_model=True, hf_text_config=configuration, hf_config=configuration
    )
    monkeypatch.setenv("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", "1")
    ModelConfig._derive_context_length(
        owner, 32768
    )  # noqa: leading-underscore  # Actual native configuration contract.
    assert owner.context_len == owner.hf_text_config.max_position_embeddings == 32768
    assert json.loads(configuration_path.read_text())["max_position_embeddings"] == 8192
    rotary_embedding = RotaryEmbedding(128, 128, 8192, 1000000.0, True, torch.bfloat16)
    original_rows = rotary_embedding.cos_sin_cache.clone()
    rotary_embedding._ensure_cos_sin_cache_length(
        32775
    )  # noqa: leading-underscore  # Actual native cache extension contract.
    assert rotary_embedding.cos_sin_cache.shape[0] >= 32776
    assert torch.equal(rotary_embedding.cos_sin_cache[:8192], original_rows)
    assert torch.isfinite(rotary_embedding.cos_sin_cache[14463:14471]).all()
