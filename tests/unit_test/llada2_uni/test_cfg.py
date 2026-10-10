# SPDX-License-Identifier: Apache-2.0
"""Behavior tests for LLaDA2-Uni classifier-free guidance."""

from __future__ import annotations

from array import array
from types import SimpleNamespace as NS
from unittest.mock import create_autospec

import pytest
import torch
from flashinfer.prefill import (
    BatchPrefillWithPagedKVCacheWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
)
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.dllm.mixin.req import DllmReqPhase
from sglang.srt.layers.attention.flashinfer_backend import (
    FlashInferIndicesUpdaterPrefill,
)
from sglang.srt.layers.radix_attention import AttentionType, RadixAttention
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.kv_index_translator import KVIndexTranslator
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.sampling.sampling_params import SamplingParams

import sglang_omni.models.llada2_uni.cfg_attention_backend as cfg_attention_backend
from sglang_omni.models.llada2_uni.low_confidence_cfg import LowConfidenceCFG
from sglang_omni.models.llada2_uni.request_builders import LLaDA2UniRequest
from sglang_omni.scheduling.dllm_scheduler import DllmScheduler


class RaggedWrapperStub:
    is_cuda_graph_enabled = False

    def __init__(self) -> None:
        self._custom_mask_buf = None  # noqa: leading-underscore  # FlashInfer API
        self._mask_indptr_buf = None  # noqa: leading-underscore  # FlashInfer API
        self.plan = None

    def begin_forward(self, *args, **kwargs) -> None:
        self.plan = (args, kwargs)


def make_config(*, fdfo: bool = False) -> DllmConfig:
    return DllmConfig(
        algorithm="LowConfidenceCFG",
        algorithm_config={"threshold": 1.0, "image_token_offset": 3},
        block_size=4,
        mask_id=9,
        max_running_requests=3,
        first_done_first_out_mode=fdfo,
    )


def make_cfg_group(size: int) -> list[LLaDA2UniRequest]:
    requests = []
    for index in range(size):
        request = LLaDA2UniRequest(
            rid=f"cond-u{index}" if index else "cond",
            origin_input_text="",
            origin_input_ids=array("q", [1, 2, 3, 4]),
            sampling_params=SamplingParams(max_new_tokens=4, temperature=0.0),
            dllm_config=make_config(),
        )
        request.dllm_phase = DllmReqPhase.INCOMING_DECODE
        request._is_uncond = index > 0  # noqa: leading-underscore  # DLLM protocol
        request._is_uncond_img = index == 2  # noqa: leading-underscore  # DLLM protocol
        requests.append(request)
    if size > 1:
        for request in requests:
            request._cfg_group_rid = "cond"  # noqa: leading-underscore  # DLLM protocol
    return requests


@pytest.mark.parametrize("size", [1, 2, 3])
@pytest.mark.parametrize("rescale", [0.0, 1.0])
def test_guidance_updates_all_cfg_branches(size: int, rescale: float) -> None:
    algorithm = LowConfidenceCFG(make_config())
    algorithm.threshold = 0.94 if size == 3 else 0.8
    requests = make_cfg_group(size)
    requests[0]._task_kind = (
        "edit" if size == 3 else "t2i"
    )  # noqa: leading-underscore  # DLLM protocol
    requests[0]._dllm_steps = 3  # noqa: leading-underscore  # DLLM protocol
    requests[0]._cfg_scale = 2.0  # noqa: leading-underscore  # DLLM protocol
    requests[0]._cfg_image_scale = 1.5  # noqa: leading-underscore  # DLLM protocol
    requests[0]._cfg_rescale = rescale  # noqa: leading-underscore  # DLLM protocol

    input_ids = torch.tensor([1, 9, 9, 9] * size)
    if size > 1:
        input_ids[4] = 9
    initial_ids = input_ids.view(size, 4).clone()
    observed_inputs: list[torch.Tensor] = []

    def forward(batch: ForwardBatch, pp_proxy_tensors: None = None) -> NS:
        observed_inputs.append(batch.input_ids.view(size, 4).clone())
        conditional_logits = torch.zeros(4, 6)
        conditional_logits[:, 3] = torch.tensor([0.0, 1.5, 2.0, 3.0])
        if batch.input_ids[3] != 9:
            conditional_logits[1, 3] = 0.0
            conditional_logits[1, 4] = 1.5
        branch_scales = torch.tensor([1.0, 0.25, -0.25])[:size]
        return NS(
            logits_output=NS(
                full_logits=(
                    conditional_logits[None] * branch_scales[:, None, None]
                ).reshape(-1, 6)
            ),
            can_run_graph=False,
        )

    result = algorithm.run(
        NS(forward=forward),
        NS(input_ids=input_ids, batch_size=size, reqs=requests),
    )

    assert result[2:] == (None, None, False)
    assert len(result[1]) == size
    if size == 1 or rescale:
        partial_ids = initial_ids.clone()
        partial_ids[:, 2:] = 3
        final_ids = initial_ids.clone()
        final_ids[:, 1:] = torch.tensor([4, 3, 3])
        expected_inputs = [initial_ids, partial_ids, final_ids]
    else:
        final_ids = initial_ids.clone()
        final_ids[:, 1:] = 3
        expected_inputs = [initial_ids, final_ids]
    assert len(observed_inputs) == len(expected_inputs)
    for actual, expected in zip(observed_inputs, expected_inputs, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual, expected in zip(result[1], final_ids[:, 1:], strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_cfg_prefill_only_builds_kv_state() -> None:
    algorithm = LowConfidenceCFG(make_config())
    requests = make_cfg_group(2)
    for request in requests:
        request.dllm_phase = DllmReqPhase.INCOMING_PREFILL
    input_ids = torch.tensor([1, 2, 3, 4, 9, 9, 3, 4])
    original = input_ids.clone()
    calls = 0

    def forward(batch: ForwardBatch, pp_proxy_tensors: None = None) -> NS:
        nonlocal calls
        calls += 1
        return NS(logits_output=None, can_run_graph=False)

    result = algorithm.run(
        NS(forward=forward),
        NS(reqs=requests, batch_size=2, input_ids=input_ids),
    )

    assert result == (None, [], None, None, False)
    assert calls == 1
    torch.testing.assert_close(input_ids, original)


def test_invalid_cfg_batch_is_rejected() -> None:
    with pytest.raises(ValueError, match="FDFO"):
        LowConfidenceCFG(make_config(fdfo=True))

    algorithm = LowConfidenceCFG(make_config())
    requests = make_cfg_group(2)
    requests[1]._cfg_group_rid = "other"  # noqa: leading-underscore  # DLLM protocol
    with pytest.raises(RuntimeError, match="Malformed CFG"):
        algorithm.run(None, NS(reqs=requests, batch_size=2))


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_non_finite_cfg_scale_is_rejected(value: float) -> None:
    algorithm = LowConfidenceCFG(make_config())
    requests = make_cfg_group(2)
    requests[0]._cfg_scale = value  # noqa: leading-underscore  # DLLM protocol
    with pytest.raises(ValueError, match="_cfg_scale must be finite"):
        algorithm.run(None, NS(reqs=requests, batch_size=2))


def test_scheduler_applies_left_padding_metadata() -> None:
    scheduler = object.__new__(DllmScheduler)
    requests = make_cfg_group(2)
    requests[1]._dllm_left_pad_len = 6  # noqa: leading-underscore  # DLLM protocol
    forward_batch = NS(
        positions=torch.arange(4, 8).repeat(2),
        extend_seq_lens_cpu=[4, 4],
        forward_mode=NS(is_extend=lambda: True),
    )

    scheduler.apply_cfg_padding_metadata(forward_batch, NS(reqs=requests))

    assert forward_batch.dllm_left_pad_lens_cpu == [0, 6]
    assert forward_batch.positions[4:].tolist() == [0, 0, 0, 1]


def test_attention_masks_local_and_cached_padding() -> None:
    backend = object.__new__(cfg_attention_backend.LLaDA2CFGFlashInferAttnBackend)
    backend.cfg_prefill_wrapper_ragged = RaggedWrapperStub()
    backend.prefill_wrappers_paged = [object()]
    backend.prefill_split_tile_size = None
    begin_forward = create_autospec(
        FlashInferIndicesUpdaterPrefill, instance=True
    ).call_begin_forward
    backend.indices_updater_prefill = NS(
        call_begin_forward=begin_forward,
        kv_indptr=[None],
        qo_indptr=[None],
        num_qo_heads=1,
        num_kv_heads=1,
        head_dim=2,
        q_data_type=torch.float32,
        data_type=torch.float32,
        prefill_wrapper_ragged=RaggedWrapperStub(),
    )
    batch = NS(
        dllm_left_pad_lens_cpu=[0, 6],
        extend_prefix_lens_cpu=[4, 4],
        extend_seq_lens_cpu=[4, 4],
        forward_mode=NS(is_dllm_extend=lambda: True),
        seq_lens=torch.tensor([8, 8]),
        extend_prefix_lens=torch.tensor([4, 4]),
        req_pool_indices=torch.tensor([0, 1]),
    )

    backend.init_forward_metadata(batch)

    args = begin_forward.call_args.args
    assert args[3].tolist() == [4, 0]
    assert args[7].tolist() == [0, 4]
    mask = backend.cfg_prefill_wrapper_ragged.plan[1]["custom_mask"].reshape(2, 4, 4)
    assert mask[0].all()
    assert not mask[1, 2:, :2].any()
    assert mask[1, 0, 0] and mask[1, 1, 1]
    assert mask[1, :, 2:].all()

    batch.extend_prefix_lens_cpu = [8, 8]
    batch.seq_lens += 4
    batch.extend_prefix_lens += 4
    backend.init_forward_metadata(batch)
    assert not backend.cfg_local_left_pad_active
    assert begin_forward.call_args.args[3].tolist() == [8, 2]


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlashInfer requires CUDA")
@pytest.mark.parametrize("initial_prefix", [0, 4])
def test_attention_matches_dense_across_padding_boundary(initial_prefix: int) -> None:
    device, dtype = "cuda", torch.bfloat16
    heads, head_dim, block_size = 2, 64, 4
    pool = MHATokenToKVPool(64, 1, dtype, heads, head_dim, 1, device, False)
    req_pool = ReqToTokenPool(2, 16, device, False)
    req_pool.req_to_token[:2] = torch.arange(1, 33, device=device).view(2, 16)
    allocator = TokenToKVPoolAllocator(64, dtype, device, pool, False)
    workspace = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=device)
    backend = object.__new__(cfg_attention_backend.LLaDA2CFGFlashInferAttnBackend)
    backend.token_to_kv_pool = pool
    backend.num_wrappers = 1
    backend.is_dllm_model = True
    backend.prefill_uses_dequant_workspace = False
    backend.kv_cache_quant_method = NS(needs_global_scale=lambda: False)
    backend.prefill_split_tile_size = None
    backend.cfg_prefill_wrapper_ragged = BatchPrefillWithRaggedKVCacheWrapper(
        workspace, "NHD", backend="fa2"
    )
    backend.prefill_wrapper_ragged = BatchPrefillWithRaggedKVCacheWrapper(
        workspace, "NHD", backend="fa2"
    )
    backend.prefill_wrappers_paged = [
        BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD", backend="fa2")
    ]
    backend.kv_index_translator = KVIndexTranslator(
        req_to_token=req_pool.req_to_token,
        token_to_kv_pool_allocator=allocator,
        token_to_kv_pool=pool,
        page_size=1,
        device=device,
    )
    updater = object.__new__(FlashInferIndicesUpdaterPrefill)
    updater.attn_backend = backend
    updater.num_qo_heads = updater.num_kv_heads = heads
    updater.head_dim = head_dim
    updater.q_data_type = updater.data_type = dtype
    updater.kv_indptr = [torch.zeros(3, dtype=torch.int32, device=device)]
    updater.qo_indptr = [torch.zeros(3, dtype=torch.int32, device=device)]
    updater.kv_last_page_len = torch.ones(2, dtype=torch.int32, device=device)
    updater.prefill_wrapper_ragged = backend.prefill_wrapper_ragged
    backend.indices_updater_prefill = updater
    layer = RadixAttention(
        heads, head_dim, head_dim**-0.5, heads, 0, attn_type=AttentionType.ENCODER_ONLY
    )
    generator = torch.Generator(device=device).manual_seed(42)
    keys, values = torch.randn(
        2, 2, 16, heads, head_dim, dtype=dtype, device=device, generator=generator
    )
    cached_keys, cached_values = pool.get_kv_buffer(0)
    slots = req_pool.req_to_token[:2].long()
    cached_keys[slots[:, :initial_prefix]] = keys[:, :initial_prefix]
    cached_values[slots[:, :initial_prefix]] = values[:, :initial_prefix]
    left_padding = (0, initial_prefix + 2)

    for prefix in (initial_prefix, initial_prefix + block_size):
        batch = NS(
            dllm_left_pad_lens_cpu=left_padding,
            extend_prefix_lens_cpu=[prefix, prefix],
            extend_seq_lens_cpu=[block_size, block_size],
            forward_mode=ForwardMode.DLLM_EXTEND,
            seq_lens=torch.full(
                (2,), prefix + block_size, dtype=torch.int32, device=device
            ),
            extend_prefix_lens=torch.full(
                (2,), prefix, dtype=torch.int32, device=device
            ),
            req_pool_indices=torch.arange(2, dtype=torch.int32, device=device),
            out_cache_loc=slots[:, prefix : prefix + block_size].flatten(),
        )
        queries = torch.randn(
            2,
            block_size,
            heads,
            head_dim,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        current_keys = keys[:, prefix : prefix + block_size].contiguous()
        current_values = values[:, prefix : prefix + block_size].contiguous()
        backend.init_forward_metadata(batch)
        actual = backend.forward_extend(
            queries.flatten(0, 1),
            current_keys.flatten(0, 1),
            current_values.flatten(0, 1),
            layer,
            batch,
        ).view_as(queries)
        for row, pad in enumerate(left_padding):
            mask = torch.arange(prefix + block_size, device=device)[None, :] >= pad
            mask = mask.expand(block_size, -1).clone()
            for query in range(max(pad - prefix, 0)):
                mask[query, prefix + query] = True
            scores = (
                torch.einsum(
                    "qhd,khd->hqk",
                    queries[row].float(),
                    keys[row, : prefix + block_size].float(),
                )
                * layer.scaling
            )
            probabilities = scores.masked_fill(~mask, -torch.inf).softmax(-1)
            expected = torch.einsum(
                "hqk,khd->qhd",
                probabilities,
                values[row, : prefix + block_size].float(),
            ).to(dtype)
            torch.testing.assert_close(actual[row], expected, atol=0.016, rtol=0.016)


@pytest.mark.parametrize("cached", [False, True])
def test_attention_writes_kv_for_masked_extend(
    monkeypatch: pytest.MonkeyPatch, cached: bool
) -> None:
    backend = object.__new__(cfg_attention_backend.LLaDA2CFGFlashInferAttnBackend)
    backend.cfg_local_left_pad_active = True
    backend.cfg_has_cached_prefix = cached
    backend.forward_metadata = NS(swa_out_cache_loc=None)
    current = torch.ones(4, 1, 2)
    prefix = torch.full_like(current, 2)
    backend.cfg_prefill_wrapper_ragged = NS(
        forward=lambda *_args, **_kwargs: current,
        forward_return_lse=lambda *_args, **_kwargs: (current, torch.zeros(4, 1)),
    )
    backend.prefill_wrappers_paged = [
        NS(
            forward_return_lse=lambda *_args, **_kwargs: (
                prefix,
                torch.zeros(4, 1),
            )
        )
    ]
    monkeypatch.setattr(
        cfg_attention_backend,
        "merge_state",
        lambda output, output_lse, cached_output, _cached_lse: (
            (output + cached_output) / 2,
            output_lse,
        ),
    )
    writes = []
    backend.token_to_kv_pool = NS(
        get_kv_buffer=lambda _layer_id: object(),
        set_kv_buffer=lambda *args: writes.append(args),
    )
    backend._kv_write_scales = lambda _layer: (
        None,
        None,
    )  # noqa: leading-underscore  # SGLang API
    layer = NS(
        tp_q_head_num=1,
        tp_k_head_num=1,
        tp_v_head_num=1,
        head_dim=2,
        scaling=1.0,
        logit_cap=0.0,
        layer_id=0,
        k_scale_float=None,
        v_scale_float=None,
        is_cross_attention=False,
    )
    batch = NS(out_cache_loc=torch.arange(4))

    result = backend.forward_extend(current, current, current, layer, batch)

    expected = current if not cached else (current + prefix) / 2
    torch.testing.assert_close(result, expected.view(4, 2))
    assert len(writes) == 1
    assert torch.equal(writes[0][1].loc, batch.out_cache_loc)
