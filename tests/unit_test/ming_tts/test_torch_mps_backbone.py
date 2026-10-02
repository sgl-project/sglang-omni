# SPDX-License-Identifier: Apache-2.0
"""SGLang backbone and KV pool regression tests on Apple Metal."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace

import pytest
import torch

if not torch.backends.mps.is_available():
    pytest.skip("Requires an accessible Apple Metal device", allow_module_level=True)


@pytest.fixture(scope="module")
def runtime(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    from sglang.srt.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.layers.dp_attention import initialize_dp_attention
    from sglang.srt.layers.moe.utils import initialize_moe_config
    from sglang.srt.runtime_context import get_context

    assert not torch.distributed.is_initialized(), "Requires an isolated SGLang runtime"
    rendezvous = tmp_path_factory.mktemp("ming-gloo") / "rendezvous"
    with get_context().override_server_args(
        device="mps",
        attention_backend="torch_native",
        prefill_attention_backend="torch_native",
        decode_attention_backend="torch_native",
        sampling_backend="pytorch",
        disable_cuda_graph=True,
        disable_overlap_schedule=True,
        disable_radix_cache=True,
        chunked_prefill_size=-1,
        page_size=1,
        enable_memory_saver=False,
        max_running_requests=1,
        context_length=64,
        max_total_tokens=64,
        skip_tokenizer_init=True,
        quantization=None,
    ) as args:
        try:
            init_distributed_environment(
                world_size=1,
                rank=0,
                local_rank=0,
                backend="gloo",
                distributed_init_method=rendezvous.as_uri(),
                timeout=30,
            )
            initialize_model_parallel(backend="gloo")
            initialize_dp_attention(
                args,
                SimpleNamespace(
                    hf_config=SimpleNamespace(),
                    hidden_size=16,
                    dtype=torch.float32,
                ),
            )
            initialize_moe_config()
            yield
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()


@pytest.fixture(params=[torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def dtype(request: pytest.FixtureRequest) -> torch.dtype:
    return request.param


@pytest.fixture
def pools(runtime: None, dtype: torch.dtype) -> SimpleNamespace:
    from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
    from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool

    requests = ReqToTokenPool(3, 64, "mps", enable_memory_saver=False)
    kv = MHATokenToKVPool(
        size=18,
        page_size=1,
        dtype=dtype,
        head_num=1,
        head_dim=8,
        layer_num=2,
        device="mps",
        enable_memory_saver=False,
        enable_alt_stream=False,
        enable_kv_cache_copy=False,
    )
    allocator = TokenToKVPoolAllocator(
        size=18,
        dtype=dtype,
        device="mps",
        kvcache=kv,
        need_sort=False,
    )
    backend = TorchNativeAttnBackend(
        SimpleNamespace(
            device="mps",
            req_to_token_pool=requests,
            token_to_kv_pool=kv,
        )
    )
    return SimpleNamespace(
        requests=requests, kv=kv, allocator=allocator, backend=backend
    )


def assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.device.type == "mps" and torch.isfinite(actual).all()
    tolerance = 2e-2 if actual.dtype == torch.bfloat16 else 3e-5
    torch.testing.assert_close(
        actual.float().cpu(),
        expected.float().cpu(),
        atol=tolerance,
        rtol=tolerance,
    )


@pytest.mark.parametrize(
    "use_embeddings", [False, True], ids=["token_ids", "feedback_embeds"]
)
@torch.inference_mode()
def test_backbone_cached_decode_and_isolation(
    pools: SimpleNamespace,
    dtype: torch.dtype,
    use_embeddings: bool,
) -> None:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    from sglang_omni.model_runner.sglang_execution import attn_forward_context
    from sglang_omni.models.ming_tts.sglang_model import MingBailingMoeTextModel

    torch.manual_seed(2315)
    config = SimpleNamespace(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        moe_intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_experts=4,
        num_experts_per_tok=2,
        num_shared_experts=1,
        first_k_dense_replace=1,
        multi_gate=True,
        norm_topk_prob=True,
        rms_norm_eps=1e-5,
        rope_theta=600000.0,
        max_position_embeddings=64,
        runtime_rope_scaling={
            "type": "default",
            "rope_type": "default",
            "mrope_section": [1, 1, 2],
        },
    )
    model = MingBailingMoeTextModel(config).eval()
    for name, parameter in model.named_parameters():
        parameter.copy_(torch.randn_like(parameter) * 0.08)
        if "norm" in name and name.endswith("weight"):
            parameter.add_(1.0)
    model.to(device="mps", dtype=dtype)
    ids = torch.tensor([1, 4, 5, 6, 7, 8], device="mps")
    embeds = (
        torch.randn(6, 16).to(device="mps", dtype=dtype) if use_embeddings else None
    )
    positions = torch.arange(6, device="mps").expand(3, -1).clone()
    positions[1] += 5
    positions[2] += 11
    rows = pools.requests.alloc_rows(3)
    slots = pools.allocator.alloc(18)
    assert rows is not None and slots is not None
    # Each request's tokens occupy non-contiguous physical slots.
    full_locs, cached_locs, other_locs = slots.reshape(6, 3).T
    for row, locations in zip(rows, (full_locs, cached_locs, other_locs)):
        pools.requests.write((row, slice(0, 6)), locations.int())

    def run(
        row: int,
        locations: torch.Tensor,
        start: int,
        end: int,
        *,
        other: bool = False,
    ) -> torch.Tensor:
        decode = start > 0
        batch = ForwardBatch(
            forward_mode=ForwardMode.DECODE if decode else ForwardMode.EXTEND,
            batch_size=1,
            input_ids=(ids[start:end] + int(other)) % 32,
            req_pool_indices=torch.tensor([row], device="mps"),
            seq_lens=torch.tensor([end], dtype=torch.int32, device="mps"),
            out_cache_loc=locations[start:end],
            seq_lens_sum=end,
            seq_lens_cpu=torch.tensor([end], dtype=torch.int32),
            positions=positions[0, start:end],
            mrope_positions=positions[:, start:end],
            extend_num_tokens=end - start,
            extend_seq_lens=torch.tensor(
                [end - start], dtype=torch.int32, device="mps"
            ),
            extend_prefix_lens=torch.tensor([start], dtype=torch.int32, device="mps"),
            extend_start_loc=torch.tensor([0], dtype=torch.int32, device="mps"),
            extend_seq_lens_cpu=[end - start],
            extend_prefix_lens_cpu=[start],
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        pools.backend.init_forward_metadata(batch)
        inputs = None if embeds is None else embeds[start:end] + (0.3 if other else 0.0)
        with attn_forward_context(pools.backend):
            return model(
                batch.input_ids,
                batch.mrope_positions,
                batch,
                input_embeds=inputs,
            ).clone()

    expected = run(rows[0], full_locs, 0, 6)
    chunks = [run(rows[1], cached_locs, 0, 3)]
    other_expected = run(rows[2], other_locs, 0, 6, other=True)
    for step in range(3, 6):
        chunks.append(run(rows[1], cached_locs, step, step + 1))
    assert_close(torch.cat(chunks), expected)

    # Reuse the first request's storage without clearing stale KV entries.
    actual = [run(rows[0], full_locs, 0, 3, other=True)]
    actual.extend(run(rows[0], full_locs, i, i + 1, other=True) for i in range(3, 6))
    assert_close(torch.cat(actual), other_expected)
