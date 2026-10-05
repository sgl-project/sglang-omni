# SPDX-License-Identifier: Apache-2.0
"""The predictor kernels: the exact add-norm against the plain path; the fused layer
against the plain path, an fp32 reference and its own buffer and cache contracts."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.runtime_context import get_context
from torch import nn

from sglang_omni.models.qwen3_omni.components.talker import Qwen3OmniTalker
from sglang_omni.platforms import current_platform
from sglang_omni.platforms.cuda import CUDAOmniPlatform
from sglang_omni.platforms.rocm import ROCMOmniPlatform
from sglang_omni.utils import predictor_layers
from sglang_omni.utils.predictor_layers import (
    HIDDEN_SIZE,
    MAX_FUSED_ROWS,
    add_rmsnorm_rounded,
    resolve_fused_predictor_layers,
    resolve_predictor_layer_shape,
    split_count,
    supports_exact_add_rmsnorm,
)
from tests.unit_test.fixtures.qwen_predictor import TupleLinear

HIDDEN = 1024
HEAD_DIM = 128
NUM_HEADS = 16
NUM_KV_HEADS = 8
INTERMEDIATE = 3072
NUM_LAYERS = 5
NUM_CODE_GROUPS = 16
PREDICTOR_LEN = NUM_CODE_GROUPS + 1
MAX_BS = 64
EPS = 1e-6
DTYPE = torch.bfloat16
accelerator = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="the fused layer runs on CUDA only"
)


class PlainLinear(TupleLinear):
    """A tuple linear that looks unquantized and unpartitioned to the resolver."""

    quant_method = UnquantizedLinearMethod()
    bias = None
    tp_size = 1


class SwiGLU(nn.Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.gate_up_proj = PlainLinear(HIDDEN, 2 * INTERMEDIATE).to(device, DTYPE)
        self.down_proj = PlainLinear(INTERMEDIATE, HIDDEN).to(device, DTYPE)
        self.act_fn = SiluAndMul()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(hidden_states)
        hidden_states, _ = self.down_proj(self.act_fn(gate_up))
        return hidden_states


def norm_with_random_scale(device: torch.device, size: int) -> RMSNorm:
    norm = RMSNorm(size, eps=EPS).to(device, DTYPE)
    with torch.no_grad():
        norm.weight.normal_(1.0, 0.1)
    return norm


def build_layer(device: torch.device) -> SimpleNamespace:
    attention = SimpleNamespace(
        hidden_size=HIDDEN,
        q_size=NUM_HEADS * HEAD_DIM,
        kv_size=NUM_KV_HEADS * HEAD_DIM,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        q_norm=norm_with_random_scale(device, HEAD_DIM),
        k_norm=norm_with_random_scale(device, HEAD_DIM),
        alt_stream=None,
        qkv_proj=PlainLinear(HIDDEN, (NUM_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM).to(
            device, DTYPE
        ),
        o_proj=PlainLinear(NUM_HEADS * HEAD_DIM, HIDDEN).to(device, DTYPE),
        rotary_emb=RotaryEmbedding(HEAD_DIM, HEAD_DIM, 64, 10000, True, DTYPE).to(
            device
        ),
    )
    return SimpleNamespace(
        self_attn=attention,
        mlp=SwiGLU(device),
        input_layernorm=norm_with_random_scale(device, HIDDEN),
        post_attention_layernorm=norm_with_random_scale(device, HIDDEN),
    )


def build_talker(
    device: torch.device, seed: int, max_bs: int = MAX_BS
) -> Qwen3OmniTalker:
    """A talker whose predictor layers are real-shape modules with seeded weights."""
    torch.manual_seed(seed)
    talker = object.__new__(Qwen3OmniTalker)
    talker.code_predictor = SimpleNamespace(
        model=SimpleNamespace(
            layers=[build_layer(device) for _ in range(NUM_LAYERS)],
            norm=norm_with_random_scale(device, HIDDEN),
        )
    )
    positions = torch.arange(PREDICTOR_LEN, device=device, dtype=torch.long)
    talker.predictor_positions = positions
    talker.predictor_position_rows = (
        positions[:, None].expand(PREDICTOR_LEN, max_bs).contiguous()
    )
    talker.predictor_pair_positions = positions[:2].repeat(max_bs)
    talker.predictor_k_cache = torch.zeros(
        NUM_LAYERS,
        max_bs,
        PREDICTOR_LEN,
        NUM_KV_HEADS,
        HEAD_DIM,
        device=device,
        dtype=DTYPE,
    )
    talker.predictor_v_cache = torch.zeros_like(talker.predictor_k_cache)
    talker.predictor_k_rows = [
        layer.view(max_bs * PREDICTOR_LEN, -1) for layer in talker.predictor_k_cache
    ]
    talker.predictor_v_rows = [
        layer.view(max_bs * PREDICTOR_LEN, -1) for layer in talker.predictor_v_cache
    ]
    talker.predictor_cache_slots = (
        torch.arange(max_bs, device=device, dtype=torch.long)[None, :] * PREDICTOR_LEN
        + positions[:, None]
    ).contiguous()
    talker.predictor_pair_cache_slots = (
        talker.predictor_cache_slots[:2, :].t().reshape(-1).contiguous()
    )
    talker.predictor_rope_stores_kv = False
    talker.predictor_exact_add_norm = True
    talker.predictor_fused_layers = None
    return talker


def fuse(talker: Qwen3OmniTalker) -> Qwen3OmniTalker:
    """The fused path built as the model loader builds it, under a bf16 default dtype:
    the split scratch must come out fp32 and int32 regardless."""
    device = talker.predictor_k_cache.device
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        fused = resolve_fused_predictor_layers(
            talker.code_predictor,
            PREDICTOR_LEN,
            talker.predictor_k_cache.shape[1],
            device,
            DTYPE,
        )
    finally:
        torch.set_default_dtype(default_dtype)
    assert fused is not None
    assert fused.partials.dtype == torch.float32
    assert fused.sum_sq_partials.dtype == torch.float32
    assert fused.counters.dtype == torch.int32
    talker.predictor_fused_layers = fused
    return talker


def reset_caches(talker: Qwen3OmniTalker) -> None:
    talker.predictor_k_cache.zero_()
    talker.predictor_v_cache.zero_()


@pytest.fixture
def published_server_args() -> Iterator[None]:
    with get_context().override_server_args(
        cuda_graph_config=CudaGraphConfig(
            prefill=PhaseConfig(backend=Backend.DISABLED)
        ),
    ):
        yield


def rmsnorm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * weight.float()


def rotate(x: torch.Tensor, cos_sin: torch.Tensor) -> torch.Tensor:
    half = HEAD_DIM // 2
    cos, sin = cos_sin[..., None, :half], cos_sin[..., None, half:]
    first, second = x[..., :half], x[..., half:]
    return torch.cat((first * cos - second * sin, first * sin + second * cos), -1)


class Reference:
    """The predictor layers in fp32 from the same bf16 weights, with fp32 caches."""

    def __init__(self, talker: Qwen3OmniTalker, batch_size: int) -> None:
        self.layers = talker.code_predictor.model.layers
        self.final_norm = talker.code_predictor.model.norm
        self.k_cache = torch.zeros(
            NUM_LAYERS,
            batch_size,
            NUM_KV_HEADS,
            PREDICTOR_LEN,
            HEAD_DIM,
            device=talker.predictor_k_cache.device,
        )
        self.v_cache = torch.zeros_like(self.k_cache)

    def forward(self, token_embeds: torch.Tensor, cache_len: int) -> torch.Tensor:
        batch_size, seq_len, _ = token_embeds.shape
        end = cache_len + seq_len
        x = token_embeds.float()
        positions = torch.arange(cache_len, end, device=x.device)
        for layer_idx, layer in enumerate(self.layers):
            attention = layer.self_attn
            cos_sin = attention.rotary_emb.cos_sin_cache[positions].float()
            normed = rmsnorm(x, layer.input_layernorm.weight)
            qkv = normed @ attention.qkv_proj.weight.float().t()
            q, k, v = qkv.split(
                [attention.q_size, attention.kv_size, attention.kv_size], -1
            )
            q = q.reshape(batch_size, seq_len, NUM_HEADS, HEAD_DIM)
            k = k.reshape(batch_size, seq_len, NUM_KV_HEADS, HEAD_DIM)
            v = v.reshape(batch_size, seq_len, NUM_KV_HEADS, HEAD_DIM)
            q = rotate(rmsnorm(q, attention.q_norm.weight), cos_sin)
            k = rotate(rmsnorm(k, attention.k_norm.weight), cos_sin)
            self.k_cache[layer_idx, :, :, cache_len:end] = k.transpose(1, 2)
            self.v_cache[layer_idx, :, :, cache_len:end] = v.transpose(1, 2)
            attended = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2),
                self.k_cache[layer_idx, :, :, :end],
                self.v_cache[layer_idx, :, :, :end],
                is_causal=seq_len > 1,
                enable_gqa=True,
            )
            attended = attended.transpose(1, 2).reshape(batch_size, seq_len, -1)
            x = x + attended @ attention.o_proj.weight.float().t()
            normed = rmsnorm(x, layer.post_attention_layernorm.weight)
            gate, up = (normed @ layer.mlp.gate_up_proj.weight.float().t()).chunk(2, -1)
            x = (
                x
                + (torch.nn.functional.silu(gate) * up)
                @ layer.mlp.down_proj.weight.float().t()
            )
        return rmsnorm(x, self.final_norm.weight)


def predictor_inputs(
    device: torch.device, batch_size: int, seed: int
) -> list[torch.Tensor]:
    """The opening pair then one token per codebook, as the predictor feeds them."""
    generator = torch.Generator(device=device).manual_seed(seed)
    steps = [torch.randn(batch_size, 2, HIDDEN, device=device, generator=generator)]
    steps += [
        torch.randn(batch_size, 1, HIDDEN, device=device, generator=generator)
        for _ in range(NUM_CODE_GROUPS - 2)
    ]
    return [step.to(DTYPE) for step in steps]


def run_sequence(
    talker: Qwen3OmniTalker, steps: list[torch.Tensor]
) -> list[torch.Tensor]:
    reset_caches(talker)
    outputs = []
    cache_len = 0
    with torch.no_grad():
        for step in steps:
            # note (ratish): the plain path's residual chain writes the first sum
            # into the tokens it was given, as the predictor's scratch allows.
            outputs.append(
                talker.predictor_forward_tokens(
                    token_embeds=step.clone(),
                    batch_size=step.shape[0],
                    cache_len=cache_len,
                ).clone()
            )
            cache_len += step.shape[1]
    return outputs


def relative_error(actual: torch.Tensor, reference: torch.Tensor) -> float:
    return ((actual.float() - reference).norm() / reference.norm()).item()


@accelerator
@pytest.mark.accelerator
@pytest.mark.usefixtures("published_server_args")
@pytest.mark.parametrize("batch_size", [1, 3, 12, 32, 64])
def test_fused_layer_tracks_the_reference_like_the_plain_path(batch_size: int) -> None:
    """Over a whole predictor sequence the fused path's distance to fp32 stays within
    a quarter of the plain path's at every step, in the output and in the caches: the
    accumulation order and the moved norm scale may not cost precision."""
    device = torch.device("cuda")
    talker = build_talker(device, seed=3)
    steps = predictor_inputs(device, batch_size, seed=4)
    plain = run_sequence(talker, steps)
    plain_k = talker.predictor_k_cache[:, :batch_size].transpose(2, 3).clone()
    fused = run_sequence(fuse(talker), steps)
    fused_k = talker.predictor_k_cache[:, :batch_size].transpose(2, 3).clone()
    reference = Reference(talker, batch_size)
    cache_len = 0
    with torch.no_grad():
        for step, plain_out, fused_out in zip(steps, plain, fused):
            expected = reference.forward(step, cache_len)
            plain_error = relative_error(plain_out, expected)
            fused_error = relative_error(fused_out, expected)
            assert fused_error <= 1.25 * plain_error + 1e-4, (
                cache_len,
                plain_error,
                fused_error,
            )
            cache_len += step.shape[1]
    assert relative_error(fused_k, reference.k_cache) <= 1.25 * relative_error(
        plain_k, reference.k_cache
    )


@accelerator
@pytest.mark.accelerator
@pytest.mark.usefixtures("published_server_args")
def test_fused_layer_is_deterministic_and_batch_invariant() -> None:
    device = torch.device("cuda")
    talker = fuse(build_talker(device, seed=5))
    steps = predictor_inputs(device, 12, seed=6)
    first = run_sequence(talker, steps)
    k_first = talker.predictor_k_cache.clone()
    second = run_sequence(talker, steps)
    assert all(torch.equal(a, b) for a, b in zip(first, second))
    assert torch.equal(k_first, talker.predictor_k_cache)
    alone = run_sequence(talker, [step[:1] for step in steps])
    assert all(torch.equal(a[:1], b) for a, b in zip(first, alone))
    assert torch.equal(k_first[:, :1], talker.predictor_k_cache[:, :1])


@accelerator
@pytest.mark.accelerator
@pytest.mark.usefixtures("published_server_args")
def test_a_pass_runs_fused_up_to_the_fused_rows_and_plain_above() -> None:
    """A pass above MAX_FUSED_ROWS rows, the opening pair or a single token, must equal
    the plain path bit for bit; a pass of exactly MAX_FUSED_ROWS rows runs fused."""
    device = torch.device("cuda")
    batch_size = MAX_FUSED_ROWS + 1
    plain_talker = build_talker(device, seed=21, max_bs=batch_size)
    fused_talker = fuse(build_talker(device, seed=21, max_bs=batch_size))
    steps = predictor_inputs(device, batch_size, seed=22)
    plain = run_sequence(plain_talker, steps)
    fused = run_sequence(fused_talker, steps)
    assert all(torch.equal(a, b) for a, b in zip(plain, fused))
    assert torch.equal(plain_talker.predictor_k_cache, fused_talker.predictor_k_cache)
    assert torch.equal(plain_talker.predictor_v_cache, fused_talker.predictor_v_cache)
    residual = fused_talker.predictor_fused_layers.residual
    residual.fill_(float("nan"))
    run_sequence(fused_talker, [steps[0][: MAX_FUSED_ROWS // 2]])
    assert not torch.any(torch.isnan(residual))


@accelerator
@pytest.mark.accelerator
@pytest.mark.usefixtures("published_server_args")
def test_fused_opening_pair_matches_two_single_token_passes() -> None:
    """The second token of a pair, fed as a strided view, must equal a single pass at
    cache length 1: the residual add once read the view with the wrong row stride."""
    device = torch.device("cuda")
    talker = fuse(build_talker(device, seed=7))
    tokens = predictor_inputs(device, 3, seed=8)[0]
    paired = run_sequence(talker, [tokens])[0]
    paired_k = talker.predictor_k_cache[:, :3, :2].clone()
    reset_caches(talker)
    parent = tokens.clone()
    first, second = parent[:, 0:1], parent[:, 1:2]
    assert second.stride(0) == 2 * HIDDEN
    with torch.no_grad():
        talker.predictor_forward_tokens(token_embeds=first, batch_size=3, cache_len=0)
        single = talker.predictor_forward_tokens(
            token_embeds=second, batch_size=3, cache_len=1
        )
    torch.testing.assert_close(paired[:, 1:2], single)
    assert torch.equal(paired_k, talker.predictor_k_cache[:, :3, :2])


@accelerator
@pytest.mark.accelerator
@pytest.mark.usefixtures("published_server_args")
def test_fused_layer_replays_from_a_cuda_graph() -> None:
    device = torch.device("cuda")
    talker = fuse(build_talker(device, seed=9))
    step = predictor_inputs(device, 4, seed=10)[1]
    eager = run_sequence(talker, [step])[0]
    reset_caches(talker)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.no_grad():
        talker.predictor_forward_tokens(token_embeds=step, batch_size=4, cache_len=0)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    reset_caches(talker)
    with torch.cuda.graph(graph), torch.no_grad():
        replayed = talker.predictor_forward_tokens(
            token_embeds=step, batch_size=4, cache_len=0
        )
    reset_caches(talker)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(replayed, eager)
    fresh = predictor_inputs(device, 4, seed=11)[1]
    step.copy_(fresh)
    reset_caches(talker)
    graph.replay()
    torch.cuda.synchronize()
    replayed_fresh = replayed.clone()
    replayed_k = talker.predictor_k_cache[:, :4, :1].clone()
    eager_fresh = run_sequence(talker, [fresh])[0]
    assert torch.equal(replayed_fresh, eager_fresh)
    assert torch.equal(replayed_k, talker.predictor_k_cache[:, :4, :1])


@accelerator
@pytest.mark.accelerator
def test_resolver_keeps_the_plain_path_for_a_quantized_projection() -> None:
    device = torch.device("cuda")
    talker = build_talker(device, seed=12)
    talker.code_predictor.model.layers[2].mlp.down_proj.quant_method = SimpleNamespace()
    assert (
        resolve_predictor_layer_shape(talker.code_predictor, PREDICTOR_LEN, device)
        is None
    )


@accelerator
@pytest.mark.accelerator
@pytest.mark.usefixtures("published_server_args")
def test_split_launches_leave_the_tile_counters_at_zero() -> None:
    """The last program of a tile resets its counter; a missed reset would make the
    next launch reduce with the wrong program and corrupt that launch, not this one."""
    device = torch.device("cuda")
    talker = fuse(build_talker(device, seed=13))
    steps = predictor_inputs(device, 12, seed=14)
    run_sequence(talker, steps)
    counters = talker.predictor_fused_layers.counters
    assert torch.equal(counters, torch.zeros_like(counters))


@accelerator
@pytest.mark.accelerator
@pytest.mark.usefixtures("published_server_args")
def test_rows_beyond_the_batch_are_not_written() -> None:
    """The launches pad the batch to a tile of 16 rows; the padded rows must not reach
    the q, residual or activation buffers, which later, larger batches read."""
    device = torch.device("cuda")
    talker = fuse(build_talker(device, seed=15))
    fused = talker.predictor_fused_layers
    for buffer in (fused.q, fused.residual, fused.activated):
        buffer.fill_(7.0)
    step = predictor_inputs(device, 3, seed=16)[1]
    run_sequence(talker, [step])
    for buffer in (fused.q, fused.residual, fused.activated):
        assert torch.all(buffer[3:] == 7.0)
        assert not torch.all(buffer[:3] == 7.0)


@accelerator
@pytest.mark.accelerator
@pytest.mark.usefixtures("published_server_args")
def test_forward_writes_only_the_positions_slot_of_every_layer() -> None:
    """A token at cache length 5 lands in slot 5 of its row in every layer's K and V
    cache and nowhere else: the slot is the position, the row group is the batch row."""
    device = torch.device("cuda")
    talker = fuse(build_talker(device, seed=17))
    talker.predictor_k_cache.fill_(7.0)
    talker.predictor_v_cache.fill_(7.0)
    step = predictor_inputs(device, 4, seed=18)[1]
    with torch.no_grad():
        talker.predictor_forward_tokens(token_embeds=step, batch_size=4, cache_len=5)
    for cache in (talker.predictor_k_cache, talker.predictor_v_cache):
        written = cache[:, :4, 5]
        assert not torch.any(written == 7.0)
        untouched = cache.clone()
        untouched[:, :4, 5] = 7.0
        assert torch.all(untouched == 7.0)


@accelerator
@pytest.mark.accelerator
def test_resolver_keeps_the_plain_path_for_a_partitioned_projection() -> None:
    """A projection sharded across ranks needs the all-reduce its linear layer does;
    the fused launches add the residual to the rank-local product, so they must not run.
    """
    device = torch.device("cuda")
    talker = build_talker(device, seed=19)
    talker.code_predictor.model.layers[1].mlp.down_proj.tp_size = 2
    assert (
        resolve_predictor_layer_shape(talker.code_predictor, PREDICTOR_LEN, device)
        is None
    )


def test_without_triton_the_predictor_keeps_the_plain_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(predictor_layers, "HAS_TRITON", False)
    cuda = torch.device("cuda")
    assert not supports_exact_add_rmsnorm(HIDDEN_SIZE, torch.bfloat16, cuda)
    assert (
        resolve_fused_predictor_layers(
            SimpleNamespace(), PREDICTOR_LEN, MAX_BS, cuda, DTYPE
        )
        is None
    )


@accelerator
@pytest.mark.accelerator
def test_deterministic_inference_keeps_the_plain_path() -> None:
    """Deterministic inference promises a request the same bits at every batch size; a
    pass above MAX_FUSED_ROWS runs plain, so the fused layers must stay off."""
    device = torch.device("cuda")
    talker = build_talker(device, seed=25)
    with get_context().override_server_args(enable_deterministic_inference=True):
        assert (
            resolve_fused_predictor_layers(
                talker.code_predictor, PREDICTOR_LEN, MAX_BS, device, DTYPE
            )
            is None
        )


def test_split_count_lands_the_program_count_nearest_the_sms() -> None:
    assert split_count(32, 16, 132) == 4
    assert split_count(32, 24, 132) == 4
    assert split_count(32, 24, 264) == 8
    assert split_count(32, 12, 132) == 4
    assert split_count(32, 6, 132) == 2
    assert split_count(128, 8, 132) == 1


@pytest.mark.accelerator
@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="the kernel matches the CUDA RMSNorm",
)
@pytest.mark.parametrize("rows", [1, 3, 12, 32])
def test_add_rmsnorm_rounded_matches_add_then_rmsnorm_bit_for_bit(rows: int) -> None:
    """Same bits as torch's bf16 add followed by the plain RMSNorm, over many seeds."""
    from sgl_kernel import rmsnorm

    device = torch.device("cuda")
    for seed in range(50):
        torch.manual_seed(seed)
        x = (torch.randn(rows, HIDDEN_SIZE, device=device) * 3).to(torch.bfloat16)
        residual = (torch.randn(rows, HIDDEN_SIZE, device=device) * 3).to(
            torch.bfloat16
        )
        weight = (1 + 0.1 * torch.randn(HIDDEN_SIZE, device=device)).to(torch.bfloat16)
        expected_sum = residual + x
        expected = rmsnorm(expected_sum, weight, 1e-6)
        normed, summed = add_rmsnorm_rounded(x.clone(), residual.clone(), weight, 1e-6)
        assert torch.equal(summed, expected_sum)
        assert torch.equal(normed, expected)


def test_exact_add_rmsnorm_applies_to_the_predictor_shape_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cuda = torch.device("cuda")
    monkeypatch.setattr(predictor_layers, "HAS_TRITON", True)
    monkeypatch.setattr(predictor_layers, "current_platform", ROCMOmniPlatform())
    assert not supports_exact_add_rmsnorm(HIDDEN_SIZE, torch.bfloat16, cuda)
    monkeypatch.setattr(predictor_layers, "current_platform", CUDAOmniPlatform())
    assert supports_exact_add_rmsnorm(HIDDEN_SIZE, torch.bfloat16, cuda)
    assert not supports_exact_add_rmsnorm(2048, torch.bfloat16, cuda)
    assert not supports_exact_add_rmsnorm(HIDDEN_SIZE, torch.float16, cuda)
    assert not supports_exact_add_rmsnorm(
        HIDDEN_SIZE, torch.bfloat16, torch.device("cpu")
    )
