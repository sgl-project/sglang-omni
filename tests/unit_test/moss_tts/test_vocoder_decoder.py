# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import sglang_omni.models.moss_tts.vocoder_decoder as vocoder_decoder
import sglang_omni.models.moss_tts.vocoder_kernels as vocoder_kernels
from sglang_omni.models.moss_tts.vocoder_decoder import (
    MossAudioTokenizerAttention,
    MossAudioTokenizerProjectedTransformer,
    MossAudioTokenizerTransformerLayer,
    MossAudioTokenizerVocoderDecoder,
)


def create_sin_embedding(
    positions: torch.Tensor,
    dim: int,
    *,
    max_period: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    del max_period
    return torch.zeros(*positions.shape, dim, device=positions.device, dtype=dtype)


class _FakeLayerScale(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x


class _FakeAttention(nn.Module):
    def __init__(self, hidden_size: int, *, num_heads: int = 2) -> None:
        super().__init__()
        self.embed_dim = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // self.num_heads
        self.causal = True
        self.context = 4
        self.rope = None
        self.attention_implementation = "sdpa"
        self.in_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def resolve_attention_implementation(
        self, _: torch.Tensor, *, is_streaming: bool = False
    ) -> str:
        return "sdpa"

    def _get_backend_check_dtype(self, x: torch.Tensor) -> torch.dtype:
        return x.dtype

    def forward(self, x: torch.Tensor, **_: object) -> torch.Tensor:
        return x


class _ReferenceAttention(_FakeAttention):
    def forward(self, x: torch.Tensor, *, input_lengths: torch.Tensor) -> torch.Tensor:
        batch_size, max_seqlen, _ = x.shape
        projected = self.in_proj(x).reshape(
            batch_size, max_seqlen, 3, self.num_heads, self.head_dim
        )
        q, k, v = projected.permute(2, 0, 3, 1, 4)
        positions = torch.arange(max_seqlen, device=x.device, dtype=torch.long)
        valid_k = positions.view(1, 1, max_seqlen) < input_lengths.view(-1, 1, 1)
        delta = positions.view(1, max_seqlen, 1) - positions.view(1, 1, max_seqlen)
        attn_bias = torch.ones(
            (1, max_seqlen, max_seqlen), device=x.device, dtype=torch.bool
        )
        if self.causal:
            attn_bias = attn_bias & (delta >= 0)
        if self.context is not None:
            attn_bias = attn_bias & (delta < int(self.context))
        attn_bias = (attn_bias & valid_k)[:, None, :, :]
        out = F.scaled_dot_product_attention(q, k, v, attn_bias, dropout_p=0.0)
        valid_q = positions.view(1, max_seqlen) < input_lengths.view(-1, 1)
        out = torch.where(
            valid_q.view(batch_size, 1, max_seqlen, 1),
            out,
            torch.zeros((), device=x.device, dtype=x.dtype),
        )
        out = out.transpose(1, 2).reshape(batch_size, max_seqlen, self.embed_dim)
        return self.out_proj(out)


class _FakeLayer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.self_attn = _FakeAttention(hidden_size)
        self.layer_scale_1 = _FakeLayerScale(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.layer_scale_2 = _FakeLayerScale(hidden_size)


class _FallbackTransformer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_FakeLayer(hidden_size)])
        self.positional_embedding = "rope"
        self.positional_scale = 1.0
        self.max_period = 10000.0

    def resolve_attention_implementation(self, _: torch.Tensor) -> str:
        return "sdpa"


class _FallbackProjectedStage(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_proj = nn.Linear(3, 6)
        self.transformer = _FallbackTransformer(6)
        self.output_proj = nn.Linear(6, 7)
        self.is_streaming = False
        self.module_type = "Transformer"
        self.seen_input_shape: tuple[int, ...] | None = None

    def forward(
        self,
        x: torch.Tensor,
        input_lengths: torch.Tensor,
        **_: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.seen_input_shape = tuple(x.shape)
        return x + 10, input_lengths + 1


class _PatchStage(nn.Module):
    def __init__(self, *, patch_size: int = 2, is_downsample: bool = False) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.downsample_ratio = patch_size
        self.is_downsample = is_downsample
        self.module_type = "PatchedPretransform"

    def forward(
        self,
        x: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return x, input_lengths


class _CountingLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(in_features, out_features)
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return super().forward(x)


class _CountingLayerNorm(nn.LayerNorm):
    def __init__(self, hidden_size: int) -> None:
        super().__init__(hidden_size)
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return super().forward(x)


class _CountingLayerScale(_FakeLayerScale):
    def __init__(self, hidden_size: int) -> None:
        super().__init__(hidden_size)
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return super().forward(x)


class _CountingLayer(_FakeLayer):
    def __init__(self, hidden_size: int) -> None:
        nn.Module.__init__(self)
        self.norm1 = _CountingLayerNorm(hidden_size)
        self.self_attn = _FakeAttention(hidden_size)
        self.layer_scale_1 = _CountingLayerScale(hidden_size)
        self.norm2 = _CountingLayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            _CountingLinear(hidden_size, hidden_size * 2),
            nn.GELU(),
            _CountingLinear(hidden_size * 2, hidden_size),
        )
        self.layer_scale_2 = _CountingLayerScale(hidden_size)


class _LegacyAttention(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.embed_dim = hidden_size
        self.num_heads = 2
        self.causal = True
        self.context = 4
        self.rope = None
        self.in_projs = nn.ModuleList(
            [nn.Linear(hidden_size, 3 * hidden_size, bias=False)]
        )
        self.out_projs = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size, bias=False)]
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        del key, value
        return query


class _LegacyLayer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.self_attn = _LegacyAttention(hidden_size)
        self.layer_scale_1 = _FakeLayerScale(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.linear1 = nn.Linear(hidden_size, hidden_size * 2)
        self.linear2 = nn.Linear(hidden_size * 2, hidden_size)
        self.activation = F.gelu
        self.layer_scale_2 = _FakeLayerScale(hidden_size)


class _LegacyTransformer(_FallbackTransformer):
    def __init__(self, hidden_size: int) -> None:
        nn.Module.__init__(self)
        self.layers = nn.ModuleList([_LegacyLayer(hidden_size)])
        self.positional_embedding = "rope"
        self.positional_scale = 1.0
        self.max_period = 10000.0


class _LegacyProjectedStage(_FallbackProjectedStage):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.input_proj = nn.Linear(3, 6)
        self.transformer = _LegacyTransformer(6)
        self.output_proj = nn.Linear(6, 7)
        self.is_streaming = False
        self.module_type = "Transformer"
        self.seen_input_shape = None


def test_projected_transformer_sdpa_path_does_not_reenter_source_stage() -> None:
    source = _FallbackProjectedStage()
    wrapper = MossAudioTokenizerProjectedTransformer(source)
    x = torch.randn(2, 3, 4)
    lengths = torch.tensor([4, 3])

    out, out_lengths = wrapper(x, lengths)

    assert source.seen_input_shape is None
    assert out.shape == (2, 7, 4)
    assert torch.equal(out_lengths, lengths)


def test_projected_transformer_shares_packed_rope_cache_across_layers() -> None:
    source = _FallbackProjectedStage()
    source.transformer.layers.append(_FakeLayer(6))

    wrapper = MossAudioTokenizerProjectedTransformer(source)
    caches = [
        layer.self_attn._packed_rope_cache for layer in wrapper.transformer.layers
    ]

    assert len(caches) == 2
    assert caches[0] is caches[1]


def test_projected_transformer_uses_sglang_packed_flash_path() -> None:
    source = _FallbackProjectedStage()
    source.transformer.layers[0].self_attn.attention_implementation = (
        "flash_attention_2"
    )
    source.transformer.layers[0].self_attn.context = None
    wrapper = MossAudioTokenizerProjectedTransformer(source)
    attn = wrapper.transformer.layers[0].self_attn
    attn._can_run_packed_flash = lambda _: True  # type: ignore[method-assign]
    calls = []

    def fake_flash_attn(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_q: torch.Tensor,
        cu_k: torch.Tensor,
        max_q: int,
        max_k: int,
        *,
        causal: bool,
        window_size: tuple[int, int],
    ) -> torch.Tensor:
        calls.append((cu_q.clone(), cu_k.clone(), max_q, max_k, window_size))
        return q

    attn._flash_attn_varlen = fake_flash_attn
    x = torch.randn(2, 3, 4)
    lengths = torch.tensor([4, 3])

    out, out_lengths = wrapper(x, lengths)

    assert source.seen_input_shape is None
    assert len(calls) == 1
    cu_q, cu_k, max_q, max_k, window_size = calls[0]
    assert cu_q.tolist() == [0, 4, 7]
    assert cu_k.tolist() == [0, 4, 7]
    assert max_q == 4
    assert max_k == 4
    assert window_size == (-1, -1)
    assert out.shape == (2, 7, 4)
    assert torch.equal(out_lengths, lengths)


def test_projected_transformer_uses_host_lengths_without_tensor_max(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _FallbackProjectedStage()
    source.transformer.layers[0].self_attn.attention_implementation = (
        "flash_attention_2"
    )
    source.transformer.layers[0].self_attn.context = None
    wrapper = MossAudioTokenizerProjectedTransformer(source)
    attention = wrapper.transformer.layers[0].self_attn
    attention._can_run_packed_flash = lambda _: True  # type: ignore[method-assign]
    attention._flash_attn_varlen = lambda q, *_, **__: q

    def fail_max(*_: object, **__: object) -> None:
        raise AssertionError("host lengths must avoid reading the length tensor")

    monkeypatch.setattr(torch.Tensor, "max", fail_max)
    x = torch.randn(1, 3, 4)
    lengths = torch.tensor([4])

    out, out_lengths = wrapper(x, lengths, input_lengths_cpu=[4])

    assert out.shape == (1, 7, 4)
    assert out_lengths is lengths


def test_packed_flash_unavailable_uses_source_attention(monkeypatch) -> None:
    monkeypatch.setattr(vocoder_decoder, "flash_attn_varlen_func", None)
    source = _FallbackProjectedStage()
    source.transformer.layers[0].self_attn.attention_implementation = (
        "flash_attention_2"
    )
    source.transformer.layers[0].self_attn.context = None
    wrapper = MossAudioTokenizerProjectedTransformer(source)
    x = torch.randn(2, 3, 4)
    lengths = torch.tensor([4, 3])

    out, out_lengths = wrapper(x, lengths)

    assert wrapper.transformer.layers[0].self_attn._flash_attn_varlen is None
    assert source.seen_input_shape is None
    assert out.shape == (2, 7, 4)
    assert torch.equal(out_lengths, lengths)


def test_local_causal_flash_plan_chunks_queries_and_overlaps_keys() -> None:
    cu_seqlens = torch.tensor([0, 320, 577], dtype=torch.int32)

    plan = vocoder_decoder._build_local_causal_flash_plan(
        cu_seqlens,
        context=125,
    )

    assert plan.cu_seqlens_q.tolist() == [0, 128, 256, 320, 448, 576, 577]
    assert plan.cu_seqlens_k.tolist() == [0, 128, 380, 568, 696, 948, 1073]
    expected_kv_indices = torch.cat(
        [
            torch.arange(0, 128),
            torch.arange(4, 256),
            torch.arange(132, 320),
            torch.arange(320, 448),
            torch.arange(324, 576),
            torch.arange(452, 577),
        ]
    )
    assert torch.equal(plan.kv_indices, expected_kv_indices)
    assert plan.max_seqlen_q == 128
    assert plan.max_seqlen_k == 252
    assert plan.context == 125


def test_local_causal_flash_plan_skips_identity_kv_gather() -> None:
    cu_seqlens = torch.tensor([0, 80, 180], dtype=torch.int32)

    plan = vocoder_decoder._build_local_causal_flash_plan(
        cu_seqlens,
        context=125,
    )

    assert plan.kv_indices is None


def test_local_causal_attention_keeps_packed_flash_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    source = _FakeAttention(hidden_size=6).to(device="cuda", dtype=torch.bfloat16)
    source.attention_implementation = "flash_attention_2"
    source.causal = True
    source.context = 4
    attn = MossAudioTokenizerAttention(source)
    attn._flash_attn_varlen = lambda *_, **__: torch.empty(0, device="cuda")
    x = torch.empty(1, 6, device="cuda", dtype=torch.bfloat16)

    assert attn._can_run_packed_flash(x)


def test_projected_transformer_skips_flash_for_zero_valid_length(monkeypatch) -> None:
    source = _FallbackProjectedStage()
    source.transformer.layers[0].self_attn.attention_implementation = (
        "flash_attention_2"
    )
    source.transformer.layers[0].self_attn.context = None
    wrapper = MossAudioTokenizerProjectedTransformer(source)
    attn = wrapper.transformer.layers[0].self_attn
    attn._can_run_packed_flash = lambda _: True  # type: ignore[method-assign]

    def fail_flash(*_: object, **__: object) -> None:
        raise AssertionError("zero-length input must not call flash attention")

    def fail_pack(*_: object) -> None:
        raise AssertionError("zero-length input must not pack padded frames")

    attn._flash_attn_varlen = fail_flash
    monkeypatch.setattr(vocoder_decoder, "_pack_padded_sequence", fail_pack)
    x = torch.randn(2, 3, 4)
    lengths = torch.tensor([0, 0])

    out, out_lengths = wrapper(x, lengths)

    assert out.shape == (2, 7, 4)
    assert torch.equal(out_lengths, lengths)


@pytest.mark.parametrize(
    ("context", "causal", "expected"),
    [
        (None, True, (-1, -1)),
        (None, False, (-1, -1)),
        (1, True, (0, 0)),
        (4, True, (3, 0)),
        (4, False, (-1, -1)),
    ],
)
def test_flash_window_size_matches_moss_local_mask(
    context: int | None, causal: bool, expected: tuple[int, int]
) -> None:
    source = _FakeAttention(hidden_size=6)
    source.context = context
    source.causal = causal
    attn = MossAudioTokenizerAttention(source)

    assert attn._flash_window_size() == expected


def test_flash_window_size_keeps_same_keys_as_moss_mask() -> None:
    context = 4
    seqlen = 7
    source = _FakeAttention(hidden_size=6)
    source.context = context
    attn = MossAudioTokenizerAttention(source)
    left_window, right_window = attn._flash_window_size()
    positions = torch.arange(seqlen)
    moss_mask = (positions.view(seqlen, 1) - positions.view(1, seqlen) >= 0) & (
        positions.view(seqlen, 1) - positions.view(1, seqlen) < context
    )
    flash_mask = torch.zeros_like(moss_mask)
    for query_position in range(seqlen):
        first_key = max(0, query_position - left_window)
        last_key = min(seqlen - 1, query_position + right_window)
        flash_mask[query_position, first_key : last_key + 1] = True

    assert torch.equal(flash_mask, moss_mask)


def test_projected_transformer_uses_single_unpadded_pack_fast_path(
    monkeypatch,
) -> None:
    source = _FallbackProjectedStage()
    source.transformer.layers[0].self_attn.attention_implementation = (
        "flash_attention_2"
    )
    source.transformer.layers[0].self_attn.context = None
    wrapper = MossAudioTokenizerProjectedTransformer(source)
    attn = wrapper.transformer.layers[0].self_attn
    attn._can_run_packed_flash = lambda _: True  # type: ignore[method-assign]
    calls = []

    def fail_masked_pack(_: torch.Tensor, __: torch.Tensor) -> None:
        raise AssertionError("single unpadded input should not use masked pack")

    def fake_flash_attn(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_q: torch.Tensor,
        cu_k: torch.Tensor,
        max_q: int,
        max_k: int,
        *,
        causal: bool,
        window_size: tuple[int, int],
    ) -> torch.Tensor:
        calls.append((q.shape, cu_q.clone(), cu_k.clone(), max_q, max_k))
        return q

    monkeypatch.setattr(vocoder_decoder, "_pack_padded_sequence", fail_masked_pack)
    attn._flash_attn_varlen = fake_flash_attn
    x = torch.randn(1, 3, 4)
    lengths = torch.tensor([4])

    out, out_lengths = wrapper(x, lengths)
    _ = wrapper(x, lengths)

    assert len(calls) == 2
    q_shape, cu_q, cu_k, max_q, max_k = calls[0]
    assert q_shape[0] == 4
    assert cu_q.tolist() == [0, 4]
    assert cu_k.tolist() == [0, 4]
    assert calls[1][1].tolist() == [0, 4]
    assert calls[1][2].tolist() == [0, 4]
    assert max_q == 4
    assert max_k == 4
    assert out.shape == (1, 7, 4)
    assert torch.equal(out_lengths, lengths)


def test_single_unpadded_pack_position_cache_grows_and_slices() -> None:
    cache = vocoder_decoder._PositionIdsCache()
    x_short = torch.randn(1, 4, 6)
    x_long = torch.randn(1, 6, 6)

    _, cu_short_1, pos_short_1 = vocoder_decoder._pack_unpadded_sequence(x_short, cache)
    _, cu_long, pos_long = vocoder_decoder._pack_unpadded_sequence(x_long, cache)
    _, cu_short_2, pos_short_2 = vocoder_decoder._pack_unpadded_sequence(x_short, cache)

    assert cu_short_1.tolist() == [0, 4]
    assert cu_long.tolist() == [0, 6]
    assert cu_short_2.tolist() == [0, 4]
    assert pos_short_1.tolist() == [0, 1, 2, 3]
    assert pos_long.tolist() == [0, 1, 2, 3, 4, 5]
    assert pos_short_2.tolist() == [0, 1, 2, 3]
    assert pos_short_2.data_ptr() == pos_long.data_ptr()


def test_host_length_pack_and_unpack_matches_masked_path() -> None:
    x = torch.randn(3, 5, 7)
    lengths = torch.tensor([5, 2, 4], dtype=torch.int32)
    expected_packed, valid_mask, expected_cu, expected_positions = (
        vocoder_decoder._pack_padded_sequence(x, lengths)
    )

    packed, flat_indices, cu_seqlens, position_ids = (
        vocoder_decoder._pack_padded_sequence_from_host_lengths(
            x,
            lengths,
            [5, 2, 4],
        )
    )
    expected_unpacked = vocoder_decoder._unpack_packed_sequence(
        expected_packed,
        valid_mask,
        batch_size=3,
        max_seqlen=5,
    )
    unpacked = vocoder_decoder._unpack_packed_sequence_from_indices(
        packed,
        flat_indices,
        batch_size=3,
        max_seqlen=5,
    )

    assert torch.equal(packed, expected_packed)
    assert torch.equal(cu_seqlens, expected_cu)
    assert torch.equal(position_ids, expected_positions)
    assert torch.equal(unpacked, expected_unpacked)


def test_projected_transformer_single_padded_input_uses_masked_pack() -> None:
    source = _FallbackProjectedStage()
    source.transformer.layers[0].self_attn.attention_implementation = (
        "flash_attention_2"
    )
    source.transformer.layers[0].self_attn.context = None
    wrapper = MossAudioTokenizerProjectedTransformer(source)
    attn = wrapper.transformer.layers[0].self_attn
    attn._can_run_packed_flash = lambda _: True  # type: ignore[method-assign]
    calls = []

    def fake_flash_attn(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_q: torch.Tensor,
        cu_k: torch.Tensor,
        max_q: int,
        max_k: int,
        *,
        causal: bool,
        window_size: tuple[int, int],
    ) -> torch.Tensor:
        calls.append((q.shape, cu_q.clone(), cu_k.clone(), max_q, max_k))
        return q

    attn._flash_attn_varlen = fake_flash_attn
    x = torch.randn(1, 3, 4)
    lengths = torch.tensor([2])

    out, out_lengths = wrapper(x, lengths)

    assert len(calls) == 1
    q_shape, cu_q, cu_k, max_q, max_k = calls[0]
    assert q_shape[0] == 2
    assert cu_q.tolist() == [0, 2]
    assert cu_k.tolist() == [0, 2]
    assert max_q == 2
    assert max_k == 2
    assert out.shape == (1, 7, 4)
    assert torch.equal(out_lengths, lengths)


def test_sglang_packed_flash_matches_sdpa_reference_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if vocoder_decoder.flash_attn_varlen_func is None:
        pytest.skip("requires SGLang flash_attn_varlen_func")

    torch.manual_seed(0)
    device = torch.device("cuda")
    source = _ReferenceAttention(hidden_size=128, num_heads=2).to(
        device=device, dtype=torch.bfloat16
    )
    source.attention_implementation = "flash_attention_2"
    source.context = None
    wrapper = MossAudioTokenizerAttention(source)
    x = torch.randn(2, 6, 128, device=device, dtype=torch.bfloat16)
    input_lengths = torch.tensor([6, 4], device=device)

    packed_x, valid_mask, cu_seqlens, position_ids = (
        vocoder_decoder._pack_padded_sequence(x, input_lengths)
    )
    packed_out = wrapper(
        packed_x,
        cu_seqlens=cu_seqlens,
        max_seqlen=6,
        position_ids=position_ids,
    )
    flash_out = vocoder_decoder._unpack_packed_sequence(
        packed_out, valid_mask, batch_size=2, max_seqlen=6
    )
    sdpa_out = source(x, input_lengths=input_lengths)

    torch.testing.assert_close(flash_out, sdpa_out, atol=4e-2, rtol=3e-2)


def test_sglang_chunked_local_flash_matches_sdpa_reference_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if vocoder_decoder.flash_attn_varlen_func is None:
        pytest.skip("requires SGLang flash_attn_varlen_func")

    torch.manual_seed(0)
    device = torch.device("cuda")
    source = _ReferenceAttention(hidden_size=128, num_heads=2).to(
        device=device, dtype=torch.bfloat16
    )
    source.attention_implementation = "flash_attention_2"
    source.context = 400
    wrapper = MossAudioTokenizerAttention(source)
    x = torch.randn(2, 1024, 128, device=device, dtype=torch.bfloat16)
    input_lengths = torch.tensor([1024, 701], device=device)

    packed_x, valid_mask, cu_seqlens, position_ids = (
        vocoder_decoder._pack_padded_sequence(x, input_lengths)
    )
    local_flash_plan = vocoder_decoder._build_local_causal_flash_plan(
        cu_seqlens,
        context=400,
    )
    packed_out = wrapper(
        packed_x,
        cu_seqlens=cu_seqlens,
        max_seqlen=1024,
        position_ids=position_ids,
        local_flash_plan=local_flash_plan,
    )
    flash_out = vocoder_decoder._unpack_packed_sequence(
        packed_out, valid_mask, batch_size=2, max_seqlen=1024
    )
    sdpa_out = source(x, input_lengths=input_lengths)

    torch.testing.assert_close(flash_out, sdpa_out, atol=4e-2, rtol=3e-2)


def test_cached_packed_rope_matches_moss_interleaved_reference() -> None:
    q = torch.randn(5, 2, 6)
    k = torch.randn(5, 2, 6)
    position_ids = torch.tensor([0, 1, 2, 3, 4])
    max_period = 10000.0
    cache = vocoder_decoder._MossPackedRopeCache(max_period=max_period)

    out_q, out_k = vocoder_decoder._apply_cached_packed_rope(
        q,
        k,
        position_ids,
        max_positions=5,
        cache=cache,
    )

    half_dim = q.shape[-1] // 2
    ds = torch.arange(half_dim, dtype=torch.float32)
    freqs = torch.exp(ds * (-math.log(max_period) * 2 / q.shape[-1]))
    phase = position_ids.float().view(-1, 1, 1) * freqs.view(1, 1, -1)
    cos = torch.cos(phase)
    sin = torch.sin(phase)
    q_pair = q.view(*q.shape[:-1], half_dim, 2)
    k_pair = k.view(*k.shape[:-1], half_dim, 2)
    qr, qi = q_pair[..., 0].float(), q_pair[..., 1].float()
    kr, ki = k_pair[..., 0].float(), k_pair[..., 1].float()
    ref_q = torch.stack(
        [
            (qr * cos - qi * sin).to(q.dtype),
            (qr * sin + qi * cos).to(q.dtype),
        ],
        dim=-1,
    ).view_as(q)
    ref_k = torch.stack(
        [
            (kr * cos - ki * sin).to(k.dtype),
            (kr * sin + ki * cos).to(k.dtype),
        ],
        dim=-1,
    ).view_as(k)

    assert torch.equal(out_q, ref_q)
    assert torch.equal(out_k, ref_k)
    assert cache._cos is not None
    cos_ptr = cache._cos.data_ptr()
    _ = cache.get(device=q.device, head_dim=q.shape[-1], max_positions=3)
    assert cache._cos.data_ptr() == cos_ptr


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_exact_packed_rope_matches_reference_cuda(dtype: torch.dtype) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if vocoder_kernels._exact_interleaved_rope_kernel is None:
        pytest.skip("requires Triton")

    torch.manual_seed(0)
    projected = torch.randn(11, 3, 2, 64, device="cuda", dtype=dtype)
    q = projected[:, 0]
    k = projected[:, 1]
    position_ids = torch.tensor(
        [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5],
        device="cuda",
    )
    cache = vocoder_decoder._MossPackedRopeCache(max_period=10000.0)
    reference_q, reference_k = vocoder_decoder._apply_cached_packed_rope(
        q.cpu(),
        k.cpu(),
        position_ids.cpu(),
        max_positions=6,
        cache=vocoder_decoder._MossPackedRopeCache(max_period=10000.0),
    )
    assert not q.is_contiguous()
    assert not k.is_contiguous()

    actual_q, actual_k = vocoder_decoder._apply_cached_packed_rope(
        q,
        k,
        position_ids,
        max_positions=6,
        cache=cache,
    )

    assert torch.equal(actual_q.cpu(), reference_q)
    assert torch.equal(actual_k.cpu(), reference_k)


def test_transformer_layer_uses_source_modules_for_primitive_ops() -> None:
    source = _CountingLayer(hidden_size=6)
    wrapper = MossAudioTokenizerTransformerLayer(source)
    x = torch.randn(2, 4, 6)

    _ = wrapper(x, input_lengths=torch.tensor([4, 4]))

    assert source.norm1.calls == 1
    assert source.norm2.calls == 1
    assert source.layer_scale_1.calls == 1
    assert source.layer_scale_2.calls == 1
    assert source.ffn[0].calls == 1
    assert source.ffn[2].calls == 1


def test_vocoder_decoder_wraps_supported_stage_types() -> None:
    patch_stage = _PatchStage(patch_size=2, is_downsample=False)
    decoder = nn.ModuleList([_FallbackProjectedStage(), patch_stage])
    wrapped = MossAudioTokenizerVocoderDecoder(decoder)

    assert len(wrapped) == 2
    assert isinstance(wrapped[0], MossAudioTokenizerProjectedTransformer)
    assert wrapped[1] is patch_stage


def test_vocoder_decoder_computes_output_lengths_on_host() -> None:
    decoder = nn.ModuleList(
        [
            _PatchStage(patch_size=2, is_downsample=False),
            _FallbackProjectedStage(),
            _PatchStage(patch_size=4, is_downsample=False),
            _PatchStage(patch_size=2, is_downsample=True),
        ]
    )
    wrapped = MossAudioTokenizerVocoderDecoder(decoder)

    assert wrapped.output_lengths([3, 5]) == [12, 20]


def test_vocoder_decoder_requires_packed_flash_for_every_transformer() -> None:
    legacy_source = _LegacyProjectedStage()
    legacy = MossAudioTokenizerVocoderDecoder(nn.ModuleList([legacy_source]))
    legacy_attention = legacy[0].transformer.layers[0].self_attn
    legacy_attention._flash_attn_varlen = lambda *args, **kwargs: None

    assert legacy.supports_packed_flash("cuda", torch.bfloat16)
    assert legacy.supports_packed_flash("cuda", torch.float16)
    assert not legacy.supports_packed_flash("cpu", torch.bfloat16)
    assert not legacy.supports_packed_flash("cuda", None)

    legacy_attention._flash_attn_varlen = None
    assert not legacy.supports_packed_flash("cuda", torch.bfloat16)

    local_source = _FallbackProjectedStage()
    local_source.transformer.layers[0].self_attn.attention_implementation = (
        "flash_attention_2"
    )
    local = MossAudioTokenizerVocoderDecoder(nn.ModuleList([local_source]))
    local_attention = local[0].transformer.layers[0].self_attn
    local_attention._flash_attn_varlen = lambda *args, **kwargs: None

    assert local.supports_packed_flash("cuda", torch.bfloat16)
    assert not local.supports_packed_flash("cuda", torch.float16)


def test_vocoder_decoder_wraps_legacy_moss_audio_tokenizer_fields() -> None:
    source = _LegacyProjectedStage()
    wrapped = MossAudioTokenizerVocoderDecoder(nn.ModuleList([source]))
    source_layer = source.transformer.layers[0]
    wrapped_layer = wrapped[0].transformer.layers[0]
    attention = wrapped_layer.self_attn

    assert attention.in_proj is source_layer.self_attn.in_projs[0]
    assert attention.out_proj is source_layer.self_attn.out_projs[0]
    assert wrapped_layer.ffn.linear1 is source_layer.linear1
    assert wrapped_layer.ffn.linear2 is source_layer.linear2
    assert wrapped_layer.ffn.activation is source_layer.activation

    attention._can_run_packed_flash = lambda _: True
    assert attention.resolve_attention_implementation(torch.randn(2, 6)) == (
        "flash_attention_2"
    )

    attention._can_run_packed_flash = lambda _: False
    x = torch.randn(2, 3, 4)
    lengths = torch.tensor([4, 3])
    out, out_lengths = wrapped(x, lengths)

    assert out.shape == (2, 7, 4)
    assert torch.equal(out_lengths, lengths)
