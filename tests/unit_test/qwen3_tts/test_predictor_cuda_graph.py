# SPDX-License-Identifier: Apache-2.0
"""Bit-identity and capture-safety gates for the Qwen3-TTS predictor CUDA graph.

The per-semantic-token code-predictor chain (lm_head + seeded sampling + codec
embedding + predictor stack, num_code_groups - 1 sub-iterations) is captured as
one CUDA graph per (batch bucket, sampling signature). Graphed output codes and
summed embeddings must equal the eager chain bit-for-bit (torch.equal) at
bucket-exact batch sizes, and the dispatch/replay path must never host-sync.
"""

from __future__ import annotations

import ast
import copy
import gc
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from sglang.kernels.fused_op import get_fused_op_backend, set_fused_op_backend
from sglang.kernels.spec import KernelBackend
from sglang.srt.layers.quantization.unquant import Bf16GemmBackend
from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
from torch import nn

import sglang_omni.models.qwen3_tts.sglang_model as sglang_model_module
from sglang_omni.models.qwen3_tts.sglang_model import Qwen3TTSTalker
from sglang_omni.vendor.sglang.layers import RMSNorm
from sglang_omni.vendor.sglang.models import apply_qk_norm


@pytest.fixture(autouse=True)
def _stub_qk_norm(monkeypatch: pytest.MonkeyPatch):
    # apply_qk_norm reads global server args (unset in unit tests); the norm
    # ops themselves are covered by the real RMSNorm layer norms.
    monkeypatch.setattr(sglang_model_module, "apply_qk_norm", lambda q, k, **_: (q, k))


@pytest.fixture(autouse=True)
def _require_cuda_for_accelerator_tests(request: pytest.FixtureRequest):
    if request.node.get_closest_marker("accelerator") and not torch.cuda.is_available():
        pytest.skip("predictor CUDA graph needs CUDA")


HIDDEN = 8
NUM_HEADS = 2
NUM_KV_HEADS = 1
HEAD_DIM = 4
NUM_CODE_GROUPS = 4
PRED_VOCAB = 16
MAX_BS = 16
BUCKETS = (1, 2, 4, 8, 16)
DTYPE = torch.bfloat16


class _TupleLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_features, out_features, bias=False)
        self.quant_method = sglang_model_module.UnquantizedLinearMethod()
        self.tp_size = 1

    @property
    def weight(self) -> torch.Tensor:
        return self.proj.weight

    @property
    def bias(self) -> None:
        return None

    def forward(self, hidden_states: torch.Tensor):
        return self.proj(hidden_states), None


class _IdentityRotary(nn.Module):
    def forward(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        fused_set_kv_buffer_arg=None,
    ):
        del positions, fused_set_kv_buffer_arg
        return q, k


def _build_talker(device: torch.device) -> Qwen3TTSTalker:
    torch.manual_seed(7)
    predictor_len = NUM_CODE_GROUPS + 1
    talker = object.__new__(Qwen3TTSTalker)
    talker.training = False
    talker.config = SimpleNamespace(
        num_code_groups=NUM_CODE_GROUPS,
        code_predictor_config=SimpleNamespace(
            vocab_size=PRED_VOCAB,
            hidden_size=HIDDEN,
        ),
    )
    positions = torch.arange(predictor_len, device=device, dtype=torch.long)
    talker._predictor_positions = positions
    talker._predictor_position_rows = (
        positions[:, None].expand(predictor_len, MAX_BS).contiguous()
    )
    talker._predictor_k_cache = torch.zeros(
        1, MAX_BS, predictor_len, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=DTYPE
    )
    talker._predictor_v_cache = torch.zeros_like(talker._predictor_k_cache)
    talker._predictor_rope_stores_kv = False
    talker._output_codes = torch.zeros(
        MAX_BS, NUM_CODE_GROUPS, dtype=torch.long, device=device
    )
    talker._output_embeds = torch.zeros(MAX_BS, HIDDEN, device=device, dtype=DTYPE)
    talker._predictor_embedding_buffer = torch.empty(
        MAX_BS, HIDDEN, device=device, dtype=DTYPE
    )
    talker._sampled_token_ids = torch.zeros(MAX_BS, dtype=torch.long, device=device)

    talker._sub_batch_size = 0
    talker._sub_temperature_tensor = torch.full(
        (MAX_BS,), 0.9, device=device, dtype=torch.float32
    )
    talker._sub_top_p_tensor = torch.ones(MAX_BS, device=device, dtype=torch.float32)
    talker._sub_top_k_tensor = torch.full(
        (MAX_BS,), 50, device=device, dtype=torch.long
    )
    talker._semantic_sampling_seed_tensor = torch.zeros(
        MAX_BS, device=device, dtype=torch.long
    )
    talker._sub_sampling_seed_tensor = torch.zeros(
        MAX_BS, device=device, dtype=torch.long
    )
    talker._sub_do_sample_tensor = torch.zeros(MAX_BS, device=device, dtype=torch.bool)
    talker._sub_seed_offsets = torch.arange(
        1, NUM_CODE_GROUPS, device=device, dtype=torch.long
    )
    talker._sub_has_sampled_rows = False
    talker._sub_has_argmax_rows = False
    talker._sub_sampled_has_top_p = False
    talker._sub_sampled_max_top_k = 0
    talker._sub_sampled_has_unbounded_top_k = False

    layer = SimpleNamespace(
        input_layernorm=RMSNorm(HIDDEN, eps=1e-6).to(device, DTYPE),
        post_attention_layernorm=RMSNorm(HIDDEN, eps=1e-6).to(device, DTYPE),
        mlp=nn.Linear(HIDDEN, HIDDEN, bias=False).to(device, DTYPE),
    )
    layer.self_attn = SimpleNamespace(
        q_size=NUM_HEADS * HEAD_DIM,
        kv_size=NUM_KV_HEADS * HEAD_DIM,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        q_norm=RMSNorm(HEAD_DIM, eps=1e-6).to(device, DTYPE),
        k_norm=RMSNorm(HEAD_DIM, eps=1e-6).to(device, DTYPE),
        alt_stream=None,
        qkv_proj=_TupleLinear(HIDDEN, (NUM_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM).to(
            device, DTYPE
        ),
        o_proj=_TupleLinear(NUM_HEADS * HEAD_DIM, HIDDEN).to(device, DTYPE),
        rotary_emb=_IdentityRotary(),
    )
    projection = nn.Linear(HIDDEN, HIDDEN, bias=True).to(device, DTYPE)
    talker.code_predictor = SimpleNamespace(
        model=SimpleNamespace(
            layers=[layer],
            norm=RMSNorm(HIDDEN, eps=1e-6).to(device, DTYPE),
            codec_embedding=nn.ModuleList(
                [
                    nn.Embedding(PRED_VOCAB, HIDDEN).to(device, DTYPE)
                    for _ in range(NUM_CODE_GROUPS - 1)
                ]
            ),
        ),
        lm_head=nn.ModuleList(
            [
                _TupleLinear(HIDDEN, PRED_VOCAB).to(device, DTYPE)
                for _ in range(NUM_CODE_GROUPS - 1)
            ]
        ),
        project_input=lambda hidden: projection(hidden),
    )
    layer0_embedding = nn.Embedding(PRED_VOCAB, HIDDEN).to(device, DTYPE)
    talker.get_input_embeddings = lambda: layer0_embedding

    talker._predictor_graphs = {}
    talker._predictor_graph_disabled = set()
    talker._predictor_graph_batch_sizes = BUCKETS
    talker._predictor_graph_enabled = True
    talker._predictor_graph_failure_count = 0
    talker._predictor_graph_capacity_fallback_count = 0
    talker._predictor_graph_capacity_warned = False
    talker._predictor_graph_capture_count = 0
    talker._predictor_graph_startup_count = 0
    talker._predictor_graph_pool = None
    talker._predictor_capture_stream = None
    return talker


def _request(
    *,
    dosample: bool = True,
    temperature: float = 0.9,
    top_p: float = 1.0,
    top_k: int = 5,
    sub_seed: int = 1234,
    semantic_seed: int = 99,
) -> SimpleNamespace:
    return SimpleNamespace(
        data=SimpleNamespace(
            semantic_sampling_seed=semantic_seed,
            subtalker_dosample=dosample,
            subtalker_temperature=temperature,
            subtalker_top_p=top_p,
            subtalker_top_k=top_k,
            subtalker_sampling_seed=sub_seed,
        )
    )


def _uniform_requests(batch_size: int, **kwargs) -> list[SimpleNamespace]:
    return [
        _request(sub_seed=1000 + idx, semantic_seed=2000 + idx, **kwargs)
        for idx in range(batch_size)
    ]


def _step_inputs(batch_size: int, device: torch.device, *, step: int = 0):
    generator = torch.Generator(device="cpu").manual_seed(31 * batch_size + step)
    layer0 = torch.randint(
        0, PRED_VOCAB, (batch_size, 1), generator=generator, dtype=torch.long
    ).to(device)
    hidden = torch.randn(
        batch_size, 1, HIDDEN, generator=generator, dtype=torch.float32
    ).to(device, DTYPE)
    positions = torch.arange(
        step * 3, step * 3 + batch_size, device=device, dtype=torch.long
    )
    return layer0, hidden, positions


def _run_eager(talker, layer0, hidden, positions):
    with torch.no_grad():
        codes, embeds = talker._code_predictor_forward_incremental(
            layer0, hidden, semantic_positions=positions
        )
    return codes.detach().clone(), embeds.detach().clone()


def _run_forward(talker, layer0, hidden, positions):
    with torch.no_grad():
        codes, embeds = talker.code_predictor_forward(
            layer0, hidden, semantic_positions=positions
        )
        torch.cuda.synchronize()
    return codes.detach().clone(), embeds.detach().clone()


@pytest.mark.accelerator
def test_greedy_prediction_reads_no_seed_state():
    device = torch.device("cuda")
    talker = _build_talker(device)
    talker.prepare_decode_buffers(_uniform_requests(3, dosample=False))
    layer0, hidden, positions = _step_inputs(3, device)
    expected_codes, expected_embeds = _run_eager(talker, layer0, hidden, positions)

    talker._sub_seed_offsets = None
    eager_codes, eager_embeds = _run_eager(talker, layer0, hidden, positions)
    graph_codes, graph_embeds = _run_forward(talker, layer0, hidden, positions)

    assert torch.equal(eager_codes, expected_codes)
    assert torch.equal(eager_embeds, expected_embeds)
    assert torch.equal(graph_codes, expected_codes)
    assert torch.equal(graph_embeds, expected_embeds)


@pytest.mark.accelerator
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16])
@pytest.mark.parametrize(
    "sampling_kwargs",
    [
        {"top_k": 5, "top_p": 1.0},
        {"top_k": 5, "top_p": 0.9},
        {"top_k": 0, "top_p": 1.0},
    ],
    ids=["topk", "topk-topp", "fullsort"],
)
def test_graph_bit_identity_sampled(batch_size: int, sampling_kwargs: dict):
    device = torch.device("cuda")
    talker = _build_talker(device)
    talker.prepare_decode_buffers(_uniform_requests(batch_size, **sampling_kwargs))
    layer0, hidden, positions = _step_inputs(batch_size, device)

    eager_codes, eager_embeds = _run_eager(talker, layer0, hidden, positions)
    graph_codes, graph_embeds = _run_forward(talker, layer0, hidden, positions)

    assert talker._predictor_graphs, "no predictor graph captured"
    assert torch.equal(graph_codes, eager_codes), (
        f"codes not bit-identical (bs={batch_size}): "
        f"mismatches={(graph_codes != eager_codes).sum().item()}"
    )
    assert torch.equal(graph_embeds, eager_embeds), (
        f"summed embeddings not bit-identical (bs={batch_size}): "
        f"max|delta|={(graph_embeds - eager_embeds).abs().max().item():.3e}"
    )


@pytest.mark.accelerator
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16])
def test_graph_bit_identity_argmax(batch_size: int):
    device = torch.device("cuda")
    talker = _build_talker(device)
    talker.prepare_decode_buffers(_uniform_requests(batch_size, dosample=False))
    layer0, hidden, positions = _step_inputs(batch_size, device)

    eager_codes, eager_embeds = _run_eager(talker, layer0, hidden, positions)
    graph_codes, graph_embeds = _run_forward(talker, layer0, hidden, positions)

    assert talker._predictor_graphs
    assert torch.equal(graph_codes, eager_codes)
    assert torch.equal(graph_embeds, eager_embeds)


@pytest.mark.accelerator
def test_missing_embedding_buffer_uses_original_graph_path():
    """The captured fused path must retain the original embedding operation."""
    device = torch.device("cuda")
    fused_talker = _build_talker(device)
    fallback_talker = _build_talker(device)
    requests = _uniform_requests(4)
    fused_talker.prepare_decode_buffers(requests)
    fallback_talker.prepare_decode_buffers(requests)
    object.__delattr__(fallback_talker, "_predictor_embedding_buffer")
    layer0, hidden, positions = _step_inputs(4, device)

    fused_codes, fused_embeds = _run_forward(fused_talker, layer0, hidden, positions)
    fallback_codes, fallback_embeds = _run_forward(
        fallback_talker, layer0, hidden, positions
    )

    assert torch.equal(fused_codes, fallback_codes)
    assert torch.equal(fused_embeds, fallback_embeds)


@pytest.mark.accelerator
def test_eager_predictor_leaves_the_talker_hidden_untouched_for_an_identity_projection():
    device = torch.device("cuda")
    talker = _build_talker(device)
    talker.code_predictor.project_input = lambda hidden: hidden
    talker.prepare_decode_buffers(_uniform_requests(2))
    layer0, hidden, positions = _step_inputs(2, device)
    hidden_before = hidden.clone()

    _run_eager(talker, layer0, hidden, positions)

    assert torch.equal(hidden, hidden_before)


def _with_predictor_layers(talker: Qwen3TTSTalker, num_layers: int) -> Qwen3TTSTalker:
    """Deep copy the fixture layer so the predictor has num_layers of them."""
    model = talker.code_predictor.model
    first = model.layers[0]
    model.layers = [first] + [copy.deepcopy(first) for _ in range(num_layers - 1)]
    cache = talker._predictor_k_cache
    talker._predictor_k_cache = torch.zeros(
        num_layers, *cache.shape[1:], device=cache.device, dtype=cache.dtype
    )
    talker._predictor_v_cache = torch.zeros_like(talker._predictor_k_cache)
    return talker


def _predictor_one_token_out_of_place(
    talker: Qwen3TTSTalker, token_embeds: torch.Tensor, *, cache_len: int
) -> torch.Tensor:
    """The predictor layer stack in the residual form without aliasing.

    Same calls as the talker's forward. The three calls that overwrite an
    operand, the fused add and norm, the o_proj epilogue and the final fused
    norm, get clones, so no tensor is ever read after it was overwritten."""
    batch_size, _, hidden_size = token_embeds.shape
    positions = talker._predictor_position_rows[cache_len, :batch_size]
    residual = token_embeds
    mlp_out = None
    for layer_idx, layer in enumerate(talker.code_predictor.model.layers):
        if mlp_out is None:
            normed = layer.input_layernorm(residual.reshape(-1, hidden_size))
        else:
            normed, residual = layer.input_layernorm(
                mlp_out.clone(), residual.reshape(-1, hidden_size).clone()
            )
            residual = residual.reshape(batch_size, 1, hidden_size)
        attn_input = talker._predictor_cached_self_attention(
            layer_idx=layer_idx,
            attn=layer.self_attn,
            hidden_states=normed.reshape(batch_size, 1, hidden_size),
            positions=positions,
            batch_size=batch_size,
            cache_len=cache_len,
        )
        residual = talker._predictor_o_proj_add_residual(
            layer.self_attn.o_proj, attn_input, residual.clone()
        )
        normed = layer.post_attention_layernorm(residual.reshape(-1, hidden_size))
        mlp_out = layer.mlp(normed)
    normed, _ = talker.code_predictor.model.norm(
        mlp_out.clone(), residual.reshape(-1, hidden_size).clone()
    )
    return normed.reshape(batch_size, 1, hidden_size)


@pytest.mark.accelerator
@pytest.mark.parametrize("num_layers, batch_size", [(1, 1), (3, 2), (3, 16)])
def test_eager_predictor_in_place_residual_norms_match_the_out_of_place_form(
    num_layers: int, batch_size: int
):
    device = torch.device("cuda")
    in_place = _with_predictor_layers(_build_talker(device), num_layers)
    reference = _with_predictor_layers(_build_talker(device), num_layers)
    generator = torch.Generator(device="cpu").manual_seed(num_layers * 100 + batch_size)
    embeds = torch.randn(
        batch_size, 1, HIDDEN, generator=generator, dtype=torch.float32
    ).to(device, DTYPE)

    with torch.no_grad():
        expected = _predictor_one_token_out_of_place(
            reference, embeds.clone(), cache_len=0
        )
        actual = in_place._predictor_forward_one_token(
            token_embeds=embeds.clone(), batch_size=batch_size, cache_len=0
        )

    assert actual.shape == (batch_size, 1, HIDDEN)
    assert actual.dtype == DTYPE
    assert torch.equal(actual, expected)
    assert torch.equal(in_place._predictor_k_cache, reference._predictor_k_cache)
    assert torch.equal(in_place._predictor_v_cache, reference._predictor_v_cache)


@pytest.mark.accelerator
def test_eager_predictor_adds_each_residual_inside_the_norm_that_follows(
    monkeypatch: pytest.MonkeyPatch,
):
    device = torch.device("cuda")
    talker = _with_predictor_layers(_build_talker(device), 3)
    layers = talker.code_predictor.model.layers
    hidden_norms = {
        id(module)
        for layer in layers
        for module in (layer.input_layernorm, layer.post_attention_layernorm)
    } | {id(talker.code_predictor.model.norm)}
    calls: list[tuple[int, bool]] = []
    original_forward = RMSNorm.forward

    def recording_forward(self, x, residual=None, *args, **kwargs):
        if id(self) in hidden_norms:
            calls.append((x.dim(), residual is not None))
        return original_forward(self, x, residual, *args, **kwargs)

    monkeypatch.setattr(RMSNorm, "forward", recording_forward)
    embeds = torch.randn(2, 1, HIDDEN, device=device, dtype=DTYPE)
    with torch.no_grad():
        talker._predictor_forward_one_token(
            token_embeds=embeds, batch_size=2, cache_len=0
        )

    assert all(dim == 2 for dim, _ in calls)
    fused_calls = [fused for _, fused in calls]
    assert fused_calls == [False, False] + [True, False] * (len(layers) - 1) + [True]


@pytest.mark.accelerator
def test_eager_predictor_output_survives_the_next_token():
    device = torch.device("cuda")
    talker = _with_predictor_layers(_build_talker(device), 2)
    generator = torch.Generator(device="cpu").manual_seed(5)
    embeds = [
        torch.randn(2, 1, HIDDEN, generator=generator, dtype=torch.float32).to(
            device, DTYPE
        )
        for _ in range(2)
    ]

    with torch.no_grad():
        first = talker._predictor_forward_one_token(
            token_embeds=embeds[0], batch_size=2, cache_len=0
        )
        snapshot = first.clone()
        second = talker._predictor_forward_one_token(
            token_embeds=embeds[1], batch_size=2, cache_len=1
        )

    assert torch.equal(first, snapshot)
    assert second.data_ptr() != first.data_ptr()
    assert not torch.equal(second, first)


@pytest.mark.accelerator
def test_eager_predictor_accepts_a_strided_input_and_leaves_its_neighbours():
    device = torch.device("cuda")
    strided_talker = _with_predictor_layers(_build_talker(device), 2)
    contiguous_talker = _with_predictor_layers(_build_talker(device), 2)
    wide = torch.randn(2, 1, 2 * HIDDEN, device=device, dtype=DTYPE)
    strided = wide[:, :, :HIDDEN]
    assert not strided.is_contiguous()
    neighbours_before = wide[:, :, HIDDEN:].clone()
    contiguous = strided.clone()

    with torch.no_grad():
        from_strided = strided_talker._predictor_forward_one_token(
            token_embeds=strided, batch_size=2, cache_len=0
        )
        from_contiguous = contiguous_talker._predictor_forward_one_token(
            token_embeds=contiguous, batch_size=2, cache_len=0
        )

    torch.testing.assert_close(from_strided, from_contiguous)
    assert torch.equal(wide[:, :, HIDDEN:], neighbours_before)


@pytest.mark.accelerator
def test_graph_bit_identity_argmax_none_positions():
    device = torch.device("cuda")
    talker = _build_talker(device)
    talker.prepare_decode_buffers(_uniform_requests(2, dosample=False))
    layer0, hidden, _ = _step_inputs(2, device)

    eager_codes, eager_embeds = _run_eager(talker, layer0, hidden, None)
    graph_codes, graph_embeds = _run_forward(talker, layer0, hidden, None)

    assert talker._predictor_graphs
    assert torch.equal(graph_codes, eager_codes)
    assert torch.equal(graph_embeds, eager_embeds)


@pytest.mark.accelerator
def test_graph_padded_bucket_bit_identity():
    """Live bs=3 replays through the bucket-4 graph with padded rows."""
    device = torch.device("cuda")
    talker = _build_talker(device)
    talker.prepare_decode_buffers(_uniform_requests(3))
    layer0, hidden, positions = _step_inputs(3, device)

    eager_codes, eager_embeds = _run_eager(talker, layer0, hidden, positions)
    graph_codes, graph_embeds = _run_forward(talker, layer0, hidden, positions)

    assert any(key[0] == 4 for key in talker._predictor_graphs)
    assert not any(key[0] == 3 for key in talker._predictor_graphs)
    assert graph_codes.shape == eager_codes.shape
    assert torch.equal(graph_codes, eager_codes)
    assert torch.equal(graph_embeds, eager_embeds)


@pytest.mark.accelerator
def test_mixed_padded_bucket_bit_identity_and_reuse():
    """Mixed live bs=3 replays through one bucket-4 graph across row masks."""
    device = torch.device("cuda")
    talker = _build_talker(device)

    requests = [
        _request(dosample=True),
        _request(dosample=False),
        _request(dosample=True),
    ]
    talker.prepare_decode_buffers(requests)
    layer0, hidden, positions = _step_inputs(3, device)
    eager_codes, eager_embeds = _run_eager(talker, layer0, hidden, positions)
    graph_codes, graph_embeds = _run_forward(talker, layer0, hidden, positions)

    assert any(key[0] == 4 and key[1] == "sampled" for key in talker._predictor_graphs)
    assert torch.equal(graph_codes, eager_codes)
    assert torch.equal(graph_embeds, eager_embeds)

    requests = [
        _request(dosample=False),
        _request(dosample=True),
        _request(dosample=True),
    ]
    talker.prepare_decode_buffers(requests)
    layer0, hidden, positions = _step_inputs(3, device, step=1)
    eager_codes, eager_embeds = _run_eager(talker, layer0, hidden, positions)
    graph_codes, graph_embeds = _run_forward(talker, layer0, hidden, positions)

    assert len(talker._predictor_graphs) == 1
    assert torch.equal(graph_codes, eager_codes)
    assert torch.equal(graph_embeds, eager_embeds)


@pytest.mark.accelerator
def test_graph_multi_step_replay_bit_identity():
    """Consecutive steps reuse one captured graph and stay bit-identical."""
    device = torch.device("cuda")
    talker = _build_talker(device)
    talker.prepare_decode_buffers(_uniform_requests(4))

    for step in range(3):
        layer0, hidden, positions = _step_inputs(4, device, step=step)
        eager_codes, eager_embeds = _run_eager(talker, layer0, hidden, positions)
        graph_codes, graph_embeds = _run_forward(talker, layer0, hidden, positions)
        assert torch.equal(graph_codes, eager_codes), f"step={step}"
        assert torch.equal(graph_embeds, eager_embeds), f"step={step}"

    assert len(talker._predictor_graphs) == 1


@pytest.mark.accelerator
def test_mixed_sampled_argmax_rows_use_graph_bit_identity(
    monkeypatch: pytest.MonkeyPatch,
):
    device = torch.device("cuda")
    talker = _build_talker(device)
    requests = [_request(dosample=True), _request(dosample=False)]
    talker.prepare_decode_buffers(requests)
    layer0, hidden, positions = _step_inputs(2, device)

    real_seeded = Qwen3TTSTalker._sample_subtalker_token_seeded

    def _sentinel_seeded(self, logits, *, sub_positions):
        del self, sub_positions
        return (torch.argmax(logits, dim=-1) + 1) % PRED_VOCAB

    monkeypatch.setattr(
        Qwen3TTSTalker, "_sample_subtalker_token_seeded", _sentinel_seeded
    )

    eager_codes, eager_embeds = _run_eager(talker, layer0, hidden, positions)
    graph_codes, graph_embeds = _run_forward(talker, layer0, hidden, positions)

    assert talker._predictor_graphs, "mixed batch did not capture a graph"
    assert torch.equal(graph_codes, eager_codes)
    assert torch.equal(graph_embeds, eager_embeds)

    monkeypatch.setattr(Qwen3TTSTalker, "_sample_subtalker_token_seeded", real_seeded)
    talker.prepare_decode_buffers([_request(dosample=False), _request(dosample=False)])
    argmax_codes, _ = talker._code_predictor_forward_incremental(
        layer0, hidden, semantic_positions=positions
    )

    assert not torch.equal(
        graph_codes[0, 1:], argmax_codes[0, 1:]
    ), "sampled row matches pure argmax -- the seeded path was not exercised"
    assert torch.equal(
        graph_codes[1, 1:], argmax_codes[1, 1:]
    ), "argmax row was affected by the seeded-sampling sentinel"


@pytest.mark.accelerator
def test_mixed_sampled_argmax_rows_preserve_argmax_tie_break():
    device = torch.device("cuda")
    talker = _build_talker(device)
    talker.prepare_decode_buffers([_request(dosample=True), _request(dosample=False)])
    logits = torch.full((2, PRED_VOCAB), -10.0, device=device)
    logits[0, 3] = 9.0
    logits[1, 5] = 9.0
    logits[1, 7] = 9.0

    tokens = talker._sample_subtalker_token(
        logits,
        sub_positions=talker._sub_seed_positions(
            torch.zeros(2, dtype=torch.long, device=device)
        )[0],
    )

    assert tokens[1].item() == torch.argmax(logits[1]).item() == 5


@pytest.mark.accelerator
def test_mixed_sampling_masks_reuse_one_graph():
    device = torch.device("cuda")
    talker = _build_talker(device)

    requests = [
        _request(dosample=True),
        _request(dosample=False),
        _request(dosample=True),
        _request(dosample=False),
    ]
    talker.prepare_decode_buffers(requests)
    layer0, hidden, positions = _step_inputs(4, device)
    eager_codes, eager_embeds = _run_eager(talker, layer0, hidden, positions)
    graph_codes, graph_embeds = _run_forward(talker, layer0, hidden, positions)
    assert torch.equal(graph_codes, eager_codes)
    assert torch.equal(graph_embeds, eager_embeds)

    requests = [
        _request(dosample=False),
        _request(dosample=True),
        _request(dosample=False),
        _request(dosample=True),
    ]
    talker.prepare_decode_buffers(requests)
    layer0, hidden, positions = _step_inputs(4, device, step=1)
    eager_codes, eager_embeds = _run_eager(talker, layer0, hidden, positions)
    graph_codes, graph_embeds = _run_forward(talker, layer0, hidden, positions)

    assert len(talker._predictor_graphs) == 1
    assert torch.equal(graph_codes, eager_codes)
    assert torch.equal(graph_embeds, eager_embeds)


@pytest.mark.accelerator
def test_all_sampled_and_mixed_batches_capture_separate_graphs():
    device = torch.device("cuda")
    talker = _build_talker(device)
    layer0, hidden, positions = _step_inputs(4, device)

    talker.prepare_decode_buffers(_uniform_requests(4))
    _run_forward(talker, layer0, hidden, positions)

    talker.prepare_decode_buffers(
        [
            _request(dosample=True),
            _request(dosample=False),
            _request(dosample=True),
            _request(dosample=False),
        ]
    )
    _run_forward(talker, layer0, hidden, positions)

    keys = sorted(talker._predictor_graphs)
    assert len(keys) == 2
    assert len({key[:-1] for key in keys}) == 1
    assert {key[1] for key in keys} == {"sampled"}
    assert {key[-1] for key in keys} == {False, True}


@pytest.mark.accelerator
def test_kill_switch_disables_graph_path():
    device = torch.device("cuda")
    talker = _build_talker(device)
    talker._predictor_graph_enabled = False
    talker.prepare_decode_buffers(_uniform_requests(2))
    layer0, hidden, positions = _step_inputs(2, device)

    eager_codes, eager_embeds = _run_eager(talker, layer0, hidden, positions)
    graph_codes, graph_embeds = _run_forward(talker, layer0, hidden, positions)

    assert not talker._predictor_graphs
    assert torch.equal(graph_codes, eager_codes)
    assert torch.equal(graph_embeds, eager_embeds)


def test_env_switch_parsing(monkeypatch: pytest.MonkeyPatch):
    env = sglang_model_module.QTTS_PREDICTOR_GRAPH_ENV
    monkeypatch.delenv(env, raising=False)
    assert sglang_model_module._predictor_graph_env_enabled() is True
    monkeypatch.setenv(env, "0")
    assert sglang_model_module._predictor_graph_env_enabled() is False
    monkeypatch.setenv(env, "false")
    assert sglang_model_module._predictor_graph_env_enabled() is False
    monkeypatch.setenv(env, "no")
    assert sglang_model_module._predictor_graph_env_enabled() is False
    monkeypatch.setenv(env, "1")
    assert sglang_model_module._predictor_graph_env_enabled() is True


@pytest.mark.accelerator
def test_capture_failure_disables_key_and_falls_back(
    monkeypatch: pytest.MonkeyPatch,
):
    device = torch.device("cuda")
    talker = _build_talker(device)
    talker.prepare_decode_buffers(_uniform_requests(2))
    layer0, hidden, positions = _step_inputs(2, device)

    calls = []

    class _BoomGraph:
        def __init__(self, *args, **kwargs) -> None:
            calls.append(1)
            raise RuntimeError("simulated capture failure")

    monkeypatch.setattr(sglang_model_module, "_PredictorDecodeGraph", _BoomGraph)

    eager_codes, eager_embeds = _run_eager(talker, layer0, hidden, positions)
    graph_codes, graph_embeds = _run_forward(talker, layer0, hidden, positions)

    assert len(calls) == 1
    assert talker._predictor_graph_disabled, "failed key must be disabled"
    assert torch.equal(graph_codes, eager_codes)
    assert torch.equal(graph_embeds, eager_embeds)

    _run_forward(talker, layer0, hidden, positions)
    assert len(calls) == 1, "disabled key must not retry capture"


@pytest.mark.accelerator
def test_capture_failure_restores_live_sub_state(monkeypatch: pytest.MonkeyPatch):
    device = torch.device("cuda")
    talker = _build_talker(device)
    talker.prepare_decode_buffers(
        [
            _request(dosample=True, sub_seed=1000, semantic_seed=2000),
            _request(dosample=False, sub_seed=1001, semantic_seed=2001),
        ]
    )
    layer0, hidden, positions = _step_inputs(2, device)

    real_forward = Qwen3TTSTalker._code_predictor_forward_incremental

    def _boom_forward(self, *args, **kwargs):
        if torch.cuda.current_stream(device) != torch.cuda.default_stream(device):
            raise RuntimeError("simulated capture failure")
        return real_forward(self, *args, **kwargs)

    monkeypatch.setattr(
        Qwen3TTSTalker, "_code_predictor_forward_incremental", _boom_forward
    )
    _run_forward(talker, layer0, hidden, positions)

    assert talker._predictor_graph_disabled
    assert talker._sub_batch_size == 2
    assert talker._sub_has_sampled_rows is True
    assert talker._sub_do_sample_tensor[:2].tolist() == [True, False]


class _NoHostReadbackTensor(torch.Tensor):
    """Tensor whose host-materialization entry points fail the test."""

    def cpu(self, *args, **kwargs):
        raise RuntimeError("host readback (cpu) on the predictor graph path")

    def tolist(self):
        raise RuntimeError("host readback (tolist) on the predictor graph path")

    def numpy(self, *args, **kwargs):
        raise RuntimeError("host readback (numpy) on the predictor graph path")

    def item(self):
        raise RuntimeError("host readback (item) on the predictor graph path")

    def __float__(self):
        raise RuntimeError("host readback (float) on the predictor graph path")

    def __int__(self):
        raise RuntimeError("host readback (int) on the predictor graph path")

    def __bool__(self):
        raise RuntimeError("host readback (bool) on the predictor graph path")

    def __iter__(self):
        raise RuntimeError("host readback (iter) on the predictor graph path")

    def to(self, *args, **kwargs):
        if any(str(a) == "cpu" for a in args) or str(kwargs.get("device")) == "cpu":
            raise RuntimeError("host readback (to cpu) on the predictor graph path")
        return super().to(*args, **kwargs)


@pytest.mark.accelerator
def test_no_host_readback_on_graph_dispatch_and_replay():
    """Live per-step inputs must reach the graph via device-side copies only."""
    device = torch.device("cuda")
    talker = _build_talker(device)
    talker.prepare_decode_buffers(_uniform_requests(2))
    layer0, hidden, positions = _step_inputs(2, device)
    guarded_layer0 = layer0.as_subclass(_NoHostReadbackTensor)
    guarded_hidden = hidden.as_subclass(_NoHostReadbackTensor)
    guarded_positions = positions.as_subclass(_NoHostReadbackTensor)

    with torch.no_grad():
        talker.code_predictor_forward(
            guarded_layer0, guarded_hidden, semantic_positions=guarded_positions
        )
        assert talker._predictor_graphs, "expected graph capture"
        talker.code_predictor_forward(
            guarded_layer0, guarded_hidden, semantic_positions=guarded_positions
        )
        torch.cuda.synchronize()


@pytest.mark.accelerator
def test_no_host_readback_in_eager_chain():
    """The captured body itself must be free of host materialization."""
    device = torch.device("cuda")
    talker = _build_talker(device)
    talker.prepare_decode_buffers(_uniform_requests(2))
    layer0, hidden, positions = _step_inputs(2, device)

    with torch.no_grad():
        talker._code_predictor_forward_incremental(
            layer0.as_subclass(_NoHostReadbackTensor),
            hidden.as_subclass(_NoHostReadbackTensor),
            semantic_positions=positions.as_subclass(_NoHostReadbackTensor),
        )
        torch.cuda.synchronize()


def test_capture_uses_thread_local_error_mode():
    source = (
        Path(__file__).resolve().parents[3]
        / "sglang_omni"
        / "models"
        / "qwen3_tts"
        / "sglang_model.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    graph_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "graph"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "cuda"
    ]
    assert graph_calls, "Qwen3-TTS predictor CUDA graph capture call not found"
    assert any(
        keyword.arg == "capture_error_mode"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value == "thread_local"
        for call in graph_calls
        for keyword in call.keywords
    )


def test_normalize_predictor_graph_batch_sizes():
    normalize = Qwen3TTSTalker._normalize_predictor_graph_batch_sizes

    def _args(bs):
        return SimpleNamespace(
            cuda_graph_config=SimpleNamespace(decode=SimpleNamespace(bs=bs))
        )

    assert normalize(_args(None), max_batch_size=16) == (1, 2, 4, 8, 12, 16)
    assert normalize(_args([4, 2, 2, 64]), max_batch_size=16) == (2, 4, 16)
    assert normalize(_args([1, 3, 7]), max_batch_size=8) == (1, 3, 7, 8)
    assert normalize(_args(None), max_batch_size=2) == (1, 2)


def test_quantize_predictor_top_k_ladder():
    quantize = sglang_model_module._quantize_predictor_top_k
    assert quantize(1, 2048) == 4
    assert quantize(37, 2048) == 50
    assert quantize(50, 2048) == 50
    assert quantize(51, 2048) == 64
    assert quantize(600, 2048) == 1024
    assert quantize(1500, 2048) is None
    assert quantize(4, 16) == 4
    assert quantize(5, 16) == 8
    assert quantize(9, 16) is None


@pytest.mark.accelerator
def test_graph_key_shared_across_request_top_k_values():
    """top_k=5 and top_k=7 land in the same ladder bucket and share one graph."""
    device = torch.device("cuda")
    talker = _build_talker(device)

    for top_k in (5, 7):
        talker.prepare_decode_buffers(_uniform_requests(2, top_k=top_k))
        layer0, hidden, positions = _step_inputs(2, device, step=top_k)
        eager_codes, eager_embeds = _run_eager(talker, layer0, hidden, positions)
        graph_codes, graph_embeds = _run_forward(talker, layer0, hidden, positions)
        assert torch.equal(graph_codes, eager_codes), f"top_k={top_k}"
        assert torch.equal(graph_embeds, eager_embeds), f"top_k={top_k}"

    assert len(talker._predictor_graphs) == 1, (
        "distinct request top_k values within one ladder bucket must share "
        f"one graph, got keys {sorted(talker._predictor_graphs)}"
    )


@pytest.mark.accelerator
def test_row_top_k_below_bucket_width_bit_identity():
    """Rows with k below the captured bucket width stay bit-identical to eager."""
    device = torch.device("cuda")
    talker = _build_talker(device)
    requests = [
        _request(top_k=3, sub_seed=1000, semantic_seed=2000),
        _request(top_k=5, sub_seed=1001, semantic_seed=2001),
    ]
    talker.prepare_decode_buffers(requests)
    layer0, hidden, positions = _step_inputs(2, device)

    eager_codes, eager_embeds = _run_eager(talker, layer0, hidden, positions)
    graph_codes, graph_embeds = _run_forward(talker, layer0, hidden, positions)

    assert any(
        key[2] == 8 for key in talker._predictor_graphs
    ), f"expected capture at ladder width 8, got keys {sorted(talker._predictor_graphs)}"
    assert torch.equal(graph_codes, eager_codes)
    assert torch.equal(graph_embeds, eager_embeds)


@pytest.mark.accelerator
def test_replay_tracks_per_step_sampling_params():
    """One graph key, three replays with fresh temps/top_p/top_k/seeds per step;
    stale captured params would break bit-identity."""
    device = torch.device("cuda")
    talker = _build_talker(device)
    step_params = [
        {"temperature": 0.9, "top_p": 0.8, "top_k": 5},
        {"temperature": 1.1, "top_p": 0.95, "top_k": 7},
        {"temperature": 0.7, "top_p": 0.85, "top_k": 6},
    ]

    for step, params in enumerate(step_params):
        requests = [
            _request(
                sub_seed=5000 + 10 * step + idx, semantic_seed=6000 + idx, **params
            )
            for idx in range(4)
        ]
        talker.prepare_decode_buffers(requests)
        layer0, hidden, positions = _step_inputs(4, device, step=step)
        eager_codes, eager_embeds = _run_eager(talker, layer0, hidden, positions)
        graph_codes, graph_embeds = _run_forward(talker, layer0, hidden, positions)
        assert torch.equal(graph_codes, eager_codes), f"step={step} params={params}"
        assert torch.equal(graph_embeds, eager_embeds), f"step={step} params={params}"

    assert len(talker._predictor_graphs) == 1


@pytest.mark.accelerator
def test_global_disable_after_max_capture_failures(monkeypatch: pytest.MonkeyPatch):
    device = torch.device("cuda")
    talker = _build_talker(device)
    calls = []

    class _BoomGraph:
        def __init__(self, *args, **kwargs) -> None:
            calls.append(1)
            raise RuntimeError("simulated capture failure")

    monkeypatch.setattr(sglang_model_module, "_PredictorDecodeGraph", _BoomGraph)

    compositions = [(bs, True) for bs in (1, 2, 4, 8, 16)] + [
        (bs, False) for bs in (1, 2, 4)
    ]
    for batch_size, dosample in compositions:
        talker.prepare_decode_buffers(_uniform_requests(batch_size, dosample=dosample))
        layer0, hidden, positions = _step_inputs(batch_size, device)
        _run_forward(talker, layer0, hidden, positions)

    assert len(calls) == 8
    assert talker._predictor_graph_enabled is False, (
        "predictor graphs must self-disable after "
        f"{len(compositions)} distinct capture failures"
    )

    talker.prepare_decode_buffers(_uniform_requests(8, dosample=False))
    layer0, hidden, positions = _step_inputs(8, device)
    _run_forward(talker, layer0, hidden, positions)
    assert len(calls) == 8, "globally disabled graphs must not attempt new captures"


@pytest.mark.accelerator
def test_capture_failure_resets_cuda_graph(monkeypatch: pytest.MonkeyPatch):
    """A failed capture must release its CUDA graph (pool) via reset()."""
    device = torch.device("cuda")
    talker = _build_talker(device)
    reset_calls = []
    original_reset = torch.cuda.CUDAGraph.reset

    def spy_reset(self, *args, **kwargs):
        reset_calls.append(1)
        return original_reset(self, *args, **kwargs)

    monkeypatch.setattr(torch.cuda.CUDAGraph, "reset", spy_reset)

    @contextmanager
    def boom_capture_state(bucket_size, signature):
        raise RuntimeError("simulated capture failure")
        yield

    talker._predictor_graph_capture_state = boom_capture_state
    talker.prepare_decode_buffers(_uniform_requests(2))
    layer0, hidden, positions = _step_inputs(2, device)
    _run_forward(talker, layer0, hidden, positions)

    assert talker._predictor_graph_disabled, "failed key must be disabled"
    assert reset_calls, "failed capture must reset() its CUDAGraph"


@pytest.mark.accelerator
def test_widened_top_k_masked_ranks_never_sampled():
    """Ranks past a row's true k remain impossible after log conversion."""
    device = torch.device("cuda")
    talker = _build_talker(device)
    logits = torch.linspace(2.0, -2.0, PRED_VOCAB, device=device).unsqueeze(0)
    allowed = set(torch.topk(logits[0], 2).indices.tolist())
    positions = torch.zeros(1, dtype=torch.long, device=device)

    for seed in range(100):
        # top_k=2 quantizes to ladder width 4, leaving ranks 2-3 masked
        talker.prepare_decode_buffers([_request(top_k=2, sub_seed=seed)])
        token = talker._sample_subtalker_token_seeded(
            logits,
            sub_positions=talker._sub_seed_positions(positions)[0],
        )
        assert token.item() in allowed, (
            f"seed={seed} sampled rank outside the request's top_k=2: "
            f"{token.item()} not in {sorted(allowed)}"
        )


@pytest.mark.accelerator
def test_graph_keys_share_memory_pool(monkeypatch: pytest.MonkeyPatch):
    """Distinct graph keys must capture into one model-owned memory pool."""
    device = torch.device("cuda")
    talker = _build_talker(device)
    pools = []
    real_graph = torch.cuda.graph

    class _SpyGraph(real_graph):
        def __init__(self, cuda_graph, pool=None, **kwargs):
            pools.append(pool)
            super().__init__(cuda_graph, pool=pool, **kwargs)

    monkeypatch.setattr(torch.cuda, "graph", _SpyGraph)

    layer0, hidden, positions = _step_inputs(2, device)
    talker.prepare_decode_buffers(_uniform_requests(2))
    _run_forward(talker, layer0, hidden, positions)
    talker.prepare_decode_buffers(_uniform_requests(2, dosample=False))
    _run_forward(talker, layer0, hidden, positions)

    assert len(talker._predictor_graphs) == 2
    assert len(pools) == 2
    assert pools[0] is not None
    assert pools[0] == pools[1]
    assert pools[0] == talker._predictor_graph_pool


@pytest.mark.accelerator
def test_graph_key_cache_capacity_fallback_without_eviction(
    monkeypatch: pytest.MonkeyPatch,
):
    device = torch.device("cuda")
    talker = _build_talker(device)
    layer0, hidden, positions = _step_inputs(2, device)
    monkeypatch.setattr(sglang_model_module, "_PREDICTOR_GRAPH_MAX_LAZY_KEYS", 2)

    def assert_graph_matches(requests) -> None:
        talker.prepare_decode_buffers(requests)
        eager_codes, eager_embeds = _run_eager(talker, layer0, hidden, positions)
        graph_codes, graph_embeds = _run_forward(talker, layer0, hidden, positions)
        assert torch.equal(graph_codes, eager_codes)
        assert torch.equal(graph_embeds, eager_embeds)

    sampled = _uniform_requests(2)
    argmax = _uniform_requests(2, dosample=False)
    other_sampled = _uniform_requests(2, top_k=3)

    assert_graph_matches(sampled)
    assert_graph_matches(argmax)
    assert_graph_matches(other_sampled)

    assert {key[1] for key in talker._predictor_graphs} == {"sampled", "argmax"}
    assert len(talker._predictor_graphs) == 2
    assert talker._predictor_graph_capture_count == 2
    assert talker._predictor_graph_capacity_fallback_count == 1


@pytest.mark.accelerator
def test_top_p_removed_ranks_never_sampled():
    """Nucleus-removed ranks must be impossible, same rationale as the top-k mask."""
    device = torch.device("cuda")
    talker = _build_talker(device)
    logits = torch.zeros(1, PRED_VOCAB, device=device)
    logits[0, 3] = 4.0
    positions = torch.zeros(1, dtype=torch.long, device=device)

    for seed in range(100):
        # rank 0 alone holds ~0.97 mass, so top_p=0.5 removes ranks 1-3
        talker.prepare_decode_buffers([_request(top_k=4, top_p=0.5, sub_seed=seed)])
        token = talker._sample_subtalker_token_seeded(
            logits,
            sub_positions=talker._sub_seed_positions(positions)[0],
        )
        assert (
            token.item() == 3
        ), f"seed={seed} sampled a nucleus-removed rank: {token.item()}"


def test_capture_state_body_failure_restores_state():
    """An exception inside capture must restore the live sampling state."""
    device = torch.device("cpu")
    talker = _build_talker(device)
    talker.prepare_decode_buffers(_uniform_requests(3, dosample=False))
    saved = (
        talker._sub_batch_size,
        talker._sub_has_sampled_rows,
        talker._sub_has_argmax_rows,
        talker._sub_sampled_has_top_p,
        talker._sub_sampled_max_top_k,
        talker._sub_sampled_has_unbounded_top_k,
    )

    with pytest.raises(RuntimeError, match="simulated capture failure"):
        with talker._predictor_graph_capture_state(
            4, ("sampled", 8, True, False, True)
        ):
            assert talker._sub_batch_size == 4
            assert talker._sub_has_sampled_rows is True
            assert talker._sub_has_argmax_rows is True
            assert talker._sub_sampled_has_top_p is True
            assert talker._sub_sampled_max_top_k == 8
            assert talker._sub_sampled_has_unbounded_top_k is False
            raise RuntimeError("simulated capture failure")

    assert (
        talker._sub_batch_size,
        talker._sub_has_sampled_rows,
        talker._sub_has_argmax_rows,
        talker._sub_sampled_has_top_p,
        talker._sub_sampled_max_top_k,
        talker._sub_sampled_has_unbounded_top_k,
    ) == saved


def test_resolve_predictor_graph_enabled(monkeypatch: pytest.MonkeyPatch):
    talker = object.__new__(Qwen3TTSTalker)
    graph = SimpleNamespace(disable_cuda_graph=False)
    parallel = SimpleNamespace(tp_size=1)
    monkeypatch.setattr(
        sglang_model_module, "get_exec", lambda: SimpleNamespace(graph=graph)
    )
    monkeypatch.setattr(sglang_model_module, "get_parallel", lambda: parallel)
    monkeypatch.delenv(sglang_model_module.QTTS_PREDICTOR_GRAPH_ENV, raising=False)

    assert talker._resolve_predictor_graph_enabled() is True
    graph.disable_cuda_graph = True
    assert talker._resolve_predictor_graph_enabled() is False
    graph.disable_cuda_graph = False
    parallel.tp_size = 2
    assert talker._resolve_predictor_graph_enabled() is False
    parallel.tp_size = 1
    monkeypatch.setenv(sglang_model_module.QTTS_PREDICTOR_GRAPH_ENV, "0")
    assert talker._resolve_predictor_graph_enabled() is False


@pytest.mark.accelerator
def test_server_disable_cuda_graph_gates_predictor(monkeypatch: pytest.MonkeyPatch):
    """server_args.disable_cuda_graph must gate the lazily resolved graph path."""
    device = torch.device("cuda")
    talker = _build_talker(device)
    talker._predictor_graph_enabled = None
    monkeypatch.delenv(sglang_model_module.QTTS_PREDICTOR_GRAPH_ENV, raising=False)
    monkeypatch.setattr(
        sglang_model_module,
        "get_exec",
        lambda: SimpleNamespace(graph=SimpleNamespace(disable_cuda_graph=True)),
    )
    monkeypatch.setattr(
        sglang_model_module, "get_parallel", lambda: SimpleNamespace(tp_size=1)
    )
    talker.prepare_decode_buffers(_uniform_requests(2))
    layer0, hidden, positions = _step_inputs(2, device)

    eager_codes, eager_embeds = _run_eager(talker, layer0, hidden, positions)
    graph_codes, graph_embeds = _run_forward(talker, layer0, hidden, positions)

    assert not talker._predictor_graphs
    assert talker._predictor_graph_enabled is False
    assert torch.equal(graph_codes, eager_codes)
    assert torch.equal(graph_embeds, eager_embeds)


@pytest.mark.accelerator
def test_failed_capture_restores_the_current_stream(monkeypatch: pytest.MonkeyPatch):
    device = torch.device("cuda")
    talker = _build_talker(device)
    real_forward = Qwen3TTSTalker._code_predictor_forward_incremental

    def _sync_inside_capture(self, *args, **kwargs):
        if torch.cuda.is_current_stream_capturing():
            torch.cuda.synchronize()
        return real_forward(self, *args, **kwargs)

    monkeypatch.setattr(
        Qwen3TTSTalker, "_code_predictor_forward_incremental", _sync_inside_capture
    )
    talker.prepare_decode_buffers(_uniform_requests(2))
    layer0, hidden, positions = _step_inputs(2, device)
    eager_codes, eager_embeds = _run_eager(talker, layer0, hidden, positions)

    graph_codes, graph_embeds = _run_forward(talker, layer0, hidden, positions)

    assert torch.cuda.current_stream(device) == torch.cuda.default_stream(device)
    assert gc.isenabled()
    assert not talker._predictor_graphs
    assert talker._predictor_graph_failure_count == 1
    assert torch.equal(graph_codes, eager_codes)
    assert torch.equal(graph_embeds, eager_embeds)


@pytest.mark.accelerator
def test_startup_capture_builds_the_ladder_for_both_sampled_signatures():
    device = torch.device("cuda")
    talker = _build_talker(device)
    all_sampled = ("sampled", 8, False, False, False)
    mixed = ("sampled", 8, False, False, True)
    expected_keys = {(bucket, *all_sampled) for bucket in BUCKETS} | {
        (bucket, *mixed) for bucket in BUCKETS if bucket >= 2
    }

    assert talker.capture_predictor_graphs(do_sample=True, top_k=5, top_p=1.0) == len(
        expected_keys
    )
    assert set(talker._predictor_graphs) == expected_keys
    assert talker._predictor_graph_startup_count == len(expected_keys)
    assert talker.capture_predictor_graphs(do_sample=True, top_k=5, top_p=1.0) == 0
    assert talker._predictor_graph_startup_count == len(expected_keys)

    for batch_size in BUCKETS:
        talker.prepare_decode_buffers(_uniform_requests(batch_size, top_k=5))
        layer0, hidden, positions = _step_inputs(batch_size, device)
        eager_codes, eager_embeds = _run_eager(talker, layer0, hidden, positions)
        graph_codes, graph_embeds = _run_forward(talker, layer0, hidden, positions)
        assert torch.equal(graph_codes, eager_codes)
        assert torch.equal(graph_embeds, eager_embeds)

    talker.prepare_decode_buffers([_request(top_k=5), _request(dosample=False)])
    layer0, hidden, positions = _step_inputs(2, device)
    eager_codes, eager_embeds = _run_eager(talker, layer0, hidden, positions)
    graph_codes, graph_embeds = _run_forward(talker, layer0, hidden, positions)
    assert torch.equal(graph_codes, eager_codes)
    assert torch.equal(graph_embeds, eager_embeds)

    assert len(talker._predictor_graphs) == len(expected_keys)


@pytest.mark.accelerator
def test_startup_set_stays_outside_the_lazy_capture_budget(
    monkeypatch: pytest.MonkeyPatch,
):
    device = torch.device("cuda")
    talker = _build_talker(device)
    monkeypatch.setattr(sglang_model_module, "_PREDICTOR_GRAPH_MAX_LAZY_KEYS", 1)
    startup = talker.capture_predictor_graphs(do_sample=True, top_k=5, top_p=1.0)
    assert startup > 1
    layer0, hidden, positions = _step_inputs(2, device)

    def assert_graph_matches(requests) -> None:
        talker.prepare_decode_buffers(requests)
        eager_codes, eager_embeds = _run_eager(talker, layer0, hidden, positions)
        graph_codes, graph_embeds = _run_forward(talker, layer0, hidden, positions)
        assert torch.equal(graph_codes, eager_codes)
        assert torch.equal(graph_embeds, eager_embeds)

    assert_graph_matches(_uniform_requests(2, top_k=3))
    assert len(talker._predictor_graphs) == startup + 1
    assert talker._predictor_graph_capacity_fallback_count == 0

    assert_graph_matches(_uniform_requests(2, dosample=False))
    assert len(talker._predictor_graphs) == startup + 1
    assert talker._predictor_graph_capacity_fallback_count == 1

    assert_graph_matches(_uniform_requests(2, top_k=5))
    assert_graph_matches([_request(top_k=5), _request(dosample=False)])
    assert len(talker._predictor_graphs) == startup + 1
    assert talker._predictor_graph_capture_count == startup + 1
    assert talker._predictor_graph_capacity_fallback_count == 1


@pytest.mark.accelerator
def test_startup_capture_failure_raises_and_restores_state(
    monkeypatch: pytest.MonkeyPatch,
):
    device = torch.device("cuda")
    talker = _build_talker(device)
    real_forward = Qwen3TTSTalker._code_predictor_forward_incremental

    def _boom_forward(self, *args, **kwargs):
        if torch.cuda.current_stream(device) != torch.cuda.default_stream(device):
            raise RuntimeError("simulated capture failure")
        return real_forward(self, *args, **kwargs)

    monkeypatch.setattr(
        Qwen3TTSTalker, "_code_predictor_forward_incremental", _boom_forward
    )

    with pytest.raises(RuntimeError, match="simulated capture failure"):
        talker.capture_predictor_graphs(do_sample=True, top_k=8, top_p=1.0)

    assert not talker._predictor_graphs
    assert talker._sub_batch_size == 0
    assert gc.isenabled()


def test_signature_rule_is_shared_by_batch_and_startup_paths(
    monkeypatch: pytest.MonkeyPatch,
):
    talker = _build_talker(torch.device("cpu"))
    startup_keys: list[tuple] = []

    def _record_capture(self, bucket_size, signature):
        startup_keys.append((bucket_size, *signature))
        return object()

    monkeypatch.setattr(Qwen3TTSTalker, "_capture_predictor_graph", _record_capture)
    cases = [
        (True, 5, 1.0),
        (True, 5, 0.9),
        (True, 0, 1.0),
        (True, 3, 0.5),
        (True, PRED_VOCAB, 1.0),
        (True, 4, 1.0),
        (True, 9, 1.0),
        (False, 5, 1.0),
    ]
    for dosample, top_k, top_p in cases:
        talker.prepare_decode_buffers(
            _uniform_requests(3, dosample=dosample, top_k=top_k, top_p=top_p)
        )
        if talker._sub_has_sampled_rows:
            batch_terms = (
                "sampled",
                talker._sub_sampled_max_top_k,
                talker._sub_sampled_has_top_p,
                talker._sub_sampled_has_unbounded_top_k,
            )
            expected = {batch_terms + (mixed,) for mixed in (False, True)}
        else:
            expected = {("argmax", 0, False, False, False)}
        talker._predictor_graphs.clear()
        startup_keys.clear()
        talker.capture_predictor_graphs(do_sample=dosample, top_k=top_k, top_p=top_p)
        assert {key[1:] for key in startup_keys} == expected, (dosample, top_k, top_p)

    talker.prepare_decode_buffers([_request(dosample=True), _request(dosample=False)])
    assert talker._sub_has_argmax_rows is True
    assert talker._sub_has_sampled_rows is True


@pytest.mark.accelerator
def test_graph_object_holds_no_reference_to_the_talker():
    device = torch.device("cuda")
    talker = _build_talker(device)
    talker.prepare_decode_buffers(_uniform_requests(2))
    layer0, hidden, positions = _step_inputs(2, device)
    _run_forward(talker, layer0, hidden, positions)
    (graph,) = talker._predictor_graphs.values()

    assert all(value is not talker for value in vars(graph).values())
    assert talker not in gc.get_referents(graph)

    graph_ref = weakref.ref(graph)
    talker._predictor_graphs.clear()
    del graph
    assert graph_ref() is None


@pytest.mark.accelerator
@pytest.mark.parametrize(
    "sglang_gemm_override",
    [
        ("is_batch_invariant_mode_enabled", lambda: True),
        ("get_bf16_gemm_backend", lambda: Bf16GemmBackend.CUTEDSL),
    ],
    ids=["batch-invariant", "optimized-backend"],
)
def test_sglang_gemm_overrides_keep_the_eager_gemm_on_both_paths(
    monkeypatch: pytest.MonkeyPatch, sglang_gemm_override
):
    device = torch.device("cuda")
    talker = _build_talker(device)
    monkeypatch.setattr(sglang_model_module, *sglang_gemm_override)
    original_addmm = torch.addmm
    calls = []

    def _record_addmm(*args, **kwargs):
        calls.append(None)
        return original_addmm(*args, **kwargs)

    monkeypatch.setattr(torch, "addmm", _record_addmm)
    talker.prepare_decode_buffers(_uniform_requests(2))
    layer0, hidden, positions = _step_inputs(2, device)

    eager_codes, eager_embeds = _run_eager(talker, layer0, hidden, positions)
    graph_codes, graph_embeds = _run_forward(talker, layer0, hidden, positions)

    assert talker._predictor_graphs
    assert not calls
    assert torch.equal(graph_codes, eager_codes)
    assert torch.equal(graph_embeds, eager_embeds)


ROPE_HEAD_DIM = 128
ROPE_NUM_HEADS = 16
ROPE_NUM_KV_HEADS = 8
ROPE_HIDDEN = 1024
ROPE_PREDICTOR_LEN = 17


def _rope_store_talker(device: torch.device, *, stores: bool) -> Qwen3TTSTalker:
    predictor_len = ROPE_PREDICTOR_LEN
    talker = object.__new__(Qwen3TTSTalker)
    positions = torch.arange(predictor_len, device=device, dtype=torch.long)
    talker._predictor_position_rows = (
        positions[:, None].expand(predictor_len, MAX_BS).contiguous()
    )
    talker._predictor_k_cache = torch.zeros(
        1,
        MAX_BS,
        predictor_len,
        ROPE_NUM_KV_HEADS,
        ROPE_HEAD_DIM,
        device=device,
        dtype=DTYPE,
    )
    talker._predictor_v_cache = torch.zeros_like(talker._predictor_k_cache)
    talker._predictor_k_rows = [
        layer.view(MAX_BS * predictor_len, -1) for layer in talker._predictor_k_cache
    ]
    talker._predictor_v_rows = [
        layer.view(MAX_BS * predictor_len, -1) for layer in talker._predictor_v_cache
    ]
    talker._predictor_cache_slots = (
        torch.arange(MAX_BS, device=device, dtype=torch.long)[None, :] * predictor_len
        + positions[:, None]
    ).contiguous()
    talker._predictor_rope_stores_kv = stores
    return talker


def _rope_copy_reference(
    attn: SimpleNamespace,
    hidden: torch.Tensor,
    positions: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_len: int,
) -> torch.Tensor:
    """Plain RoPE followed by the former [batch, head, slot, dim] cache writes."""
    batch_size = hidden.shape[0]
    qkv, _ = attn.qkv_proj(hidden.reshape(batch_size, -1))
    q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
    q, k = apply_qk_norm(
        q, k, attn.q_norm, attn.k_norm, attn.head_dim, alt_stream=attn.alt_stream
    )
    q, k = attn.rotary_emb(positions, q, k, fused_set_kv_buffer_arg=None)
    k_cache[:batch_size, :, cache_len : cache_len + 1].copy_(
        k.reshape(batch_size, 1, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
    )
    v_cache[:batch_size, :, cache_len : cache_len + 1].copy_(
        v.reshape(batch_size, 1, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
    )
    output = torch.nn.functional.scaled_dot_product_attention(
        q.reshape(batch_size, 1, attn.num_heads, attn.head_dim).transpose(1, 2),
        k_cache[:batch_size, :, : cache_len + 1],
        v_cache[:batch_size, :, : cache_len + 1],
        is_causal=False,
        enable_gqa=True,
    )
    return output.transpose(1, 2).reshape(batch_size, -1)


@pytest.fixture(params=["cuda", "torch"])
def predictor_rope_dispatch(request: pytest.FixtureRequest) -> Iterator[str]:
    from sglang.srt.model_executor.cuda_graph_config import (
        Backend,
        CudaGraphConfig,
        PhaseConfig,
    )
    from sglang.srt.runtime_context import get_context

    mode = request.param
    with get_context().override_server_args(
        cuda_graph_config=CudaGraphConfig(
            prefill=PhaseConfig(backend=Backend.DISABLED)
        ),
    ):
        previous_backend = get_fused_op_backend()
        try:
            set_fused_op_backend(KernelBackend.TORCH if mode == "torch" else None)
            yield mode
        finally:
            set_fused_op_backend(previous_backend)


@pytest.mark.accelerator
@pytest.mark.parametrize("batch_size", [1, 16])
def test_rope_store_writes_the_cache_the_copy_path_writes(
    batch_size: int,
    predictor_rope_dispatch: str,
    monkeypatch: pytest.MonkeyPatch,
):
    """Compare cache rows and attention against the former layout, then replay
    the fused path with fresh inputs to check that captured stores overwrite."""
    device = torch.device("cuda")
    torch.manual_seed(11)
    # Other tests isolate the graph machinery with a stub; this test covers
    # the actual normalized Q/K tensors handed to the upstream rotary.
    monkeypatch.setattr(sglang_model_module, "apply_qk_norm", apply_qk_norm)
    attn = SimpleNamespace(
        q_size=ROPE_NUM_HEADS * ROPE_HEAD_DIM,
        kv_size=ROPE_NUM_KV_HEADS * ROPE_HEAD_DIM,
        num_heads=ROPE_NUM_HEADS,
        num_kv_heads=ROPE_NUM_KV_HEADS,
        head_dim=ROPE_HEAD_DIM,
        q_norm=RMSNorm(ROPE_HEAD_DIM, eps=1e-6).to(device, DTYPE),
        k_norm=RMSNorm(ROPE_HEAD_DIM, eps=1e-6).to(device, DTYPE),
        alt_stream=None,
        qkv_proj=_TupleLinear(
            ROPE_HIDDEN, (ROPE_NUM_HEADS + 2 * ROPE_NUM_KV_HEADS) * ROPE_HEAD_DIM
        ).to(device, DTYPE),
        # A fresh rotary resolves this fixture's dispatch instead of reusing
        # get_rope's process-wide cache from another parameterized case.
        rotary_emb=RotaryEmbedding(
            ROPE_HEAD_DIM, ROPE_HEAD_DIM, 64, 10000, True, DTYPE
        ).to(device),
        compatible_with_fused_kv_buffer=True,
    )
    stores = Qwen3TTSTalker._resolve_predictor_rope_store(attn, device=device)
    stored = _rope_store_talker(device, stores=stores)
    copied = _rope_store_talker(device, stores=False)
    # Allocate the old layout independently, not as another view of the new cache.
    reference_k = torch.zeros(
        MAX_BS,
        ROPE_NUM_KV_HEADS,
        ROPE_PREDICTOR_LEN,
        ROPE_HEAD_DIM,
        device=device,
        dtype=DTYPE,
    )
    reference_v = torch.zeros_like(reference_k)
    hidden_steps = torch.randn(
        ROPE_PREDICTOR_LEN, batch_size, 1, ROPE_HIDDEN, device=device, dtype=DTYPE
    )

    def run_attention(talker: Qwen3TTSTalker) -> torch.Tensor:
        return torch.stack(
            [
                talker._predictor_cached_self_attention(
                    layer_idx=0,
                    attn=attn,
                    hidden_states=hidden_steps[slot],
                    positions=talker._predictor_position_rows[slot, :batch_size],
                    batch_size=batch_size,
                    cache_len=slot,
                )
                for slot in range(ROPE_PREDICTOR_LEN)
            ]
        )

    def run_reference() -> torch.Tensor:
        return torch.stack(
            [
                _rope_copy_reference(
                    attn,
                    hidden_steps[slot],
                    stored._predictor_position_rows[slot, :batch_size],
                    reference_k,
                    reference_v,
                    slot,
                )
                for slot in range(ROPE_PREDICTOR_LEN)
            ]
        )

    def assert_matches_reference(output: torch.Tensor) -> None:
        expected = run_reference()
        torch.cuda.synchronize()
        assert torch.equal(output, expected)
        assert torch.equal(stored._predictor_k_cache[0].transpose(1, 2), reference_k)
        assert torch.equal(stored._predictor_v_cache[0].transpose(1, 2), reference_v)

    with torch.no_grad():
        output = run_attention(stored)
        assert torch.equal(output, run_attention(copied))
        assert torch.equal(stored._predictor_k_cache, copied._predictor_k_cache)
        assert torch.equal(stored._predictor_v_cache, copied._predictor_v_cache)
        assert_matches_reference(output)
        assert stores == (predictor_rope_dispatch == "cuda")
        if not stores:
            return

        stream = torch.cuda.Stream(device=device)
        current_stream = torch.cuda.current_stream(device)
        stream.wait_stream(current_stream)
        with torch.cuda.stream(stream):
            for _ in range(2):
                run_attention(stored)
        current_stream.wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            replay_output = run_attention(stored)

        # The same graph must consume changed inputs and replace the prior frame.
        for _ in range(2):
            hidden_steps.normal_()
            graph.replay()
            assert_matches_reference(replay_output)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
