# SPDX-License-Identifier: Apache-2.0
"""Bit-identity and capture-safety gates for the Voxtral-TTS frame CUDA graph.

``decode_one_frame`` runs a 7 step flow-matching ODE whose body is a CFG
batch-doubled 3 layer acoustic stack; the whole per-frame chain is captured as
one CUDA graph per batch bucket. Graphed codes must equal the eager chain
bit-for-bit (``torch.equal``) at bucket-exact and padded batch sizes, the
per-frame noise must stay on the eager host RNG stream, and dispatch/replay must
never read back to the host.
"""

from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path

import pytest
import torch

import sglang_omni.models.voxtral_tts.acoustic_transformer as at_module
from sglang_omni.cuda_graph import env_graph_enabled
from sglang_omni.models.voxtral_tts.acoustic_transformer import (
    AudioSpecialTokens,
    FlowMatchingAudioTransformer,
)

_HAS_CUDA = torch.cuda.is_available()

DTYPE = torch.bfloat16
INPUT_DIM = 16
N_ACOUSTIC = 4
MAX_BS = 8
BUCKETS = (1, 2, 4, 8)

_AUDIO_ARGS = {
    "semantic_codebook_size": 32,
    "acoustic_codebook_size": 24,
    "n_acoustic_codebook": N_ACOUSTIC,
    "acoustic_transformer_args": {
        "input_dim": INPUT_DIM,
        "dim": 16,
        "n_layers": 3,
        "head_dim": 8,
        "hidden_dim": 32,
        "n_heads": 4,
        "n_kv_heads": 2,
        "use_biases": False,
        "norm_eps": 1e-5,
        "sigma": 1e-5,
    },
}

_END_AUDIO = AudioSpecialTokens.id(AudioSpecialTokens.end_audio)


def _build_transformer(device: torch.device) -> FlowMatchingAudioTransformer:
    torch.manual_seed(7)
    model = FlowMatchingAudioTransformer(deepcopy(_AUDIO_ARGS))
    return model.to(device=device, dtype=DTYPE).eval()


def _frame_inputs(batch_size: int, device: torch.device, *, masked_rows=()):
    generator = torch.Generator(device="cpu").manual_seed(31 * batch_size + 5)
    semantic = torch.randint(
        len(AudioSpecialTokens),
        len(AudioSpecialTokens) + 20,
        (batch_size,),
        generator=generator,
        dtype=torch.long,
    )
    for row in masked_rows:
        semantic[row] = _END_AUDIO
    llm_hidden = torch.randn(
        batch_size, INPUT_DIM, generator=generator, dtype=torch.float32
    )
    return semantic.to(device), llm_hidden.to(device=device, dtype=DTYPE)


def _run(model, semantic, llm_hidden, *, seed: int = 1234, steps: int = 1):
    """Run ``steps`` consecutive frames off one host RNG seed."""
    torch.manual_seed(seed)
    outputs = []
    with torch.no_grad():
        for _ in range(steps):
            outputs.append(model.decode_one_frame(semantic, llm_hidden).clone())
    if _HAS_CUDA:
        torch.cuda.synchronize()
    return outputs


def _eager(model, semantic, llm_hidden, **kwargs):
    cache = model._frame_graph_cache
    model._frame_graph_cache = None
    try:
        return _run(model, semantic, llm_hidden, **kwargs)
    finally:
        model._frame_graph_cache = cache


def _graphed_model(device: torch.device) -> FlowMatchingAudioTransformer:
    model = _build_transformer(device)
    model.enable_frame_graph(max_batch_size=MAX_BS, cuda_graph_bs=BUCKETS)
    model._frame_graph_runtime_checked = True
    return model


def _reference_frame(model, semantic_code, llm_hidden):
    """The pre-graph chain, transcribed, as an absolute reference.

    The graph-vs-eager comparisons alone would not catch a refactor that moved
    both paths together, so the ODE body and the host RNG draw are pinned here.
    """
    B = semantic_code.shape[0]
    should_decode = semantic_code != model._end_audio_token_id

    x_0 = torch.randn(B, model.model_args.n_acoustic_codebook).to(
        dtype=llm_hidden.dtype, device=llm_hidden.device
    )
    x_0 = model._noise_scale * x_0

    timesteps = model._timesteps.to(dtype=llm_hidden.dtype)
    llm_hidden_zero = torch.zeros_like(llm_hidden)

    sampled = x_0
    for i in range(len(timesteps) - 1):
        t = timesteps[i]
        dt = timesteps[i + 1] - timesteps[i]
        t_emb = model.time_embedding(t.view(-1, 1).repeat(B, 1)).to(llm_hidden.dtype)
        x_batched = torch.cat([sampled, sampled], dim=0)
        llm_batched = torch.cat([llm_hidden, llm_hidden_zero], dim=0)
        t_emb_batched = torch.cat([t_emb, t_emb], dim=0)
        v_all = model._predict_velocity(
            x_t=x_batched, llm_output=llm_batched, t_emb=t_emb_batched
        )
        v_t, uncond_v_t = v_all[:B], v_all[B:]
        v_t = model._cfg_alpha * v_t + (1 - model._cfg_alpha) * uncond_v_t
        sampled = sampled + v_t * dt

    sampled = torch.clamp(sampled, -1, 1)
    quantized_levels = ((sampled + 1) / 2) * (model.acoustic_embeddings_levels - 1)
    output_codes = quantized_levels.round().long()
    output_codes[~should_decode] = model._empty_audio_token_id
    return output_codes + len(AudioSpecialTokens)


# -- reference equivalence -------------------------------------------------


@pytest.mark.skipif(not _HAS_CUDA, reason="frame CUDA graph needs CUDA")
@pytest.mark.parametrize("batch_size", [1, 3, 8])
def test_graph_and_eager_match_the_pre_graph_chain(batch_size: int):
    device = torch.device("cuda")
    model = _graphed_model(device)
    semantic, llm_hidden = _frame_inputs(batch_size, device, masked_rows=(0,))

    torch.manual_seed(777)
    with torch.no_grad():
        reference = _reference_frame(model, semantic, llm_hidden).clone()

    (eager,) = _eager(model, semantic, llm_hidden, seed=777)
    (graphed,) = _run(model, semantic, llm_hidden, seed=777)

    assert torch.equal(eager, reference), "refactored eager chain drifted"
    assert torch.equal(graphed, reference), "graphed chain drifted"


@pytest.mark.skipif(not _HAS_CUDA, reason="frame CUDA graph needs CUDA")
@pytest.mark.parametrize("batch_size", [1, 3, 8])
def test_frame_noise_matches_the_pre_graph_host_rng_draw(batch_size: int):
    """The draw must consume the same host RNG values, in the same order."""
    device = torch.device("cuda")
    model = _build_transformer(device)
    llm_hidden = torch.zeros(batch_size, INPUT_DIM, dtype=DTYPE, device=device)

    torch.manual_seed(4321)
    # Note: (Jiaxin Deng) reference draws on the CPU generator explicitly; an
    # ambient default-device switch must not silently move the draw to CUDA.
    reference = model._noise_scale * torch.randn(
        batch_size, N_ACOUSTIC, device="cpu"
    ).to(dtype=DTYPE, device=device)
    torch.manual_seed(4321)
    drawn = model._draw_frame_noise(batch_size, llm_hidden)

    assert torch.equal(drawn, reference)


# -- bit identity ----------------------------------------------------------


@pytest.mark.skipif(not _HAS_CUDA, reason="frame CUDA graph needs CUDA")
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
def test_graph_bit_identity(batch_size: int):
    device = torch.device("cuda")
    model = _graphed_model(device)
    semantic, llm_hidden = _frame_inputs(batch_size, device)

    (eager,) = _eager(model, semantic, llm_hidden)
    (graphed,) = _run(model, semantic, llm_hidden)

    assert model._frame_graph_cache.graphs, "no frame graph captured"
    assert graphed.shape == eager.shape
    assert torch.equal(graphed, eager), (
        f"codes not bit-identical (bs={batch_size}): "
        f"mismatches={(graphed != eager).sum().item()}"
    )


@pytest.mark.skipif(not _HAS_CUDA, reason="frame CUDA graph needs CUDA")
@pytest.mark.parametrize("batch_size", [2, 4, 8])
def test_graph_bit_identity_partial_should_decode_mask(batch_size: int):
    """Rows at the end-audio token must be forced to the empty-audio id."""
    device = torch.device("cuda")
    model = _graphed_model(device)
    masked = tuple(range(0, batch_size, 2))
    semantic, llm_hidden = _frame_inputs(batch_size, device, masked_rows=masked)

    (eager,) = _eager(model, semantic, llm_hidden)
    (graphed,) = _run(model, semantic, llm_hidden)

    empty_code = AudioSpecialTokens.id(AudioSpecialTokens.empty_audio) + len(
        AudioSpecialTokens
    )
    assert (eager[list(masked)] == empty_code).all(), "eager mask precondition"
    assert torch.equal(graphed, eager)


@pytest.mark.skipif(not _HAS_CUDA, reason="frame CUDA graph needs CUDA")
@pytest.mark.parametrize("batch_size", [3, 5])
def test_graph_padded_bucket_bit_identity(batch_size: int):
    """A live batch smaller than its bucket replays with padded rows."""
    device = torch.device("cuda")
    model = _graphed_model(device)
    semantic, llm_hidden = _frame_inputs(batch_size, device, masked_rows=(1,))

    (eager,) = _eager(model, semantic, llm_hidden)
    (graphed,) = _run(model, semantic, llm_hidden)

    bucket = 4 if batch_size == 3 else 8
    assert any(key[0] == bucket for key in model._frame_graph_cache.graphs)
    assert not any(key[0] == batch_size for key in model._frame_graph_cache.graphs)
    assert graphed.shape == eager.shape
    assert torch.equal(graphed, eager)


@pytest.mark.skipif(not _HAS_CUDA, reason="frame CUDA graph needs CUDA")
def test_graph_multi_frame_rng_stream_parity():
    """Consecutive frames off one seed must match eager frame for frame."""
    device = torch.device("cuda")
    model = _graphed_model(device)
    semantic, llm_hidden = _frame_inputs(4, device)

    eager = _eager(model, semantic, llm_hidden, steps=4)
    graphed = _run(model, semantic, llm_hidden, steps=4)

    assert (
        len({tuple(frame.flatten().tolist()) for frame in eager}) > 1
    ), "frames must differ across the RNG stream for this test to bite"
    for step, (want, got) in enumerate(zip(eager, graphed)):
        assert torch.equal(got, want), f"frame {step} diverged from eager"
    assert len(model._frame_graph_cache.graphs) == 1


@pytest.mark.skipif(not _HAS_CUDA, reason="frame CUDA graph needs CUDA")
def test_reseeding_reproduces_the_same_frames_on_the_graph_path():
    device = torch.device("cuda")
    model = _graphed_model(device)
    semantic, llm_hidden = _frame_inputs(2, device)

    first = _run(model, semantic, llm_hidden, seed=99, steps=3)
    second = _run(model, semantic, llm_hidden, seed=99, steps=3)

    for step, (a, b) in enumerate(zip(first, second)):
        assert torch.equal(a, b), f"frame {step} not reproducible under one seed"


# -- fallbacks and kill switch --------------------------------------------


@pytest.mark.skipif(not _HAS_CUDA, reason="frame CUDA graph needs CUDA")
def test_batch_above_max_bucket_falls_back_to_eager():
    device = torch.device("cuda")
    model = _graphed_model(device)
    semantic, llm_hidden = _frame_inputs(MAX_BS + 1, device)

    (eager,) = _eager(model, semantic, llm_hidden)
    (graphed,) = _run(model, semantic, llm_hidden)

    assert not model._frame_graph_cache.graphs
    assert torch.equal(graphed, eager)


@pytest.mark.skipif(not _HAS_CUDA, reason="frame CUDA graph needs CUDA")
def test_kill_switch_disables_graph_path():
    device = torch.device("cuda")
    model = _graphed_model(device)
    model._frame_graph_cache.disable("kill switch")
    semantic, llm_hidden = _frame_inputs(2, device)

    (eager,) = _eager(model, semantic, llm_hidden)
    (graphed,) = _run(model, semantic, llm_hidden)

    assert not model._frame_graph_cache.graphs
    assert torch.equal(graphed, eager)


def test_env_switch_parsing(monkeypatch: pytest.MonkeyPatch):
    env = at_module.VOXTRAL_FRAME_GRAPH_ENV
    monkeypatch.delenv(env, raising=False)
    assert env_graph_enabled(env) is True
    for falsy in ("0", "false", "no", "off"):
        monkeypatch.setenv(env, falsy)
        assert env_graph_enabled(env) is False
    monkeypatch.setenv(env, "1")
    assert env_graph_enabled(env) is True


@pytest.mark.skipif(not _HAS_CUDA, reason="frame CUDA graph needs CUDA")
def test_capture_failure_disables_key_and_falls_back(monkeypatch: pytest.MonkeyPatch):
    device = torch.device("cuda")
    model = _graphed_model(device)
    semantic, llm_hidden = _frame_inputs(2, device)
    calls = []

    class _BoomGraph:
        def __init__(self, *args, **kwargs) -> None:
            calls.append(1)
            raise RuntimeError("simulated capture failure")

    monkeypatch.setattr(at_module, "_FrameDecodeGraph", _BoomGraph)

    (eager,) = _eager(model, semantic, llm_hidden)
    (graphed,) = _run(model, semantic, llm_hidden)

    assert len(calls) == 1
    assert model._frame_graph_cache.disabled_keys, "failed key must be disabled"
    assert torch.equal(graphed, eager)

    _run(model, semantic, llm_hidden)
    assert len(calls) == 1, "disabled key must not retry capture"


@pytest.mark.skipif(not _HAS_CUDA, reason="frame CUDA graph needs CUDA")
def test_frame_graph_off_by_default_without_enable():
    """A transformer built outside the serving model stays fully eager."""
    device = torch.device("cuda")
    model = _build_transformer(device)
    semantic, llm_hidden = _frame_inputs(2, device)

    assert model._frame_graph_cache is None
    (out,) = _run(model, semantic, llm_hidden)
    assert out.shape == (2, N_ACOUSTIC)


# -- capture safety --------------------------------------------------------


class _NoHostReadbackTensor(torch.Tensor):
    """Tensor whose host-materialization entry points fail the test."""

    def cpu(self, *args, **kwargs):
        raise RuntimeError("host readback (cpu) on the frame graph path")

    def tolist(self):
        raise RuntimeError("host readback (tolist) on the frame graph path")

    def numpy(self, *args, **kwargs):
        raise RuntimeError("host readback (numpy) on the frame graph path")

    def item(self):
        raise RuntimeError("host readback (item) on the frame graph path")

    def __float__(self):
        raise RuntimeError("host readback (float) on the frame graph path")

    def __int__(self):
        raise RuntimeError("host readback (int) on the frame graph path")

    def __bool__(self):
        raise RuntimeError("host readback (bool) on the frame graph path")

    def __iter__(self):
        raise RuntimeError("host readback (iter) on the frame graph path")

    def to(self, *args, **kwargs):
        if any(str(a) == "cpu" for a in args) or str(kwargs.get("device")) == "cpu":
            raise RuntimeError("host readback (to cpu) on the frame graph path")
        return super().to(*args, **kwargs)


@pytest.mark.skipif(not _HAS_CUDA, reason="frame CUDA graph needs CUDA")
def test_no_host_readback_on_graph_dispatch_and_replay():
    device = torch.device("cuda")
    model = _graphed_model(device)
    semantic, llm_hidden = _frame_inputs(2, device)
    guarded_semantic = semantic.as_subclass(_NoHostReadbackTensor)
    guarded_hidden = llm_hidden.as_subclass(_NoHostReadbackTensor)

    with torch.no_grad():
        model.decode_one_frame(guarded_semantic, guarded_hidden)
        assert model._frame_graph_cache.graphs, "expected graph capture"
        model.decode_one_frame(guarded_semantic, guarded_hidden)
        torch.cuda.synchronize()


@pytest.mark.skipif(not _HAS_CUDA, reason="frame CUDA graph needs CUDA")
def test_no_host_readback_in_eager_chain():
    """The captured body itself must be free of host materialization."""
    device = torch.device("cuda")
    model = _build_transformer(device)
    semantic, llm_hidden = _frame_inputs(2, device)

    with torch.no_grad():
        model.decode_one_frame(
            semantic.as_subclass(_NoHostReadbackTensor),
            llm_hidden.as_subclass(_NoHostReadbackTensor),
        )
        torch.cuda.synchronize()


def test_capture_uses_thread_local_error_mode():
    source = (
        Path(__file__).resolve().parents[3]
        / "sglang_omni"
        / "models"
        / "voxtral_tts"
        / "acoustic_transformer.py"
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
    assert graph_calls, "Voxtral frame CUDA graph capture call not found"
    assert any(
        keyword.arg == "capture_error_mode"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value == "thread_local"
        for call in graph_calls
        for keyword in call.keywords
    )


def test_frame_chain_has_no_boolean_mask_assignment():
    """The captured chain must stay branchless: no data-dependent index_put_."""
    source = (
        Path(__file__).resolve().parents[3]
        / "sglang_omni"
        / "models"
        / "voxtral_tts"
        / "acoustic_transformer.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    chain = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_decode_one_frame_chain"
    )
    subscript_targets = [
        target
        for node in ast.walk(chain)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Subscript)
    ]
    assert (
        not subscript_targets
    ), "boolean-mask assignment is not capturable; use torch.where"


@pytest.mark.skipif(not _HAS_CUDA, reason="frame graph requires CUDA")
def test_frame_noise_is_pinned_to_the_cpu_generator(monkeypatch: pytest.MonkeyPatch):
    """The draw must stay on the host RNG even under a default-device switch.

    A CUDA draw is a different generator, so a seeded run would silently change
    its output; the graph path depends on the two paths sharing one stream.
    """
    device = torch.device("cuda")
    model = _build_transformer(device)
    llm_hidden = torch.zeros(2, INPUT_DIM, dtype=DTYPE, device=device)

    torch.manual_seed(99)
    expected = model._draw_frame_noise(2, llm_hidden)

    torch.set_default_device("cuda")
    try:
        torch.manual_seed(99)
        drawn = model._draw_frame_noise(2, llm_hidden)
    finally:
        torch.set_default_device("cpu")

    assert torch.equal(drawn, expected)
