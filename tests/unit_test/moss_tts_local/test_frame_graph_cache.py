# SPDX-License-Identifier: Apache-2.0
"""Frame-decode CUDA-graph cache adoption and the persistent cross-step state
contract for MOSS-TTS Local.

The GPU cases drive the real capture path (``init_frame_decode_graphs`` /
``decode_frame_graphed``) on a miniature model, so bit-identity of the captured
region is gated rather than assumed. The CPU cases cover the KV reserve/freeze
contract and the reassignment detector.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from sglang_omni.cuda_graph import PersistentStateError, PersistentStateRegistry
from sglang_omni.models.moss_tts_local.local_transformer import MossTTSLocalTransformer
from sglang_omni.models.moss_tts_local.sglang_model import (
    MOSS_LOCAL_FRAME_GRAPH_ENV,
    MossTTSLocalSGLangModel,
)

_HAS_CUDA = torch.cuda.is_available()

HIDDEN = 32
N_VQ = 4
AUDIO_VOCAB = 16
TEXT_VOCAB = 8
MAX_BS = 4
BUCKETS = [1, 2, 4]


def _transformer(*, num_layers: int = 1) -> MossTTSLocalTransformer:
    return MossTTSLocalTransformer(
        hidden_size=HIDDEN,
        num_heads=4,
        inner_size=2 * HIDDEN,
        num_layers=num_layers,
        max_positions=N_VQ + 1,
        rope_base=1_000_000.0,
    )


def _mini_model(device: torch.device) -> MossTTSLocalSGLangModel:
    """A MOSS-TTS Local model carrying only what the frame-decode path reads.

    Building the real model needs a checkpoint and an SGLang backbone; the frame
    decode only touches the local transformer, the embedding tables, the binary
    head and the decode staging table, so those are supplied directly.
    """
    model = object.__new__(MossTTSLocalSGLangModel)
    nn.Module.__init__(model)
    model.hidden_size = HIDDEN
    model.n_vq = N_VQ
    model.config = SimpleNamespace(
        audio_vocab_size=AUDIO_VOCAB,
        audio_assistant_slot_token_id=1,
        audio_end_token_id=2,
        channels=N_VQ + 1,
    )
    model.embedding_list = nn.ModuleList(
        [nn.Embedding(TEXT_VOCAB, HIDDEN)]
        + [nn.Embedding(AUDIO_VOCAB + 1, HIDDEN) for _ in range(N_VQ)]
    )
    model.local_text_lm_head = nn.Linear(HIDDEN, 2, bias=False)
    model.local_transformer = _transformer()
    model._decode_input_embedding = nn.Embedding(MAX_BS, HIDDEN)
    model.to(device=device, dtype=torch.bfloat16)
    model._compiled_frame_sampler = None
    model._frame_compile_configured = True
    model._frame_graph_cache = None
    model._frame_graph_buckets = ()
    model._persistent_state = PersistentStateRegistry()
    # torch.compile of the sampler costs minutes and is not what these gate.
    model._ensure_frame_sampler_compile = lambda: None
    return model


def _frame_inputs(batch_size: int, device: torch.device, *, seed: int = 0) -> dict:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    hidden = torch.randn(
        batch_size, HIDDEN, generator=generator, dtype=torch.float32
    ).to(device=device, dtype=torch.bfloat16)
    return {
        "hidden_states": hidden,
        "text_temperature": torch.full(
            (batch_size,), 0.9, device=device, dtype=torch.float32
        ),
        "text_top_p": torch.full(
            (batch_size,), 0.8, device=device, dtype=torch.float32
        ),
        "text_top_k": torch.full((batch_size,), 2, device=device, dtype=torch.long),
        "audio_temperature": torch.full(
            (batch_size,), 1.1, device=device, dtype=torch.float32
        ),
        "audio_top_p": torch.full(
            (batch_size,), 0.95, device=device, dtype=torch.float32
        ),
        "audio_top_k": torch.full((batch_size,), 5, device=device, dtype=torch.long),
        "seeds": torch.arange(batch_size, device=device, dtype=torch.long) * 977 + 3,
        "base_positions": torch.full(
            (batch_size,), 7 * (N_VQ + 1), device=device, dtype=torch.long
        ),
    }


# ---------------------------------------------------------------------------
# CPU: KV reserve/freeze contract and the reassignment detector
# ---------------------------------------------------------------------------


def test_reserve_and_freeze_kv_cache_sizes_then_blocks_growth():
    module = _transformer()
    module.reserve_and_freeze_kv_cache(4, torch.device("cpu"), torch.float32)
    assert module._kv_capacity >= 4
    module.step(torch.zeros(4, HIDDEN), 0)
    with pytest.raises(RuntimeError, match="frozen"):
        module.step(torch.zeros(8, HIDDEN), 0)


def test_local_transformer_declares_every_kv_buffer():
    module = _transformer(num_layers=2)
    module.reserve_and_freeze_kv_cache(2, torch.device("cpu"), torch.float32)
    registry = PersistentStateRegistry()
    module.register_persistent_state(registry)
    assert len(registry.declared_names()) == 4


def test_declared_kv_buffer_survives_steps_but_trips_on_reassignment():
    """The declared buffers must keep their addresses across steps; swapping in
    a fresh (value-identical) cache is exactly the failure a graph replay cannot
    see, so the registry has to catch it."""
    module = _transformer()
    module.reserve_and_freeze_kv_cache(2, torch.device("cpu"), torch.float32)
    registry = PersistentStateRegistry()
    module.register_persistent_state(registry)
    registry.snapshot_addresses()

    for position in range(3):
        module.step(torch.zeros(2, HIDDEN), position)
    registry.assert_addresses_stable()

    module._kv_cache = [(k.clone(), v.clone()) for k, v in module._kv_cache]
    with pytest.raises(PersistentStateError, match="moved"):
        registry.assert_addresses_stable()


# ---------------------------------------------------------------------------
# GPU: real capture through the shared keyed cache
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.skipif(not _HAS_CUDA, reason="needs CUDA")
@pytest.mark.parametrize("bucket", BUCKETS)
def test_frame_graph_replay_is_bit_identical_to_eager(bucket):
    """Acceptance gate for the captured region: replaying the graph for a full
    bucket must reproduce the eager branchless frame decode bit-for-bit."""
    device = torch.device("cuda")
    torch.manual_seed(17)
    model = _mini_model(device)
    model.init_frame_decode_graphs(BUCKETS)
    assert model.frame_graph_max_bs == max(BUCKETS)

    inputs = _frame_inputs(bucket, device, seed=bucket)
    stop_choice, codes, feedback = model.decode_frame_graphed(
        inputs["hidden_states"],
        **{k: v for k, v in inputs.items() if k != "hidden_states"},
    )
    graphed = (stop_choice.clone(), codes.clone(), feedback.clone())
    eager = model._decode_frame_graphable(**inputs)

    assert torch.equal(graphed[0], eager[0]), "stop choice diverged"
    assert torch.equal(graphed[1], eager[1]), "codes diverged"
    assert torch.equal(graphed[2], eager[2]), (
        "feedback embedding diverged: max|delta|="
        f"{(graphed[2].float() - eager[2].float()).abs().max().item():.3e}"
    )


@pytest.mark.gpu
@pytest.mark.skipif(not _HAS_CUDA, reason="needs CUDA")
def test_interleaved_bucket_replays_stay_bit_identical():
    """Buckets share one graph pool, so a later capture's retained outputs can
    sit on an earlier capture's freed intermediates. Serving survives that by
    consuming each replay before the next one; this drives the buckets in a
    scrambled order under exactly that discipline."""
    device = torch.device("cuda")
    torch.manual_seed(29)
    model = _mini_model(device)
    model.init_frame_decode_graphs(BUCKETS)

    for step, bucket in enumerate([4, 1, 2, 4, 1, 2, 4]):
        inputs = _frame_inputs(bucket, device, seed=100 + step)
        stop_choice, codes, feedback = model.decode_frame_graphed(
            inputs["hidden_states"],
            **{k: v for k, v in inputs.items() if k != "hidden_states"},
        )
        graphed = (stop_choice.clone(), codes.clone(), feedback.clone())
        eager = model._decode_frame_graphable(**inputs)
        for got, want, what in zip(graphed, eager, ("stop", "codes", "feedback")):
            assert torch.equal(got, want), f"{what} diverged at step {step} bs={bucket}"


@pytest.mark.gpu
@pytest.mark.skipif(not _HAS_CUDA, reason="needs CUDA")
def test_env_kill_switch_skips_capture_and_leaves_kv_growable(monkeypatch):
    monkeypatch.setenv(MOSS_LOCAL_FRAME_GRAPH_ENV, "0")
    model = _mini_model(torch.device("cuda"))
    model.init_frame_decode_graphs(BUCKETS)
    assert model.frame_graph_max_bs == 0
    assert not model.local_transformer._kv_frozen


@pytest.mark.gpu
@pytest.mark.skipif(not _HAS_CUDA, reason="needs CUDA")
def test_capture_failure_falls_back_to_eager_instead_of_raising():
    model = _mini_model(torch.device("cuda"))

    def boom(bucket, device):
        raise RuntimeError("simulated capture OOM")

    model._capture_frame_graph = boom
    model.init_frame_decode_graphs(BUCKETS)
    assert model.frame_graph_max_bs == 0
    assert len(model._frame_graph_cache.disabled_keys) == len(BUCKETS)


@pytest.mark.gpu
@pytest.mark.skipif(not _HAS_CUDA, reason="needs CUDA")
def test_repeated_capture_failures_blow_the_global_fuse():
    model = _mini_model(torch.device("cuda"))
    many_buckets = list(range(1, 12))

    def boom(bucket, device):
        raise RuntimeError("simulated capture OOM")

    model._capture_frame_graph = boom
    model.init_frame_decode_graphs(many_buckets)
    assert not model._frame_graph_cache.enabled


@pytest.mark.gpu
@pytest.mark.skipif(not _HAS_CUDA, reason="needs CUDA")
def test_frame_graphs_share_one_memory_pool(monkeypatch):
    model = _mini_model(torch.device("cuda"))
    pools = []
    original = torch.cuda.graph

    class _RecordingGraph(original):
        def __init__(self, cuda_graph, pool=None, **kwargs):
            pools.append(pool)
            super().__init__(cuda_graph, pool=pool, **kwargs)

    monkeypatch.setattr(torch.cuda, "graph", _RecordingGraph)
    model.init_frame_decode_graphs(BUCKETS)
    assert len(pools) == len(BUCKETS)
    assert all(pool is not None for pool in pools)
    assert len(set(map(id, pools))) == 1


@pytest.mark.gpu
@pytest.mark.skipif(not _HAS_CUDA, reason="needs CUDA")
def test_captured_graphs_declare_their_cross_step_state():
    model = _mini_model(torch.device("cuda"))
    model.init_frame_decode_graphs(BUCKETS)
    names = model._persistent_state.declared_names()
    assert any("kv" in name for name in names)
    assert any("decode_input_embedding" in name for name in names)
    model.verify_persistent_state()


@pytest.mark.gpu
@pytest.mark.skipif(not _HAS_CUDA, reason="needs CUDA")
def test_reassigned_kv_cache_is_caught():
    device = torch.device("cuda")
    model = _mini_model(device)
    model.init_frame_decode_graphs(BUCKETS)
    model.verify_persistent_state()

    model.local_transformer._kv_frozen = False
    model.local_transformer._kv_cache = [
        (k.clone(), v.clone()) for k, v in model.local_transformer._kv_cache
    ]
    with pytest.raises(PersistentStateError, match="moved"):
        model.verify_persistent_state()


@pytest.mark.gpu
@pytest.mark.skipif(not _HAS_CUDA, reason="needs CUDA")
def test_reassigned_decode_embedding_goes_stale_and_is_caught():
    """The feedback staging table is read inside the backbone decode graph, so
    it is the buffer whose reassignment reproduces the streaming-cache failure:
    the replay keeps reading the old address while the writer fills a new one.
    Written in place the replay tracks eager; reassigned it does not, and the
    registry must fire either way before the divergence reaches audio."""
    device = torch.device("cuda")
    model = _mini_model(device)
    model.init_frame_decode_graphs(BUCKETS)

    row_ids = torch.arange(MAX_BS, device=device, dtype=torch.long)
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            model._decode_input_embedding(row_ids)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        replayed = model._decode_input_embedding(row_ids)

    fresh = torch.randn(MAX_BS, HIDDEN, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        model._decode_input_embedding.weight.copy_(fresh)
    graph.replay()
    assert torch.equal(replayed, fresh)
    model.verify_persistent_state()

    with torch.no_grad():
        model._decode_input_embedding.weight = nn.Parameter(
            torch.zeros(MAX_BS, HIDDEN, device=device, dtype=torch.bfloat16),
            requires_grad=False,
        )
        model._decode_input_embedding.weight.copy_(fresh * -1)
    graph.replay()
    assert torch.equal(
        replayed, fresh
    ), "replay must still read the stale buffer, else this gate proves nothing"
    with pytest.raises(PersistentStateError, match="moved"):
        model.verify_persistent_state()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
