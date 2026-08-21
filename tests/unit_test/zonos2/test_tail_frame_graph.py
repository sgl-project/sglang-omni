# SPDX-License-Identifier: Apache-2.0
"""Bit-identity and key-sharing gates for the ZONOS2 per-frame tail CUDA graph.

The tail (multi-codebook head GEMM, loop-break mask, per-request sampling, frame
embed, radix hash) is captured once per (decode bucket, sampling signature). The
signature pins only the host branches the capture bakes in: the ladder-quantized
top-k bound and the top-p / min-p passes. Every per-request value rides a device
buffer, so one graph has to serve a heterogeneous batch bit-for-bit.

``sample_tts`` draws with ``torch.multinomial`` on the global RNG, so this file
separates two different properties and never lets one stand in for the other:

* per-frame transform identity: graphed and eager outputs for ONE frame started
  from the same generator state. ``_run_eager`` / ``_run_graph`` reseed on
  purpose, because a shared starting state is what makes the comparison mean
  anything. The draw is row-indexed, so a padded bucket does not shift the live
  rows' draws within a frame.
* global RNG stream accounting across frames, which the reseeding above would
  hide. Those tests never reseed mid-run: they step a graphed and an eager model
  in lockstep off one seed, so any drift in how much of the generator the tail
  consumes compounds into the codes. See the "RNG stream" section.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import sglang_omni.models.zonos2.sglang_model as zonos2_model_module
from sglang_omni.cuda_graph import env_graph_enabled
from sglang_omni.models.zonos2.sglang_model import Zonos2SGLangModel

_HAS_CUDA = torch.cuda.is_available()

N_CODEBOOKS = 9
CODEBOOK_SIZE = 1024
AUDIO_VOCAB = CODEBOOK_SIZE + 2
TEXT_VOCAB = 32
DIM = 64
BUCKETS = (1, 2, 4, 8)
WINDOW = 50
DTYPE = torch.bfloat16
SEED = 20260727

# Heterogeneous batch: distinct temperature / top_k / top_p / min_p /
# repetition_penalty per row, with a greedy row (temperature 0) at index 1 and a
# top-k-disabled row at index 3.
_ROWS: list[dict] = [
    dict(temperature=1.15, top_k=100, top_p=0.0, min_p=0.18, repetition_penalty=1.2),
    dict(temperature=0.0, top_k=106, top_p=0.0, min_p=0.0, repetition_penalty=1.0),
    dict(temperature=0.8, top_k=16, top_p=0.9, min_p=0.05, repetition_penalty=1.5),
    dict(temperature=1.3, top_k=0, top_p=0.95, min_p=0.0, repetition_penalty=1.0),
    dict(temperature=0.7, top_k=8, top_p=0.0, min_p=0.3, repetition_penalty=1.1),
    dict(temperature=1.0, top_k=64, top_p=0.8, min_p=0.0, repetition_penalty=1.3),
    dict(temperature=1.5, top_k=32, top_p=0.0, min_p=0.02, repetition_penalty=1.0),
    dict(temperature=0.9, top_k=50, top_p=0.99, min_p=0.18, repetition_penalty=1.4),
]


def _default_params() -> SimpleNamespace:
    return SimpleNamespace(
        temperature=1.15,
        top_k=106,
        top_p=0.0,
        min_p=0.18,
        repetition_penalty=1.2,
        repetition_window=WINDOW,
        repetition_codebooks=8,
    )


def _build_model(device: torch.device) -> Zonos2SGLangModel:
    """Fake-weight ZONOS2 head/embedder surface: everything the tail touches."""
    torch.manual_seed(11)
    model = object.__new__(Zonos2SGLangModel)
    model.training = False
    model.n_codebooks = N_CODEBOOKS
    model.audio_vocab = AUDIO_VOCAB
    model.frame_width = N_CODEBOOKS + 1
    model.config = SimpleNamespace(
        dim=DIM,
        text_vocab=TEXT_VOCAB,
        codebook_size=CODEBOOK_SIZE,
        loss_softcap=20.0,
        audio_pad_id=CODEBOOK_SIZE + 1,
    )
    # Plain list / tensor: the model is not nn.Module-initialised here, so the
    # Parameter and submodule registration paths are unavailable.
    model.embedders = [
        nn.Embedding(CODEBOOK_SIZE + 2, DIM).to(device, DTYPE)
        for _ in range(N_CODEBOOKS)
    ] + [nn.Embedding(TEXT_VOCAB + 1, DIM).to(device, DTYPE)]
    model.multi_output = (
        torch.randn(AUDIO_VOCAB * N_CODEBOOKS, DIM, device=device) * 0.05
    ).to(DTYPE)
    model._tail_cache = None
    model._tail_window = 0
    model._cg = {}
    return model


def _armed_model(device: torch.device, buckets=BUCKETS) -> Zonos2SGLangModel:
    model = _build_model(device)
    model.capture_tail_graphs(list(buckets), _default_params())
    return model


def _hetero(batch_size: int) -> list[dict]:
    return [_ROWS[i % len(_ROWS)] for i in range(batch_size)]


def _host_flags(rows: list[dict], vocab: int) -> tuple[int, bool, bool]:
    """The runner's eager host flags for this batch."""
    top_k_max = max((r["top_k"] for r in rows if 0 < r["top_k"] < vocab), default=0)
    any_top_p = any(0.0 < r["top_p"] < 1.0 for r in rows)
    any_min_p = any(r["min_p"] > 0.0 for r in rows)
    return top_k_max, any_top_p, any_min_p


def _step_inputs(rows: list[dict], device: torch.device, *, step: int = 0) -> dict:
    bs = len(rows)
    generator = torch.Generator(device="cpu").manual_seed(97 * bs + step)
    hidden = torch.randn(bs, DIM, generator=generator).to(device, DTYPE)
    rep_ids = torch.randint(
        -1, CODEBOOK_SIZE, (bs, N_CODEBOOKS, WINDOW), generator=generator
    ).to(device, torch.long)
    break_mask = torch.zeros(bs, AUDIO_VOCAB, device=device, dtype=torch.float32)
    break_mask[:, (7 * step + 13) % AUDIO_VOCAB] = float("-inf")
    return dict(
        hidden=hidden,
        temperature=torch.tensor([r["temperature"] for r in rows], device=device),
        top_k=torch.tensor([r["top_k"] for r in rows], device=device),
        top_p=torch.tensor([r["top_p"] for r in rows], device=device),
        min_p=torch.tensor([r["min_p"] for r in rows], device=device),
        rep_pen=torch.tensor([r["repetition_penalty"] for r in rows], device=device),
        rep_ids=rep_ids,
        break_mask=break_mask,
    )


def _stage(model: Zonos2SGLangModel, bs: int, inputs: dict) -> None:
    cg = model._cg
    for name in (
        "hidden",
        "temperature",
        "top_k",
        "top_p",
        "min_p",
        "rep_pen",
        "rep_ids",
        "break_mask",
    ):
        cg[name][:bs].copy_(inputs[name])


def _outputs(model: Zonos2SGLangModel, bs: int) -> tuple:
    cg = model._cg
    return (
        cg["codes"][:bs].clone(),
        cg["keys"][:bs].clone(),
        cg["feedback"][:bs].clone(),
    )


def _eager_frame(model: Zonos2SGLangModel, rows: list[dict], inputs: dict) -> tuple:
    """One eager tail frame from the generator's current state (no reseed)."""
    bs = len(rows)
    top_k_max, any_top_p, any_min_p = _host_flags(rows, model.audio_vocab)
    _stage(model, bs, inputs)
    model._tail_compute(
        bs, top_k_max=top_k_max, any_top_p=any_top_p, any_min_p=any_min_p
    )
    torch.cuda.synchronize()
    return _outputs(model, bs)


def _graph_frame(model: Zonos2SGLangModel, rows: list[dict], inputs: dict) -> tuple:
    """One replayed tail frame from the generator's current state (no reseed)."""
    graph = _graph_for(model, rows)
    codes, keys, feedback = model.run_tail_graph(
        graph,
        inputs["hidden"],
        inputs["temperature"],
        inputs["top_k"],
        inputs["top_p"],
        inputs["min_p"],
        inputs["rep_pen"],
        inputs["rep_ids"],
        inputs["break_mask"],
    )
    torch.cuda.synchronize()
    return codes.clone(), keys.clone(), feedback.clone()


def _graph_for(model: Zonos2SGLangModel, rows: list[dict]):
    top_k_max, any_top_p, any_min_p = _host_flags(rows, model.audio_vocab)
    graph = model.tail_graph(
        len(rows),
        top_k_max=top_k_max,
        any_top_p=any_top_p,
        any_min_p=any_min_p,
        window=WINDOW,
    )
    assert graph is not None, "expected a captured tail graph for this batch"
    return graph


def _run_eager(model: Zonos2SGLangModel, rows: list[dict], inputs: dict) -> tuple:
    """The runner's eager branch, reseeded so one frame is comparable to _run_graph."""
    torch.manual_seed(SEED)
    return _eager_frame(model, rows, inputs)


def _run_graph(model: Zonos2SGLangModel, rows: list[dict], inputs: dict) -> tuple:
    bs = len(rows)
    top_k_max, any_top_p, any_min_p = _host_flags(rows, model.audio_vocab)
    graph = model.tail_graph(
        bs,
        top_k_max=top_k_max,
        any_top_p=any_top_p,
        any_min_p=any_min_p,
        window=WINDOW,
    )
    assert graph is not None, "expected a captured tail graph for this batch"
    torch.manual_seed(SEED)
    codes, keys, feedback = model.run_tail_graph(
        graph,
        inputs["hidden"],
        inputs["temperature"],
        inputs["top_k"],
        inputs["top_p"],
        inputs["min_p"],
        inputs["rep_pen"],
        inputs["rep_ids"],
        inputs["break_mask"],
    )
    torch.cuda.synchronize()
    return codes.clone(), keys.clone(), feedback.clone()


def _assert_identical(graph_out: tuple, eager_out: tuple, label: str) -> None:
    names = ("codes", "keys", "feedback")
    for name, got, want in zip(names, graph_out, eager_out):
        assert torch.equal(got, want), (
            f"{name} not bit-identical ({label}): "
            f"mismatched rows={(got != want).any(dim=-1).sum().item() if got.ndim > 1 else (got != want).sum().item()}"
        )


# ---- (a) bit-identity on heterogeneous batches ----


@pytest.mark.skipif(not _HAS_CUDA, reason="tail CUDA graph needs CUDA")
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
def test_heterogeneous_batch_bit_identity(batch_size: int):
    device = torch.device("cuda")
    model = _armed_model(device)
    rows = _ROWS[:batch_size]
    inputs = _step_inputs(rows, device)

    eager_out = _run_eager(model, rows, inputs)
    graph_out = _run_graph(model, rows, inputs)

    _assert_identical(graph_out, eager_out, f"bs={batch_size}")


@pytest.mark.skipif(not _HAS_CUDA, reason="tail CUDA graph needs CUDA")
def test_heterogeneous_padded_bucket_bit_identity():
    """Live bs=3 replays through the bucket-4 graph with padded rows."""
    device = torch.device("cuda")
    model = _armed_model(device)
    rows = _ROWS[:3]
    inputs = _step_inputs(rows, device)

    eager_out = _run_eager(model, rows, inputs)
    graph_out = _run_graph(model, rows, inputs)

    keys = model._tail_cache.graphs
    assert any(key[0] == 4 for key in keys)
    assert not any(key[0] == 3 for key in keys)
    assert graph_out[0].shape == (3, N_CODEBOOKS)
    _assert_identical(graph_out, eager_out, "padded bs=3 -> bucket 4")


@pytest.mark.skipif(not _HAS_CUDA, reason="tail CUDA graph needs CUDA")
def test_heterogeneous_batch_reuses_one_graph_across_steps():
    """Per-request inputs change every step; one key must still serve them.

    Frames run off ONE seed against an eager model stepping in lockstep, so a
    graph that consumed the stream differently would show up in the codes."""
    device = torch.device("cuda")
    model = _armed_model(device)
    eager_model = _armed_model(device)
    rows = _ROWS[:4]
    before = set(model._tail_cache.graphs)
    _graph_for(model, rows)

    torch.manual_seed(SEED)
    graph_frames = [
        _graph_frame(model, rows, _step_inputs(rows, device, step=step))
        for step in range(3)
    ]
    torch.manual_seed(SEED)
    eager_frames = [
        _eager_frame(eager_model, rows, _step_inputs(rows, device, step=step))
        for step in range(3)
    ]
    for step, (got, want) in enumerate(zip(graph_frames, eager_frames)):
        _assert_identical(got, want, f"step={step}")

    minted = set(model._tail_cache.graphs) - before
    assert len(minted) == 1, f"expected one new key, got {sorted(minted)}"


# ---- (b) ladder key sharing ----


def test_quantize_tail_top_k_ladder():
    quantize = zonos2_model_module._quantize_tail_top_k
    assert quantize(1, AUDIO_VOCAB) == 8
    assert quantize(40, AUDIO_VOCAB) == 64
    assert quantize(60, AUDIO_VOCAB) == 64
    assert quantize(64, AUDIO_VOCAB) == 64
    assert quantize(106, AUDIO_VOCAB) == 106
    assert quantize(107, AUDIO_VOCAB) == 128
    assert quantize(1025, AUDIO_VOCAB) == AUDIO_VOCAB
    assert quantize(300, 128) == 128


@pytest.mark.skipif(not _HAS_CUDA, reason="tail CUDA graph needs CUDA")
def test_graph_key_shared_across_request_top_k_values():
    """top_k bounds 40 and 60 land in the same ladder bucket and share one key."""
    device = torch.device("cuda")
    model = _armed_model(device)

    for step, top_k in enumerate((40, 60)):
        rows = [
            dict(
                temperature=1.0,
                top_k=top_k,
                top_p=0.0,
                min_p=0.18,
                repetition_penalty=1.2,
            ),
            dict(
                temperature=0.9,
                top_k=top_k // 2,
                top_p=0.0,
                min_p=0.18,
                repetition_penalty=1.3,
            ),
        ]
        inputs = _step_inputs(rows, device, step=step)
        eager_out = _run_eager(model, rows, inputs)
        graph_out = _run_graph(model, rows, inputs)
        _assert_identical(graph_out, eager_out, f"top_k={top_k}")

    ladder_keys = [key for key in model._tail_cache.graphs if key[1] == 64]
    assert len(ladder_keys) == 1, (
        "distinct request top_k values inside one ladder bucket must share one "
        f"graph, got keys {sorted(model._tail_cache.graphs)}"
    )


# ---- (c) rows below the captured bucket width ----


@pytest.mark.skipif(not _HAS_CUDA, reason="tail CUDA graph needs CUDA")
def test_row_top_k_below_bucket_width_bit_identity():
    """A row whose k is far below the captured width still equals eager exactly."""
    device = torch.device("cuda")
    model = _armed_model(device)
    rows = [
        dict(temperature=1.0, top_k=8, top_p=0.0, min_p=0.18, repetition_penalty=1.2),
        dict(temperature=1.2, top_k=100, top_p=0.0, min_p=0.18, repetition_penalty=1.2),
    ]
    inputs = _step_inputs(rows, device)

    eager_out = _run_eager(model, rows, inputs)
    graph_out = _run_graph(model, rows, inputs)

    assert any(
        key[1] == 106 for key in model._tail_cache.graphs
    ), f"expected capture at ladder width 106, got {sorted(model._tail_cache.graphs)}"
    _assert_identical(graph_out, eager_out, "top_k=8 row under a width-106 capture")


# ---- RNG stream: capture neutrality and padded-replay accounting ----

# A signature the startup pre-capture does not already hold, so asking for it
# forces a lazy capture mid-stream.
_LAZY_ROWS: list[dict] = [
    dict(temperature=1.15, top_k=100, top_p=0.85, min_p=0.18, repetition_penalty=1.2),
    dict(temperature=0.9, top_k=64, top_p=0.9, min_p=0.18, repetition_penalty=1.3),
    dict(temperature=1.2, top_k=32, top_p=0.95, min_p=0.18, repetition_penalty=1.1),
    dict(temperature=0.75, top_k=16, top_p=0.8, min_p=0.18, repetition_penalty=1.4),
]


def _reference_eager_frame(device: torch.device, rows: list[dict], inputs: dict):
    """Eager frame with no capture anywhere between the seed and the sample."""
    model = _armed_model(device)
    torch.manual_seed(SEED)
    return _eager_frame(model, rows, inputs)


@pytest.mark.skipif(not _HAS_CUDA, reason="tail CUDA graph needs CUDA")
def test_lazy_capture_does_not_consume_the_request_rng_stream():
    """A capture triggered by a live request must not burn its random draws.

    The runner asks for the graph before it samples, and capture warms up and
    records the chain, each pass drawing from the global generator. Without a
    save/restore the first request of a new signature samples from a shifted
    stream, so it is not reseeded here: the reference and the live run share one
    seed and nothing in between may move the generator."""
    device = torch.device("cuda")
    rows = _LAZY_ROWS
    inputs = _step_inputs(rows, device, step=11)
    reference = _reference_eager_frame(device, rows, inputs)

    model = _armed_model(device)
    assert not any(
        key[2] for key in model._tail_cache.graphs
    ), "this signature must not be pre-captured, or the test proves nothing"
    torch.manual_seed(SEED)
    live = _graph_frame(model, rows, inputs)

    assert any(key[2] for key in model._tail_cache.graphs), "expected a lazy capture"
    _assert_identical(live, reference, "lazy capture shifted the RNG stream")


@pytest.mark.skipif(not _HAS_CUDA, reason="tail CUDA graph needs CUDA")
def test_failed_capture_does_not_consume_the_request_rng_stream(
    monkeypatch: pytest.MonkeyPatch,
):
    """The eager fallback after a failed capture must also see a pristine stream.

    Warmups have already drawn by the time the capture raises, so the restore
    has to happen on the failure path too."""
    device = torch.device("cuda")
    rows = _LAZY_ROWS
    inputs = _step_inputs(rows, device, step=12)
    reference = _reference_eager_frame(device, rows, inputs)

    model = _armed_model(device)

    def boom(*args, **kwargs):
        raise RuntimeError("simulated capture failure")

    monkeypatch.setattr(torch.cuda, "graph", boom)
    torch.manual_seed(SEED)
    top_k_max, any_top_p, any_min_p = _host_flags(rows, model.audio_vocab)
    assert (
        model.tail_graph(
            len(rows),
            top_k_max=top_k_max,
            any_top_p=any_top_p,
            any_min_p=any_min_p,
            window=WINDOW,
        )
        is None
    )
    assert model._tail_cache.disabled_keys, "expected the capture to have failed"
    live = _eager_frame(model, rows, inputs)

    _assert_identical(live, reference, "failed capture shifted the RNG stream")


_CROSS_FRAME_BUCKETS = (1, 2, 4, 8, 16, 32, 48)


def _consecutive_frames(
    device: torch.device, batch_size: int, *, graphed: bool, frames: int = 4
):
    """Run consecutive frames off ONE seed, the way serving does: no reseeding
    between frames, so any drift in global RNG consumption compounds and shows
    up in the codes. Returns the frames plus the generator state left behind."""
    rows = _hetero(batch_size)
    model = _armed_model(device, _CROSS_FRAME_BUCKETS)
    if graphed:
        _graph_for(model, rows)  # capture up front; its neutrality is tested above
    torch.manual_seed(SEED)
    out = []
    for step in range(frames):
        inputs = _step_inputs(rows, device, step=100 + step)
        run = _graph_frame if graphed else _eager_frame
        out.append(run(model, rows, inputs)[0])
    return out, torch.cuda.get_rng_state(device).clone()


@pytest.mark.skipif(not _HAS_CUDA, reason="tail CUDA graph needs CUDA")
@pytest.mark.parametrize(
    "batch_size, bucket",
    [(1, 1), (3, 4), (5, 8), (12, 16), (17, 32), (33, 48)],
    ids=["exact-1", "pad-3of4", "pad-5of8", "pad-12of16", "pad-17of32", "pad-33of48"],
)
def test_consecutive_frames_track_the_eager_rng_stream(batch_size: int, bucket: int):
    """Graphed and eager decode must stay on the same stream across frames.

    A padded replay executes the whole captured bucket, so it also draws for the
    padding rows. That does NOT drift here: ``torch.multinomial`` derives its
    philox offset increment from the launch configuration, not the row count, so
    a bucket-sized draw and a live-sized draw bump the generator identically for
    every bucket in this ladder (measured; the increment is flat at 4 until
    ~1062 rows, and bucket 48 is 432 rows). Buckets 128 and 256 straddle that
    threshold and are the only ones where a padded replay could drift, which is
    unreachable at the shipped cuda_graph_max_bs.

    No reseeding between frames, so a drift would compound into the codes."""
    device = torch.device("cuda")
    assert next(b for b in _CROSS_FRAME_BUCKETS if b >= batch_size) == bucket

    eager_frames, eager_state = _consecutive_frames(device, batch_size, graphed=False)
    graph_frames, graph_state = _consecutive_frames(device, batch_size, graphed=True)

    for step, (got, want) in enumerate(zip(graph_frames, eager_frames)):
        assert torch.equal(got, want), (
            f"frame {step} diverged from eager (bs={batch_size}, bucket={bucket}); "
            "a shifted global RNG stream is the likely cause"
        )
    assert torch.equal(graph_state, eager_state), (
        f"replay left the generator somewhere else than eager "
        f"(bs={batch_size}, bucket={bucket})"
    )


# ---- cache plumbing: env switch, fuse, ceiling, pool ----


def test_env_switch_parsing(monkeypatch: pytest.MonkeyPatch):
    env = zonos2_model_module.ZONOS2_FRAME_GRAPH_ENV
    monkeypatch.delenv(env, raising=False)
    assert env_graph_enabled(env) is True
    for falsy in ("0", "false", "no", "off"):
        monkeypatch.setenv(env, falsy)
        assert env_graph_enabled(env) is False
    monkeypatch.setenv(env, "1")
    assert env_graph_enabled(env) is True


@pytest.mark.skipif(not _HAS_CUDA, reason="tail CUDA graph needs CUDA")
def test_env_switch_off_keeps_the_tail_eager(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(zonos2_model_module.ZONOS2_FRAME_GRAPH_ENV, "0")
    device = torch.device("cuda")
    model = _armed_model(device)

    assert not model._tail_cache.graphs
    assert (
        model.tail_graph(
            2, top_k_max=106, any_top_p=False, any_min_p=True, window=WINDOW
        )
        is None
    )


@pytest.mark.skipif(not _HAS_CUDA, reason="tail CUDA graph needs CUDA")
def test_window_mismatch_falls_back_to_eager():
    """The rep_ids buffer width is allocated once; a different window is eager."""
    device = torch.device("cuda")
    model = _armed_model(device)
    assert (
        model.tail_graph(
            2, top_k_max=106, any_top_p=False, any_min_p=True, window=WINDOW + 1
        )
        is None
    )


@pytest.mark.skipif(not _HAS_CUDA, reason="tail CUDA graph needs CUDA")
def test_capture_failure_disables_key_and_falls_back(monkeypatch: pytest.MonkeyPatch):
    device = torch.device("cuda")
    model = _armed_model(device)
    calls = []

    def boom(self, key):
        calls.append(key)
        raise RuntimeError("simulated capture failure")

    monkeypatch.setattr(Zonos2SGLangModel, "_capture_tail_graph", boom)
    args = dict(top_k_max=32, any_top_p=True, any_min_p=False, window=WINDOW)

    assert model.tail_graph(2, **args) is None
    assert model._tail_cache.disabled_keys
    assert model.tail_graph(2, **args) is None
    assert len(calls) == 1, "a disabled key must not retry capture"


@pytest.mark.skipif(not _HAS_CUDA, reason="tail CUDA graph needs CUDA")
def test_repeated_capture_failures_blow_the_global_fuse(
    monkeypatch: pytest.MonkeyPatch,
):
    device = torch.device("cuda")
    model = _armed_model(device)
    model._tail_cache.max_failures = 3

    monkeypatch.setattr(
        Zonos2SGLangModel,
        "_capture_tail_graph",
        lambda self, key: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    for top_k_max in (8, 16, 32):
        model.tail_graph(
            2, top_k_max=top_k_max, any_top_p=True, any_min_p=False, window=WINDOW
        )

    assert model._tail_cache.enabled is False
    assert (
        model.tail_graph(
            4, top_k_max=64, any_top_p=True, any_min_p=False, window=WINDOW
        )
        is None
    )


@pytest.mark.skipif(not _HAS_CUDA, reason="tail CUDA graph needs CUDA")
def test_key_ceiling_stops_minting_new_graphs():
    device = torch.device("cuda")
    model = _armed_model(device)
    cache = model._tail_cache
    cache.max_keys = len(cache.graphs)

    assert (
        model.tail_graph(
            2, top_k_max=16, any_top_p=True, any_min_p=False, window=WINDOW
        )
        is None
    )


@pytest.mark.skipif(not _HAS_CUDA, reason="tail CUDA graph needs CUDA")
def test_graph_keys_share_one_memory_pool(monkeypatch: pytest.MonkeyPatch):
    device = torch.device("cuda")
    model = _build_model(device)
    pools = []
    real_graph = torch.cuda.graph

    class _SpyGraph(real_graph):
        def __init__(self, cuda_graph, pool=None, **kwargs):
            pools.append(pool)
            super().__init__(cuda_graph, pool=pool, **kwargs)

    monkeypatch.setattr(torch.cuda, "graph", _SpyGraph)
    model.capture_tail_graphs(list(BUCKETS), _default_params())

    assert len(pools) == len(BUCKETS)
    assert pools[0] is not None
    assert all(pool == pools[0] for pool in pools)
    assert pools[0] == model._tail_cache.memory_pool()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
