# SPDX-License-Identifier: Apache-2.0
"""Bit-identity test: the stateful multi-chunk CUDA-graph replay must equal the eager codec decode bit-for-bit (torch.equal, maxdelta 0). GPU + real MOSS-Audio-Tokenizer-v2 codec required."""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest
import torch

pytestmark = pytest.mark.gpu

CODEC_GLOB = (
    "/root/.cache/huggingface/hub/"
    "models--OpenMOSS-Team--MOSS-Audio-Tokenizer-v2/snapshots/*/"
)
N_VQ = 12  # MOSS-TTS-Local v1.5 uses the first 12 RVQ codebooks
STREAM_SLOTS = 8
OFFLINE_SLOTS = 8
# T values the gate exercises (chunk sizes); warmup captures these + remainders.
CHUNK_TS = [1, 5, 25, 100]
_HAS_CUDA = torch.cuda.is_available()


def _codebook_size(codec) -> int:
    q = getattr(codec, "quantizer", None)
    qs = getattr(q, "quantizers", None)
    if qs:
        for attr in ("codebook_size", "n_codes", "num_embeddings", "codebook_dim"):
            v = getattr(qs[0], attr, None)
            if isinstance(v, int) and v > 0:
                return v
    v = getattr(getattr(codec, "config", None), "codebook_size", None)
    return v if isinstance(v, int) and v > 0 else 1024


@pytest.fixture(scope="module")
def session_bundle():
    # Load the codec DIRECTLY (the sglang processor pulls librosa/soxr audio deps absent in this
    # serving container; the codec modeling file is self-contained). streaming_vocoder imports clean.
    import glob

    from transformers import AutoModel

    from sglang_omni.models.moss_tts_local.streaming_vocoder import _CodecStreamSession

    snaps = glob.glob(CODEC_GLOB)
    if not snaps:
        pytest.skip("MOSS-Audio-Tokenizer-v2 codec snapshot not found")
    codec = (
        AutoModel.from_pretrained(snaps[0], trust_remote_code=True).to("cuda").eval()
    )
    n_vq = N_VQ
    vocab = _codebook_size(codec)
    session = _CodecStreamSession(
        codec, stream_slots=STREAM_SLOTS, offline_slots=OFFLINE_SLOTS, n_vq=n_vq
    )
    # Capture every length the test emits (each chunk_t and its remainder) once, sealed.
    wanted = set()
    for chunk_t in CHUNK_TS:
        total = chunk_t * 3 + max(1, chunk_t // 2)
        pos = 0
        while pos < total:
            wanted.add(min(chunk_t, total - pos))
            pos += min(chunk_t, total - pos)
    captured = session.warmup_cuda_graph(sorted(wanted))
    return session, n_vq, vocab, set(captured)


def _decode_chunks(session, slot_seqs, chunk_t):
    """Decode dict{slot: [n_vq, T_total]} in lockstep chunks of chunk_t. Resets slots first."""
    slots = list(slot_seqs)
    session._reset_slots(slots)
    total = next(iter(slot_seqs.values())).shape[1]
    parts = {s: [] for s in slots}
    pos = 0
    while pos < total:
        t = min(chunk_t, total - pos)
        out = session.step({s: slot_seqs[s][:, pos : pos + t] for s in slots})
        for s in slots:
            parts[s].append(out[s])
        pos += t
    return {s: torch.cat(parts[s], dim=-1) for s in slots}


@pytest.mark.skipif(not _HAS_CUDA, reason="needs CUDA + real codec")
def test_some_graphs_captured(session_bundle):
    _, _, _, captured = session_bundle
    # If zero captured, the whole line is eager-only (no prize) -- surface it loudly.
    assert (
        captured
    ), "no codec-decode CUDA graphs captured (all shapes fell back to eager)"


def test_cuda_graph_capture_uses_thread_local_error_mode():
    from sglang_omni.models.moss_tts_local.vocoder_cuda_graph import (
        MossVocoderCudaGraphRunner,
    )

    source = textwrap.dedent(
        inspect.getsource(MossVocoderCudaGraphRunner._capture_frame_count)
    )
    tree = ast.parse(source)
    graph_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "graph"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "cuda"
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == "torch"
    ]
    assert graph_calls, "MOSS vocoder CUDA graph capture call not found"
    assert any(
        keyword.arg == "capture_error_mode"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value == "thread_local"
        for call in graph_calls
        for keyword in call.keywords
    ), "MOSS vocoder CUDA graph capture must use thread-local error mode"


@pytest.mark.skipif(not _HAS_CUDA, reason="needs CUDA + real codec")
@pytest.mark.parametrize("chunk_t", CHUNK_TS)
@pytest.mark.parametrize("n_active", [1, 8])
def test_streaming_pcm_bit_identical(session_bundle, chunk_t, n_active):
    session, n_vq, vocab, captured = session_bundle
    if chunk_t not in captured:
        pytest.skip(
            f"T={chunk_t} fell back to eager (not captured); nothing to compare"
        )
    torch.manual_seed(1000 * chunk_t + n_active)
    total = chunk_t * 3 + max(1, chunk_t // 2)  # multiple full chunks + a remainder
    slot_seqs = {
        s: torch.randint(0, vocab, (n_vq, total), device="cuda", dtype=torch.long)
        for s in range(n_active)
    }
    runner = session._cg_runner
    session._cg_runner = None  # force eager
    eager = _decode_chunks(session, slot_seqs, chunk_t)
    session._cg_runner = runner  # graph path
    graphed = _decode_chunks(session, slot_seqs, chunk_t)
    for s in range(n_active):
        assert torch.equal(eager[s], graphed[s]), (
            f"streaming PCM not bit-identical (chunk_t={chunk_t}, n_active={n_active}, slot={s}): "
            f"max|delta|={(eager[s] - graphed[s]).abs().max().item():.3e}"
        )


@pytest.mark.skipif(not _HAS_CUDA, reason="needs CUDA + real codec")
@pytest.mark.parametrize("chunk_t", [5, 25])
def test_graph_tracks_eager_across_chunkings(session_bundle, chunk_t):
    """Graph PCM must equal eager PCM bit-for-bit at EACH chunking (the codec is chunk-boundary
    dependent, so we assert per-chunking, not cross-chunking)."""
    session, n_vq, vocab, captured = session_bundle
    if chunk_t not in captured:
        pytest.skip(
            f"T={chunk_t} fell back to eager (not captured); nothing to compare"
        )
    torch.manual_seed(7)
    total = 75
    seq = {0: torch.randint(0, vocab, (n_vq, total), device="cuda", dtype=torch.long)}
    runner = session._cg_runner
    session._cg_runner = None  # force eager
    eager = _decode_chunks(session, seq, chunk_t)[0]
    session._cg_runner = runner  # graph path
    graphed = _decode_chunks(session, seq, chunk_t)[0]
    assert torch.equal(eager, graphed), (
        f"graph decode not bit-identical to eager at chunk_t={chunk_t}: "
        f"max|delta|={(eager - graphed).abs().max().item():.3e}"
    )


@pytest.mark.skipif(not _HAS_CUDA, reason="needs CUDA + real codec")
def test_replay_failure_disables_runner_and_serves_eager_bit_identical(session_bundle):
    """A replay exception disables the runner (future steps go eager, bit-identical to a pure-eager
    reference); the failing step itself raises so its participants abort."""
    session, n_vq, vocab, captured = session_bundle
    chunk_t = next((t for t in (5, 25) if t in captured), None)
    if chunk_t is None:
        pytest.skip("need T=5 or T=25 captured")
    torch.manual_seed(4242)
    seq = {
        0: torch.randint(0, vocab, (n_vq, chunk_t * 3), device="cuda", dtype=torch.long)
    }
    runner = session._cg_runner
    session._cg_runner = None  # pure-eager reference
    eager_ref = _decode_chunks(session, seq, chunk_t)[0]

    session._cg_runner = runner  # graph path, but make the next replay blow up
    session._reset_slots([0])

    def boom(*args, **kwargs):
        raise RuntimeError("simulated replay failure")

    orig_decode = runner.decode_step
    runner.decode_step = boom
    try:
        with pytest.raises(RuntimeError):
            session.step({0: seq[0][:, :chunk_t]})
        assert (
            session._cg_runner is None
        ), "runner must be disabled after a replay failure"
        # session is now eager-only -> a fresh decode must be bit-identical to the pure-eager reference
        after = _decode_chunks(session, seq, chunk_t)[0]
        assert torch.equal(after, eager_ref), (
            "post-failure eager output not bit-identical to eager reference: "
            f"max|delta|={(after - eager_ref).abs().max().item():.3e}"
        )
    finally:
        # restore the module-scoped session for the remaining tests
        runner.decode_step = orig_decode
        session._cg_runner = runner


@pytest.mark.skipif(not _HAS_CUDA, reason="needs CUDA + real codec")
def test_vram_guard_skips_capture_and_falls_back_to_eager(session_bundle):
    """Below the configured VRAM headroom, warmup skips capture (empty graph set, serving uses eager);
    forced via an absurd min_free_gb."""
    from sglang_omni.models.moss_tts_local.vocoder_cuda_graph import (
        MossVocoderCudaGraphRunner,
    )

    session, n_vq, vocab, captured = session_bundle
    guarded = MossVocoderCudaGraphRunner(
        session._codec,
        batch_size=STREAM_SLOTS + OFFLINE_SLOTS,
        n_vq=n_vq,
        min_free_gb=100000.0,  # 100 TB headroom -> always trips
    )
    guarded.warmup([5, 25])
    assert (
        guarded.captured_frames() == []
    ), "VRAM guard must skip all captures under insufficient headroom"


@pytest.mark.skipif(not _HAS_CUDA, reason="needs CUDA + real codec")
def test_capture_failure_falls_back_to_eager(session_bundle):
    """A capture exception is caught per-T, that T dropped, serving uses eager; forced via _capture_frame_count raising. (A real mid-capture OOM does not wedge the CUDA context, verified empirically.)"""
    from sglang_omni.models.moss_tts_local.vocoder_cuda_graph import (
        MossVocoderCudaGraphRunner,
    )

    session, n_vq, vocab, captured = session_bundle
    runner = MossVocoderCudaGraphRunner(
        session._codec, batch_size=STREAM_SLOTS + OFFLINE_SLOTS, n_vq=n_vq
    )

    def boom(frame_count):
        raise RuntimeError("simulated capture OOM")

    runner._capture_frame_count = boom
    runner.warmup([5, 25])
    assert (
        runner.captured_frames() == []
    ), "capture failures must be caught per-T -> no graphs -> eager"
    assert runner._sealed, "runner must still seal after capture failures"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
