# SPDX-License-Identifier: Apache-2.0
"""Bit-identity test: the stateful multi-chunk CUDA-graph replay must equal the eager codec decode bit-for-bit (torch.equal, maxdelta 0). GPU + real MOSS-Audio-Tokenizer-v2 codec required."""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest
import torch

pytestmark = pytest.mark.accelerator

CODEC_MODEL_ID = "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2"
N_VQ = 12  # MOSS-TTS-Local v1.5 uses the first 12 RVQ codebooks
STREAM_SLOTS = 8
# T values the gate exercises (chunk sizes); warmup captures these + remainders.
CHUNK_TS = [1, 5, 25, 100]
PCM_CASES = [
    (chunk_t, n_active, chunk_t * 3 + max(1, chunk_t // 2))
    for chunk_t in CHUNK_TS
    for n_active in [1, 3, 8]
] + [(5, 1, 75), (25, 1, 75)]
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
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    from sglang_omni.models.moss_tts.audio_tokenizer import load_moss_audio_vocoder
    from sglang_omni.models.moss_tts_local.streaming_vocoder import _CodecStreamSession

    try:
        model_path = snapshot_download(CODEC_MODEL_ID, local_files_only=True)
    except LocalEntryNotFoundError:
        pytest.skip("MOSS-Audio-Tokenizer-v2 codec snapshot not found")
    # Exercise the production loader. The remote AutoModel lacks the native
    # state API and would silently skip every graph correctness test.
    codec = load_moss_audio_vocoder(
        model_path,
        device="cuda:0",
        decoder_dtype=torch.float32,
        compute_dtype=torch.bfloat16,
    ).model
    n_vq = N_VQ
    vocab = _codebook_size(codec)
    session = _CodecStreamSession(
        codec,
        stream_slots=STREAM_SLOTS,
        n_vq=n_vq,
    )
    # note (Zhang Yiyang): Include final remainders and the dynamic-batch trace.
    wanted = {7}
    for chunk_t, _, total in PCM_CASES:
        wanted.add(chunk_t)
        if total % chunk_t:
            wanted.add(total % chunk_t)
    try:
        captured = session.warmup_cuda_graph(sorted(wanted))
        assert session._cg_runner is not None
        assert set(session._cg_runner.capture_sizes) == {
            (batch_size, length)
            for batch_size in session._graph_batch_sizes()
            for length in wanted
        }
        yield session, n_vq, vocab, set(captured)
    finally:
        session.close()


def _step(session, slot_codes, *, require_graph=False):
    if require_graph:
        batch_size = next(
            size for size in session._graph_batch_sizes() if size >= len(slot_codes)
        )
        length = next(iter(slot_codes.values())).shape[1]
        assert session._cg_runner is not None
        assert (batch_size, length) in session._cg_runner.capture_sizes
        graph_steps = sum(session._cg_graph_t.values())
        eager_steps = sum(session._cg_eager_t.values())
    output = session.step(slot_codes)
    if require_graph:
        assert sum(session._cg_graph_t.values()) == graph_steps + 1
        assert sum(session._cg_eager_t.values()) == eager_steps
    return output


def _decode_chunks(session, slot_seqs, chunk_t, *, require_graph=False):
    """Decode dict{slot: [n_vq, T_total]} in lockstep chunks of chunk_t. Resets slots first."""
    slots = list(slot_seqs)
    session._reset_slots(slots)
    total = next(iter(slot_seqs.values())).shape[1]
    parts = {s: [] for s in slots}
    pos = 0
    while pos < total:
        t = min(chunk_t, total - pos)
        out = _step(
            session,
            {s: slot_seqs[s][:, pos : pos + t] for s in slots},
            require_graph=require_graph,
        )
        for s in slots:
            parts[s].append(out[s])
        pos += t
    return {s: torch.cat(parts[s], dim=-1) for s in slots}


@pytest.mark.skipif(not _HAS_CUDA, reason="needs CUDA + real codec")
def test_indexed_streaming_matches_sequential_precision_and_cache_order(session_bundle):
    from sglang_omni.models.moss_tts.attention import MossAudioTokenizerStreamingModule
    from sglang_omni.models.moss_tts.audio_tokenizer import (
        MossAudioTokenizerVocoderDecoder,
    )

    session, n_vq, vocab, _ = session_bundle
    codec = session._codec
    # note (Zhang Yiyang): Compare indexed and sequential state at the same
    # batch width; independent attention math is tested in test_audio_tokenizer.
    reference = MossAudioTokenizerStreamingModule()
    reference.decoder = MossAudioTokenizerVocoderDecoder(source_decoder=codec.decoder)
    batch_size = 4
    active_slots = [7, 2, 4]
    session._reset_slots(active_slots)
    torch.manual_seed(912)
    with reference.streaming(batch_size), torch.no_grad():
        mask = torch.arange(batch_size, device="cuda") < len(active_slots)
        reference._set_streaming_exec_mask(mask)
        for length in [5, 25, 7]:
            codes = torch.randint(0, vocab, (n_vq, batch_size, length), device="cuda")
            codes[:, len(active_slots) :] = 0
            hidden = codec.quantizer.decode_codes(codes).float()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                expected, lengths = reference.decoder(hidden, mask.long() * length)
            expected, lengths = codec._restore_channels_from_codec(expected, lengths)
            actual = _step(
                session,
                {slot: codes[:, i] for i, slot in enumerate(active_slots)},
                require_graph=True,
            )
            for i, slot in enumerate(active_slots):
                torch.testing.assert_close(
                    actual[slot], expected[i].float().cpu(), rtol=0, atol=0
                )


def test_cuda_graph_capture_uses_thread_local_error_mode():
    from sglang_omni.models.moss_tts_local.vocoder_cuda_graph import (
        MossVocoderCudaGraphRunner,
    )

    source = textwrap.dedent(inspect.getsource(MossVocoderCudaGraphRunner._capture))
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
    assert graph_calls, "MOSS-Audio-Tokenizer vocoder CUDA graph capture call not found"
    assert any(
        keyword.arg == "capture_error_mode"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value == "thread_local"
        for call in graph_calls
        for keyword in call.keywords
    ), "MOSS-Audio-Tokenizer vocoder CUDA graph capture must use thread-local error mode"


@pytest.mark.skipif(not _HAS_CUDA, reason="needs CUDA + real codec")
@pytest.mark.parametrize("chunk_t,n_active,total", PCM_CASES)
def test_streaming_pcm_bit_identical(session_bundle, chunk_t, n_active, total):
    session, n_vq, vocab, _ = session_bundle
    torch.manual_seed(1000 * chunk_t + n_active)
    slot_seqs = {
        s: torch.randint(0, vocab, (n_vq, total), device="cuda", dtype=torch.long)
        for s in range(n_active)
    }
    runner = session._cg_runner
    try:
        session._cg_runner = None
        eager = _decode_chunks(session, slot_seqs, chunk_t)
    finally:
        session._cg_runner = runner
    graphed = _decode_chunks(session, slot_seqs, chunk_t, require_graph=True)
    for s in range(n_active):
        assert torch.equal(eager[s], graphed[s]), (
            f"streaming PCM not bit-identical (chunk_t={chunk_t}, n_active={n_active}, slot={s}): "
            f"max|delta|={(eager[s] - graphed[s]).abs().max().item():.3e}"
        )


@pytest.mark.skipif(not _HAS_CUDA, reason="needs CUDA + real codec")
def test_graph_tracks_eager_with_changing_batches_and_slot_reuse(session_bundle):
    session, n_vq, vocab, _ = session_bundle
    torch.manual_seed(123)
    trace = [
        {name: torch.randint(0, vocab, (n_vq, length), device="cuda") for name in names}
        for names, length in [
            (("a",), 5),
            (("c", "a", "b"), 25),
            (("b",), 7),
            (("c", "d"), 5),
            (("b",), 25),
        ]
    ]

    def decode_trace(*, require_graph):
        session._reset_slots(list(range(STREAM_SLOTS)))
        slots = {name: session.acquire() for name in ("a", "b", "c")}
        outputs = []
        try:
            for index, codes in enumerate(trace):
                if index == 3:
                    released = slots.pop("a")
                    session.release(released)
                    slots["d"] = session.acquire()
                    assert slots["d"] == released
                output = _step(
                    session,
                    {slots[name]: value for name, value in codes.items()},
                    require_graph=require_graph,
                )
                outputs.append({name: output[slots[name]] for name in codes})
        finally:
            for slot in sorted(slots.values()):
                session.release(slot)
        return outputs

    runner = session._cg_runner
    try:
        session._cg_runner = None
        eager = decode_trace(require_graph=False)
    finally:
        session._cg_runner = runner
    graphed = decode_trace(require_graph=True)
    for actual, expected in zip(graphed, eager, strict=True):
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


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
        real_state_capacity=STREAM_SLOTS,
        scratch_capacity=STREAM_SLOTS,
        batch_sizes=session._graph_batch_sizes(),
        frame_sizes=[5, 25],
        num_quantizers=n_vq,
        min_free_gb=100000.0,  # 100 TB headroom -> always trips
    )
    guarded.warmup([5, 25])
    assert (
        guarded.captured_frames() == []
    ), "VRAM guard must skip all captures under insufficient headroom"


@pytest.mark.skipif(not _HAS_CUDA, reason="needs CUDA + real codec")
def test_capture_failure_falls_back_to_eager(session_bundle):
    """A capture exception drops that shape and leaves eager decode available."""
    from sglang_omni.models.moss_tts_local.vocoder_cuda_graph import (
        MossVocoderCudaGraphRunner,
    )

    session, n_vq, vocab, captured = session_bundle
    runner = MossVocoderCudaGraphRunner(
        session._codec,
        real_state_capacity=STREAM_SLOTS,
        scratch_capacity=STREAM_SLOTS,
        batch_sizes=session._graph_batch_sizes(),
        frame_sizes=[5, 25],
        num_quantizers=n_vq,
    )

    def boom(batch_size, frame_size):
        raise RuntimeError("simulated capture OOM")

    runner._capture = boom
    runner.warmup([5, 25])
    assert (
        runner.captured_frames() == []
    ), "capture failures must be caught per-T -> no graphs -> eager"
    assert runner._sealed, "runner must still seal after capture failures"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
