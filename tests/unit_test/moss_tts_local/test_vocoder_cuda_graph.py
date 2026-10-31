# SPDX-License-Identifier: Apache-2.0
"""Bit-identity tests for stateful multi-chunk device-graph replay with the real codec."""

from __future__ import annotations

import ast
import inspect
import textwrap
from unittest.mock import Mock

import pytest
import torch

pytestmark = pytest.mark.accelerator

CODEC_MODEL_ID = "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2"
N_VQ = 12  # MOSS-TTS-Local v1.5 uses the first 12 RVQ codebooks
STREAM_SLOTS = 8
HAS_CUDA = torch.cuda.is_available()
HAS_NPU = hasattr(torch, "npu") and torch.npu.is_available()
HAS_ACCELERATOR = HAS_CUDA or HAS_NPU
DEVICE_TYPE = "cuda" if HAS_CUDA else "npu"
DEVICE = f"{DEVICE_TYPE}:0"
CHUNK_TS = [1, 5, 25, 100] if HAS_CUDA else [1, 2, 3, 4, 5, 6, 7]
PCM_CASES = [
    (chunk_t, n_active, chunk_t * 3 + max(1, chunk_t // 2))
    for chunk_t in CHUNK_TS
    for n_active in [1, 3, 8]
] + ([(5, 1, 75), (25, 1, 75)] if HAS_CUDA else [(5, 1, 75)])


def codebook_size(codec) -> int:
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
    from sglang_omni.models.moss_tts.codec_session import CodecStreamSession

    try:
        model_path = snapshot_download(CODEC_MODEL_ID, local_files_only=True)
    except LocalEntryNotFoundError:
        pytest.skip("MOSS-Audio-Tokenizer-v2 codec snapshot not found")
    # Exercise the production loader. The remote AutoModel lacks the native
    # state API and would silently skip every graph correctness test.
    codec = load_moss_audio_vocoder(
        model_path,
        device=DEVICE,
        decoder_dtype=torch.float32,
        compute_dtype=torch.bfloat16,
    ).model
    n_vq = N_VQ
    vocab = codebook_size(codec)
    session = CodecStreamSession(
        codec,
        stream_slots=STREAM_SLOTS,
        n_vq=n_vq,
    )
    # note (Zhang Yiyang): CUDA includes final remainders and the dynamic-batch
    # trace; NPU leaves longer lengths uncaptured.
    wanted = set(CHUNK_TS) if DEVICE_TYPE == "npu" else {7}
    if DEVICE_TYPE == "cuda":
        for chunk_t, _, total in PCM_CASES:
            wanted.add(chunk_t)
            if total % chunk_t:
                wanted.add(total % chunk_t)
    else:
        pass
    try:
        captured = session.warmup_cuda_graph(sorted(wanted))
        assert session.cg_runner is not None
        assert set(session.cg_runner.capture_sizes) == {
            (batch_size, length)
            for batch_size in session.graph_batch_sizes()
            for length in wanted
        }
        yield session, n_vq, vocab, set(captured)
    finally:
        session.close()


def step(session, slot_codes, *, require_graph=False):
    if require_graph:
        batch_size = next(
            size for size in session.graph_batch_sizes() if size >= len(slot_codes)
        )
        length = next(iter(slot_codes.values())).shape[1]
        assert session.cg_runner is not None
        assert (batch_size, length) in session.cg_runner.capture_sizes
        graph_steps = sum(session.cg_graph_t.values())
        eager_steps = sum(session.cg_eager_t.values())
    output = session.step(slot_codes)
    if require_graph:
        assert sum(session.cg_graph_t.values()) == graph_steps + 1
        assert sum(session.cg_eager_t.values()) == eager_steps
    return output


def decode_chunks(session, slot_seqs, chunk_t, *, require_graph=False):
    """Decode dict{slot: [n_vq, T_total]} in lockstep chunks of chunk_t. Resets slots first."""
    slots = list(slot_seqs)
    session.reset_slots(slots)
    total = next(iter(slot_seqs.values())).shape[1]
    parts = {s: [] for s in slots}
    pos = 0
    while pos < total:
        t = min(chunk_t, total - pos)
        out = step(
            session,
            {s: slot_seqs[s][:, pos : pos + t] for s in slots},
            require_graph=require_graph and t in session.captured_frames(),
        )
        for s in slots:
            parts[s].append(out[s])
        pos += t
    return {s: torch.cat(parts[s], dim=-1) for s in slots}


@pytest.mark.skipif(not HAS_ACCELERATOR, reason="needs CUDA/NPU + real codec")
def test_indexed_streaming_matches_sequential_precision_and_cache_order(session_bundle):
    from sglang_omni.models.moss_tts.attention import MossAudioTokenizerStreamingModule
    from sglang_omni.models.moss_tts.audio_tokenizer import (
        MossAudioTokenizerVocoderDecoder,
    )

    session, n_vq, vocab, _ = session_bundle
    codec = session.codec
    # note (Zhang Yiyang): Compare indexed and sequential state at the same
    # batch width; independent attention math is tested in test_audio_tokenizer.
    reference = MossAudioTokenizerStreamingModule()
    reference.decoder = MossAudioTokenizerVocoderDecoder(source_decoder=codec.decoder)
    batch_size = 4
    active_slots = [7, 2, 4]
    session.reset_slots(active_slots)
    torch.manual_seed(912)
    with reference.streaming(batch_size), torch.no_grad():
        mask = torch.arange(batch_size, device=DEVICE) < len(active_slots)
        reference.set_streaming_exec_mask(mask)
        for length in [5, 25, 7]:
            codes = torch.randint(0, vocab, (n_vq, batch_size, length), device=DEVICE)
            codes[:, len(active_slots) :] = 0
            hidden = codec.quantizer.decode_codes(codes).float()
            with torch.autocast(DEVICE_TYPE, dtype=torch.bfloat16):
                expected, lengths = reference.decoder(hidden, mask.long() * length)
            expected, lengths = codec.restore_channels_from_codec(expected, lengths)
            actual = step(
                session,
                {slot: codes[:, i] for i, slot in enumerate(active_slots)},
                require_graph=length in session.captured_frames(),
            )
            for i, slot in enumerate(active_slots):
                torch.testing.assert_close(
                    actual[slot], expected[i].float().cpu(), rtol=0, atol=0
                )


def test_cuda_graph_capture_uses_thread_local_error_mode():
    from sglang_omni.models.moss_tts.vocoder_cuda_graph import (
        MossVocoderCudaGraphRunner,
    )

    source = textwrap.dedent(inspect.getsource(MossVocoderCudaGraphRunner.capture))
    tree = ast.parse(source)
    capture_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "capture"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "graph_backend"
    ]
    assert capture_calls, "MOSS-Audio-Tokenizer device graph capture call not found"
    assert any(
        keyword.arg == "thread_local_errors"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is True
        for call in capture_calls
        for keyword in call.keywords
    ), "MOSS-Audio-Tokenizer device graph capture must use thread-local errors"


def test_capture_sync_failure_is_fatal():
    from sglang_omni.models.moss_tts.vocoder_cuda_graph import (
        VocoderGraphCaptureError,
        synchronize_capture_stream,
    )

    stream = Mock()
    stream.synchronize.side_effect = [RuntimeError("sync failed"), None]
    with pytest.raises(VocoderGraphCaptureError, match="failed to synchronize"):
        synchronize_capture_stream(stream)
    synchronize_capture_stream(stream)


@pytest.mark.skipif(not HAS_ACCELERATOR, reason="needs CUDA/NPU + real codec")
@pytest.mark.parametrize("chunk_t,n_active,total", PCM_CASES)
def test_streaming_pcm_bit_identical(session_bundle, chunk_t, n_active, total):
    session, n_vq, vocab, _ = session_bundle
    torch.manual_seed(1000 * chunk_t + n_active)
    slot_seqs = {
        s: torch.randint(0, vocab, (n_vq, total), device=DEVICE, dtype=torch.long)
        for s in range(n_active)
    }
    runner = session.cg_runner
    try:
        session.cg_runner = None
        eager = decode_chunks(session, slot_seqs, chunk_t)
    finally:
        session.cg_runner = runner
    graph_steps = sum(session.cg_graph_t.values())
    graphed = decode_chunks(session, slot_seqs, chunk_t, require_graph=True)
    assert sum(session.cg_graph_t.values()) > graph_steps
    for s in range(n_active):
        assert torch.equal(eager[s], graphed[s]), (
            f"streaming PCM not bit-identical (chunk_t={chunk_t}, n_active={n_active}, slot={s}): "
            f"max|delta|={(eager[s] - graphed[s]).abs().max().item():.3e}"
        )


@pytest.mark.skipif(not HAS_ACCELERATOR, reason="needs CUDA/NPU + real codec")
def test_graph_tracks_eager_with_changing_batches_and_slot_reuse(session_bundle):
    session, n_vq, vocab, _ = session_bundle
    torch.manual_seed(123)
    trace = [
        {name: torch.randint(0, vocab, (n_vq, length), device=DEVICE) for name in names}
        for names, length in [
            (("a",), 5),
            (("c", "a", "b"), 25 if HAS_CUDA else 1),
            (("b",), 7 if HAS_CUDA else 4),
            (("c", "d"), 5 if HAS_CUDA else 2),
            (("b",), 25),
        ]
    ]

    def decode_trace(*, require_graph):
        session.reset_slots(list(range(STREAM_SLOTS)))
        slots = {name: session.acquire() for name in ("a", "b", "c")}
        outputs = []
        try:
            for index, codes in enumerate(trace):
                if index == 3:
                    released = slots.pop("a")
                    session.release(released)
                    slots["d"] = session.acquire()
                    assert slots["d"] == released
                output = step(
                    session,
                    {slots[name]: value for name, value in codes.items()},
                    require_graph=require_graph
                    and next(iter(codes.values())).shape[1]
                    in session.captured_frames(),
                )
                outputs.append({name: output[slots[name]] for name in codes})
        finally:
            for slot in sorted(slots.values()):
                session.release(slot)
        return outputs

    runner = session.cg_runner
    try:
        session.cg_runner = None
        eager = decode_trace(require_graph=False)
    finally:
        session.cg_runner = runner
    graph_steps = sum(session.cg_graph_t.values())
    eager_steps = sum(session.cg_eager_t.values())
    graphed = decode_trace(require_graph=True)
    assert sum(session.cg_graph_t.values()) > graph_steps
    if DEVICE_TYPE == "npu":
        assert sum(session.cg_eager_t.values()) > eager_steps
    else:
        pass
    for actual, expected in zip(graphed, eager, strict=True):
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


@pytest.mark.skipif(not HAS_NPU or HAS_CUDA, reason="needs NPU + real codec")
def test_npu_rope_overflow_is_rejected_before_replay(session_bundle):
    session, n_vq, _, _ = session_bundle
    runner = session.cg_runner
    assert runner is not None
    assert session.streaming_max_positions > 0
    session.reset_slots([0])
    session.slot_offsets[0] = session.streaming_max_positions - 1
    original_decode = runner.decode_step

    def unexpected_replay(*args, **kwargs):
        raise AssertionError("replay must not run after a failed RoPE preflight")

    try:
        runner.decode_step = unexpected_replay
        with pytest.raises(ValueError, match="streaming RoPE position exceeds"):
            session.step({0: torch.zeros(n_vq, 5, dtype=torch.long, device=DEVICE)})
        assert session.cg_runner is runner
    finally:
        runner.decode_step = original_decode
        session.reset_slots([0])


@pytest.mark.skipif(not HAS_ACCELERATOR, reason="needs CUDA/NPU + real codec")
def test_replay_failure_disables_runner_and_serves_eager_bit_identical(session_bundle):
    """A replay exception disables the runner (future steps go eager, bit-identical to a pure-eager
    reference); the failing step itself raises so its participants abort."""
    session, n_vq, vocab, captured = session_bundle
    chunk_t = next((t for t in (5, 25) if t in captured), None)
    if chunk_t is None:
        pytest.skip("need T=5 or T=25 captured")
    torch.manual_seed(4242)
    seq = {
        0: torch.randint(0, vocab, (n_vq, chunk_t * 3), device=DEVICE, dtype=torch.long)
    }
    runner = session.cg_runner
    session.cg_runner = None  # pure-eager reference
    eager_ref = decode_chunks(session, seq, chunk_t)[0]

    session.cg_runner = runner  # graph path, but make the next replay blow up
    session.reset_slots([0])

    def boom(*args, **kwargs):
        raise RuntimeError("simulated replay failure")

    orig_decode = runner.decode_step
    runner.decode_step = boom
    try:
        with pytest.raises(RuntimeError):
            session.step({0: seq[0][:, :chunk_t]})
        assert session.slot_offsets[0] == 0
        assert (
            session.cg_runner is None
        ), "runner must be disabled after a replay failure"
        # session is now eager-only -> a fresh decode must be bit-identical to the pure-eager reference
        after = decode_chunks(session, seq, chunk_t)[0]
        assert torch.equal(after, eager_ref), (
            "post-failure eager output not bit-identical to eager reference: "
            f"max|delta|={(after - eager_ref).abs().max().item():.3e}"
        )
    finally:
        # restore the module-scoped session for the remaining tests
        runner.decode_step = orig_decode
        session.cg_runner = runner


@pytest.mark.skipif(not HAS_ACCELERATOR, reason="needs CUDA/NPU + real codec")
def test_vram_guard_skips_capture_and_falls_back_to_eager(session_bundle):
    """Below the configured VRAM headroom, warmup skips capture (empty graph set, serving uses eager);
    forced via an absurd min_free_gb."""
    from sglang_omni.models.moss_tts.vocoder_cuda_graph import (
        MossVocoderCudaGraphRunner,
    )

    session, n_vq, vocab, captured = session_bundle
    guarded = MossVocoderCudaGraphRunner(
        session.codec,
        real_state_capacity=STREAM_SLOTS,
        scratch_capacity=STREAM_SLOTS,
        batch_sizes=session.graph_batch_sizes(),
        frame_sizes=[5, 25],
        num_quantizers=n_vq,
        min_free_gb=100000.0,  # 100 TB headroom -> always trips
    )
    guarded.warmup([5, 25])
    assert (
        guarded.captured_frames() == []
    ), "VRAM guard must skip all captures under insufficient headroom"


@pytest.mark.skipif(not HAS_ACCELERATOR, reason="needs CUDA/NPU + real codec")
def test_capture_failure_contract(session_bundle):
    """NPU capture failure aborts initialization; CUDA keeps its eager fallback."""
    from sglang_omni.models.moss_tts.vocoder_cuda_graph import (
        MossVocoderCudaGraphRunner,
    )

    session, n_vq, vocab, captured = session_bundle
    runner = MossVocoderCudaGraphRunner(
        session.codec,
        real_state_capacity=STREAM_SLOTS,
        scratch_capacity=STREAM_SLOTS,
        batch_sizes=session.graph_batch_sizes(),
        frame_sizes=[5, 25],
        num_quantizers=n_vq,
    )

    def boom(batch_size, frame_size):
        raise RuntimeError("simulated capture OOM")

    runner.capture = boom
    if DEVICE_TYPE == "npu":
        with pytest.raises(RuntimeError, match="simulated capture OOM"):
            runner.warmup([5, 25])
    else:
        runner.warmup([5, 25])
    assert (
        runner.captured_frames() == []
    ), "capture failures must be caught per-T -> no graphs -> eager"
    assert runner.sealed, "runner must still seal after capture failures"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
