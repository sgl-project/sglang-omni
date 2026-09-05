# SPDX-License-Identifier: Apache-2.0
"""GPU correctness tests for the MOSS-TTS-Realtime codec CUDA graphs."""

from __future__ import annotations

import ast
import glob
import inspect
import textwrap

import pytest
import torch

from sglang_omni.models.moss_tts_realtime.stages import load_moss_tts_realtime_codec
from sglang_omni.models.moss_tts_realtime.streaming_vocoder import _CodecStreamSession
from sglang_omni.models.moss_tts_realtime.vocoder_cuda_graph import (
    MossTTSRealtimeVocoderCudaGraphRunner,
)
from sglang_omni.models.moss_tts_realtime.vocoder_decoder import (
    configure_moss_tts_realtime_vocoder_decoder,
)

pytestmark = pytest.mark.accelerator

CODEC_GLOB = (
    "/opt/scratch/cache/hf/hub/models--OpenMOSS-Team--MOSS-Audio-Tokenizer/snapshots/*"
)
N_VQ = 16
STREAM_SLOTS = 2
SAMPLES_PER_FRAME = 1920
CAPTURE_FRAME_COUNTS = list(range(1, 13))
FRAME_COUNTS = [1, 2, 3, 6, 12]


@pytest.fixture(
    scope="module",
    params=[torch.float32, torch.bfloat16],
    ids=["float32", "bfloat16"],
)
def session_bundle(request):
    snapshots = glob.glob(CODEC_GLOB)
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if not snapshots:
        pytest.skip("MOSS-Audio-Tokenizer snapshot is unavailable")

    codec = load_moss_tts_realtime_codec(
        snapshots[0],
        component="decoder",
        device="cuda",
        dtype=request.param,
    )
    if request.param != torch.float32:
        assert (
            configure_moss_tts_realtime_vocoder_decoder(
                codec,
                dtype=request.param,
            )
            == 68
        )
    session = _CodecStreamSession(
        codec,
        stream_slots=STREAM_SLOTS,
        n_vq=N_VQ,
        samples_per_frame=SAMPLES_PER_FRAME,
    )
    slots = [session.acquire() for _ in range(STREAM_SLOTS)]
    captured = session.warmup_cuda_graph(CAPTURE_FRAME_COUNTS, min_free_gb=0.0)
    assert captured == CAPTURE_FRAME_COUNTS
    try:
        yield session, slots, request.param
    finally:
        for slot in slots:
            if slot in session._leased_slots:
                session.release(slot)
        session.close()


def _decode_chunks(
    session: _CodecStreamSession,
    slot_codes: dict[int, torch.Tensor],
    frame_count: int,
) -> dict[int, torch.Tensor]:
    slots = list(slot_codes)
    session._state_adapter.reset_slots(slots, batch_size=STREAM_SLOTS)
    total_frames = int(next(iter(slot_codes.values())).shape[1])
    parts = {slot: [] for slot in slots}
    for start in range(0, total_frames, frame_count):
        decoded = session.step(
            {
                slot: codes[:, start : start + frame_count]
                for slot, codes in slot_codes.items()
            }
        )
        for slot in slots:
            parts[slot].append(decoded[slot])
    return {slot: torch.cat(chunks, dim=-1) for slot, chunks in parts.items()}


def test_capture_uses_thread_local_error_mode() -> None:
    source = textwrap.dedent(
        inspect.getsource(MossTTSRealtimeVocoderCudaGraphRunner._capture_frame_count)
    )
    tree = ast.parse(source)
    graph_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "graph"
    ]
    assert graph_calls
    assert any(
        keyword.arg == "capture_error_mode"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value == "thread_local"
        for call in graph_calls
        for keyword in call.keywords
    )


@pytest.mark.parametrize("frame_count", FRAME_COUNTS)
@pytest.mark.parametrize("active_slots", [1, STREAM_SLOTS])
def test_streaming_cuda_graph_is_bit_identical(
    session_bundle,
    frame_count: int,
    active_slots: int,
) -> None:
    session, slots, decoder_dtype = session_bundle
    assert next(session._codec.decoder.parameters()).dtype is decoder_dtype
    torch.manual_seed(100 * frame_count + active_slots)
    slot_codes = {
        slot: torch.randint(
            0,
            1024,
            (N_VQ, frame_count * 3),
            device="cuda",
            dtype=torch.long,
        )
        for slot in slots[:active_slots]
    }

    runner = session._cg_runner
    session._cg_runner = None
    eager = _decode_chunks(session, slot_codes, frame_count)
    session._cg_runner = runner
    graphed = _decode_chunks(session, slot_codes, frame_count)

    for slot in slot_codes:
        assert torch.equal(eager[slot], graphed[slot]), (
            f"frame_count={frame_count} active_slots={active_slots} slot={slot} "
            f"max_delta={(eager[slot] - graphed[slot]).abs().max().item():.3e}"
        )


def test_released_slot_restarts_from_fresh_codec_state(session_bundle) -> None:
    session, slots, _ = session_bundle
    slot = slots[0]
    torch.manual_seed(20260728)
    codes = torch.randint(
        0,
        1024,
        (N_VQ, 6),
        device="cuda",
        dtype=torch.long,
    )

    first = _decode_chunks(session, {slot: codes}, 3)[slot]
    session.release(slot)
    reused = session.acquire()
    assert reused == slot
    second = _decode_chunks(session, {reused: codes}, 3)[reused]

    assert torch.equal(first, second)
