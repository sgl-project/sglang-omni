# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict

import numpy as np
import pytest
import torch
from transformers.cache_utils import DynamicCache

from sglang_omni.models.nemotron3_5_asr.decoder import Nemotron3_5ASRDecodeState
from sglang_omni.models.nemotron3_5_asr.hf_compat import (
    Nemotron3_5AsrConfig,
    Nemotron3_5AsrRNNTDecoderCache,
)
from sglang_omni.models.nemotron3_5_asr.model_runner import (
    Nemotron3_5ASRModelRunner,
    Nemotron3_5ASRPreparedChunk,
    Nemotron3_5ASRStreamingBatchResult,
)
from sglang_omni.models.nemotron3_5_asr.streaming import (
    Nemotron3_5ASRStreamingChunkSpec,
    Nemotron3_5ASRStreamState,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto.request import OmniRequest, StagePayload

LOOKAHEAD_3 = Nemotron3_5ASRStreamingChunkSpec(
    sample_rate=16000,
    first_samples=4040,
    subsequent_samples=5520,
    first_frames=25,
    subsequent_frames=32,
    hop_length=160,
    n_fft=512,
    streaming_latency_ms=80,
)


def make_decode_state() -> Nemotron3_5ASRDecodeState:
    return Nemotron3_5ASRDecodeState(
        tokens=[99],
        durations=[0],
        attention_cache=DynamicCache(),
        decoder_cache=Nemotron3_5AsrRNNTDecoderCache(Nemotron3_5AsrConfig()),
    )


def make_payload(request_id: str, *, language: str = "en-US") -> StagePayload:
    payload = StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs=None, params={"language": language}),
        data=None,
    )
    return payload


def make_pcm_item(request_id: str, samples: np.ndarray) -> tuple[str, StreamItem]:
    return request_id, StreamItem(
        chunk_id=0,
        data=torch.from_numpy(samples.astype(np.int16, copy=False)),
        from_stage="test",
        metadata={"sample_rate": 16000, "modality": "pcm16"},
    )


class FakeRunner(Nemotron3_5ASRModelRunner):
    def __init__(self) -> None:
        self.batches: list[list[Nemotron3_5ASRDecodeState]] = []
        self.is_closed = False

    @property
    def streaming_state_budget_bytes(self) -> int:
        return 1024

    @property
    def prompt_dictionary(self) -> dict[str, int]:
        return {"auto": 101, "en-US": 0, "zh-CN": 4}

    @property
    def streaming_chunk_spec(self) -> dict[str, int]:
        return asdict(LOOKAHEAD_3)

    def new_streaming_decode_state(self) -> Nemotron3_5ASRDecodeState:
        return make_decode_state()

    def prepare_streaming_chunk(
        self, waveform: np.ndarray, *, language: str, is_first: bool
    ) -> Nemotron3_5ASRPreparedChunk:
        return Nemotron3_5ASRPreparedChunk(
            input_features=torch.from_numpy(waveform),
            prompt_ids=torch.tensor([self.prompt_dictionary[language]]),
        )

    def run_streaming_batch(
        self,
        states: Sequence[Nemotron3_5ASRDecodeState],
        chunks: Sequence[Nemotron3_5ASRPreparedChunk],
        *,
        requested_languages: Sequence[str],
        max_new_tokens: Sequence[int | None] | None = None,
    ) -> Nemotron3_5ASRStreamingBatchResult:
        self.batches.append(list(states))
        raw_texts = []
        clean_texts = []
        for state in states:
            state.tokens.append(len(state.tokens))
            state.durations.append(1)
            state.decoder_steps += 1
            state.encoder_frames += 2
            text = "word" + " more" * (state.decoder_steps - 1)
            raw_texts.append(f"<en-US> {text}")
            clean_texts.append(text)
        return Nemotron3_5ASRStreamingBatchResult(
            elapsed_s=0.001,
            raw_texts=raw_texts,
            clean_texts=clean_texts,
            languages=["en-US"] * len(states),
        )

    def close(self) -> None:
        self.is_closed = True


def test_pcm16_fragmentation_and_final_padding_geometry() -> None:
    state = Nemotron3_5ASRStreamState(
        request_id="r",
        payload=make_payload("r"),
        language="en-US",
        spec=LOOKAHEAD_3,
        decode=make_decode_state(),
    )
    waveform = np.arange(5000, dtype=np.int16)
    raw_bytes = waveform.astype("<i2", copy=False).view(np.uint8)
    boundaries = (0, 1025, 3074, 7171, raw_bytes.size)
    for start, end in zip(boundaries, boundaries[1:]):
        state.append_bytes(raw_bytes[start:end].tobytes())

    first = state.pop_ready_window()
    np.testing.assert_array_equal(first.waveform, waveform[:4040] / 32768.0)

    state.mark_done()
    final = state.pop_ready_window()
    np.testing.assert_array_equal(
        final.waveform, np.pad(waveform[3744:] / 32768.0, (0, 4264))
    )
    assert state.total_samples / LOOKAHEAD_3.sample_rate == pytest.approx(0.3125)


def test_lookahead_zero_preserves_negative_stft_start() -> None:
    spec = Nemotron3_5ASRStreamingChunkSpec(
        sample_rate=16000,
        first_samples=200,
        subsequent_samples=1680,
        first_frames=1,
        subsequent_frames=8,
        hop_length=160,
        n_fft=512,
        streaming_latency_ms=20,
    )
    state = Nemotron3_5ASRStreamState(
        request_id="r",
        payload=make_payload("r"),
        language="en-US",
        spec=spec,
        decode=make_decode_state(),
    )
    state.append_bytes(np.arange(300, dtype="<i2").tobytes())
    state.pop_ready_window()
    state.mark_done()
    final = state.pop_ready_window()
    np.testing.assert_array_equal(
        final.waveform, np.pad(np.arange(300) / 32768.0, (96, 1284))
    )
