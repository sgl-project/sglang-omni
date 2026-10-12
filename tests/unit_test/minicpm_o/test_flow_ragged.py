# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
import torch

from sglang_omni.models.minicpm_o.components.token2wav.conformer import (
    UpsampleConformerEncoderV2,
)
from sglang_omni.models.minicpm_o.components.token2wav.dit import DiT
from sglang_omni.models.minicpm_o.components.token2wav.flow import (
    CausalConditionalCFM,
    CausalMaskedDiffWithXvec,
)
from sglang_omni.models.minicpm_o.components.token2wav.vocoder import (
    FLOW_CACHE_TAIL_FRAMES,
    MEL_CACHE_FRAMES,
    SAMPLES_PER_MEL_FRAME,
    SpeakerPrompt,
    Token2Wav,
)
from sglang_omni.models.minicpm_o.components.tts_runtime import (
    CODEC_CHUNK_SIZE,
    MiniCPMOVocoderRuntime,
    SynthesisRequest,
)
from sglang_omni.models.minicpm_o.native_config import (
    DEFAULT_SPEECH_STATE_BYTES_PER_SESSION,
)

MEL_CHANNELS = 8
# note (Junnan Li): The silence token that pads every stream is id 4218, so the vocabulary keeps the checkpoint's size.
VOCABULARY_SIZE = 6561
SPEAKER_DIMENSION = 4
PROMPT_TOKENS = 8
UP_RATE = 2
N_TIMESTEPS = 3
WINDOW_TOKENS = 28


def tiny_flow() -> CausalMaskedDiffWithXvec:
    torch.manual_seed(0)
    encoder = UpsampleConformerEncoderV2(
        input_size=16,
        output_size=16,
        num_blocks=1,
        num_up_blocks=1,
        up_stride=UP_RATE,
        attention_heads=2,
        linear_units=32,
        dropout_rate=0.0,
        positional_dropout_rate=0.0,
        attention_dropout_rate=0.0,
    )
    estimator = DiT(
        in_channels=4 * MEL_CHANNELS,
        out_channels=MEL_CHANNELS,
        depth=2,
        num_heads=2,
        head_dim=8,
        hidden_size=16,
    )
    for parameter in estimator.parameters():
        torch.nn.init.normal_(parameter, std=0.1)
    flow = CausalMaskedDiffWithXvec(
        encoder,
        CausalConditionalCFM(estimator),
        input_size=16,
        output_size=MEL_CHANNELS,
        spk_embed_dim=SPEAKER_DIMENSION,
        vocab_size=VOCABULARY_SIZE,
    )
    return flow.eval()


def random_token_ids(length: int, seed: int) -> torch.Tensor:
    return torch.randint(
        0, VOCABULARY_SIZE, (1, length), generator=torch.Generator().manual_seed(seed)
    )


def speaker_embedding(seed: int) -> torch.Tensor:
    return torch.randn(
        1, SPEAKER_DIMENSION, generator=torch.Generator().manual_seed(seed)
    )


def prompt_mel(seed: int) -> torch.Tensor:
    return torch.randn(
        1,
        PROMPT_TOKENS * UP_RATE,
        MEL_CHANNELS,
        generator=torch.Generator().manual_seed(seed),
    )


def clone(cache: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in cache.items()}


def warmed_caches(
    flow: CausalMaskedDiffWithXvec, warmup_chunks: list[int]
) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    speakers, caches = [], []
    for stream, chunks in enumerate(warmup_chunks):
        speaker = speaker_embedding(stream)
        cache = flow.setup_cache(
            random_token_ids(PROMPT_TOKENS + flow.pre_lookahead_len, stream),
            prompt_mel(stream),
            speaker,
            n_timesteps=N_TIMESTEPS,
        )
        for step in range(chunks):
            ((_, cache),) = flow.inference_chunks(
                [random_token_ids(WINDOW_TOKENS, 100 * stream + step)],
                speaker,
                [cache],
                is_last_chunk=[False],
                n_timesteps=N_TIMESTEPS,
            )
        speakers.append(speaker)
        caches.append(cache)
    return speakers, caches


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.accelerator)]
)
@pytest.mark.parametrize(
    ("lengths", "is_last_chunk", "warmup_chunks"),
    [
        ((28, 10, 15), [False, False, False], [0, 1, 2]),
        ((28, 9, 15, 28), [False, True, False, False], [1, 1, 0, 2]),
        ((28, 28), [False, False], [1, 1]),
    ],
)
def test_ragged_chunks_match_each_stream_alone(
    device: str,
    lengths: tuple[int, ...],
    is_last_chunk: list[bool],
    warmup_chunks: list[int],
) -> None:
    flow = tiny_flow()
    with torch.inference_mode():
        speakers, caches = warmed_caches(flow, warmup_chunks)
        flow.to(device)
        speakers = [speaker.to(device) for speaker in speakers]
        caches = [
            {key: value.to(device) for key, value in cache.items()} for cache in caches
        ]
        if device == "cuda":
            flow.decoder.capture_chunk_graphs(
                stream_counts=(1, 2, 4),
                frame_counts=(
                    (WINDOW_TOKENS - flow.pre_lookahead_len) * UP_RATE,
                    (WINDOW_TOKENS - 1) * UP_RATE,
                ),
                history_capacity=PROMPT_TOKENS * UP_RATE + FLOW_CACHE_TAIL_FRAMES,
                convolution_cache=caches[0]["estimator_convolution_cache"],
                attention_cache=caches[0]["estimator_attention_cache"],
            )
        else:
            pass
        token_ids = [
            random_token_ids(length, 1000 + row).to(device)
            for row, length in enumerate(lengths)
        ]
        together = flow.inference_chunks(
            token_ids,
            torch.cat(speakers),
            [clone(cache) for cache in caches],
            is_last_chunk=is_last_chunk,
            n_timesteps=N_TIMESTEPS,
        )
        for row in range(len(lengths)):
            ((mel, cache),) = flow.inference_chunks(
                [token_ids[row]],
                speakers[row],
                [clone(caches[row])],
                is_last_chunk=[is_last_chunk[row]],
                n_timesteps=N_TIMESTEPS,
            )
            batched_mel, batched_cache = together[row]
            torch.testing.assert_close(batched_mel, mel, rtol=1e-4, atol=1e-4)
            assert batched_cache.keys() == cache.keys()
            for key in cache:
                torch.testing.assert_close(
                    batched_cache[key], cache[key], rtol=1e-4, atol=1e-4
                )


class MelEnvelopeVocoder(torch.nn.Module):
    """Deterministic stand-in for HiFT, which draws fresh noise on every call."""

    def forward(
        self, mel: torch.Tensor, source: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        speech = torch.tanh(mel.mean(dim=1)).repeat_interleave(
            SAMPLES_PER_MEL_FRAME, dim=1
        )
        return speech, speech[:, None, :]


def tiny_token2wav() -> Token2Wav:
    token2wav = Token2Wav.__new__(Token2Wav)
    torch.nn.Module.__init__(token2wav)
    token2wav.device = torch.device("cpu")
    token2wav.dtype = torch.float32
    token2wav.n_timesteps = N_TIMESTEPS
    token2wav.flow = tiny_flow()
    token2wav.hift = MelEnvelopeVocoder()
    token2wav.mel_cache_len = MEL_CACHE_FRAMES
    token2wav.source_cache_len = MEL_CACHE_FRAMES * SAMPLES_PER_MEL_FRAME
    token2wav.speech_window = torch.from_numpy(
        np.hamming(2 * token2wav.source_cache_len)
    )
    return token2wav


class TinyCode2Wav:
    def __init__(self, token2wav: Token2Wav) -> None:
        self.token2wav = token2wav
        self.decode_stream = torch.cpu.Stream()

    def resolve_reference_key(self, reference: bytes) -> tuple[str, bytes]:
        return reference.decode(), reference

    def prepare_references(self, references: list[bytes]) -> list[SpeakerPrompt]:
        return [
            SpeakerPrompt(
                prompt_tokens=random_token_ids(PROMPT_TOKENS, int(reference)),
                prompt_token_lengths=torch.tensor([PROMPT_TOKENS]),
                speaker_embedding=speaker_embedding(int(reference)),
                prompt_mel=prompt_mel(int(reference)),
            )
            for reference in references
        ]


SESSIONS = 6
# note (Junnan Li): Turn-start flushes of 25, 10 and 40 tokens give chunks of 28, 13, then 28 and 18 tokens; 30 tokens mid-turn give one 28-token chunk; the last two units end their turns with a last chunk.
FINAL_UNITS = (
    (CODEC_CHUNK_SIZE, True, False),
    (10, True, False),
    (40, True, False),
    (30, False, False),
    (12, False, True),
    (CODEC_CHUNK_SIZE, True, True),
)


def unit(
    session: int, index: int, length: int, is_turn_start: bool, end_of_turn: bool
) -> SynthesisRequest:
    return SynthesisRequest(
        session_id=str(session),
        codec_token_ids=random_token_ids(length, 10_000 * session + index)[0].tolist(),
        is_turn_start=is_turn_start,
        end_of_turn=end_of_turn,
    )


def run_sessions(
    token2wav: Token2Wav, *, together: bool
) -> tuple[MiniCPMOVocoderRuntime, dict[str, np.ndarray | None]]:
    runtime = MiniCPMOVocoderRuntime(
        TinyCode2Wav(token2wav),
        max_state_bytes_per_session=DEFAULT_SPEECH_STATE_BYTES_PER_SESSION,
        max_open_sessions=SESSIONS,
    )
    waveforms: dict[str, np.ndarray | None] = {}

    def synthesize(requests: list[SynthesisRequest]) -> None:
        for position, waveform in runtime.synthesize_batch(requests):
            waveforms[requests[position].session_id] = waveform

    for session in range(SESSIONS):
        runtime.open_session(str(session), reference_audio=str(session).encode())
        for index in range(session % 4):
            synthesize([unit(session, index, CODEC_CHUNK_SIZE, index == 0, False)])
    final_units = [
        unit(session, 99, *FINAL_UNITS[session % len(FINAL_UNITS)])
        for session in range(SESSIONS)
    ]
    if together:
        synthesize(final_units)
    else:
        for final_unit in final_units:
            synthesize([final_unit])
    return runtime, waveforms


def test_sessions_decoded_together_match_each_session_alone() -> None:
    token2wav = tiny_token2wav()
    # note (Junnan Li): Two warm-up chunks reach exactly the prompt plus the kept tail, so session 2's next chunk is its first trimmed one.
    assert 2 * CODEC_CHUNK_SIZE * UP_RATE == FLOW_CACHE_TAIL_FRAMES
    alone, alone_waveforms = run_sessions(token2wav, together=False)
    together, together_waveforms = run_sessions(token2wav, together=True)

    for session in map(str, range(SESSIONS)):
        alone_flow_cache = alone.sessions[session].caches[0]
        together_flow_cache = together.sessions[session].caches[0]
        assert together_flow_cache.keys() == alone_flow_cache.keys()
        for key in alone_flow_cache:
            torch.testing.assert_close(
                together_flow_cache[key], alone_flow_cache[key], rtol=1e-5, atol=1e-5
            )
        np.testing.assert_allclose(
            together_waveforms[session], alone_waveforms[session], atol=2 / 32768
        )
    assert (
        together.sessions["2"].caches[0]["estimator_attention_cache"].shape[4]
        == PROMPT_TOKENS * UP_RATE + FLOW_CACHE_TAIL_FRAMES
    )
