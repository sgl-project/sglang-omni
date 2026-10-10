# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for native duplex image and audio unit plans."""

import copy
import wave
from collections.abc import Callable
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from PIL import Image, UnidentifiedImageError
from transformers import AutoProcessor, ProcessorMixin, WhisperFeatureExtractor

from sglang_omni.models.minicpm_o.components.audio_encoder import (
    MiniCPMOAudioEncoder,
    StreamingAudioChunk,
)
from sglang_omni.models.minicpm_o.components.streaming_perception import (
    CONTEXT_FRAMES,
    DYNAMIC_NORM_MIN_SAMPLES,
    DYNAMIC_RANGE_DB,
    FIRST_CHUNK_MS,
    FIRST_CHUNK_SAMPLES,
    HOP_LENGTH,
    LOG_FLOOR_DB,
    SAMPLE_RATE,
    SLIDE_STRIDE_SAMPLES,
    SLIDE_TRIGGER_SAMPLES,
    UNIT_FRAMES,
    UNIT_MS,
    UNIT_SAMPLES,
    LogMelFilterBank,
    MiniCPMOPerceptionState,
    PerceptionStepPlan,
)
from sglang_omni.models.minicpm_o.native_stages import PerceptionHooks
from sglang_omni.preprocessing.cache_key import hash_bytes
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, TimedChunk
from sglang_omni.scheduling.session import SessionAppend
from tests.unit_test.minicpm_o.test_audio_encoder import (
    checkpoint_dir,
    tiny_audio_encoder,
)


@pytest.fixture
def state() -> MiniCPMOPerceptionState:
    tokenizer = Mock(unk_token_id=0)
    tokenizer.convert_tokens_to_ids.side_effect = {
        "<unit>": 1,
        "<image>": 2,
        "</image>": 3,
        "<slice>": 4,
        "</slice>": 5,
    }.__getitem__
    return MiniCPMOPerceptionState(
        tokenizer=tokenizer,
        processor=Mock(),
        audio_encoder=Mock(),
        image_encoder=Mock(),
        max_slice_nums=1,
        mel_filter_bank=Mock(),
    )


@pytest.mark.parametrize(
    ("chunk_index", "reference_audio"), [(1, False), (1, True), (2, True)]
)
def test_audio_plan_without_image(
    state: MiniCPMOPerceptionState, chunk_index: int, reference_audio: bool
) -> None:
    state.audio_chunk_index = chunk_index
    if reference_audio:
        state.prefix_token_ids = [7, 0, 0, 8]
        state.prefix_schema = [("token", 1), ("audio", 2), ("token", 1)]
        state.prefix_embeds = torch.full((2, 4), 5.0)
        prefix_spans = [
            dict(
                modality="audio",
                token_start=1,
                token_end=3,
                embedding_start=0,
                embedding_end=2,
            )
        ]
    else:
        state.prefix_token_ids = [7, 8]
        state.prefix_schema = [("token", 2)]
        prefix_spans = []
    audio = torch.full((10, 4), 9.0)
    plan = state.build_step_plan(audio)
    is_first_unit = chunk_index == 1
    prefix_ids = state.prefix_token_ids if is_first_unit else []
    spans = prefix_spans if is_first_unit else []
    embedding_start = 2 if is_first_unit and reference_audio else 0
    token_start = len(prefix_ids) + 1
    assert plan["token_ids"] == prefix_ids + [1] + [0] * 10
    assert plan["embedding_spans"] == spans + [
        dict(
            modality="audio",
            token_start=token_start,
            token_end=token_start + 10,
            embedding_start=embedding_start,
            embedding_end=embedding_start + 10,
        )
    ]
    assert plan["input_embeds"].shape[0] == embedding_start + 10
    assert torch.equal(plan["input_embeds"][embedding_start:], audio)


@pytest.mark.parametrize("chunk_index", [1, 2])
def test_image_plan_follows_official_slice_order(
    state: MiniCPMOPerceptionState, chunk_index: int
) -> None:
    state.audio_chunk_index = chunk_index
    state.prefix_token_ids = [7, 0, 0, 8]
    state.prefix_schema = [("token", 1), ("audio", 2), ("token", 1)]
    state.prefix_embeds = torch.full((2, 4), 5.0)
    first = torch.cat([torch.full((64, 4), float(i)) for i in (1, 2, 3)])
    second = torch.full((64, 4), 4.0)
    audio = torch.full((10, 4), 9.0)
    plan = state.build_step_plan(audio, (first, second))
    prefix = state.prefix_token_ids if chunk_index == 1 else []
    offset = len(prefix)
    # note (Junnan Li): Overview, its two slices, then the second frame's overview.
    expected = prefix + [1]
    for open_id, close_id in [(2, 3), (4, 5), (4, 5), (2, 3)]:
        expected += [open_id, *[0] * 64, close_id]
    assert plan["token_ids"] == expected + [0] * 10
    unit_spans = plan["embedding_spans"][-5:]
    assert [span["modality"] for span in unit_spans] == ["image"] * 4 + ["audio"]
    assert [(span["token_start"], span["token_end"]) for span in unit_spans] == [
        (offset + 2 + 66 * index, offset + 66 + 66 * index) for index in range(4)
    ] + [(offset + 265, offset + 275)]
    blocks = [state.prefix_embeds] if prefix else []
    blocks += [*first.split(64), second, audio]
    assert torch.equal(plan["input_embeds"], torch.cat(blocks))
    for span, block in zip(plan["embedding_spans"], blocks, strict=True):
        assert torch.equal(
            plan["input_embeds"][span["embedding_start"] : span["embedding_end"]], block
        )


def encoded_image(format: str) -> bytes:
    encoded = BytesIO()
    Image.new("RGB", (16, 16)).save(encoded, format=format)
    return encoded.getvalue()


@pytest.mark.parametrize(
    ("encoded", "error", "match", "pixel_limit"),
    [
        (encoded_image("GIF"), ValueError, "JPEG or PNG", None),
        (b"\x89PNG\r\n\x1a\ninvalid", UnidentifiedImageError, None, None),
        (encoded_image("PNG")[:45], OSError, None, None),
        (encoded_image("PNG"), ValueError, "pixel limit", 15),
    ],
)
def test_reject_frame_before_processor(
    state: MiniCPMOPerceptionState,
    monkeypatch: pytest.MonkeyPatch,
    encoded: bytes,
    error: type[Exception],
    match: str | None,
    pixel_limit: int | None,
) -> None:
    if pixel_limit is not None:
        monkeypatch.setattr(
            "sglang_omni.models.minicpm_o.components.streaming_perception.MAX_FRAME_PIXELS",
            pixel_limit,
        )
    else:
        pass
    with pytest.raises(error, match=match):
        state.encode_image(encoded)
    state.processor.process_image.assert_not_called()


SESSION_STARTS = {"early": 0, "next": 1, "late": 7}
UNITS_PER_SESSION = 31


# note (Junnan Li): The unit whose last samples become the mel buffer's first ones at the 30 s slide.
SLIDE_EDGE_UNIT = 9
SLIDE_EDGE_OFFSET = UNIT_SAMPLES - (FIRST_CHUNK_MS * SAMPLE_RATE // 1000 - UNIT_SAMPLES)
CLICK_SAMPLES = 40
# note (Junnan Li): The slide drops the burst, so the buffer peak moves to the click in the reflect-padded edge frames.
BURST_UNIT = 3
BURST_LOUDNESS = 0.9
CLICK_LOUDNESS = 0.3
SILENT_UNIT_PERIOD = 6


def session_units(seed: int) -> list[np.ndarray]:
    """One session's 1 s PCM units: noise, silent units, a loud burst and a click at the slide's new buffer start."""
    generator = np.random.default_rng(seed)
    units = []
    for index in range(UNITS_PER_SESSION):
        if index == BURST_UNIT:
            loudness = BURST_LOUDNESS
        elif index % SILENT_UNIT_PERIOD == 1:
            loudness = 0.0
        else:
            loudness = generator.uniform(0.001, 0.2)
        samples = generator.standard_normal(UNIT_SAMPLES) * loudness
        if index == SLIDE_EDGE_UNIT:
            samples[SLIDE_EDGE_OFFSET : SLIDE_EDGE_OFFSET + CLICK_SAMPLES] = (
                CLICK_LOUDNESS
            )
        else:
            pass
        pcm = np.clip(samples * 32768, -32768, 32767).astype("<i2")
        units.append(pcm.astype(np.float32) / 32768.0)
    return units


def whole_buffer_mel_chunks(
    units: list[np.ndarray], mel_filter_bank: LogMelFilterBank
) -> list[torch.Tensor]:
    """The checkpoint's exact streaming mel: the whole buffer's mel recomputed every unit."""
    buffer = np.zeros(0, dtype=np.float32)
    pending = np.zeros(0, dtype=np.float32)
    first_frame = 0
    core_frame = 0
    chunks = []
    for index, waveform in enumerate(units):
        if index == 0:
            padding = max(FIRST_CHUNK_MS * SAMPLE_RATE // 1000 - waveform.size, 0)
            chunk_samples = FIRST_CHUNK_SAMPLES
        else:
            padding = 0
            chunk_samples = UNIT_SAMPLES
        pending = np.concatenate(
            [np.zeros(padding, dtype=np.float32), pending, waveform]
        )
        buffer = np.concatenate([buffer, pending[:chunk_samples]])
        pending = pending[chunk_samples:]
        if buffer.size >= SLIDE_TRIGGER_SAMPLES:
            buffer = buffer[SLIDE_STRIDE_SAMPLES:]
            first_frame += SLIDE_STRIDE_SAMPLES // HOP_LENGTH
        else:
            pass
        log_mel = mel_filter_bank.log_mel(torch.from_numpy(buffer)[None])[0]
        if buffer.size < DYNAMIC_NORM_MIN_SAMPLES:
            threshold = torch.tensor(LOG_FLOOR_DB)
        else:
            threshold = log_mel.max() - DYNAMIC_RANGE_DB
        mel = (torch.maximum(log_mel, threshold) + 4.0) / 4.0
        start = max(core_frame - CONTEXT_FRAMES, 0) - first_frame
        end = core_frame + UNIT_FRAMES + CONTEXT_FRAMES - first_frame
        chunks.append(mel[:, start:end])
        core_frame += UNIT_FRAMES
    return chunks


def checkpoint_mel_chunks(
    processor: ProcessorMixin, units: list[np.ndarray]
) -> list[torch.Tensor]:
    """The checkpoint processor's own streaming mel, configured as the official duplex demo does."""
    session_processor = copy.copy(processor)
    session_processor.set_streaming_mode(
        mode="exact",
        chunk_ms=UNIT_MS,
        first_chunk_ms=FIRST_CHUNK_MS,
        cnn_redundancy_ms=20,
        enable_sliding_window=True,
        slide_trigger_seconds=30.0,
        slide_stride_seconds=10.0,
    )
    pending = np.zeros(0, dtype=np.float32)
    chunks = []
    for index, waveform in enumerate(units):
        if index == 0:
            padding = max(FIRST_CHUNK_MS * SAMPLE_RATE // 1000 - waveform.size, 0)
            chunk_samples = FIRST_CHUNK_SAMPLES
        else:
            padding = 0
            chunk_samples = UNIT_SAMPLES
        pending = np.concatenate(
            [np.zeros(padding, dtype=np.float32), pending, waveform]
        )
        assert session_processor.get_streaming_chunk_size() == chunk_samples
        features = session_processor.process_audio_streaming(
            pending[:chunk_samples].copy(), reset=False, return_batch_feature=True
        )["audio_features"]
        chunks.append(features[0])
        pending = pending[chunk_samples:]
    return chunks


def run_staggered_sessions(
    mel_filter_bank: LogMelFilterBank, units: dict[str, list[np.ndarray]]
) -> tuple[
    dict[str, list[torch.Tensor]],
    dict[str, list[PerceptionStepPlan]],
    MiniCPMOAudioEncoder,
]:
    """Drive PerceptionHooks with sessions that start at different units, so batches mix first and later units."""
    tokenizer = Mock(unk_token_id=0)
    tokenizer.convert_tokens_to_ids.side_effect = {
        "<unit>": 1,
        "<image>": 2,
        "</image>": 3,
        "<slice>": 4,
        "</slice>": 5,
    }.__getitem__
    encoder = tiny_audio_encoder(pool_step=5)
    hooks = PerceptionHooks(
        tokenizer,
        Mock(),
        encoder,
        reference_audio=b"",
        image_encoder=Mock(),
        mel_filter_bank=mel_filter_bank,
        reference_cache_capacity=1,
    )
    features = {name: [] for name in units}
    plans = {name: [] for name in units}
    for name in units:
        state = MiniCPMOPerceptionState(
            tokenizer=tokenizer,
            processor=Mock(),
            audio_encoder=encoder,
            image_encoder=Mock(),
            max_slice_nums=1,
            mel_filter_bank=mel_filter_bank,
        )
        cut_chunk = state.mel_chunk

        def recording_mel_chunk(
            window_log_mel: torch.Tensor,
            cut_chunk: Callable[[torch.Tensor], StreamingAudioChunk] = cut_chunk,
            name: str = name,
        ) -> StreamingAudioChunk:
            chunk = cut_chunk(window_log_mel)
            features[name].append(chunk.audio_features[0])
            return chunk

        state.mel_chunk = recording_mel_chunk
        hooks.states[SessionIdentity(name)] = state
    last_tick = max(SESSION_STARTS.values()) + UNITS_PER_SESSION
    for tick in range(last_tick):
        appends = []
        for name, start in SESSION_STARTS.items():
            index = tick - start
            if 0 <= index < UNITS_PER_SESSION:
                pcm = (units[name][index] * 32768).astype("<i2").tobytes()
                appends.append(
                    SessionAppend(
                        chunk=TimedChunk("audio", 0, UNIT_MS, index, pcm),
                        payload=StagePayload("unit", OmniRequest(None), None),
                        context=SimpleNamespace(session_identity=SessionIdentity(name)),
                    )
                )
            else:
                pass
        for append, payload in zip(appends, hooks.append_batch(appends), strict=True):
            plans[append.context.session_identity.id].append(payload.data)
    return features, plans, encoder


@pytest.mark.parametrize("reference", ["whole_buffer", "checkpoint"])
def test_batched_perception_matches_each_session_alone(reference: str) -> None:
    """Batched mel and encoding give every session the checkpoint's own features and step plans."""
    units = {
        name: session_units(seed) for seed, name in enumerate(SESSION_STARTS, start=1)
    }
    if reference == "checkpoint":
        checkpoint = checkpoint_dir()
        if checkpoint is None:
            pytest.skip("no MiniCPM-o checkpoint with remote processing files")
        else:
            processor = AutoProcessor.from_pretrained(
                str(checkpoint), trust_remote_code=True
            )
            mel_filter_bank = LogMelFilterBank.from_feature_extractor(
                processor.audio_processor
            )
            expected_features = {
                name: checkpoint_mel_chunks(processor, session)
                for name, session in units.items()
            }
    else:
        mel_filter_bank = LogMelFilterBank.from_feature_extractor(
            WhisperFeatureExtractor(feature_size=80)
        )
        expected_features = {
            name: whole_buffer_mel_chunks(session, mel_filter_bank)
            for name, session in units.items()
        }
    features, plans, encoder = run_staggered_sessions(mel_filter_bank, units)
    for name in units:
        assert len(features[name]) == UNITS_PER_SESSION
        state = None
        for index, (actual, expected, plan) in enumerate(
            zip(features[name], expected_features[name], plans[name], strict=True)
        ):
            assert torch.equal(actual, expected), (name, index)
            [(alone_embeds, state)] = encoder.forward_streaming_batch(
                [
                    StreamingAudioChunk(
                        audio_features=expected[None],
                        state=state,
                        prefix_extra_frames=0 if index == 0 else CONTEXT_FRAMES,
                        suffix_extra_frames=CONTEXT_FRAMES,
                    )
                ]
            )
            assert plan["token_ids"] == [1] + [0] * alone_embeds.shape[0]
            torch.testing.assert_close(
                plan["input_embeds"], alone_embeds, rtol=1e-5, atol=1e-5
            )


REFERENCE_MEL_FRAMES = 200


def wav_bytes(seed: int) -> bytes:
    """One second of 16 kHz mono PCM noise as a WAV file."""
    samples = (np.random.default_rng(seed).standard_normal(SAMPLE_RATE) * 3000).astype(
        "<i2"
    )
    encoded = BytesIO()
    with wave.open(encoded, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(samples.tobytes())
    return encoded.getvalue()


def reference_processor() -> Mock:
    return Mock(
        process_audio=Mock(
            side_effect=lambda audio, sampling_rate: {
                "audio_features": torch.from_numpy(
                    np.resize(audio, (1, 80, REFERENCE_MEL_FRAMES))
                ),
                "audio_feature_lens": [torch.tensor([REFERENCE_MEL_FRAMES])],
            }
        )
    )


def test_sessions_share_one_encoding_per_reference_audio() -> None:
    """Opening sessions encodes each distinct reference once; the least recently used one is evicted past the capacity."""
    tokenizer = Mock(unk_token_id=0)
    tokenizer.encode.return_value = [7]
    encoder = tiny_audio_encoder(pool_step=5)
    encoder.forward = Mock(wraps=encoder.forward)
    default_reference, other_reference = wav_bytes(seed=1), wav_bytes(seed=2)
    hooks = PerceptionHooks(
        tokenizer,
        reference_processor(),
        encoder,
        reference_audio=default_reference,
        image_encoder=Mock(),
        mel_filter_bank=Mock(),
        reference_cache_capacity=1,
    )
    for name, params in (
        ("first", {}),
        ("second", {}),
        ("own", {"reference_audio": other_reference}),
        ("third", {}),
    ):
        hooks.open(
            SessionIdentity(name),
            OmniRequest(
                None,
                params={"instructions": "", "max_slice_nums": 1, **params},
            ),
        )
    states = {identity.id: state for identity, state in hooks.states.items()}
    assert states["first"].prefix_embeds is states["second"].prefix_embeds
    assert not torch.equal(states["first"].prefix_embeds, states["own"].prefix_embeds)
    assert torch.equal(states["first"].prefix_embeds, states["third"].prefix_embeds)
    # note (Junnan Li): Capacity 1: the default, the session's own, then the default again after its eviction.
    assert encoder.forward.call_count == 3


def test_warm_up_keeps_no_session_state() -> None:
    hooks = PerceptionHooks(
        Mock(unk_token_id=0),
        reference_processor(),
        tiny_audio_encoder(pool_step=5),
        reference_audio=wav_bytes(seed=0),
        image_encoder=Mock(),
        mel_filter_bank=LogMelFilterBank.from_feature_extractor(
            WhisperFeatureExtractor(feature_size=80)
        ),
        reference_cache_capacity=1,
    )
    hooks.warm_up(2)
    assert hooks.states == {}
    assert list(hooks.reference_embeds_cache) == [hash_bytes(hooks.reference_audio)]
