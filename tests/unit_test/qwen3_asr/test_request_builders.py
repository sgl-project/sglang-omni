# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import sglang_omni.models.qwen3_asr.request_builders as request_builders
from sglang_omni.models.qwen3_asr.audio_lengths import (
    qwen3_asr_audio_token_lengths,
    qwen3_asr_num_audio_tokens,
)
from sglang_omni.models.qwen3_asr.configuration_qwen3_asr import Qwen3ASRProcessor
from sglang_omni.models.qwen3_asr.request_builders import (
    Qwen3ASRRequestData,
    _try_load_pcm_wav_fast,
    load_audio,
    make_qwen3_asr_scheduler_adapters,
)
from sglang_omni.proto import OmniRequest, StagePayload


def _wav_bytes(audio: np.ndarray, *, sample_rate: int) -> bytes:
    if audio.ndim == 1:
        pcm = audio[:, None]
    else:
        pcm = audio
    pcm16 = np.clip(pcm * 32767.0, -32768, 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(pcm16.shape[1])
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())
    return buffer.getvalue()


def _wav_bytes_uint8(audio: np.ndarray, *, sample_rate: int) -> bytes:
    if audio.ndim == 1:
        pcm = audio[:, None]
    else:
        pcm = audio
    pcm8 = np.clip((pcm + 1.0) * 127.5, 0, 255).astype("u1")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(pcm8.shape[1])
        wav_file.setsampwidth(1)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm8.tobytes())
    return buffer.getvalue()


def test_qwen3_asr_fast_pcm_wav_loads_mono_16k_bytes() -> None:
    audio = np.array([0.0, 0.5, -0.5, 0.25], dtype=np.float32)
    loaded = load_audio(_wav_bytes(audio, sample_rate=16000))

    expected = np.array([0.0, 16383 / 32768.0, -16383 / 32768.0, 8191 / 32768.0])
    np.testing.assert_allclose(loaded, expected, atol=1e-6)


def test_qwen3_asr_fast_pcm_wav_loads_path(tmp_path: Path) -> None:
    audio = np.array([0.0, 0.5, -0.5, 0.25], dtype=np.float32)
    wav_path = tmp_path / "mono.wav"
    wav_path.write_bytes(_wav_bytes(audio, sample_rate=16000))

    loaded = load_audio(str(wav_path))

    expected = np.array([0.0, 16383 / 32768.0, -16383 / 32768.0, 8191 / 32768.0])
    np.testing.assert_allclose(loaded, expected, atol=1e-6)


def test_qwen3_asr_fast_pcm_wav_loads_stereo_without_resample() -> None:
    audio = np.array(
        [
            [0.25, -0.25],
            [0.5, -0.5],
        ],
        dtype=np.float32,
    )
    fast = _try_load_pcm_wav_fast(_wav_bytes(audio, sample_rate=16000))

    assert fast is not None
    loaded_channels_first, sample_rate = fast
    assert sample_rate == 16000
    expected = np.array(
        [
            [8191 / 32768.0, 16383 / 32768.0],
            [-8191 / 32768.0, -16383 / 32768.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(loaded_channels_first, expected, atol=1e-6)


def test_qwen3_asr_fast_pcm_wav_rejects_malformed_bytes() -> None:
    assert _try_load_pcm_wav_fast(b"not a wav") is None


def test_qwen3_asr_fast_pcm_wav_rejects_unsupported_sample_width() -> None:
    audio = np.array([0.0, 0.5, -0.5, 0.25], dtype=np.float32)

    assert _try_load_pcm_wav_fast(_wav_bytes_uint8(audio, sample_rate=16000)) is None


def test_qwen3_asr_fast_pcm_wav_matches_torchaudio_path() -> None:
    import torchaudio

    rng = np.random.default_rng(0)
    audio = rng.uniform(-0.8, 0.8, size=(257, 2)).astype(np.float32)
    wav = _wav_bytes(audio, sample_rate=16000)
    fast = load_audio(wav)

    baseline, sample_rate = torchaudio.load(io.BytesIO(wav))
    assert sample_rate == 16000
    baseline = baseline.mean(dim=0).to(torch.float32).cpu().numpy()

    np.testing.assert_allclose(fast, baseline, atol=1e-6)


class _FakeTokenizer:
    eos_token_id = 2
    vocab_size = 1000

    def __init__(self) -> None:
        self.encode_calls: list[str] = []
        self.decode_calls: list[dict] = []

    def convert_tokens_to_ids(self, token: str) -> int:
        assert token == "<|audio_pad|>"
        return 42

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        self.encode_calls.append(text)
        assert text == "<asr_text>"
        return [100, 101]

    def __call__(self, text: str, *, add_special_tokens: bool = False):
        assert not add_special_tokens
        audio_pad_count = text.count("<|audio_pad|>")
        return SimpleNamespace(input_ids=[11] + [42] * audio_pad_count + [12, 13, 14])

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
    ) -> str:
        self.decode_calls.append(
            {
                "token_ids": list(token_ids),
                "skip_special_tokens": skip_special_tokens,
                "clean_up_tokenization_spaces": clean_up_tokenization_spaces,
            }
        )
        pieces = {
            10: "language English",
            100: "<asr_text>",
            101: "",
            20: " leading",
            21: "\u00a0middle",
            22: "  ",
            99: "<|endoftext|>",
        }
        text = "".join(pieces[token_id] for token_id in token_ids)
        if skip_special_tokens:
            text = text.replace("<|endoftext|>", "")
        return text


def test_qwen3_asr_audio_token_length_formula_is_shared() -> None:
    lengths = torch.tensor([0, 1, 99, 100, 101, 3000], dtype=torch.long)
    expected = torch.tensor([0, 1, 13, 13, 14, 390], dtype=torch.long)

    processor = object.__new__(Qwen3ASRProcessor)

    assert torch.equal(qwen3_asr_audio_token_lengths(lengths), expected)
    assert torch.equal(processor._get_feat_extract_output_lengths(lengths), expected)
    assert qwen3_asr_num_audio_tokens(3000) == 390


def test_qwen3_asr_request_builder_records_inclusive_audio_offsets(monkeypatch) -> None:
    num_mel_frames = 101
    num_audio_tokens = qwen3_asr_num_audio_tokens(num_mel_frames)
    feature_extractor = lambda *args, **kwargs: SimpleNamespace(
        input_features=torch.zeros((1, 128, 3000)),
        attention_mask=torch.ones((1, num_mel_frames), dtype=torch.long),
    )
    monkeypatch.setattr(
        request_builders,
        "load_audio",
        lambda source: np.zeros(1600, dtype=np.float32),
    )
    request_builder, _ = make_qwen3_asr_scheduler_adapters(
        tokenizer=_FakeTokenizer(),
        max_new_tokens=32,
        feature_extractor=feature_extractor,
    )
    payload = StagePayload(
        request_id="req-asr",
        request=OmniRequest(inputs={"audio_bytes": b"wav"}),
        data={},
    )

    data = request_builder(payload)

    audio_item = data.req.multimodal_inputs.mm_items[0]
    start, end = audio_item.offsets[0]
    assert audio_item.feature_attention_mask.shape == (1, num_mel_frames)
    assert end - start + 1 == num_audio_tokens
    assert data.prompt_token_ids[start : end + 1] == (
        [audio_item.pad_value] * num_audio_tokens
    )


def test_qwen3_asr_result_adapter_decodes_without_text_round_trip() -> None:
    tokenizer = _FakeTokenizer()
    _, result_adapter = make_qwen3_asr_scheduler_adapters(
        tokenizer=tokenizer,
        max_new_tokens=32,
        feature_extractor=object(),
    )
    payload = StagePayload(
        request_id="req-asr",
        request=OmniRequest(inputs={}),
        data={},
    )
    data = Qwen3ASRRequestData(
        output_ids=[10, 100, 101, 20, 21, 22, 99],
        stage_payload=payload,
        language="en",
        audio_duration_s=1.25,
    )

    result = result_adapter(data)

    assert result.data["text"] == " leading\u00a0middle  "
    assert tokenizer.encode_calls == ["<asr_text>"]
    assert tokenizer.decode_calls[-1] == {
        "token_ids": [20, 21, 22, 99],
        "skip_special_tokens": True,
        "clean_up_tokenization_spaces": False,
    }
