# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import sglang_omni.models.qwen3_asr.request_builders as request_builders
from sglang_omni.models.qwen3_asr.audio_lengths import (
    qwen3_asr_audio_token_lengths,
    qwen3_asr_num_audio_tokens,
)
from sglang_omni.models.qwen3_asr.configuration_qwen3_asr import Qwen3ASRProcessor
from sglang_omni.models.qwen3_asr.request_builders import (
    Qwen3ASRRequestData,
    make_qwen3_asr_scheduler_adapters,
)
from sglang_omni.proto import OmniRequest, StagePayload


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


def _wav_bytes(samples: np.ndarray, sample_rate: int, subtype: str) -> bytes:
    sf = pytest.importorskip("soundfile")
    buffer = io.BytesIO()
    sf.write(buffer, samples, sample_rate, format="WAV", subtype=subtype)
    return buffer.getvalue()


def test_qwen3_asr_audio_token_length_formula_is_shared() -> None:
    lengths = torch.tensor([0, 1, 99, 100, 101, 3000], dtype=torch.long)
    expected = torch.tensor([0, 1, 13, 13, 14, 390], dtype=torch.long)

    processor = object.__new__(Qwen3ASRProcessor)

    assert torch.equal(qwen3_asr_audio_token_lengths(lengths), expected)
    assert torch.equal(processor._get_feat_extract_output_lengths(lengths), expected)
    assert qwen3_asr_num_audio_tokens(3000) == 390


@pytest.mark.parametrize("subtype", ["PCM_U8", "PCM_16", "PCM_24", "PCM_32"])
@pytest.mark.parametrize("channels", [1, 2])
def test_qwen3_asr_read_pcm_wav_bytes_matches_soundfile_for_integer_pcm(
    subtype,
    channels,
) -> None:
    sf = pytest.importorskip("soundfile")

    sample_rate = 16000
    mono = np.array(
        [-0.75, -0.25, -0.01, 0.0, 0.01, 0.25, 0.75],
        dtype=np.float32,
    )
    if channels == 1:
        samples = mono
    else:
        samples = np.stack([mono, -0.5 * mono], axis=1)
    data = _wav_bytes(samples, sample_rate, subtype)

    audio, decoded_sample_rate = request_builders._read_pcm_wav_bytes(data)
    expected, expected_sample_rate = sf.read(
        io.BytesIO(data),
        dtype="float32",
        always_2d=False,
    )

    assert decoded_sample_rate == expected_sample_rate == sample_rate
    assert audio.dtype == np.float32
    assert audio.flags["C_CONTIGUOUS"]
    assert audio.shape == expected.shape
    np.testing.assert_allclose(audio, expected, atol=1.0e-7, rtol=0.0)


def test_qwen3_asr_load_audio_uses_pcm_wav_bytes_fast_path(monkeypatch) -> None:
    sf = pytest.importorskip("soundfile")

    sample_rate = 16000
    mono = np.linspace(-0.25, 0.25, sample_rate // 4, dtype=np.float32)
    data = _wav_bytes(mono, sample_rate, "PCM_16")
    expected, _ = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)

    def fail_if_soundfile_is_called(source):
        raise AssertionError("soundfile should not run for PCM WAV bytes")

    monkeypatch.setattr(
        request_builders,
        "_read_audio_with_soundfile",
        fail_if_soundfile_is_called,
    )

    audio = request_builders.load_audio(data)

    assert audio.dtype == np.float32
    assert audio.flags["C_CONTIGUOUS"]
    assert audio.shape == expected.shape
    np.testing.assert_allclose(audio, expected, atol=1.0e-7, rtol=0.0)


def test_qwen3_asr_load_audio_falls_back_to_soundfile_for_float_wav_bytes(
    monkeypatch,
) -> None:
    sample_rate = 16000
    mono = np.linspace(-0.25, 0.25, sample_rate // 4, dtype=np.float32)
    data = _wav_bytes(mono, sample_rate, "FLOAT")
    original_read = request_builders._read_audio_with_soundfile
    calls: list[str] = []

    def record_soundfile_read(source):
        calls.append(type(source).__name__)
        return original_read(source)

    monkeypatch.setattr(
        request_builders,
        "_read_audio_with_soundfile",
        record_soundfile_read,
    )

    audio = request_builders.load_audio(data)

    assert calls == ["BytesIO"]
    assert audio.dtype == np.float32
    assert audio.flags["C_CONTIGUOUS"]
    np.testing.assert_allclose(audio, mono, atol=1.0e-7, rtol=0.0)


def test_qwen3_asr_load_audio_accepts_pathlike_wav(tmp_path) -> None:
    sf = pytest.importorskip("soundfile")

    sample_rate = 24000
    left = np.linspace(-0.5, 0.5, sample_rate, dtype=np.float32)
    right = np.linspace(0.25, -0.25, sample_rate, dtype=np.float32)
    stereo = np.stack([left, right], axis=1)
    wav_path = tmp_path / "stereo.wav"
    sf.write(wav_path, stereo, sample_rate)

    audio = request_builders.load_audio(wav_path)

    assert audio.dtype == np.float32
    assert audio.flags["C_CONTIGUOUS"]
    assert audio.ndim == 1
    assert len(audio) == 16000


def test_qwen3_asr_load_audio_accepts_wav_bytes() -> None:
    sf = pytest.importorskip("soundfile")

    sample_rate = 16000
    mono = np.linspace(-0.25, 0.25, sample_rate // 2, dtype=np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, mono, sample_rate, format="WAV")

    audio = request_builders.load_audio(buffer.getvalue())

    assert audio.dtype == np.float32
    assert audio.flags["C_CONTIGUOUS"]
    assert audio.shape == mono.shape
    np.testing.assert_allclose(audio, mono, atol=1.0e-4)


@pytest.mark.parametrize(
    ("sample_rate", "subtype", "channels"),
    [
        (16000, "PCM_16", 1),
        (24000, "PCM_16", 2),
        (44100, "FLOAT", 1),
        (48000, "FLOAT", 2),
    ],
)
def test_qwen3_asr_load_audio_matches_torchaudio_path_for_wav_variants(
    tmp_path,
    sample_rate,
    subtype,
    channels,
) -> None:
    sf = pytest.importorskip("soundfile")
    torchaudio = pytest.importorskip("torchaudio")

    duration_s = 0.25
    frames = int(sample_rate * duration_s)
    t = np.arange(frames, dtype=np.float32) / float(sample_rate)
    mono = (0.5 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
    if channels == 1:
        samples = mono
    else:
        samples = np.stack([mono, -0.25 * mono], axis=1)

    wav_path = tmp_path / f"{sample_rate}_{subtype}_{channels}ch.wav"
    sf.write(wav_path, samples, sample_rate, subtype=subtype)

    fast = request_builders.load_audio(wav_path)
    expected_tensor, expected_sample_rate = request_builders._load_audio_with_torchaudio(
        wav_path
    )
    if expected_sample_rate != 16000:
        expected_tensor = torchaudio.functional.resample(
            expected_tensor,
            expected_sample_rate,
            16000,
        )
    expected = np.ascontiguousarray(expected_tensor.cpu().numpy(), dtype=np.float32)

    assert fast.dtype == expected.dtype == np.float32
    assert fast.flags["C_CONTIGUOUS"]
    assert fast.shape == expected.shape
    assert len(fast) == int(16000 * duration_s)
    np.testing.assert_allclose(fast, expected, atol=2.0e-4, rtol=1.0e-4)


def test_qwen3_asr_load_audio_falls_back_when_soundfile_decode_fails(
    monkeypatch,
) -> None:
    sf = pytest.importorskip("soundfile")
    torchaudio = pytest.importorskip("torchaudio")

    calls: list[str] = []

    def fail_soundfile_read(*args, **kwargs):
        calls.append("soundfile")
        raise RuntimeError("decode failed")

    def fake_torchaudio_load(source):
        calls.append(type(source).__name__)
        return torch.tensor([[0.0, 0.25, -0.25]], dtype=torch.float32), 16000

    monkeypatch.setattr(sf, "read", fail_soundfile_read)
    monkeypatch.setattr(torchaudio, "load", fake_torchaudio_load)

    audio = request_builders.load_audio(b"not-a-soundfile-wav")

    assert calls == ["soundfile", "BytesIO"]
    assert audio.dtype == np.float32
    np.testing.assert_allclose(audio, np.array([0.0, 0.25, -0.25], dtype=np.float32))


def test_qwen3_asr_load_audio_does_not_fallback_on_processing_error(
    monkeypatch,
) -> None:
    sf = pytest.importorskip("soundfile")
    torchaudio = pytest.importorskip("torchaudio")

    class DecodedAudioWithProcessingError:
        ndim = 2

        def mean(self, *, axis):
            assert axis == 1
            raise ValueError("processing bug after decode")

    def fake_soundfile_read(*args, **kwargs):
        return DecodedAudioWithProcessingError(), 16000

    def fake_torchaudio_load(source):
        return torch.zeros((1, 3), dtype=torch.float32), 16000

    monkeypatch.setattr(sf, "read", fake_soundfile_read)
    monkeypatch.setattr(torchaudio, "load", fake_torchaudio_load)

    with pytest.raises(ValueError, match="processing bug after decode"):
        request_builders.load_audio(b"decoded-but-processing-fails")


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
