# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys

import numpy as np
import pytest
import torch

from sglang_omni.models.fun_cosyvoice3 import utils
from sglang_omni.models.fun_cosyvoice3.sglang_model import (
    EOS_ID,
    FILL_ID,
    SOS_ID,
    TASK_ID,
    TOTAL_VOCAB_SIZE,
    VOCAB_SIZE,
)
from sglang_omni.models.fun_cosyvoice3.utils import (
    SpeechTokenizerV3,
    build_llm_prompt_embeddings,
)


def _speech_embed(ids: torch.Tensor) -> torch.Tensor:
    return ids.to(dtype=torch.float32).unsqueeze(-1).expand(*ids.shape, 4)


def test_speech_tokenizer_v3_loads_onnx_weights_on_requested_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []

    class _FakeModel:
        def to(self, device: torch.device) -> _FakeModel:
            calls.append(device)
            return self

        def eval(self) -> _FakeModel:
            calls.append("eval")
            return self

    class _FakeBackend:
        @staticmethod
        def load_model(model_path: str) -> _FakeModel:
            calls.append(model_path)
            return _FakeModel()

    monkeypatch.setitem(sys.modules, "s3tokenizer", _FakeBackend())

    tokenizer = SpeechTokenizerV3("speech_tokenizer_v3.onnx", device="cpu")

    assert calls == ["speech_tokenizer_v3.onnx", torch.device("cpu"), "eval"]
    assert tokenizer.device == torch.device("cpu")


def test_speech_tokenizer_v3_batches_ragged_audio_and_slices_codes() -> None:
    class _FakeBackend:
        @staticmethod
        def log_mel_spectrogram(audio: torch.Tensor) -> torch.Tensor:
            return torch.full(
                (128, audio.numel()),
                float(audio[0].item()),
                dtype=torch.float32,
            )

        @staticmethod
        def padding(mels: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
            max_frames = max(mel.shape[-1] for mel in mels)
            padded = torch.zeros(len(mels), 128, max_frames)
            lengths = torch.tensor([mel.shape[-1] for mel in mels])
            for index, mel in enumerate(mels):
                padded[index, :, : mel.shape[-1]] = mel
            return padded, lengths

    class _FakeModel:
        def __init__(self) -> None:
            self.seen_mels: torch.Tensor | None = None
            self.seen_lengths: torch.Tensor | None = None

        def quantize(
            self, mels: torch.Tensor, lengths: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            assert torch.is_inference_mode_enabled()
            self.seen_mels = mels.clone()
            self.seen_lengths = lengths.clone()
            return (
                torch.tensor([[10, 11, 99], [20, 21, 22]]),
                torch.tensor([2, 3]),
            )

    tokenizer = SpeechTokenizerV3.__new__(SpeechTokenizerV3)
    tokenizer._backend = _FakeBackend()
    tokenizer.device = torch.device("cpu")
    tokenizer.model = _FakeModel()

    tokens = tokenizer.extract_speech_tokens(
        [
            np.ones(3, dtype=np.float32),
            np.full(5, 2.0, dtype=np.float32),
        ]
    )

    assert tokenizer.model.seen_mels is not None
    assert tokenizer.model.seen_lengths is not None
    assert tokenizer.model.seen_mels.shape == (2, 128, 5)
    assert tokenizer.model.seen_lengths.tolist() == [3, 5]
    assert [token.tolist() for token in tokens] == [[[10, 11]], [[20, 21, 22]]]
    assert all(token.dtype == torch.int32 for token in tokens)
    assert all(token.device.type == "cpu" for token in tokens)


def test_speech_tokenizer_v3_empty_batch_returns_empty_list() -> None:
    tokenizer = SpeechTokenizerV3.__new__(SpeechTokenizerV3)

    class _Backend:
        def log_mel_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
            raise AssertionError("empty batches must not invoke the backend")

    tokenizer._backend = _Backend()
    assert tokenizer.extract_speech_tokens([]) == []


def test_speech_tokenizer_v3_accepts_audio_at_30_second_boundary() -> None:
    audio = np.ones(30 * 16000, dtype=np.float32)

    normalized = SpeechTokenizerV3._normalize_audio(audio, 16000)

    assert normalized.shape == (30 * 16000,)
    assert normalized.dtype == torch.float32


@pytest.mark.parametrize(
    ("codes", "code_lengths", "message"),
    [
        (
            torch.zeros(1, 2, 1),
            torch.tensor([1]),
            "invalid batch shapes",
        ),
        (
            torch.zeros(1, 2),
            torch.tensor([3]),
            "invalid code length",
        ),
        (
            torch.zeros(1, 2),
            torch.tensor([-1]),
            "invalid code length",
        ),
    ],
)
def test_speech_tokenizer_v3_rejects_invalid_model_outputs(
    codes: torch.Tensor,
    code_lengths: torch.Tensor,
    message: str,
) -> None:
    class _Backend:
        @staticmethod
        def log_mel_spectrogram(audio: torch.Tensor) -> torch.Tensor:
            return torch.zeros(128, 4)

        @staticmethod
        def padding(mels: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
            return torch.zeros(len(mels), 128, 4), torch.tensor([4] * len(mels))

    class _Model:
        def quantize(
            self, mels: torch.Tensor, lengths: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return codes, code_lengths

    tokenizer = SpeechTokenizerV3.__new__(SpeechTokenizerV3)
    tokenizer._backend = _Backend()
    tokenizer.device = torch.device("cpu")
    tokenizer.model = _Model()

    with pytest.raises(RuntimeError, match=message):
        tokenizer.extract_speech_tokens([np.ones(4, dtype=np.float32)])


@pytest.mark.parametrize(
    ("audio", "sample_rate", "message"),
    [
        (np.ones(4, dtype=np.float32), 24000, "16kHz"),
        (np.ones((2, 4), dtype=np.float32), 16000, "shape"),
        (np.array([], dtype=np.float32), 16000, "must not be empty"),
        (np.ones(480001, dtype=np.float32), 16000, "30s"),
    ],
)
def test_speech_tokenizer_v3_rejects_invalid_audio(
    audio: np.ndarray, sample_rate: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        SpeechTokenizerV3._normalize_audio(audio, sample_rate)


def test_cosyvoice3_prompt_embeddings_use_speech_control_tokens_and_reference_tokens() -> (
    None
):
    text_embed = torch.tensor([[[10.0, 11.0, 12.0, 13.0], [20.0, 21.0, 22.0, 23.0]]])
    prompt_tokens = torch.tensor([[30, 31]], dtype=torch.int32)

    result = build_llm_prompt_embeddings(
        text_token=torch.tensor([[1, 2]]),
        text_embed=text_embed,
        prompt_speech_token=prompt_tokens,
        speech_embed=_speech_embed,
        embedding=torch.full((1, 192), 999.0),
        sos_id=SOS_ID,
        task_id=TASK_ID,
        hidden_size=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert result.shape == (1, 6, 4)
    assert torch.equal(result[0, 0], torch.full((4,), float(SOS_ID)))
    assert torch.equal(result[0, 1:3], text_embed[0])
    assert torch.equal(result[0, 3], torch.full((4,), float(TASK_ID)))
    assert torch.equal(result[0, 4], torch.full((4,), 30.0))
    assert torch.equal(result[0, 5], torch.full((4,), 31.0))
    assert not torch.any(result == 999.0)


def test_cosyvoice3_prompt_embeddings_keep_empty_reference_shape_and_dtype() -> None:
    result = build_llm_prompt_embeddings(
        text_token=torch.tensor([[1]]),
        text_embed=torch.ones(1, 1, 3, dtype=torch.float16),
        prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
        speech_embed=lambda ids: torch.ones(*ids.shape, 3, dtype=torch.float32),
        embedding=torch.zeros(0, 192),
        sos_id=SOS_ID,
        task_id=TASK_ID,
        hidden_size=3,
        device=torch.device("cpu"),
        dtype=torch.float16,
    )

    assert result.shape == (1, 3, 3)
    assert result.dtype == torch.float32
    assert result[:, 3:].numel() == 0


def test_cosyvoice3_speech_vocabulary_layout_is_explicit() -> None:
    assert SOS_ID == VOCAB_SIZE
    assert EOS_ID == VOCAB_SIZE + 1
    assert TASK_ID == VOCAB_SIZE + 2
    assert FILL_ID == VOCAB_SIZE + 3
    assert TOTAL_VOCAB_SIZE == VOCAB_SIZE + 200


def test_cosyvoice3_prompt_mel_uses_flow_layout_and_fixed_configuration(
    monkeypatch,
) -> None:
    captured: dict[str, torch.Tensor] = {}

    def fake_mel(waveform: torch.Tensor) -> torch.Tensor:
        captured["waveform"] = waveform
        return torch.arange(1 * 80 * 3, dtype=torch.float32).reshape(1, 80, 3)

    monkeypatch.setattr(utils, "_run_cosyvoice3_mel_spectrogram", fake_mel)

    result = utils.extract_prompt_speech_feat(np.zeros(12, dtype=np.float64))

    assert captured["waveform"].shape == (1, 12)
    assert captured["waveform"].dtype == torch.float32
    assert result.shape == (1, 3, 80)
    assert torch.equal(result[0, 0], torch.arange(0, 80 * 3, 3, dtype=torch.float32))
