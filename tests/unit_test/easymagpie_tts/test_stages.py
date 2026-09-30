# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang_omni.models.easymagpie_tts import codec as codec_module
from sglang_omni.models.easymagpie_tts import stages
from sglang_omni.models.easymagpie_tts.config import EasyMagpieTTSPipelineConfig
from sglang_omni.models.easymagpie_tts.payload_types import EasyMagpieTTSState
from sglang_omni.proto import OmniRequest, StagePayload


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [ord(char) % 60 for char in text]


@pytest.fixture
def checkpoint(tmp_path, monkeypatch, tiny_raw_config):
    (tmp_path / "config.json").write_text(json.dumps(tiny_raw_config))
    voices = tmp_path / stages.SPEAKER_SUBDIR
    voices.mkdir()
    torch.save({"speaker_encoding": torch.ones(3, 8)}, voices / "eng.pt")
    torch.save(torch.zeros(2, 8), voices / "alt.pt")
    monkeypatch.setattr(
        stages,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda *args, **kwargs: FakeTokenizer()),
    )
    return tmp_path


def make_payload(inputs, tts_params=None, data=None) -> StagePayload:
    return StagePayload(
        request_id="req-0",
        request=OmniRequest(
            inputs=inputs, params={}, metadata={"tts_params": tts_params or {}}
        ),
        data=data or {},
    )


def test_pipeline_is_a_three_stage_offline_topology() -> None:
    config = EasyMagpieTTSPipelineConfig(model_path="unused")
    assert [stage.name for stage in config.stages] == [
        "preprocessing",
        "tts_engine",
        "vocoder",
    ]
    assert config.stages[1].factory.dtype == "float16"
    assert config.stages[-1].terminal is True


def test_preprocessing_tokenizes_text_context_and_attaches_the_voice(
    checkpoint,
) -> None:
    scheduler = stages.create_preprocessing_executor(str(checkpoint))
    result = scheduler.fn(make_payload("ab", tts_params={"voice": "alt"}))
    state = EasyMagpieTTSState.from_dict(result.data)

    assert state.text_token_ids == [ord("a") % 60, ord("b") % 60, 63]
    assert state.context_token_ids == [ord(char) % 60 for char in "[EN]"]
    assert (state.phoneme_delay, state.speech_delay, state.text_prefill_num) == (
        3,
        5,
        4,
    )
    assert state.speaker_embedding.shape == (2, 8)
    assert state.speaker_embedding.dtype == torch.float16


def test_preprocessing_rejects_unknown_voices(checkpoint) -> None:
    scheduler = stages.create_preprocessing_executor(str(checkpoint))
    with pytest.raises(ValueError, match="available voices: \\['alt', 'eng'\\]"):
        scheduler.fn(make_payload("hi", tts_params={"voice": "nobody"}))


def test_preprocessing_rejects_speech_before_phoneme_delay(
    checkpoint, tiny_raw_config
) -> None:
    config = dict(tiny_raw_config, streaming_speech_delay=3)
    (checkpoint / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="streaming_speech_delay"):
        stages.create_preprocessing_executor(str(checkpoint))


def test_vocoder_batches_requests_and_reports_usage(tmp_path, monkeypatch) -> None:
    decoded = []

    class FakeCodec:
        config = SimpleNamespace(output_sample_rate=22050)

        def decode_batch(self, codes):
            decoded.append([item.shape[0] for item in codes])
            return [torch.full((item.shape[0] * 2,), 0.5) for item in codes]

    monkeypatch.setattr(codec_module, "load_codec", lambda *args: FakeCodec())
    scheduler = stages.create_vocoder_executor(str(tmp_path), device="cpu")
    payloads = [
        make_payload(
            "x",
            data=EasyMagpieTTSState(
                audio_codes=torch.zeros(frames, 4, dtype=torch.long),
                prompt_tokens=9,
                completion_tokens=frames,
            ).to_dict(),
        )
        for frames in (3, 1)
    ]

    results = scheduler.batch_fn(payloads)

    assert decoded == [[3, 1]]
    assert results[0].data["sample_rate"] == 22050
    assert results[0].data["usage"]["completion_tokens"] == 3
    audio = results[1].data
    assert audio["modality"] == "audio"
    assert audio["audio_waveform_shape"] == [2]
    waveform = np.frombuffer(audio["audio_waveform"], dtype=np.float32)
    assert waveform.tolist() == [0.5, 0.5]

    empty = make_payload("x", data=EasyMagpieTTSState().to_dict())
    with pytest.raises(ValueError, match="no audio frames"):
        scheduler.fn(empty)
