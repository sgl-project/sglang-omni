# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.fishaudio_s2_pro.request_builders import (
    make_tts_scheduler_adapters,
)
from sglang_omni.scheduling.streaming_vocoder import INITIAL_CODEC_CHUNK_FRAMES_PARAM
from tests.unit_test.fixtures.fish_fakes import FakeFishTokenizer, make_s2pro_payload


@pytest.fixture(autouse=True)
def fast_sampling_params(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sglang.srt.sampling.sampling_params.SamplingParams.normalize",
        lambda self, tokenizer: None,
    )
    monkeypatch.setattr(
        "sglang.srt.sampling.sampling_params.SamplingParams.verify",
        lambda self, vocab_size: None,
    )


@pytest.mark.parametrize(
    "override,expected",
    [
        (None, None),
        (0, 0),
        (5, 5),
        (41, 41),
    ],
)
def test_fish_stream_output_forwards_explicit_initial_chunk_override(
    override: int | None,
    expected: int | None,
) -> None:
    params = {"stream": True}
    if override is not None:
        params[INITIAL_CODEC_CHUNK_FRAMES_PARAM] = override
    payload = make_s2pro_payload(params=params)
    request_builder, _, stream_output_builder = make_tts_scheduler_adapters(
        tokenizer=FakeFishTokenizer()
    )

    data = request_builder(payload)
    data.latest_stream_code_chunk = torch.ones((11, 1), dtype=torch.long)
    outputs = stream_output_builder(payload.request_id, data, None)

    assert len(outputs) == 1
    metadata = outputs[0].metadata
    assert metadata is not None
    if expected is None:
        assert INITIAL_CODEC_CHUNK_FRAMES_PARAM not in metadata
    else:
        assert metadata[INITIAL_CODEC_CHUNK_FRAMES_PARAM] == expected

    data.latest_stream_code_chunk = torch.ones((11, 1), dtype=torch.long)
    followup = stream_output_builder(payload.request_id, data, None)
    assert followup[0].metadata == metadata


@pytest.mark.parametrize(
    ("override", "error_type", "message"),
    [
        (-1, ValueError, f"{INITIAL_CODEC_CHUNK_FRAMES_PARAM} must be >= 0"),
        (
            "invalid",
            TypeError,
            f"{INITIAL_CODEC_CHUNK_FRAMES_PARAM} must be an integer",
        ),
    ],
)
def test_fish_request_builder_rejects_invalid_initial_chunk_override(
    override: object,
    error_type: type[Exception],
    message: str,
) -> None:
    payload = make_s2pro_payload(
        params={"stream": True, INITIAL_CODEC_CHUNK_FRAMES_PARAM: override}
    )
    request_builder, _, _ = make_tts_scheduler_adapters(tokenizer=FakeFishTokenizer())

    with pytest.raises(error_type, match=message):
        request_builder(payload)


def test_fish_stream_output_direct_call_resolves_metadata_fallback() -> None:
    _, _, stream_output_builder = make_tts_scheduler_adapters(
        tokenizer=FakeFishTokenizer()
    )
    codes = torch.ones((11, 1), dtype=torch.long)
    data = SimpleNamespace(
        stage_payload=make_s2pro_payload(
            params={"stream": True, INITIAL_CODEC_CHUNK_FRAMES_PARAM: 3}
        ),
        latest_stream_code_chunk=codes,
    )

    outputs = stream_output_builder("direct", data, None)

    assert len(outputs) == 1
    assert outputs[0].metadata == {
        "modality": "audio_codes",
        INITIAL_CODEC_CHUNK_FRAMES_PARAM: 3,
    }
    assert outputs[0].data is codes
