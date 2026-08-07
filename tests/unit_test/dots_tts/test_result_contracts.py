# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang_omni.models.dots_tts.codec import _DotsReferenceHook
from sglang_omni.models.dots_tts.payload_types import DotsTTSState
from sglang_omni.models.dots_tts.vocoder import (
    DotsTTSBatchVocoder,
    DotsTTSStreamingVocoder,
)
from sglang_omni.proto import OmniRequest, StagePayload


def test_inline_reference_keys_are_content_sensitive() -> None:
    hook = object.__new__(_DotsReferenceHook)

    first = hook.input_key("data:audio/wav;base64,AAAA")
    repeated = hook.input_key("data:audio/wav;base64,AAAA")
    different = hook.input_key("data:audio/wav;base64,BBBB")

    assert first is not None
    assert repeated == first
    assert different != first


def test_terminal_vocoder_results_do_not_serialize_internal_tts_state() -> None:
    prompt_audio = "data:audio/wav;base64,PRIVATE"
    state = DotsTTSState(
        prompt_audio_path=prompt_audio,
        generated_latents=torch.ones(1, 4, 3),
    )
    payload = StagePayload(
        request_id="rid",
        request=OmniRequest(inputs="hello"),
        data=state.to_dict(),
    )
    codec = SimpleNamespace(sample_rate=48000)

    stream = object.__new__(DotsTTSStreamingVocoder)
    stream.codec = codec
    stream_result = stream.final_result_data("rid", payload, SimpleNamespace())

    batch = DotsTTSBatchVocoder(codec)
    batch_result = batch.store_result(
        payload,
        state,
        torch.ones(1, 16),
        48000,
    )

    assert "state" not in batch_result.data
    assert "state" not in stream_result
    assert prompt_audio not in repr(batch_result.data)
    assert prompt_audio not in repr(stream_result)
