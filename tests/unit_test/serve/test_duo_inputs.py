"""Realization model work leaves the websocket loop live and closes after GPU work."""

import os
import wave
from contextlib import nullcontext
from email.parser import BytesParser
from email.policy import default
from io import BytesIO
from unittest.mock import Mock

import httpx
import numpy as np
import pytest
import torch
from sglang.multimodal_gen.configs.pipeline_configs.duo import DuoDecodeMode
from sglang.multimodal_gen.runtime.models.vaes.duo.codec import DuoCodecs

from sglang_omni.models.duo.live import LiveDuoSessionHooks, LiveQwenASR, LiveWhisper
from sglang_omni.proto.request import OmniRequest
from sglang_omni.proto.session import SessionIdentity


@pytest.mark.parametrize("mode", ["duo_1_4", "duo_5_17"])
def test_reference_cache_keeps_seed_semantics_and_bounds_retained_images(
    monkeypatch: pytest.MonkeyPatch, mode: DuoDecodeMode
) -> None:
    monkeypatch.setattr(torch.cuda, "device", Mock(return_value=nullcontext()))
    codecs = Mock(spec=DuoCodecs)
    codecs.device = torch.device("cpu")
    realization = Mock()
    realization.device = torch.device("cpu")
    encoded: list[tuple[int, int, DuoDecodeMode]] = []

    def encode_reference(
        pixels: torch.Tensor, *, seed: int, mode: DuoDecodeMode
    ) -> torch.Tensor:
        pixel = int(pixels[0, 0, 0])
        encoded.append((pixel, seed, mode))
        value = pixel + (seed if mode == "duo_1_4" else 0)
        return torch.full((24, 1, 4, 4), value, dtype=torch.bfloat16)

    codecs.encode_reference.side_effect = encode_reference
    hooks = LiveDuoSessionHooks(
        Mock(), realization, codecs, max_blocks=3, reference_cache_entries=2
    )
    other_mode: DuoDecodeMode = "duo_5_17" if mode == "duo_1_4" else "duo_1_4"
    requests = [
        (16, 1, mode),
        (16, 2, mode),
        (64, 1, mode),
        (128, 1, mode),
        (16, 1, mode),
        (16, 1, other_mode),
    ]
    for index, (pixel, seed, decode_mode) in enumerate(requests):
        identity = SessionIdentity(f"reference-{index}")
        request = OmniRequest(
            {
                "reference_rgb": bytes([pixel]) * (32 * 32 * 3),
                "height": 32,
                "width": 32,
                "description": {
                    "appearance": "a person",
                    "environment": "a room",
                    "initial_pose": "standing",
                },
                "seed": seed,
            }
        )
        hooks.open(identity, request, decode_mode=decode_mode)
        expected = pixel + (seed if decode_mode == "duo_1_4" else 0)
        torch.testing.assert_close(
            hooks.states[identity].reference,
            torch.full((24, 1, 4, 4), expected, dtype=torch.bfloat16),
        )
        with torch.inference_mode():
            hooks.states[identity].reference.fill_(-1)
        hooks.close(identity)
    expected_encodes = (
        requests if mode == "duo_1_4" else [requests[i] for i in [0, 2, 3, 4, 5]]
    )
    assert encoded == expected_encodes
    assert not hooks.states


@pytest.mark.accelerator
@pytest.mark.skipif(
    not torch.cuda.is_available() or "WHISPER_MODEL_PATH" not in os.environ,
    reason="requires CUDA and WHISPER_MODEL_PATH",
)
def test_whisper_reuses_encoder_for_language_detection_and_transcription() -> None:
    recognizer = LiveWhisper(os.environ["WHISPER_MODEL_PATH"], device="cuda:0")
    samples = np.zeros(16000, dtype=np.float32)
    inputs = recognizer.processor(
        samples, sampling_rate=16000, return_tensors="pt", return_attention_mask=True
    )
    with torch.inference_mode():
        tokens = recognizer.model.generate(
            inputs.input_features.to("cuda:0", dtype=torch.float16),
            attention_mask=inputs.attention_mask.to("cuda:0"),
            task="transcribe",
            max_new_tokens=128,
        )
    expected = recognizer.processor.batch_decode(tokens, skip_special_tokens=True)[
        0
    ].strip()
    calls: list[torch.Size] = []

    def count_encoder(
        module: torch.nn.Module, arguments: tuple[torch.Tensor, ...]
    ) -> None:
        calls.append(arguments[0].shape)

    hook = recognizer.model.model.encoder.register_forward_pre_hook(count_encoder)
    try:
        actual = recognizer.transcribe(np.zeros(16000, dtype="<i2").tobytes())
    finally:
        hook.remove()
    assert actual == expected
    assert len(calls) == 1


@pytest.mark.parametrize("status", [200, 503])
def test_intent_asr_preserves_pcm_and_propagates_service_errors(status: int) -> None:
    pcm = np.arange(16000, dtype="<i2").tobytes()

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/audio/transcriptions"
        message = BytesParser(policy=default).parsebytes(
            f"Content-Type: {request.headers['content-type']}\r\n\r\n".encode()
            + request.read()
        )
        audio_part = next(part for part in message.iter_parts() if part.get_filename())
        with wave.open(BytesIO(audio_part.get_payload(decode=True))) as reader:
            assert reader.getframerate() == 16000
            assert reader.getnchannels() == 1
            assert reader.readframes(16000) == pcm
        return httpx.Response(status, json={"text": "  周航，你现在方便吗？  "})

    recognizer = LiveQwenASR("http://asr.local/v1", "intent-asr")
    recognizer.client.close()
    recognizer.client = httpx.Client(
        base_url="http://asr.local/v1/", transport=httpx.MockTransport(handle)
    )
    try:
        if status == 200:
            assert recognizer.transcribe(pcm) == "周航，你现在方便吗？"
        else:
            with pytest.raises(httpx.HTTPStatusError):
                recognizer.transcribe(pcm)
    finally:
        recognizer.close()
