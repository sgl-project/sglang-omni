# SPDX-License-Identifier: Apache-2.0
"""Opt-in checkpoint parity against Tencent-Hunyuan/AuK (docs/cookbook/auk.md)."""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

pytestmark = pytest.mark.accelerator


def _capture(monkeypatch, obj, method, output, key, *, argument=False, first=False):
    original = getattr(obj, method)

    def wrapped(*args, **kwargs):
        result = original(*args, **kwargs)
        value = args[0] if argument else result
        if first:
            value = value[0]
        output[key] = value.detach().float().cpu().clone()
        return result

    monkeypatch.setattr(obj, method, wrapped)


@pytest.fixture(scope="module")
def models():
    source = os.environ.get("AUK_UPSTREAM_SOURCE")
    checkpoint = os.environ.get("AUK_PARITY_CHECKPOINT")
    if not source or not checkpoint:
        pytest.skip(
            "Set AUK_UPSTREAM_SOURCE and AUK_PARITY_CHECKPOINT for real checkpoint parity"
        )
    if not torch.cuda.is_available():
        pytest.skip("AuK checkpoint parity requires CUDA")
    from sglang_omni.models.auk.stages import (
        create_auk_engine_executor,
        create_conditioning_executor,
        create_decode_executor,
    )
    from sglang_omni.models.auk.weight_loader import resolve_weight_file
    from sglang_omni.utils.checkpoint import resolve_checkpoint

    sys.path.insert(0, str(Path(source) / "src"))
    with patch.dict(sys.modules, {"flash_attn": None}):
        from auk.infer.infer_auk import AukInfer

    checkpoint = resolve_checkpoint(checkpoint)
    qwen = os.environ.get("AUK_QWEN_CHECKPOINT", "Qwen/Qwen2.5-Omni-3B")
    upstream = AukInfer(
        str(Path(checkpoint) / "config.yaml"),
        str(resolve_weight_file(checkpoint)),
        device="cuda:0",
        qwen_path=qwen,
    )
    conditioning = create_conditioning_executor(
        checkpoint, device="cuda:0", text_encoder_path=qwen
    )
    engine = create_auk_engine_executor(checkpoint, device="cuda:0")
    decode = create_decode_executor(checkpoint, device="cuda:0")

    def generate(payload):
        return decode._fn(engine._fn(conditioning._fn(payload)))

    return upstream, generate, checkpoint, Path(source)


@pytest.mark.parametrize("reference", [False, True])
def test_speech_matches_upstream(models, monkeypatch, reference):
    from sglang_omni.client.client import Client
    from sglang_omni.models.auk.flow_matching import AuKFlowMatching
    from sglang_omni.models.auk.hf_config import make_runtime_config
    from sglang_omni.models.auk.request_builders import build_auk_state
    from sglang_omni.models.auk.vae import BigVGANFlowVAE
    from sglang_omni.proto import StagePayload
    from sglang_omni.serve.protocol import CreateSpeechRequest
    from sglang_omni.serve.speech_service import SpeechRequestValidator

    upstream, engine, checkpoint, source = models
    ref_path = (
        str(source / "assets/demo-input-audio/zero-shot-tts/ref.wav")
        if reference
        else None
    )
    request = CreateSpeechRequest(
        input="Welcome home.",
        instructions=None if reference else "A warm, relaxed female voice",
        ref_audio=ref_path,
        seed=1234,
        stage_params={"auk_engine": {"gen_seconds": 2.01}},
    )
    generated = SpeechRequestValidator(
        default_model="tencent/AuK"
    ).build_generate_request(request)
    payload = StagePayload(
        request_id="parity", request=Client._build_omni_request(generated), data={}
    )
    state = build_auk_state(payload, make_runtime_config(checkpoint))
    payload.data = state.to_dict()
    instruction = (
        'Say the following with the same voice: "Welcome home."'
        if reference
        else 'Generate speech based on the following description: "A warm, relaxed female voice". '
        'The content to speak is: "Welcome home.".'
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": instruction}]}]
    if reference:
        messages[0]["content"].append({"type": "audio", "audio": ref_path})

    expected = {}
    actual = {}
    _capture(
        monkeypatch,
        upstream.vae_model,
        "encoding_and_normalization",
        expected,
        "reference",
        first=True,
    )
    _capture(
        monkeypatch, upstream.model, "encode_text", expected, "conditioning", first=True
    )
    _capture(
        monkeypatch,
        upstream.vae_model,
        "denormalize",
        expected,
        "latent",
        argument=True,
    )

    original_encode = BigVGANFlowVAE.encoding_and_normalization
    original_fuse = AuKFlowMatching.fuse
    original_denormalize = BigVGANFlowVAE.denormalize

    def encode(self, *args, **kwargs):
        result = original_encode(self, *args, **kwargs)
        actual["reference"] = result[0].detach().float().cpu()
        return result

    def fuse(self, *args, **kwargs):
        result = original_fuse(self, *args, **kwargs)
        actual["conditioning"] = result.detach().float().cpu()
        return result

    def denormalize(self, latent):
        actual["latent"] = latent.detach().float().cpu()
        return original_denormalize(self, latent)

    monkeypatch.setattr(BigVGANFlowVAE, "encoding_and_normalization", encode)
    monkeypatch.setattr(AuKFlowMatching, "fuse", fuse)
    monkeypatch.setattr(BigVGANFlowVAE, "denormalize", denormalize)
    torch.manual_seed(request.seed)
    expected["waveform"], sample_rate = upstream.generate(
        messages,
        audio=ref_path,
        gen_seconds=2.01,
        seed=1234,
        nfe=32,
        cfg_strength=2.0,
    )
    torch.manual_seed(request.seed)
    result = engine(payload)
    actual["waveform"] = torch.frombuffer(
        bytearray(result.data["audio_waveform"]), dtype=torch.float32
    ).reshape(1, -1)
    assert result.data["sample_rate"] == sample_rate == 24000
    for key in expected:
        print(
            f"{key}: shape={tuple(expected[key].shape)}, max_abs_error={(actual[key] - expected[key]).abs().max().item():.8g}"
        )
        torch.testing.assert_close(
            actual[key],
            expected[key],
            rtol=1e-4,
            atol=1e-5,
            msg=lambda message: f"{key}: {message}",
        )

    if reference:
        waveform = result.data["audio_waveform"]
        torch.rand(17, device="cuda:0")
        rng = torch.cuda.get_rng_state()
        payload.data = state.to_dict()
        repeated = engine(payload)
        assert repeated.data["audio_waveform"] == waveform
        assert torch.equal(torch.cuda.get_rng_state(), rng)
