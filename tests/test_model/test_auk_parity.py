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
        checkpoint, device="cuda", gpu_id=0, text_encoder_path=qwen
    )
    engine = create_auk_engine_executor(
        checkpoint,
        device="cuda",
        gpu_id=0,
        enable_dit_fused_qk_norm_rope=False,
    )
    decode = create_decode_executor(checkpoint, device="cuda", gpu_id=0)

    def generate(payload):
        return decode._fn(engine._fn(conditioning._fn(payload)))

    return upstream, generate, checkpoint, Path(source)


@pytest.mark.parametrize("reference", [False, True])
def test_speech_matches_upstream(models, monkeypatch, reference):
    from sglang_omni.client.client import Client
    from sglang_omni.models.auk import stages
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
        request_id="parity", request=Client.build_omni_request(generated), data={}
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
    original_denormalize = BigVGANFlowVAE.denormalize

    def encode(self, *args, **kwargs):
        result = original_encode(self, *args, **kwargs)
        actual["reference"] = result[0].detach().float().cpu()
        return result

    def denormalize(self, latent):
        actual["latent"] = latent.detach().float().cpu()
        return original_denormalize(self, latent)

    monkeypatch.setattr(BigVGANFlowVAE, "encoding_and_normalization", encode)
    monkeypatch.setattr(BigVGANFlowVAE, "denormalize", denormalize)
    _capture(monkeypatch, stages, "fuse_hidden_states", actual, "conditioning")
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


@pytest.fixture(scope="module")
def checkpoint_flow():
    """The real DiT alone, in the serving recipe (bf16 weights, fused Q/K)."""
    checkpoint = os.environ.get("AUK_PARITY_CHECKPOINT")
    if not checkpoint:
        pytest.skip("Set AUK_PARITY_CHECKPOINT for real checkpoint packed parity")
    if not torch.cuda.is_available():
        pytest.skip("AuK packed parity requires CUDA")
    from sglang_omni.models.auk import packed
    from sglang_omni.models.auk.fused_qk_norm_rope import fused_norm_rope
    from sglang_omni.models.auk.stages import load_flow
    from sglang_omni.utils.checkpoint import resolve_checkpoint

    try:
        packed.resolve_flash_version(torch.device("cuda", 0))
    except (ImportError, ValueError) as exc:
        pytest.skip(str(exc))
    flow = load_flow(resolve_checkpoint(checkpoint), "cuda", torch.bfloat16)
    for block in (
        *flow.transformer.transformer_blocks,
        *flow.transformer.single_transformer_blocks,
    ):
        block.attn.qk_fusion = fused_norm_rope
    return flow


def _drift(output, reference):
    """Relative L2 error and cosine between two latents of one request."""
    relative = ((output - reference).norm() / reference.norm()).item()
    cosine = torch.nn.functional.cosine_similarity(
        output.flatten(), reference.flatten(), dim=0
    ).item()
    return relative, cosine


@pytest.mark.parametrize("cfg_strength", [0.0, 2.0])
@torch.inference_mode()
def test_packed_batch_matches_single_requests(
    checkpoint_flow, record_property, cfg_strength
):
    """Fixed seeds, real weights, the 32-step serving recipe: each request
    sampled alone (padded path) against the same requests sampled together on
    the packed path.

    The BF16 backbone is batch-shape sensitive before this change: sampling the
    same requests together on the padded path already drifts from sampling
    them alone (measured on RTX 5090 with FA4 forced: padded-vs-single relative
    L2 up to 0.11 at cfg=0 and cfg=2; packed-vs-padded up to 0.13). Packing must stay
    inside that pre-existing envelope, so the padded batch is measured in the
    same run and packing may add at most ``ENVELOPE`` times the largest padded
    drift plus a ``FLOOR`` for plain BF16 reduction noise. Text masks are all
    True, as the conditioning stage emits them; masks with holes are covered
    by the fp64 parity test in tests/unit_test/auk/test_packed.py (a hole in a
    no-reference request is attended in single-request mode but masked in any
    batch, upstream behaviour that would dominate this comparison). This test
    records what the real model does and catches gross mixing between requests
    or CFG branches."""
    from sglang_omni.models.auk.flow_matching import AuKSampleItem

    ENVELOPE = 2.0
    FLOOR = 0.05
    flow = checkpoint_flow
    dim, text_dim = flow.transformer.latent_dim, flow.transformer.txt_proj.in_features
    generator = torch.Generator("cuda").manual_seed(2026)
    items = []
    for seed, (text, frames, ref) in enumerate(
        [(23, 180, 60), (41, 95, 0), (12, 240, 33), (30, 150, 60)], start=1
    ):
        text_mask = torch.ones(text, dtype=torch.bool, device="cuda")
        items.append(
            AuKSampleItem(
                torch.randn(text, text_dim, device="cuda", generator=generator),
                text_mask,
                frames,
                (
                    torch.randn(ref, dim, device="cuda", generator=generator) * 0.5
                    if ref
                    else None
                ),
                seed=seed,
                ref_length=max(0, ref - 5),
            )
        )
    sampling = dict(steps=32, cfg_strength=cfg_strength, sway_sampling_coef=-1.0)
    single = [flow.sample_batch([item], **sampling)[0] for item in items]
    padded = flow.sample_batch(items, **sampling)
    packed = flow.sample_batch(items, enable_packed_dit=True, **sampling)
    padded_drift = [_drift(p, s)[0] for p, s in zip(padded, single)]
    budget = ENVELOPE * max(padded_drift) + FLOOR
    record_property("padded_vs_single_max_relative", max(padded_drift))
    for i, (output, reference) in enumerate(zip(packed, single)):
        assert output.shape == reference.shape
        assert torch.isfinite(output).all()
        relative, cosine = _drift(output, reference)
        relative_padded, cosine_padded = _drift(output, padded[i])
        record_property(f"request_{i}_packed_vs_single_relative", relative)
        record_property(f"request_{i}_packed_vs_single_cosine", cosine)
        record_property(f"request_{i}_packed_vs_padded_relative", relative_padded)
        record_property(f"request_{i}_padded_vs_single_relative", padded_drift[i])
        print(
            f"cfg={cfg_strength} request {i} {tuple(output.shape)}: "
            f"packed-vs-single rel {relative:.4f} cos {cosine:.5f} | "
            f"packed-vs-padded rel {relative_padded:.4f} cos {cosine_padded:.5f} | "
            f"padded-vs-single rel {padded_drift[i]:.4f}"
        )
        assert relative <= budget, (
            f"request {i}: packed drift {relative:.4f} exceeds {ENVELOPE}x the "
            f"padded batch drift ({max(padded_drift):.4f})"
        )
        assert relative_padded <= budget
