# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest
import tiktoken
import torch
from safetensors.torch import save_file

from sglang_omni.config import EngineStageConfig
from sglang_omni.config.runtime import resolve_factory_signature_args
from sglang_omni.models.yue2.acoustic import synthesize_latents
from sglang_omni.models.yue2.config import Yue2PipelineConfig
from sglang_omni.models.yue2.generation import masked_scores
from sglang_omni.models.yue2.layers import (
    MLP,
    Attention,
    RMSNorm,
    RotaryEmbedding,
    TokenCache,
    Yue2ModelConfig,
)
from sglang_omni.models.yue2.model import Yue2Model
from sglang_omni.models.yue2.payload_types import Yue2State
from sglang_omni.models.yue2.request import (
    ABC_END,
    CODEC_OFFSET,
    CODEC_SIZE,
    MUSIC_END,
    TEXT_END,
    SamplingConfig,
    Yue2Tokenizer,
    build_request,
)
from sglang_omni.models.yue2.runtime import Yue2Runtime, load_weights
from sglang_omni.models.yue2.stages import (
    Yue2AcousticScheduler,
    create_acoustic_executor,
)
from sglang_omni.models.yue2.vae import DecoderConfig, VaeConfig, Yue2Vae
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.pipeline_state import store_state
from sglang_omni.serve.openai_errors import generation_error_status_code


@pytest.fixture
def model() -> Yue2Model:
    with torch.random.fork_rng():
        torch.manual_seed(7)
        return Yue2Model(
            Yue2ModelConfig(
                hidden_size=8,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=4,
                intermediate_size=16,
                max_position_embeddings=128,
                max_latent_frames=128,
            )
        ).eval()


@pytest.fixture
def tokenizer() -> Yue2Tokenizer:
    tokenizer = object.__new__(Yue2Tokenizer)
    tokenizer.encoding = tiktoken.Encoding(
        "test-yue2",
        pat_str=r"[\s\S]",
        special_tokens={},
        mergeable_ranks={bytes([index]): index for index in range(256)},
    )
    return tokenizer


def payload(seed: int = 7, token_budget: int = 8, **options: object) -> StagePayload:
    return StagePayload(
        request_id=f"song-{seed}",
        data={},
        request=OmniRequest(
            inputs="Sing",
            metadata={"tts_params": {"instructions": "pop", "seed": seed}},
            params={
                "max_new_tokens": token_budget,
                "stage_params": {"synthesis": options},
            },
        ),
    )


def test_pipeline_runs_sglang_ar_then_acoustic_stage() -> None:
    configuration = Yue2PipelineConfig(model_path="m-a-p/YuE2-3B")
    ar_stage, acoustic_stage = configuration.stages
    assert isinstance(ar_stage, EngineStageConfig)
    assert ar_stage.next == "yue2_acoustic" and not ar_stage.terminal
    assert acoustic_stage.terminal and acoustic_stage.process != ar_stage.process
    arguments = resolve_factory_signature_args(
        create_acoustic_executor,
        acoustic_stage.factory.model_dump(exclude_none=True),
        defaults={"model_path": configuration.model_path, "gpu_id": 0},
    )
    assert arguments["vae_path"] == "m-a-p/YuE2-Vae"
    assert not arguments["fused_kernels"] and arguments["gpu_id"] == 0


@pytest.mark.parametrize("seed", [-1, 2**63, True, 1.5])
def test_invalid_seed_rejected(seed: object) -> None:
    message = payload()
    message.request.metadata["tts_params"]["seed"] = seed
    with pytest.raises(ValueError, match="seed"):
        build_request(message)


@pytest.mark.parametrize(
    "options",
    [
        {"cot": "unknown"},
        {"cfg_scale": float("nan")},
        {"cfg_scale": -1.0},
        {"abc": "X:1", "cot": "off"},
        {"abc": " "},
        {"ode_steps": 0},
        {"abc_max_tokens": True},
        {"unknown_knob": 3},
    ],
)
def test_invalid_options_rejected(options: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        build_request(payload(**options))


def test_style_is_required_and_kept_verbatim() -> None:
    message = payload()
    message.request.metadata["tts_params"]["instructions"] = ""
    assert build_request(message).style == ""
    del message.request.metadata["tts_params"]["instructions"]
    with pytest.raises(ValueError, match="instructions"):
        build_request(message)


@pytest.mark.parametrize("token_budget", [0, -1, True, 1.5])
def test_invalid_token_budget_rejected(token_budget: object) -> None:
    message = payload()
    message.request.params["max_new_tokens"] = token_budget
    with pytest.raises(ValueError):
        build_request(message)


@pytest.mark.parametrize("cot", ["full", "melody"])
def test_cfg_keeps_exact_score_tokens(cot: str, tokenizer: Yue2Tokenizer) -> None:
    request = build_request(payload(cot=cot, cfg_scale=0.0))
    assert request.guidance == 0.0
    score = tokenizer.encode("X:1\nK:C\nCDEF")
    assert tokenizer.negative_prefix(request, score)[-len(score) - 2 :] == [
        *score,
        ABC_END,
        151851,
    ]
    assert tokenizer.prefix(request, score)[-len(score) - 2 :] == [
        *score,
        ABC_END,
        151851,
    ]


@torch.inference_mode()
def test_cached_decode_matches_full_prefix(model: Yue2Model) -> None:
    tokens = torch.tensor([[1, 2, 3, 4]])
    expected = TokenCache(capacity=4)
    model(tokens, expected)
    actual = TokenCache(capacity=4)
    model(tokens[:, :3], actual)
    model(tokens[:, 3:], actual)
    assert actual.length == expected.length == 4
    for actual_tensor, expected_tensor in zip(
        actual.keys + actual.values, expected.keys + expected.values
    ):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("phase,end_token", [("abc", ABC_END), ("semantic", MUSIC_END)])
def test_sampling_masks_phase_vocabulary_and_gates_end_per_row(
    phase: str, end_token: int
) -> None:
    sampling = SamplingConfig(temperature=0.0, min_tokens=1, max_tokens=2)
    logits = torch.zeros(2, 184704)
    logits[:, end_token] = torch.inf
    scores = masked_scores(
        logits,
        sampling,
        torch.full((2, sampling.penalty_window), -1),
        torch.tensor([False, True]),
        phase,
        False,
    )
    assert torch.isneginf(scores[0, end_token]) and scores[1].argmax() == end_token
    if phase == "semantic":
        assert torch.isneginf(scores[0, :CODEC_OFFSET]).all()
    else:
        assert torch.isneginf(scores[:, CODEC_OFFSET:]).all()


def full_vocabulary_scores(
    logits: torch.Tensor,
    sampling: SamplingConfig,
    recent_tokens: torch.Tensor,
    allow_end: torch.Tensor,
    phase: str,
    legacy_off: bool,
) -> torch.Tensor:
    """The validated formulation: frequencies over the whole vocabulary."""
    scores = logits.to(torch.bfloat16 if legacy_off else torch.float32)
    end_token = ABC_END if phase == "abc" else MUSIC_END
    allowed = torch.full_like(scores, -torch.inf)
    if phase == "abc":
        allowed[:, :TEXT_END] = 0
    else:
        allowed[:, CODEC_OFFSET : CODEC_OFFSET + CODEC_SIZE] = 0
    allowed[:, end_token] = 0
    scores = scores + allowed
    scores[:, end_token] = torch.where(allow_end, scores[:, end_token], -torch.inf)
    frequencies = torch.zeros_like(scores)
    frequencies.scatter_add_(
        -1,
        recent_tokens.clamp_min(0),
        (recent_tokens >= 0).to(frequencies.dtype),
    )
    penalty = sampling.repetition_penalty**frequencies
    return torch.where(scores < 0, scores * penalty, scores / penalty)


@pytest.mark.parametrize("legacy_off", [False, True])
@pytest.mark.parametrize("phase,first", [("abc", 10), ("semantic", CODEC_OFFSET)])
def test_window_penalty_matches_full_vocabulary_penalty(
    phase: str, first: int, legacy_off: bool
) -> None:
    sampling = SamplingConfig(repetition_penalty=1.2, penalty_window=6)
    with torch.random.fork_rng():
        torch.manual_seed(0)
        logits = torch.randn(3, 184704) * 4
    recent = torch.tensor(
        [
            [first, first + 1, first, -1, -1, -1],
            [first + 2] * 6,
            [-1] * 6,
        ]
    )
    allow_end = torch.tensor([True, False, True])
    torch.testing.assert_close(
        masked_scores(logits, sampling, recent, allow_end, phase, legacy_off),
        full_vocabulary_scores(logits, sampling, recent, allow_end, phase, legacy_off),
        rtol=0,
        atol=0,
    )


def test_acoustic_context_overflow_fails_before_prefill(
    model: Yue2Model, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AR-fit prompts can still leave no room for an acoustic frame."""
    forward = Mock(side_effect=AssertionError("must not execute"))
    monkeypatch.setattr(model.model.embed_tokens, "forward", forward)
    with pytest.raises(ValueError, match="context") as invalid_request:
        synthesize_latents(model, list(range(124)), [0], 7, 2, lambda: False)
    forward.assert_not_called()
    assert generation_error_status_code(RuntimeError(str(invalid_request.value))) == 400


def test_acoustic_stage_renders_state_and_reports_truncation() -> None:
    runtime = Mock(spec=Yue2Runtime)
    runtime.vae = Mock()
    runtime.vae.config.sample_rate = 48000
    runtime.render.return_value = torch.zeros(2, 96)
    scheduler = Yue2AcousticScheduler(runtime)
    state = Yue2State(
        semantic_prefix=[1, 2],
        semantic_tokens=[CODEC_OFFSET + 3],
        seed=11,
        ode_steps=5,
        semantic_truncated=True,
        finish_reason="length",
    )
    message = store_state(payload(), state)
    result = scheduler.render(message)
    prefix, tokens, seed, ode_steps, cancelled = runtime.render.call_args.args
    assert (prefix, tokens, seed, ode_steps) == ([1, 2], [CODEC_OFFSET + 3], 11, 5)
    assert not cancelled()
    assert result.request_id == message.request_id
    assert result.data["finish_reason"] == "length"


@torch.inference_mode()
def test_acoustic_seed_and_cancellation(model: Yue2Model) -> None:
    codec_tokens = list(range(65))
    first = synthesize_latents(model, [1, 2], codec_tokens, 7, 2, lambda: False)
    assert first.shape == (65, 64)
    assert torch.isfinite(first).all()
    polls = 0

    def cancelled() -> bool:
        nonlocal polls
        polls += 1
        return polls >= 3

    with pytest.raises(InterruptedError):
        synthesize_latents(model, [1, 2], codec_tokens, 7, 2, cancelled)
    recovered = synthesize_latents(model, [1, 2], codec_tokens, 7, 2, lambda: False)
    torch.testing.assert_close(first, recovered, rtol=0, atol=0)


@torch.inference_mode()
def test_tiled_vae_preserves_full_waveform_and_tail() -> None:
    decoder = DecoderConfig(channels=2, c_mults=[1, 2], strides=[2, 2])
    vae = Yue2Vae(VaeConfig(decoder_config=decoder, downsampling_ratio=4)).eval()
    torch.manual_seed(42)
    latents = torch.randn(1, 64, 400)
    expected = vae.decoder(latents).clamp(-1, 1)
    actual = vae.decode_tiled(latents, 31, vae.required_halo(31), lambda: False)
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError, match="halo"):
        vae.decode_tiled(latents, 31, 0, lambda: False)


def test_weight_loading_rejects_missing_tensors(
    model: Yue2Model, tmp_path: Path
) -> None:
    weights = dict(model.state_dict())
    weights["lm_head.weight"] = torch.zeros(1)
    save_file(weights, tmp_path / "model.safetensors")
    load_weights(model, tmp_path, skipped_prefix="lm_head.")
    weights.pop("llm2vae.weight")
    save_file(weights, tmp_path / "model.safetensors")
    with pytest.raises(RuntimeError, match="llm2vae.weight"):
        load_weights(model, tmp_path, skipped_prefix="lm_head.")


@torch.inference_mode()
def test_projection_fusion_preserves_outputs_and_checkpoint(model: Yue2Model) -> None:
    tokens = torch.tensor([[1, 2, 3, 4]])
    hidden_states = torch.linspace(-1, 1, 32).reshape(1, 4, 8)
    layer = model.model.layers[0]
    expected_cache = TokenCache(capacity=4)
    model(tokens, expected_cache)
    expected_mlp = layer.mlp(hidden_states)
    names = set(model.state_dict())
    layer.self_attn.fuse_projections()
    layer.mlp.fuse_projections()
    actual_cache = TokenCache(capacity=4)
    model(tokens, actual_cache)
    for actual, expected in [
        (actual_cache.keys[0], expected_cache.keys[0]),
        (actual_cache.values[0], expected_cache.values[0]),
        (layer.mlp(hidden_states), expected_mlp),
    ]:
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert set(model.state_dict()) == names


def test_fused_kernels_reject_cpu_weights(model: Yue2Model) -> None:
    with pytest.raises(ValueError, match="CUDA BF16"):
        model.enable_fused_kernels()


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("request_count,token_count", [(1, 1), (3, 1), (1, 258)])
@torch.inference_mode()
def test_fused_layers_preserve_normalization_rotation_and_residual(
    request_count: int, token_count: int
) -> None:
    torch.manual_seed(42)
    config = Yue2ModelConfig(intermediate_size=512)
    attention = Attention(config).cuda().bfloat16()
    normalization = RMSNorm(config.hidden_size, config.rms_norm_eps).cuda().bfloat16()
    mlp = MLP(config).cuda().bfloat16()
    attention.q_norm.weight.normal_(1.0, 0.2)
    attention.k_norm.weight.normal_(0.8, 0.3)
    normalization.weight.normal_(1.0, 0.2)
    hidden_states = torch.randn(
        request_count,
        token_count,
        config.hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    residual = torch.randn_like(hidden_states)
    positions = (
        torch.arange(
            request_count * token_count, device="cuda", dtype=torch.int32
        ).view(request_count, token_count)
        + 1848
    )
    cosine, sine = RotaryEmbedding(config.head_dim, config.rope_theta)(positions)
    attention.fuse_projections()
    mlp.fuse_projections()
    expected_qkv = attention.project_qkv(hidden_states, cosine, sine, positions)
    expected_normalized, expected_residual = normalization.add(
        hidden_states.clone(), residual.clone()
    )
    expected_mlp = mlp(hidden_states)
    expected_norm = normalization(hidden_states)
    attention.fused_kernels = normalization.fused_kernels = mlp.fused_kernels = True
    actual_qkv = attention.project_qkv(hidden_states, cosine, sine, positions)
    actual_normalized, actual_residual = normalization.add(
        hidden_states.clone(), residual.clone()
    )
    torch.testing.assert_close(actual_residual, expected_residual, rtol=0, atol=0)
    torch.testing.assert_close(actual_qkv[2], expected_qkv[2], rtol=0, atol=0)
    # note (ruiling): Fused kernels keep intermediates in FP32; seeded BF16 probes measured relative RMS differences below 0.005.
    for actual, expected in zip(
        (
            *actual_qkv[:2],
            actual_normalized,
            mlp(hidden_states),
            normalization(hidden_states),
        ),
        (*expected_qkv[:2], expected_normalized, expected_mlp, expected_norm),
    ):
        difference = (actual.float() - expected.float()).square().mean().sqrt()
        assert torch.isfinite(actual).all()
        assert difference / expected.float().square().mean().sqrt() < 0.01
