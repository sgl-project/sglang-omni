# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import tiktoken
import torch
from safetensors.torch import save_file

from sglang_omni.config.runtime import resolve_factory_signature_args
from sglang_omni.models.yue2.acoustic import AcousticChunk, synthesize_latents
from sglang_omni.models.yue2.config import Yue2PipelineConfig
from sglang_omni.models.yue2.cuda_graph import DecodeGraph
from sglang_omni.models.yue2.generation import (
    SamplingGraph,
    TokenResult,
    generate_tokens,
    sampling_scores,
)
from sglang_omni.models.yue2.layers import TokenCache, Yue2ModelConfig
from sglang_omni.models.yue2.model import Yue2Model
from sglang_omni.models.yue2.request import (
    ABC_END,
    CODEC_OFFSET,
    CODEC_SIZE,
    MUSIC_END,
    SamplingConfig,
    Yue2Tokenizer,
    build_request,
)
from sglang_omni.models.yue2.runtime import SongResult, Yue2Runtime, load_weights
from sglang_omni.models.yue2.stages import Yue2Scheduler, create_synthesis_executor
from sglang_omni.models.yue2.vae import DecoderConfig, VaeConfig, Yue2Vae
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.message import IncomingMessage
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


def test_pipeline_configuration_reaches_factory() -> None:
    configuration = Yue2PipelineConfig(model_path="m-a-p/YuE2-3B")
    stage = configuration.stages[0]
    arguments = resolve_factory_signature_args(
        create_synthesis_executor,
        stage.factory.model_dump(exclude_none=True),
        defaults={"model_path": configuration.model_path, "gpu_id": 0},
    )
    assert stage.process == "synthesis" and stage.terminal
    assert arguments["model_path"] == configuration.model_path
    assert arguments["vae_path"] == "m-a-p/YuE2-Vae"
    assert arguments["gpu_id"] == 0


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


@pytest.mark.parametrize("token_budget", [0, -1, True, 1.5])
def test_invalid_token_budget_rejected(token_budget: object) -> None:
    message = payload()
    message.request.params["max_new_tokens"] = token_budget
    with pytest.raises(ValueError):
        build_request(message)


def test_scheduler_classifies_request_and_compute_failures() -> None:
    """Invalid stage options must not become HTTP 500 responses."""
    runtime = Mock(spec=Yue2Runtime)
    runtime.generate.side_effect = ValueError("Invalid tensor shape")
    scheduler = Yue2Scheduler(runtime)
    with pytest.raises(ValueError) as invalid_request:
        scheduler.generate(payload(cot="unknown"))
    assert generation_error_status_code(RuntimeError(str(invalid_request.value))) == 400
    runtime.generate.assert_not_called()
    with pytest.raises(ValueError) as compute_failure:
        scheduler.generate(payload())
    assert generation_error_status_code(RuntimeError(str(compute_failure.value))) == 500


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
    expected = model(tokens, TokenCache(capacity=4))
    cache = TokenCache(capacity=4)
    model(tokens[:, :3], cache)
    actual = model(tokens[:, 3:], cache)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("phase,end_token", [("abc", ABC_END), ("semantic", MUSIC_END)])
def test_sampling_masks_and_minimum_length(phase: str, end_token: int) -> None:
    sampling = SamplingConfig(temperature=0.0, min_tokens=1, max_tokens=2)
    logits = torch.zeros(1, 184704)
    logits[:, end_token] = torch.inf
    blocked = sampling_scores(logits, sampling, [], phase, False)
    allowed = sampling_scores(logits, sampling, [1], phase, False)
    assert torch.isneginf(blocked[:, end_token]).all()
    assert allowed.argmax().item() == end_token
    if phase == "semantic":
        assert torch.isneginf(blocked[:, :CODEC_OFFSET]).all()
    else:
        assert torch.isneginf(blocked[:, CODEC_OFFSET:]).all()


def test_seed_and_budget_remain_request_local(model: Yue2Model) -> None:
    sampling = SamplingConfig(min_tokens=8, max_tokens=8)
    first = generate_tokens(model, [1, 2], sampling, 7, "semantic", lambda: False)
    other = generate_tokens(model, [1, 2], sampling, 9, "semantic", lambda: False)
    repeat = generate_tokens(model, [1, 2], sampling, 7, "semantic", lambda: False)
    assert first.tokens == repeat.tokens
    assert first.tokens != other.tokens
    assert len(first.tokens) == 8 and first.truncated
    assert all(
        CODEC_OFFSET <= token < CODEC_OFFSET + CODEC_SIZE for token in first.tokens
    )
    short = generate_tokens(
        model,
        [1, 2],
        SamplingConfig(min_tokens=2, max_tokens=2),
        7,
        "semantic",
        lambda: False,
    )
    assert len(short.tokens) == 2 and short.truncated


@pytest.mark.parametrize("phase", ["abc", "acoustic"])
def test_context_overflow_fails_before_prefill(
    phase: str, model: Yue2Model, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AR-fit prompts can still leave no room for an acoustic frame."""
    forward = Mock(side_effect=AssertionError("must not execute"))
    monkeypatch.setattr(model.model.embed_tokens, "forward", forward)
    with pytest.raises(ValueError, match="context") as invalid_request:
        if phase == "abc":
            generate_tokens(
                model,
                [1, 2],
                SamplingConfig(min_tokens=0, max_tokens=128),
                7,
                "abc",
                lambda: False,
            )
        else:
            synthesize_latents(model, list(range(124)), [0], 7, 2, lambda: False)
    forward.assert_not_called()
    assert generation_error_status_code(RuntimeError(str(invalid_request.value))) == 400


def test_abort_then_recovery_releases_request_cache(model: Yue2Model) -> None:
    polls = 0

    def cancelled() -> bool:
        nonlocal polls
        polls += 1
        return polls >= 3

    sampling = SamplingConfig(min_tokens=2, max_tokens=2)
    with pytest.raises(InterruptedError):
        generate_tokens(model, [1, 2], sampling, 7, "semantic", cancelled)
    recovered = generate_tokens(model, [1, 2], sampling, 7, "semantic", lambda: False)
    assert len(recovered.tokens) == 2


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
    weights.pop("lm_head.weight")
    save_file(weights, tmp_path / "model.safetensors")
    with pytest.raises(RuntimeError, match="lm_head.weight"):
        load_weights(model, tmp_path, decoder_only=False)


@pytest.mark.parametrize(
    "cot,abc,abc_truncated,semantic_truncated",
    [
        ("full", None, True, False),
        ("melody", "score", False, True),
        ("off", None, False, False),
    ],
)
def test_runtime_propagates_options_and_truncation(
    cot: str,
    abc: str | None,
    abc_truncated: bool,
    semantic_truncated: bool,
    model: Yue2Model,
    tokenizer: Yue2Tokenizer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generate = Mock(
        side_effect=(
            [
                TokenResult(tokens=[1, 2], truncated=abc_truncated),
                TokenResult(tokens=[CODEC_OFFSET], truncated=semantic_truncated),
            ]
            if cot != "off" and abc is None
            else [TokenResult(tokens=[CODEC_OFFSET], truncated=semantic_truncated)]
        )
    )
    acoustic = Mock(return_value=torch.zeros(1, 64))
    vae = Mock(spec=Yue2Vae)
    vae.config = VaeConfig(decoder_config=DecoderConfig())
    vae.required_halo.return_value = 16
    vae.decode_tiled.return_value = torch.zeros(1, 2, 1856)
    monkeypatch.setattr("sglang_omni.models.yue2.runtime.generate_tokens", generate)
    monkeypatch.setattr("sglang_omni.models.yue2.runtime.synthesize_latents", acoustic)
    runtime = Yue2Runtime(
        model, vae, tokenizer, vae_core_frames=1024, vae_halo_frames=16
    )
    request = build_request(
        payload(19, 3, cot=cot, abc=abc, cfg_scale=1.5, ode_steps=7)
    )
    song = runtime.generate(request, lambda: False)
    assert song.finish_reason == (
        "length" if abc_truncated or semantic_truncated else "stop"
    )
    assert generate.call_args.args[2].max_tokens == 3
    assert generate.call_args.args[3] == 19
    assert generate.call_args.kwargs["negative"] == tokenizer.negative_prefix(
        request, song.abc_tokens
    )
    assert acoustic.call_args.args[3:5] == (19, 7)


def test_scheduler_queues_distinct_requests_and_handles_abort() -> None:
    entered = threading.Event()
    released = threading.Event()
    runtime = Mock(spec=Yue2Runtime)

    def generate(request, cancelled):
        if request.seed == 1:
            entered.set()
            assert released.wait(5)
            assert cancelled()
            raise InterruptedError("cancelled")
        return SongResult(
            audio=torch.ones(2, 10),
            sample_rate=48000,
            abc_tokens=[],
            codec_tokens=[1],
            abc_truncated=False,
            semantic_truncated=True,
        )

    runtime.generate.side_effect = generate
    scheduler = Yue2Scheduler(runtime)
    worker = threading.Thread(target=scheduler.start)
    worker.start()
    try:
        for seed, budget in [(1, 3), (2, 9)]:
            message = payload(seed, budget, ode_steps=seed)
            scheduler.enqueue(
                IncomingMessage(
                    request_id=message.request_id, type="new_request", data=message
                )
            )
        assert entered.wait(5)
        scheduler.abort("song-1")
        released.set()
        result = scheduler.outbox.get(timeout=5)
        assert result.request_id == "song-2" and result.type == "result"
        assert result.data.data["finish_reason"] == "length"
        assert result.data.data["audio_waveform_shape"] == [2, 10]
        np.testing.assert_array_equal(
            np.frombuffer(result.data.data["audio_waveform"], dtype=np.float32),
            np.ones(20),
        )
        requests = [call.args[0] for call in runtime.generate.call_args_list]
        assert [
            (
                request.seed,
                request.semantic_sampling.max_tokens,
                request.options.ode_steps,
            )
            for request in requests
        ] == [(1, 3, 1), (2, 9, 2)]
        assert scheduler.outbox.empty()
    finally:
        released.set()
        scheduler.stop()
        worker.join(5)
    assert not worker.is_alive()


@torch.inference_mode()
def test_projection_fusion_preserves_logits_and_checkpoint(model: Yue2Model) -> None:
    tokens = torch.tensor([[1, 2, 3, 4]])
    expected = model(tokens, TokenCache(capacity=4))
    names = set(model.state_dict())
    for layer in model.model.layers:
        layer.self_attn.fuse_projections()
        layer.mlp.fuse_projections()
    actual = model(tokens, TokenCache(capacity=4))
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert set(model.state_dict()) == names


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("device_index", [0, 1])
@torch.inference_mode()
def test_decode_graph_replay_and_request_reset(device_index: int) -> None:
    if torch.cuda.device_count() <= device_index:
        pytest.skip("Second GPU required")
    else:
        pass
    device = torch.device("cuda", device_index)
    torch.manual_seed(17)
    model = (
        Yue2Model(
            Yue2ModelConfig(
                hidden_size=64,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=32,
                intermediate_size=128,
                max_position_embeddings=128,
                max_latent_frames=128,
            )
        )
        .to(device)
        .bfloat16()
        .eval()
    )
    runner = DecodeGraph(model)
    for prefix in [[1, 2, 3, 4, 5], [9, 8]]:
        runner.prefill(prefix)
        token = torch.tensor([[7]], device=device)
        runner.tokens.copy_(token)
        expected = runner.forward().clone()
        runner.position.fill_(len(prefix))
        actual = runner.step(token)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        fresh = DecodeGraph(model)
        fresh.prefill(prefix)
        torch.testing.assert_close(actual, fresh.step(token), rtol=0, atol=0)
        sampling = SamplingConfig(min_tokens=1, max_tokens=2)
        sampler = SamplingGraph(actual, sampling, "semantic", False)
        torch.testing.assert_close(
            sampler.step(actual, 0, None),
            sampling_scores(actual, sampling, [], "semantic", False),
            rtol=0,
            atol=0,
        )
        chunk = AcousticChunk(model, prefix, torch.randn(8, 64))
        expected_latents = chunk.solve(2, lambda: False)
        torch.testing.assert_close(
            chunk.solve(2, lambda: False, cuda_graph=True),
            expected_latents,
            rtol=0,
            atol=0,
        )


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "phase,legacy_off,temperature",
    [
        ("abc", False, 0.7),
        ("semantic", False, 1.0),
        ("semantic", True, 1.0),
        ("semantic", False, 0.0),
    ],
)
@torch.inference_mode()
def test_sampling_graph_preserves_masks_ties_and_history(
    phase: str,
    legacy_off: bool,
    temperature: float,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(17)
    logits = torch.randn(
        1, 184704, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    sampling = SamplingConfig(
        temperature=temperature, min_tokens=2, max_tokens=8, penalty_window=3, top_k=4
    )
    runner = SamplingGraph(logits, sampling, phase, legacy_off)
    history: list[int] = []
    previous = None
    for step in range(8):
        # note (ruiling): BF16 ties at the top-k boundary must retain the eager distribution.
        logits[:, CODEC_OFFSET : CODEC_OFFSET + 6] = 10
        expected = sampling_scores(logits, sampling, history, phase, legacy_off)
        actual = runner.step(logits, step, previous)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        token = CODEC_OFFSET + step % 2
        history.append(token)
        previous = torch.tensor([[token]], device="cuda")
