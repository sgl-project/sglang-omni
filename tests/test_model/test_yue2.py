# SPDX-License-Identifier: Apache-2.0
"""Real-checkpoint generation and cancellation checks for one CUDA device."""

import os

import pytest
import torch

from sglang_omni.models.yue2.acoustic import synthesize_latents
from sglang_omni.models.yue2.cuda_graph import DecodeGraph
from sglang_omni.models.yue2.generation import SamplingGraph
from sglang_omni.models.yue2.request import (
    SamplingConfig,
    Yue2Options,
    Yue2Request,
    build_request,
)
from sglang_omni.models.yue2.runtime import SongResult, Yue2Runtime
from sglang_omni.proto.request import OmniRequest, StagePayload

pytestmark = [
    pytest.mark.accelerator,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


@pytest.fixture(scope="module", params=[False, True])
def runtime(request: pytest.FixtureRequest) -> Yue2Runtime:
    return Yue2Runtime.from_pretrained(
        os.environ.get("YUE2_MODEL_PATH", "m-a-p/YuE2-3B"),
        os.environ.get("YUE2_VAE_PATH", "m-a-p/YuE2-Vae"),
        torch.device("cuda:0"),
        vae_core_frames=1024,
        vae_halo_frames=16,
        ar_cuda_graph=request.param,
        nar_cuda_graph=request.param,
        sampling_cuda_graph=request.param,
        fuse_projections=request.param,
    )


@pytest.mark.parametrize(
    "cot,abc,cfg_scale",
    [
        ("full", None, 1.0),
        ("melody", None, 1.5),
        ("off", None, None),
        ("melody", "X:1\nT:Morning\nM:4/4\nL:1/4\nK:C\nC D E F |", 1.5),
    ],
)
def test_real_checkpoint_generation(
    runtime: Yue2Runtime,
    cot: str,
    abc: str | None,
    cfg_scale: float | None,
) -> None:
    message = StagePayload(
        request_id="song",
        data={},
        request=OmniRequest(
            inputs="[Verse]\nMorning light is shining through",
            metadata={
                "tts_params": {
                    "instructions": "Gentle acoustic pop, warm female vocals",
                    "seed": 42,
                }
            },
            params={
                "max_new_tokens": 16,
                "stage_params": {
                    "synthesis": {
                        "cot": cot,
                        "abc": abc,
                        "cfg_scale": cfg_scale,
                        "abc_max_tokens": 8,
                        "ode_steps": 2,
                    }
                },
            },
        ),
    )
    request = build_request(message)
    song = runtime.generate(request, lambda: False)
    assert song.audio.shape[0] == 2 and song.audio.shape[1] > 0
    assert song.audio.dtype == torch.float32 and torch.isfinite(song.audio).all()
    assert song.sample_rate == 48000
    assert len(song.codec_tokens) == 16 and song.finish_reason == "length"
    cancellation_checks = 0

    def cancelled_after_prefill() -> bool:
        nonlocal cancellation_checks
        cancellation_checks += 1
        return cancellation_checks >= 3

    with pytest.raises(InterruptedError):
        runtime.generate(request, cancelled_after_prefill)
    recovered = runtime.generate(request, lambda: False)
    assert recovered.abc_tokens == song.abc_tokens
    assert recovered.codec_tokens == song.codec_tokens
    torch.testing.assert_close(recovered.audio, song.audio, rtol=0, atol=0)


def test_acoustic_graph_preserves_solver_and_abort_recovery(
    runtime: Yue2Runtime,
) -> None:
    prefix, codecs = [1, 2, 3], list(range(64))
    expected = synthesize_latents(runtime.model, prefix, codecs, 42, 7, lambda: False)
    polls = 0

    def cancelled() -> bool:
        nonlocal polls
        polls += 1
        return polls >= 5

    with pytest.raises(InterruptedError):
        synthesize_latents(
            runtime.model, prefix, codecs, 42, 7, cancelled, cuda_graph=True
        )
    actual = synthesize_latents(
        runtime.model, prefix, codecs, 42, 7, lambda: False, cuda_graph=True
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@torch.inference_mode()
def test_decode_graph_survives_temporary_graphs(runtime: Yue2Runtime) -> None:
    """Destroying sampling graphs must not invalidate a persistent decode graph."""
    runner = DecodeGraph(runtime.model)
    prefix = [1, 2, 3]
    logits = runner.prefill([prefix])
    token = torch.tensor([[7]], device=logits.device)
    expected = runner.step(token, [True]).clone()
    sampling = SamplingConfig(min_tokens=1, max_tokens=2)
    # note (ruiling): Forty captures wrap PyTorch's 32-stream pool and reproduce workspace invalidation.
    for _ in range(40):
        sampler = SamplingGraph(logits, sampling, "semantic", False)
        sampler.step(logits, 0, None)
        del sampler
    torch.cuda.empty_cache()
    runner.prefill([prefix])
    torch.testing.assert_close(runner.step(token, [True]), expected, rtol=0, atol=0)


def test_real_batch_isolates_settings_cancellation_and_context_errors(
    runtime: Yue2Runtime,
) -> None:
    requests = [
        Yue2Request(
            lyrics="[Verse]\nMorning light is shining through",
            style="Gentle acoustic pop, warm female vocals",
            seed=42 + index,
            options=Yue2Options(
                cot=cot,
                abc=abc,
                cfg_scale=guidance,
                abc_max_tokens=4 + index,
                ode_steps=2 + index,
            ),
            semantic_sampling=SamplingConfig(
                min_tokens=8 + index,
                max_tokens=8 + index,
                temperature=0.7 + index / 10,
            ),
        )
        for index, (cot, abc, guidance) in enumerate(
            [
                ("full", None, 1.0),
                ("melody", "X:1\nK:C\nC D E F |", 1.0),
                ("off", None, 1.0),
                ("full", None, 1.0),
                ("full", None, 1.0),
                ("melody", None, 1.5),
            ]
        )
    ]
    requests[4].semantic_sampling.max_tokens = (
        runtime.model.config.max_position_embeddings
    )
    callbacks = [lambda: False for _ in requests]
    callbacks[3] = lambda: True
    original = runtime.generate_batch(requests, callbacks, 2)
    assert isinstance(original[3], InterruptedError)
    assert isinstance(original[4], ValueError)
    for index in (0, 1, 2, 5):
        song = original[index]
        assert isinstance(song, SongResult)
        assert len(song.codec_tokens) == 8 + index
        assert song.audio.shape == (2, (8 + index) * 1920 - 64)
        assert torch.isfinite(song.audio).all() and song.sample_rate == 48000
        assert song.semantic_truncated and song.finish_reason == "length"
    assert original[0].abc_truncated and not original[1].abc_truncated
    assert not original[2].abc_tokens
    repeated = runtime.generate_batch(requests, callbacks, 2)
    for index in (0, 1, 2, 5):
        assert original[index].abc_tokens == repeated[index].abc_tokens
        assert original[index].codec_tokens == repeated[index].codec_tokens
        torch.testing.assert_close(
            original[index].audio, repeated[index].audio, rtol=0, atol=0
        )
