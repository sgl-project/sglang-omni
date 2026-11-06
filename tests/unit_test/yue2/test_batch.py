# SPDX-License-Identifier: Apache-2.0
"""Request isolation, completion, and error handling for YuE2 batches."""

from collections import OrderedDict
from unittest.mock import Mock

import pytest
import torch

from sglang_omni.models.yue2.batch_generation import TokenRequest, generate_batch_tokens
from sglang_omni.models.yue2.generation import TokenResult
from sglang_omni.models.yue2.layers import Yue2ModelConfig
from sglang_omni.models.yue2.model import Yue2Model
from sglang_omni.models.yue2.request import (
    CODEC_OFFSET,
    SamplingConfig,
    Yue2Options,
    Yue2Request,
    Yue2Tokenizer,
)
from sglang_omni.models.yue2.runtime import SongResult, Yue2Runtime
from sglang_omni.models.yue2.stages import Yue2Scheduler
from sglang_omni.models.yue2.vae import Yue2Vae
from sglang_omni.proto.request import OmniRequest, StagePayload


def test_scheduler_preserves_batch_identity_and_invalid_requests() -> None:
    runtime = Mock(spec=Yue2Runtime)
    runtime.generate_batch.return_value = [
        SongResult(
            audio=torch.zeros(2, 32),
            sample_rate=48000,
            abc_tokens=[],
            codec_tokens=[0],
            abc_truncated=False,
            semantic_truncated=True,
        ),
        ValueError("YuE2 request: context exhausted"),
    ]
    scheduler = Yue2Scheduler(
        runtime, max_batch_size=3, max_batch_wait_ms=0, batch_graph_cache_size=2
    )
    payloads = [
        StagePayload(
            request_id=str(index),
            data={},
            request=OmniRequest(
                inputs="Sing",
                metadata={"tts_params": {"seed": index}},
                params={"stage_params": {"synthesis": {"cot": cot}}},
            ),
        )
        for index, cot in enumerate(["full", "invalid", "melody"])
    ]
    results = scheduler.generate_batch(payloads)
    assert results[0].request_id == "0"
    assert results[0].data["finish_reason"] == "length"
    assert isinstance(results[1], ValueError) and "YuE2 request:" in str(results[1])
    assert isinstance(results[2], ValueError) and "context exhausted" in str(results[2])
    requests, cancellations, cache_size = runtime.generate_batch.call_args.args
    assert [request.seed for request in requests] == [0, 2]
    scheduler.abort("2")
    assert not cancellations[0]() and cancellations[1]()
    assert cache_size == 2


def test_runtime_batches_planning_and_preserves_per_request_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = Mock(spec=Yue2Tokenizer)
    tokenizer.prefix.side_effect = lambda request, abc: [request.seed + 1, *(abc or [])]
    tokenizer.encode.return_value = [11, 12]
    vae = Mock(spec=Yue2Vae)
    vae.required_halo.return_value = 16
    runtime = Yue2Runtime(
        Mock(spec=Yue2Model),
        vae,
        tokenizer,
        vae_core_frames=1024,
        vae_halo_frames=16,
        ar_cuda_graph=True,
    )
    requests = [
        Yue2Request(
            lyrics="Sing",
            style="pop",
            seed=index,
            options=Yue2Options(
                cot=cot, abc=abc, abc_max_tokens=index + 3, ode_steps=index + 2
            ),
            semantic_sampling=SamplingConfig(min_tokens=1, max_tokens=index + 2),
        )
        for index, (cot, abc) in enumerate(
            [("full", None), ("melody", None), ("full", "score")]
        )
    ]
    batched = Mock(
        side_effect=[
            [
                TokenResult(tokens=[8], truncated=True),
                ValueError("YuE2 request: bad plan"),
            ],
            [
                TokenResult(tokens=[CODEC_OFFSET], truncated=False),
                TokenResult(tokens=[CODEC_OFFSET + 1], truncated=True),
            ],
        ]
    )
    monkeypatch.setattr(
        "sglang_omni.models.yue2.runtime.generate_batch_tokens", batched
    )
    song = SongResult(
        audio=torch.zeros(2, 32),
        sample_rate=48000,
        abc_tokens=[8],
        codec_tokens=[0],
        abc_truncated=True,
        semantic_truncated=False,
    )
    rendered = Mock(side_effect=[song, InterruptedError("cancelled")])
    monkeypatch.setattr(runtime, "render", rendered)
    results = runtime.generate_batch(requests, [lambda: False] * 3, 2)
    assert results[0] is song
    assert isinstance(results[1], ValueError)
    assert isinstance(results[2], InterruptedError)
    planning_inputs = batched.call_args_list[0].args[1]
    semantic_inputs = batched.call_args_list[1].args[1]
    assert [item.sampling.max_tokens for item in planning_inputs] == [3, 4]
    assert [item.seed for item in semantic_inputs] == [0, 2]
    assert [item.sampling.max_tokens for item in semantic_inputs] == [2, 4]
    assert semantic_inputs[1].prefix == [3, 11, 12]
    assert rendered.call_args_list[0].args[2].truncated
    assert rendered.call_args_list[1].args[0].options.ode_steps == 4


@pytest.fixture
def cuda_model() -> Yue2Model:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    else:
        pass
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(17)
        model = Yue2Model(
            Yue2ModelConfig(
                hidden_size=64,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=32,
                intermediate_size=128,
                max_position_embeddings=64,
                max_latent_frames=64,
            )
        )
    return model.to("cuda").bfloat16().eval().requires_grad_(False)


@pytest.mark.accelerator
@torch.inference_mode()
def test_batch_finishes_and_cancels_rows_at_the_context_boundary(
    cuda_model: Yue2Model,
) -> None:
    cuda_model.lm_head.weight.zero_()
    polls = 0

    def cancel_during_decode() -> bool:
        nonlocal polls
        polls += 1
        return polls >= 4

    requests = [
        TokenRequest(
            prefix=prefix,
            seed=seed,
            sampling=SamplingConfig(
                temperature=0.0,
                min_tokens=minimum,
                max_tokens=budget,
            ),
            cancelled=cancelled,
            legacy_off=False,
        )
        for prefix, seed, minimum, budget, cancelled in [
            ([1] * 61, 0, 2, 3, lambda: False),
            ([2, 3], 1, 12, 13, lambda: False),
            ([4], 2, 6, 7, cancel_during_decode),
            ([5] * 64, 3, 1, 2, lambda: False),
        ]
    ]
    graphs = OrderedDict()
    results = generate_batch_tokens(cuda_model, requests, "semantic", graphs, 2)
    assert results[0].tokens == [CODEC_OFFSET] * 2 and not results[0].truncated
    assert results[1].tokens == [CODEC_OFFSET] * 12 and not results[1].truncated
    assert isinstance(results[2], InterruptedError)
    assert isinstance(results[3], ValueError)
    requests[2].cancelled = lambda: False
    recovered = generate_batch_tokens(cuda_model, requests[:3], "semantic", graphs, 2)
    assert recovered[2].tokens == [CODEC_OFFSET] * 6 and not recovered[2].truncated


@pytest.mark.accelerator
@torch.inference_mode()
def test_batch_keeps_seed_parameter_and_order_independent(
    cuda_model: Yue2Model,
) -> None:
    requests = [
        TokenRequest(
            prefix=[index + 1, 3, 7],
            seed=seed,
            sampling=SamplingConfig(
                min_tokens=8,
                max_tokens=8,
                temperature=temperature,
                top_k=top_k,
            ),
            cancelled=lambda: False,
            legacy_off=False,
        )
        for index, (seed, temperature, top_k) in enumerate(
            [(7, 0.7, 30), (19, 1.2, 100)]
        )
    ]
    graphs = OrderedDict()
    original = generate_batch_tokens(cuda_model, requests, "semantic", graphs, 1)
    repeated = generate_batch_tokens(cuda_model, requests, "semantic", graphs, 1)
    assert original == repeated
    reverse = generate_batch_tokens(
        cuda_model, list(reversed(requests)), "semantic", graphs, 1
    )
    assert original == list(reversed(reverse))
    requests[0].seed = 38
    changed = generate_batch_tokens(cuda_model, requests, "semantic", graphs, 1)
    assert original[1] == changed[1]
    generate_batch_tokens(cuda_model, requests[:1], "abc", graphs, 1)
    requests[0].seed = 7
    after_eviction = generate_batch_tokens(cuda_model, requests, "semantic", graphs, 1)
    assert original == after_eviction
