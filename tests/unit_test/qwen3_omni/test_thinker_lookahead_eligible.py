# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ThinkerModelRunner.lookahead_eligible and the lookahead sampling.

lookahead_eligible reads only per-request flags (never other instance state), so it
is exercised on a bare instance built with ``object.__new__`` and stand-in requests.
Audio-output detection (should_generate_audio_output) is stubbed on the stand-in
stage_payload.
"""
from __future__ import annotations

import types

import pytest
import torch
from sglang.srt.layers.sampler import multinomial_with_seed

from sglang_omni.model_runner.base import rank_shared_unseeded_sampling_seed
from sglang_omni.model_runner.thinker_model_runner import ThinkerModelRunner


@pytest.fixture(autouse=True)
def stub_audio_output(monkeypatch):
    monkeypatch.setattr(
        "sglang_omni.models.qwen3_omni.request_builders.should_generate_audio_output",
        lambda payload: payload == "audio",
    )


def runner() -> ThinkerModelRunner:
    return object.__new__(ThinkerModelRunner)


def sp(**kw):
    d = dict(
        repetition_penalty=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        min_new_tokens=0,
        sampling_seed=None,
        logit_bias=None,
        custom_params=None,
    )
    d.update(kw)
    return types.SimpleNamespace(**d)


def req(return_logprob=False, stage_payload="text", **sp_kw):
    return types.SimpleNamespace(
        sampling_params=sp(**sp_kw),
        omni_data=types.SimpleNamespace(
            return_logprob=return_logprob, stage_payload=stage_payload
        ),
    )


def batch(*reqs):
    return types.SimpleNamespace(reqs=list(reqs))


def test_plain_greedy_is_eligible():
    assert runner().lookahead_eligible(batch(req(), req())) is True


def test_audio_output_disables_lookahead():
    # an audio-output request captures hidden for the talker -> route to sync.
    assert runner().lookahead_eligible(batch(req(stage_payload="audio"))) is False


def test_return_logprob_disables_lookahead():
    assert runner().lookahead_eligible(batch(req(return_logprob=True))) is False


def test_missing_or_noneomni_data_falls_to_sync():
    # request data missing or None cannot be inspected -> fail closed to sync
    # (never raise, never let a possible hidden-capture batch onto async).
    no_data = types.SimpleNamespace(sampling_params=sp())
    assert runner().lookahead_eligible(batch(no_data)) is False
    none_data = types.SimpleNamespace(sampling_params=sp(), omni_data=None)
    assert runner().lookahead_eligible(batch(none_data)) is False


def test_each_gated_sampling_param_disables_lookahead():
    for kw in (
        dict(repetition_penalty=1.3),
        dict(presence_penalty=0.5),
        dict(frequency_penalty=0.5),
        dict(min_new_tokens=5),
        dict(logit_bias={1: 2.0}),
        dict(custom_params={"x": 1}),
    ):
        assert runner().lookahead_eligible(batch(req(**kw))) is False, kw


def test_one_gated_request_disables_whole_batch():
    audio_mix = batch(req(), req(stage_payload="audio"), req())
    assert runner().lookahead_eligible(audio_mix) is False
    param_mix = batch(req(), req(repetition_penalty=1.3), req())
    assert runner().lookahead_eligible(param_mix) is False


def test_seeded_batch_is_eligible() -> None:
    seeded_mix = batch(req(sampling_seed=42), req())
    assert runner().lookahead_eligible(seeded_mix) is True


def scheduler_request(
    request_id: str, sampling_seed: int | None
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        request_id=request_id,
        data=types.SimpleNamespace(
            suppress_tokens=None,
            req=types.SimpleNamespace(
                sampling_params=types.SimpleNamespace(sampling_seed=sampling_seed)
            ),
        ),
    )


def seeded_sampling_info(device: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        sampling_seed=None,
        device=device,
        need_min_p_sampling=False,
        need_top_p_sampling=False,
        need_top_k_sampling=False,
    )


def sample_lookahead_seeds(
    requests: list[types.SimpleNamespace],
) -> torch.Tensor | None:
    def sample(
        logits_output: types.SimpleNamespace, forward_batch: types.SimpleNamespace
    ) -> torch.Tensor:
        return torch.zeros(len(requests), dtype=torch.long)

    thinker_runner = runner()
    thinker_runner.tp_worker = types.SimpleNamespace(
        model_runner=types.SimpleNamespace(sample=sample)
    )
    thinker_runner.sampling_seed_batch_key = ()
    thinker_runner.sampling_seed_batch_tensor = None
    forward_batch = types.SimpleNamespace(sampling_info=seeded_sampling_info("cpu"))
    logits_output = types.SimpleNamespace(next_token_logits=None)
    thinker_runner.sample_lookahead(logits_output, forward_batch, requests)
    return forward_batch.sampling_info.sampling_seed


def test_sample_lookahead_installs_request_seeds() -> None:
    seeds = sample_lookahead_seeds(
        [scheduler_request("a", 42), scheduler_request("b", 7)]
    )
    assert seeds.tolist() == [42, 7]


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_overlapped_lookahead_steps_draw_the_synchronous_seeded_tokens() -> None:
    """Launching step N+1 before step N resolves draws, per step, the tokens a
    synchronous sample at that step's positions draws."""
    requests = [scheduler_request("seeded", 42), scheduler_request("unseeded", None)]
    generator = torch.Generator(device="cpu").manual_seed(0)
    step_logprobs = [
        torch.log_softmax(
            torch.randn(2, 4096, generator=generator, dtype=torch.float64), dim=-1
        ).cuda()
        for _ in range(2)
    ]
    step_positions = [
        torch.tensor([5, 9], device="cuda"),
        torch.tensor([6, 10], device="cuda"),
    ]

    def sample(
        logits_output: types.SimpleNamespace, forward_batch: types.SimpleNamespace
    ) -> torch.Tensor:
        return multinomial_with_seed(
            logits_output.next_token_logits,
            forward_batch.sampling_info.sampling_seed,
            forward_batch.positions,
        ).view(-1)

    thinker_runner = runner()
    thinker_runner.tp_worker = types.SimpleNamespace(
        model_runner=types.SimpleNamespace(sample=sample)
    )
    thinker_runner.sampling_seed_batch_key = ()
    thinker_runner.sampling_seed_batch_tensor = None
    thinker_runner.th_host_bufs = None
    thinker_runner.th_slot = 0
    pending_steps = []
    for logprobs, positions in zip(step_logprobs, step_positions):
        forward_batch = types.SimpleNamespace(
            positions=positions, sampling_info=seeded_sampling_info("cuda")
        )
        result = types.SimpleNamespace(
            next_token_ids=None,
            logits_output=types.SimpleNamespace(next_token_logits=logprobs),
        )
        launch_buffer = thinker_runner.post_decode_launch(
            result, forward_batch, requests
        )
        pending_steps.append((launch_buffer, result))
    torch.cuda.synchronize()
    for launch_buffer, result in pending_steps:
        thinker_runner.post_decode_resolve(launch_buffer, result, None, None, requests)

    row_seeds = torch.tensor(
        [42, rank_shared_unseeded_sampling_seed(requests[1], 1)], device="cuda"
    )
    synchronous_tokens = [
        multinomial_with_seed(logprobs, row_seeds, positions).view(-1).tolist()
        for logprobs, positions in zip(step_logprobs, step_positions)
    ]
    assert [
        result.next_token_ids.tolist() for _, result in pending_steps
    ] == synchronous_tokens
