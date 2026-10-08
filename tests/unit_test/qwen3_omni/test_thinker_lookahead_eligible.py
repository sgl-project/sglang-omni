# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ThinkerModelRunner.lookahead_eligible.

lookahead_eligible reads only per-request flags (never other instance state), so it
is exercised on a bare instance built with ``object.__new__`` and stand-in requests.
Audio-output detection (should_generate_audio_output) is stubbed on the stand-in
stage_payload.
"""
from __future__ import annotations

import types

import pytest

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


def test_empty_batch_is_eligible():
    assert runner().lookahead_eligible(batch()) is True


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
        dict(sampling_seed=42),
        dict(logit_bias={1: 2.0}),
        dict(custom_params={"x": 1}),
    ):
        assert runner().lookahead_eligible(batch(req(**kw))) is False, kw


def test_one_gated_request_disables_whole_batch():
    audio_mix = batch(req(), req(stage_payload="audio"), req())
    assert runner().lookahead_eligible(audio_mix) is False
    param_mix = batch(req(), req(repetition_penalty=1.3), req())
    assert runner().lookahead_eligible(param_mix) is False
