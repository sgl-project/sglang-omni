# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ThinkerModelRunner.lookahead_eligible.

lookahead_eligible reads only per-request flags (never other instance state), so it
is exercised on a bare instance built with ``object.__new__`` and stand-in requests.
"""

from __future__ import annotations

import types

from sglang_omni.model_runner.thinker_model_runner import ThinkerModelRunner


def _runner() -> ThinkerModelRunner:
    return object.__new__(ThinkerModelRunner)


def _sp(**kw):
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


def _req(return_logprob=False, stage_payload="text", **sp_kw):
    return types.SimpleNamespace(
        sampling_params=_sp(**sp_kw),
        _omni_data=types.SimpleNamespace(
            return_logprob=return_logprob, stage_payload=stage_payload
        ),
    )


def _batch(*reqs):
    return types.SimpleNamespace(reqs=list(reqs))


def test_plain_greedy_is_eligible():
    assert _runner().lookahead_eligible(_batch(_req(), _req())) is True


def test_empty_batch_is_eligible():
    assert _runner().lookahead_eligible(_batch()) is True


def test_audio_output_is_eligible_for_lookahead():
    assert _runner().lookahead_eligible(_batch(_req(stage_payload="audio"))) is True


def test_return_logprob_disables_lookahead():
    assert _runner().lookahead_eligible(_batch(_req(return_logprob=True))) is False


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
        assert _runner().lookahead_eligible(_batch(_req(**kw))) is False, kw


def test_one_gated_request_disables_whole_batch():
    audio_mix = _batch(_req(), _req(stage_payload="audio"), _req())
    assert _runner().lookahead_eligible(audio_mix) is True
    param_mix = _batch(_req(), _req(repetition_penalty=1.3), _req())
    assert _runner().lookahead_eligible(param_mix) is False
