# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from sglang.srt.sampling.sampling_params import (
    MAX_STOP_COUNT,
    MAX_STOP_REGEX_COUNT,
    MAX_STOP_REGEX_LEN,
    SamplingParams,
)

from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.proto import SubmitMessage
from sglang_omni.serve.openai_errors import is_bad_request_error
from tests.unit_test.fixtures.pipeline_fakes import (
    RecordingCoordinatorControlPlane,
    make_stage_payload,
)
from tests.unit_test.pipeline.helpers import make_stage


def normalize_error(**kwargs) -> ValueError:
    with pytest.raises(ValueError) as raised:
        SamplingParams(**kwargs).normalize(None)
    return raised.value


def test_too_many_stop_strings_is_a_bad_request() -> None:
    error = normalize_error(stop=["."] * (MAX_STOP_COUNT + 1))

    assert is_bad_request_error(error)


def test_too_many_stop_regexes_is_a_bad_request() -> None:
    error = normalize_error(stop_regex=[r"\."] * (MAX_STOP_REGEX_COUNT + 1))

    assert is_bad_request_error(error)


def test_an_oversized_stop_regex_is_a_bad_request() -> None:
    error = normalize_error(stop_regex=["a" * (MAX_STOP_REGEX_LEN + 1)])

    assert is_bad_request_error(error)


def test_the_stop_bounds_themselves_normalize() -> None:
    params = SamplingParams(
        stop=["."] * MAX_STOP_COUNT,
        stop_regex=["a" * MAX_STOP_REGEX_LEN] * MAX_STOP_REGEX_COUNT,
    )

    tokenizer = SimpleNamespace(encode=lambda text, **_: list(text.encode()))
    params.normalize(tokenizer)

    assert len(params.stop_strs) == MAX_STOP_COUNT
    assert len(params.stop_regex_strs) == MAX_STOP_REGEX_COUNT


def test_request_id_conflicts_are_bad_requests() -> None:
    async def conflict_messages() -> list[str]:
        coordinator = Coordinator(
            "inproc://complete", "inproc://abort", entry_stage="preprocess"
        )
        coordinator.control_plane = RecordingCoordinatorControlPlane()
        coordinator.register_stage("preprocess", "inproc://preprocess")
        await coordinator.submit_request("req-live", "hello")
        with pytest.raises(ValueError) as duplicate:
            await coordinator.submit_request("req-live", "hello")

        stage = make_stage(name="preprocess")
        stage.record_aborted_request_id("req-retired")
        await stage.on_submit(
            SubmitMessage(
                request_id="req-retired",
                data=make_stage_payload(request_id="req-retired"),
            )
        )
        (retired,) = stage.control_plane.completions
        return [str(duplicate.value), retired.error]

    for message in asyncio.run(conflict_messages()):
        assert is_bad_request_error(RuntimeError(message))


def test_an_unrelated_failure_stays_internal() -> None:
    for message in (
        "CUDA out of memory",
        "AuK generated latent contains NaN/Inf",
        "internal cache size is a server-level setting",
        "index out of range in self",
    ):
        assert not is_bad_request_error(RuntimeError(message))
