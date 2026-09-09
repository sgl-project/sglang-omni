# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import pytest
from sglang.srt.sampling.sampling_params import (
    MAX_STOP_COUNT,
    MAX_STOP_REGEX_COUNT,
    MAX_STOP_REGEX_LEN,
    SamplingParams,
)

from sglang_omni.serve.openai_errors import is_bad_request_error


def _normalize_error(**kwargs) -> ValueError:
    with pytest.raises(ValueError) as raised:
        SamplingParams(**kwargs).normalize(None)
    return raised.value


def test_too_many_stop_strings_is_a_bad_request() -> None:
    error = _normalize_error(stop=["."] * (MAX_STOP_COUNT + 1))

    assert is_bad_request_error(error)


def test_too_many_stop_regexes_is_a_bad_request() -> None:
    error = _normalize_error(stop_regex=[r"\."] * (MAX_STOP_REGEX_COUNT + 1))

    assert is_bad_request_error(error)


def test_an_oversized_stop_regex_is_a_bad_request() -> None:
    error = _normalize_error(stop_regex=["a" * (MAX_STOP_REGEX_LEN + 1)])

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


def test_an_unrelated_failure_stays_internal() -> None:
    for message in (
        "CUDA out of memory",
        "AuK generated latent contains NaN/Inf",
        "internal cache size is a server-level setting",
    ):
        assert not is_bad_request_error(RuntimeError(message))
