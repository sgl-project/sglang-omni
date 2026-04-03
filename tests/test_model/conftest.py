# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures and hooks for test_model tests."""

from __future__ import annotations

import pytest

S2PRO_TTS_ALLOWED_CONCURRENCIES = (1, 2, 4, 8, 16)
S2PRO_TTS_CONCURRENCY_OPTION = "--concurrency"
S2PRO_TTS_FULL_SWEEP_VALUE = "all"
SELECTED_S2PRO_TTS_CONCURRENCIES = pytest.StashKey[tuple[int, ...]]()


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        S2PRO_TTS_CONCURRENCY_OPTION,
        action="store",
        default="1",
        help=(
            "Select the S2-Pro TTS benchmark concurrency. "
            "Use one of {1,2,4,8,16} or 'all' for the full sweep."
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    option_value = config.getoption(S2PRO_TTS_CONCURRENCY_OPTION)
    config.stash[SELECTED_S2PRO_TTS_CONCURRENCIES] = _parse_s2pro_tts_concurrency(
        option_value
    )


@pytest.fixture(scope="session")
def selected_s2pro_tts_concurrencies(
    pytestconfig: pytest.Config,
) -> tuple[int, ...]:
    return pytestconfig.stash[SELECTED_S2PRO_TTS_CONCURRENCIES]


def _parse_s2pro_tts_concurrency(option_value: str) -> tuple[int, ...]:
    normalized_value = option_value.strip().lower()
    if normalized_value == S2PRO_TTS_FULL_SWEEP_VALUE:
        return S2PRO_TTS_ALLOWED_CONCURRENCIES

    try:
        concurrency = int(normalized_value)
    except ValueError as exc:
        raise pytest.UsageError(
            "Invalid value for --concurrency. " "Use one of {1,2,4,8,16} or 'all'."
        ) from exc

    if concurrency not in S2PRO_TTS_ALLOWED_CONCURRENCIES:
        raise pytest.UsageError(
            f"Unsupported concurrency {concurrency}. "
            f"Use one of {S2PRO_TTS_ALLOWED_CONCURRENCIES} or 'all'."
        )
    return (concurrency,)
