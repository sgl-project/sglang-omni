# SPDX-License-Identifier: Apache-2.0

import pytest

from sglang_omni.scheduling.prefill_coalesce import (
    validate_prefill_coalesce_requests,
    validate_prefill_coalesce_wait_ms,
)


def test_prefill_coalesce_validators_normalize_values():
    assert validate_prefill_coalesce_requests(32) == 32
    assert validate_prefill_coalesce_requests(32.0) == 32
    assert validate_prefill_coalesce_wait_ms(300) == 300.0


@pytest.mark.parametrize(
    "requests",
    [
        None,
        -1,
        True,
        False,
        -0.5,
        2.9,
        # int(inf) raises OverflowError, which is not a ValueError, so callers
        # wrapping this in typer.BadParameter would leak a traceback instead
        float("inf"),
        float("-inf"),
        "32",
    ],
)
def test_prefill_coalesce_requests_rejects_invalid_values(requests):
    with pytest.raises(ValueError):
        validate_prefill_coalesce_requests(requests)


@pytest.mark.parametrize(
    "wait_ms",
    [None, 0.0, float("nan"), float("inf"), True, "300"],
)
def test_prefill_coalesce_wait_rejects_invalid_values(wait_ms):
    with pytest.raises(ValueError):
        validate_prefill_coalesce_wait_ms(wait_ms)


def test_prefill_coalesce_requests_warns_on_one(caplog):
    # 1 passes validation but the gate only engages at >= 2, so it must warn
    # the user that coalescing stays disabled; 0 stays silent (explicit off).
    with caplog.at_level("WARNING", logger="sglang_omni.scheduling.prefill_coalesce"):
        assert validate_prefill_coalesce_requests(1) == 1
    assert any("disables coalescing" in r.message for r in caplog.records)

    caplog.clear()
    with caplog.at_level("WARNING", logger="sglang_omni.scheduling.prefill_coalesce"):
        assert validate_prefill_coalesce_requests(0) == 0
    assert not caplog.records
