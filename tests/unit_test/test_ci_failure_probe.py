# SPDX-License-Identifier: Apache-2.0
"""Temporary probe for PR Test failure handling; reverted before merge."""


def test_deliberate_failure_does_not_stop_the_job() -> None:
    assert False, "deliberate failure: the rest of this partition must still run"
