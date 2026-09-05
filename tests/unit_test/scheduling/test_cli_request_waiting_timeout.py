# SPDX-License-Identifier: Apache-2.0
"""CLI override tests for the waiting-queue timeout.

``--req-waiting-timeout`` bounds how long one request may wait, where
``--max-queued-requests`` bounds how many may wait. Upstream reads the bound
from the environment inside ``Scheduler._abort_on_waiting_timeout``, so the
flag has to publish it there rather than onto ``server_args``.
"""

from __future__ import annotations

import os

from sglang_omni.cli.serve import apply_request_waiting_timeout


def test_the_timeout_reaches_the_environment(monkeypatch) -> None:
    monkeypatch.delenv("SGLANG_REQ_WAITING_TIMEOUT", raising=False)

    apply_request_waiting_timeout(30.0)

    assert os.environ["SGLANG_REQ_WAITING_TIMEOUT"] == "30.0"


def test_omitting_the_flag_leaves_an_exported_value_alone(monkeypatch) -> None:
    monkeypatch.setenv("SGLANG_REQ_WAITING_TIMEOUT", "12")

    apply_request_waiting_timeout(None)

    assert os.environ["SGLANG_REQ_WAITING_TIMEOUT"] == "12"


def test_upstream_parses_back_what_the_flag_wrote(monkeypatch) -> None:
    """The scheduler calls EnvFloat.get(); a value it cannot parse raises there."""
    from sglang.srt.environ import envs

    monkeypatch.delenv("SGLANG_REQ_WAITING_TIMEOUT", raising=False)

    apply_request_waiting_timeout(2.5)

    assert envs.SGLANG_REQ_WAITING_TIMEOUT.get() == 2.5


def test_the_default_leaves_the_bound_off(monkeypatch) -> None:
    """A negative default is how upstream spells 'no timeout'."""
    from sglang.srt.environ import envs

    monkeypatch.delenv("SGLANG_REQ_WAITING_TIMEOUT", raising=False)

    apply_request_waiting_timeout(None)

    assert envs.SGLANG_REQ_WAITING_TIMEOUT.get() <= 0
