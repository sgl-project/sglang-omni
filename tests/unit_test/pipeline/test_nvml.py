# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from sglang_omni.utils.nvml import nvml_session


def test_failed_nested_initialization_does_not_release_outer_session():
    references = 0

    def init():
        nonlocal references
        if references:
            raise RuntimeError("initialization failed")
        references += 1

    def shutdown():
        nonlocal references
        references -= 1

    pynvml = SimpleNamespace(nvmlInit=init, nvmlShutdown=shutdown)
    with nvml_session(pynvml):
        with pytest.raises(RuntimeError, match="initialization failed"):
            with nvml_session(pynvml):
                pytest.fail("failed initialization must not enter the session")
        assert references == 1
    assert references == 0


def test_shutdown_failure_preserves_query_error():
    def shutdown():
        raise RuntimeError("shutdown failed")

    pynvml = SimpleNamespace(nvmlInit=lambda: None, nvmlShutdown=shutdown)
    with pytest.raises(ValueError, match="query failed"):
        with nvml_session(pynvml):
            raise ValueError("query failed")
