# SPDX-License-Identifier: Apache-2.0
import pytest

from sglang_omni.models.cosmos3.stages import native_server_kwargs


@pytest.mark.parametrize("timeout", [1, 600])
def test_finite_rpc_deadline_is_rejected_before_native_startup(timeout):
    with pytest.raises(ValueError, match="before work settles"):
        native_server_kwargs("checkpoint", 0, {"scheduler_rpc_timeout": timeout})
