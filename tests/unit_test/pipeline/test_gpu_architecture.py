# SPDX-License-Identifier: Apache-2.0

import pytest

from sglang_omni.utils.gpu_compat import gpu_architecture_for_sm


@pytest.mark.parametrize(
    ("sm_version", "architecture"),
    [
        (89, "ada"),
        (90, "hopper"),
        (100, "blackwell"),
        (103, "blackwell"),
        (120, "blackwell"),
        (None, "unknown"),
    ],
)
def test_gpu_architecture_identity_is_explicit(
    sm_version: int | None,
    architecture: str,
) -> None:
    assert gpu_architecture_for_sm(sm_version) == architecture
