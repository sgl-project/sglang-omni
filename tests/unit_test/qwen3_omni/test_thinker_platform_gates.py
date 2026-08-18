# SPDX-License-Identifier: Apache-2.0
"""The thinker's CUDA-only gates must key off the resolved platform.

sglang's is_cuda() reports only torch's CUDA build and availability, so on a host
carrying both CUDA and XPU it stays true for an XPU run. Gating on it would enter
CUDA-only branches -- a torch.cuda alt_stream that apply_qk_norm() feeds to
torch.cuda.stream(), and the fused set_kv_buffer arg -- with the accelerator set to
XPU. Exercising those branches needs a fully initialized distributed thinker, so
this guards the gate itself: the module must consult current_platform and must not
snapshot torch's view at import time.
"""

from __future__ import annotations

import inspect

from sglang_omni.models.qwen3_omni.components import talker, thinker_model


def test_thinker_gates_read_the_resolved_platform_not_torch_cuda() -> None:
    source = inspect.getsource(thinker_model)

    assert not hasattr(thinker_model, "_is_cuda")
    assert "from sglang.srt.utils import is_cuda" not in source
    assert "current_platform.is_cuda()" in source


def test_talker_alt_streams_are_gated_on_the_resolved_platform() -> None:
    """Both talker constructors feed alt_stream to the same inherited apply_qk_norm
    path, so an accelerator stream off CUDA would reach torch.cuda.stream() there.
    get_device_module().Stream() would hand it an XPU stream on an XPU run.
    """
    source = inspect.getsource(talker)

    assert "torch.get_device_module().Stream()" not in source
    assert source.count("torch.cuda.Stream() if current_platform.is_cuda()") == 2
