# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

from sglang.srt.model_executor.cuda_graph_config import Backend


def cuda_graph_server_args(**selectors: object) -> SimpleNamespace:
    return SimpleNamespace(
        **{
            "disable_cuda_graph": False,
            "disable_decode_cuda_graph": False,
            "disable_prefill_cuda_graph": False,
            "disable_cuda_graph_padding": False,
            "cuda_graph_backend_decode": Backend.FULL,
            "cuda_graph_backend_prefill": Backend.BREAKABLE,
            "cuda_graph_config": None,
            **selectors,
        }
    )
