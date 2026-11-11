# SPDX-License-Identifier: Apache-2.0
"""Compiled DiT execution and optional optimized projection resources."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar

import torch
from torch._dynamo.backends.registry import CompiledFn
from torch.utils._python_dispatch import TorchDispatchMode

from sglang_omni.models.firered_tts3.dit import DiT
from sglang_omni.models.firered_tts3.ffn_gemm import FFN_CONFIGS as FFN_CONFIGS
from sglang_omni.models.firered_tts3.ffn_gemm import FFN_LAYERS as FFN_LAYERS
from sglang_omni.models.firered_tts3.ffn_gemm import QKV_CONFIGS as QKV_CONFIGS
from sglang_omni.models.firered_tts3.ffn_gemm import CompileBackend as CompileBackend
from sglang_omni.models.firered_tts3.ffn_gemm import FfnGemmMode as FfnGemmMode
from sglang_omni.models.firered_tts3.ffn_gemm import GemmOutput as GemmOutput
from sglang_omni.models.firered_tts3.ffn_gemm import gemm_out as gemm_out

__all__ = [
    "CompileBackend",
    "CompiledDiTExecutor",
    "DIT_RECOMPILE_LIMIT",
    "FFN_CONFIGS",
    "FFN_LAYERS",
    "FfnGemmMode",
    "GemmOutput",
    "QKV_CONFIGS",
    "gemm_out",
]

# note (Codex): each bucket uses separate Dynamo entries for both backends.
DIT_RECOMPILE_LIMIT = 64


class CompiledDiTExecutor:
    """Keep dispatch modes outside Dynamo tracing and the original graph intact."""

    def __init__(self, dit: DiT, original: torch.nn.Module | None = None) -> None:
        self.source = dit
        self.active_modes: ContextVar[tuple[TorchDispatchMode, ...]] = ContextVar(
            "firered_dit_dispatch_modes", default=()
        )
        self.original_compiled = (
            torch.compile(dit, dynamic=False, options={"epilogue_fusion": False})
            if original is None
            else original
        )
        self.observed = torch.compile(dit, backend=self.compile_backend, dynamic=False)

    def compile_backend(
        self, graph_module: torch.fx.GraphModule, example_inputs: list[torch.Tensor]
    ) -> CompiledFn:
        compiled = torch._inductor.compile(  # noqa: SLF001  # leading-underscore: PyTorch compiler API
            graph_module, example_inputs, options={"epilogue_fusion": False}
        )

        # note (kermit): TorchDynamo passes the graph's flattened inputs positionally.
        def execute(*graph_inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
            with ExitStack() as stack:
                for mode in self.active_modes.get():
                    stack.enter_context(mode)
                return compiled(*graph_inputs)

        return execute

    @contextmanager
    def dispatch(self, modes: tuple[TorchDispatchMode, ...]) -> Iterator[None]:
        token = self.active_modes.set(modes)
        try:
            yield
        finally:
            self.active_modes.reset(token)

    def original(self, *, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        with torch._dynamo.config.patch(  # noqa: SLF001  # leading-underscore: PyTorch compiler API
            recompile_limit=max(
                DIT_RECOMPILE_LIMIT,
                torch._dynamo.config.recompile_limit,  # noqa: SLF001  # leading-underscore: PyTorch compiler API
            ),
            fail_on_recompile_limit_hit=True,
            suppress_errors=False,
        ):
            return self.original_compiled(x=x, t=t)

    def run(self, *, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        packed_mode = next(
            (
                mode
                for mode in self.active_modes.get()
                if isinstance(mode, FfnGemmMode)
                and x.shape[0] * x.shape[1] in QKV_CONFIGS
            ),
            None,
        )
        if packed_mode is None:
            observed = self.observed
        else:
            assert packed_mode.packed_compiled is not None
            observed = packed_mode.packed_compiled
        with torch._dynamo.config.patch(  # noqa: SLF001  # leading-underscore: PyTorch compiler API
            recompile_limit=max(
                DIT_RECOMPILE_LIMIT,
                torch._dynamo.config.recompile_limit,  # noqa: SLF001  # leading-underscore: PyTorch compiler API
            ),
            fail_on_recompile_limit_hit=True,
            suppress_errors=False,
        ):
            return observed(x=x, t=t)

    def __call__(self, *, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.original(x=x, t=t)
