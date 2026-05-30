# SPDX-License-Identifier: Apache-2.0
"""Pytest collection defaults for local/sandboxed test runs."""

from __future__ import annotations

import os
import sys
import types

import pytest

# FlashInfer initializes a file logger at import time through SGLang. In the
# Codex sandbox, /root/.cache is read-only, so collection fails before tests can
# monkeypatch anything. Keep its import-time workspace in writable /tmp.
os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", "/tmp")


def _install_vllm_layer_stubs() -> None:
    """Provide narrow CPU fallback classes for SGLang collection.

    Upstream SGLang imports these vLLM layer classes only when running on a CPU
    platform without sgl-kernel AMX support. The encoder unit tests do not
    exercise vLLM itself, but collection still needs the fallback symbols.
    """
    try:
        import vllm  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class RMSNorm(nn.Module):
        def __init__(self, hidden_size: int, eps: float = 1e-6, **kwargs) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.variance_epsilon = eps
            self.cast_x_before_out_mul = False

        def forward_native(
            self,
            x: torch.Tensor,
            residual: torch.Tensor | None = None,
            post_residual_addition: torch.Tensor | None = None,
            **kwargs,
        ):
            y = x
            if residual is not None:
                y = y + residual
            if post_residual_addition is not None:
                y = y + post_residual_addition
            out = y.float()
            out = out * torch.rsqrt(
                out.pow(2).mean(dim=-1, keepdim=True) + self.variance_epsilon
            )
            out = out.to(dtype=x.dtype) * self.weight
            return out if residual is None else (out, y)

        def forward_cuda(
            self,
            x: torch.Tensor,
            residual: torch.Tensor | None = None,
            post_residual_addition: torch.Tensor | None = None,
            **kwargs,
        ):
            return self.forward_native(
                x,
                residual,
                post_residual_addition=post_residual_addition,
                **kwargs,
            )

        def forward_with_allreduce_fusion(
            self,
            x: torch.Tensor,
            residual: torch.Tensor | None = None,
            post_residual_addition: torch.Tensor | None = None,
            **kwargs,
        ):
            return self.forward_native(
                x,
                residual,
                post_residual_addition=post_residual_addition,
                **kwargs,
            )

        def forward(
            self,
            x: torch.Tensor,
            residual: torch.Tensor | None = None,
            post_residual_addition: torch.Tensor | None = None,
            **kwargs,
        ):
            return self.forward_native(
                x,
                residual,
                post_residual_addition=post_residual_addition,
                **kwargs,
            )

    class GemmaRMSNorm(nn.Module):
        def __init__(self, hidden_size: int, eps: float = 1e-6, **kwargs) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
            out = x.float()
            out = out * torch.rsqrt(
                out.pow(2).mean(dim=-1, keepdim=True) + self.variance_epsilon
            )
            return (out * (1.0 + self.weight.float())).to(dtype=x.dtype)

    class SiluAndMul(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            d = x.shape[-1] // 2
            return F.silu(x[..., :d]) * x[..., d:]

    class GeluAndMul(nn.Module):
        def __init__(self, approximate: str = "tanh") -> None:
            super().__init__()
            self.approximate = approximate

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            d = x.shape[-1] // 2
            return F.gelu(x[..., :d], approximate=self.approximate) * x[..., d:]

    vllm_mod = types.ModuleType("vllm")
    model_executor_mod = types.ModuleType("vllm.model_executor")
    layers_mod = types.ModuleType("vllm.model_executor.layers")
    layernorm_mod = types.ModuleType("vllm.model_executor.layers.layernorm")
    activation_mod = types.ModuleType("vllm.model_executor.layers.activation")

    vllm_mod.__path__ = []
    model_executor_mod.__path__ = []
    layers_mod.__path__ = []

    layernorm_mod.RMSNorm = RMSNorm
    layernorm_mod.GemmaRMSNorm = GemmaRMSNorm
    activation_mod.SiluAndMul = SiluAndMul
    activation_mod.GeluAndMul = GeluAndMul

    vllm_mod.model_executor = model_executor_mod
    model_executor_mod.layers = layers_mod
    layers_mod.layernorm = layernorm_mod
    layers_mod.activation = activation_mod

    sys.modules.setdefault("vllm", vllm_mod)
    sys.modules.setdefault("vllm.model_executor", model_executor_mod)
    sys.modules.setdefault("vllm.model_executor.layers", layers_mod)
    sys.modules.setdefault("vllm.model_executor.layers.layernorm", layernorm_mod)
    sys.modules.setdefault("vllm.model_executor.layers.activation", activation_mod)


_install_vllm_layer_stubs()


@pytest.fixture(autouse=True)
def _deterministic_nccl_ports(monkeypatch):
    """Avoid socket probing in CPU-only unit tests.

    The sandbox can block during ``socket.socket`` construction. Unit tests only
    need stable, non-null ports to verify propagation through process specs.
    """
    try:
        from sglang_omni.pipeline import mp_runner
    except Exception:
        return

    next_port = 29500

    def _allocate(self):
        nonlocal next_port
        port = next_port
        next_port += 1
        return port

    monkeypatch.setattr(mp_runner._NcclPortAllocator, "allocate", _allocate)
