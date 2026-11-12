# SPDX-License-Identifier: Apache-2.0
"""Optional optimized feed-forward and packed QKV graph resources."""

from __future__ import annotations

from typing import Protocol

import torch
from torch._dynamo.backends.registry import CompiledFn
from torch._ops import OpOverload
from torch.utils._python_dispatch import TorchDispatchMode

from sglang_omni.models.firered_tts3.modules import Attention
from sglang_omni.models.firered_tts3.weights import FfnWeight, PackedQkvModel

try:
    from sglang_omni.models.firered_tts3.ffn_gemm_kernel import gemm_out
except ImportError:
    gemm_out = None

# note (Codex): measured SM120 tiles; small FFN1 and M192 QKV use K128 rounding.
FFN_CONFIGS = {
    24: (16, 64, 64, 4, 3),
    48: (32, 64, 64, 4, 3),
    96: (32, 128, 64, 4, 3),
}
QKV_CONFIGS = {192: (64, 128, 64, 4, 3), 384: (64, 64, 64, 4, 3)}
FFN_LAYERS = 11


class CompileBackend(Protocol):
    def __call__(
        self, graph_module: torch.fx.GraphModule, example_inputs: list[torch.Tensor]
    ) -> CompiledFn: ...


class GemmOutput(Protocol):
    def __call__(
        self,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        out: torch.Tensor,
        config: tuple[int, int, int, int, int],
    ) -> torch.Tensor: ...


class FfnGemmMode(TorchDispatchMode):
    """Own FFN1 and packed-QKV graph resources; rebuild after any weight update."""

    def __init__(self, source: torch.nn.Module) -> None:
        super().__init__()
        self.source: torch.nn.Module = source
        self.weights: dict[int, FfnWeight] = {}
        self.packed_weights: dict[int, FfnWeight] = {}
        self.packed_qkv: PackedQkvModel | None = None
        self.packed_compiled: torch.nn.Module | None = None
        self.hits: int = 0
        self.misses: int = 0
        self.enabled: bool = False
        self.reason: str = (
            "DiT GEMM optimization requires eleven CUDA BF16 Linear layers"
        )
        self.gemm_out: GemmOutput | None = None
        layers = [
            (name, layer)
            for name, layer in source.named_modules()
            if isinstance(layer, torch.nn.Linear) and name.endswith(".mlp.ff.0.0")
        ]
        if len(layers) != FFN_LAYERS or any(
            layer.weight.shape != (3072, 1024)
            or layer.bias is None
            or layer.bias.shape != (3072,)
            or not layer.weight.is_contiguous()
            or not layer.bias.is_contiguous()
            or not layer.weight.is_cuda
            or layer.weight.dtype != torch.bfloat16
            or layer.bias.dtype != torch.bfloat16
            or layer.weight.device != layer.bias.device
            for _, layer in layers
        ):
            return
        else:
            device = layers[0][1].weight.device
            if any(layer.weight.device != device for _, layer in layers):
                return
            elif torch.cuda.get_device_capability(device) != (12, 0):
                self.reason = "DiT GEMM optimization requires SM120"
                return
            elif not torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction:
                self.reason = (
                    "DiT GEMM optimization requires BF16 reduced precision reduction"
                )
                return
            else:
                if gemm_out is None:
                    raise RuntimeError("DiT GEMM optimization requires Triton")
                else:
                    pass
                self.gemm_out = gemm_out
                self.weights = {
                    layer.weight.data_ptr(): FfnWeight.from_linear(name, layer)
                    for name, layer in layers
                }
                if len(self.weights) != FFN_LAYERS:
                    self.reason = (
                        "DiT GEMM optimization requires independent Linear weights"
                    )
                    return
                else:
                    self.enabled = True
                    self.reason = ""

    def prepare_packed(
        self, compile_backend: CompileBackend, device: torch.device
    ) -> None:
        """Prepare large-bucket QKV before capture without changing the small FFN path."""
        if self.packed_compiled is not None:
            return
        elif not self.enabled:
            raise RuntimeError(self.reason)
        else:
            with torch.cuda.device(device):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("Packed QKV must be prepared before capture")
                else:
                    pass
            self.validate_weights()
            attentions = [
                (name, layer)
                for name, layer in self.source.named_modules()
                if name.endswith(".attn")
            ]
            if len(attentions) != FFN_LAYERS or any(
                type(layer) is not Attention for _, layer in attentions
            ):
                raise RuntimeError(
                    "Packed QKV requires eleven original DiT Attention layers"
                )
            else:
                pass
            for name, attention in attentions:
                for role in ("to_q", "to_k", "to_v"):
                    layer = attention.get_submodule(role)
                    if (
                        layer.weight.shape != (1024, 1024)
                        or layer.bias is None
                        or layer.bias.shape != (1024,)
                        or any(
                            tensor.dtype != torch.bfloat16
                            or tensor.device != device
                            or not tensor.is_contiguous()
                            for tensor in (layer.weight, layer.bias)
                        )
                    ):
                        raise RuntimeError(
                            f"Unsupported packed QKV parameters: {name}.{role}"
                        )
                    else:
                        pass
            packed_qkv = PackedQkvModel(self.source)
            packed_weights = {
                packed.to_qkv.weight.data_ptr(): FfnWeight.from_linear(
                    f"{name}.to_qkv", packed.to_qkv
                )
                for name, packed in packed_qkv.layers.items()
            }
            packed_compiled = torch.compile(
                packed_qkv.model, backend=compile_backend, dynamic=False
            )
            self.packed_qkv = packed_qkv
            self.packed_weights = packed_weights
            self.packed_compiled = packed_compiled

    def supports(self, rows: int) -> bool:
        return self.enabled and (
            rows in FFN_CONFIGS
            or (rows in QKV_CONFIGS and self.packed_compiled is not None)
        )

    def expected_hits(self, rows: int) -> int:
        if rows in FFN_CONFIGS:
            return len(self.weights)
        elif rows in QKV_CONFIGS:
            return len(self.packed_weights)
        else:
            return 0

    def record_stream(self, stream: torch.cuda.Stream) -> None:
        """Called once per graph replay stream by the owning graph entry."""
        for entry in self.weights.values():
            entry.source.record_stream(stream)
            entry.bias.record_stream(stream)
        if self.packed_qkv is not None:
            self.packed_qkv.record_stream(stream)
        else:
            pass

    def validate_weights(self) -> None:
        """Validate cached parameters before capturing a new graph."""
        if self.packed_qkv is not None:
            self.packed_qkv.validate_weights()
        else:
            pass
        for pointer, entry in self.weights.items():
            current = self.source.get_parameter(entry.name)
            bias = self.source.get_parameter(entry.name.removesuffix("weight") + "bias")
            version = (
                None if torch.is_inference(current) else current._version
            )  # noqa: SLF001  # leading-underscore: PyTorch tensor version API
            bias_version = (
                None if torch.is_inference(bias) else bias._version
            )  # noqa: SLF001  # leading-underscore: PyTorch tensor version API
            if (
                current is not entry.source
                or bias is not entry.bias
                or current.data_ptr() != pointer
                or current.shape != (entry.output_size, 1024)
                or current.stride() != (1024, 1)
                or bias.data_ptr() != entry.bias_pointer
                or bias.shape != (entry.output_size,)
                or bias.stride() != (1,)
                or current.dtype != torch.bfloat16
                or bias.dtype != torch.bfloat16
                or current.device != entry.device
                or bias.device != entry.device
                or version != entry.version
                or bias_version != entry.bias_version
            ):
                raise RuntimeError(
                    f"DiT GEMM weights changed; rebuild mode: {entry.name}"
                )
            else:
                pass

    def __torch_dispatch__(
        self,
        func: OpOverload,
        types: tuple[type[torch.Tensor], ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        """Preserve PyTorch's open operator argument and return contract."""
        kwargs = kwargs or {}
        if not self.enabled or func is not torch.ops.aten.addmm.out:
            return func(*args, **kwargs)
        else:
            parameter_names = ("self", "mat1", "mat2")
            provided_names = set(parameter_names[: len(args)]) | set(kwargs)
            provided_names.update(("alpha", "beta"))
            weight = kwargs.get("mat2", args[2] if len(args) > 2 else None)
            weight_address = (
                weight.data_ptr() if isinstance(weight, torch.Tensor) else None
            )
            entry = self.weights.get(weight_address)
            if entry is None:
                entry = self.packed_weights.get(weight_address)
                configs = QKV_CONFIGS
            else:
                configs = FFN_CONFIGS
            if entry is None:
                return func(*args, **kwargs)
            elif len(args) > 3 or provided_names != {
                "self",
                "mat1",
                "mat2",
                "out",
                "alpha",
                "beta",
            }:
                self.misses += 1
                return func(*args, **kwargs)
            else:
                inputs = kwargs.get("mat1", args[1] if len(args) > 1 else None)
                bias = kwargs.get("self", args[0] if args else None)
                out = kwargs.get("out")
                if (
                    not isinstance(inputs, torch.Tensor)
                    or not isinstance(bias, torch.Tensor)
                    or not isinstance(out, torch.Tensor)
                    or not isinstance(weight, torch.Tensor)
                ):
                    self.misses += 1
                    return func(*args, **kwargs)
                else:
                    if (
                        kwargs.get("alpha", 1) != 1
                        or kwargs.get("beta", 1) != 1
                        or inputs.ndim != 2
                        or inputs.shape[0] not in configs
                        or inputs.shape[1] != 1024
                        or not inputs.is_contiguous()
                        or weight.shape != (1024, 3072)
                        or weight.stride() != (1, 1024)
                        or bias.shape != (3072,)
                        or bias.stride() != (1,)
                        or bias.data_ptr() != entry.bias_pointer
                        or out.shape != (inputs.shape[0], 3072)
                        or not out.is_contiguous()
                        or any(
                            tensor.dtype != torch.bfloat16
                            or tensor.device != entry.device
                            or tensor.is_neg()
                            or tensor.is_conj()
                            for tensor in (inputs, weight, bias, out)
                        )
                        or (
                            torch.is_grad_enabled()
                            and any(
                                tensor.requires_grad
                                for tensor in (inputs, weight, bias, out)
                            )
                        )
                    ):
                        self.misses += 1
                        return func(*args, **kwargs)
                    else:
                        output_start_address = out.data_ptr()
                        output_end_address = output_start_address + out.numel() * 2
                        if any(
                            max(output_start_address, tensor.data_ptr())
                            < min(
                                output_end_address,
                                tensor.data_ptr() + tensor.numel() * 2,
                            )
                            for tensor in (inputs, weight, bias)
                        ):
                            self.misses += 1
                            return func(*args, **kwargs)
                        else:
                            assert self.gemm_out is not None
                            result = self.gemm_out(
                                inputs, weight, bias, out, configs[inputs.shape[0]]
                            )
                            self.hits += 1
                            return result
