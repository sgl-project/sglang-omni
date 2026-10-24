# SPDX-License-Identifier: Apache-2.0
"""Convolution weight caching and controlled cuDNN execution for compiled DiT."""

from __future__ import annotations

import weakref
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import torch
from torch._ops import OpOverload
from torch.utils._python_dispatch import TorchDispatchMode

try:
    import cudnn
except ImportError:
    cudnn = None

# note (Codex): verified against the original NCHW plan on SM120 with cuDNN 9.20.
CUDNN_VERSION = 92000
FRONTEND_VERSION = "1.28.0"
ENGINE_ID = 23
TENSOR_ALIGNMENT = 16  # note (Codex): frontend 1.28 tensor descriptor default.
INPUT_SHAPE = (8, 1024, 1, 12)
INPUT_STRIDE = (12288, 1, 12288, 1024)
WEIGHT_SHAPE = (1024, 1024, 1, 3)
WEIGHT_STRIDE = (3072, 1, 3072, 1024)


def validate_tensors(inputs: torch.Tensor, weight: torch.Tensor) -> None:
    if (
        inputs.device.type != "cuda"
        or weight.device != inputs.device
        or inputs.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or tuple(inputs.shape) != INPUT_SHAPE
        or tuple(inputs.stride()) != INPUT_STRIDE
        or tuple(weight.shape) != WEIGHT_SHAPE
        or tuple(weight.stride()) != WEIGHT_STRIDE
        or inputs.data_ptr() % TENSOR_ALIGNMENT
        or weight.data_ptr() % TENSOR_ALIGNMENT
    ):
        raise RuntimeError("Unsupported controlled convolution tensor metadata")
    else:
        pass


class ControlledConvPlan:
    """Own one plan and workspace; calls and captured replays must be serialized."""

    def __init__(self, inputs: torch.Tensor, weight: torch.Tensor) -> None:
        validate_tensors(inputs, weight)
        with torch.cuda.device(inputs.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Controlled convolution must be prepared before capture"
                )
            else:
                pass
        if (
            not torch.backends.cudnn.enabled
            or torch.backends.cudnn.deterministic
            or torch.are_deterministic_algorithms_enabled()
        ):
            raise RuntimeError(
                "Controlled convolution requires default cuDNN determinism settings"
            )
        else:
            pass
        if (
            torch.cuda.get_device_capability(inputs.device) != (12, 0)
            or torch.backends.cudnn.version() != CUDNN_VERSION
        ):
            raise RuntimeError("Controlled convolution requires SM120 and cuDNN 9.20")
        else:
            pass
        if cudnn is None:
            raise RuntimeError(
                f"Controlled convolution requires nvidia-cudnn-frontend=={FRONTEND_VERSION}"
            )
        else:
            pass
        if (
            cudnn.__version__ != FRONTEND_VERSION
            or cudnn.backend_version() != CUDNN_VERSION
        ):
            raise RuntimeError(
                f"Controlled convolution requires frontend {FRONTEND_VERSION} and cuDNN 9.20"
            )
        else:
            pass
        self.frontend = cudnn
        self.device = inputs.device
        with torch.cuda.device(self.device):
            self.handle = cudnn.create_handle()
            self.release_handle = weakref.finalize(
                self, cudnn.destroy_handle, self.handle
            )
            self.graph = cudnn.pygraph(
                handle=self.handle,
                io_data_type=cudnn.data_type.BFLOAT16,
                intermediate_data_type=cudnn.data_type.FLOAT,
                compute_data_type=cudnn.data_type.FLOAT,
            )
            self.x = self.graph.tensor(dim=INPUT_SHAPE, stride=INPUT_STRIDE, name="x")
            self.w = self.graph.tensor(dim=WEIGHT_SHAPE, stride=WEIGHT_STRIDE, name="w")
            self.y = self.graph.conv_fprop(
                image=self.x,
                weight=self.w,
                padding=[0, 1],
                stride=[1, 1],
                dilation=[1, 1],
            )
            self.y.set_output(True).set_dim(INPUT_SHAPE).set_stride(INPUT_STRIDE)
            self.y.set_data_type(cudnn.data_type.BFLOAT16)
            knobs = {
                cudnn.knob_type.TILE_SIZE: 8,
                cudnn.knob_type.SPLIT_K_BUF: 2,
                cudnn.knob_type.TILEK: 1,
                cudnn.knob_type.STAGES: 4,
                cudnn.knob_type.REDUCTION_MODE: 0,
                cudnn.knob_type.SPLIT_K_SLC: 342,
                cudnn.knob_type.IDX_MODE: 1,
                cudnn.knob_type.SPECFILT: 0,
            }
            self.graph.create_execution_plan(ENGINE_ID, knobs)
            self.graph.select_plan(0)
            self.graph.check_support()
            self.graph.build_plan_at_index(0)
            self.workspace = torch.empty(
                self.graph.get_workspace_size(), dtype=torch.uint8, device=self.device
            )

    def __call__(self, inputs: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        validate_tensors(inputs, weight)
        if inputs.device != self.device:
            raise RuntimeError("Controlled convolution plan belongs to another device")
        else:
            pass
        with torch.cuda.device(self.device):
            stream = torch.cuda.current_stream(self.device)
            self.frontend.set_stream(self.handle, stream.cuda_stream)
            inputs.record_stream(stream)
            weight.record_stream(stream)
            self.workspace.record_stream(stream)
            output = torch.empty_like(inputs, memory_format=torch.preserve_format)
            self.graph.execute(
                {self.x: inputs, self.w: weight, self.y: output},
                self.workspace,
                handle=self.handle,
            )
            return output


@dataclass(frozen=True)
class CachedConvWeight:
    name: str
    source: torch.Tensor
    packed: torch.Tensor
    version: int | None


class Conv1dWeightCache(TorchDispatchMode):
    """Keep the original compiled graph and replace its external convolution calls.

    Construct after final weight loading, device placement, and dtype conversion.
    Retain the cache until captured graphs are destroyed; recreate after weights change.
    """

    def __init__(
        self, module: torch.nn.Module, *, use_controlled_plan: bool = False
    ) -> None:
        super().__init__()
        self.module = module
        self.use_controlled_plan = use_controlled_plan
        self.weights: dict[int, CachedConvWeight] = {}
        for name, layer in module.named_modules():
            if isinstance(layer, torch.nn.Conv1d):
                weight = layer.weight
                source = weight.detach()
                self.weights[weight.data_ptr()] = CachedConvWeight(
                    name=f"{name}.weight" if name else "weight",
                    source=source,
                    packed=source.unsqueeze(2).contiguous(
                        memory_format=torch.channels_last
                    ),
                    version=(
                        None if torch.is_inference(weight) else weight._version
                    ),  # noqa: SLF001  # leading-underscore: PyTorch tensor version API
                )
            else:
                pass
        self.version_checks_available = all(
            entry.version is not None for entry in self.weights.values()
        )
        self.controlled_plan: ControlledConvPlan | None = None
        self.hits = 0
        self.misses = 0

    def validate_weights(self) -> None:
        """Check weights before capture; live weight updates require a new worker."""
        for pointer, entry in self.weights.items():
            current = self.module.get_parameter(entry.name)
            version = (
                None if torch.is_inference(current) else current._version
            )  # noqa: SLF001  # leading-underscore: PyTorch tensor version API
            if (
                current.data_ptr() != pointer
                or current.shape != entry.source.shape
                or current.stride() != entry.source.stride()
                or current.dtype != entry.source.dtype
                or current.device != entry.source.device
                or version != entry.version
            ):
                raise RuntimeError(
                    f"Convolution weight changed; rebuild cache: {entry.name}"
                )
            else:
                pass

    def __torch_dispatch__(
        self,
        func: OpOverload,
        types: tuple[type, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        if func not in (
            torch.ops.aten.convolution.default,
            torch.ops.aten.conv1d.default,
        ):
            return func(*args, **(kwargs or {}))
        else:
            pass
        if func is torch.ops.aten.conv1d.default:
            names = (
                "input",
                "weight",
                "bias",
                "stride",
                "padding",
                "dilation",
                "groups",
            )
        else:
            names = (
                "input",
                "weight",
                "bias",
                "stride",
                "padding",
                "dilation",
                "transposed",
                "output_padding",
                "groups",
            )
        fields: dict[str, object] = {
            "bias": None,
            "stride": [1],
            "padding": [0],
            "dilation": [1],
            "transposed": False,
            "output_padding": [0],
            "groups": 1,
        }
        fields.update(zip(names, args))
        fields.update(kwargs or {})
        inputs = cast(torch.Tensor, fields["input"])
        weight = cast(torch.Tensor, fields["weight"])
        entry = self.weights.get(weight.data_ptr())
        if entry is None:
            self.misses += 1
            return func(*args, **(kwargs or {}))
        else:
            pass
        expected = entry.source if weight.ndim == 3 else entry.source.unsqueeze(2)
        if (
            weight.shape != expected.shape
            or weight.dtype != expected.dtype
            or weight.device != expected.device
            or any(
                size != 1 and got != want
                for size, got, want in zip(
                    weight.shape, weight.stride(), expected.stride()
                )
            )
        ):
            self.misses += 1
            return func(*args, **(kwargs or {}))
        else:
            pass
        if fields["transposed"] or any(cast(Sequence[int], fields["output_padding"])):
            self.misses += 1
            return func(*args, **(kwargs or {}))
        else:
            pass
        stride = list(cast(Sequence[int], fields["stride"]))
        padding = list(cast(Sequence[int], fields["padding"]))
        dilation = list(cast(Sequence[int], fields["dilation"]))
        if inputs.ndim == weight.ndim == 3:
            inputs = inputs.unsqueeze(2)
            stride, padding, dilation = [1, *stride], [0, *padding], [1, *dilation]
            squeeze = True
        elif (
            inputs.ndim == weight.ndim == 4
            and inputs.shape[2] == weight.shape[2] == 1
            and stride[0] == dilation[0] == 1
            and padding[0] == 0
        ):
            squeeze = False
        else:
            self.misses += 1
            return func(*args, **(kwargs or {}))
        self.hits += 1
        inputs = inputs.contiguous(memory_format=torch.channels_last)
        if (
            self.use_controlled_plan
            and inputs.device.type == "cuda"
            and inputs.dtype == torch.bfloat16
            and tuple(inputs.shape) == INPUT_SHAPE
            and tuple(entry.packed.shape) == WEIGHT_SHAPE
            and fields["bias"] is None
            and stride == dilation == [1, 1]
            and padding == [0, 1]
            and fields["groups"] == 1
        ):
            # note (Codex): preserve the NCHW reduction plan for this NHWC shape.
            if self.controlled_plan is None:
                self.controlled_plan = ControlledConvPlan(inputs, entry.packed)
            else:
                pass
            result = self.controlled_plan(inputs, entry.packed)
        else:
            result = torch.ops.aten.convolution.default(
                inputs,
                entry.packed,
                fields["bias"],
                stride,
                padding,
                dilation,
                False,
                [0, 0],
                fields["groups"],
            )
        return (result.squeeze(2) if squeeze else result).contiguous()
