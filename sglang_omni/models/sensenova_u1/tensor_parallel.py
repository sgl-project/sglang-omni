# SPDX-License-Identifier: Apache-2.0
"""Shard dense SenseNova Qwen3 projections while keeping hidden states replicated."""

from __future__ import annotations

import asyncio
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from sglang_omni.models.sensenova_u1.cancellation import SenseNovaCancelledError
from sglang_omni.scheduling.message import IncomingMessage
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.scheduling.tensor_parallel_scheduler import (
    RequestCompute,
    RequestValidator,
    ShutdownCallback,
    TensorParallelScheduler,
)

SUPPORTED_TP_SIZES: tuple[int, ...] = (1, 2, 4, 8)
# The process-group timeout covers initialization and every collective in the group.
TP_COLLECTIVE_TIMEOUT: timedelta = timedelta(minutes=10)


def validate_tp_settings(
    tensor_parallel_rank: int, tensor_parallel_size: int, max_batch_size: int
) -> None:
    if tensor_parallel_size not in SUPPORTED_TP_SIZES:
        raise ValueError("SenseNova TP supports tp_size=1, 2, 4, or 8")
    else:
        pass
    if not 0 <= tensor_parallel_rank < tensor_parallel_size:
        raise ValueError("SenseNova tp_rank must be in [0, tp_size)")
    else:
        pass
    if tensor_parallel_size > 1 and max_batch_size != 1:
        raise ValueError("SenseNova TP requires max_batch_size=1")
    else:
        pass


def validate_tp_device(device: torch.device, nccl_port: int | None) -> None:
    if device.type not in {"cuda", "npu"}:
        raise ValueError("SenseNova TP requires CUDA or Ascend NPU")
    else:
        pass
    if nccl_port is None:
        raise ValueError("SenseNova TP requires nccl_port from the stage launcher")
    else:
        pass


def initialize_tp(
    device: torch.device,
    tensor_parallel_rank: int,
    tensor_parallel_size: int,
    nccl_port: int | None,
) -> dist.ProcessGroup:
    """Create the process-local TP group on the launcher's reserved port."""
    validate_tp_device(device, nccl_port)
    if dist.is_initialized():
        raise RuntimeError("SenseNova TP requires its own distributed worker process")
    else:
        pass
    getattr(torch, device.type).set_device(device)
    dist.init_process_group(
        backend="hccl" if device.type == "npu" else "nccl",
        init_method=f"tcp://127.0.0.1:{nccl_port}",
        rank=tensor_parallel_rank,
        world_size=tensor_parallel_size,
        timeout=TP_COLLECTIVE_TIMEOUT,
    )
    return dist.group.WORLD


class RowParallelLinear(nn.Linear):
    """Linear with a local input shard and a replicated, summed output."""

    def __init__(
        self,
        linear: nn.Linear,
        tensor_parallel_rank: int,
        tensor_parallel_size: int,
        process_group: dist.ProcessGroup | None,
    ) -> None:
        input_feature_count = linear.in_features // tensor_parallel_size
        # note (Wes): Meta placeholders skip nn.Linear's random init; the checkpoint slices below replace them.
        super().__init__(
            input_feature_count,
            linear.out_features,
            bias=linear.bias is not None,
            device="meta",
        )
        self.weight = nn.Parameter(
            linear.weight.detach()[
                :,
                tensor_parallel_rank
                * input_feature_count : (tensor_parallel_rank + 1)
                * input_feature_count,
            ].clone(memory_format=torch.contiguous_format),
            requires_grad=False,
        )
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=False)
        else:
            pass
        self.tp_group: dist.ProcessGroup | None = process_group
        self.train(linear.training)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = F.linear(inputs, self.weight)
        dist.all_reduce(output, group=self.tp_group)
        # note (AuFlow): Summing replicated biases would multiply them by the TP size.
        if self.bias is not None:
            output = output + self.bias
        else:
            pass
        return output


def column_parallel(
    linear: nn.Linear, tensor_parallel_rank: int, tensor_parallel_size: int
) -> nn.Linear:
    output_feature_count = linear.out_features // tensor_parallel_size
    shard = nn.Linear(
        linear.in_features,
        output_feature_count,
        bias=linear.bias is not None,
        device="meta",
    )
    shard.weight = nn.Parameter(
        linear.weight.detach()[
            tensor_parallel_rank
            * output_feature_count : (tensor_parallel_rank + 1)
            * output_feature_count
        ].clone(),
        requires_grad=False,
    )
    if linear.bias is not None:
        shard.bias = nn.Parameter(
            linear.bias.detach()[
                tensor_parallel_rank
                * output_feature_count : (tensor_parallel_rank + 1)
                * output_feature_count
            ].clone(),
            requires_grad=False,
        )
    else:
        pass
    shard.train(linear.training)
    return shard


def apply_tensor_parallel(
    model: nn.Module,
    tensor_parallel_rank: int,
    tensor_parallel_size: int,
    process_group: dist.ProcessGroup | None,
) -> None:
    """Shard loaded CPU weights before moving the model to its accelerator."""
    validate_tp_settings(tensor_parallel_rank, tensor_parallel_size, 1)
    if tensor_parallel_size == 1:
        return
    else:
        pass
    if getattr(model, "sensenova_tp_size", 1) != 1:
        raise ValueError("SenseNova tensor parallelism has already been applied")
    else:
        pass
    backbone = model.language_model.model
    config = backbone.config
    if getattr(config, "model_type", None) != "qwen3":
        raise ValueError(
            "SenseNova TP currently supports the dense Qwen3 backbone only"
        )
    else:
        pass
    for name in ("num_attention_heads", "num_key_value_heads", "intermediate_size"):
        if getattr(config, name) % tensor_parallel_size:
            raise ValueError(
                f"SenseNova {name} must be divisible by tp_size={tensor_parallel_size}"
            )
        else:
            pass

    # note (AuFlow): Validation must finish before any weights are sharded.
    columns: list[tuple[nn.Module, str]] = []
    rows: list[tuple[nn.Module, str]] = []
    for layer in backbone.layers:
        attention = layer.self_attn
        for suffix in ("", "_mot_gen"):
            columns.extend(
                (attention, f"{name}{suffix}")
                for name in ("q_proj", "k_proj", "v_proj")
            )
            rows.append((attention, f"o_proj{suffix}"))
        for mlp in (layer.mlp, layer.mlp_mot_gen):
            columns.extend((mlp, name) for name in ("gate_proj", "up_proj"))
            rows.append((mlp, "down_proj"))
    for projections, dimension in ((columns, "out_features"), (rows, "in_features")):
        for module, name in projections:
            linear = getattr(module, name)
            if (
                not isinstance(linear, nn.Linear)
                or getattr(linear, dimension) % tensor_parallel_size
            ):
                raise ValueError(
                    f"SenseNova {name} cannot be sharded with tp_size={tensor_parallel_size}"
                )
            else:
                pass

    for module, name in columns:
        setattr(
            module,
            name,
            column_parallel(
                getattr(module, name), tensor_parallel_rank, tensor_parallel_size
            ),
        )
    for module, name in rows:
        setattr(
            module,
            name,
            RowParallelLinear(
                getattr(module, name),
                tensor_parallel_rank,
                tensor_parallel_size,
                process_group,
            ),
        )
    for layer in backbone.layers:
        for mlp in (layer.mlp, layer.mlp_mot_gen):
            mlp.intermediate_size //= tensor_parallel_size
            mlp.npu_gate_up_weight = None
    model.sensenova_tp_size = tensor_parallel_size
    model.sensenova_tp_group = process_group


def synchronize_tp_noise(model: nn.Module, noise: torch.Tensor) -> None:
    """Use rank zero's initial image noise on every backend."""
    if getattr(model, "sensenova_tp_size", 1) > 1:
        dist.broadcast(noise, src=0, group=model.sensenova_tp_group)
    else:
        pass


class SenseNovaTensorParallelScheduler(TensorParallelScheduler):
    """Stop TP generation at SenseNova checkpoints without breaking collectives."""

    def __init__(
        self,
        request_compute: RequestCompute,
        *,
        report_cancellation: bool,
        shutdown_join_timeout_s: float,
        validate_request: RequestValidator | None = None,
        shutdown_callback: ShutdownCallback | None = None,
    ) -> None:
        super().__init__(
            request_compute,
            shutdown_join_timeout_s=shutdown_join_timeout_s,
            validate_request=validate_request,
            shutdown_callback=shutdown_callback,
        )
        self.report_cancellation: bool = report_cancellation

    def abort(self, request_id: str) -> None:
        SimpleScheduler.abort(self, request_id)

    def consume_if_aborted(self, request_id: str) -> bool:
        return False

    def run_single(
        self, message: IncomingMessage, loop: asyncio.AbstractEventLoop
    ) -> None:
        try:
            SimpleScheduler.run_single(self, message, loop)
        except SenseNovaCancelledError as error:
            # note (Wes): A follower stage treats any scheduler error as a broken collective sequence, so only the leader reports the agreed cancellation.
            if self.report_cancellation:
                self.emit_error(message.request_id, error, self.outbox)
            else:
                pass
        except Exception as error:
            # note (AuFlow): A failed rank must not reuse collectives for the next request.
            self.compute_error = error
            self.running = False
            raise
        finally:
            with self.abort_lock:
                self.aborted.discard(message.request_id)
