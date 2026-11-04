# SPDX-License-Identifier: Apache-2.0
"""Trace the routed experts every Drama 3 backbone token selects, from outside the model.

Serve with the trace engine builder and a trace directory in the engine stage's env:

    sgl-omni serve --model-path fishaudio/drama-3-int4 \\
      --tts_engine.factory.engine_builder_path benchmarks.drama3_expert_trace.Drama3ExpertTraceEngineBuilder \\
      --tts_engine.env.DRAMA3_EXPERT_TRACE_DIR /path/to/traces

A forward hook on each MoE layer's top-k copies its expert ids into a fixed
buffer, so captured prefill and decode CUDA graphs replay the copy. After each
forward the model runner appends the real rows to per-process files, which a
killed server keeps up to its last finished step.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, TextIO

import torch
from sglang.srt.arg_groups.model_override_base import resolved_view
from sglang.srt.layers.moe.topk import (
    BypassedTopKOutput,
    StandardTopKOutput,
    StandardTopKOutputPacked,
    TopK,
)
from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock
from sglang.srt.server_args import ServerArgs
from torch import nn

from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.models.fishaudio_drama3.decode_state import (
    request_data_of,
    request_ids_of,
)
from sglang_omni.models.fishaudio_drama3.engine_builder import Drama3EngineBuilder
from sglang_omni.models.fishaudio_drama3.model_runner import Drama3ModelRunner
from sglang_omni.models.fishaudio_drama3.sglang_model import Drama3ForCausalLM
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.scheduling.types import SchedulerRequest
from sglang_omni.vendor.sglang.core import (
    ForwardBatch,
    GenerationBatchResult,
    ScheduleBatch,
)

TRACE_DIR_ENVIRONMENT_VARIABLE = "DRAMA3_EXPERT_TRACE_DIR"
EXPERT_ID_DTYPE = torch.int16
TRACE_FORMAT_VERSION = 1
EXPERT_IDS_FILE_NAME = "expert_ids.int16"
FORWARDS_FILE_NAME = "forwards.jsonl"
METADATA_FILE_NAME = "metadata.json"

logger = logging.getLogger(__name__)

__all__ = [
    "Drama3ExpertTraceEngineBuilder",
    "Drama3ExpertTraceModelRunner",
    "ExpertTrace",
    "TopKIdsRecorder",
]


class TopKIdsRecorder:
    """Forward hook that copies one layer's top-k expert ids into a fixed buffer."""

    def __init__(self, expert_ids: torch.Tensor) -> None:
        self.expert_ids = expert_ids

    def __call__(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: StandardTopKOutput | StandardTopKOutputPacked | BypassedTopKOutput,
    ) -> None:
        if isinstance(output, (StandardTopKOutput, StandardTopKOutputPacked)):
            topk_ids = output.topk_ids
        elif isinstance(output, BypassedTopKOutput):
            # The fused kernel routes internally; routing again here is deterministic.
            topk_ids = output.to_standard().topk_ids
        else:
            raise NotImplementedError(
                f"the Drama 3 expert trace cannot read {type(output).__name__}"
            )
        token_count = topk_ids.shape[0]
        if token_count > self.expert_ids.shape[0]:
            raise RuntimeError(
                f"a forward routed {token_count} tokens; the expert trace holds "
                f"{self.expert_ids.shape[0]}"
            )
        else:
            self.expert_ids[:token_count].copy_(topk_ids)


@dataclass(kw_only=True)
class ExpertTrace:
    """Fixed device buffers of each MoE layer's expert ids and the files they go to."""

    moe_layer_ids: list[int]
    layer_expert_ids: list[torch.Tensor]
    expert_ids_file: BinaryIO
    forwards_file: TextIO
    forward_index: int = 0

    @classmethod
    def attach(
        cls,
        backbone: nn.Module,
        *,
        max_token_count: int,
        trace_dir: Path,
        checkpoint_path: str,
        quantization: str | None,
    ) -> ExpertTrace:
        """Hook every MoE layer of the backbone and open this process's trace files."""
        moe_blocks = [
            (layer_id, layer.mlp)
            for layer_id, layer in enumerate(backbone.layers)
            if isinstance(layer.mlp, Qwen2MoeSparseMoeBlock)
        ]
        if not moe_blocks:
            raise RuntimeError("the Drama 3 backbone has no MoE layer to trace")
        else:
            pass
        first_block = moe_blocks[0][1]
        top_k = first_block.topk.topk_config.top_k
        device = first_block.gate.weight.device
        layer_expert_ids = []
        for layer_id, moe_block in moe_blocks:
            topk = moe_block.topk
            assert isinstance(topk, TopK)
            expert_ids = torch.zeros(
                (max_token_count, top_k), dtype=EXPERT_ID_DTYPE, device=device
            )
            topk.register_forward_hook(TopKIdsRecorder(expert_ids))
            layer_expert_ids.append(expert_ids)
        experts = first_block.experts
        routed_expert_count = first_block.num_experts
        expert_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in experts.parameters()
            if parameter.dim() > 0 and parameter.shape[0] == experts.num_local_experts
        )
        # Container pids repeat across restarts; the start time keeps two runs apart.
        process_dir = trace_dir / f"pid{os.getpid()}-{time.strftime('%Y%m%d-%H%M%S')}"
        process_dir.mkdir(parents=True, exist_ok=False)
        metadata = {
            "format_version": TRACE_FORMAT_VERSION,
            "checkpoint_path": checkpoint_path,
            "quantization": quantization,
            "moe_layer_ids": [layer_id for layer_id, _ in moe_blocks],
            "routed_expert_count": routed_expert_count,
            "top_k": top_k,
            "expert_id_dtype": "int16",
            "bytes_per_routed_expert": expert_bytes // experts.num_local_experts,
            "device_name": torch.cuda.get_device_name(device),
        }
        (process_dir / METADATA_FILE_NAME).write_text(json.dumps(metadata, indent=2))
        return cls(
            moe_layer_ids=metadata["moe_layer_ids"],
            layer_expert_ids=layer_expert_ids,
            expert_ids_file=(process_dir / EXPERT_IDS_FILE_NAME).open("ab"),
            forwards_file=(process_dir / FORWARDS_FILE_NAME).open("a"),
        )

    def record(
        self, *, is_prefill: bool, request_ids: list[str], token_counts: list[int]
    ) -> None:
        """Append the expert ids of the forward that just ran, token rows in batch order."""
        token_count = sum(token_counts)
        if len(request_ids) != len(token_counts):
            raise ValueError(
                f"{len(request_ids)} request ids for {len(token_counts)} token counts"
            )
        else:
            pass
        # [tokens, layers, top_k], the order the simulator reads.
        expert_ids = torch.stack(
            [layer_ids[:token_count] for layer_ids in self.layer_expert_ids], dim=1
        ).cpu()
        self.expert_ids_file.write(expert_ids.numpy().tobytes())
        self.expert_ids_file.flush()
        self.forwards_file.write(
            json.dumps(
                {
                    "forward_index": self.forward_index,
                    "is_prefill": is_prefill,
                    "request_ids": request_ids,
                    "token_counts": token_counts,
                }
            )
            + "\n"
        )
        self.forwards_file.flush()
        self.forward_index += 1

    def record_prefill(self, forward_batch: ForwardBatch) -> None:
        """Append the extend that just ran, every request's rows in batch order."""
        assert forward_batch.extend_seq_lens_cpu is not None
        self.record(
            is_prefill=True,
            request_ids=list(forward_batch.rids),
            token_counts=list(forward_batch.extend_seq_lens_cpu),
        )

    def record_decode(self, requests: list[SchedulerRequest]) -> None:
        """Append the decode step that just ran, one row per request in batch order."""
        self.record(
            is_prefill=False,
            request_ids=request_ids_of(
                [request_data_of(request) for request in requests]
            ),
            token_counts=[1] * len(requests),
        )


class Drama3ExpertTraceModelRunner(Drama3ModelRunner):
    """Drama 3 model runner that appends each forward's expert ids once it finished."""

    def __init__(
        self,
        tp_worker: ModelWorker,
        output_processor: SGLangOutputProcessor,
        expert_trace: ExpertTrace,
    ) -> None:
        super().__init__(tp_worker, output_processor)
        self.expert_trace = expert_trace

    def post_prefill(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        super().post_prefill(result, forward_batch, schedule_batch, requests)
        assert forward_batch is not None
        self.expert_trace.record_prefill(forward_batch)

    def post_decode(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        super().post_decode(result, forward_batch, schedule_batch, requests)
        self.expert_trace.record_decode(requests)


class Drama3ExpertTraceEngineBuilder(Drama3EngineBuilder):
    """The Drama 3 engine with its MoE top-k hooked before graph capture."""

    expert_trace: ExpertTrace | None = None

    def validate_before_infrastructure(self, server_args: ServerArgs) -> None:
        super().validate_before_infrastructure(server_args)
        if TRACE_DIR_ENVIRONMENT_VARIABLE not in os.environ:
            raise ValueError(
                f"the Drama 3 expert trace needs {TRACE_DIR_ENVIRONMENT_VARIABLE}; set "
                f"--tts_engine.env.{TRACE_DIR_ENVIRONMENT_VARIABLE} /path/to/traces"
            )
        elif self.enable_async_decode:
            # A lookahead decode overwrites the traced expert ids before they are read.
            raise ValueError(
                "the Drama 3 expert trace needs tts_engine.factory.enable_async_decode=false"
            )
        else:
            pass

    def setup_model(
        self,
        *,
        model_worker: ModelWorker,
        checkpoint_dir: str,
        device: str,
        gpu_id: int,
        server_args: ServerArgs,
    ) -> None:
        super().setup_model(
            model_worker=model_worker,
            checkpoint_dir=checkpoint_dir,
            device=device,
            gpu_id=gpu_id,
            server_args=server_args,
        )
        model = model_worker.model_runner.model
        assert isinstance(model, Drama3ForCausalLM)
        tower_prefill = model.model.tower_prefill
        assert tower_prefill is not None
        trace_dir = Path(os.environ[TRACE_DIR_ENVIRONMENT_VARIABLE])
        # The tower prefill buffers already hold the largest forward's rows.
        self.expert_trace = ExpertTrace.attach(
            model.model,
            max_token_count=tower_prefill.input_frames.shape[0],
            trace_dir=trace_dir,
            checkpoint_path=server_args.model_path,
            quantization=resolved_view(server_args).quantization,
        )
        logger.info(
            f"Drama 3 expert trace: on for {len(self.expert_trace.moe_layer_ids)} MoE "
            f"layers, appending to {trace_dir}"
        )

    def make_model_runner(
        self, model_worker: ModelWorker, output_proc: SGLangOutputProcessor
    ) -> Drama3ExpertTraceModelRunner:
        assert self.expert_trace is not None
        return Drama3ExpertTraceModelRunner(
            model_worker, output_proc, self.expert_trace
        )
