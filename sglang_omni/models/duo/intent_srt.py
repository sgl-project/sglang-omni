"""Qwen3-VL H50 execution on Omni's native SGLang worker and KV pools."""

import time
from array import array
from collections import deque
from dataclasses import dataclass, field
from typing import Literal

import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.mm_schedule import init_mm_embedding_cache
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    Req,
    ScheduleBatch,
)
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

from sglang_omni.model_runner.base import ModelRunner, resolve_deferred_prefill_inputs
from sglang_omni.scheduling.bootstrap import (
    create_sglang_infrastructure,
    init_sglang_cuda_graphs,
)
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.vendor.sglang.core import ServerArgs


@dataclass(frozen=True)
class SRTPlan:
    shared_h50: torch.Tensor | None
    plan_h50: torch.Tensor
    generated_ids: list[int]
    kv_tokens: int
    kv_bytes: int
    graph_replays: int
    reused_tokens: int = 0
    evicted_tokens: int = 0
    prefill_submit_seconds: float = 0.0
    decode_wall_seconds: float = 0.0
    h50_return_seconds: float = 0.0


@dataclass(kw_only=True)
class IntentWindow:
    prefix_tokens: int
    evicted_tokens: int = 0
    block_tokens: deque[int] = field(default_factory=deque)


class IntentH50Worker:
    def __init__(
        self,
        model_path: str,
        *,
        device: int,
        max_context_tokens: int,
        enable_cuda_graph: bool,
        memory_fraction: float = 0.52,
        tp_size: int = 1,
        tp_rank: int = 0,
        nccl_port: int | None = None,
        quantization: Literal["fp8"] | None = None,
        kv_window_tokens: int | None = None,
    ) -> None:
        if (
            kv_window_tokens is not None
            and not 0 < kv_window_tokens <= max_context_tokens
        ):
            raise ValueError(
                "Intent KV window must be positive and fit context capacity"
            )
        else:
            pass
        self.kv_window_tokens = kv_window_tokens
        self.windows: dict[str, IntentWindow] = {}
        arguments = ServerArgs(
            model_path=model_path,
            tp_size=tp_size,
            quantization=quantization,
            dtype="bfloat16",
            trust_remote_code=True,
            attention_backend="fa3",
            disable_overlap_schedule=True,
            disable_cuda_graph=not enable_cuda_graph,
            cuda_graph_config={"decode": {"bs": [1]}},
            max_running_requests=4,
            context_length=max_context_tokens,
            max_total_tokens=max_context_tokens,
            mem_fraction_static=memory_fraction,
            page_size=1,
            enable_return_hidden_states=True,
            return_hidden_states_mode="last",
        )
        (
            self.worker,
            self.tree_cache,
            self.request_pool,
            self.allocator,
            self.model_config,
        ) = create_sglang_infrastructure(
            arguments,
            device,
            tp_rank=tp_rank,
            nccl_port=nccl_port,
            defer_cuda_graph_capture=True,
        )
        self.runner = self.worker.model_runner
        self.decode_runner = ModelRunner(self.worker, SGLangOutputProcessor())
        self.model = self.runner.model
        self.model.set_eagle3_layers_to_capture([49])
        # Match the training model's residual/deepstack ordering.
        self.model.model.use_hf_deepstack_order = True
        init_mm_embedding_cache(0)
        init_sglang_cuda_graphs(self.worker)
        self.requests: dict[str, Req] = {}
        # Session usage counts KV storage across all TP ranks.
        self.bytes_per_token = (
            2
            * self.model_config.num_hidden_layers
            * self.model_config.get_num_kv_heads(tp_size)
            * tp_size
            * self.model_config.head_dim
            * torch.tensor([], dtype=self.model_config.dtype).element_size()
        )

    def new_batch(self, request: Req) -> ScheduleBatch:
        return ScheduleBatch.init_new(
            reqs=[request],
            req_to_token_pool=self.request_pool,
            token_to_kv_pool_allocator=self.allocator,
            tree_cache=self.tree_cache,
            model_config=self.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )

    def forward(
        self, batch: ScheduleBatch, positions: torch.Tensor
    ) -> tuple[LogitsProcessorOutput, bool]:
        resolve_deferred_prefill_inputs(batch, self.runner.device)
        forward = ForwardBatch.init_new(
            batch,
            self.runner,
            capture_hidden_mode=(
                CaptureHiddenMode.LAST
                if positions.numel() == 3
                else CaptureHiddenMode.FULL
            ),
            return_hidden_states_before_norm=False,
            spec_mrope_positions=positions.to(self.runner.device),
        )
        result = self.runner.forward(forward)
        request = batch.reqs[0]
        request.kv.kv_committed_len = int(batch.seq_lens_cpu[0])
        return result.logits_output, result.can_run_graph

    @torch.inference_mode()
    def plan(
        self,
        session_id: str,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        *,
        patches: torch.Tensor,
        grid: torch.Tensor,
        keep: torch.Tensor,
        end_token_id: int,
        forbidden_token_ids: list[int],
        max_new_tokens: int,
        forced_tokens: list[int] | None = None,
        prefix_tokens: int = 0,
    ) -> SRTPlan:
        ids = input_ids.reshape(-1).tolist()
        request = self.requests.get(session_id)
        first = request is None
        if first:
            self.windows[session_id] = IntentWindow(prefix_tokens=prefix_tokens)
        else:
            pass
        window = self.windows[session_id]
        absolute_prefix_tokens = (
            0 if first else request.kv.kv_committed_len + window.evicted_tokens
        )
        new_input_tokens = len(ids) - absolute_prefix_tokens
        if self.kv_window_tokens is not None:
            if (
                window.prefix_tokens + new_input_tokens + max_new_tokens
                > self.kv_window_tokens
                and not first
            ):
                raise ValueError(
                    "Intent window cannot fit static prefix, current input and reply budget"
                )
            elif first and len(ids) + max_new_tokens > self.kv_window_tokens:
                raise ValueError(
                    "Intent window cannot fit the first input and reply budget"
                )
            else:
                pass
            dropped_tokens = 0
            while (
                len(ids) - window.evicted_tokens - dropped_tokens + max_new_tokens
                > self.kv_window_tokens
            ):
                dropped_tokens += window.block_tokens.popleft()
            if dropped_tokens:
                assert request.kv.cache_protected_len == 0
                assert request.kv.kv_allocated_len == request.kv.kv_committed_len
                slots = self.request_pool.req_to_token[request.kv.req_pool_idx]
                stop = window.prefix_tokens + dropped_tokens
                length = request.kv.kv_committed_len
                self.allocator.free(
                    slots[window.prefix_tokens : stop].to(torch.int64).clone()
                )
                slots[window.prefix_tokens : length - dropped_tokens].copy_(
                    slots[stop:length].clone()
                )
                request.kv.kv_committed_len -= dropped_tokens
                request.kv.kv_allocated_len -= dropped_tokens
                previous = list(request.origin_input_ids) + list(request.output_ids)
                request.origin_input_ids = array(
                    "q", previous[: window.prefix_tokens] + previous[stop:]
                )
                request.output_ids = array("q")
                window.evicted_tokens += dropped_tokens
            else:
                pass
        else:
            pass
        if window.evicted_tokens:
            stop = window.prefix_tokens + window.evicted_tokens
            ids = ids[: window.prefix_tokens] + ids[stop:]
            positions = torch.cat(
                (positions[:, : window.prefix_tokens], positions[:, stop:]), dim=1
            )
            keep = torch.cat((keep[: window.prefix_tokens], keep[stop:]))
        else:
            pass
        if first:
            request = Req(
                rid=session_id,
                origin_input_text="",
                origin_input_ids=array("q", ids),
                sampling_params=SamplingParams(
                    temperature=0, max_new_tokens=max_new_tokens
                ),
            )
            request.return_hidden_states = True
            request.skip_radix_cache_insert = True
            self.requests[session_id] = request
            prefix_length = 0
        else:
            prefix_length = request.kv.kv_committed_len
            assert ids[:prefix_length] == list(request.origin_input_ids) + list(
                request.output_ids
            )
            request.origin_input_ids = array("q", ids)
            request.output_ids = array("q")
        request.full_untruncated_fill_ids = request.origin_input_ids
        request.prefix_len = prefix_length
        request.extend_end = len(ids)
        request.logprob_start_len = -1
        video_id = self.model.config.video_token_id
        offsets: list[tuple[int, int]] = []
        for index in range(prefix_length, len(ids)):
            if ids[index] == video_id:
                if offsets and offsets[-1][1] == index - 1:
                    offsets[-1] = (offsets[-1][0], index)
                else:
                    offsets.append((index, index))
            else:
                pass
        item = MultimodalDataItem(
            modality=Modality.VIDEO,
            feature=patches,
            pad_value=video_id,
            offsets=offsets,
            model_specific_data={"video_grid_thw": grid.reshape(-1, 3)},
        )
        request.multimodal_inputs = MultimodalInputs(
            mm_items=[item],
            video_token_id=video_id,
            mrope_positions=positions,
            mrope_position_delta=positions.max().reshape(1, 1) + 1 - len(ids),
        )
        batch = self.new_batch(request)
        batch.prepare_for_extend()
        prefill_started = time.monotonic()
        output, graph = self.forward(batch, positions[:, prefix_length:])
        prefill_submitted = time.monotonic()
        shared = (
            output.hidden_states[keep[prefix_length:].to(output.hidden_states.device)]
            .cpu()
            .clone()
            if first
            else None
        )
        generated, hidden = [], []
        graph_replays = int(graph)
        next_position = int(positions.max()) + 1
        forbidden = torch.tensor(
            forbidden_token_ids, device=self.runner.device, dtype=torch.long
        )
        group = self.worker.get_tp_group()
        decode_positions = torch.empty(
            (3, 1), device=self.runner.device, dtype=torch.long
        )
        forced_ids = (
            torch.tensor(forced_tokens, device=self.runner.device, dtype=torch.long)
            if forced_tokens is not None
            else None
        )
        decode_started = time.monotonic()
        for index in range(max_new_tokens):
            logits = output.next_token_logits[0].float()
            logits[forbidden] = -torch.inf
            token_tensor = logits.argmax().reshape(1)
            if group.world_size > 1:
                torch.distributed.broadcast(
                    token_tensor, src=0, group=group.device_group
                )
            else:
                pass
            if forced_ids is not None:
                token_tensor = forced_ids[index : index + 1]
            else:
                pass
            host_token = self.decode_runner.next_token_id_host_buf(token_tensor, 1)
            host_token.copy_(token_tensor, non_blocking=True)
            token_ready = self.decode_runner.next_token_id_event()
            token_ready.record()
            batch.input_ids = token_tensor
            batch.prepare_for_decode()
            decode_positions.fill_(next_position + index)
            output, graph = self.forward(batch, decode_positions)
            # Capture H50 before the next graph replay overwrites its output buffer.
            token_hidden = output.hidden_states[:1].clone()
            token_ready.synchronize()
            token = int(host_token[0])
            generated.append(token)
            request.output_ids.append(token)
            graph_replays += int(graph)
            if token == end_token_id:
                break
            else:
                hidden.append(token_hidden)
        else:
            raise RuntimeError("Intent reply did not finish within max_new_tokens")
        decode_done = time.monotonic()
        if not hidden:
            raise RuntimeError("Intent emitted an empty plan")
        else:
            pass
        window.block_tokens.append(
            new_input_tokens + len(generated) - (window.prefix_tokens if first else 0)
        )
        plan_h50 = torch.cat(hidden).cpu()
        h50_returned = time.monotonic()
        return SRTPlan(
            shared,
            plan_h50,
            generated,
            request.kv.kv_committed_len,
            request.kv.kv_committed_len * self.bytes_per_token,
            graph_replays,
            prefix_length,
            window.evicted_tokens,
            prefill_submitted - prefill_started,
            decode_done - decode_started,
            h50_returned - decode_done,
        )

    def close(self, session_id: str) -> None:
        self.windows.pop(session_id, None)
        request = self.requests.pop(session_id, None)
        if request is not None and request.kv.holds_kv:
            release_kv_cache(request, self.tree_cache, checkpoint=False)
        else:
            pass
