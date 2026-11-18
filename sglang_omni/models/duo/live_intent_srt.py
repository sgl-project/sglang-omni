"""Persistent native Intent worker, isolated from Diffusion's process globals."""

import atexit
import gc
import json
import multiprocessing
import time
import traceback
from argparse import Namespace
from dataclasses import dataclass
from multiprocessing.connection import Connection, wait
from pathlib import Path
from threading import Event
from typing import ClassVar, Literal

import numpy as np
import torch
from sglang.srt.utils import freeze_gc
from torchtitan.models.duo_realization.modules.intent import SPECIALS, prepare_tokenizer
from transformers import AutoConfig, AutoProcessor
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

from sglang_omni.models.duo.frame_intent import FrameIntentPlanner
from sglang_omni.models.duo.intent_srt import IntentH50Worker, SRTPlan
from sglang_omni.models.duo.live_intent import IntentVisual, LiveIntentState, LivePlan
from sglang_omni.utils.connection import find_available_port


@dataclass(frozen=True)
class IntentPlanRequest:
    session_id: str
    input_ids: torch.Tensor
    positions: torch.Tensor
    patches: torch.Tensor
    grid: torch.Tensor
    keep: torch.Tensor
    prefix_tokens: int


@dataclass(frozen=True)
class IntentReply:
    plan: SRTPlan | None = None
    error: str | None = None


class IntentPositions:
    # Reference preprocessing; the forward itself uses SRT's MRoPE kernel.
    get_rope_index = Qwen3VLModel.get_rope_index
    get_vision_position_ids = Qwen3VLModel.get_vision_position_ids

    def __init__(self, config: Qwen3VLConfig) -> None:
        self.config = config


def run_intent_worker(
    connection: Connection,
    model_path: str,
    device: int,
    tp_rank: int,
    tp_size: int,
    nccl_port: int,
    max_context_tokens: int,
    enable_cuda_graph: bool,
    end_token_id: int,
    forbidden: list[int],
    max_new_tokens: int,
    quantization: Literal["fp8"] | None,
    kv_window_tokens: int | None,
) -> None:
    try:
        worker = IntentH50Worker(
            model_path,
            device=device,
            tp_rank=tp_rank,
            tp_size=tp_size,
            nccl_port=nccl_port,
            max_context_tokens=max_context_tokens,
            enable_cuda_graph=enable_cuda_graph,
            quantization=quantization,
            kv_window_tokens=kv_window_tokens,
        )
        gc.collect()
        freeze_gc(f"Intent rank {tp_rank}")
        connection.send(IntentReply())
        while True:
            request: IntentPlanRequest | str | None = connection.recv()
            if request is None:
                return
            elif isinstance(request, str):
                worker.close(request)
                connection.send(IntentReply())
            else:
                result = worker.plan(
                    request.session_id,
                    request.input_ids,
                    request.positions,
                    patches=request.patches,
                    grid=request.grid,
                    keep=request.keep,
                    end_token_id=end_token_id,
                    forbidden_token_ids=forbidden,
                    max_new_tokens=max_new_tokens,
                    prefix_tokens=request.prefix_tokens,
                )
                connection.send(IntentReply(plan=result if tp_rank == 0 else None))
    except BaseException:
        connection.send(IntentReply(error=traceback.format_exc()))
        raise
    finally:
        connection.close()


class SRTFrameIntentPlanner(FrameIntentPlanner):
    incremental_inputs: ClassVar[bool] = True

    def __init__(
        self,
        args: Namespace,
        *,
        device: torch.device,
        max_context_tokens: int,
        max_new_tokens: int,
        enable_cuda_graph: bool,
        devices: tuple[int, ...] | None = None,
        quantization: Literal["fp8"] | None = None,
        kv_window_tokens: int | None = None,
    ) -> None:
        devices = devices if devices is not None else (device.index,)
        self.tokenizer, _ = prepare_tokenizer(args)
        if not args.origin_intent:
            manifest_path = Path(args.intent) / "duplex_adapter_merge.json"
            if not manifest_path.is_file():
                raise ValueError(
                    "Native adapted Intent requires a merged checkpoint; run "
                    "scripts/merge_duo_intent_adapter.py first"
                )
            else:
                manifest = json.loads(manifest_path.read_text())
            if manifest["adapter_weights"] != args.intent_identity["adapter_weights"]:
                raise ValueError("Merged Intent checkpoint has a different adapter")
            else:
                pass
        else:
            pass
        self.processor = AutoProcessor.from_pretrained(
            args.intent, local_files_only=True
        )
        self.processor.tokenizer = self.tokenizer
        self.processor.video_processor.size = {
            "shortest_edge": 1024,
            "longest_edge": args.intent_max_pixels,
        }
        self.config = AutoConfig.from_pretrained(args.intent, local_files_only=True)
        self.positions = IntentPositions(self.config)
        self.kv_window_tokens = kv_window_tokens
        self.max_context_tokens = max_context_tokens
        self.max_new_tokens = max_new_tokens
        self.states: dict[str, LiveIntentState] = {}
        self.usages: dict[str, tuple[int, int]] = {}
        self.worker_device = torch.device("cuda", devices[0])
        self.end_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        allowed = {
            self.tokenizer.convert_tokens_to_ids(token)
            for token in SPECIALS
            if token == "<wait>" or token.startswith("<emo_")
        } | {self.end_token_id}
        forbidden = sorted(set(self.tokenizer.all_special_ids) - allowed)
        context = multiprocessing.get_context("spawn")
        self.connections: list[Connection] = []
        self.processes: list[multiprocessing.Process] = []
        nccl_port = find_available_port("127.0.0.1")
        atexit.register(self.shutdown)
        try:
            for rank, gpu in enumerate(devices):
                parent, child = context.Pipe()
                process = context.Process(
                    target=run_intent_worker,
                    args=(
                        child,
                        args.intent,
                        gpu,
                        rank,
                        len(devices),
                        nccl_port,
                        max_context_tokens,
                        enable_cuda_graph,
                        self.end_token_id,
                        forbidden,
                        max_new_tokens,
                        quantization,
                        kv_window_tokens,
                    ),
                    daemon=True,
                )
                process.start()
                child.close()
                self.connections.append(parent)
                self.processes.append(process)
            self.receive()
        except BaseException:
            self.shutdown(abort=True)
            raise

    @property
    def device(self) -> torch.device:
        return self.worker_device

    def receive(self) -> IntentReply:
        replies = []
        for connection in self.connections:
            ready = wait(
                [connection, *(p.sentinel for p in self.processes)], timeout=600
            )
            if connection not in ready:
                raise RuntimeError("Intent TP worker exited or timed out")
            else:
                pass
            reply: IntentReply = connection.recv()
            if reply.error is not None:
                raise RuntimeError(reply.error)
            else:
                replies.append(reply)
        return replies[0]

    @torch.inference_mode()
    def append(
        self,
        session_id: str,
        frames: np.ndarray,
        user_text: str,
        block_index: int,
        cancelled: Event,
        *,
        frames_per_block: int = 4,
        block_duration_ms: float = 1000 / 6,
        prepared_visual: IntentVisual | None = None,
    ) -> LivePlan:
        if cancelled.is_set():
            raise RuntimeError("Intent session cancelled")
        else:
            pass
        serialization_started = time.monotonic()
        state, visual, prefix, input_ids, reused_tokens = self.prepare_prefix(
            session_id,
            frames,
            user_text,
            block_index,
            frames_per_block,
            block_duration_ms,
            prepared_visual=prepared_visual,
        )
        serialization_done = time.monotonic()
        position_input_ids = (
            input_ids[:, reused_tokens:] if self.incremental_inputs else input_ids
        )
        grid = torch.as_tensor(prefix["video_grid_thw"]).reshape(-1, 3)
        new_positions, _ = self.positions.get_rope_index(
            position_input_ids,
            (position_input_ids == self.config.video_token_id).long() * 2,
            video_grid_thw=grid,
            attention_mask=torch.ones_like(position_input_ids),
        )
        if self.incremental_inputs:
            new_positions += state.next_position
            positions = (
                new_positions
                if state.position_ids is None
                else torch.cat((state.position_ids, new_positions), dim=-1)
            )
        else:
            positions = new_positions
        positions_done = time.monotonic()
        request = IntentPlanRequest(
            session_id,
            input_ids,
            positions[:, 0],
            visual["patches"].cpu(),
            torch.as_tensor(visual["grid"]),
            torch.tensor(prefix["keep"], dtype=torch.bool),
            min(span["start"] for span in prefix["input_spans"] if span["block"] >= 0),
        )
        for connection in self.connections:
            connection.send(request)
        result = self.receive().plan
        assert result is not None
        if state.shared_h50 is None:
            state.shared_h50 = result.shared_h50
        else:
            pass
        body_ids = result.generated_ids[:-1]
        state.bodies[block_index] = body_ids
        state.input_ids = torch.cat(
            (input_ids, torch.tensor([result.generated_ids])), dim=1
        )
        state.input_keep = list(prefix["keep"]) + [True] * len(body_ids) + [False]
        next_position = int(positions.max()) + 1
        generated_positions = (
            torch.arange(next_position, next_position + len(result.generated_ids))
            .reshape(1, 1, -1)
            .expand(3, 1, -1)
        )
        state.position_ids = torch.cat((positions, generated_positions), dim=-1)
        state.next_position = next_position + len(result.generated_ids)
        self.usages[session_id] = (result.kv_tokens, result.kv_bytes)
        return LivePlan(
            shared_h50=state.shared_h50,
            plan_h50=result.plan_h50,
            body_ids=body_ids,
            text=self.tokenizer.decode(body_ids, skip_special_tokens=False),
            prefix_tokens=input_ids.shape[1] - result.evicted_tokens,
            reused_tokens=result.reused_tokens,
            kv_tokens=result.kv_tokens,
            description_tokens=state.description_tokens,
            graph_replays=result.graph_replays,
            evicted_tokens=result.evicted_tokens,
            serialization_seconds=serialization_done - serialization_started,
            position_seconds=positions_done - serialization_done,
            worker_seconds=time.monotonic() - positions_done,
            prefill_submit_seconds=result.prefill_submit_seconds,
            decode_wall_seconds=result.decode_wall_seconds,
            h50_return_seconds=result.h50_return_seconds,
        )

    def usage(self, session_id: str) -> tuple[int, int]:
        return self.usages.get(session_id, (0, 0))

    def close(self, session_id: str) -> None:
        if session_id in self.usages:
            for connection in self.connections:
                connection.send(session_id)
            self.receive()
        else:
            pass
        self.states.pop(session_id, None)
        self.usages.pop(session_id, None)

    def shutdown(self, *, abort: bool = False) -> None:
        if not abort:
            for connection in self.connections:
                try:
                    connection.send(None)
                except (BrokenPipeError, EOFError, OSError):
                    pass
            for process in self.processes:
                process.join(timeout=10)
        else:
            pass
        for process in self.processes:
            if process.is_alive():
                process.terminate()
            else:
                pass
        for process in self.processes:
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join()
            else:
                pass
        for connection in self.connections:
            connection.close()
        self.processes.clear()
        self.connections.clear()
