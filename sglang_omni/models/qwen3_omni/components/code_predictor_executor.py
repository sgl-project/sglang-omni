# SPDX-License-Identifier: Apache-2.0
"""Code Predictor executor — streaming RVQ + feedback generator."""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from sglang_omni.executors._batching import run_batch_loop
from sglang_omni.executors.interface import Executor
from sglang_omni.models.qwen3_omni.pipeline.next_stage import (
    CODE2WAV_STAGE,
    TALKER_AR_STAGE,
)
from sglang_omni.proto import StagePayload

logger = logging.getLogger(__name__)


@dataclass
class _PredictRequest:
    request_id: str
    talker_hidden: torch.Tensor
    layer0_code: torch.Tensor
    result_future: asyncio.Future[dict[str, torch.Tensor]] | None = None


class _CodePredictorWrapper(nn.Module):
    """Batched HF code predictor forward + feedback embedding reduction."""

    def __init__(self, talker_model: nn.Module):
        super().__init__()
        self._talker = talker_model

    def forward(
        self, talker_hidden: torch.Tensor, layer0_code: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        hidden = talker_hidden.unsqueeze(1)
        codes_input = layer0_code.unsqueeze(1)
        layer0_embed = self._talker.get_input_embeddings()(codes_input)

        predictor_result = self._talker.code_predictor.generate(
            inputs_embeds=torch.cat((hidden, layer0_embed), dim=1),
            max_new_tokens=self._talker.config.num_code_groups - 1,
            do_sample=False,
            temperature=0.0,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        result_codes = torch.cat(
            (codes_input, predictor_result.sequences.to(codes_input.device)),
            dim=-1,
        )
        mid_residual_hiddens = [
            hid[0].to(layer0_embed.device) for hid in predictor_result.hidden_states[1:]
        ]
        last_residual_hidden = self._talker.code_predictor.get_input_embeddings()[-1](
            predictor_result.sequences[..., -1:]
        ).to(layer0_embed.device)
        codec_hiddens = torch.cat(
            [layer0_embed] + mid_residual_hiddens + [last_residual_hidden], dim=1
        )
        summed_embeddings = codec_hiddens.sum(1)

        return {
            "codes": result_codes,
            "summed_embeddings": summed_embeddings,
        }


class _LightweightTalkerShell(nn.Module):
    """Minimal talker shell exposing only what _CodePredictorWrapper needs."""

    def __init__(self, config, code_predictor: nn.Module, codec_embedding: nn.Module):
        super().__init__()
        self.config = config
        self.code_predictor = code_predictor
        self.model = nn.Module()
        self.model.codec_embedding = codec_embedding

    def get_input_embeddings(self):
        return self.model.codec_embedding


def _load_talker_model(model_path: str, gpu_id: int = 0):
    """Load only the code predictor + layer-0 codec embedding from the checkpoint."""
    from transformers import AutoConfig
    from transformers.models.qwen3_omni_moe import (
        modeling_qwen3_omni_moe as hf_modeling,
    )

    from sglang_omni.models.weight_loader import load_module, load_weights_by_prefix

    device = f"cuda:{gpu_id}"
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    code_predictor = hf_modeling.Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration._from_config(
        cfg.talker_config.code_predictor_config
    )
    code_predictor = load_module(
        code_predictor,
        model_path,
        prefix="talker.code_predictor.",
        dtype=torch.bfloat16,
        device=device,
        strict=False,
    )

    text_cfg = cfg.talker_config.text_config
    codec_embedding = nn.Embedding(text_cfg.vocab_size, text_cfg.hidden_size)
    emb_weights = load_weights_by_prefix(
        model_path, prefix="talker.model.codec_embedding."
    )
    codec_embedding.load_state_dict(emb_weights, strict=True, assign=True)
    codec_embedding = codec_embedding.to(device=device, dtype=torch.bfloat16)

    model = _LightweightTalkerShell(
        config=cfg.talker_config,
        code_predictor=code_predictor,
        codec_embedding=codec_embedding,
    )
    model.eval()
    return model


class _CodePredictorStreamingExecutor(Executor):
    """Stream talker hidden states through the RVQ predictor.

    Fans out ``codes`` to code2wav and ``summed_embeddings`` to talker_ar as feedback.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str | torch.device,
        *,
        max_batch_size: int = 16,
    ):
        self._model = model
        self._device = torch.device(device)
        self._max_batch_size = max(int(max_batch_size), 1)
        self._results: asyncio.Queue[StagePayload] = asyncio.Queue()
        self._aborted: set[str] = set()
        self._stream_queue: Any | None = None
        self._stream_fn: Any | None = None
        self._pending: asyncio.Queue[_PredictRequest] = asyncio.Queue()
        self._batch_loop_task: asyncio.Task[None] | None = None
        self._request_tasks: dict[str, asyncio.Task[None]] = {}
        self._forward_pool = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="code-predictor-fwd",
        )

    async def add_request(self, payload: StagePayload) -> None:
        request_id = payload.request_id
        if request_id in self._aborted:
            return
        if self._stream_queue is None:
            raise RuntimeError("Code predictor requires a stream queue")

        self._ensure_batch_loop()
        run_task = asyncio.create_task(self._run_request(payload))
        self._request_tasks[request_id] = run_task

        def _cleanup(_t: asyncio.Task, rid: str = request_id) -> None:
            self._request_tasks.pop(rid, None)
            self._aborted.discard(rid)

        run_task.add_done_callback(_cleanup)

    async def _run_request(self, payload: StagePayload) -> None:
        request_id = payload.request_id
        loop = asyncio.get_running_loop()
        chunk_count = 0

        while True:
            if request_id in self._aborted:
                break

            item = await self._stream_queue.get(request_id)
            if item is None:
                break

            talker_hidden = item.data.to(device=self._device)
            layer0_code = torch.tensor(
                item.metadata["codec_code"],
                dtype=torch.long,
                device=self._device,
            )
            req = _PredictRequest(
                request_id=request_id,
                talker_hidden=talker_hidden,
                layer0_code=layer0_code,
                result_future=loop.create_future(),
            )
            self._pending.put_nowait(req)

            try:
                output = await req.result_future
            except asyncio.CancelledError:
                break

            self._dispatch_outputs(request_id, output)
            chunk_count += 1

        if request_id not in self._aborted:
            payload.data = {"chunk_count": chunk_count}
            await self._results.put(payload)

    def _dispatch_outputs(
        self, request_id: str, output: dict[str, torch.Tensor]
    ) -> None:
        if request_id in self._aborted:
            return
        if self._stream_fn is None:
            return
        codes = output["codes"]
        summed_embeddings = output["summed_embeddings"]
        self._stream_fn(request_id, summed_embeddings, target_stage=TALKER_AR_STAGE)
        self._stream_fn(request_id, codes, target_stage=CODE2WAV_STAGE)

    def set_stream_fn(self, fn) -> None:
        """Set the streaming output callback. Sync, non-blocking."""
        self._stream_fn = fn

    async def get_result(self) -> StagePayload:
        return await self._results.get()

    async def abort(self, request_id: str) -> None:
        self._aborted.add(request_id)
        task = self._request_tasks.pop(request_id, None)
        if task is not None and not task.done():
            task.cancel()
        if self._stream_queue is not None:
            try:
                self._stream_queue.close(request_id)
            except KeyError:
                pass

    async def stop(self) -> None:
        if self._batch_loop_task is not None and not self._batch_loop_task.done():
            self._batch_loop_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._batch_loop_task
        self._batch_loop_task = None

        while not self._pending.empty():
            try:
                req = self._pending.get_nowait()
            except asyncio.QueueEmpty:
                break
            fut = req.result_future
            if fut is not None and not fut.done():
                fut.cancel()

        for task in list(self._request_tasks.values()):
            if not task.done():
                task.cancel()
        self._request_tasks.clear()

        self._forward_pool.shutdown(wait=False)

    def _ensure_batch_loop(self) -> None:
        if self._batch_loop_task is None or self._batch_loop_task.done():
            self._batch_loop_task = asyncio.create_task(
                run_batch_loop(self._batch_predict_step, "CodePredictor")
            )

    async def _batch_predict_step(self) -> None:
        loop = asyncio.get_running_loop()

        first = await self._pending.get()
        batch: list[_PredictRequest] = [first]

        while not self._pending.empty() and len(batch) < self._max_batch_size:
            try:
                batch.append(self._pending.get_nowait())
            except asyncio.QueueEmpty:
                break

        live: list[_PredictRequest] = []
        for r in batch:
            if r.request_id in self._aborted:
                fut = r.result_future
                if fut is not None and not fut.done():
                    fut.cancel()
            else:
                live.append(r)
        if not live:
            return

        try:
            outputs = await loop.run_in_executor(
                self._forward_pool,
                self._run_batch,
                live,
            )
            for req, output in zip(live, outputs):
                fut = req.result_future
                if req.request_id in self._aborted:
                    if fut is not None and not fut.done():
                        fut.cancel()
                elif fut is not None and not fut.done():
                    fut.set_result(output)
        except Exception as exc:
            for req in live:
                fut = req.result_future
                if fut is not None and not fut.done():
                    fut.set_exception(exc)
            raise

    @torch.no_grad()
    def _run_batch(self, batch: list[_PredictRequest]) -> list[dict[str, torch.Tensor]]:
        if self._device.type == "cuda":
            torch.cuda.set_device(self._device)

        talker_hidden = torch.stack([r.talker_hidden for r in batch], dim=0)
        layer0_code = torch.stack([r.layer0_code for r in batch], dim=0)

        out = self._model(talker_hidden=talker_hidden, layer0_code=layer0_code)
        codes = out["codes"]
        summed_embeddings = out["summed_embeddings"]

        return [
            {
                "codes": codes[i],
                "summed_embeddings": summed_embeddings[i],
            }
            for i in range(len(batch))
        ]


def create_code_predictor_executor(
    model_path: str,
    *,
    gpu_id: int = 0,
    max_batch_size: int = 16,
) -> Executor:
    """Create streaming Code Predictor executor."""
    device = f"cuda:{gpu_id}"
    model = _load_talker_model(model_path, gpu_id=gpu_id)
    wrapper = _CodePredictorWrapper(model)
    return _CodePredictorStreamingExecutor(
        model=wrapper, device=device, max_batch_size=max_batch_size
    )


def create_code_predictor_executor_from_config(
    model_path: str,
    *,
    gpu_id: int = 0,
    server_args_overrides: dict[str, Any] | None = None,
) -> Executor:
    """Create Code Predictor executor from config args."""
    max_batch_size = 16
    if server_args_overrides:
        max_batch_size = int(
            server_args_overrides.get("code_predictor_max_batch_size", max_batch_size)
        )
    return create_code_predictor_executor(
        model_path=model_path,
        gpu_id=gpu_id,
        max_batch_size=max_batch_size,
    )
