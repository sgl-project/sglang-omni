# SPDX-License-Identifier: Apache-2.0
"""Streaming Talker executor with prompt-aware prefill and gated feedback."""

from __future__ import annotations

import asyncio
import collections
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from sglang_omni.engines.omni.types import SchedulerStatus
from sglang_omni.executors.interface import Executor
from sglang_omni.models.qwen3_omni.components.talker_input import build_prefill_input
from sglang_omni.models.qwen3_omni.io import PipelineState
from sglang_omni.models.qwen3_omni.pipeline.engine_io import build_sglang_talker_request
from sglang_omni.models.qwen3_omni.pipeline.next_stage import (
    CODE_PREDICTOR_STAGE,
    THINKER_STAGE,
)
from sglang_omni.models.weight_loader import resolve_model_path
from sglang_omni.pipeline.stage.stream_queue import (
    StreamItem,
    StreamQueue,
    StreamSignal,
)
from sglang_omni.proto import StagePayload

logger = logging.getLogger(__name__)

_THINKER_EMBED_CANDIDATE_KEYS = (
    "thinker.model.embed_tokens.weight",
    "model.embed_tokens.weight",
)


def _validate_talker_context(
    *,
    request_id: str,
    prefill_tokens: int,
    max_new_tokens: int,
    max_seq_len: int | None,
) -> None:
    if max_seq_len is None:
        return
    total_tokens = int(prefill_tokens) + int(max_new_tokens)
    if prefill_tokens >= max_seq_len:
        raise ValueError(
            f"Talker prefill for request {request_id} has {prefill_tokens} tokens, "
            f"reaching or exceeding talker context length {max_seq_len}."
        )
    if total_tokens >= max_seq_len:
        raise ValueError(
            f"Talker request {request_id} needs {total_tokens} total tokens "
            f"({prefill_tokens} prefill + {int(max_new_tokens)} generated), reaching "
            f"or exceeding talker context length {max_seq_len}."
        )


@dataclass
class _TalkerRequestState:
    payload: StagePayload
    pending_feedbacks: collections.deque[torch.Tensor] = field(
        default_factory=collections.deque
    )
    thinker_chunks_done: bool = False
    feedback_chunk_id: int = 0
    bridge_task: asyncio.Task | None = None


def _read_rows_from_safetensor(
    file_path: Path, tensor_name: str, row_ids: list[int]
) -> torch.Tensor:
    with safe_open(str(file_path), framework="pt", device="cpu") as handle:
        if tensor_name not in handle.keys():
            raise KeyError(f"{tensor_name} not found in {file_path}")
        try:
            tensor_slice = handle.get_slice(tensor_name)
            rows = [tensor_slice[row_id] for row_id in row_ids]
            return torch.stack(rows, dim=0)
        except Exception:
            tensor = handle.get_tensor(tensor_name)
            return tensor[row_ids]


def _load_thinker_embedding_rows(model_path: str, row_ids: list[int]) -> torch.Tensor:
    model_dir = Path(model_path)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    for index_path in model_dir.glob("*.safetensors.index.json"):
        index_data = json.loads(index_path.read_text())
        weight_map = index_data.get("weight_map", {})
        for tensor_name in _THINKER_EMBED_CANDIDATE_KEYS:
            shard_name = weight_map.get(tensor_name)
            if shard_name is None:
                continue
            shard_path = model_dir / shard_name
            return _read_rows_from_safetensor(shard_path, tensor_name, row_ids)

    for shard_path in model_dir.glob("*.safetensors"):
        try:
            with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
                for tensor_name in _THINKER_EMBED_CANDIDATE_KEYS:
                    if tensor_name in handle.keys():
                        return _read_rows_from_safetensor(
                            shard_path, tensor_name, row_ids
                        )
        except Exception:
            continue

    raise KeyError(
        "Unable to locate thinker embedding weights in "
        f"{model_path} (tried {list(_THINKER_EMBED_CANDIDATE_KEYS)})"
    )


class TalkerStreamingExecutor(Executor):
    """Run Talker with prompt-aware streaming prefill and gated feedback."""

    def __init__(
        self,
        *,
        engine,
        model_path: str,
        tokenizer: Any,
        codec_vocab_size: int,
        tts_bos_token_id: int,
        tts_eos_token_id: int,
        tts_pad_token_id: int,
        im_start_token_id: int,
        im_end_token_id: int,
        system_token_id: int,
        user_token_id: int,
        assistant_token_id: int,
        accept_hidden_layer: int,
        audio_token_id: int | None,
        image_token_id: int | None,
        video_token_id: int | None,
        speaker_map: dict[str, int] | None,
        enqueue_fn_holder: dict[str, Any],
        max_seq_len: int | None = None,
        thinker_config: Any = None,
    ):
        self._engine = engine
        self._model_path = model_path
        try:
            self._resolved_model_path = str(
                resolve_model_path(model_path, local_files_only=False)
            )
        except Exception:
            self._resolved_model_path = model_path
        self._tokenizer = tokenizer
        self._codec_vocab_size = codec_vocab_size
        self._tts_bos_token_id = int(tts_bos_token_id)
        self._tts_eos_token_id = int(tts_eos_token_id)
        self._tts_pad_token_id = int(tts_pad_token_id)
        self._im_start_token_id = int(im_start_token_id)
        self._im_end_token_id = int(im_end_token_id)
        self._system_token_id = int(system_token_id)
        self._user_token_id = int(user_token_id)
        self._assistant_token_id = int(assistant_token_id)
        self._accept_hidden_layer = int(accept_hidden_layer)
        self._audio_token_id = (
            int(audio_token_id) if audio_token_id is not None else None
        )
        self._image_token_id = (
            int(image_token_id) if image_token_id is not None else None
        )
        self._video_token_id = (
            int(video_token_id) if video_token_id is not None else None
        )
        self._speaker_map = {
            str(k).lower(): int(v) for k, v in (speaker_map or {}).items()
        }
        self._enqueue_fn_holder = enqueue_fn_holder
        self._max_seq_len = int(max_seq_len) if max_seq_len is not None else None
        self._thinker_config = thinker_config

        self._talker_model = engine.model_runner._inner_model
        self._device = next(self._talker_model.parameters()).device
        self._dtype = next(self._talker_model.parameters()).dtype

        self._engine_feedback_mailbox = StreamQueue(max_pending=4096)
        if hasattr(self._engine, "_feedback_mailbox"):
            self._engine._feedback_mailbox = self._engine_feedback_mailbox

        self._stream_queue: Any | None = None  # Set by compiler
        self._results: asyncio.Queue[StagePayload] = asyncio.Queue()
        self._done: asyncio.Queue[str] = asyncio.Queue()
        self._tasks: dict[str, asyncio.Task[StagePayload]] = {}
        self._payloads: dict[str, StagePayload] = {}
        self._states: dict[str, _TalkerRequestState] = {}
        self._aborted: set[str] = set()
        self._tts_special_cache: (
            tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None
        ) = None
        self._thinker_embed_cache: dict[int, torch.Tensor] = {}

    async def start(self) -> None:
        start = getattr(self._engine, "start", None)
        if callable(start):
            await start()

    async def stop(self) -> None:
        stop = getattr(self._engine, "stop", None)
        if callable(stop):
            await stop()

    def set_stream_fn(self, fn) -> None:
        self._enqueue_fn_holder["fn"] = fn

    def set_feedback_mailbox(self, mailbox: Any) -> None:
        # Receiver wiring uses one shared inbound stream queue for thinker + feedback.
        self._stream_queue = mailbox
        if hasattr(self._engine, "_feedback_mailbox"):
            self._engine._feedback_mailbox = self._engine_feedback_mailbox
        scheduler = getattr(self._engine, "scheduler", None)
        iter_ctrl = getattr(scheduler, "iteration_controller", None)
        if iter_ctrl is not None and hasattr(iter_ctrl, "_feedback_enabled"):
            iter_ctrl._feedback_enabled = True

    async def add_request(self, payload: StagePayload) -> None:
        request_id = payload.request_id
        if request_id in self._aborted:
            return
        if self._stream_queue is None:
            raise RuntimeError("Talker executor requires a stream queue")

        self._payloads[request_id] = payload
        self._engine_feedback_mailbox.open(request_id)

        min_thinker_chunks = 1
        thinker_chunks, pending_feedbacks, thinker_done = (
            await self._collect_initial_chunks(
                request_id,
                min_thinker_chunks=min_thinker_chunks,
            )
        )
        if request_id in self._aborted:
            return

        engine_input = self._build_initial_request(
            payload=payload,
            request_id=request_id,
            thinker_chunks=thinker_chunks,
            thinker_done=thinker_done,
        )
        if engine_input is None:
            # Thinker produced nothing — complete immediately with empty output
            payload.data = {"chunk_count": 0}

            async def _noop_result():
                return payload

            task = asyncio.create_task(_noop_result())
            self._tasks[request_id] = task
            task.add_done_callback(lambda _t: self._done.put_nowait(request_id))
            self._engine_feedback_mailbox.close(request_id)
            return
        await self._engine.add_request(request_id, engine_input)

        state = _TalkerRequestState(
            payload=payload,
            pending_feedbacks=pending_feedbacks,
            thinker_chunks_done=thinker_done,
        )
        self._states[request_id] = state
        bridge_task = asyncio.create_task(self._bridge_inbound(request_id))
        state.bridge_task = bridge_task

        result_task = asyncio.create_task(self._await_result(payload, bridge_task))
        self._tasks[request_id] = result_task
        result_task.add_done_callback(lambda _t: self._done.put_nowait(request_id))

    async def get_result(self) -> StagePayload:
        while True:
            request_id = await self._done.get()
            if request_id in self._aborted:
                self._tasks.pop(request_id, None)
                self._payloads.pop(request_id, None)
                self._states.pop(request_id, None)
                continue
            task = self._tasks.pop(request_id, None)
            if task is None:
                continue
            self._payloads.pop(request_id, None)
            self._states.pop(request_id, None)
            try:
                return await task
            except Exception as e:
                e.request_id = request_id  # Allow worker to map error to request
                raise

    async def abort(self, request_id: str) -> None:
        self._aborted.add(request_id)
        state = self._states.pop(request_id, None)
        if state is not None and state.bridge_task is not None:
            state.bridge_task.cancel()
        self._payloads.pop(request_id, None)
        task = self._tasks.pop(request_id, None)
        if task is not None:
            task.cancel()
        self._engine_feedback_mailbox.close(request_id)
        await self._engine.abort(request_id)

    async def stream(self, request_id: str):
        stream_fn = getattr(self._engine, "stream", None)
        if not callable(stream_fn):
            return
        async for item in stream_fn(request_id):
            if request_id in self._aborted:
                break
            yield item

    async def _await_result(
        self, payload: StagePayload, bridge_task: asyncio.Task | None
    ) -> StagePayload:
        request_id = payload.request_id
        try:
            result = await self._engine.get_result(request_id)
        finally:
            if bridge_task is not None:
                bridge_task.cancel()
                try:
                    await bridge_task
                except asyncio.CancelledError:
                    pass
            self._engine_feedback_mailbox.close(request_id)

        # EOS is signaled by the Worker after executor completes

        payload.data = {"codec_codes": getattr(result, "output_ids", [])}
        return payload

    async def _collect_initial_chunks(
        self,
        request_id: str,
        *,
        min_thinker_chunks: int,
        thinker_chunks: list[StreamItem] | None = None,
        pending_feedbacks: collections.deque[torch.Tensor] | None = None,
        thinker_done: bool = False,
    ) -> tuple[list[StreamItem], collections.deque[torch.Tensor], bool]:
        thinker_chunks = thinker_chunks or []
        pending_feedbacks = (
            pending_feedbacks if pending_feedbacks is not None else collections.deque()
        )

        while len(thinker_chunks) < min_thinker_chunks and not thinker_done:
            item = await self._stream_queue.get_with_source(request_id)
            thinker_done = self._route_inbound_item(
                request_id,
                item,
                thinker_chunks=thinker_chunks,
                pending_feedbacks=pending_feedbacks,
                update_request=False,
            )

        return thinker_chunks, pending_feedbacks, thinker_done

    def _build_initial_request(
        self,
        *,
        payload: StagePayload,
        request_id: str,
        thinker_chunks: list[StreamItem],
        thinker_done: bool,
    ):
        sampling_cfg = self._resolve_talker_sampling_config(payload)

        if not thinker_chunks:
            # Thinker produced no output (e.g. max_new_tokens=0 or immediate abort).
            # Return a minimal payload so the request completes gracefully.
            return None

        return self._build_prompt_prefill_request(
            payload=payload,
            request_id=request_id,
            thinker_chunks=thinker_chunks,
            thinker_done=thinker_done,
            sampling_cfg=sampling_cfg,
        )

    def _build_prompt_prefill_request(
        self,
        *,
        payload: StagePayload,
        request_id: str,
        thinker_chunks: list[StreamItem],
        thinker_done: bool,
        sampling_cfg: dict[str, Any],
    ):
        if not thinker_chunks:
            raise ValueError("prompt prefill requires at least one thinker chunk")

        prompt_ids, prompt_embed, prompt_hidden = self._reconstruct_prompt_states(
            payload
        )
        assistant_token_ids = self._extract_thinker_chunk_token_ids(thinker_chunks)
        assistant_embed = self._load_prompt_token_embeddings(assistant_token_ids).to(
            device=self._device, dtype=self._dtype
        )
        assistant_hidden = torch.stack(
            [self._chunk_layer_hidden_or_embed(chunk) for chunk in thinker_chunks],
            dim=0,
        ).to(device=self._device, dtype=self._dtype)

        thinker_input_ids = torch.cat([prompt_ids, assistant_token_ids], dim=0)
        thinker_embed = torch.cat([prompt_embed, assistant_embed], dim=0)
        thinker_hidden = torch.cat([prompt_hidden, assistant_hidden], dim=0)
        multimodal_mask = self._build_multimodal_mask(thinker_input_ids)

        tts_bos_embed, tts_eos_embed, tts_pad_embed = self._get_tts_special_embeds()
        speaker_id = self._resolve_speaker_id(payload)
        prefill = build_prefill_input(
            thinker_embed=thinker_embed,
            thinker_hidden=thinker_hidden,
            thinker_input_ids=thinker_input_ids,
            multimodal_mask=multimodal_mask,
            text_projection=self._talker_model.text_projection,
            hidden_projection=self._talker_model.hidden_projection,
            codec_embed_fn=self._talker_model.get_input_embeddings(),
            tts_bos_embed=tts_bos_embed,
            tts_eos_embed=tts_eos_embed,
            tts_pad_embed=tts_pad_embed,
            im_start_token_id=self._im_start_token_id,
            system_token_id=self._system_token_id,
            user_token_id=self._user_token_id,
            assistant_token_id=self._assistant_token_id,
            speaker_id=speaker_id,
            codec_nothink_id=self._talker_model.config.codec_nothink_id,
            codec_think_bos_id=self._talker_model.config.codec_think_bos_id,
            codec_think_eos_id=self._talker_model.config.codec_think_eos_id,
            codec_pad_id=self._talker_model.config.codec_pad_id,
            codec_bos_id=self._talker_model.config.codec_bos_id,
            tts_pad_token_id=self._tts_pad_token_id,
            include_assistant_eos=thinker_done,
            im_end_token_id=self._im_end_token_id,
        )
        sampling_cfg["max_new_tokens"] = self._resolve_effective_max_new_tokens(
            request_id,
            prefill["input_ids"],
            max_new_tokens=sampling_cfg["max_new_tokens"],
            explicit=bool(sampling_cfg["max_new_tokens_explicit"]),
        )
        self._log_assistant_component_summary(
            request_id=request_id,
            assistant_embed=assistant_embed,
            speaker_id=speaker_id,
            tts_bos_embed=tts_bos_embed,
            tts_pad_embed=tts_pad_embed,
            assembled_input_embeds=prefill["input_embeds"],
        )
        self._log_prefill_summary(
            request_id=request_id,
            prefill_type="prompt",
            input_ids=prefill["input_ids"],
            input_embeds=prefill["input_embeds"],
            source_token_ids=thinker_input_ids,
            assistant_token_ids=assistant_token_ids,
        )

        trailing_list = self._tensor_rows_to_list(prefill["trailing_text_hidden"])
        data = build_sglang_talker_request(
            torch.empty(0),
            tokenizer=self._tokenizer,
            codec_vocab_size=self._codec_vocab_size,
            max_new_tokens=sampling_cfg["max_new_tokens"],
            temperature=sampling_cfg["temperature"],
            top_k=sampling_cfg["top_k"],
            top_p=sampling_cfg["top_p"],
            repetition_penalty=sampling_cfg["repetition_penalty"],
            request_id=request_id,
            codec_eos_id=sampling_cfg["codec_eos_id"],
            suppress_tokens=sampling_cfg["suppress_tokens"],
            talker_input_embeds=prefill["input_embeds"],
            talker_input_ids=prefill["input_ids"],
            input_embeds_are_projected=True,
            trailing_text_hidden=trailing_list,
            tts_pad_embed=tts_pad_embed[0].detach().cpu(),
            thinker_chunks_done=thinker_done,
            thinker_config=self._thinker_config,
            talker_model_inputs=self._get_prompt_model_inputs(payload),
        )
        data.tts_eos_embed = tts_eos_embed[0].detach().cpu()
        return data

    def _resolve_effective_max_new_tokens(
        self,
        request_id: str,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        explicit: bool,
    ) -> int:
        if self._max_seq_len is None:
            return int(max_new_tokens)

        prefill_tokens = int(input_ids.numel())
        if explicit:
            _validate_talker_context(
                request_id=request_id,
                prefill_tokens=prefill_tokens,
                max_new_tokens=max_new_tokens,
                max_seq_len=self._max_seq_len,
            )
            return int(max_new_tokens)

        if prefill_tokens >= self._max_seq_len:
            _validate_talker_context(
                request_id=request_id,
                prefill_tokens=prefill_tokens,
                max_new_tokens=max_new_tokens,
                max_seq_len=self._max_seq_len,
            )

        # Reserve 1 token of slack so prefill + generated stays strictly below
        # max_seq_len (room for EOS / talker bos token handling).
        max_available = max(int(self._max_seq_len) - prefill_tokens - 1, 0)
        return min(int(max_new_tokens), max_available)

    def _resolve_speaker_id(self, payload: StagePayload) -> int:
        speaker_name = str(payload.request.params.get("speaker", "Ethan")).lower()
        speaker_id = self._speaker_map.get(speaker_name)
        if speaker_id is not None:
            return speaker_id
        if self._speaker_map:
            return next(iter(self._speaker_map.values()))
        return int(payload.request.params.get("speaker_id", 0))

    def _resolve_talker_sampling_config(self, payload: StagePayload) -> dict[str, Any]:
        params = payload.request.params
        codec_eos_id = int(getattr(self._talker_model.config, "codec_eos_token_id", -1))
        suppress_tokens = [
            token_id
            for token_id in range(
                max(self._codec_vocab_size - 1024, 0), self._codec_vocab_size
            )
            if token_id != codec_eos_id
        ]
        return {
            "max_new_tokens": int(params.get("talker_max_new_tokens", 4096)),
            "max_new_tokens_explicit": "talker_max_new_tokens" in params,
            "temperature": float(params.get("talker_temperature", 0.9)),
            "top_k": int(params.get("talker_top_k", 50)),
            "top_p": float(params.get("talker_top_p", 1.0)),
            "repetition_penalty": float(params.get("talker_repetition_penalty", 1.05)),
            "codec_eos_id": codec_eos_id if codec_eos_id >= 0 else None,
            "suppress_tokens": suppress_tokens,
        }

    def _get_tts_special_embeds(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._tts_special_cache is not None:
            return self._tts_special_cache

        special_ids = [
            self._tts_bos_token_id,
            self._tts_eos_token_id,
            self._tts_pad_token_id,
        ]
        thinker_rows = _load_thinker_embedding_rows(
            self._resolved_model_path, special_ids
        )
        thinker_rows = thinker_rows.to(device=self._device, dtype=self._dtype)

        projected = self._talker_model.text_projection(thinker_rows)
        tts_bos_embed, tts_eos_embed, tts_pad_embed = projected.chunk(3, dim=0)
        self._tts_special_cache = (tts_bos_embed, tts_eos_embed, tts_pad_embed)
        return self._tts_special_cache

    def _reconstruct_prompt_states(
        self,
        payload: StagePayload,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = PipelineState.from_dict(payload.data)
        prompt = state.prompt or {}
        prompt_input_ids = prompt.get("input_ids")
        if not isinstance(prompt_input_ids, torch.Tensor):
            raise TypeError("prompt.input_ids missing for talker prompt prefill")
        if prompt_input_ids.dim() == 2:
            if prompt_input_ids.shape[0] != 1:
                raise ValueError("talker prompt prefill only supports batch size 1")
            prompt_input_ids = prompt_input_ids[0]
        prompt_ids = prompt_input_ids.to(dtype=torch.long).cpu()

        prompt_embed = self._load_prompt_token_embeddings(prompt_ids)
        prompt_hidden = prompt_embed.clone()

        thinker_inputs = state.thinker_inputs or {}
        model_inputs = thinker_inputs.get("model_inputs", thinker_inputs)
        if isinstance(model_inputs, dict):
            self._merge_prompt_multimodal_embeddings(
                prompt_ids,
                prompt_embed,
                prompt_hidden,
                model_inputs,
            )

        return prompt_ids, prompt_embed, prompt_hidden

    def _log_prefill_summary(
        self,
        *,
        request_id: str,
        prefill_type: str,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        source_token_ids: torch.Tensor | None = None,
        assistant_token_ids: torch.Tensor | None = None,
    ) -> None:
        if not logger.isEnabledFor(logging.DEBUG):
            return
        ids = input_ids.detach().cpu().view(-1)
        embeds = input_embeds.detach().float().cpu()
        source_ids = (
            source_token_ids.detach().cpu().view(-1).tolist()
            if isinstance(source_token_ids, torch.Tensor)
            else None
        )
        assistant_ids = (
            assistant_token_ids.detach().cpu().view(-1).tolist()
            if isinstance(assistant_token_ids, torch.Tensor)
            else None
        )
        logger.debug(
            "Talker %s prefill summary rid=%s seq_len=%s ids=%s "
            "first_row_head=%s last_row_head=%s embed_sum=%.6f embed_norm=%.6f "
            "source_ids=%s assistant_ids=%s",
            prefill_type,
            request_id,
            int(ids.numel()),
            ids.tolist(),
            [round(float(v), 6) for v in embeds[0, :8].tolist()],
            [round(float(v), 6) for v in embeds[-1, :8].tolist()],
            float(embeds.sum().item()),
            float(embeds.norm().item()),
            source_ids,
            assistant_ids,
        )

    def _log_assistant_component_summary(
        self,
        *,
        request_id: str,
        assistant_embed: torch.Tensor,
        speaker_id: int,
        tts_bos_embed: torch.Tensor,
        tts_pad_embed: torch.Tensor,
        assembled_input_embeds: torch.Tensor,
    ) -> None:
        if not logger.isEnabledFor(logging.DEBUG):
            return
        projected = self._talker_model.text_projection(assistant_embed.detach())
        codec_special_ids = torch.tensor(
            [
                self._talker_model.config.codec_nothink_id,
                self._talker_model.config.codec_think_bos_id,
                self._talker_model.config.codec_think_eos_id,
                speaker_id,
                self._talker_model.config.codec_pad_id,
                self._talker_model.config.codec_bos_id,
            ],
            device=assistant_embed.device,
            dtype=torch.long,
        )
        codec_special_embeds = self._talker_model.get_input_embeddings()(
            codec_special_ids
        ).detach()
        logger.debug(
            "Talker assistant components rid=%s speaker_id=%s codec_special_ids=%s "
            "projected_last_head=%s codec_bos_head=%s codec_pad_head=%s "
            "tts_bos_head=%s tts_pad_head=%s assembled_last_head=%s "
            "codec_bos_norm=%.6f codec_pad_norm=%.6f assembled_last_norm=%.6f",
            request_id,
            int(speaker_id),
            codec_special_ids.detach().cpu().tolist(),
            [round(float(v), 6) for v in projected[-1, :8].float().cpu().tolist()],
            [
                round(float(v), 6)
                for v in codec_special_embeds[-1, :8].float().cpu().tolist()
            ],
            [
                round(float(v), 6)
                for v in codec_special_embeds[-2, :8].float().cpu().tolist()
            ],
            [round(float(v), 6) for v in tts_bos_embed[0, :8].float().cpu().tolist()],
            [round(float(v), 6) for v in tts_pad_embed[0, :8].float().cpu().tolist()],
            [
                round(float(v), 6)
                for v in assembled_input_embeds[-1, :8].float().cpu().tolist()
            ],
            float(codec_special_embeds[-1].float().norm().item()),
            float(codec_special_embeds[-2].float().norm().item()),
            float(assembled_input_embeds[-1].float().norm().item()),
        )

    def _get_prompt_model_inputs(self, payload: StagePayload) -> dict[str, Any]:
        state = PipelineState.from_dict(payload.data)
        thinker_inputs = state.thinker_inputs or {}
        model_inputs = thinker_inputs.get("model_inputs", thinker_inputs)
        return model_inputs if isinstance(model_inputs, dict) else {}

    def _load_prompt_token_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        token_ids = token_ids.to(dtype=torch.long).view(-1).cpu()
        unique_ids, inverse = torch.unique(token_ids, sorted=False, return_inverse=True)
        missing_ids = [
            int(token_id)
            for token_id in unique_ids.tolist()
            if int(token_id) not in self._thinker_embed_cache
        ]
        if missing_ids:
            loaded_rows = _load_thinker_embedding_rows(
                self._resolved_model_path, missing_ids
            ).to(device=self._device, dtype=self._dtype)
            for token_id, row in zip(missing_ids, loaded_rows):
                self._thinker_embed_cache[int(token_id)] = row.detach().clone()

        unique_rows = torch.stack(
            [
                self._thinker_embed_cache[int(token_id)]
                for token_id in unique_ids.tolist()
            ],
            dim=0,
        )
        gathered = unique_rows.index_select(0, inverse.to(device=unique_rows.device))
        return gathered.view(token_ids.shape[0], unique_rows.shape[-1])

    def _merge_prompt_multimodal_embeddings(
        self,
        prompt_ids: torch.Tensor,
        prompt_embed: torch.Tensor,
        prompt_hidden: torch.Tensor,
        model_inputs: dict[str, Any],
    ) -> None:
        self._apply_prompt_modality(
            prompt_ids,
            prompt_embed,
            prompt_hidden,
            token_id=self._audio_token_id,
            features=model_inputs.get("audio_embeds"),
            modality="audio",
        )
        self._apply_prompt_modality(
            prompt_ids,
            prompt_embed,
            prompt_hidden,
            token_id=self._image_token_id,
            features=model_inputs.get("image_embeds"),
            modality="image",
        )
        self._apply_prompt_modality(
            prompt_ids,
            prompt_embed,
            prompt_hidden,
            token_id=self._video_token_id,
            features=model_inputs.get("video_embeds"),
            modality="video",
        )

    def _apply_prompt_modality(
        self,
        prompt_ids: torch.Tensor,
        prompt_embed: torch.Tensor,
        prompt_hidden: torch.Tensor,
        *,
        token_id: int | None,
        features: Any,
        modality: str,
    ) -> None:
        if token_id is None:
            return
        feature_tensor = self._coerce_feature_tensor(features)
        if feature_tensor is None:
            return
        mask = prompt_ids == int(token_id)
        if not mask.any():
            return
        if feature_tensor.shape[0] != int(mask.sum().item()):
            raise ValueError(
                f"{modality} placeholder count mismatch: "
                f"tokens={int(mask.sum().item())} embeds={feature_tensor.shape[0]}"
            )
        feature_tensor = feature_tensor.to(device=self._device, dtype=self._dtype)
        prompt_embed[mask] = feature_tensor
        # Zero out prompt_hidden at multimodal positions.  build_user_part
        # applies hidden_projection to these positions, which expects actual
        # thinker layer-24 hidden states.  We don't have those for the prompt
        # (only streaming decode hidden states arrive later).  Leaving token
        # embeddings here would feed untransformed embeddings into a linear
        # layer trained on layer-24 outputs, producing garbage.  Zeros yield
        # a zero vector after projection — a safer neutral value.
        prompt_hidden[mask] = 0.0

    def _coerce_feature_tensor(self, value: Any) -> torch.Tensor | None:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            tensor = value
        elif isinstance(value, (list, tuple)):
            tensors = [item for item in value if isinstance(item, torch.Tensor)]
            if not tensors:
                return None
            tensor = torch.cat(tensors, dim=0)
        else:
            return None
        if tensor.dim() == 3 and tensor.shape[0] == 1:
            tensor = tensor[0]
        elif tensor.dim() > 2:
            tensor = tensor.reshape(-1, tensor.shape[-1])
        return tensor

    def _extract_thinker_chunk_token_ids(
        self, thinker_chunks: list[StreamItem]
    ) -> torch.Tensor:
        token_ids: list[int] = []
        for chunk in thinker_chunks:
            metadata = chunk.metadata if isinstance(chunk.metadata, dict) else {}
            token_id = metadata.get("token_id")
            if token_id is None:
                raise ValueError("thinker chunk missing token_id metadata")
            token_ids.append(int(token_id))
        return torch.tensor(token_ids, dtype=torch.long)

    def _chunk_layer_hidden_or_embed(self, chunk: StreamItem) -> torch.Tensor:
        metadata = chunk.metadata if isinstance(chunk.metadata, dict) else {}
        layer_hidden = metadata.get("layer_hidden")
        if isinstance(layer_hidden, torch.Tensor):
            return layer_hidden
        return chunk.data

    def _build_multimodal_mask(self, token_ids: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros(token_ids.shape[0], dtype=torch.bool, device=self._device)
        for token_id in (
            self._audio_token_id,
            self._image_token_id,
            self._video_token_id,
        ):
            if token_id is not None:
                mask |= token_ids.to(device=self._device) == int(token_id)
        return mask

    def _tensor_rows_to_list(self, tensor: torch.Tensor | None) -> list[torch.Tensor]:
        if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
            return []
        return [row.detach().cpu() for row in tensor]

    def _project_assistant_chunk(self, chunk: StreamItem) -> torch.Tensor:
        metadata = chunk.metadata if isinstance(chunk.metadata, dict) else {}
        token_id = metadata.get("token_id")
        if token_id is not None:
            chunk_tensor = self._load_prompt_token_embeddings(
                torch.tensor([int(token_id)], dtype=torch.long)
            ).to(device=self._device, dtype=self._dtype)
        else:
            chunk_tensor = chunk.data.to(
                device=self._device, dtype=self._dtype
            ).unsqueeze(0)
        projected = self._talker_model.text_projection(chunk_tensor)
        return projected[0].detach()

    async def _bridge_inbound(self, request_id: str) -> None:
        pending_feedbacks = self._states[request_id].pending_feedbacks
        try:
            while request_id not in self._aborted:
                self._flush_feedback(request_id)
                try:
                    item = await asyncio.wait_for(
                        self._stream_queue.get_with_source(request_id),
                        timeout=0.01,
                    )
                except asyncio.TimeoutError:
                    continue

                thinker_done = self._route_inbound_item(
                    request_id,
                    item,
                    thinker_chunks=None,
                    pending_feedbacks=pending_feedbacks,
                    update_request=True,
                )
                if thinker_done:
                    self._states[request_id].thinker_chunks_done = True
                self._flush_feedback(request_id)
        except asyncio.CancelledError:
            raise

    def _route_inbound_item(
        self,
        request_id: str,
        item: StreamItem | StreamSignal,
        *,
        thinker_chunks: list[StreamItem] | None,
        pending_feedbacks: list[torch.Tensor],
        update_request: bool,
    ) -> bool:
        if isinstance(item, StreamSignal):
            if item.error is not None:
                raise item.error
            if item.is_done:
                # from_stage=None means the queue was closed (abort/cleanup)
                # — treat as terminal to prevent infinite loops
                is_terminal = item.from_stage is None or self._is_thinker_source(item)
                if is_terminal:
                    if update_request:
                        self._mark_thinker_done(request_id)
                    return True
            return False

        if self._is_thinker_source(item):
            if thinker_chunks is not None:
                thinker_chunks.append(item)
            if update_request:
                self._append_trailing_chunk(request_id, item)
            return False

        pending_feedbacks.append(item.data.detach().cpu())
        return False

    def _is_thinker_source(self, item: StreamItem | StreamSignal) -> bool:
        from_stage = getattr(item, "from_stage", None)
        if from_stage == THINKER_STAGE:
            return True
        metadata = getattr(item, "metadata", None)
        return isinstance(metadata, dict) and "token_id" in metadata

    def _append_trailing_chunk(self, request_id: str, chunk: StreamItem) -> None:
        request = self._engine.scheduler.requests.get(request_id)
        if request is None:
            return
        # Skip <|im_end|> token — HF's thinker_embed never contains a hidden
        # state for the EOS token, so including it creates an off-by-one.
        metadata = chunk.metadata if isinstance(chunk.metadata, dict) else {}
        token_id = metadata.get("token_id")
        if token_id is not None and int(token_id) == self._im_end_token_id:
            return
        trailing = getattr(request.data, "trailing_text_hidden", None)
        if not isinstance(trailing, list):
            return
        projected = self._project_assistant_chunk(chunk).cpu()
        trailing.append(projected)

    def _mark_thinker_done(self, request_id: str) -> None:
        request = self._engine.scheduler.requests.get(request_id)
        if request is None:
            return
        if bool(getattr(request.data, "thinker_chunks_done", False)):
            return
        request.data.thinker_chunks_done = True
        trailing = getattr(request.data, "trailing_text_hidden", None)
        tts_eos_embed = getattr(request.data, "tts_eos_embed", None)
        if isinstance(trailing, list) and isinstance(tts_eos_embed, torch.Tensor):
            eos_embed = tts_eos_embed.detach().cpu()
            trailing.append(eos_embed)

    def _flush_feedback(self, request_id: str) -> None:
        state = self._states.get(request_id)
        if state is None or not state.pending_feedbacks:
            return

        request = self._engine.scheduler.requests.get(request_id)
        if request is None or request.status != SchedulerStatus.WAITING_FEEDBACK:
            return

        trailing = getattr(request.data, "trailing_text_hidden", None)
        trailing_len = 0
        if isinstance(trailing, list):
            trailing_len = len(trailing)
        elif isinstance(trailing, torch.Tensor):
            trailing_len = trailing.shape[0]

        step_index = max(int(getattr(request.data, "generation_steps", 0)) - 1, 0)
        thinker_done = bool(getattr(request.data, "thinker_chunks_done", True))
        if not thinker_done and step_index >= trailing_len:
            return

        feedback = state.pending_feedbacks.popleft()
        self._engine_feedback_mailbox.put(
            request_id,
            StreamItem(
                chunk_id=state.feedback_chunk_id,
                data=feedback,
                from_stage=CODE_PREDICTOR_STAGE,
            ),
        )
        state.feedback_chunk_id += 1
