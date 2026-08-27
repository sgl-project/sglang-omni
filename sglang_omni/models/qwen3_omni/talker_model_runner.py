# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni talker runner with FIFO text/feedback decode handoff."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.prefill_inputs import (
    OmniPrefillInputs,
    attach_omni_prefill_inputs,
)
from sglang_omni.scheduling.messages import OutgoingMessage


class QwenTalkerModelRunner(ModelRunner):
    def __init__(
        self,
        tp_worker: Any,
        output_processor: Any,
        outbox: Any,
        *,
        code2wav_target: str = "code2wav",
        feedback_enabled: bool = True,
    ) -> None:
        super().__init__(tp_worker, output_processor)
        self._outbox = outbox
        self._code2wav_target = code2wav_target
        self._feedback_enabled = bool(feedback_enabled)
        if self._feedback_enabled:
            self._check_feedback_slots_cover_pool()

    def _check_feedback_slots_cover_pool(self) -> None:
        """Validate feedback slots against the request pool's actual row count."""
        slots = self.model._feedback_slots
        pool = self.tp_worker.model_runner.req_to_token_pool
        required = pool.req_to_token.shape[0]
        if slots.shape[0] < required:
            raise RuntimeError(
                "Talker feedback slots are too small for the request pool: "
                f"_feedback_slots has {slots.shape[0]} rows but req_to_token_pool "
                f"of size {pool.size} allocates req_pool_idx in [1, {pool.size}], "
                f"needing {required} rows"
            )

    def execute(self, scheduler_output: Any):
        return super().execute(scheduler_output)

    def before_prefill(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        del schedule_batch
        composed = self._compose_prefill_embeds(forward_batch, requests)
        if composed is None:
            return
        input_embeds, input_embeds_are_projected = composed
        attach_omni_prefill_inputs(
            forward_batch,
            OmniPrefillInputs(
                input_embeds=input_embeds,
                input_embeds_are_projected=input_embeds_are_projected,
            ),
        )

    def before_decode(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
        *,
        is_lookahead: bool = False,
    ) -> None:
        del is_lookahead
        del schedule_batch
        if not self._feedback_enabled:
            return

        if not self._requests_ready_for_decode(requests):
            raise RuntimeError(
                "Talker decode reached model runner without ready feedback/text input"
            )

        self.model.prepare_decode_buffers(requests)
        self._write_feedback_buffers(
            requests, self._batch_pool_indices(forward_batch, len(requests))
        )

    def post_prefill(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        # Note (Xuesong): Do not clear data.prefill_input_embeds: decode retract may requeue
        # the Req for another prefill pass and Req.input_embeds is None.
        if not self._feedback_enabled:
            return

        if result.next_token_ids is None:
            return
        layer0_codes = result.next_token_ids
        if layer0_codes.ndim == 1:
            layer0_codes = layer0_codes.unsqueeze(1)
        talker_hidden = result.logits_output.hidden_states
        if isinstance(talker_hidden, torch.Tensor) and talker_hidden.ndim == 2:
            talker_hidden = talker_hidden.unsqueeze(1)
        self.model.code_predictor_forward(layer0_codes, talker_hidden)
        self._stage_token_ids(result, result.next_token_ids)
        self._emit_code_chunks_and_feedback(
            schedule_batch=schedule_batch,
            requests=requests,
            pool_indices=self._batch_pool_indices(forward_batch, len(requests)),
        )

    def post_decode(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        if not self._feedback_enabled:
            return

        batch_size = len(requests)
        result.next_token_ids = self.model._sampled_token_ids[:batch_size].clone()
        self._stage_token_ids(result, result.next_token_ids)
        self._emit_code_chunks_and_feedback(
            schedule_batch=schedule_batch,
            requests=requests,
            pool_indices=self._batch_pool_indices(forward_batch, len(requests)),
        )

    @staticmethod
    def _batch_pool_indices(forward_batch: Any, bs: int) -> torch.Tensor:
        """Return device-resident pool rows without a per-step host copy."""
        rows = forward_batch.req_pool_indices
        if int(rows.shape[0]) < bs:
            raise RuntimeError(
                "Talker forward batch carries fewer pool indices than requests: "
                f"{int(rows.shape[0])} rows for {bs} requests"
            )
        return rows[:bs]

    def _emit_code_chunks_and_feedback(
        self,
        *,
        schedule_batch: Any,
        requests: list,
        pool_indices: torch.Tensor,
    ) -> None:
        bs = len(requests)
        # Note (wenyao): preserve codes before the next graph replay overwrites them.
        codes_snap = self.model._output_codes[:bs].detach().clone()
        # Note (wenyao): same-stream ordering avoids a synchronization here.
        self.model._feedback_slots[pool_indices] = self.model._output_embeds[:bs]
        for idx, sched_req in enumerate(requests):
            req = schedule_batch.reqs[idx]
            code_chunk = codes_snap[idx]
            # Tell code2wav whether to forward audio chunks to the Coordinator.
            stage_payload = sched_req.data.stage_payload
            is_streaming = bool(
                stage_payload is not None
                and (stage_payload.request.params or {}).get("stream", False)
            )
            self._outbox.put(
                OutgoingMessage(
                    request_id=req.rid,
                    type="stream",
                    data=code_chunk,
                    target=self._code2wav_target,
                    metadata={"stream": is_streaming},
                )
            )
            sched_req.data.pending_feedback_count += 1
            # Note (wenyao): reading pool_indices on the host would synchronize.
            sched_req.data.feedback_slot_idx = req.req_pool_idx

    def snapshot_feedback_for_retract(self, req: Any) -> None:
        """Snapshot pending feedback before retract lets the pool reuse its row."""
        if not self._feedback_enabled:
            return
        data = req._omni_data
        if data is None:
            return
        slot_idx = data.feedback_slot_idx
        data.feedback_slot_idx = None
        pending = data.pending_feedback_count
        if pending <= 0:
            return
        if data.retracted_feedback_embed is not None:
            # Note (wenyao): re-prefill regenerates the newer row; keep the old one.
            return
        if slot_idx is None:
            raise RuntimeError(
                "Talker request has pending feedback but no recorded slot to "
                f"snapshot on retract (pending_feedback_count={pending}): the "
                "recorded slot is retired by every retract, so a second retract "
                "before a new emit has no row left to read and the freed pool "
                "index may already belong to another request"
            )
        data.retracted_feedback_embed = self.model._feedback_slots[slot_idx].clone()

    def sample_before_post_prefill(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> bool:
        del forward_batch, schedule_batch, requests
        return True

    def sample_before_post_decode(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> bool:
        del forward_batch, schedule_batch, requests
        return False

    def is_decode_batch_ready(self, schedule_batch: Any) -> bool:
        if not self._feedback_enabled or not schedule_batch.forward_mode.is_decode():
            return True
        return all(
            self._data_has_next_decode_input(req._omni_data)
            for req in schedule_batch.reqs
        )

    def _compose_prefill_embeds(
        self,
        forward_batch: Any,
        requests: list,
    ) -> tuple[torch.Tensor, bool] | None:
        """Assemble prefill rows and preserve whether they are projected."""
        projected_flags = [
            bool(req.data.input_embeds_are_projected) for req in requests
        ]
        has_tensor_requests = any(
            req.data.prefill_input_embeds is not None for req in requests
        )
        if not any(projected_flags) and not has_tensor_requests:
            return None

        has_projected_requests = any(projected_flags)
        if has_projected_requests and not all(projected_flags):
            raise RuntimeError(
                "Talker projected and unprojected prefill requests cannot be "
                "batched together"
            )

        parts: list[torch.Tensor] = []
        for sched_req in requests:
            req = sched_req.data.req
            prefix_len = len(req.prefix_indices)
            extend_len = int(req.extend_range.length)
            part = self._projected_prefill_slice(
                sched_req=sched_req,
                prefix_len=prefix_len,
                extend_len=extend_len,
                device=forward_batch.input_ids.device,
                take_next_decode_input_embed=self._take_next_decode_input_embed,
            )
            if part is not None and part.shape[0] > 0:
                parts.append(part)
        if not parts:
            return None
        input_embeds = torch.cat(parts, dim=0)

        expected_rows = int(forward_batch.input_ids.shape[0])
        if input_embeds.shape[0] != expected_rows:
            raise RuntimeError(
                "Talker prefill embeds must align with forward input_ids: "
                f"got {input_embeds.shape[0]} rows for {expected_rows} input ids"
            )
        return (
            input_embeds.to(
                device=forward_batch.input_ids.device,
                dtype=self.model.activation_dtype,
            ),
            has_projected_requests,
        )

    @staticmethod
    def _projected_prefill_slice(
        *,
        sched_req: Any,
        prefix_len: int,
        extend_len: int,
        device: torch.device,
        take_next_decode_input_embed: Callable[..., torch.Tensor | None],
    ) -> torch.Tensor | None:
        if extend_len <= 0:
            return None

        data = sched_req.data
        req = data.req
        end = prefix_len + extend_len
        tensor = data.prefill_input_embeds
        if tensor is not None:
            prompt_len = int(tensor.shape[0])
            dtype = tensor.dtype
            embed_device = tensor.device
            parts = QwenTalkerModelRunner._prefill_prompt_parts_from_tensor(
                tensor=tensor,
                prefix_len=prefix_len,
                end=end,
            )
        else:
            embeds = req.input_embeds
            if not embeds:
                return None
            prompt_len = len(embeds)
            dtype = torch.float32
            embed_device = device
            parts = QwenTalkerModelRunner._prefill_prompt_parts_from_list(
                embeds=embeds,
                prefix_len=prefix_len,
                end=end,
                device=device,
            )

        if end > prompt_len:
            generated = QwenTalkerModelRunner._generated_prefill_slice(
                sched_req=sched_req,
                gen_start=max(prefix_len, prompt_len) - prompt_len,
                gen_end=end - prompt_len,
                device=embed_device,
                dtype=dtype,
                take_next_decode_input_embed=take_next_decode_input_embed,
            )
            if generated is not None:
                parts.append(generated)

        if not parts:
            return None
        return torch.cat(parts, dim=0)

    @staticmethod
    def _prefill_prompt_parts_from_tensor(
        *,
        tensor: torch.Tensor,
        prefix_len: int,
        end: int,
    ) -> list[torch.Tensor]:
        prompt_len = int(tensor.shape[0])
        start = min(prefix_len, prompt_len)
        stop = min(end, prompt_len)
        return [tensor[start:stop]] if stop > start else []

    @staticmethod
    def _prefill_prompt_parts_from_list(
        *,
        embeds: list,
        prefix_len: int,
        end: int,
        device: torch.device,
    ) -> list[torch.Tensor]:
        prompt_len = len(embeds)
        start = min(prefix_len, prompt_len)
        stop = min(end, prompt_len)
        if stop <= start:
            return []
        return [
            torch.as_tensor(
                embeds[start:stop],
                device=device,
                dtype=torch.float32,
            )
        ]

    @staticmethod
    def _generated_prefill_slice(
        *,
        sched_req: Any,
        gen_start: int,
        gen_end: int,
        device: torch.device,
        dtype: torch.dtype,
        take_next_decode_input_embed: Callable[..., torch.Tensor | None],
    ) -> torch.Tensor | None:
        if gen_end <= gen_start:
            return None

        data = sched_req.data
        history = QwenTalkerModelRunner._decode_input_history(data)
        while len(history) < gen_end:
            combined = take_next_decode_input_embed(
                sched_req=sched_req,
                device=device,
                dtype=dtype,
            )
            if combined is None:
                raise RuntimeError(
                    "Cannot replay retracted talker decode tokens: missing "
                    "feedback/text input embeds for generated-token prefill "
                    "(pending_feedback_count="
                    f"{data.pending_feedback_count}). A retract "
                    "recovers at most one feedback row, so a request retracted with "
                    "more than one unconsumed frame cannot be fully replayed"
                )
            QwenTalkerModelRunner._append_decode_input_history(data, combined)

        rows = [
            QwenTalkerModelRunner._decode_row(row, device=device, dtype=dtype)
            for row in history[gen_start:gen_end]
        ]
        if not rows:
            return None
        return torch.stack(rows, dim=0)

    def _write_feedback_buffers(
        self, requests: list, pool_indices: torch.Tensor | None = None
    ) -> None:
        batch_size = len(requests)
        if batch_size == 0:
            return

        feedback_buffer = self.model._feedback_buffer
        feedback_mask = self.model._feedback_mask
        device = feedback_buffer.device
        dtype = feedback_buffer.dtype
        feedback_mask[:batch_size] = False

        rows: list[int] = []
        datas: list[Any] = []
        pool_ids: list[int] = []
        overrides: list[torch.Tensor | None] = []
        text_rows: list[torch.Tensor] = []
        any_missing_pool_idx = False
        for row_idx, sched_req in enumerate(requests):
            data = sched_req.data
            if data.pending_feedback_count <= 0:
                continue
            override = data.retracted_feedback_embed
            pool_idx = data.req.req_pool_idx
            if override is None and pool_idx is None:
                raise RuntimeError(
                    "Talker request has pending feedback but no pool slot to read it "
                    "from: req_pool_idx is None and no retracted feedback snapshot "
                    "was taken"
                )
            next_text = self._peek_next_text_row(data)
            if next_text is None:
                continue
            rows.append(row_idx)
            datas.append(data)
            any_missing_pool_idx = any_missing_pool_idx or pool_idx is None
            pool_ids.append(0 if pool_idx is None else int(pool_idx))
            overrides.append(override)
            text_rows.append(self._decode_row(next_text, device=device, dtype=dtype))
        if not rows:
            return

        if (
            pool_indices is not None
            and len(rows) == batch_size
            and not any_missing_pool_idx
        ):
            # Note (wenyao): avoid a host-built index tensor on steady-state decode.
            pool_ids_t = pool_indices
        else:
            pool_ids_t = torch.tensor(pool_ids, dtype=torch.long, device=device)
        feedback_rows = self.model._feedback_slots[pool_ids_t]
        for i, override in enumerate(overrides):
            if override is not None:
                feedback_rows[i] = self._decode_row(
                    override, device=device, dtype=dtype
                )
        combined = feedback_rows + torch.stack(text_rows, dim=0)

        for i, data in enumerate(datas):
            self._append_decode_input_history(data, combined[i])
            self._consume_feedback_and_text(data)

        if len(rows) == batch_size:
            feedback_buffer[:batch_size] = combined
            feedback_mask[:batch_size] = True
            return
        rows_t = torch.tensor(rows, dtype=torch.long, device=device)
        feedback_buffer[rows_t] = combined
        feedback_mask[rows_t] = True

    @staticmethod
    def _data_has_next_decode_input(data: Any) -> bool:
        if data is None:
            return False
        if data.pending_feedback_count <= 0:
            return False
        if data.pending_text_queue:
            return True
        return bool(data.thinker_chunks_done and data.tts_pad_embed is not None)

    def _requests_ready_for_decode(self, requests: list) -> bool:
        return all(
            self._data_has_next_decode_input(sched_req.data) for sched_req in requests
        )

    @staticmethod
    def _pop_left(queue: Any) -> torch.Tensor | None:
        if not queue:
            return None
        if hasattr(queue, "popleft"):
            return queue.popleft()
        if isinstance(queue, list):
            return queue.pop(0)
        return None

    @staticmethod
    def _peek_left(queue: Any) -> torch.Tensor | None:
        if not queue:
            return None
        if isinstance(queue, list):
            return queue[0]
        if hasattr(queue, "__getitem__"):
            return queue[0]
        return None

    @staticmethod
    def _decode_input_history(data: Any) -> list[torch.Tensor]:
        return data.decode_input_embeds

    @staticmethod
    def _append_decode_input_history(data: Any, row: torch.Tensor) -> None:
        QwenTalkerModelRunner._decode_input_history(data).append(row.detach())

    @staticmethod
    def _decode_row(
        row: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        row = row.reshape(-1)
        if row.device != device or row.dtype != dtype:
            raise RuntimeError(
                "Talker decode rows must already match the feedback buffer "
                f"device/dtype, got {row.device}/{row.dtype}, "
                f"expected {device}/{dtype}"
            )
        return row

    @staticmethod
    def _peek_next_text_row(data: Any) -> torch.Tensor | None:
        next_text = QwenTalkerModelRunner._peek_left(data.pending_text_queue)
        if next_text is not None:
            return next_text
        if not data.thinker_chunks_done:
            return None
        return data.tts_pad_embed

    @staticmethod
    def _consume_feedback_and_text(data: Any) -> None:
        data.pending_feedback_count -= 1
        data.retracted_feedback_embed = None
        if data.pending_text_queue:
            QwenTalkerModelRunner._pop_left(data.pending_text_queue)

    @staticmethod
    def _combine_feedback_with_next_text(
        *,
        data: Any,
        feedback: torch.Tensor | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if feedback is None:
            return None
        next_text = QwenTalkerModelRunner._peek_next_text_row(data)
        if next_text is None:
            return None
        return QwenTalkerModelRunner._decode_row(
            feedback,
            device=device,
            dtype=dtype,
        ) + QwenTalkerModelRunner._decode_row(
            next_text,
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def _take_next_decode_input_embed(
        *,
        sched_req: Any,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        # Note (wenyao): the pool row may be reused after retract.
        data = sched_req.data
        if data.pending_feedback_count <= 0:
            return None
        combined = QwenTalkerModelRunner._combine_feedback_with_next_text(
            data=data,
            feedback=data.retracted_feedback_embed,
            device=device,
            dtype=dtype,
        )
        if combined is None:
            return None

        QwenTalkerModelRunner._consume_feedback_and_text(data)
        return combined
