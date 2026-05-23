# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni talker runner with FIFO text/feedback decode handoff."""

from __future__ import annotations

from typing import Any

import torch
from sglang.srt.managers.scheduler import GenerationBatchResult

from sglang_omni.model_runner.base import ModelRunner
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

    def execute(self, scheduler_output: Any):
        return super().execute(scheduler_output)

    def prepare_prefill(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> GenerationBatchResult | None:
        return self._run_projected_prefill_forward(
            forward_batch, schedule_batch, requests
        )

    def prepare_decode(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> GenerationBatchResult | None:
        del forward_batch
        del schedule_batch
        if not self._feedback_enabled:
            return None

        if not self._requests_ready_for_decode(requests):
            raise RuntimeError(
                "Talker decode reached model runner without ready feedback/text input"
            )

        self.model.prepare_decode_buffers(requests)
        self._write_feedback_buffers(requests)
        return None

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
        schedule_batch.output_ids = result.next_token_ids
        self._emit_code_chunks_and_feedback(
            schedule_batch=schedule_batch,
            requests=requests,
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
        schedule_batch.output_ids = result.next_token_ids
        self._emit_code_chunks_and_feedback(
            schedule_batch=schedule_batch,
            requests=requests,
        )

    def _emit_code_chunks_and_feedback(
        self,
        *,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        for idx, sched_req in enumerate(requests):
            req = schedule_batch.reqs[idx]
            code_chunk = self.model._output_codes[idx].detach().clone()
            feedback_row = self.model._output_embeds[idx].detach().clone()
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
            sched_req.data.pending_feedback_queue.append(feedback_row)

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
            self._data_has_next_decode_input(getattr(req, "_omni_data", None))
            for req in schedule_batch.reqs
        )

    def _run_projected_prefill_forward(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> GenerationBatchResult | None:
        del schedule_batch
        has_projected = forward_batch.input_embeds is not None or any(
            bool(req.data.input_embeds_are_projected) for req in requests
        )
        if not has_projected:
            return None

        projected_flags = [
            bool(req.data.input_embeds_are_projected) for req in requests
        ]
        has_projected_requests = any(projected_flags)
        if has_projected_requests and not all(projected_flags):
            raise RuntimeError(
                "Talker projected and unprojected prefill requests cannot be "
                "batched together"
            )

        input_embeds_are_projected = has_projected_requests
        input_embeds = forward_batch.input_embeds
        if has_projected_requests:
            parts: list[torch.Tensor] = []
            for sched_req in requests:
                req = sched_req.data.req
                prefix_len = len(req.prefix_indices)
                extend_len = int(req.extend_input_len)
                part = self._projected_prefill_slice(
                    sched_req=sched_req,
                    prefix_len=prefix_len,
                    extend_len=extend_len,
                    device=forward_batch.input_ids.device,
                )
                if part is not None and part.shape[0] > 0:
                    parts.append(part)
            if not parts:
                return None
            input_embeds = torch.cat(parts, dim=0)
        elif input_embeds is None:
            return None

        expected_rows = int(forward_batch.input_ids.shape[0])
        if input_embeds.shape[0] != expected_rows:
            raise RuntimeError(
                "Talker projected prefill embeds must align with forward input_ids: "
                f"got {input_embeds.shape[0]} rows for {expected_rows} input ids"
            )

        result = self._forward_with_input_embeds(
            forward_batch,
            input_embeds=input_embeds,
            input_embeds_are_projected=input_embeds_are_projected,
        )
        return result

    @staticmethod
    def _projected_prefill_slice(
        *,
        sched_req: Any,
        prefix_len: int,
        extend_len: int,
        device: torch.device,
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
    ) -> torch.Tensor | None:
        if gen_end <= gen_start:
            return None

        data = sched_req.data
        history = QwenTalkerModelRunner._decode_input_history(data)
        while len(history) < gen_end:
            combined = QwenTalkerModelRunner._take_next_decode_input_embed(
                sched_req=sched_req,
                device=device,
                dtype=dtype,
            )
            if combined is None:
                raise RuntimeError(
                    "Cannot replay retracted talker decode tokens: missing "
                    "feedback/text input embeds for generated-token prefill"
                )
            QwenTalkerModelRunner._append_decode_input_history(data, combined)

        rows = [
            QwenTalkerModelRunner._decode_row(row, device=device, dtype=dtype)
            for row in history[gen_start:gen_end]
        ]
        if not rows:
            return None
        return torch.stack(rows, dim=0)

    def _write_feedback_buffers(self, requests: list) -> None:
        batch_size = len(requests)
        if batch_size == 0:
            return

        feedback_buffer = self.model._feedback_buffer
        feedback_mask = self.model._feedback_mask
        feedback_mask[:batch_size] = False

        rows: list[int] = []
        embeds: list[torch.Tensor] = []
        for row_idx, sched_req in enumerate(requests):
            combined = self._take_next_decode_input_embed(
                sched_req=sched_req,
                device=feedback_buffer.device,
                dtype=feedback_buffer.dtype,
            )
            if combined is None:
                continue
            self._append_decode_input_history(sched_req.data, combined)
            rows.append(row_idx)
            embeds.append(combined)
        if rows:
            rows_t = torch.tensor(rows, dtype=torch.long, device=feedback_buffer.device)
            embeds_stacked = torch.stack(embeds, dim=0)
            feedback_buffer[rows_t] = embeds_stacked
            feedback_mask[rows_t] = True

    @staticmethod
    def _data_has_next_decode_input(data: Any) -> bool:
        if data is None:
            return False
        pending_feedback_queue = getattr(data, "pending_feedback_queue", None)
        if not pending_feedback_queue:
            return False
        pending_text_queue = getattr(data, "pending_text_queue", None)
        if pending_text_queue:
            return True
        return bool(
            getattr(data, "thinker_chunks_done", False)
            and getattr(data, "tts_pad_embed", None) is not None
        )

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
        history = getattr(data, "decode_input_embeds", None)
        if history is None:
            history = []
            data.decode_input_embeds = history
        return history

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
    def _combine_feedback_with_next_text(
        *,
        data: Any,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        pending_feedback_queue = getattr(data, "pending_feedback_queue", None)
        feedback = QwenTalkerModelRunner._peek_left(pending_feedback_queue)
        if feedback is None:
            return None

        combined = QwenTalkerModelRunner._decode_row(
            feedback,
            device=device,
            dtype=dtype,
        )
        next_text = QwenTalkerModelRunner._peek_left(
            getattr(data, "pending_text_queue", None)
        )
        if next_text is None:
            if not data.thinker_chunks_done:
                return None
            next_text = data.tts_pad_embed

        return combined + QwenTalkerModelRunner._decode_row(
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
        data = sched_req.data
        combined = QwenTalkerModelRunner._combine_feedback_with_next_text(
            data=data,
            device=device,
            dtype=dtype,
        )
        if combined is None:
            return None

        QwenTalkerModelRunner._pop_left(getattr(data, "pending_feedback_queue", None))
        if getattr(data, "pending_text_queue", None):
            QwenTalkerModelRunner._pop_left(data.pending_text_queue)
        return combined

    def _forward_with_input_embeds(
        self,
        forward_batch: Any,
        *,
        input_embeds: torch.Tensor,
        input_deepstack_embeds: torch.Tensor | None = None,
        input_deepstack_mask: torch.Tensor | None = None,
        input_embeds_are_projected: bool = False,
    ) -> GenerationBatchResult:
        model_runner = self.tp_worker.model_runner
        model_dtype = next(self.model.parameters()).dtype

        model_runner.attn_backend.init_forward_metadata(forward_batch)

        positions = forward_batch.positions
        if forward_batch.mrope_positions is not None:
            positions = forward_batch.mrope_positions

        input_embeds = input_embeds.to(
            device=forward_batch.input_ids.device,
            dtype=model_dtype,
        )
        if input_deepstack_embeds is not None:
            input_deepstack_embeds = input_deepstack_embeds.to(
                device=forward_batch.input_ids.device,
                dtype=model_dtype,
            )

        logits_output = self.model(
            input_ids=forward_batch.input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
            input_deepstack_embeds=input_deepstack_embeds,
            input_deepstack_mask=input_deepstack_mask,
            input_embeds_are_projected=input_embeds_are_projected,
        )
        return GenerationBatchResult(
            logits_output=logits_output,
            can_run_cuda_graph=False,
        )
