# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni talker runner with buffer-backed feedback + batched code predictor."""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
from sglang.srt.managers.scheduler import GenerationBatchResult

from sglang_omni.debug_trace import append_trace, summarize_tensor
from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.qwen3_omni.debug_dump import (
    dump_precision_pt,
    precision_dump_path,
)
from sglang_omni.scheduling.messages import OutgoingMessage

logger = logging.getLogger(__name__)
_TRACE_TALKER = os.environ.get("QWEN_TALKER_TRACE") == "1"


def _tensor_head(tensor: torch.Tensor, n: int = 4) -> list[float | int]:
    return tensor.detach().reshape(-1)[:n].cpu().tolist()


def _tensor_norm(tensor: torch.Tensor) -> float:
    return float(tensor.detach().float().norm().cpu().item())


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
        del schedule_batch
        if not self._feedback_enabled:
            return None

        self.model.prepare_decode_buffers(requests)
        self._write_feedback_buffers(requests)
        for row_idx, sched_req in enumerate(requests):
            dump_precision_pt(
                prefix="talker_decode_context",
                request_id=sched_req.request_id,
                step=int(getattr(sched_req.data, "generation_steps", 0)),
                payload={
                    "request_id": sched_req.request_id,
                    "generation_steps": int(
                        getattr(sched_req.data, "generation_steps", 0)
                    ),
                    "input_id": (
                        int(forward_batch.input_ids[row_idx].item())
                        if hasattr(forward_batch, "input_ids")
                        and isinstance(forward_batch.input_ids, torch.Tensor)
                        and forward_batch.input_ids.ndim == 1
                        else None
                    ),
                    "position": (
                        int(forward_batch.positions[row_idx].item())
                        if hasattr(forward_batch, "positions")
                        and isinstance(forward_batch.positions, torch.Tensor)
                        and forward_batch.positions.ndim == 1
                        else None
                    ),
                    "mrope_positions": (
                        forward_batch.mrope_positions[:, row_idx].detach().cpu()
                        if hasattr(forward_batch, "mrope_positions")
                        and isinstance(forward_batch.mrope_positions, torch.Tensor)
                        and forward_batch.mrope_positions.ndim == 2
                        else None
                    ),
                    "feedback_mask": bool(self.model._feedback_mask[row_idx].item()),
                    "feedback_buffer": self.model._feedback_buffer[row_idx]
                    .detach()
                    .cpu(),
                },
            )
        return None

    def post_prefill(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
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
            result=result,
            forward_batch=forward_batch,
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
            result=result,
            forward_batch=forward_batch,
            schedule_batch=schedule_batch,
            requests=requests,
        )

    def _emit_code_chunks_and_feedback(
        self,
        *,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        for idx, sched_req in enumerate(requests):
            req = schedule_batch.reqs[idx]
            if (
                forward_batch is not None
                and hasattr(result, "logits_output")
                and getattr(result.logits_output, "next_token_logits", None) is not None
            ):
                generation_steps = int(getattr(sched_req.data, "generation_steps", 0))
                logits = (
                    result.logits_output.next_token_logits[idx].detach().cpu().float()
                )
                top_scores, top_ids = torch.topk(logits, k=min(5, int(logits.numel())))
                input_id_value = (
                    int(forward_batch.input_ids[idx].item())
                    if hasattr(forward_batch, "input_ids")
                    and isinstance(forward_batch.input_ids, torch.Tensor)
                    and forward_batch.input_ids.ndim == 1
                    else None
                )
                pos_value = (
                    int(forward_batch.positions[idx].item())
                    if hasattr(forward_batch, "positions")
                    and isinstance(forward_batch.positions, torch.Tensor)
                    and forward_batch.positions.ndim == 1
                    else None
                )
                dump_precision_pt(
                    prefix="talker_decode_input",
                    request_id=sched_req.request_id,
                    step=generation_steps,
                    payload={
                        "request_id": sched_req.request_id,
                        "generation_steps": generation_steps,
                        "input_id": input_id_value,
                        "position": pos_value,
                        "mrope_positions": (
                            forward_batch.mrope_positions[:, idx].detach().cpu()
                            if hasattr(forward_batch, "mrope_positions")
                            and isinstance(forward_batch.mrope_positions, torch.Tensor)
                            and forward_batch.mrope_positions.ndim == 2
                            else None
                        ),
                        "logits": logits,
                        "top_ids": [int(v) for v in top_ids.tolist()],
                        "top_scores": [float(v) for v in top_scores.tolist()],
                    },
                )
            code_chunk = self.model._output_codes[idx].detach().clone()
            feedback_row = self.model._output_embeds[idx].detach().clone()
            talker_hidden = result.logits_output.hidden_states[idx].detach().clone()
            append_trace(
                "code_predictor",
                req.rid,
                "dispatch",
                codes=code_chunk.detach().cpu().reshape(-1).tolist(),
                feedback=summarize_tensor(feedback_row),
            )
            self._append_code_predictor_dump(
                request_id=req.rid,
                output_codes=code_chunk.detach().cpu(),
                talker_hidden=talker_hidden.detach().cpu(),
                feedback=feedback_row.detach().cpu(),
            )
            self._outbox.put(
                OutgoingMessage(
                    request_id=req.rid,
                    type="stream",
                    data=code_chunk,
                    target=self._code2wav_target,
                )
            )
            sched_req.data.feedback_embeds = feedback_row

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

    def trace_pre_sample(
        self,
        logits_output: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        logits = getattr(logits_output, "next_token_logits", None)
        if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
            return

        input_ids = getattr(forward_batch, "input_ids", None)
        positions = getattr(forward_batch, "positions", None)
        mrope_positions = getattr(forward_batch, "mrope_positions", None)

        for row_idx, sched_req in enumerate(requests):
            if row_idx >= logits.shape[0]:
                continue
            top_scores, top_ids = torch.topk(
                logits[row_idx].detach().float(),
                k=min(5, int(logits.shape[1])),
            )
            generation_steps = int(
                getattr(getattr(sched_req, "data", None), "generation_steps", 0)
            )
            input_id_value = (
                int(input_ids[row_idx].item())
                if isinstance(input_ids, torch.Tensor) and input_ids.ndim == 1
                else None
            )
            pos_value = (
                int(positions[row_idx].item())
                if isinstance(positions, torch.Tensor) and positions.ndim == 1
                else None
            )
            dump_precision_pt(
                prefix="talker_decode_input",
                request_id=sched_req.request_id,
                step=generation_steps,
                payload={
                    "request_id": sched_req.request_id,
                    "generation_steps": generation_steps,
                    "input_id": input_id_value,
                    "position": pos_value,
                    "mrope_positions": (
                        mrope_positions[:, row_idx].detach().cpu()
                        if isinstance(mrope_positions, torch.Tensor)
                        and mrope_positions.ndim == 2
                        else None
                    ),
                    "logits": logits[row_idx].detach().cpu().float(),
                    "top_ids": [int(v) for v in top_ids.cpu().tolist()],
                    "top_scores": [float(v) for v in top_scores.cpu().tolist()],
                },
            )

    def _run_projected_prefill_forward(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> GenerationBatchResult | None:
        has_projected = forward_batch.input_embeds is not None or any(
            bool(req.data.input_embeds_are_projected) for req in requests
        )
        if not has_projected:
            return None

        input_embeds = forward_batch.input_embeds
        projected_flags = [
            bool(req.data.input_embeds_are_projected) for req in requests
        ]
        input_embeds_are_projected = bool(projected_flags) and all(projected_flags)
        if input_embeds is None:
            rows = []
            for sched_req in requests:
                req = sched_req.data.req
                embeds = req.input_embeds
                if embeds:
                    prefix_len = len(getattr(req, "prefix_indices", []))
                    rows.extend(embeds[prefix_len:])
            if not rows:
                return None
            input_embeds = torch.as_tensor(
                rows,
                device=forward_batch.input_ids.device,
                dtype=torch.float32,
            )

        result = self._forward_with_input_embeds(
            forward_batch,
            input_embeds=input_embeds,
            input_embeds_are_projected=input_embeds_are_projected,
        )
        request_id = getattr(
            requests[0],
            "request_id",
            getattr(getattr(requests[0].data, "req", None), "rid", "prefill"),
        )
        if hasattr(result, "logits_output") and result.logits_output is not None:
            dump_precision_pt(
                prefix="talker_prefill_logits",
                request_id=request_id,
                payload={
                    "next_token_logits": result.logits_output.next_token_logits.detach().cpu(),
                    "hidden_states": (
                        result.logits_output.hidden_states.detach().cpu()
                        if isinstance(result.logits_output.hidden_states, torch.Tensor)
                        else None
                    ),
                },
            )
        return result

    def _write_feedback_buffers(self, requests: list) -> None:
        batch_size = len(requests)
        if batch_size == 0:
            return

        feedback_buffer = self.model._feedback_buffer
        feedback_mask = self.model._feedback_mask
        feedback_mask[:batch_size] = False

        for row_idx, sched_req in enumerate(requests):
            combined = self._combine_feedback_embed(
                sched_req=sched_req,
                row_idx=row_idx,
                device=feedback_buffer.device,
                dtype=feedback_buffer.dtype,
            )
            if combined is None:
                continue
            feedback_buffer[row_idx].copy_(combined)
            feedback_mask[row_idx] = True
            request_id = getattr(
                sched_req,
                "request_id",
                getattr(getattr(sched_req.data, "req", None), "rid", f"row-{row_idx}"),
            )
            dump_precision_pt(
                prefix="talker_feedback_input",
                request_id=request_id,
                step=max(int(getattr(sched_req.data, "generation_steps", 0)), 0),
                payload={
                    "request_id": request_id,
                    "generation_steps": int(
                        getattr(sched_req.data, "generation_steps", 0)
                    ),
                    "step_index": int(
                        getattr(sched_req.data, "_last_feedback_step_index", 0)
                    ),
                    "row_idx": row_idx,
                    "combined_feedback_input_embeds": combined.detach().cpu(),
                    "trailing_source": getattr(
                        sched_req.data, "_last_feedback_trailing_source", None
                    ),
                    "trailing_value": (
                        getattr(sched_req.data, "_last_feedback_trailing_value")
                        .detach()
                        .cpu()
                        if isinstance(
                            getattr(
                                sched_req.data, "_last_feedback_trailing_value", None
                            ),
                            torch.Tensor,
                        )
                        else None
                    ),
                },
            )
            sched_req.data.feedback_embeds = None

    @staticmethod
    def _combine_feedback_embed(
        *,
        sched_req: Any,
        row_idx: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        data = sched_req.data
        feedback = data.feedback_embeds
        if feedback is None:
            return None

        request_id = getattr(
            sched_req,
            "request_id",
            getattr(getattr(data, "req", None), "rid", f"row-{row_idx}"),
        )

        combined = feedback.to(device=device, dtype=dtype).reshape(-1)
        projected = bool(
            getattr(data, "input_embeds_are_projected", False)
            or getattr(getattr(data, "req", None), "_input_embeds_are_projected", False)
        )
        step_offset = 2 if projected else 1
        step_index = max(int(data.generation_steps) - step_offset, 0)
        trailing = data.trailing_text_hidden
        tts_pad_embed = data.tts_pad_embed
        thinker_chunks_done = bool(data.thinker_chunks_done)

        trailing_value = None
        if isinstance(trailing, list) and step_index < len(trailing):
            trailing_value = trailing[step_index]
        elif isinstance(trailing, torch.Tensor) and step_index < trailing.shape[0]:
            trailing_value = trailing[step_index]

        if trailing_value is not None:
            combined = combined + trailing_value.to(
                device=device,
                dtype=dtype,
            ).reshape(-1)
            trailing_source = "trailing_text_hidden"
        elif thinker_chunks_done and tts_pad_embed is not None:
            combined = combined + tts_pad_embed.to(
                device=device,
                dtype=dtype,
            ).reshape(-1)
            trailing_source = "tts_pad_embed"
        else:
            trailing_source = "none"
        setattr(data, "_last_feedback_step_index", step_index)
        setattr(data, "_last_feedback_trailing_source", trailing_source)
        setattr(data, "_last_feedback_trailing_value", trailing_value)
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

    @staticmethod
    def _append_code_predictor_dump(
        *,
        request_id: str,
        output_codes: torch.Tensor,
        talker_hidden: torch.Tensor,
        feedback: torch.Tensor,
    ) -> None:
        path = precision_dump_path(prefix="code_predictor_debug", request_id=request_id)
        if path is None:
            return
        if path.exists():
            payload = torch.load(path, map_location="cpu")
        else:
            payload = {
                "request_id": request_id,
                "layer0_codes": torch.empty((0,), dtype=torch.long),
                "output_codes": torch.empty(
                    (0, int(output_codes.numel())), dtype=torch.long
                ),
                "talker_hidden": torch.empty(
                    (0, int(talker_hidden.numel())), dtype=talker_hidden.dtype
                ),
                "feedbacks": torch.empty(
                    (0, int(feedback.numel())), dtype=feedback.dtype
                ),
            }

        codec_row = output_codes.reshape(-1).to(dtype=torch.long)
        payload["layer0_codes"] = torch.cat(
            [payload["layer0_codes"].to(dtype=torch.long), codec_row[:1]],
            dim=0,
        )
        payload["output_codes"] = torch.cat(
            [payload["output_codes"].to(dtype=torch.long), codec_row.reshape(1, -1)],
            dim=0,
        )
        payload["talker_hidden"] = torch.cat(
            [
                payload["talker_hidden"].to(dtype=talker_hidden.dtype),
                talker_hidden.reshape(1, -1),
            ],
            dim=0,
        )
        payload["feedbacks"] = torch.cat(
            [payload["feedbacks"].to(dtype=feedback.dtype), feedback.reshape(1, -1)],
            dim=0,
        )
        torch.save(payload, path)
