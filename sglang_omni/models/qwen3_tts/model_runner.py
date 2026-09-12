# SPDX-License-Identifier: Apache-2.0
"""Qwen3-TTS model runner for the OmniScheduler AR stage."""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.prefill_inputs import (
    OmniPrefillInputs,
    attach_omni_prefill_inputs,
)
from sglang_omni.models.qwen3_omni.talker_model_runner import QwenTalkerModelRunner
from sglang_omni.scheduling.types import RequestOutput


def _ensure_mrope_positions(forward_batch: Any, *, prefill_graph_runner: Any) -> None:
    """Give a graph-replayed batch MRoPE positions mirroring its plain ones.

    The Talker declares ``is_mrope_enabled``, so a captured prefill graph binds
    the runner's ``mrope_positions`` slot, and that slot is only refreshed at
    replay when the live batch carries mrope positions. A TTS request has no
    multimodal inputs to provide them, so a replay would otherwise rotate on
    whatever positions capture happened to leave behind.

    All three MRoPE rows are equal for the Talker, and ``MRotaryEmbedding``
    selects row ``i`` of each mrope section, so a mirrored ``[3, T]`` collapses
    to exactly the 1-D result. It is not the same kernel though: on CUDA a 1-D
    positions tensor runs ``forward_native`` while a 2-D one runs the fused
    ``forward_triton``. Mirroring only the batches that replay keeps every
    other prefill on the kernel it already used: SGLang hands out an
    ``EagerRunner`` when prefill graphs are off, and a runner that holds graphs
    still declines batches outside its captured shapes.
    """
    if prefill_graph_runner is None or not prefill_graph_runner.can_run_graph(
        forward_batch
    ):
        return
    if forward_batch.mrope_positions is None:
        forward_batch.mrope_positions = (
            forward_batch.positions.unsqueeze(0).expand(3, -1).contiguous()
        )


class Qwen3TTSModelRunner(ModelRunner):
    """Runs Qwen3-TTS AR steps and stores generated codec frames per request."""

    def __init__(self, tp_worker: Any, output_processor: Any):
        super().__init__(tp_worker, output_processor)
        self._has_pending_code_step = False
        self._row_ids_cache: torch.Tensor | None = None

    def before_prefill(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        del schedule_batch
        _ensure_mrope_positions(
            forward_batch,
            prefill_graph_runner=self.tp_worker.model_runner.prefill_cuda_graph_runner,
        )
        self.model.prepare_decode_buffers(requests)
        attach_omni_prefill_inputs(
            forward_batch,
            OmniPrefillInputs(
                input_embeds=self._build_prefill_input_embeds(
                    forward_batch,
                    requests,
                ),
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
        self.model.prepare_decode_buffers(requests)
        self._write_feedback_buffers(forward_batch, requests)

    def post_prefill(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        self._collect_codes(result, forward_batch, schedule_batch, requests)

    def post_decode(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        self._collect_codes(result, forward_batch, schedule_batch, requests)

    def sample_before_post_prefill(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> bool:
        del forward_batch, schedule_batch, requests
        return True

    def sample_before_post_decode(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> bool:
        del forward_batch, schedule_batch, requests
        return True

    def _sample_next_token_ids(
        self,
        logits_output: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> Any:
        self._install_semantic_sampling_seeds(forward_batch, requests)
        return super()._sample_next_token_ids(
            logits_output,
            forward_batch,
            schedule_batch,
            requests,
        )

    # ------------------------------------------------------------------
    # Qwen3-TTS logit shaping
    # ------------------------------------------------------------------

    def _apply_codec_suppress_tokens(self, logits_output: Any, requests: list) -> None:
        logits = logits_output.next_token_logits
        if logits is None or logits.ndim != 2 or not requests:
            return

        # Qwen3-TTS reserves the final 1024 configured token IDs for codec
        # control and suppresses that range except for codec EOS.
        configured_vocab = int(self.model.config.vocab_size)
        suppress_start = max(0, configured_vocab - 1024)
        suppress_stop = min(configured_vocab, logits.shape[1])
        if suppress_start >= suppress_stop:
            return

        active_logits = logits[: len(requests)]
        codec_eos = int(self.model.config.codec_eos_token_id)
        if suppress_start <= codec_eos < suppress_stop:
            active_logits[:, suppress_start:codec_eos] = float("-inf")
            active_logits[:, codec_eos + 1 : suppress_stop] = float("-inf")
        else:
            active_logits[:, suppress_start:suppress_stop] = float("-inf")

    def _install_semantic_sampling_seeds(
        self,
        forward_batch: Any,
        requests: list,
    ) -> None:
        batch_size = len(requests)
        forward_batch.sampling_info.sampling_seed = (
            self.model._semantic_sampling_seed_tensor[:batch_size]
        )

    def _collect_codes(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        self._has_pending_code_step = False
        if result.next_token_ids is None:
            return
        layer0_codes = result.next_token_ids
        if layer0_codes.ndim == 1:
            layer0_codes = layer0_codes.unsqueeze(1)

        hidden = result.logits_output.hidden_states
        if isinstance(hidden, torch.Tensor) and hidden.ndim == 2:
            hidden = hidden.unsqueeze(1)
        semantic_positions = self._sample_positions(forward_batch, layer0_codes.device)
        self.model.code_predictor_forward(
            layer0_codes,
            hidden,
            semantic_positions=semantic_positions,
        )
        # Note: (Jiaxin Deng) stage the ids into pinned host memory now so the
        # output processor's .tolist() waits on an event instead of issuing a
        # blocking pageable copy inside the decode loop.
        self._stage_token_ids(result, result.next_token_ids)
        self._has_pending_code_step = True

    def post_process_outputs(
        self,
        result: Any,
        scheduler_output: Any,
        outputs: dict[str, RequestOutput],
    ) -> None:
        del result
        if not self._has_pending_code_step:
            return
        self._has_pending_code_step = False
        eos_id = int(self.model.config.codec_eos_token_id)
        # Note: (Jiaxin Deng) per-row clones were a c32 decode-loop hot spot;
        # rows must stay views of a snapshot, never of the reused graph buffers.
        batch_size = len(scheduler_output.requests)
        codes_snap = self.model._output_codes[:batch_size].detach().clone()
        embeds_snap = self.model._output_embeds[:batch_size].detach().clone()
        codes_ready = None
        if codes_snap.is_cuda:
            codes_ready = torch.cuda.Event()
            codes_ready.record()
        for row_idx, sched_req in enumerate(scheduler_output.requests):
            req_output = outputs[sched_req.request_id]
            if req_output.data is None or int(req_output.data) == eos_id:
                continue
            code_chunk = codes_snap[row_idx]
            sched_req.data.output_codes.append(code_chunk)
            sched_req.data.latest_stream_code_chunk = code_chunk
            sched_req.data.codes_ready_event = codes_ready
            sched_req.data.pending_feedback_queue.append(embeds_snap[row_idx])

    def _sample_positions(
        self, forward_batch: Any, device: torch.device
    ) -> torch.Tensor:
        forward_mode = getattr(forward_batch, "forward_mode", None)
        is_decode = (
            forward_mode is not None
            and hasattr(forward_mode, "is_decode")
            and bool(forward_mode.is_decode())
        )
        if is_decode:
            positions = getattr(forward_batch, "positions", None)
            if positions is not None:
                return positions.to(device=device, dtype=torch.long)

        seq_lens = getattr(forward_batch, "seq_lens", None)
        if seq_lens is not None:
            return (seq_lens.to(device=device, dtype=torch.long) - 1).clamp_min(0)

        positions = getattr(forward_batch, "positions", None)
        if positions is not None:
            return positions.to(device=device, dtype=torch.long)

        raise RuntimeError("Qwen3-TTS subtalker sampling requires semantic positions")

    def _write_feedback_buffers(self, forward_batch: Any, requests: list) -> None:
        batch_size = len(requests)
        if batch_size == 0:
            return
        decode_feedback_embedding = self.model._decode_feedback_embedding
        input_ids = forward_batch.input_ids
        if input_ids.numel() < batch_size:
            raise RuntimeError(
                "Qwen3-TTS decode input_ids must contain one row id per request"
            )
        if batch_size > decode_feedback_embedding.num_embeddings:
            raise RuntimeError(
                "Qwen3-TTS decode batch exceeds staged feedback embedding rows"
            )
        row_ids = self._decode_row_ids(batch_size, input_ids)
        weight = decode_feedback_embedding.weight
        device = weight.device
        dtype = weight.dtype
        feedback_rows: list[torch.Tensor] = []
        text_rows: list[torch.Tensor] = []
        batched_row_ids: list[int] = []
        rows: list[torch.Tensor | None] = [None] * batch_size
        for row_idx, sched_req in enumerate(requests):
            data = sched_req.data
            inputs = QwenTalkerModelRunner._peek_next_decode_inputs(data)
            if inputs is None:
                token_id = input_ids[row_idx : row_idx + 1].to(device=device)
                rows[row_idx] = self.model.get_input_embeddings()(token_id).reshape(-1)
                continue
            feedback, text = inputs
            feedback_rows.append(
                QwenTalkerModelRunner._decode_row(feedback, device=device, dtype=dtype)
            )
            text_rows.append(
                QwenTalkerModelRunner._decode_row(text, device=device, dtype=dtype)
            )
            batched_row_ids.append(row_idx)
            QwenTalkerModelRunner._pop_next_decode_inputs(data)

        with torch.no_grad():
            target = weight[:batch_size]
            if len(batched_row_ids) == batch_size:
                torch.stack(feedback_rows, dim=0, out=target)
                target.add_(torch.stack(text_rows, dim=0))
            else:
                if feedback_rows:
                    combined = torch.stack(feedback_rows, dim=0) + torch.stack(
                        text_rows, dim=0
                    )
                    for slot, row_idx in enumerate(batched_row_ids):
                        rows[row_idx] = combined[slot]
                torch.stack(rows, dim=0, out=target)
            # note(ratish): the history outlives the buffer, a retracted request
            # replays it in its re-prefill.
            history = target.detach().clone()
        for row_idx, sched_req in enumerate(requests):
            QwenTalkerModelRunner._append_decode_input_history(
                sched_req.data, history[row_idx]
            )
        # During graph decode, input_ids carries staged embedding row ids.
        input_ids[:batch_size].copy_(row_ids)

    def _decode_row_ids(self, batch_size: int, input_ids: torch.Tensor) -> torch.Tensor:
        cached = getattr(self, "_row_ids_cache", None)
        if (
            cached is None
            or cached.numel() < batch_size
            or cached.dtype != input_ids.dtype
            or cached.device != input_ids.device
        ):
            cached = torch.arange(
                max(batch_size, 64),
                device=input_ids.device,
                dtype=input_ids.dtype,
            )
            self._row_ids_cache = cached
        return cached[:batch_size]

    def _build_prefill_input_embeds(
        self,
        forward_batch: Any,
        requests: list,
    ) -> torch.Tensor:
        pieces = []
        for sched_req in requests:
            data = sched_req.data
            req = data.req
            req_len = int(req.extend_range.length)
            prefix_len = len(req.prefix_indices)
            if data.prefill_input_embeds is None:
                data.prefill_input_embeds = data.prompt_input_embeds
            if data.prefill_input_embeds is None:
                raise RuntimeError("Qwen3-TTS prefill requires prompt_input_embeds")
            piece = QwenTalkerModelRunner._projected_prefill_slice(
                sched_req=sched_req,
                prefix_len=prefix_len,
                extend_len=req_len,
                device=forward_batch.input_ids.device,
            )
            if piece is None or int(piece.shape[0]) != req_len:
                have = 0 if piece is None else int(piece.shape[0])
                raise RuntimeError(
                    f"Qwen3-TTS prefill embed mismatch for {req.rid}: "
                    f"have {have} rows, need {req_len}"
                )
            pieces.append(piece)
        return torch.cat(pieces, dim=0).to(
            device=forward_batch.input_ids.device,
            dtype=next(self.model.parameters()).dtype,
        )
