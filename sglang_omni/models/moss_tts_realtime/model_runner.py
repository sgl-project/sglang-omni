# SPDX-License-Identifier: Apache-2.0
"""Eager model runner for MOSS-TTS-Realtime."""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.moss_tts_realtime.observability import (
    emit_realtime_event as _emit_event,
)
from sglang_omni.models.moss_tts_realtime.observability import (
    realtime_events_active,
    realtime_identity_metadata,
)
from sglang_omni.models.moss_tts_realtime.state_pool import MossTTSRealtimeDecodeJournal
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.types import RequestOutput


class MossTTSRealtimeModelRunner(ModelRunner):
    """Projects canonical rows and samples local audio frames."""

    _vocoder_target = "vocoder"

    def __init__(self, tp_worker: Any, output_processor: Any) -> None:
        super().__init__(tp_worker, output_processor)
        self._outbox: Any | None = None
        server_args = getattr(tp_worker, "server_args", None)
        if server_args is None:
            server_args = getattr(tp_worker.model_runner, "server_args", None)
        max_active_turns = int(
            getattr(
                tp_worker,
                "moss_tts_realtime_max_active_turns",
                getattr(server_args, "max_running_requests", 1),
            )
            or 1
        )
        context_length = getattr(server_args, "context_length", None)
        max_history_frames = int(
            getattr(
                tp_worker,
                "moss_tts_realtime_max_history_frames",
                context_length,
            )
            or context_length
            or 1
        )
        self.model.init_decode_state_pool(
            max_running_requests=max_active_turns,
            max_history_frames=max_history_frames,
        )

    def set_stream_outbox(self, outbox: Any) -> None:
        self._outbox = outbox

    @property
    def _pool(self):
        return self.model.state_pool

    def resource_snapshot(self) -> dict[str, int]:
        snapshot = self._pool.resource_snapshot()
        graph_snapshot = getattr(
            self.model,
            "local_cuda_graph_resource_snapshot",
            None,
        )
        if callable(graph_snapshot):
            snapshot.update(graph_snapshot())
        return snapshot

    @staticmethod
    def _request_event_metadata(sched_req: Any) -> dict[str, Any]:
        data = sched_req.data
        metadata = realtime_identity_metadata(getattr(data, "state", None))
        req = getattr(data, "req", None)
        turn = getattr(data, "turn_state", None)
        if turn is not None:
            metadata.setdefault("session_id", turn.session_id)
            metadata.setdefault("turn_id", turn.turn_id)
        prompt_rows = getattr(data, "prompt_rows", None)
        prefix_indices = (
            getattr(req, "prefix_indices", None) if req is not None else None
        )
        metadata.update(
            {
                "seq_no": (
                    turn.pending_input.next_seq_no - 1
                    if turn is not None and turn.pending_input.next_seq_no
                    else None
                ),
                "stable_token_count": (
                    turn.pending_input.total_received_tokens
                    if turn is not None
                    else len(getattr(data, "initial_token_ids", ()) or ())
                ),
                "prompt_rows": (
                    int(prompt_rows.shape[0])
                    if isinstance(prompt_rows, torch.Tensor)
                    else None
                ),
                "prefill_cached_rows": (
                    len(prefix_indices) if prefix_indices is not None else 0
                ),
                "prefill_dispatch_rows": (
                    int(
                        getattr(
                            getattr(req, "extend_range", None),
                            "length",
                            0,
                        )
                        or 0
                    )
                    if req is not None
                    else 0
                ),
            }
        )
        return metadata

    def _emit_prefill_events(
        self,
        event_name: str,
        schedule_batch: Any,
        requests: list[Any],
        *,
        can_run_cuda_graph: bool | None = None,
    ) -> None:
        if not realtime_events_active():
            return
        batch_size = len(requests)
        for sched_req in requests:
            metadata = self._request_event_metadata(sched_req)
            metadata.update(
                {
                    "batch_size": batch_size,
                    "is_prefill_only": bool(
                        getattr(schedule_batch, "is_prefill_only", False)
                    ),
                    "is_extend_in_batch": bool(
                        getattr(schedule_batch, "is_extend_in_batch", False)
                    ),
                    "is_chunked": self._is_chunked_request(sched_req),
                }
            )
            if can_run_cuda_graph is not None:
                metadata["can_run_cuda_graph"] = can_run_cuda_graph
            _emit_event(
                request_id=sched_req.request_id,
                stage=None,
                event_name=event_name,
                metadata=metadata,
            )

    def before_prefill(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list[Any],
    ) -> None:
        del forward_batch
        self._emit_prefill_events(
            "prefill_dispatch_start",
            schedule_batch,
            requests,
        )

    def custom_prefill_forward(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list[Any],
    ) -> None:
        del schedule_batch
        forward_batch.input_embeds = self._build_prefill_input_embeds(
            forward_batch,
            requests,
        )
        for request in requests:
            request.data.input_embeds_are_projected = True
        return None

    def _build_prefill_input_embeds(
        self,
        forward_batch: Any,
        requests: list[Any],
    ) -> torch.Tensor:
        pieces: list[torch.Tensor] = []
        for sched_req in requests:
            data = sched_req.data
            req = data.req
            rows = data.prompt_rows
            if rows is None:
                raise RuntimeError("MOSS-TTS-Realtime prefill requires prompt_rows")
            if not isinstance(rows, torch.Tensor):
                rows = torch.as_tensor(rows, dtype=torch.long)
            req_len = int(req.extend_range.length)
            prefix_len = len(req.prefix_indices)
            current_rows = rows[prefix_len : prefix_len + req_len]
            if int(current_rows.shape[0]) != req_len:
                raise RuntimeError(
                    f"MOSS-TTS-Realtime prefill row mismatch for {req.rid}: have "
                    f"{int(current_rows.shape[0])} rows, need {req_len} "
                    f"(prefix={prefix_len}, total={int(rows.shape[0])})"
                )
            pieces.append(
                self.model._prepare_multi_modal_inputs(
                    current_rows.to(device=forward_batch.input_ids.device)
                )
            )
        if not pieces:
            return torch.empty(
                (0, self.model.hidden_size),
                dtype=self.model.dtype,
                device=forward_batch.input_ids.device,
            )
        return torch.cat(pieces, dim=0).to(
            device=forward_batch.input_ids.device,
            dtype=self.model.dtype,
        )

    def before_decode(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list[Any],
        *,
        is_lookahead: bool = False,
    ) -> None:
        del schedule_batch
        if is_lookahead:
            raise RuntimeError("MOSS-TTS-Realtime does not support async lookahead")
        if not requests:
            return

        row_t, pool_rows = self._pool.prepare_active_rows(requests)
        materialized_rows: list[tuple[int, ...]] = []
        cache_keys: list[int] = []
        for sched_req, row_idx in zip(requests, pool_rows, strict=True):
            data = sched_req.data
            turn_state = data.turn_state
            if turn_state is None:
                raise RuntimeError(
                    "MOSS-TTS-Realtime decode requires scheduler-owned turn state"
                )
            if turn_state.provisional_frame is not None:
                raise RuntimeError(
                    f"request {sched_req.request_id!r} reached decode with an "
                    "unresolved provisional frame"
                )
            materialized = turn_state.last_materialized_row
            if materialized is None:
                raise RuntimeError(
                    f"request {sched_req.request_id!r} has no materialized decode row"
                )
            if turn_state.model_state_slot_id != row_idx:
                raise RuntimeError(
                    "turn/model state-pool ownership changed before decode"
                )
            materialized_rows.append(materialized.row)
            cache_keys.append(int(materialized.cache_key))

        if forward_batch.input_ids.numel() < len(requests):
            raise RuntimeError(
                "MOSS-TTS-Realtime decode input_ids must contain one cache key "
                "per request"
            )
        expected_keys = torch.tensor(
            cache_keys,
            dtype=torch.long,
            device=forward_batch.input_ids.device,
        )
        actual_keys = forward_batch.input_ids[: len(requests)].to(dtype=torch.long)
        if not torch.equal(actual_keys, expected_keys):
            raise RuntimeError(
                "MOSS-TTS-Realtime decode ids do not match materialized row hashes"
            )

        rows = torch.tensor(
            materialized_rows,
            dtype=torch.long,
            device=forward_batch.input_ids.device,
        )
        embeddings = self.model._prepare_multi_modal_inputs(rows)
        for sched_req, materialized in zip(
            requests,
            materialized_rows,
            strict=True,
        ):
            self._pool.ensure_materialized(sched_req.request_id, materialized)
            sched_req.data.input_embeds_are_projected = True
        self._pool.stage_feedback(row_t, embeddings)
        batch_size = len(requests)
        weight = self.model._decode_input_embedding.weight
        if batch_size > int(weight.shape[0]):
            raise RuntimeError(
                "MOSS-TTS-Realtime decode batch exceeds the staged decode-embedding "
                f"rows ({batch_size} > {int(weight.shape[0])})"
            )
        with torch.no_grad():
            weight[:batch_size].copy_(
                self._pool.feedback_for(row_t).to(
                    device=weight.device,
                    dtype=weight.dtype,
                )
            )
        forward_batch.input_ids[:batch_size].copy_(
            torch.arange(
                batch_size,
                device=forward_batch.input_ids.device,
                dtype=torch.long,
            )
        )
        forward_batch.input_embeds = None
        forward_batch.moss_realtime_pool_row_t = row_t
        forward_batch.moss_realtime_pool_rows = pool_rows

    def on_realtime_row_materialized(
        self,
        sched_req: Any,
        materialized: Any,
    ) -> int:
        """Clear model-pool provisional state before decode can be prepared."""

        row_idx = self._pool.mark_materialized(
            sched_req.request_id,
            materialized.row,
        )
        turn_state = sched_req.data.turn_state
        if turn_state is None or turn_state.model_state_slot_id != row_idx:
            raise RuntimeError(
                "scheduler materialization changed model-state slot ownership"
            )
        return row_idx

    def post_prefill(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list[Any],
    ) -> None:
        self._emit_prefill_events(
            "prefill_dispatch_end",
            schedule_batch,
            requests,
            can_run_cuda_graph=bool(getattr(result, "can_run_cuda_graph", False)),
        )
        if bool(getattr(schedule_batch, "is_prefill_only", False)):
            return
        self._collect_frame(result, forward_batch, schedule_batch, requests)

    def post_decode(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list[Any],
    ) -> None:
        self._collect_frame(result, forward_batch, schedule_batch, requests)

    def _provisional_ids(
        self,
        requests: list[Any],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        values: list[int] = []
        vocab_size = int(self.model.config.vocab_size)
        for sched_req in requests:
            value = getattr(sched_req.data, "provisional_output_id", None)
            if isinstance(value, bool) or not isinstance(value, int):
                raise RuntimeError(
                    f"request {sched_req.request_id!r} has no integer provisional id"
                )
            if value < 0 or value >= vocab_size:
                raise ValueError(
                    f"request {sched_req.request_id!r} provisional id is outside "
                    "the text vocabulary"
                )
            if value == int(self.model.config.audio_eos_token):
                raise ValueError("the provisional id must differ from audio EOS")
            values.append(value)
        return torch.tensor(values, dtype=torch.long, device=device)

    def _collect_frame(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list[Any],
    ) -> None:
        if not requests:
            return
        try:
            hidden_states = result.logits_output.hidden_states
        except AttributeError as exc:
            raise RuntimeError(
                "MOSS-TTS-Realtime model output did not include hidden states"
            ) from exc
        if not isinstance(hidden_states, torch.Tensor):
            raise RuntimeError(
                "MOSS-TTS-Realtime model output did not include hidden states"
            )
        if hidden_states.ndim == 3:
            hidden_states = hidden_states[:, -1, :]
        if hidden_states.ndim != 2 or int(hidden_states.shape[0]) != len(requests):
            raise RuntimeError(
                "MOSS-TTS-Realtime sampled hidden states must have shape "
                "[batch, hidden]"
            )

        try:
            pool_rows = forward_batch.moss_realtime_pool_rows
        except AttributeError:
            _, pool_rows = self._pool.prepare_active_rows(requests)
        else:
            if len(pool_rows) != len(requests):
                raise RuntimeError("decode state-pool row count does not match batch")

        provisional_ids = self._provisional_ids(
            requests,
            device=hidden_states.device,
        )
        rids = [request.request_id for request in requests]
        sample_positions = self._pool.sample_positions_for(pool_rows)
        generator_states = self._pool.snapshot_generator_states(pool_rows)

        def sample_audio(logits: torch.Tensor, codebook: int) -> torch.Tensor:
            return self._pool.sample_audio(logits, codebook, pool_rows)

        try:
            frames = self.model.decode_local_frame(
                hidden_states,
                sample_audio=sample_audio,
            )
            if frames.ndim != 2 or tuple(frames.shape) != (
                len(requests),
                int(self.model.config.rvq),
            ):
                raise RuntimeError(
                    "MOSS-TTS-Realtime local decoder returned an invalid frame shape"
                )
            eos_mask = frames[:, 0].eq(int(self.model.config.audio_eos_token))
            journal = MossTTSRealtimeDecodeJournal(
                rids=rids,
                pool_rows=list(pool_rows),
                sample_positions=sample_positions,
                frames=frames,
                eos_mask=eos_mask,
                generator_states_before=generator_states,
                model_config=self.model.config,
            )
            committed_eos = self._pool.commit_frames(
                rids=rids,
                pool_rows=list(pool_rows),
                sample_positions=sample_positions,
                frames=frames,
            )
            if not torch.equal(committed_eos, eos_mask):
                raise RuntimeError("state-pool EOS commit diverged from frame journal")
        except BaseException:
            self._pool.restore_generator_states(pool_rows, generator_states)
            raise

        next_token_ids = torch.where(
            eos_mask,
            torch.full_like(provisional_ids, int(self.model.config.audio_eos_token)),
            provisional_ids,
        )
        result.moss_realtime_journal = journal
        result.next_token_ids = next_token_ids
        schedule_batch.output_ids = next_token_ids

    @staticmethod
    def _is_chunked_request(sched_req: Any) -> bool:
        req = getattr(sched_req.data, "req", None)
        if req is None:
            return False
        return int(getattr(req, "is_chunked", 0) or 0) > 0

    def finalize_skip_rids(self, scheduler_output: Any) -> set[str]:
        return {
            sched_req.request_id
            for sched_req in scheduler_output.requests
            if self._is_chunked_request(sched_req)
        }

    def lookahead_eligible(self, batch: Any) -> bool:
        del batch
        return False

    def on_generation_steps_advanced(
        self,
        advanced_steps: list[tuple[Any, int]],
        forward_batch: Any,
    ) -> None:
        del forward_batch
        for sched_req, generation_steps in advanced_steps:
            pool_position = self._pool.frame_position_for(sched_req.request_id)
            if pool_position != int(generation_steps):
                raise RuntimeError(
                    f"MOSS-TTS-Realtime generation position mismatch for "
                    f"{sched_req.request_id!r}: {pool_position} != "
                    f"{generation_steps}"
                )

    def post_process_outputs(
        self,
        result: Any,
        scheduler_output: Any,
        outputs: dict[str, RequestOutput],
    ) -> None:
        journal = getattr(result, "moss_realtime_journal", None)
        if journal is None:
            return
        expected_rids = [request.request_id for request in scheduler_output.requests]
        if journal.rids != expected_rids:
            raise RuntimeError(
                "MOSS-TTS-Realtime journal/batch alignment broken: "
                f"{journal.rids} != {expected_rids}"
            )
        if int(journal.frames.shape[0]) != len(expected_rids):
            raise RuntimeError("MOSS-TTS-Realtime journal frame count mismatch")

        frames_cpu = journal.frames.detach().to(device="cpu", dtype=torch.long)
        eos_cpu = frames_cpu[:, 0].eq(int(self.model.config.audio_eos_token))
        for index, sched_req in enumerate(scheduler_output.requests):
            rid = sched_req.request_id
            if rid not in outputs:
                raise RuntimeError(f"MOSS-TTS-Realtime output is missing {rid!r}")
            if self._pool.row_for(rid) != journal.pool_rows[index]:
                raise RuntimeError(
                    f"MOSS-TTS-Realtime journal pool ownership changed for {rid!r}"
                )
            expected_position = journal.sample_positions[index] + 1
            if self._pool.frame_position_for(rid) != expected_position:
                raise RuntimeError(
                    f"MOSS-TTS-Realtime journal position changed for {rid!r}"
                )

            data = sched_req.data
            turn_state = data.turn_state
            if turn_state is None:
                raise RuntimeError(
                    "MOSS-TTS-Realtime output requires scheduler-owned turn state"
                )
            frame = tuple(int(value) for value in frames_cpu[index].tolist())
            observed = turn_state.observe_audio_frame(
                frame,
                generation_step=journal.sample_positions[index],
            )
            if observed.is_audio_eos != bool(eos_cpu[index].item()):
                raise RuntimeError("host and device audio-EOS decisions diverged")
            if observed.is_audio_eos:
                continue
            is_first_frame = journal.sample_positions[index] == 0
            observe_first_frame = is_first_frame and realtime_events_active()
            if observe_first_frame:
                metadata = self._request_event_metadata(sched_req)
                metadata.update(
                    {
                        "frame_index": 0,
                        "ar_batch_size": len(scheduler_output.requests),
                        "backbone_can_run_cuda_graph": bool(
                            getattr(result, "can_run_cuda_graph", False)
                        ),
                    }
                )
                _emit_event(
                    request_id=rid,
                    stage=None,
                    event_name="first_codec_frame_ready",
                    metadata=metadata,
                )
            stream_metadata = getattr(data, "stream_metadata", None)
            if not stream_metadata or self._outbox is None:
                continue
            outgoing_stream_metadata = stream_metadata
            if observe_first_frame:
                outgoing_stream_metadata = {
                    **stream_metadata,
                    **realtime_identity_metadata(data.state),
                }
            self._outbox.put(
                OutgoingMessage(
                    request_id=rid,
                    type="stream",
                    target=self._vocoder_target,
                    data=frames_cpu[index].clone(),
                    metadata=outgoing_stream_metadata,
                )
            )

    def release_request(self, sched_req: Any) -> int | None:
        data = sched_req.data
        return self.model.reset_request(
            sched_req.request_id,
            getattr(data, "turn_state", None),
        )
