# SPDX-License-Identifier: Apache-2.0
"""ZONOS2 model runner for OmniScheduler.

Per step the radix-cached backbone yields one hidden state per request; the
runner runs the multi-codebook head, samples all 9 DAC codes for the whole
batch at once, advances the any-of-9 EOS state machine, and stages the next
frame's summed embedding into a fixed row-indexed buffer so the backbone decode
stays CUDA-graph-replayable (decode input_ids are row indices). No frame loop.
"""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.zonos2 import callbacks
from sglang_omni.models.zonos2.radix_hash import EOS_SENTINEL, poly_row_hash
from sglang_omni.models.zonos2.sampler import sample_tts
from sglang_omni.models.zonos2.streaming_contract import (
    DEFAULT_ZONOS2_PRODUCER_FIRST_FLUSH_ROWS,
)


class Zonos2ModelRunner(ModelRunner):
    def __init__(
        self,
        tp_worker: Any,
        output_processor: Any,
        *,
        compile_sampler: bool = False,
        frame_graph: bool = False,
        async_decode: bool = False,
        stream_emit_chunk_frames: int = 1,
        stream_emit_first_chunk_frames: int = (
            DEFAULT_ZONOS2_PRODUCER_FIRST_FLUSH_ROWS
        ),
    ):
        super().__init__(tp_worker, output_processor)
        self._outbox: Any | None = None
        # Streaming emission granularity: coalesce this many newly sampled frame
        # rows into a single [k, 9] outbox message instead of one put() per row.
        # The per-frame puts (~16/step at c=16) run on the resolve host loop and
        # serialize against the next launch, defeating the async_decode D2H
        # overlap; batching cuts that put count by ~k. 1 == legacy per-frame path
        # (byte-identical default). The OLA vocoder is unaffected (same rows, same
        # order) -- this only regroups them into fewer cross-stage messages.
        self._stream_emit_chunk_frames = max(1, int(stream_emit_chunk_frames))
        # Adaptive emit: emit a SMALLER first chunk so the vocoder can produce its
        # first audio sooner (recovers the TTFC that a large steady chunk costs),
        # then fall back to the steady stream_emit_chunk_frames for throughput.
        # 0 disables (first chunk == steady chunk = fixed behavior). Only applies on
        # the coalesced path (stream_emit_chunk_frames > 1).
        self._stream_emit_first_chunk_frames = max(
            0, int(stream_emit_first_chunk_frames)
        )
        # Side stream for the lagged resolve D2H: gated by a launch event so the
        # copy waits only for codes(N), not the next forward queued on the main
        # stream -> the host resolve overlaps forward(N+1).
        self._copy_stream: Any | None = None
        # Per-request sampler (preserves the ZH-CER fix): per-request
        # temperature/top_k/top_p/min_p/repetition_penalty, no requests[0]
        # broadcast. Opt-in torch.compile fuses its elementwise launches.
        self._sampler = (
            torch.compile(sample_tts, dynamic=True) if compile_sampler else sample_tts
        )
        # Opt-in: replay the captured per-frame tail graph (head+sample+embed+hash)
        # for the GPU tail, fed rep_ids/break_mask from the on-device ring. Inert
        # unless the model captured tail graphs.
        self._frame_graph = frame_graph
        # Opt-in async-decode lookahead (overlap the resolve D2H with the next
        # forward); also gates disable_overlap_schedule in sglang_stages.
        self._async_decode = async_decode

    def set_stream_outbox(self, outbox: Any) -> None:
        self._outbox = outbox

    # ---- FeedbackAR hooks: model-specific bodies live in callbacks.py ----

    def post_process_outputs(self, result, scheduler_output, outputs) -> None:
        callbacks.extract_zonos2_output(self, result, scheduler_output, outputs)

    def custom_prefill_forward(self, forward_batch, schedule_batch, requests):
        return callbacks.zonos2_prefill_forward(
            self, forward_batch, schedule_batch, requests
        )

    def before_decode(
        self, forward_batch, schedule_batch, requests, *, is_lookahead=False
    ):
        return callbacks.write_zonos2_buffers(
            self, forward_batch, schedule_batch, requests, is_lookahead=is_lookahead
        )

    def _build_prefill_embeds(self, forward_batch, requests) -> torch.Tensor:
        model = self.model
        pieces = []
        for sr in requests:
            data = sr.data
            req = data.req
            prefix_len = len(req.prefix_indices)
            req_len = int(req.extend_range.length)
            rows = data.prompt_rows[prefix_len : prefix_len + req_len].to(model.device)
            emb = model.embed_frames(rows)
            if data.speaker_emb is not None:
                pos = int(data.speaker_position) - prefix_len
                if 0 <= pos < req_len:
                    s = model.speaker_lda_projection(
                        data.speaker_emb.to(
                            model.device, model.speaker_lda_projection.weight.dtype
                        )
                    )
                    s = model.speaker_projection(
                        s.to(model.speaker_projection.weight.dtype)
                    )
                    emb[pos] = s.to(emb.dtype)
            pieces.append(emb)
        return torch.cat(pieces, dim=0).to(device=model.device, dtype=model.dtype)

    # ---- frame collection (head + batched sample + EOS + feedback) ----

    def post_prefill(self, result, forward_batch, schedule_batch, requests):
        if bool(getattr(schedule_batch, "is_prefill_only", False)):
            return
        buf = self._collect_launch(
            result, forward_batch, schedule_batch, requests, is_prefill=True
        )
        self._collect_resolve(buf, result)

    def post_decode(self, result, forward_batch, schedule_batch, requests):
        buf = self._collect_launch(
            result, forward_batch, schedule_batch, requests, is_prefill=False
        )
        self._collect_resolve(buf, result)

    # note (Yue Yin): async-decode lookahead splits post_decode into a GPU launch
    # half (publishes next_ids on-device, no host sync) and a host resolve half
    # (the deferred output codes.to('cpu')), so the resolve D2H overlaps the next
    # decode forward. No lookahead_eligible gate needed: the rep-history ring is
    # updated on-device IN launch, so there is no one-step lag (default True holds).
    def post_decode_launch(self, result, forward_batch, requests):
        return self._collect_launch(
            result, forward_batch, None, requests, is_prefill=False
        )

    def post_decode_resolve(
        self, launch_buf, result, forward_batch, schedule_batch, requests
    ):
        self._collect_resolve(launch_buf, result)

    def _last_token_hidden(self, hidden, forward_batch, is_prefill) -> torch.Tensor:
        if not is_prefill:
            return hidden
        lens = forward_batch.extend_seq_lens
        idx = torch.cumsum(lens.to(hidden.device, torch.long), dim=0) - 1
        return hidden[idx]

    def _collect_launch(
        self, result, forward_batch, schedule_batch, requests, *, is_prefill
    ):
        model = self.model
        n = model.n_codebooks
        eoa = model.config.eoa_id
        text_pad = model.config.text_vocab
        cb_size = model.config.codebook_size

        b = len(requests)
        hidden = self._last_token_hidden(
            result.logits_output.hidden_states, forward_batch, is_prefill
        )
        if hidden.shape[0] < b:
            raise ValueError(
                f"hidden batch size ({hidden.shape[0]}) < len(requests) ({b})"
            )
        hidden = hidden[:b]
        pool = model._decode_state_pool
        row_t = pool.prepare_active_rows(requests)
        # Per-request sampling params (preserves the ZH-CER fix): each request's
        # own temperature/top_k/top_p/min_p/repetition_penalty, no requests[0]
        # broadcast. window/codebooks are uniform (requests[0]) as in _rep_window.
        params = requests[0].data.params
        p = [sr.data.params for sr in requests]
        dev = hidden.device

        # Compose with the tail CUDA-graph (ZONOS2_FRAME_GRAPH): when armed, run
        # head+sample+embed+hash as one captured replay, with rep_ids/break_mask
        # fed from the ON-DEVICE ring (not host-built) so no host work re-enters.
        use_graph = (
            self._frame_graph
            and not is_prefill
            and model._tail_buckets
            and b <= model._tail_buckets[-1]
            and all(self._params_match(x, model._tail_params) for x in p)
        )
        if use_graph:
            rep_ids = self._rep_window_ring(
                row_t, n, int(params.repetition_window), cb_size, dev
            )
            break_mask = self._break_mask_ring(row_t, model.audio_vocab, dev)
            codes, keys, feedback = model.run_tail_graph(
                hidden,
                torch.tensor([x.temperature for x in p], device=dev),
                torch.tensor([x.top_k for x in p], device=dev),
                torch.tensor([x.top_p for x in p], device=dev),
                torch.tensor([x.min_p for x in p], device=dev),
                torch.tensor([x.repetition_penalty for x in p], device=dev),
                rep_ids,
                break_mask,
            )
        else:
            logits = model.compute_logits(hidden).float()  # [B, 9, 1026]
            rep_ids = self._rep_window(row_t, n, cb_size, params)
            self._break_frame_loops(logits, row_t)
            top_k_max = max(
                (x.top_k for x in p if 0 < x.top_k < model.audio_vocab), default=0
            )
            any_top_p = any(0.0 < x.top_p < 1.0 for x in p)
            any_min_p = any(x.min_p > 0.0 for x in p)
            codes = self._sampler(
                logits,
                temperature=torch.tensor([x.temperature for x in p], device=dev),
                top_k=torch.tensor([x.top_k for x in p], device=dev),
                top_p=torch.tensor([x.top_p for x in p], device=dev),
                min_p=torch.tensor([x.min_p for x in p], device=dev),
                repetition_penalty=torch.tensor(
                    [x.repetition_penalty for x in p], device=dev
                ),
                top_k_max=top_k_max,
                rep_token_ids=rep_ids,
                any_top_p=any_top_p,
                any_min_p=any_min_p,
            )  # [B, 9]
            text_col = torch.full(
                (b, 1), text_pad, device=codes.device, dtype=torch.long
            )
            rows = torch.cat([codes, text_col], dim=1)
            feedback = model.embed_frames(rows)  # [B, dim]
            keys = poly_row_hash(rows)  # [B] (< RADIX_HASH_SPACE)

        # Stage next-step feedback into the on-device pool row (one batched scatter).
        pool.feedback_embeds[row_t] = feedback.detach()
        # Append this frame to the rep-history ring (read by the rep functions on
        # the NEXT step; matches the scalar rep_hist.append timing: read-then-append).
        pool.rep_hist[row_t] = torch.cat(
            [pool.rep_hist[row_t][:, 1:, :], codes.unsqueeze(1)], dim=1
        )
        pool.rep_len[row_t] = torch.clamp_max(pool.rep_len[row_t] + 1, pool.rep_ring)

        # note (Yue Yin): vectorized EOS state machine on the on-device pool rows
        # (replaces the per-request Python loop + keys.tolist()). Byte-exact to the
        # scalar loop (unit-verified): eos_frame = max(0, step - rightmost eoa col),
        # set on the first eoa frame; countdown = n+1 then decremented while > 0;
        # finished = countdown <= 0; next_ids = radix key, or EOS_SENTINEL when done.
        jidx = torch.arange(n, device=codes.device)
        hits = codes == eoa  # [B, n]
        any_hit = hits.any(dim=1)
        last_hit_j = (hits.long() * jidx).amax(dim=1)
        gstep = pool.generation_step[row_t]
        first_set = any_hit & (~pool.eos_frame_set[row_t])
        new_eos = torch.clamp_min(gstep - last_hit_j, 0)
        pool.eos_frame_val[row_t] = torch.where(
            first_set, new_eos, pool.eos_frame_val[row_t]
        )
        pool.eos_countdown[row_t] = torch.where(
            first_set, torch.full_like(gstep, n + 1), pool.eos_countdown[row_t]
        )
        pool.eos_frame_set[row_t] = pool.eos_frame_set[row_t] | first_set
        cd = pool.eos_countdown[row_t]
        dec = pool.eos_frame_set[row_t] & (cd > 0)
        cd = torch.where(dec, cd - 1, cd)
        pool.eos_countdown[row_t] = cd
        finished = pool.eos_frame_set[row_t] & (cd <= 0)
        pool.generation_step[row_t] = gstep + 1
        next_ids = torch.where(finished, torch.full_like(keys, EOS_SENTINEL), keys)
        result.next_token_ids = next_ids

        # launch_buf snapshots what the (lagged) resolve reads. Pack codes +
        # per-row EOS metadata into one fresh int64 tensor (advanced indexing
        # already copies, and cat makes a fresh tensor, so it remains stable
        # while the next launch mutates these pool rows). Record a CUDA event
        # right after, so resolve's side-stream D2H waits only for this (codes
        # ready), not the next forward. next_ids is cloned: the base aliases it
        # onto output_ids, overwritten in place before this resolve.
        meta = torch.stack(
            (
                pool.eos_frame_set[row_t].to(torch.int64),
                pool.eos_frame_val[row_t],
            ),
            dim=1,
        )
        packed = torch.cat((codes, meta), dim=1)  # [B, n+2] int64
        ev = torch.cuda.Event()
        ev.record()
        return (requests, packed, n, next_ids.clone(), ev)

    def _collect_resolve(self, launch_buf, result) -> None:
        # Host half (runs lagged under async, overlapping the next decode forward):
        # the deferred codes.to('cpu') -> per-request output_codes + eos_frame.
        # Restores next_ids to the launch-time (un-clobbered) value for the
        # shared finalize tail. Request release belongs to the scheduler's
        # terminal result / abort adapters, which cover every finish reason.
        if launch_buf is None:
            return
        requests, packed, n, next_ids_snap, ev = launch_buf
        if result is not None:
            result.next_token_ids = next_ids_snap
        # Copy on a side stream gated by the launch event: the copy starts as soon
        # as codes(N) is ready (event recorded before the next forward was queued)
        # and runs on a separate stream, so this no longer whole-stream-syncs on
        # forward(N+1). One D2H of the packed [B, n+2] snapshot.
        if self._copy_stream is None:
            self._copy_stream = torch.cuda.Stream(device=packed.device)
        self._copy_stream.wait_event(ev)
        with torch.cuda.stream(self._copy_stream):
            packed_cpu = packed.to("cpu", non_blocking=True)
        self._copy_stream.synchronize()
        codes_cpu = packed_cpu[:, :n]
        eos_set_cpu = packed_cpu[:, n]
        eos_val_cpu = packed_cpu[:, n + 1]
        for i, sr in enumerate(requests):
            data = sr.data
            data.output_codes.append(codes_cpu[i].clone())
            data.eos_frame = int(eos_val_cpu[i]) if bool(eos_set_cpu[i]) else None

    def _rep_window(self, row_t, n, cb_size, params):
        # Vectorized rep-penalty token window from the on-device history ring.
        # Byte-exact to the old per-request build (unit-tested): an all -1 window
        # is a no-op penalty == None, so no special first-step handling is needed.
        if params.repetition_penalty == 1.0:
            return None
        w = params.repetition_window
        ring = self.model._decode_state_pool.rep_hist[row_t]  # [B, ring, n]
        t = ring[:, -w:, :].transpose(1, 2)  # last w frames -> [B, n, w]
        rc = min(params.repetition_codebooks, n)
        rep = torch.full_like(t, -1)
        rep[:, :rc] = torch.where(t[:, :rc] < cb_size, t[:, :rc], rep[:, :rc])
        return rep

    def _break_frame_loops(self, logits, row_t, run: int = 8):
        # Loop-collapse guard (vectorized over the on-device history ring): mask
        # the primary-codebook token where a request's full 9-codebook frame has
        # repeated identically for `run` steps -- a degenerate loop the windowed
        # rep-penalty misses (validated WER/CER-neutral). Byte-exact to the scalar.
        pool = self.model._decode_state_pool
        ring = pool.rep_hist[row_t]  # [B, ring, n]
        last = ring[:, -1, :]  # [B, n]
        window = ring[:, -run:, :]  # [B, run, n]
        all_eq = (window == last.unsqueeze(1)).all(dim=2).all(dim=1)
        mask = all_eq & (pool.rep_len[row_t] >= run) & (last[:, 0] >= 0)
        bi = torch.arange(logits.shape[0], device=logits.device)
        tok = last[:, 0].clamp(min=0)
        cur = logits[bi, 0, tok]
        logits[bi, 0, tok] = torch.where(mask, torch.full_like(cur, float("-inf")), cur)

    @staticmethod
    def _params_match(a, b) -> bool:
        # The tail graph bakes the structural sampler flags (top_k_max/any_top_p/
        # any_min_p), so only replay when the request's params match those captured.
        if b is None:
            return False
        return (
            a.temperature == b.temperature
            and a.top_k == b.top_k
            and a.top_p == b.top_p
            and a.min_p == b.min_p
            and a.repetition_penalty == b.repetition_penalty
            and a.repetition_window == b.repetition_window
            and a.repetition_codebooks == b.repetition_codebooks
        )

    def _rep_window_ring(self, row_t, n, w, cb_size, device):
        # Fixed-shape [B, n, w] rep window from the on-device ring, feeding the
        # tail graph's captured rep_ids input (mirrors _rep_window_graph but reads
        # the GPU ring -- no host rep_hist build). Params are uniform here (graph
        # replay requires params == _tail_params), so codebooks come from those.
        ring = self.model._decode_state_pool.rep_hist[row_t]  # [B, ring, n]
        t = ring[:, -w:, :].transpose(1, 2).contiguous()  # [B, n, w]
        rc = min(self.model._tail_params.repetition_codebooks, n)
        rep = torch.full_like(t, -1)
        rep[:, :rc] = torch.where(t[:, :rc] < cb_size, t[:, :rc], rep[:, :rc])
        return rep

    def _break_mask_ring(self, row_t, vocab, device, run: int = 8):
        # Additive [B, vocab] codebook-0 mask (-inf at a looping token) from the
        # on-device ring, feeding the tail graph's break_mask input (mirrors
        # _break_mask_graph + _break_frame_loops, vectorized on the GPU ring).
        pool = self.model._decode_state_pool
        ring = pool.rep_hist[row_t]  # [B, ring, n]
        last = ring[:, -1, :]  # [B, n]
        window = ring[:, -run:, :]  # [B, run, n]
        all_eq = (window == last.unsqueeze(1)).all(dim=2).all(dim=1)
        active = all_eq & (pool.rep_len[row_t] >= run) & (last[:, 0] >= 0)
        mask = torch.zeros(last.shape[0], vocab, device=device, dtype=torch.float32)
        tok = last[:, 0].clamp(min=0)
        vals = torch.where(
            active,
            torch.full((last.shape[0],), float("-inf"), device=device),
            torch.zeros(last.shape[0], device=device),
        )
        mask.scatter_(1, tok.unsqueeze(1), vals.unsqueeze(1))
        return mask
