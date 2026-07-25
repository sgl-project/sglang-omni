# SPDX-License-Identifier: Apache-2.0
"""FeedbackAR callbacks for ZONOS2 (model-dir convention).

The three model-specific hooks the (future shared) FeedbackARModelRunner drives
around the backbone forward:

  - write_zonos2_buffers   — pre-decode: stage the previous step's feedback into
    the row-indexed decode buffer (write_buffers_fn)
  - extract_zonos2_output  — post-forward: push newly sampled codes to the vocoder
    (extract_output_fn)
  - zonos2_prefill_forward — custom prefill with the speaker-injected summed embeds
    (prefill_forward_fn)

Each takes the runner so it can reach model buffers + per-request stream state;
``Zonos2ModelRunner`` delegates its hooks here. Behavior is byte-identical to the
pre-refactor inline runner hooks (pure structural extraction).
"""

from __future__ import annotations

import torch

from sglang_omni.scheduling.messages import OutgoingMessage


def write_zonos2_buffers(
    runner, forward_batch, schedule_batch, requests, *, is_lookahead=False
):
    del schedule_batch, is_lookahead
    n_real = len(requests)
    if n_real == 0:
        return
    bs = int(forward_batch.batch_size)
    if bs < n_real:
        raise ValueError(f"forward_batch.batch_size ({bs}) < len(requests) ({n_real})")
    buf = runner.model._decode_input_embedding.weight
    # note (Yue Yin): gather each request's last feedback from its on-device
    # pool row into the positional decode buffer (buf[i] = request i), instead
    # of a per-request Python deque. Byte-identical: the deque held exactly the
    # latest feedback the pool row now holds. Reconcile-release finished rows.
    pool = runner.model._decode_state_pool
    pool.release_inactive({sr.request_id for sr in requests})
    row_t = pool.prepare_active_rows(requests)
    with torch.no_grad():
        buf[:n_real].copy_(pool.feedback_embeds[row_t])
        buf[n_real:bs].zero_()
    # Decode reads the staged buffer by row index -> stable input for graph replay.
    forward_batch.input_ids = torch.arange(bs, device=buf.device, dtype=torch.long)
    forward_batch.input_embeds = None


def extract_zonos2_output(runner, result, scheduler_output, outputs) -> None:
    # note (Yue Yin): additive stream hook — push each newly sampled delayed
    # [9] row to the vocoder; reads the already-CPU output_codes and never
    # touches the decode/EOS/feedback state.
    # When stream_emit_chunk_frames > 1, coalesce the newly sampled rows into
    # one [k, 9] message instead of one put() per row: the per-frame puts run
    # on the resolve host loop and serialize against the next launch, defeating
    # the async_decode D2H overlap. Same rows in the same order reach the OLA
    # decoder, so the audio is unchanged — only the message grouping differs.
    del result, outputs
    if runner._outbox is None:
        return
    chunk = runner._stream_emit_chunk_frames
    for sched_req in scheduler_output.requests:
        data = sched_req.data
        stream_metadata = getattr(data, "stream_metadata", None)
        if stream_metadata is None:
            continue
        done = False
        req = getattr(data, "req", None)
        if req is not None:
            finished = getattr(req, "finished", None)
            done = (callable(finished) and finished()) or bool(
                getattr(req, "is_retracted", False)
            )
        codes = data.output_codes
        start = int(data._stream_emit_idx)
        n_new = len(codes) - start
        if n_new <= 0:
            continue
        if chunk == 1:
            # Legacy per-frame path (byte-identical default): one put per row;
            # a finishing step's tail is left to the on_stream_done flush.
            if done:
                continue
            for row in codes[start:]:
                runner._outbox.put(
                    OutgoingMessage(
                        request_id=sched_req.request_id,
                        type="stream",
                        target="vocoder",
                        data=row.clone(),
                        metadata=stream_metadata,
                    )
                )
            data._stream_emit_idx = len(codes)
            continue
        # Coalesced path: hold rows until >= threshold have accumulated, but
        # always flush the remainder on finish so the OLA decoder receives every
        # row (on_stream_done's eos_frame cap trims the tail to the aligned
        # length). Adaptive: the FIRST chunk uses a smaller threshold so the
        # first audio is produced sooner (lower TTFC); steady chunks use `chunk`.
        first = runner._stream_emit_first_chunk_frames
        threshold = first if (first > 0 and start == 0) else chunk
        if not done and n_new < threshold:
            continue
        rows = torch.stack(list(codes[start:]), dim=0)
        runner._outbox.put(
            OutgoingMessage(
                request_id=sched_req.request_id,
                type="stream",
                target="vocoder",
                data=rows.clone(),
                metadata=stream_metadata,
            )
        )
        data._stream_emit_idx = len(codes)


def zonos2_prefill_forward(runner, forward_batch, schedule_batch, requests):
    del schedule_batch
    forward_batch.input_embeds = runner._build_prefill_embeds(forward_batch, requests)
    return None
