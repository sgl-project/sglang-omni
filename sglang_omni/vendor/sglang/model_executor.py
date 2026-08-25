# SPDX-License-Identifier: Apache-2.0
"""Vendor wrapper for sglang.srt.model_executor.runner.decode_cuda_graph_runner.

Patches ``DecodeCudaGraphRunner.can_run_graph`` and
``_can_run_ragged_verify_graph`` to remove an implicit GPU sync that fires on
every decode step for encoder-decoder models (e.g. Whisper ASR): both methods
compute

    is_encoder_lens_supported = (
        torch.all(forward_batch.encoder_lens > 0) if self.is_encoder_decoder else True
    )

``torch.all(...)`` on a CUDA tensor returns a 0-dim CUDA bool tensor; the
surrounding ``if`` forces ``bool()`` on it, which triggers a device
synchronize + device-to-host copy. ``ForwardBatch`` already carries a
CPU-side mirror of the same data (``encoder_lens_cpu: Optional[List[int]]``),
so we compute the identical predicate from that list instead and only fall
back to the original tensor-based check when the CPU mirror isn't populated
(preserving exact behavior for any caller that doesn't set it).

Captured against sglang==0.5.16
(sglang/srt/model_executor/runner/decode_cuda_graph_runner.py, lines
502-631). Re-diff ``can_run_graph`` / ``_can_run_ragged_verify_graph``
against upstream on version bumps -- this is a full-method override, not a
call-through wrapper, because the sync sits in the middle of a larger
boolean expression.
"""

from __future__ import annotations

import torch
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
)
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.multiplex.pdmux_context import get_current_stream_idx
from sglang.srt.speculative.ragged_verify import resolve_ragged_verify_layout


def _is_encoder_lens_supported(
    self: DecodeCudaGraphRunner, forward_batch: ForwardBatch
) -> bool:
    if not self.is_encoder_decoder:
        return True
    encoder_lens_cpu = forward_batch.encoder_lens_cpu
    if encoder_lens_cpu is not None:
        return all(v > 0 for v in encoder_lens_cpu)
    return bool(torch.all(forward_batch.encoder_lens > 0))


def _can_run_graph(self: DecodeCudaGraphRunner, forward_batch: ForwardBatch) -> bool:
    # Disable for token embedding overrides (dynamic per-request)
    if forward_batch.replace_embeds is not None:
        return False

    ragged_layout = (
        resolve_ragged_verify_layout(forward_batch) if self.ragged_verify_mode else None
    )
    if ragged_layout is not None:
        return self._can_run_ragged_verify_graph(forward_batch, ragged_layout)
    if self.ragged_verify_mode and forward_batch.forward_mode.is_target_verify():
        return False

    # Uniform-width replay invariant: the batch's actual per-request width
    # must match this runner's capture width; anything else falls back to
    # eager. (Unset widths pass: not every path fills the field yet.)
    spec_info = forward_batch.spec_info
    if (
        spec_info is not None
        and spec_info.num_tokens_per_req > 0
        and spec_info.num_tokens_per_req != self.captured_req_width
    ):
        return False

    if self.require_mlp_tp_gather:
        # Raw sync values are per-rank request counts on decode-family
        # rounds -- no width division, no per-algorithm enumeration.
        cuda_graph_bs = max(forward_batch.original_global_num_tokens_cpu)
    else:
        cuda_graph_bs = forward_batch.batch_size

    graph_key = self._make_graph_key(
        cuda_graph_bs,
        stream_idx=get_current_stream_idx() if self.enable_pdmux else None,
        variant_label=self._resolve_lora_variant(forward_batch),
    )

    is_bs_supported = (
        self.backend.can_run(forward_batch, graph_key)
        if self.disable_padding
        else cuda_graph_bs <= self.max_bs
    )

    if self.require_mlp_sync:
        is_bs_supported = is_bs_supported and forward_batch.can_run_dp_cuda_graph

    # NOTE: cuda graph cannot handle mixed batch (encoder_len = 0)
    # If mixed batch cannot be supported, then encoder_lens can be removed in cuda graph
    # because the full_text_row_masked_out_mask tensor will always be ones
    # omni patch: sync-free equivalent of torch.all(forward_batch.encoder_lens > 0)
    is_encoder_lens_supported = _is_encoder_lens_supported(self, forward_batch)

    requested_capture_hidden_mode = max(
        forward_batch.capture_hidden_mode,
        (
            forward_batch.spec_info.capture_hidden_mode
            if getattr(forward_batch.spec_info, "capture_hidden_mode", None) is not None
            else CaptureHiddenMode.NULL
        ),
    )
    capture_hidden_mode_matches = (
        requested_capture_hidden_mode == CaptureHiddenMode.NULL
        or requested_capture_hidden_mode == self.capture_hidden_mode
    )
    is_tbo_supported = (
        forward_batch.can_run_tbo if self.enable_two_batch_overlap else True
    )

    is_ngram_supported = (
        (
            forward_batch.batch_size * self.captured_req_width
            == forward_batch.input_ids.numel()
        )
        if self.model_runner.spec_algorithm.is_ngram()
        else True
    )

    return (
        is_bs_supported
        and is_encoder_lens_supported
        and is_tbo_supported
        and capture_hidden_mode_matches
        and is_ngram_supported
    )


def _can_run_ragged_verify_graph(
    self: DecodeCudaGraphRunner, forward_batch: ForwardBatch, ragged_layout
) -> bool:
    if not self.attn_backend.supports_ragged_verify_graph:
        return False

    admission_tokens = ragged_layout.graph_num_tokens
    is_tokens_supported = admission_tokens <= self.capture_num_tokens[
        -1
    ] and forward_batch.batch_size <= self._ragged_capture_slots(admission_tokens)

    is_dp_supported = (
        forward_batch.can_run_dp_cuda_graph if self.require_mlp_sync else True
    )

    # omni patch: sync-free equivalent of torch.all(forward_batch.encoder_lens > 0)
    is_encoder_lens_supported = _is_encoder_lens_supported(self, forward_batch)

    requested_capture_hidden_mode = max(
        forward_batch.capture_hidden_mode,
        (
            forward_batch.spec_info.capture_hidden_mode
            if getattr(forward_batch.spec_info, "capture_hidden_mode", None) is not None
            else CaptureHiddenMode.NULL
        ),
    )
    capture_hidden_mode_matches = (
        requested_capture_hidden_mode == CaptureHiddenMode.NULL
        or requested_capture_hidden_mode == self.capture_hidden_mode
    )

    return (
        is_tokens_supported
        and is_dp_supported
        and is_encoder_lens_supported
        and capture_hidden_mode_matches
    )


# Patch the source class so any direct imports also see the change.
DecodeCudaGraphRunner.can_run_graph = _can_run_graph
DecodeCudaGraphRunner._can_run_ragged_verify_graph = _can_run_ragged_verify_graph

__all__ = ["DecodeCudaGraphRunner"]
