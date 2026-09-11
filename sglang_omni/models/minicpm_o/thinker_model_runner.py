# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o thinker model runner."""

from __future__ import annotations

from typing import Any

from sglang_omni.model_runner.thinker_model_runner import ThinkerModelRunner


class MiniCPMOThinkerModelRunner(ThinkerModelRunner):
    """Thinker runner over the MiniCPM-o text backbone.

    The base class resolves the embedding table through
    ``model.thinker.model.embed_tokens`` (satisfied by the wrapper's ``thinker``
    / ``model`` properties) but reads modality token ids from a Qwen-style
    ``hf_config.thinker_config``, which MiniCPM-o's flat config does not have.
    MiniCPM-o marks multimodal spans with ``<unk>`` runs plus bound intervals
    instead of dedicated placeholder tokens, so the id-based injection path is
    unused; the ids are set to -1 (matching no token).

    Hidden capture follows the deployment, not individual request modalities.
    Speech and explicit hidden-return configurations use FULL, matching their
    captured graphs; text-only deployments default to NULL. FULL is also needed
    for optional prefill graphs, which require an exact hidden-mode match.
    """

    def __init__(self, tp_worker: Any, output_processor: Any):
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

        # Skip ThinkerModelRunner.__init__ (it requires hf_config.thinker_config)
        # but keep its grandparent initialization.
        super(ThinkerModelRunner, self).__init__(tp_worker, output_processor)

        model = self.model
        self._outer_model = model.thinker
        self._text_model = self._outer_model.model
        self._embed_tokens = self._text_model.embed_tokens
        self._th_host_bufs = None
        self._th_slot = 0

        self._image_token_id = -1
        self._video_token_id = -1
        self._audio_token_id = -1

        self._capture_hidden_mode = (
            CaptureHiddenMode.FULL
            if output_processor._capture_hidden
            or getattr(
                tp_worker.model_runner.server_args,
                "enable_return_hidden_states",
                False,
            )
            else CaptureHiddenMode.NULL
        )

        # Per-request GPU-side hidden-state accumulators; flushed to CPU once
        # per request in on_request_finished.
        self._pending_hidden: dict[str, list[Any]] = {}

    def requested_capture_hidden_mode_prefill(
        self, schedule_batch: Any, requests: list
    ) -> Any:
        del schedule_batch, requests
        return self._capture_hidden_mode

    def requested_capture_hidden_mode_decode(
        self, schedule_batch: Any, requests: list
    ) -> Any:
        del schedule_batch, requests
        return self._capture_hidden_mode

    def post_process_outputs(
        self,
        result: Any,
        scheduler_output: Any,
        outputs: dict[str, Any],
    ) -> None:
        """Accumulate per-step last-layer hidden states for the talker.

        ``_finalize`` merges ``extra`` into ``extra_model_outputs`` with a
        plain ``update``, which would keep only the final step's hidden. The
        talker needs the whole sequence, so collect each step's vector: entry
        0 is the last prompt position (prefill), entry i>0 is the position of
        output token i-1 (its decode-step input).

        The vectors stay on the GPU here; one stack + copy per request in
        ``on_request_finished`` replaces a synchronous D2H copy per request
        per decode step. ``clone()`` is required: the slice aliases the CUDA
        graph's output buffer, which the next replay overwrites.
        """
        del result
        for sched_req in scheduler_output.requests:
            req_output = outputs.get(sched_req.request_id)
            extra = getattr(req_output, "extra", None)
            if not isinstance(extra, dict):
                continue
            hidden = extra.pop("hidden_states", None)
            if hidden is None:
                continue
            hidden = hidden.reshape(-1, hidden.shape[-1])[-1]
            seq = self._pending_hidden.setdefault(sched_req.request_id, [])
            seq.append(hidden.detach().clone())

    def on_request_finished(self, request_id: str, req_data: Any) -> None:
        """Flush the request's hidden accumulator with a single D2H copy."""
        import torch

        seq = self._pending_hidden.pop(request_id, None)
        if not seq:
            return
        stacked = torch.stack(seq).to("cpu")
        req_data.extra_model_outputs["hidden_states_seq"] = list(stacked.unbind(0))

    def reset_request(self, request_id: str) -> None:
        """Drop accumulated hidden states on abort (no terminal flush runs)."""
        self._pending_hidden.pop(request_id, None)
