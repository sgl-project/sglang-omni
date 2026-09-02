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
    """

    def __init__(self, tp_worker: Any, output_processor: Any):
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

    # The base ThinkerModelRunner pins both hooks to NULL (qwen3_omni captures
    # hidden states via forward hooks instead). MiniCPM-o's talker consumes the
    # per-step last-layer hidden state through the output processor, so request
    # capture here. FULL rather than LAST: decode CUDA graphs are captured with
    # FULL (enable_return_hidden_states) and their can_run gate requires an
    # exact hidden-mode match; for decode both modes return the same rows, and
    # post_process_outputs keeps only the last row per request anyway.
    def requested_capture_hidden_mode_prefill(
        self, schedule_batch: Any, requests: list
    ) -> Any:
        del schedule_batch, requests
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

        return CaptureHiddenMode.FULL

    def requested_capture_hidden_mode_decode(
        self, schedule_batch: Any, requests: list
    ) -> Any:
        del schedule_batch, requests
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

        return CaptureHiddenMode.FULL

    def post_process_outputs(
        self,
        result: Any,
        scheduler_output: Any,
        outputs: dict[str, Any],
    ) -> None:
        """Accumulate per-step last-layer hidden states for the talker.

        ``_finalize`` merges ``extra`` into ``extra_model_outputs`` with a
        plain ``update``, which would keep only the final step's hidden. The
        talker needs the whole sequence, so collect each step's vector into
        ``hidden_states_seq``: entry 0 is the last prompt position (prefill),
        entry i>0 is the position of output token i-1 (its decode-step input).
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
            seq = sched_req.data.extra_model_outputs.setdefault("hidden_states_seq", [])
            seq.append(hidden.detach().to("cpu"))
