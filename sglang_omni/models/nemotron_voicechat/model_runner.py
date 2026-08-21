# SPDX-License-Identifier: Apache-2.0
"""Model-output bridge for VoiceChat's per-frame auxiliary tokens."""

from __future__ import annotations

from typing import Any

from sglang_omni.model_runner.base import ModelRunner


class VoiceChatModelRunner(ModelRunner):
    """Bridge VoiceChat's per-frame payload around SGLang's eager batch copy."""

    @staticmethod
    def _custom_inputs(requests: list[Any]) -> list[dict[str, Any]]:
        return [request.data.custom_inputs for request in requests]

    def _install_custom_inputs(self, requests: list[Any]) -> None:
        # SGLang's eager runner copies ForwardBatch through a fixed-field
        # registry before calling the model, so dynamically adding a field to
        # ForwardBatch is not preserved. The model-local, one-shot payload
        # survives that copy and is safe because VoiceChat disables overlap.
        self.model.set_voicechat_custom_inputs(self._custom_inputs(requests))

    def before_prefill(
        self, forward_batch: Any, schedule_batch: Any, requests: list[Any]
    ) -> None:
        del forward_batch, schedule_batch
        self._install_custom_inputs(requests)

    def before_decode(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list[Any],
        *,
        is_lookahead: bool = False,
    ) -> None:
        del forward_batch, schedule_batch, is_lookahead
        self._install_custom_inputs(requests)

    def post_process_outputs(
        self,
        result: Any,
        scheduler_output: Any,
        outputs: dict[str, Any],
    ) -> None:
        logits_output = result.logits_output
        customized = None if logits_output is None else logits_output.customized_info
        if not isinstance(customized, dict):
            return
        for index, sched_req in enumerate(scheduler_output.requests):
            output = outputs[sched_req.request_id]
            if output.extra is None:
                output.extra = {}
            for key, values in customized.items():
                if index >= len(values):
                    raise RuntimeError(
                        f"VoiceChat customized_info {key!r} has {len(values)} "
                        f"rows for batch index {index}"
                    )
                output.extra[key] = values[index]


__all__ = ["VoiceChatModelRunner"]
