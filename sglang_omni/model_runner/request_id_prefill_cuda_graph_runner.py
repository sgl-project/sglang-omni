# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
from typing_extensions import Unpack

from sglang_omni.model_runner.whisper_prefill_cuda_graph_runner import (
    PrefillReplayKwargs,
)


class RequestIdPrefillCudaGraphRunner(PrefillCudaGraphRunner):
    def load_batch(
        self,
        forward_batch: ForwardBatch,
        **kwargs: Unpack[PrefillReplayKwargs],
    ) -> ForwardBatch:
        static_forward_batch = super().load_batch(forward_batch, **kwargs)
        static_forward_batch.rids = forward_batch.rids
        return static_forward_batch


__all__ = ["RequestIdPrefillCudaGraphRunner"]
