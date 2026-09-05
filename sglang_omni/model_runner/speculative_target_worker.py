# SPDX-License-Identifier: Apache-2.0
"""Expose an already loaded Omni target through SGLang's speculative interface."""

from __future__ import annotations

from typing import Any

from sglang.srt.distributed.parallel_state import get_pp_group
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.utils.hf_transformers_utils import get_tokenizer


class SpeculativeTargetWorker(TpModelWorker):
    """Reuse upstream prefill sampling and verification without loading a target twice."""

    def __init__(self, worker: Any) -> None:
        self._worker = worker
        self._model_runner = worker.model_runner
        self.ps = self.model_runner.ps
        self.pp_group = get_pp_group()
        self.model_runner_list = []
        self.enable_overlap = False
        self.enable_spec = True
        self.hicache_layer_transfer_counter = None
        self.tokenizer = get_tokenizer(
            worker.server_args.tokenizer_path or worker.server_args.model_path,
            tokenizer_mode=worker.server_args.tokenizer_mode,
            trust_remote_code=worker.server_args.trust_remote_code,
            revision=worker.server_args.revision,
            tokenizer_backend=worker.server_args.tokenizer_backend,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._worker, name)
