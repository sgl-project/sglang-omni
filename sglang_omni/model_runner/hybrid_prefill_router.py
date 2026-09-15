# SPDX-License-Identifier: Apache-2.0
"""Shape-scoped hybrid prefill CUDA graphs: BREAKABLE small, FULL large."""

from __future__ import annotations

import logging
from typing import Any, Sequence

logger = logging.getLogger(__name__)

_HYBRID_PREFILL_BACKEND = "hybrid"
_HYBRID_FULL_BS_KEY = "cuda_graph_bs_prefill_full"


def extract_hybrid_prefill_overrides(
    server_args_overrides: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, list[int] | None]:
    """Rewrite prefill ``backend: hybrid`` into the BREAKABLE config SGLang sees.

    ``hybrid`` is an omni-level policy: ServerArgs gets a plain BREAKABLE
    prefill config (the production small-batch path), and the FULL big-bucket
    ladder (``cuda_graph_bs_prefill_full``) is returned for a second capture
    after attestation. Returns ``(overrides, None)`` untouched when hybrid is
    not selected.
    """
    from sglang_omni.scheduling.generation_batch_policy import (
        CudaGraphBackend,
        CudaGraphConfig,
        nested_prefill_overrides,
    )

    if (
        server_args_overrides
        and server_args_overrides.get("cuda_graph_backend_prefill")
        == _HYBRID_PREFILL_BACKEND
    ):
        raise ValueError(
            "backend 'hybrid' must be set via cuda_graph_config prefill, "
            "not cuda_graph_backend_prefill"
        )
    nested = nested_prefill_overrides(server_args_overrides or {})
    if nested.get("backend") != _HYBRID_PREFILL_BACKEND:
        if server_args_overrides and _HYBRID_FULL_BS_KEY in server_args_overrides:
            raise ValueError(
                f"{_HYBRID_FULL_BS_KEY} requires cuda_graph_config prefill "
                f"backend {_HYBRID_PREFILL_BACKEND!r}"
            )
        return server_args_overrides, None
    overrides = dict(server_args_overrides)
    full_bs = overrides.pop(_HYBRID_FULL_BS_KEY, None)
    if not full_bs:
        raise ValueError(
            f"cuda_graph_config prefill backend {_HYBRID_PREFILL_BACKEND!r} "
            f"requires {_HYBRID_FULL_BS_KEY} buckets"
        )
    config = overrides["cuda_graph_config"]
    config = config.to_dict() if isinstance(config, CudaGraphConfig) else dict(config)
    prefill = dict(config["prefill"])
    prefill["backend"] = CudaGraphBackend.BREAKABLE
    config["prefill"] = prefill
    overrides["cuda_graph_config"] = config
    return overrides, [int(b) for b in full_bs]


class HybridPrefillGraphRouter:
    # Note (wenyao): BREAKABLE retains the production buckets; each runner owns
    # eligibility, so routing needs no second copy of its shape thresholds.

    def __init__(self, breakable_runner: Any, full_runner: Any) -> None:
        self.breakable_runner = breakable_runner
        self.full_runner = full_runner
        # Note (wenyao): ModelWorker bisects capture_num_tokens when accounting
        # for replays, so it needs a sorted union of both runners' buckets.
        self.capture_num_tokens = sorted(
            [*breakable_runner.capture_num_tokens, *full_runner.capture_num_tokens]
        )

    def _select(self, forward_batch: Any) -> Any:
        if self.breakable_runner.can_run_graph(forward_batch):
            return self.breakable_runner
        if forward_batch.forward_mode.is_mixed():
            return None
        if self.full_runner.can_run_graph(forward_batch):
            return self.full_runner
        return None

    def can_run_graph(self, forward_batch: Any) -> bool:
        return self._select(forward_batch) is not None

    def execute(self, forward_batch: Any, **kwargs: Any) -> Any:
        runner = self._select(forward_batch)
        assert runner is not None, "execute() without a passing can_run_graph()"
        return runner.execute(forward_batch, **kwargs)


def install_hybrid_full_prefill(model_worker: Any, full_bs: Sequence[int]) -> None:
    """Capture the FULL big-bucket runner and install the hybrid router.

    Runs after ``attest_prefill_cuda_graphs`` validated the production
    BREAKABLE runner. PrefillCudaGraphRunner snapshots its whole policy from
    ``get_exec().graph.cuda_graph_config.prefill`` at ``__init__``, so the FULL
    runner is constructed under a temporary view of that config (backend,
    buckets, request slots, multimodal flag) and every field is restored:
    the rest of the process keeps seeing the production BREAKABLE config.
    """
    from sglang.srt.model_executor.cuda_graph_config import Backend
    from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
        PrefillCudaGraphRunner,
    )
    from sglang.srt.runtime_context import get_exec, get_schedule

    model_runner = model_worker.model_runner
    breakable_runner = model_runner.prefill_cuda_graph_runner
    if not isinstance(breakable_runner, PrefillCudaGraphRunner):
        raise RuntimeError(
            "hybrid prefill requires the BREAKABLE runner to have captured; "
            f"got {type(breakable_runner).__name__}"
        )
    full_bs = sorted(int(b) for b in full_bs)
    if full_bs[0] <= breakable_runner.max_num_tokens:
        raise ValueError(
            "hybrid FULL buckets must exceed the breakable ladder: "
            f"{full_bs[0]} <= {breakable_runner.max_num_tokens}"
        )

    prefill_cfg = get_exec().graph.cuda_graph_config.prefill
    model_config = model_runner.model_config
    saved = (prefill_cfg.backend, prefill_cfg.bs, prefill_cfg.full_prefill_max_req)
    saved_is_multimodal = model_config.is_multimodal

    max_req = prefill_cfg.full_prefill_max_req
    if max_req is None:
        chunked = get_schedule().chunked_prefill_size
        max_req = max(chunked // 512, 1) if chunked and chunked > 0 else 1
    max_req = min(int(max_req), model_runner.req_to_token_pool.size)

    prefill_cfg.backend = Backend.FULL
    prefill_cfg.bs = full_bs
    prefill_cfg.full_prefill_max_req = max_req
    if model_worker.enable_prefill_input_embeds:
        model_config.is_multimodal = True
    try:
        with get_schedule().override(enable_mixed_chunk=False):
            full_runner = PrefillCudaGraphRunner(model_runner)
    finally:
        model_config.is_multimodal = saved_is_multimodal
        (
            prefill_cfg.backend,
            prefill_cfg.bs,
            prefill_cfg.full_prefill_max_req,
        ) = saved

    if list(full_runner.capture_num_tokens) != full_bs:
        raise RuntimeError(
            "hybrid FULL capture shapes differ from the declared ladder: "
            f"declared={full_bs}, captured={list(full_runner.capture_num_tokens)}"
        )
    model_runner.prefill_cuda_graph_runner = HybridPrefillGraphRouter(
        breakable_runner, full_runner
    )
    logger.info(
        "hybrid prefill CUDA graphs installed: breakable buckets=%s, "
        "full buckets=%s (req slots=%d)",
        list(breakable_runner.capture_num_tokens),
        list(full_runner.capture_num_tokens),
        max_req,
    )
