# SPDX-License-Identifier: Apache-2.0
"""Higgs scheduler adapter for MLX forwards and request KV lifecycle."""
from __future__ import annotations

import torch

from sglang_omni.models.higgs_tts.torch_mps_runner import HiggsTorchMpsModelRunner


class HiggsMlxModelRunner(HiggsTorchMpsModelRunner):
    """Single active request; native MLX transformer and per-request KV cache."""

    def __init__(self, tp_worker, output_processor):
        super().__init__(tp_worker, output_processor)
        owner = getattr(tp_worker, "_mlx_runner", None)
        if owner is not None:
            self._past_key_values = owner.request_caches

    def _build_forward_batch(self, scheduler_output):
        from types import SimpleNamespace

        from sglang_omni.model_runner.base import resolve_deferred_prefill_inputs

        batch = scheduler_output.batch_data
        if batch is None:
            return None
        resolve_deferred_prefill_inputs(batch, torch.device("mps"))
        view = SimpleNamespace(
            input_ids=batch.input_ids.to("mps"),
            batch_size=len(batch.reqs),
            sampling_info=batch.sampling_info,
            forward_mode=batch.forward_mode,
            replace_embeds=None,
        )
        return view, batch, bool(batch.forward_mode.is_extend())

    def _forward(self, request_id, embeddings, *, prefill):
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache
        from sglang.srt.utils.tensor_bridge import mlx_to_torch, torch_to_mlx

        if prefill:
            self._past_key_values.pop(request_id, None)
            cache = [KVCache() for _ in self.model.mlx_language_model.layers]
        else:
            if request_id not in self._past_key_values:
                raise RuntimeError(f"Higgs MLX decode has no KV cache for {request_id}")
            cache = self._past_key_values[request_id]
        try:
            inputs = torch_to_mlx(embeddings.unsqueeze(0))
            hidden = self.model.mlx_language_model(
                None, cache=cache, input_embeddings=inputs
            )[:, -1, :]
            mx.eval(hidden, [c.state for c in cache])
            result = mlx_to_torch(hidden, device=embeddings.device)
        except Exception:
            self.reset_request(request_id)
            raise
        self._past_key_values[request_id] = cache
        return result
