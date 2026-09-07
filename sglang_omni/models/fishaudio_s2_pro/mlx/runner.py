# SPDX-License-Identifier: Apache-2.0
"""Fish's synchronous native MLX adapter for the shared OmniScheduler."""
from __future__ import annotations

import mlx.core as mx
import numpy as np
import torch

from sglang_omni.models.fishaudio_s2_pro.model_runner import FishS2ProModelRunner
from sglang_omni.models.fishaudio_s2_pro.sglang_model import S2ProSGLangTextModel
from sglang_omni.models.fishaudio_s2_pro.torch_mps import seeded_choice

from .model import FishModel, to_mlx, to_torch


class FishMlxModel(S2ProSGLangTextModel):
    """CPU sampler/output buffers around native MLX transformer weights.

    The inherited sampler is the same one used by CUDA/MPS. Only logits cross
    to CPU for semantic sampling; the entire Fast-AR greedy chain stays in MLX.
    """

    def __init__(self, model_path, *, context_length):
        torch.nn.Module.__init__(self)
        self.native = FishModel.from_pretrained(model_path)
        self.vocab_size = self.native._config["text_config"]["vocab_size"]
        self.context_length = context_length
        self._request_caches = {}
        self._vq_ready = False

    def configure(self, tokenizer, ras_window):
        c = self.native._config["audio_decoder_config"]
        self.setup_decode_buffers(
            device="cpu",
            num_codebooks=c["num_codebooks"],
            codebook_size=c["vocab_size"],
            semantic_begin_id=tokenizer.semantic_begin_id,
            semantic_end_id=tokenizer.semantic_end_id,
            im_end_token_id=tokenizer.eos_token_ids[0],
            max_batch_size=1,
            rep_history_len=ras_window,
        )

    def _sample_semantic_choice(self, probs, seeds, positions):
        return (
            torch.multinomial(probs, num_samples=1)
            if int(seeds[0]) < 0
            else seeded_choice(probs, seeds, positions)
        )

    def clear_request(self, request_id):
        self._request_caches.pop(request_id, None)

    def forward_request(self, request, *, prefill):
        data = request.data
        rid = request.request_id
        native = self.native
        if prefill:
            if len(data.req.prefix_indices):
                raise ValueError("Fish MLX requires radix caching disabled")
            self._request_caches.clear()
            cache = native.text_model.model.make_cache()
            tokens = mx.array(data.req.origin_input_ids, dtype=mx.int32)
        else:
            if rid not in self._request_caches:
                raise RuntimeError(f"Fish MLX has no cache for {rid}")
            cache = self._request_caches[rid]
            tokens = mx.array([data.req.output_ids[-1]], dtype=mx.int32)
        if cache[0].offset + tokens.size > self.context_length:
            raise ValueError("Fish MLX request exceeds the maximum allowed length")
        embeds = native.text_model.model.embeddings(tokens)
        if prefill and data.vq_parts:
            if data.vq_mask_tokens is None:
                raise ValueError("Fish MLX reference codes require a prompt mask")
            mask = data.vq_mask_tokens.cpu().reshape(-1).numpy().astype(bool)
            parts = [to_mlx(part.T) for part in data.vq_parts]
            codes = mx.concatenate(parts, axis=0)
            if len(mask) != tokens.size or mask.sum() != codes.shape[0]:
                raise ValueError("Fish MLX reference mask/code length mismatch")
            indices = mx.array(np.flatnonzero(mask), dtype=mx.int32)
            embeds[indices] = native.audio_decoder.mix_embeddings(
                embeds[indices], codes
            )
        elif not prefill:
            token = data.req.output_ids[-1]
            if self._semantic_begin_id <= token <= self._semantic_end_id:
                if data.last_codebook_values is None:
                    raise RuntimeError("Fish MLX decode is missing codebook feedback")
                codes = to_mlx(data.last_codebook_values).reshape(1, -1)
                embeds = native.audio_decoder.mix_embeddings(embeds, codes)
        logits, hidden = native.text_model(embeds[None], cache)
        mx.eval(logits, hidden, [state.state for state in cache])
        semantic = self._sample_semantic_token(to_torch(logits))
        token = int(semantic[0])
        semantic_code = (
            0
            if token == self._im_end_token_id
            else max(0, min(token - self._semantic_begin_id, self._codebook_size - 1))
        )
        codes = native.audio_decoder.generate(hidden, semantic_code)
        mx.eval(codes)
        self._output_codes[0, 0] = token
        self._output_codes[0, 1:] = torch.from_numpy(np.array(codes[0], dtype=np.int64))
        self._output_semantic_ids[0] = token
        self._request_caches[rid] = cache


class FishMlxSchedulerRunner(FishS2ProModelRunner):
    """Keep Fish's state/stream adapters and Omni's admission/completion path."""

    def lookahead_eligible(self, batch):
        return False

    def _build_forward_batch(self, scheduler_output):
        batch = scheduler_output.batch_data
        if batch is None:
            return None
        return None, batch, bool(batch.forward_mode.is_extend())

    def before_prefill(self, forward_batch, schedule_batch, requests):
        if len(requests) != 1:
            raise ValueError("Fish MLX requires max_running_requests=1")
        self._sync_decode_state(requests)

    def before_decode(
        self, forward_batch, schedule_batch, requests, *, is_lookahead=False
    ):
        self.before_prefill(forward_batch, schedule_batch, requests)

    def _forward(self, requests, *, prefill):
        from sglang.srt.managers.scheduler import GenerationBatchResult

        self.model.forward_request(requests[0], prefill=prefill)
        return GenerationBatchResult(
            logits_output=None,
            next_token_ids=self.model._output_semantic_ids.clone(),
            can_run_cuda_graph=False,
        )

    def custom_prefill_forward(self, forward_batch, schedule_batch, requests):
        return self._forward(requests, prefill=True)

    def custom_decode_forward(self, forward_batch, schedule_batch, requests):
        return self._forward(requests, prefill=False)

    def on_request_finished(self, request_id, req_data):
        self.model.clear_request(request_id)

    def abort_request(self, request_id):
        self.model.clear_request(request_id)


class FishMlxWorkerAdapter:
    """Connect Fish's synchronous model to the registry-based MLX worker."""

    @classmethod
    def from_runtime_config(cls):
        from sglang.srt.runtime_context import get_model, get_schedule

        adapter = cls()
        adapter.pool_size = get_schedule().max_total_tokens
        adapter.scheduler_model = FishMlxModel(
            get_model().model_path, context_length=get_model().context_length
        )
        return adapter

    def prepare_for_kv_cache_release(self, req):
        # Fish has no SGLang auxiliary/radix state to snapshot. The scheduler's
        # completion/abort callbacks release its native per-request cache.
        pass


def make_fish_mlx_runner_class():
    return FishMlxWorkerAdapter
