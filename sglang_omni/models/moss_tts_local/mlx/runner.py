# SPDX-License-Identifier: Apache-2.0
"""SGLang MLX runner extension for MOSS-TTS Local frame generation."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import mlx.core as mx
import torch
from mlx_lm.utils import load_model
from sglang.srt.hardware_backend.mlx.kv_cache import ContiguousAttentionKVCache
from sglang.srt.hardware_backend.mlx.model_runner import (
    MlxModelRunner,
    MlxPendingDecode,
    MlxPendingPrefill,
)
from sglang.srt.hardware_backend.mlx.remote_code_gate import (
    ensure_remote_code_allowed,
    resolve_model_directory,
)
from sglang.srt.hardware_backend.mlx.sampling import MlxLogprobSpec
from sglang.srt.managers.schedule_batch import Req

from sglang_omni.models.moss_tts_local.mlx.config import ModelConfig
from sglang_omni.models.moss_tts_local.mlx.model import MossTTSLocalModel
from sglang_omni.models.moss_tts_local.request_builders import (
    MossTTSLocalSGLangRequestData,
)

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class MossPendingPrefill(MlxPendingPrefill):
    moss_row: mx.array
    moss_step: int


@dataclass(kw_only=True)
class MossPendingDecode(MlxPendingDecode):
    moss_rows: mx.array
    moss_step: int


def apply_top_k(logits: mx.array, value: int) -> mx.array:
    if value <= 0 or value >= logits.shape[-1]:
        return logits
    else:
        pass
    threshold = mx.topk(logits, k=value, axis=-1)[..., :1]
    return mx.where(logits < threshold, -mx.inf, logits)


def apply_top_p(logits: mx.array, value: float) -> mx.array:
    if value >= 1.0:
        return logits
    else:
        pass
    order = mx.argsort(-logits, axis=-1)
    sorted_logits = mx.take_along_axis(logits, order, axis=-1)
    cumulative = mx.cumsum(mx.softmax(sorted_logits, axis=-1), axis=-1)
    remove = mx.concatenate(
        [mx.zeros_like(cumulative[..., :1]), cumulative[..., :-1] >= value], axis=-1
    )
    filtered = mx.where(remove, -mx.inf, sorted_logits)
    inverse = mx.argsort(order, axis=-1)
    return mx.take_along_axis(filtered, inverse, axis=-1)


def sample(
    logits: mx.array,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    position: int,
) -> mx.array:
    if temperature <= 0:
        return mx.argmax(logits, axis=-1).astype(mx.int32)
    else:
        pass
    logits = logits.astype(mx.float32) / temperature
    logits = apply_top_k(logits, top_k)
    logits = apply_top_p(logits, top_p)
    key = mx.random.key((int(seed) + int(position)) & 0xFFFFFFFF)
    return mx.random.categorical(logits, axis=-1, key=key).astype(mx.int32)


class MossTTSLocalMlxModelRunner(MlxModelRunner):
    """One-request native MLX runner with lazy, chainable frame decoding."""

    @property
    def config(self) -> ModelConfig:
        return self.model.config

    def _load_model(
        self,
    ) -> None:  # noqa: leading-underscore  # SGLang initialization hook.
        model_path = resolve_model_directory(self.model_path, revision=self.revision)
        ensure_remote_code_allowed(model_path, self.trust_remote_code)
        logger.info(f"Loading native MLX MOSS-TTS Local model: {model_path}")
        started = time.perf_counter()
        self.model, _ = load_model(
            model_path,
            get_model_classes=lambda config: (MossTTSLocalModel, ModelConfig),
        )
        elapsed_seconds = time.perf_counter() - started
        logger.info(f"Loaded native MLX MOSS-TTS Local model in {elapsed_seconds:.2f}s")
        self.request_rows: dict[str, mx.array] = {}
        self.request_steps: dict[str, int] = {}
        self.request_params: dict[str, MossTTSLocalSGLangRequestData] = {}
        self.completed_rows: dict[str, list[list[int]]] = {}

    def decode_frame(self, req_id: str, hidden: mx.array, step: int) -> mx.array:
        params = self.request_params[req_id]
        channels = self.model.config.channels

        def sample_text(logits: mx.array) -> mx.array:
            return sample(
                logits,
                temperature=params.text_temperature,
                top_p=params.text_top_p,
                top_k=params.text_top_k,
                seed=params.sampling_seed,
                position=step * channels,
            )

        def sample_audio(logits: mx.array, channel: int) -> mx.array:
            return sample(
                logits,
                temperature=params.audio_temperature,
                top_p=params.audio_top_p,
                top_k=params.audio_top_k,
                seed=params.sampling_seed,
                position=step * channels + channel + 1,
            )

        return self.model.decode_frame(
            hidden, sample_text=sample_text, sample_audio=sample_audio
        )

    def prefill_start(
        self,
        req_id: str,
        new_token_ids: list[int],
        full_token_ids: list[int],
        prefix_slot_ids: list[int],
        new_slot_ids: list[int],
        req_pool_idx: int,
        req: Req | None = None,
        needs_logits: bool = True,
        logit_edit_row: mx.array | None = None,
        logprob_spec: MlxLogprobSpec | None = None,
    ) -> MossPendingPrefill:
        assert req is not None
        assert not prefix_slot_ids and self.disable_radix_cache
        if logit_edit_row is not None or logprob_spec is not None:
            raise NotImplementedError(
                "MOSS-TTS Local MLX does not expose text logprobs"
            )
        else:
            pass

        data: MossTTSLocalSGLangRequestData = req.omni_data
        if data.audio_repetition_penalty != 1.0:
            raise NotImplementedError(
                "MOSS-TTS Local MLX currently requires audio_repetition_penalty=1"
            )
        else:
            pass
        self.request_params[req_id] = data
        rows = data.prompt_rows
        if data.output_rows:
            rows = torch.cat([rows, torch.stack(data.output_rows)], dim=0)
        else:
            pass
        rows_mx = mx.array(rows.detach().cpu().numpy(), dtype=mx.int32)[None, ...]
        cache = self._acquire_cache()  # noqa: leading-underscore  # SGLang MLX API
        hidden = self.model.backbone(rows_mx, cache)[:, -1, :]
        step = len(data.output_rows)
        next_row = self.decode_frame(req_id, hidden, step)
        return MossPendingPrefill(
            lazy_token=next_row[:, 0],
            cache=cache,
            req_id=req_id,
            full_token_ids=list(full_token_ids),
            req_pool_idx=req_pool_idx,
            synced_offset=0,
            lazy_logprobs=None,
            moss_row=next_row,
            moss_step=step,
        )

    def prefill_finalize(self, pending: MossPendingPrefill) -> int:
        token = super().prefill_finalize(pending)
        row = [int(value) for value in pending.moss_row[0].tolist()]
        self.request_rows[pending.req_id] = pending.moss_row[:, None, :]
        self.request_steps[pending.req_id] = pending.moss_step + 1
        self.completed_rows.setdefault(pending.req_id, []).append(row)
        return token

    def decode_pending(
        self,
        req_ids: list[str],
        rows: mx.array,
        caches: list[list[ContiguousAttentionKVCache]],
        step: int,
    ) -> MossPendingDecode:
        hidden = self.model.backbone(rows, caches[0])[:, -1, :]
        next_rows = self.decode_frame(req_ids[0], hidden, step)
        return MossPendingDecode(
            lazy_tokens=next_rows[:, 0],
            req_ids=req_ids,
            caches=caches,
            lazy_logprobs=None,
            logprob_spec=None,
            edit_rows=None,
            moss_rows=next_rows,
            moss_step=step,
        )

    def decode_batch_start(
        self,
        req_ids: list[str],
        edit_rows: mx.array | None = None,
        logprob_spec: MlxLogprobSpec | None = None,
        logits_hook: None = None,
    ) -> MossPendingDecode:
        if len(req_ids) != 1:
            raise NotImplementedError(
                "MOSS-TTS Local MLX currently supports one request"
            )
        else:
            pass
        if edit_rows is not None or logprob_spec is not None or logits_hook is not None:
            raise NotImplementedError(
                "MOSS-TTS Local MLX does not expose text logprobs"
            )
        else:
            pass
        rid = req_ids[0]
        return self.decode_pending(
            req_ids,
            self.request_rows[rid],
            [self._req_caches[rid]],  # noqa: leading-underscore  # SGLang MLX API
            self.request_steps[rid],
        )

    def decode_batch_start_chained(
        self, previous: MossPendingDecode
    ) -> MossPendingDecode:
        return self.decode_pending(
            previous.req_ids,
            previous.moss_rows[:, None, :],
            previous.caches,
            previous.moss_step + 1,
        )

    def decode_batch_finalize(self, pending: MossPendingDecode) -> list[int]:
        tokens = super().decode_batch_finalize(pending)
        rows = pending.moss_rows.tolist()
        for rid, row_array, row in zip(pending.req_ids, pending.moss_rows, rows):
            self.request_rows[rid] = row_array[None, None, :]
            self.request_steps[rid] = pending.moss_step + 1
            self.completed_rows.setdefault(rid, []).append(
                [int(value) for value in row]
            )
        return tokens

    def pop_completed_rows(self, req_id: str) -> list[list[int]]:
        return self.completed_rows.pop(req_id, [])

    def remove_request(self, req_id: str) -> None:
        super().remove_request(req_id)
        self.request_rows.pop(req_id, None)
        self.request_steps.pop(req_id, None)
        self.request_params.pop(req_id, None)
        self.completed_rows.pop(req_id, None)

    def reset_request(self, req_id: str) -> None:
        self.remove_request(req_id)


def make_moss_tts_local_mlx_runner_class() -> type[MossTTSLocalMlxModelRunner]:
    return MossTTSLocalMlxModelRunner
