# SPDX-License-Identifier: Apache-2.0
"""SGLang MLX runner for the Qwen3-TTS talker.

Qwen3-TTS does not decode like a text model, so three things differ from the
Qwen3-ASR MLX runner:

*Steps are frames, not tokens.*  Every step samples codec group 0 from the
talker and then expands it to a full ``num_code_groups`` frame through the code
predictor.  Group 0 is what SGLang tracks as the request's "next token" (the
CUDA path does the same), and the rest of the frame is handed to the vocoder.

*Input is an embedding, not a token id.*  The next step's input is the summed
codec embedding of the frame just produced plus the next trailing-text
embedding, so ``decode_batch_start`` cannot read ``_req_token_ids`` the way the
base class does.

*The frame stays lazy.*  A step builds talker -> sample -> predictor -> feedback
embedding as one graph and evaluates once, and a chained step is built on the
previous step's still-lazy feedback embedding, so consecutive frames run
back-to-back with no host round trip in between.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import numpy as np

from .sampling import SamplingParams, sample_codec_token, special_codec_token_ids

logger = logging.getLogger(__name__)


@dataclass
class Qwen3TTSRequestSpec:
    """Per-request generation state the scheduler hands to the runner.

    ``prompt_embeds`` is the assembled prefill sequence ``[1, T, hidden]`` and
    ``trailing_text_embeds`` is the text stream consumed one position per frame
    after it (``[1, N, hidden]``); once exhausted, ``pad_embed`` repeats.
    """

    prompt_embeds: mx.array
    trailing_text_embeds: mx.array
    pad_embed: mx.array
    semantic: SamplingParams = field(default_factory=SamplingParams)
    subtalker: SamplingParams = field(default_factory=SamplingParams)
    seed: int | None = None
    trailing_index: int = 0
    # The next step's input embedding, kept lazy between steps.
    pending_input: Any = None

    def next_text_embed(self) -> mx.array:
        """The text half of the next frame's input, advancing the stream."""
        trailing = self.trailing_text_embeds
        if trailing is not None and self.trailing_index < trailing.shape[1]:
            embed = trailing[:, self.trailing_index : self.trailing_index + 1, :]
            self.trailing_index += 1
            return embed
        return self.pad_embed


def _as_mlx(value: Any) -> mx.array:
    """Accept an MLX array, a NumPy array, or a Torch tensor."""
    if isinstance(value, mx.array):
        return value
    if isinstance(value, np.ndarray):
        return mx.array(value)
    detach = getattr(value, "detach", None)
    if detach is not None:  # torch.Tensor
        tensor = detach().cpu()
        if str(tensor.dtype) == "torch.bfloat16":
            return mx.array(tensor.float().numpy()).astype(mx.bfloat16)
        return mx.array(tensor.numpy())
    raise TypeError(f"Cannot convert {type(value).__name__} to an mx.array")


class Qwen3TTSMlxModelRunner:
    """Qwen3-TTS support layered on SGLang's native MLX model runner.

    The base runner keeps owning cache layout, pool sizing and batched decode;
    this mixin supplies the model class, the embedding-driven step, and the
    per-request codec frames the vocoder stage consumes.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._tts_specs: dict[str, Qwen3TTSRequestSpec] = {}
        self._tts_frames: dict[str, list[np.ndarray]] = {}
        self._tts_predictor_cache: list[Any] | None = None
        self._tts_staged_prefill: dict[str, tuple[mx.array, mx.array]] = {}
        self._tts_suppress: list[int] | None = None
        # Emitted group-0 codes per request. The base class's _req_token_ids
        # starts from the prompt's *text* ids, so it cannot feed a codec-domain
        # repetition penalty.
        self._tts_emitted: dict[str, list[int]] = {}

    # -- model ----------------------------------------------------------

    def _load_model(self) -> None:
        from mlx_lm.utils import load_model
        from sglang.srt.hardware_backend.mlx.remote_code_gate import (
            ensure_remote_code_allowed,
            resolve_model_directory,
        )

        from .config import ModelConfig
        from .model import Qwen3TTSTalkerModel

        model_path = resolve_model_directory(self.model_path, revision=self.revision)
        ensure_remote_code_allowed(model_path, self.trust_remote_code)
        logger.info("Loading native MLX Qwen3-TTS talker: %s", model_path)
        started = time.perf_counter()
        self.model, _config = load_model(
            model_path,
            get_model_classes=lambda config: (Qwen3TTSTalkerModel, ModelConfig),
        )
        logger.info(
            "Loaded native MLX Qwen3-TTS talker in %.2fs",
            time.perf_counter() - started,
        )

    @property
    def _talker_config(self):
        return self.model.config

    @property
    def _num_code_groups(self) -> int:
        return int(self._talker_config.code_predictor_config.num_code_groups)

    # -- request registration ------------------------------------------

    def register_request(self, req_id: str, spec: Qwen3TTSRequestSpec) -> None:
        """Attach generation state before the request's prefill is launched.

        SGLang's ``prefill_start`` only receives its own ``Req``, so Omni's
        scheduler-side runner registers the assembled prompt here first.
        """
        self._tts_specs[req_id] = spec
        self._tts_frames.setdefault(req_id, [])
        self._tts_emitted.setdefault(req_id, [])

    def drain_frames(self, req_id: str) -> list[np.ndarray]:
        """Take and clear the frames finalised for one request."""
        frames = self._tts_frames.get(req_id)
        if not frames:
            return []
        self._tts_frames[req_id] = []
        return frames

    def remove_request(self, req_id: str) -> None:
        self._tts_specs.pop(req_id, None)
        self._tts_frames.pop(req_id, None)
        self._tts_emitted.pop(req_id, None)
        self._tts_staged_prefill.pop(req_id, None)
        super().remove_request(req_id)

    def clear(self) -> None:
        self._tts_specs.clear()
        self._tts_frames.clear()
        self._tts_emitted.clear()
        self._tts_staged_prefill.clear()
        self._tts_predictor_cache = None
        super().clear()

    # -- one autoregressive frame ---------------------------------------

    def _predictor_cache(self) -> list[Any]:
        """A reusable code-predictor cache.

        The predictor restarts at position 0 for every frame, so one cache is
        reset and reused rather than reallocated per frame; ``predict_codes``
        does the reset. Batch size changes are safe because a reset cache
        re-allocates from the first write.
        """
        if self._tts_predictor_cache is None:
            self._tts_predictor_cache = self.model.code_predictor.make_cache()
        return self._tts_predictor_cache

    def _suppress_tokens(self) -> list[int]:
        if self._tts_suppress is None:
            config = self._talker_config
            self._tts_suppress = special_codec_token_ids(
                int(config.vocab_size), keep=int(config.codec_eos_token_id)
            )
        return self._tts_suppress

    def _frame_from_hidden(
        self,
        logits: mx.array,
        hidden: mx.array,
        specs: list[Qwen3TTSRequestSpec],
        recent_tokens: list[list[int]] | None = None,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Sample one frame per row and build the next input embedding.

        Returns ``(group-0 tokens [B, 1], frame [B, groups], next input
        [B, 1, hidden])``, all still lazy.
        """
        # Sampling parameters are per request, so rows are sampled separately
        # and restacked; a homogeneous batch collapses to a single call.
        semantic_rows = []
        for index, spec in enumerate(specs):
            history = recent_tokens[index] if recent_tokens else None
            semantic_rows.append(
                sample_codec_token(
                    logits[index : index + 1],
                    spec.semantic,
                    recent_tokens=history,
                    suppress_tokens=self._suppress_tokens(),
                )
            )
        semantic = (
            semantic_rows[0]
            if len(semantic_rows) == 1
            else mx.concatenate(semantic_rows, axis=0)
        )

        def sample_group(group_logits: mx.array, _group_index: int) -> mx.array:
            # The predictor runs batched, so honour each row's own subtalker
            # settings rather than letting row 0 speak for the batch.
            if len(specs) == 1:
                return sample_codec_token(group_logits, specs[0].subtalker)
            return mx.concatenate(
                [
                    sample_codec_token(group_logits[index : index + 1], spec.subtalker)
                    for index, spec in enumerate(specs)
                ],
                axis=0,
            )

        frame, codec_embed = self.model.predict_codes(
            semantic,
            hidden,
            cache=self._predictor_cache(),
            sampler=sample_group,
        )

        text_embed = mx.concatenate([spec.next_text_embed() for spec in specs], axis=0)
        return semantic, frame, text_embed + codec_embed

    # -- prefill --------------------------------------------------------

    def prefill_start(
        self,
        req_id: str,
        new_token_ids: list[int],
        full_token_ids: list[int],
        prefix_slot_ids: list[int],
        new_slot_ids: list[int],
        req_pool_idx: int,
        req: Any | None = None,
        needs_logits: bool = True,
        logit_edit_row: mx.array | None = None,
        logprob_spec: Any = None,
    ):
        """Prefill the assembled prompt and produce the request's first frame."""
        from sglang.srt.hardware_backend.mlx.model_runner import MlxPendingPrefill

        del new_token_ids, new_slot_ids, req, needs_logits
        spec = self._tts_specs.get(req_id)
        if spec is None:
            raise RuntimeError(
                f"Qwen3-TTS MLX prefill has no registered prompt for {req_id!r}; "
                "the scheduler runner must call register_request first"
            )
        if prefix_slot_ids:
            raise NotImplementedError(
                "Qwen3-TTS MLX prefill does not support a radix prefix; the "
                "prompt is an embedding sequence, not a token prefix"
            )
        if not self.disable_radix_cache:
            raise RuntimeError("Qwen3-TTS MLX requires disable_radix_cache=True")
        if logit_edit_row is not None or logprob_spec is not None:
            raise NotImplementedError(
                "Qwen3-TTS MLX prefill supports neither logit edits nor logprobs"
            )
        if spec.seed is not None:
            mx.random.seed(spec.seed)

        cache = self._acquire_cache()
        logits, hidden = self.model(spec.prompt_embeds, cache=cache)
        semantic, frame, next_embeds = self._frame_from_hidden(
            logits[:, -1, :], hidden, [spec]
        )

        pending = MlxPendingPrefill(
            lazy_token=semantic.reshape(-1),
            cache=cache,
            req_id=req_id,
            full_token_ids=list(full_token_ids),
            req_pool_idx=req_pool_idx,
            synced_offset=0,
            lazy_logprobs=None,
        )
        # The frame and the next input ride alongside the pending; the base
        # class never looks at them, and prefill_finalize collects them.
        self._tts_staged_prefill[req_id] = (frame, next_embeds)
        return pending

    def prefill_finalize(self, pending: Any) -> int:
        token = super().prefill_finalize(pending)
        staged = self._tts_staged_prefill.pop(pending.req_id, None)
        if staged is not None:
            frame, next_embeds = staged
            self._commit_frames([pending.req_id], frame)
            self._tts_specs[pending.req_id].pending_input = next_embeds
            self._tts_emitted[pending.req_id].append(token)
        return token

    # -- decode ---------------------------------------------------------

    def _commit_frames(self, req_ids: list[str], frame: mx.array) -> None:
        """Materialise a lazy frame and stash one row per request."""
        codes = np.array(frame, copy=True)
        if codes.ndim == 1:
            codes = codes[None, :]
        for index, req_id in enumerate(req_ids):
            self._tts_frames.setdefault(req_id, []).append(codes[index])

    def decode_batch_start(
        self,
        req_ids: list[str],
        edit_rows: mx.array | None = None,
        logprob_spec: Any = None,
        logits_hook: Any = None,
    ):
        """Run one frame for every request from its stored feedback embedding."""
        if edit_rows is not None or logprob_spec is not None or logits_hook is not None:
            raise NotImplementedError(
                "Qwen3-TTS MLX decode supports neither logit edits, logprobs, "
                "nor logits hooks"
            )
        specs = [self._tts_specs[rid] for rid in req_ids]
        caches = [self._req_caches[rid] for rid in req_ids]
        batched_input = mx.concatenate([spec.pending_input for spec in specs], axis=0)
        return self._decode_step(req_ids, specs, caches, batched_input)

    def decode_batch_start_chained(self, prev: Any):
        """Continue from the previous step's still-lazy feedback embedding."""
        specs = [self._tts_specs[rid] for rid in prev.req_ids]
        return self._decode_step(prev.req_ids, specs, prev.caches, prev.lazy_feedback)

    def _decode_step(
        self,
        req_ids: list[str],
        specs: list[Qwen3TTSRequestSpec],
        caches: list[list[Any]],
        batched_input: mx.array,
    ):
        from .pending import Qwen3TTSMlxPendingDecode

        logits, hidden = self._talker_forward(caches, req_ids, batched_input)
        recent = [self._tts_emitted[rid] for rid in req_ids]
        semantic, frame, next_embeds = self._frame_from_hidden(
            logits, hidden, specs, recent_tokens=recent
        )
        return Qwen3TTSMlxPendingDecode(
            lazy_tokens=semantic.reshape(-1),
            req_ids=list(req_ids),
            caches=caches,
            lazy_logprobs=None,
            logprob_spec=None,
            edit_rows=None,
            lazy_codes=frame,
            lazy_feedback=next_embeds,
        )

    def _talker_forward(
        self,
        caches: list[list[Any]],
        req_ids: list[str],
        batched_input: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """One batched talker step over embeddings, via SGLang's KV plumbing."""
        from sglang.srt.hardware_backend.mlx.kv_cache import (
            AttentionOffsetCache,
            clear_context,
            set_context,
        )

        if len(req_ids) == 1:
            logits, hidden = self.model(batched_input, cache=caches[0])
            return logits[:, -1, :], hidden

        context = self._build_batched_decode_context(caches, list(req_ids))
        set_context(context)
        try:
            shim = [
                AttentionOffsetCache(offset=max(context.seq_lens))
                for _ in range(self._cache_layout.num_layers)
            ]
            logits, hidden = self.model(batched_input, cache=shim)
        finally:
            clear_context()
        return logits[:, -1, :], hidden

    def decode_batch_finalize(self, pending: Any) -> list[int]:
        """Materialise the step, stash its frames, carry the feedback forward."""
        next_tokens = super().decode_batch_finalize(pending)
        self._commit_frames(pending.req_ids, pending.lazy_codes)
        for index, req_id in enumerate(pending.req_ids):
            self._tts_specs[req_id].pending_input = pending.feedback_row(index)
            self._tts_emitted[req_id].append(next_tokens[index])
        return next_tokens


def make_qwen3_tts_mlx_runner_class():
    """Build the extension class after the MLX backend has been selected."""
    from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner

    class Qwen3TTSMlxRunner(Qwen3TTSMlxModelRunner, MlxModelRunner):
        pass

    return Qwen3TTSMlxRunner
