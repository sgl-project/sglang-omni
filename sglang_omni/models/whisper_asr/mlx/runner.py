# SPDX-License-Identifier: Apache-2.0
"""SGLang MLX runner extension for Whisper.

Whisper is the first encoder-decoder model on this path, so it does not fit
SGLang's MLX cache discovery: that discovery classifies a layer by looking for
an attention module exposing ``("q_proj", "k_proj", "v_proj", "o_proj",
"rope")``, and Whisper has no RoPE — it uses learned absolute positions — and
names its output projection ``out_proj``. Discovery therefore finds no layers
and ``MlxModelRunner.__init__`` rejects the model. Declaring the layout up
front (see ``_declared_cache_layout``) skips the classifier entirely, so
nothing here has to pretend Whisper has a rotary embedding.
"""

from __future__ import annotations

import contextlib
import logging
import time
from typing import Any
from unittest import mock

import mlx.core as mx

from sglang_omni.model_runner.audio_mlx import AudioMlxModelRunner

logger = logging.getLogger(__name__)

_RUNNER_MODULE = "sglang.srt.hardware_backend.mlx.model_runner"


def _whisper_attention_layout(model: Any) -> tuple[list[Any], list[str]]:
    """Report the decoder stack as one attention layer per decoder block.

    Each block also owns an ``encoder_attn``, but its keys and values live in
    the per-layer cross-attention cache rather than in SGLang's KV pool, so the
    pool only needs to size the self-attention half.
    """
    layers = list(model.model.decoder.layers)
    return layers, ["self_attn"] * len(layers)


@contextlib.contextmanager
def _declared_cache_layout():
    """Replace attention discovery for the duration of base ``__init__``.

    ``patch_model_attention`` is also disabled: it wraps each discovered
    attention in ``MLXAttentionWrapper`` for batched decode, which assumes the
    rotary, single-attention-per-layer shape Whisper does not have.
    """
    with (
        mock.patch(
            f"{_RUNNER_MODULE}.find_attention_layers",
            side_effect=_whisper_attention_layout,
        ),
        mock.patch(f"{_RUNNER_MODULE}.patch_model_attention", return_value=0),
    ):
        yield


class WhisperMlxModelRunner(AudioMlxModelRunner):
    """Whisper support layered on SGLang's native MLX model runner.

    The SGLang base runner keeps ownership of pool sizing and request
    bookkeeping. ``AudioMlxModelRunner`` supplies the audio-item lookup, the
    tensor conversion and the chained decode step.

    Its prefill does not carry over. That path expects audio to arrive as
    contiguous placeholder tokens spliced into ``inputs_embeds`` through
    ``_build_inputs_embeds``, which is the decoder-only shape. Whisper is
    encoder-decoder: the decoder stream holds no placeholders, and the encoder
    projection lives in the per-layer cross-attention cache instead.

    ``decode_batch_start`` is also overridden rather than inherited. The shared
    version falls back to the batched path for anything it cannot handle, and
    that path reads ``caches[i][layer].offset`` straight off the layer entry —
    here a ``CacheList`` pair, whose shared KV pool has no room for the
    cross-attention half. Failing loudly beats failing inside the base.
    """

    model_name = "Whisper"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        with _declared_cache_layout():
            super().__init__(*args, **kwargs)

    def _load_model(self) -> None:
        from mlx_lm.utils import load_model
        from sglang.srt.hardware_backend.mlx.remote_code_gate import (
            ensure_remote_code_allowed,
            resolve_model_directory,
        )

        from .config import ModelConfig
        from .model import WhisperMlxModel

        model_path = resolve_model_directory(self.model_path, revision=self.revision)
        ensure_remote_code_allowed(model_path, self.trust_remote_code)
        logger.info("Loading native MLX Whisper model: %s", model_path)
        started = time.perf_counter()
        self.model, _config = load_model(
            model_path,
            get_model_classes=lambda config: (WhisperMlxModel, ModelConfig),
        )
        logger.info(
            "Loaded native MLX Whisper model in %.2fs", time.perf_counter() - started
        )

    def _new_native_cache(self) -> list[Any]:
        """One self-attention cache plus one cross-attention cache per layer.

        The base implementation installs SGLang's pooled attention caches, which
        hold a single growing KV stream per layer and cannot represent the
        cross-attention half.
        """
        return self.model.make_cache()

    def _first_attention_cache(self, cache: list[Any]) -> Any:
        """Point offset bookkeeping at the self-attention half.

        The base runner reads ``.offset`` off the layer's cache entry to learn
        how many tokens are committed. Here that entry is the ``CacheList``
        pair, and only its self-attention slot advances per token; the
        cross-attention slot is written once and stays put.
        """
        return cache[self._cache_layout.first_attention_layer_index][0]

    def _release_cache(self, cache: list[Any]) -> None:
        """Drop the cache instead of pooling it.

        Pool reuse calls ``reset()`` on every entry, which a ``CacheList`` pair
        does not implement. Rebuilding is cheap next to a 30 s encode, and the
        Apple path runs one request at a time.
        """
        del cache

    def _encoder_token_count(self, req: Any) -> int:
        item = self._audio_item(req)
        extra = getattr(item, "model_specific_data", None) or {}
        count = extra.get("num_audio_tokens")
        if count is None:
            count = getattr(req.multimodal_inputs, "num_image_tokens", None)
        if count is None:
            raise ValueError("Whisper MLX prefill needs the encoder token count")
        return int(count)

    def _decoder_prompt_ids(self, req: Any, token_ids: list[int]) -> list[int]:
        """Return the decoder prompt, with any encoder placeholders removed.

        The shared request builder emits ``[pad] * encoder_token_count`` ahead
        of the prompt so the scheduler reserves KV slots for the CUDA path's
        cross-attention entries. Whether the runner still sees that prefix
        depends on how the batch was assembled, so accept both shapes: this
        path holds the encoder projection in its own cross-attention cache, and
        decoding the placeholders would emit tokens from meaningless positions.
        """
        count = self._encoder_token_count(req)
        pad_token_id = getattr(self.model.config, "pad_token_id", None)
        if (
            pad_token_id is not None
            and len(token_ids) > count
            and all(token == pad_token_id for token in token_ids[:count])
        ):
            token_ids = token_ids[count:]
        if not token_ids:
            raise ValueError("Whisper MLX prefill got an empty decoder prompt")
        return list(token_ids)

    def _audio_prefill_inputs(self, req: Any, token_ids: list[int]):
        """Reject the inherited decoder-only prefill inputs.

        ``AudioMlxModelRunner`` builds them by splicing encoder output into
        ``inputs_embeds`` at the audio placeholder positions. Whisper has no
        placeholders to splice at and no ``_build_inputs_embeds``; its encoder
        output goes to the cross-attention cache in ``prefill_start`` instead.
        Nothing should reach this, so say why rather than fail on a missing
        model attribute.
        """
        raise NotImplementedError(
            "Whisper builds its prefill from the encoder-decoder path in "
            "prefill_start, not from spliced audio placeholders"
        )

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
        from sglang.srt.hardware_backend.mlx.model_runner import MlxPendingPrefill

        if req is None:
            raise ValueError("Whisper MLX prefill requires its scheduler request")
        if prefix_slot_ids:
            raise NotImplementedError(
                "Whisper MLX prefill does not support a radix prefix yet"
            )
        if not self.disable_radix_cache:
            raise RuntimeError("Whisper MLX requires disable_radix_cache=True")
        if logit_edit_row is not None or logprob_spec is not None:
            raise NotImplementedError(
                "Whisper MLX prefill supports greedy decoding only"
            )
        del new_slot_ids, needs_logits

        item = self._audio_item(req)
        if item.feature is None:
            raise ValueError("Whisper MLX prefill requires audio features")

        prompt_ids = self._decoder_prompt_ids(req, new_token_ids)
        encoder_hidden_states = self.model.encode(
            mx.array(self._to_numpy(item.feature))
        )

        cache = self._acquire_cache()
        # This call is what fills every layer's cross-attention cache; decode
        # steps after it reach cross-attention with tokens alone.
        logits = self.model.decode(
            mx.array([prompt_ids], dtype=mx.int32),
            encoder_hidden_states,
            cache=cache,
        )
        lazy_token = mx.argmax(logits[:, -1, :], axis=-1)
        return MlxPendingPrefill(
            lazy_token=lazy_token,
            cache=cache,
            req_id=req_id,
            full_token_ids=self._decoder_prompt_ids(req, full_token_ids),
            req_pool_idx=req_pool_idx,
            synced_offset=0,
            lazy_logprobs=None,
        )

    def decode_batch_start(
        self,
        req_ids: list[str],
        edit_rows: mx.array | None = None,
        logprob_spec: Any = None,
        logits_hook: Any = None,
    ):
        """Decode through the per-request cache rather than the batched path.

        Batched decode reads ``caches[i][layer].offset`` directly, but that
        entry is the ``CacheList`` pair here, and its shared KV pool has no
        room for cross-attention. The single-request path calls the model with
        its own cache, which is what this model is built for.
        """
        if (
            len(req_ids) != 1
            or edit_rows is not None
            or logprob_spec is not None
            or logits_hook is not None
        ):
            raise NotImplementedError(
                "Whisper MLX decode supports one greedy request at a time; got "
                f"{len(req_ids)} requests"
            )

        from sglang.srt.hardware_backend.mlx.model_runner import MlxPendingDecode

        req_id = req_ids[0]
        cache = self._req_caches[req_id]
        input_ids = mx.array([[self._req_token_ids[req_id][-1]]], dtype=mx.int32)
        lazy_logits = self._decode_with_native_cache([cache], [input_ids])
        return MlxPendingDecode(
            lazy_tokens=mx.argmax(lazy_logits, axis=-1),
            req_ids=[req_id],
            caches=[cache],
            lazy_logprobs=None,
            logprob_spec=None,
            edit_rows=None,
        )


def make_whisper_mlx_runner_class():
    """Build the extension class after the MLX backend has been selected."""
    from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner

    class WhisperMlxRunner(WhisperMlxModelRunner, MlxModelRunner):
        pass

    return WhisperMlxRunner


__all__ = ["WhisperMlxModelRunner", "make_whisper_mlx_runner_class"]
