# SPDX-License-Identifier: Apache-2.0
"""SGLang MLX runner extension for Whisper.

Whisper has no RoPE and names its output projection out_proj, so SGLang's MLX
attention discovery finds no layers; the cache layout is declared here instead.
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


def whisper_attention_layout(model: Any) -> tuple[list[Any], list[str]]:
    """Report one self-attention layer per block; cross-attention is not pooled."""
    layers = list(model.model.decoder.layers)
    return layers, ["self_attn"] * len(layers)


@contextlib.contextmanager
def declared_cache_layout():
    """Replace attention discovery and patching for the duration of base __init__."""
    with (
        mock.patch(
            f"{_RUNNER_MODULE}.find_attention_layers",
            side_effect=whisper_attention_layout,
        ),
        mock.patch(f"{_RUNNER_MODULE}.patch_model_attention", return_value=0),
    ):
        yield


class WhisperMlxModelRunner(AudioMlxModelRunner):
    """Whisper encoder-decoder support on SGLang's native MLX model runner."""

    model_name = "Whisper"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        with declared_cache_layout():
            super().__init__(*args, **kwargs)

    def _load_model(self) -> None:  # noqa: leading-underscore
        from mlx_lm.utils import load_model
        from sglang.srt.hardware_backend.mlx.remote_code_gate import (
            ensure_remote_code_allowed,
            resolve_model_directory,
        )

        from .config import ModelConfig
        from .model import WhisperMlxModel

        model_path = resolve_model_directory(self.model_path, revision=self.revision)
        ensure_remote_code_allowed(model_path, self.trust_remote_code)
        logger.info(f"Loading native MLX Whisper model: {model_path}")
        started = time.perf_counter()
        self.model, _config = load_model(
            model_path,
            get_model_classes=lambda config: (WhisperMlxModel, ModelConfig),
        )
        logger.info(
            f"Loaded native MLX Whisper model in {time.perf_counter() - started:.2f}s"
        )

    def _new_native_cache(self) -> list[Any]:  # noqa: leading-underscore
        """One self-attention cache plus one cross-attention cache per layer."""
        return self.model.make_cache()

    def _first_attention_cache(  # noqa: leading-underscore
        self, cache: list[Any]
    ) -> Any:
        """Point offset bookkeeping at the self-attention half."""
        return cache[self._cache_layout.first_attention_layer_index][0]

    def _release_cache(self, cache: list[Any]) -> None:  # noqa: leading-underscore
        """Drop the cache instead of pooling it; CacheList has no reset()."""
        del cache

    def encoder_token_count(self, req: Any) -> int:
        item = self.audio_item(req)
        extra = getattr(item, "model_specific_data", None) or {}
        count = extra.get("num_audio_tokens")
        if count is None:
            count = getattr(req.multimodal_inputs, "num_image_tokens", None)
        if count is None:
            raise ValueError("Whisper MLX prefill needs the encoder token count")
        return int(count)

    def decoder_prompt_ids(self, req: Any, token_ids: list[int]) -> list[int]:
        """Strip the encoder pad placeholders from the decoder prompt when present."""
        count = self.encoder_token_count(req)
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

    def audio_prefill_inputs(self, req: Any, token_ids: list[int]):
        """Whisper prefills through prefill_start, not spliced audio placeholders."""
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

        item = self.audio_item(req)
        if item.feature is None:
            raise ValueError("Whisper MLX prefill requires audio features")

        prompt_ids = self.decoder_prompt_ids(req, new_token_ids)
        encoder_hidden_states = self.model.encode(mx.array(self.to_numpy(item.feature)))

        cache = self._acquire_cache()
        # Fills every layer's cross-attention cache for the decode steps.
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
            full_token_ids=self.decoder_prompt_ids(req, full_token_ids),
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
        """Decode via the per-request cache; the batched pool lacks cross-attention."""
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
