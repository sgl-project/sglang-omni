# SPDX-License-Identifier: Apache-2.0
"""Native MLX Qwen3-Omni thinker runner and its SGLang scheduler wiring.

This module is the bridge between the reviewed native model
(:class:`sglang_omni.models.qwen3_omni.mlx.thinker.Qwen3OmniMlxThinker`) and
SGLang's MLX worker/scheduler surface.

Design notes
------------

* **Standalone runner, not an ``MlxModelRunner`` subclass.** SGLang's generic
  ``MlxModelRunner`` discovers attention layers by duck typing (``q_proj``,
  ``k_proj``, ``v_proj``, ``o_proj`` *and* ``rope``) and then replaces every
  attention module with its own batched-decode wrapper and pool-backed KV
  caches. The Qwen3-Omni thinker has interleaved three-axis M-RoPE applied from
  externally supplied ``[3, sequence]`` position rows, so it exposes no ``rope``
  module and its attention cannot be driven by that wrapper. Rather than
  half-disable the generic machinery, this runner implements exactly the
  surface ``MlxTpModelWorker`` calls, and owns one ``mlx-lm`` KV cache list per
  request.
* **Apple policy.** ``apply_qwen3_omni_apple_profile`` pins
  ``max_running_requests=1``, ``disable_radix_cache=True``,
  ``chunked_prefill_size=-1`` and greedy sampling, so this runner requires one
  request, refuses a radix prefix, refuses chunked continuations, and refuses
  logit edits / logprobs rather than silently degrading.
* **Hidden capture.** The scheduler-selected capture list arrives through
  ``ModelWorkerConfig.capture_hidden_layers`` (``[0, 24]`` for the production
  speech path); ``0`` maps to the model's ``"embed"`` capture and the single
  nonzero entry is cross-checked against ``talker_config.accept_hidden_layer``.
  Captures are carried on the pending step (never keyed by request id) so a
  chained decode cannot overwrite its predecessor's rows, and are converted to
  CPU Torch tensors only when the worker pops them.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import torch

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.qwen3_omni.components.apple_adapter import (
    emit_talker_step,
    projected_prefill_rows,
    release_talker_host_queues,
    require_single_request,
    validate_capture_layers,
)
from sglang_omni.models.qwen3_omni.mlx.common import load_qwen3_omni_mlx_component
from sglang_omni.models.qwen3_omni.mlx.config import Qwen3OmniMlxConfig
from sglang_omni.models.qwen3_omni.mlx.talker import (
    Qwen3OmniMlxTalker,
    build_suppress_mask,
)
from sglang_omni.models.qwen3_omni.mlx.talker_prefill import (
    Qwen3OmniMlxTalkerPrefillBuilder,
)
from sglang_omni.models.qwen3_omni.mlx.tensor_utils import (
    mlx_to_torch as _mlx_to_torch,
)
from sglang_omni.models.qwen3_omni.mlx.tensor_utils import (
    torch_to_mlx as _torch_to_mlx,
)
from sglang_omni.models.qwen3_omni.mlx.thinker import (
    Qwen3OmniMlxThinker,
    merge_thinker_input_embeddings,
    visual_placeholder_mask,
)
from sglang_omni.models.qwen3_omni.mrope_positions import linear_mrope_positions
from sglang_omni.models.qwen3_omni.talker_model_runner import QwenTalkerModelRunner

logger = logging.getLogger(__name__)

__all__ = [
    "Qwen3OmniMlxSchedulerModelRunner",
    "Qwen3OmniMlxTalkerModelRunner",
    "Qwen3OmniThinkerMlxRunner",
    "build_qwen3_omni_thinker_mlx_runner",
    "create_qwen3_omni_mlx_worker",
    "load_qwen3_omni_mlx_talker",
    "make_qwen3_omni_talker_mlx_runner_class",
    "make_qwen3_omni_thinker_mlx_runner_class",
    "read_qwen3_omni_component_weights",
]

_MODALITIES = ("image", "video", "audio")
# Only the thinker text stack belongs to this runner; the vision/audio towers
# run in their own stages and the talker/code2wav weights belong elsewhere.
_THINKER_WEIGHT_PREFIXES = (
    "thinker.model.",
    "thinker.lm_head.",
    "thinker.language_model.model.",
    "thinker.language_model.lm_head.",
)
# Component-local key prefixes of an already-converted (prefix-stripped) MLX
# checkpoint. They are deliberately ambiguous across components -- both the
# thinker and the talker own a ``model.`` namespace -- which is exactly why a
# converted export has to be disambiguated by *shard ownership* (see
# :func:`read_qwen3_omni_component_weights`) and never by key text alone.
_THINKER_LOCAL_PREFIXES = ("model.", "lm_head.")
_TALKER_LOCAL_PREFIXES = (
    "model.",
    "codec_head.",
    "text_projection.",
    "hidden_projection.",
    "code_predictor.",
)
_TALKER_WEIGHT_PREFIXES = ("talker.",)
# Directory names a converted multi-component MLX export uses, matching
# ``apple_runtime._components_for_key``.
_COMPONENT_DIRECTORIES = ("thinker", "talker", "vision", "audio", "code2wav")
# Official (HF) component namespaces. A key carrying one of these is owned by
# that component and by no other, wherever the shard happens to live.
_OFFICIAL_COMPONENT_PREFIXES = ("thinker.", "talker.", "code2wav.")

_CPU = torch.device("cpu")


def _shard_component_directory(shard: Path, root: Path) -> str | None:
    """The component subdirectory a shard lives in, or ``None`` at the root."""

    try:
        parts = shard.relative_to(root).parts
    except ValueError:  # pragma: no cover - rglob results are always under root
        return None
    if len(parts) > 1 and parts[0] in _COMPONENT_DIRECTORIES:
        return parts[0]
    return None


def read_qwen3_omni_component_weights(
    directory: Path | str,
    *,
    component: str,
    official_prefixes: Sequence[str],
    local_prefixes: Sequence[str],
) -> dict[str, mx.array]:
    """Collect one component's tensors, one shard at a time, by *ownership*.

    Two checkpoint layouts have to be told apart, and the second is why key
    text alone is not a safe filter:

    * **Official HF layout.** Every key carries its component namespace
      (``thinker.``/``talker.``/``code2wav.``), so ownership is unambiguous and
      the sibling components' tensors are dropped.
    * **Converted MLX layout.** The converter strips that namespace and puts
      each component in its own subdirectory (``thinker/``, ``talker/``,
      ``code2wav/``). The stripped keys collide -- the thinker and the talker
      both own ``model.layers.*`` -- so a recursive scan that accepted
      ``model.`` from any shard would load the *thinker's* decoder into the
      talker (or vice versa). Ownership is therefore taken from the shard's
      own component directory, and an unprefixed key from a foreign component
      directory is never claimed.

    A root-level unprefixed key is claimed only when the checkpoint has no
    component directories at all (a genuinely single-component export);
    otherwise it is ambiguous and is left alone.

    Returns the raw (still component-prefixed where the source had a prefix)
    key/value map; the model's own ``sanitize`` performs the prefix strip.
    """

    root = Path(directory)
    shards = sorted(path for path in root.rglob("*.safetensors") if path.is_file())
    if not shards:
        raise FileNotFoundError(
            f"no safetensors shard found under {root} for the MLX Qwen3-Omni "
            f"{component}"
        )
    component_dirs = {
        found
        for found in (_shard_component_directory(shard, root) for shard in shards)
        if found is not None
    }
    official_prefixes = tuple(official_prefixes)
    local_prefixes = tuple(local_prefixes)

    weights: dict[str, mx.array] = {}
    for shard in shards:
        shard_component = _shard_component_directory(shard, root)
        if shard_component is not None and shard_component != component:
            # A sibling component's converted shard: it can only hold that
            # component's (unprefixed) tensors, so do not even open it.
            continue
        loaded = mx.load(str(shard))
        for key, value in loaded.items():
            if key.startswith(_OFFICIAL_COMPONENT_PREFIXES):
                if key.startswith(official_prefixes):
                    weights[key] = value
                continue
            if not key.startswith(local_prefixes):
                continue
            if shard_component == component:
                weights[key] = value
            elif shard_component is None and not component_dirs:
                weights[key] = value
        del loaded

    if not weights:
        raise ValueError(f"checkpoint {root} carries no Qwen3-Omni {component} weights")
    return weights


class Qwen3OmniThinkerMlxRunner:
    """Per-request native MLX execution for the Qwen3-Omni thinker stage."""

    def __init__(
        self,
        model_path: str,
        trust_remote_code: bool = False,
        disable_radix_cache: bool = False,
        pool_size: int | None = None,
        mem_fraction_static: float = 0.8,
        quantization: str | None = None,
        revision: str | None = None,
        enable_sampling: bool = False,
        sampling_rng_seed: int = 0,
        deterministic_seeding: bool = False,
        *,
        capture_hidden_layers: tuple[int, ...] | list[int] | None = None,
    ):
        if enable_sampling:
            raise NotImplementedError(
                "Apple Qwen3-Omni thinker supports greedy generation only; "
                "MLX sampling is not enabled for this stage"
            )
        del sampling_rng_seed, deterministic_seeding
        if quantization is not None:
            raise NotImplementedError(
                "Apple Qwen3-Omni thinker reads quantization from the "
                "checkpoint's own metadata; on-the-fly preset "
                f"{quantization!r} is not supported"
            )
        self.model_path = model_path
        self.trust_remote_code = trust_remote_code
        self.revision = revision
        self.disable_radix_cache = disable_radix_cache
        self._mem_fraction_static = mem_fraction_static

        loaded = self._load_model()
        self.model = loaded["model"]
        self.placeholder_token_ids = loaded["placeholder_token_ids"]
        self.quantization = loaded["quantization"]
        self._init_capture(
            capture_hidden_layers,
            accept_hidden_layer=loaded["accept_hidden_layer"],
        )
        self._pool_size = int(
            pool_size
            if pool_size is not None
            else self.model.config.max_position_embeddings
        )
        self._init_request_state()

    # -- construction ------------------------------------------------------

    @classmethod
    def from_model(
        cls,
        *,
        model: Qwen3OmniMlxThinker,
        placeholder_token_ids: dict[str, int],
        capture_hidden_layers: tuple[int, ...] | list[int] | None,
        accept_hidden_layer: int | None,
        disable_radix_cache: bool = True,
        pool_size: int | None = None,
        quantization: Any = None,
    ) -> "Qwen3OmniThinkerMlxRunner":
        """Build a runner around an already-constructed model.

        Used by the worker factory's tests and by any caller that owns model
        construction; ``__init__`` remains the production path that loads a
        checkpoint from disk.
        """

        runner = cls.__new__(cls)
        runner.model_path = None
        runner.trust_remote_code = False
        runner.revision = None
        runner.disable_radix_cache = disable_radix_cache
        runner._mem_fraction_static = 0.8
        runner.model = model
        runner.placeholder_token_ids = dict(placeholder_token_ids)
        runner.quantization = quantization
        runner._init_capture(
            capture_hidden_layers, accept_hidden_layer=accept_hidden_layer
        )
        runner._pool_size = int(
            pool_size if pool_size is not None else model.config.max_position_embeddings
        )
        runner._init_request_state()
        return runner

    def _init_capture(
        self,
        capture_hidden_layers: tuple[int, ...] | list[int] | None,
        *,
        accept_hidden_layer: int | None,
    ) -> None:
        self.capture_layers = validate_capture_layers(
            capture_hidden_layers,
            accept_hidden_layer=accept_hidden_layer,
        )
        self.accept_hidden_layer = (
            None if accept_hidden_layer is None else int(accept_hidden_layer)
        )

    def _init_request_state(self) -> None:
        self._req_caches: dict[str, list[Any]] = {}
        self._req_token_ids: dict[str, list[int]] = {}
        self._req_mrope_delta: dict[str, int] = {}
        # Prefills that have been launched but not yet committed to
        # ``_req_caches``; counted for the single-request admission check so a
        # second request in the same extend batch is rejected at launch time.
        self._inflight_prefills: set[str] = set()
        self._req_to_token_pool: Any = None
        # Diagnostics for the most recent step; also what the unit tests read to
        # pin placeholder restoration, dtype conversion, and M-RoPE forwarding.
        self.last_prefill_input_ids: list[int] | None = None
        self.last_prefill_embeddings: mx.array | None = None
        self.last_prefill_positions: mx.array | None = None
        self.last_decode_positions: mx.array | None = None

    # -- weight loading ----------------------------------------------------

    def _load_model(self) -> dict[str, Any]:
        from sglang.srt.hardware_backend.mlx.remote_code_gate import (
            ensure_remote_code_allowed,
            resolve_model_directory,
        )

        directory = resolve_model_directory(self.model_path, revision=self.revision)
        ensure_remote_code_allowed(directory, self.trust_remote_code)
        directory = Path(directory)
        raw = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        config = Qwen3OmniMlxConfig.from_dict(raw)
        thinker_raw = raw["thinker_config"]
        text_config = config.thinker.text_config

        logger.info(
            "Loading native MLX Qwen3-Omni thinker: %s (attention_bias=%s, "
            "quantization=%s)",
            directory,
            text_config.attention_bias,
            config.quantization,
        )
        started = time.perf_counter()

        model = Qwen3OmniMlxThinker(text_config)
        weights = self._read_thinker_weights(directory)
        load_qwen3_omni_mlx_component(
            model,
            weights,
            sanitizer=model.sanitize,
            quantization=config.quantization,
        )
        model.eval()

        logger.info(
            "Loaded native MLX Qwen3-Omni thinker in %.2fs (%d layers)",
            time.perf_counter() - started,
            model.num_layers,
        )
        return {
            "model": model,
            "placeholder_token_ids": {
                modality: int(thinker_raw[f"{modality}_token_id"])
                for modality in _MODALITIES
            },
            "accept_hidden_layer": (raw.get("talker_config") or {}).get(
                "accept_hidden_layer"
            ),
            "quantization": config.quantization,
        }

    @staticmethod
    def _read_thinker_weights(directory: Path) -> dict[str, mx.array]:
        """Collect the thinker text tensors, one shard at a time.

        Ownership is decided by :func:`read_qwen3_omni_component_weights`, so a
        converted export that puts each component in its own subdirectory
        cannot leak the talker's identically named ``model.*`` keys into the
        thinker.
        """

        return read_qwen3_omni_component_weights(
            directory,
            component="thinker",
            official_prefixes=_THINKER_WEIGHT_PREFIXES,
            local_prefixes=_THINKER_LOCAL_PREFIXES,
        )

    # -- scheduler bookkeeping surface -------------------------------------

    @property
    def pool_size(self) -> int:
        return self._pool_size

    def init_cache_pools(self, req_to_token_pool: Any) -> None:
        """This runner owns per-request caches; there is no shared KV pool."""

        if not self.disable_radix_cache:
            raise RuntimeError(
                "Apple Qwen3-Omni thinker requires disable_radix_cache=True"
            )
        self._req_to_token_pool = req_to_token_pool

    def flush_all_decode_kv(self) -> None:
        """No pool-backed KV mirror exists, so there is nothing to flush."""

    def store_auxiliary_state_for_request(self, req_id: str) -> None:
        """No auxiliary (non-attention) cache state exists for this model."""

    @staticmethod
    def cache_state_arrays(caches: list[list[Any]]) -> list[mx.array]:
        """Every live KV array in ``caches``, for ``mx.async_eval``."""

        arrays: list[mx.array] = []
        for cache in caches:
            for layer_cache in cache:
                state = getattr(layer_cache, "state", None)
                if state is None:
                    continue
                for value in state if isinstance(state, (tuple, list)) else (state,):
                    if isinstance(value, mx.array):
                        arrays.append(value)
        return arrays

    def collect_logprobs(self, lazy_logprobs: Any) -> None:
        """Greedy-only stage: no logprobs are ever produced."""

        if lazy_logprobs is not None:
            raise NotImplementedError(
                "Apple Qwen3-Omni thinker supports greedy generation only"
            )
        return None

    def has_request(self, req_id: str) -> bool:
        return req_id in self._req_caches

    def remove_request(self, req_id: str) -> None:
        """Drop every per-request MLX resource (abort or completion)."""

        self._req_caches.pop(req_id, None)
        self._req_token_ids.pop(req_id, None)
        self._req_mrope_delta.pop(req_id, None)
        self._inflight_prefills.discard(req_id)

    def clear(self) -> None:
        self._req_caches.clear()
        self._req_token_ids.clear()
        self._req_mrope_delta.clear()
        self._inflight_prefills.clear()

    def extend_start(self, *args: Any, **kwargs: Any):
        raise NotImplementedError(
            "Apple Qwen3-Omni thinker runs with chunked prefill disabled; "
            "a chunked-prefill continuation cannot be served"
        )

    def extend_finalize(self, *args: Any, **kwargs: Any):
        raise NotImplementedError(
            "Apple Qwen3-Omni thinker runs with chunked prefill disabled"
        )

    # -- hidden-state transport --------------------------------------------

    def pop_hidden_states(self, pending: Any) -> dict[Any, torch.Tensor] | None:
        """Take this step's captures as CPU Torch tensors, once.

        Captures live on the pending step, so a chained decode launched before
        its predecessor was finalised cannot overwrite the earlier rows.
        """

        captured = getattr(pending, "_omni_hidden_states", None)
        if captured is None:
            return None
        pending._omni_hidden_states = None
        return {key: _mlx_to_torch(value) for key, value in captured.items()}

    def _attach_hidden_states(self, pending: Any, step: Any) -> None:
        if not self.capture_layers:
            pending._omni_hidden_states = None
            return
        pending._omni_hidden_states = dict(step.hidden_states)

    # -- request preparation -----------------------------------------------

    def _restore_placeholder_token_ids(
        self, req: Any, token_ids: list[int]
    ) -> list[int]:
        """Map cache-key placeholder ids back to configured modality ids.

        ``build_sglang_thinker_request`` substitutes a per-media hash above the
        text vocabulary so SGLang's radix cache keys stay media-specific. Those
        ids are not embeddable, so they are restored here using the absolute
        positions the request builder already recorded (``_omni_mm_positions``),
        falling back to the request's ``pad_values`` map.
        """

        restored = [int(token_id) for token_id in token_ids]
        positions = getattr(req, "_omni_mm_positions", None)
        if positions:
            for modality, modality_positions in positions.items():
                token_id = self.placeholder_token_ids.get(modality)
                if token_id is None or modality_positions is None:
                    continue
                for position in np.asarray(modality_positions).reshape(-1).tolist():
                    index = int(position)
                    if 0 <= index < len(restored):
                        restored[index] = int(token_id)
        else:
            model_inputs = getattr(req, "omni_model_inputs", None) or {}
            pad_values = model_inputs.get("pad_values") or {}
            inverse = {
                int(pad): int(self.placeholder_token_ids[modality])
                for modality, pad in pad_values.items()
                if modality in self.placeholder_token_ids
            }
            if inverse:
                restored = [inverse.get(token, token) for token in restored]

        vocab_size = self.model.config.vocab_size
        out_of_range = [token for token in restored if not 0 <= token < vocab_size]
        if out_of_range:
            raise ValueError(
                "Qwen3-Omni MLX prefill still holds non-embeddable token ids "
                f"{out_of_range[:4]} after placeholder restoration; the request's "
                "modality positions or pad_values are incomplete"
            )
        return restored

    def _mrope_positions(self, req: Any, length: int) -> tuple[mx.array, int]:
        multimodal_inputs = getattr(req, "multimodal_inputs", None)
        positions = getattr(multimodal_inputs, "mrope_positions", None)
        if positions is None:
            rows = mx.broadcast_to(
                mx.arange(length, dtype=mx.int32)[None, :], (3, length)
            )
            return rows, 0
        if positions.ndim != 2 or positions.shape[0] != 3:
            raise ValueError(
                f"Qwen3-Omni M-RoPE positions must be [3, sequence], got "
                f"{tuple(positions.shape)}"
            )
        if int(positions.shape[1]) != length:
            raise ValueError(
                f"Qwen3-Omni M-RoPE positions cover {int(positions.shape[1])} tokens "
                f"but the prefill holds {length}"
            )
        delta_tensor = getattr(multimodal_inputs, "mrope_position_delta", None)
        delta = (
            0
            if delta_tensor is None
            else int(torch.as_tensor(delta_tensor).reshape(-1)[0])
        )
        rows = mx.array(positions.detach().cpu().to(torch.int32).numpy())
        return rows, delta

    def _modality_embeddings(self, model_inputs: dict[str, Any]) -> dict[str, mx.array]:
        embeddings: dict[str, mx.array] = {}
        for modality in _MODALITIES:
            value = model_inputs.get(f"{modality}_embeds")
            if value is None:
                continue
            embeddings[modality] = _torch_to_mlx(value)
        return embeddings

    def _deepstack_embeddings(
        self, req: Any, model_inputs: dict[str, Any]
    ) -> list[mx.array] | None:
        """Prompt-ordered DeepStack rows, one ``[visual_rows, hidden]`` per layer.

        Mirrors ``ThinkerModelRunner._inject_multimodal_embeds``: the visual
        rows are the image and video placeholder positions in prompt order, so a
        request carrying both modalities interleaves them by absolute position
        rather than concatenating image rows ahead of video rows.
        """

        merged = model_inputs.get("deepstack_visual_embeds")
        if merged is not None:
            return [_torch_to_mlx(layer) for layer in merged]

        image_layers = model_inputs.get("image_deepstack_visual_embeds")
        video_layers = model_inputs.get("video_deepstack_visual_embeds")
        if not image_layers and not video_layers:
            return None
        if not video_layers:
            return [_torch_to_mlx(layer) for layer in image_layers]
        if not image_layers:
            return [_torch_to_mlx(layer) for layer in video_layers]

        positions = getattr(req, "_omni_mm_positions", None) or {}
        image_positions = torch.as_tensor(
            positions.get("image", torch.zeros(0, dtype=torch.long)), dtype=torch.long
        )
        video_positions = torch.as_tensor(
            positions.get("video", torch.zeros(0, dtype=torch.long)), dtype=torch.long
        )
        order = torch.argsort(torch.cat([image_positions, video_positions]))
        slots = torch.empty_like(order)
        slots[order] = torch.arange(order.numel(), dtype=order.dtype)
        image_slots = slots[: image_positions.numel()]
        video_slots = slots[image_positions.numel() :]

        layers: list[mx.array] = []
        for image_layer, video_layer in zip(image_layers, video_layers):
            joint = image_layer.new_zeros(
                (order.numel(), image_layer.shape[-1]), dtype=image_layer.dtype
            )
            joint[image_slots] = image_layer
            joint[video_slots] = video_layer
            layers.append(_torch_to_mlx(joint))
        return layers

    def _require_single_request(self, req_id: str) -> None:
        resident = set(self._req_caches) | self._inflight_prefills
        others = sorted(resident - {req_id})
        if others:
            raise RuntimeError(
                "Apple Qwen3-Omni thinker serves one request at a time; "
                f"{others[0]!r} is still resident while {req_id!r} starts"
            )

    # -- forward -----------------------------------------------------------

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

        del new_slot_ids
        if req is None:
            raise ValueError("Qwen3-Omni MLX prefill requires its scheduler request")
        if prefix_slot_ids:
            raise NotImplementedError(
                "Qwen3-Omni MLX prefill does not support a radix prefix"
            )
        if not self.disable_radix_cache:
            raise RuntimeError(
                "Apple Qwen3-Omni thinker requires disable_radix_cache=True"
            )
        if logit_edit_row is not None or logprob_spec is not None:
            raise NotImplementedError(
                "Apple Qwen3-Omni thinker supports greedy decoding only"
            )
        if len(new_token_ids) != len(full_token_ids):
            raise NotImplementedError(
                "Apple Qwen3-Omni thinker runs with chunked prefill disabled; "
                f"got {len(new_token_ids)} of {len(full_token_ids)} prompt tokens"
            )
        self._require_single_request(req_id)
        # Chunked prefill is disabled for this stage, so every prefill is final
        # and always needs the logit head.
        del needs_logits

        restored = self._restore_placeholder_token_ids(req, new_token_ids)
        input_ids = mx.array([restored], dtype=mx.int32)
        model_inputs = getattr(req, "omni_model_inputs", None) or {}

        embeddings = self.model.embed_tokens(input_ids)
        modality_embeddings = self._modality_embeddings(model_inputs)
        if modality_embeddings:
            embeddings = merge_thinker_input_embeddings(
                embeddings,
                input_ids,
                modality_embeddings=modality_embeddings,
                placeholder_token_ids=self.placeholder_token_ids,
            )

        deepstack = self._deepstack_embeddings(req, model_inputs)
        visual_mask = (
            visual_placeholder_mask(
                input_ids, placeholder_token_ids=self.placeholder_token_ids
            )
            if deepstack is not None
            else None
        )
        positions, delta = self._mrope_positions(req, len(restored))

        step = self.model.prefill(
            input_ids,
            input_embeddings=embeddings,
            mrope_positions=positions,
            deepstack_visual_embeds=deepstack,
            visual_mask=visual_mask,
            cache=self.model.make_cache(),
            capture_layers=self.capture_layers,
        )
        self._req_mrope_delta[req_id] = delta
        self.last_prefill_input_ids = restored
        self.last_prefill_embeddings = embeddings
        self.last_prefill_positions = positions

        pending = MlxPendingPrefill(
            lazy_token=mx.argmax(step.logits[:, -1, :], axis=-1),
            cache=step.cache,
            req_id=req_id,
            # Later decode bookkeeping needs real model token ids, never the
            # out-of-vocabulary media cache keys.
            full_token_ids=self._restore_placeholder_token_ids(req, full_token_ids),
            req_pool_idx=req_pool_idx,
            synced_offset=0,
            lazy_logprobs=None,
        )
        self._attach_hidden_states(pending, step)
        self._inflight_prefills.add(req_id)
        return pending

    def prefill_finalize(self, pending: Any) -> int:
        token_id = int(pending.lazy_token.item())
        self._inflight_prefills.discard(pending.req_id)
        self._req_caches[pending.req_id] = pending.cache
        self._req_token_ids[pending.req_id] = list(pending.full_token_ids) + [token_id]
        return token_id

    def _decode_positions(self, req_id: str, cache: list[Any]) -> mx.array:
        """M-RoPE row for the next token.

        SGLang derives a decode position as ``seq_len - 1 + mrope_position_delta``
        on all three axes; ``cache[0].offset`` is exactly that ``seq_len - 1``
        (the tokens already resident in the cache).
        """

        position = int(cache[0].offset) + int(self._req_mrope_delta.get(req_id, 0))
        return mx.full((3, 1), position, dtype=mx.int32)

    def decode_batch_start(
        self,
        req_ids: list[str],
        edit_rows: mx.array | None = None,
        logprob_spec: Any = None,
        logits_hook: Any = None,
    ):
        from sglang.srt.hardware_backend.mlx.model_runner import MlxPendingDecode

        if logprob_spec is not None or edit_rows is not None or logits_hook is not None:
            raise NotImplementedError(
                "Apple Qwen3-Omni thinker supports greedy decoding only"
            )
        if len(req_ids) != 1:
            raise RuntimeError(
                "Apple Qwen3-Omni thinker decodes one request at a time, got "
                f"{len(req_ids)}"
            )
        req_id = req_ids[0]
        cache = self._req_caches[req_id]
        positions = self._decode_positions(req_id, cache)
        step = self.model.decode(
            mx.array([[self._req_token_ids[req_id][-1]]], dtype=mx.int32),
            mrope_positions=positions,
            cache=cache,
            capture_layers=self.capture_layers,
        )
        self.last_decode_positions = positions
        pending = MlxPendingDecode(
            lazy_tokens=mx.argmax(step.logits[:, -1, :], axis=-1),
            req_ids=[req_id],
            caches=[cache],
            lazy_logprobs=None,
            logprob_spec=None,
            edit_rows=None,
        )
        self._attach_hidden_states(pending, step)
        return pending

    def decode_batch_start_chained(self, prev: Any):
        from sglang.srt.hardware_backend.mlx.model_runner import MlxPendingDecode

        if len(prev.req_ids) != 1 or prev.logprob_spec is not None:
            raise NotImplementedError(
                "Apple Qwen3-Omni thinker chains one greedy request at a time"
            )
        req_id = prev.req_ids[0]
        cache = prev.caches[0]
        positions = self._decode_positions(req_id, cache)
        step = self.model.decode(
            prev.lazy_tokens[:, None],
            mrope_positions=positions,
            cache=cache,
            capture_layers=self.capture_layers,
        )
        self.last_decode_positions = positions
        pending = MlxPendingDecode(
            lazy_tokens=mx.argmax(step.logits[:, -1, :], axis=-1),
            req_ids=list(prev.req_ids),
            caches=prev.caches,
            lazy_logprobs=None,
            logprob_spec=None,
            edit_rows=None,
        )
        self._attach_hidden_states(pending, step)
        return pending

    def decode_batch_finalize(self, pending: Any) -> list[int]:
        tokens = [int(token) for token in pending.lazy_tokens.tolist()]
        for req_id, token_id in zip(pending.req_ids, tokens):
            self._req_token_ids[req_id].append(token_id)
        return tokens


def make_qwen3_omni_thinker_mlx_runner_class() -> type:
    """Return the thinker runner class once the MLX backend is selected.

    The class body holds no SGLang MLX import, but the pending-step dataclasses
    it constructs do, so this factory keeps the same "resolve after backend
    selection" contract as the Qwen3-ASR runner factory.
    """

    from sglang.srt.hardware_backend.mlx import model_runner as _mlx_model_runner

    assert hasattr(_mlx_model_runner, "MlxPendingPrefill")
    return Qwen3OmniThinkerMlxRunner


def build_qwen3_omni_thinker_mlx_runner(
    *,
    model: Qwen3OmniMlxThinker,
    placeholder_token_ids: dict[str, int],
    capture_hidden_layers: tuple[int, ...] | list[int] | None,
    accept_hidden_layer: int | None,
    disable_radix_cache: bool = True,
    pool_size: int | None = None,
) -> Qwen3OmniThinkerMlxRunner:
    """Build a thinker runner around an already-constructed native model."""

    runner_class = make_qwen3_omni_thinker_mlx_runner_class()
    return runner_class.from_model(
        model=model,
        placeholder_token_ids=placeholder_token_ids,
        capture_hidden_layers=capture_hidden_layers,
        accept_hidden_layer=accept_hidden_layer,
        disable_radix_cache=disable_radix_cache,
        pool_size=pool_size,
    )


# ---------------------------------------------------------------------------
# Native MLX talker: checkpoint loading, native prefill, scheduler runner
# ---------------------------------------------------------------------------


def load_qwen3_omni_mlx_talker(
    model_path: str,
    *,
    trust_remote_code: bool = False,
    revision: str | None = None,
) -> dict[str, Any]:
    """Load the native MLX talker (backbone, projections, code predictor).

    Handles both published layouts:

    * the **official** per-expert dense checkpoint, whose MoE experts are one
      linear per expert (fused by ``sanitize``); and
    * a **converted 4-bit** checkpoint, whose packed ``weight``/``scales``/
      ``biases`` only fit already-quantized modules -- so the checkpoint's own
      ``quantization`` metadata is parsed and applied *before* the strict load.

    Shard selection is ownership-based (see
    :func:`read_qwen3_omni_component_weights`), so neither the thinker's nor
    code2wav's tensors can enter the talker load even when a converted export
    strips the component prefixes into per-component subdirectories.

    Every failure propagates: an MLX talker was explicitly requested, so a
    Torch-MPS fallback must never be substituted.
    """

    from sglang.srt.hardware_backend.mlx.remote_code_gate import (
        ensure_remote_code_allowed,
        resolve_model_directory,
    )

    directory = resolve_model_directory(model_path, revision=revision)
    ensure_remote_code_allowed(directory, trust_remote_code)
    directory = Path(directory)
    raw = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    config = Qwen3OmniMlxConfig.from_dict(raw)
    talker_raw = raw["talker_config"]

    logger.info(
        "Loading native MLX Qwen3-Omni talker: %s (attention_bias=%s, "
        "quantization=%s, code_groups=%d)",
        directory,
        config.talker.text_config.attention_bias,
        config.quantization,
        config.talker.num_code_groups,
    )
    started = time.perf_counter()

    model = Qwen3OmniMlxTalker.from_omni_config(config)
    weights = read_qwen3_omni_component_weights(
        directory,
        component="talker",
        official_prefixes=_TALKER_WEIGHT_PREFIXES,
        local_prefixes=_TALKER_LOCAL_PREFIXES,
    )
    load_qwen3_omni_mlx_component(
        model,
        weights,
        sanitizer=model.sanitize,
        quantization=config.quantization,
    )
    model.eval()

    logger.info(
        "Loaded native MLX Qwen3-Omni talker in %.2fs (%d layers)",
        time.perf_counter() - started,
        model.num_layers,
    )
    return {
        "model": model,
        "config": config,
        "quantization": config.quantization,
        "raw": raw,
        "talker_raw": talker_raw,
        "directory": directory,
    }


@dataclass(slots=True)
class _MlxTalkerStepState:
    """One completed talker step, materialised for the CPU Torch queues."""

    codes: torch.Tensor
    feedback: torch.Tensor


class Qwen3OmniMlxTalkerModelRunner(ModelRunner):
    """Omni model runner that drives the native MLX talker.

    Ownership split, mirroring the reviewed Torch talker runner
    (:class:`~sglang_omni.models.qwen3_omni.talker_model_runner.QwenTalkerModelRunner`):

    * **MLX side (this runner).** One outer talker KV cache per request, the
      per-request codec suppression mask, and the M-RoPE row for each step.
      The recurrence (``feedback + next_text_row``), the greedy layer-0
      selection, and the residual-RVQ group expansion all live in the reviewed
      model; nothing is recomputed here.
    * **Torch side (unchanged).** The pending text FIFO
      (``PendingTextTensorQueue``) and the feedback deque stay CPU float32 and
      are consumed through ``QwenTalkerModelRunner``'s own helpers, so decode
      readiness, thinker-done padding, and row ownership are the *same* code
      as the CUDA path. Only the single row consumed by the current step
      crosses into MLX.

    Apple policy: batch-1, greedy, no chunked prefill, no radix prefix, and no
    async decode lookahead (``apply_qwen3_omni_apple_profile`` already disables
    partial talker start, and speculative queue state is refused outright).
    """

    def __init__(
        self,
        tp_worker: Any,
        output_processor: Any,
        outbox: Any,
        *,
        mlx_talker: Qwen3OmniMlxTalker | None = None,
        code2wav_target: str = "code2wav",
        feedback_enabled: bool = True,
    ) -> None:
        super().__init__(tp_worker, output_processor)
        self._outbox = outbox
        self._code2wav_target = code2wav_target
        self._feedback_enabled = bool(feedback_enabled)
        resolved = (
            mlx_talker
            if mlx_talker is not None
            else getattr(tp_worker, "mlx_talker", None)
        )
        if resolved is None:
            raise ValueError(
                "Qwen3OmniMlxTalkerModelRunner requires a loaded MLX talker; the "
                "worker exposed none"
            )
        self._mlx_model: Qwen3OmniMlxTalker = resolved
        self._caches: dict[str, list[Any]] = {}
        self._mrope_delta: dict[str, int] = {}
        self._suppress_masks: dict[str, mx.array] = {}
        # Steps that have been computed by the forward but whose codes have not
        # yet been emitted / whose feedback has not yet been queued.
        self._pending_steps: dict[str, _MlxTalkerStepState] = {}
        self._inflight_prefills: set[str] = set()
        # Diagnostics the unit tests read to pin position progression.
        self.last_positions: torch.Tensor | None = None

    # -- accessors ---------------------------------------------------------

    @property
    def mlx_model(self) -> Qwen3OmniMlxTalker:
        return self._mlx_model

    @property
    def num_code_groups(self) -> int:
        return int(self._mlx_model.num_code_groups)

    @property
    def codec_vocab_size(self) -> int:
        return int(self._mlx_model.vocab_size)

    def has_request(self, request_id: str) -> bool:
        return request_id in self._caches

    # -- lifecycle ---------------------------------------------------------

    def abort_request(self, request_id: str) -> None:
        """Scheduler abort callback: drop every per-request resource."""

        self._release_request(request_id)

    def on_request_finished(self, request_id: str, req_data: Any) -> None:
        """Normal completion: drop every per-request resource."""

        self._release_request(request_id, req_data)

    def _release_request(self, request_id: str, req_data: Any = None) -> None:
        self._caches.pop(request_id, None)
        self._mrope_delta.pop(request_id, None)
        self._suppress_masks.pop(request_id, None)
        self._pending_steps.pop(request_id, None)
        self._inflight_prefills.discard(request_id)
        if req_data is None:
            return
        release_talker_host_queues(req_data)

    def clear(self) -> None:
        self._caches.clear()
        self._mrope_delta.clear()
        self._suppress_masks.clear()
        self._pending_steps.clear()
        self._inflight_prefills.clear()

    # -- no async lookahead ------------------------------------------------

    def lookahead_eligible(self, batch: Any) -> bool:
        """The Apple talker never runs a speculative decode step.

        A lookahead launch would have to sample -- and therefore emit a code
        frame and queue a feedback row -- before its predecessor was resolved,
        which the single feedback/text row-per-step contract cannot express.
        """

        del batch
        return False

    def execute_launch(self, scheduler_output: Any):
        raise NotImplementedError(
            "Apple Qwen3-Omni MLX talker does not support async decode "
            "lookahead; every step is resolved synchronously"
        )

    def execute_resolve(self, pending: Any):
        if pending is None:
            return None
        raise NotImplementedError(
            "Apple Qwen3-Omni MLX talker does not support async decode "
            "lookahead; every step is resolved synchronously"
        )

    # -- SGLang execution contract ----------------------------------------

    def _build_forward_batch(self, scheduler_output: Any):
        """No ``ForwardBatch``: the MLX talker consumes embeddings directly.

        SGLang's bookkeeping stub carries no Torch attention backend state, so
        (exactly as the MLX thinker scheduler runner does) the batch itself is
        the only forward input.
        """

        schedule_batch = scheduler_output.batch_data
        if schedule_batch is None:
            return None
        return None, schedule_batch, bool(schedule_batch.forward_mode.is_extend())

    def sample_before_post_prefill(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> bool:
        del forward_batch, schedule_batch, requests
        return False

    def sample_before_post_decode(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> bool:
        del forward_batch, schedule_batch, requests
        return False

    def is_decode_batch_ready(self, schedule_batch: Any) -> bool:
        """Identical readiness rule to the Torch talker runner."""

        if not self._feedback_enabled or not schedule_batch.forward_mode.is_decode():
            return True
        return all(
            QwenTalkerModelRunner._data_has_next_decode_input(
                getattr(req, "_omni_data", None)
            )
            for req in schedule_batch.reqs
        )

    # -- forward -----------------------------------------------------------

    def custom_prefill_forward(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> Any:
        del forward_batch
        from sglang.srt.managers.utils import GenerationBatchResult

        sched_req = self._single_request(requests)
        request_id = sched_req.request_id
        if request_id in self._caches or request_id in self._inflight_prefills:
            raise RuntimeError(
                "Apple Qwen3-Omni MLX talker does not support re-prefilling a "
                f"resident request ({request_id!r})"
            )
        self._inflight_prefills.add(request_id)
        try:
            rows = self._prefill_rows(sched_req)
            positions = self._prefill_positions(sched_req, int(rows.shape[0]))
            cache = self._mlx_model.make_cache()
            step = self._mlx_model.prefill(
                mx.array(rows.numpy())[None, :, :],
                mrope_positions=mx.array(positions.numpy().astype(np.int32)),
                input_embeddings_are_projected=True,
                suppress_tokens=self._suppress_mask(sched_req),
                cache=cache,
            )
            self._caches[request_id] = step.cache
        finally:
            self._inflight_prefills.discard(request_id)
        return GenerationBatchResult(
            next_token_ids=self._record_step(request_id, step, positions),
        )

    def custom_decode_forward(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> Any:
        del forward_batch, schedule_batch
        from sglang.srt.managers.utils import GenerationBatchResult

        sched_req = self._single_request(requests)
        request_id = sched_req.request_id
        cache = self._caches.get(request_id)
        if cache is None:
            raise RuntimeError(
                "Apple Qwen3-Omni MLX talker decode has no prefilled cache for "
                f"{request_id!r}"
            )
        rows = self._take_next_decode_rows(sched_req)
        if rows is None:
            raise RuntimeError(
                "MLX talker decode requires feedback and text input; the "
                "scheduler must defer the batch until both rows are ready"
            )
        feedback_row, text_row = rows
        QwenTalkerModelRunner._append_decode_input_history(
            sched_req.data, feedback_row + text_row
        )
        positions = self._decode_positions(request_id, cache)
        step = self._mlx_model.decode(
            feedback=mx.array(feedback_row.numpy()),
            next_text_row=mx.array(text_row.numpy()),
            mrope_positions=mx.array(positions.numpy().astype(np.int32)),
            cache=cache,
            suppress_tokens=self._suppress_mask(sched_req),
        )
        self._caches[request_id] = step.cache
        return GenerationBatchResult(
            next_token_ids=self._record_step(request_id, step, positions),
        )

    def post_prefill(
        self, result: Any, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> None:
        del result, forward_batch
        self._emit_codes_and_queue_feedback(
            schedule_batch=schedule_batch, requests=requests
        )

    def post_decode(
        self, result: Any, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> None:
        del result, forward_batch
        self._emit_codes_and_queue_feedback(
            schedule_batch=schedule_batch, requests=requests
        )

    # -- step bookkeeping --------------------------------------------------

    def _record_step(
        self, request_id: str, step: Any, positions: torch.Tensor
    ) -> torch.Tensor:
        """Materialise one MLX step into CPU Torch and stage it for emission."""

        mx.eval(step.codes, step.feedback)
        codes = torch.from_numpy(
            np.ascontiguousarray(np.asarray(step.codes).astype(np.int64))
        ).reshape(-1)
        if int(codes.numel()) != self.num_code_groups:
            raise RuntimeError(
                f"MLX talker produced {int(codes.numel())} codes but the config "
                f"declares {self.num_code_groups} code groups"
            )
        feedback = _mlx_to_torch(step.feedback).reshape(-1).contiguous()
        self._pending_steps[request_id] = _MlxTalkerStepState(
            codes=codes,
            feedback=feedback,
        )
        self.last_positions = positions
        return codes[:1].clone()

    def _emit_codes_and_queue_feedback(
        self, *, schedule_batch: Any, requests: list
    ) -> None:
        """Emit this step's code row once and queue its feedback row.

        Matches ``QwenTalkerModelRunner._emit_code_chunks_and_feedback``: one
        ``OutgoingMessage(target="code2wav", type="stream")`` per step carrying
        ``[num_code_groups]`` codes, and one feedback row appended to the
        request's pending queue. The codec-EOS frame is emitted too -- exactly
        as the CUDA path does, and code2wav drops it -- so a terminal step
        neither duplicates nor silently swallows a frame.
        """

        if not self._feedback_enabled:
            self._pending_steps.clear()
            return
        for index, sched_req in enumerate(requests):
            request_id = sched_req.request_id
            state = self._pending_steps.pop(request_id, None)
            if state is None:
                raise RuntimeError(
                    f"MLX talker has no computed step to emit for {request_id!r}"
                )
            emit_talker_step(
                outbox=self._outbox,
                target=self._code2wav_target,
                request_id=schedule_batch.reqs[index].rid,
                data=sched_req.data,
                codes=state.codes,
                feedback=state.feedback,
            )

    # -- inputs ------------------------------------------------------------

    @staticmethod
    def _single_request(requests: list) -> Any:
        return require_single_request(requests, backend_name="MLX talker")

    @staticmethod
    def _prefill_rows(sched_req: Any) -> torch.Tensor:
        """The already-projected CPU float32 prompt rows for this prefill."""

        return projected_prefill_rows(sched_req, backend_name="MLX talker")

    @staticmethod
    def _take_next_decode_rows(
        sched_req: Any,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Consume exactly one feedback row and one text row, FIFO.

        The queue protocol is ``QwenTalkerModelRunner``'s, helper for helper:
        peek both rows, refuse the step when either is missing, fall back to
        the projected TTS pad row only once the thinker stream is done, then
        pop one feedback row and (when present) one text row. The two rows are
        returned *unsummed* because the MLX talker owns the recurrence; their
        sum is exactly ``_take_next_decode_input_embed``'s return value.
        """

        data = sched_req.data
        feedback = QwenTalkerModelRunner._peek_left(
            getattr(data, "pending_feedback_queue", None)
        )
        if feedback is None:
            return None
        next_text = QwenTalkerModelRunner._peek_left(
            getattr(data, "pending_text_queue", None)
        )
        if next_text is None:
            if not data.thinker_chunks_done:
                return None
            next_text = getattr(data, "tts_pad_embed", None)
            if next_text is None:
                return None
        # Ownership check before combination: both rows must already be the
        # request's own CPU float32 rows, never a device tensor or a
        # silently-downcast view.
        feedback_row = QwenTalkerModelRunner._decode_row(
            feedback, device=_CPU, dtype=torch.float32
        )
        text_row = QwenTalkerModelRunner._decode_row(
            next_text, device=_CPU, dtype=torch.float32
        )
        QwenTalkerModelRunner._pop_left(data.pending_feedback_queue)
        if getattr(data, "pending_text_queue", None):
            QwenTalkerModelRunner._pop_left(data.pending_text_queue)
        return feedback_row.contiguous(), text_row.contiguous()

    # -- positions ---------------------------------------------------------

    def _prefill_positions(self, sched_req: Any, length: int) -> torch.Tensor:
        """The ``[3, length]`` M-RoPE rows for the talker prompt.

        The request builder attaches the rows it computed (``linear_mrope_positions``
        for a prompt with no emitted multimodal segment, the full multimodal
        computation otherwise). When it attached none -- a request built
        without ``talker_model_inputs`` -- the same linear arange is rebuilt
        here from the shared helper rather than re-derived.
        """

        request_id = sched_req.request_id
        multimodal_inputs = getattr(sched_req.data.req, "multimodal_inputs", None)
        positions = getattr(multimodal_inputs, "mrope_positions", None)
        if positions is None:
            rows, delta = linear_mrope_positions(length)
        else:
            rows = torch.as_tensor(positions)
            if rows.ndim != 2 or int(rows.shape[0]) != 3:
                raise ValueError(
                    "Qwen3-Omni talker M-RoPE positions must be [3, sequence], "
                    f"got {tuple(rows.shape)}"
                )
            if int(rows.shape[1]) != length:
                raise ValueError(
                    f"Qwen3-Omni talker M-RoPE positions cover {int(rows.shape[1])} "
                    f"tokens but the prefill holds {length}"
                )
            delta = getattr(multimodal_inputs, "mrope_position_delta", None)
        self._mrope_delta[request_id] = (
            0 if delta is None else int(torch.as_tensor(delta).reshape(-1)[0])
        )
        return rows.detach().cpu().to(torch.int64)

    def _decode_positions(self, request_id: str, cache: list[Any]) -> torch.Tensor:
        """The ``[3, 1]`` M-RoPE row for the next talker token.

        SGLang derives a decode position as ``seq_len - 1 + mrope_position_delta``
        on all three axes; ``cache[0].offset`` is exactly that ``seq_len - 1``
        (the rows already resident in the talker cache). With the linear
        prompt positions this continues ``0..L-1`` with ``L, L+1, ...``, step
        for step with the CUDA talker.
        """

        position = int(cache[0].offset) + int(self._mrope_delta.get(request_id, 0))
        return torch.full((3, 1), position, dtype=torch.int64)

    # -- suppression -------------------------------------------------------

    def _suppress_mask(self, sched_req: Any) -> mx.array | None:
        """The request's additive codec suppression mask, built once."""

        request_id = sched_req.request_id
        cached = self._suppress_masks.get(request_id)
        if cached is not None:
            return cached
        suppress_tokens = getattr(sched_req.data, "suppress_tokens", None)
        if not suppress_tokens:
            return None
        mask = build_suppress_mask(self.codec_vocab_size, suppress_tokens)
        self._suppress_masks[request_id] = mask
        return mask


def make_qwen3_omni_talker_mlx_runner_class() -> type:
    """Return the talker model-runner class once the MLX backend is selected.

    Mirrors the thinker factory's "resolve after backend selection" contract:
    the class body holds no SGLang MLX import, but the worker it is bound to
    does, so the import is asserted here rather than at module import time.
    """

    from sglang.srt.hardware_backend.mlx import model_runner_stub as _stub

    assert hasattr(_stub, "MlxModelRunnerStub")
    return Qwen3OmniMlxTalkerModelRunner


def build_qwen3_omni_talker_mlx_runner(
    *,
    tp_worker: Any,
    output_processor: Any,
    outbox: Any,
    mlx_talker: Qwen3OmniMlxTalker | None = None,
    code2wav_target: str = "code2wav",
    feedback_enabled: bool = True,
) -> "Qwen3OmniMlxTalkerModelRunner":
    """Build the talker model runner around an already-loaded MLX talker."""

    runner_class = make_qwen3_omni_talker_mlx_runner_class()
    return runner_class(
        tp_worker,
        output_processor,
        outbox,
        mlx_talker=mlx_talker,
        code2wav_target=code2wav_target,
        feedback_enabled=feedback_enabled,
    )


def _create_qwen3_omni_talker_mlx_worker(
    *,
    config: Any,
    server_args: Any,
    gpu_id: int,
    tp_rank: int = 0,
):
    """Zero-weight scheduler worker carrying the loaded native MLX talker.

    The talker's forward consumes talker-space *embeddings*, not token ids, so
    SGLang's token-id-driven MLX worker API cannot express it. The worker is
    therefore an external-forward worker (bookkeeping pools only) and
    :class:`Qwen3OmniMlxTalkerModelRunner` owns the forward. The MLX weights
    and the native MLX prefill builder are built here, once, so the request
    builder never sees the zero-weight stub model.
    """

    from sglang_omni.model_runner.external_model_worker import (
        _build_parallel_state,
        _make_external_worker_class,
        _publish_scheduler_runtime_context,
        _resolve_nccl_port,
    )

    base_worker_class = _make_external_worker_class()

    class OmniQwen3OmniTalkerMlxWorker(base_worker_class):
        """External-forward worker whose runner is the native MLX talker."""

        def _init_model_runner(self) -> None:
            super()._init_model_runner()
            if self.server_args.quantization is not None:
                raise NotImplementedError(
                    "Apple Qwen3-Omni talker reads quantization from the "
                    "checkpoint's own metadata; on-the-fly preset "
                    f"{self.server_args.quantization!r} is not supported"
                )
            if self.server_args.mlx_enable_sampling:
                raise NotImplementedError(
                    "Apple Qwen3-Omni talker supports greedy codec generation "
                    "only; MLX sampling is not enabled for this stage"
                )
            loaded = load_qwen3_omni_mlx_talker(
                self.server_args.model_path,
                trust_remote_code=self.server_args.trust_remote_code,
                revision=self.server_args.revision,
            )
            self.mlx_talker = loaded["model"]
            self.mlx_talker_quantization = loaded["quantization"]
            raw = loaded["raw"]
            talker_raw = loaded["talker_raw"]
            self.mlx_talker_prefill_builder = (
                Qwen3OmniMlxTalkerPrefillBuilder.from_talker(
                    loaded["model"],
                    model_path=str(loaded["directory"]),
                    special_token_ids={
                        "audio_token_id": raw["thinker_config"]["audio_token_id"],
                        "image_token_id": raw["thinker_config"]["image_token_id"],
                        "video_token_id": raw["thinker_config"]["video_token_id"],
                        "tts_bos_token_id": raw["tts_bos_token_id"],
                        "tts_eos_token_id": raw["tts_eos_token_id"],
                        "tts_pad_token_id": raw["tts_pad_token_id"],
                        "im_start_token_id": raw["im_start_token_id"],
                        "im_end_token_id": raw["im_end_token_id"],
                        "system_token_id": raw["system_token_id"],
                        "user_token_id": raw["user_token_id"],
                        "assistant_token_id": raw["assistant_token_id"],
                        "codec_bos_id": talker_raw["codec_bos_id"],
                        "codec_nothink_id": talker_raw["codec_nothink_id"],
                        "codec_think_bos_id": talker_raw["codec_think_bos_id"],
                        "codec_think_eos_id": talker_raw["codec_think_eos_id"],
                        "codec_pad_id": talker_raw["codec_pad_id"],
                    },
                    speaker_map=talker_raw.get("speaker_id") or {},
                )
            )

    ps = _build_parallel_state(server_args, gpu_id=gpu_id, tp_rank=tp_rank)
    nccl_port = _resolve_nccl_port(server_args, config.nccl_port)
    _publish_scheduler_runtime_context(server_args)
    return OmniQwen3OmniTalkerMlxWorker(
        external_backend_name="Qwen3-Omni MLX talker",
        server_args=server_args,
        gpu_id=gpu_id,
        ps=ps,
        nccl_port=nccl_port,
    )


def create_qwen3_omni_mlx_worker(
    *,
    config: Any,
    server_args: Any,
    gpu_id: int,
    tp_rank: int = 0,
):
    """Construct the MLX worker for a Qwen3-Omni stage.

    Any failure here -- an unsupported architecture, a checkpoint that does not
    load, a config the native model refuses -- propagates to the caller. The
    MLX backend is explicitly requested, so a Torch-MPS worker must never be
    substituted behind the user's back.
    """

    architecture = config.model_arch_override
    if architecture == "Qwen3OmniTalker":
        return _create_qwen3_omni_talker_mlx_worker(
            config=config,
            server_args=server_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
        )
    if architecture != "Qwen3OmniThinkerForCausalLM":
        raise NotImplementedError(
            "Omni's MLX Qwen3-Omni worker supports Qwen3OmniThinkerForCausalLM "
            f"and Qwen3OmniTalker; got {architecture!r}"
        )

    from sglang.srt.hardware_backend.mlx.model_runner_stub import MlxModelRunnerStub
    from sglang.srt.hardware_backend.mlx.tp_worker import MlxTpModelWorker

    from sglang_omni.model_runner.external_model_worker import (
        _build_parallel_state,
        _publish_scheduler_runtime_context,
        _resolve_nccl_port,
    )

    capture_hidden_layers = getattr(config, "capture_hidden_layers", None)

    class OmniQwen3OmniThinkerMlxWorker(MlxTpModelWorker):
        @property
        def tp_rank(self) -> int:
            return self.ps.tp_rank

        def _init_model_runner(self):
            MlxModelRunnerStub.validate_startup_weight_load_mode(self.server_args)
            runner_class = make_qwen3_omni_thinker_mlx_runner_class()
            init_kwargs = {
                "model_path": self.server_args.model_path,
                "trust_remote_code": self.server_args.trust_remote_code,
                "disable_radix_cache": self.server_args.disable_radix_cache,
                "mem_fraction_static": self.server_args.mem_fraction_static,
                "quantization": self.server_args.quantization,
                "revision": self.server_args.revision,
                "enable_sampling": self.server_args.mlx_enable_sampling,
                "sampling_rng_seed": self.server_args.random_seed,
                "deterministic_seeding": (
                    self.server_args.enable_deterministic_inference
                ),
                "capture_hidden_layers": capture_hidden_layers,
            }
            if self.server_args.max_total_tokens is not None:
                init_kwargs["pool_size"] = self.server_args.max_total_tokens
            self._mlx_runner = runner_class(**init_kwargs)
            self._model_runner = MlxModelRunnerStub(
                model_config=self.model_config,
                mem_fraction_static=self.server_args.mem_fraction_static,
                gpu_id=self.gpu_id,
                ps=self.ps,
                nccl_port=self.nccl_port,
                server_args=self.server_args,
                is_draft_worker=self.is_draft_worker,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                memory_pool_config=self.memory_pool_config,
                mlx_pool_size=self._mlx_runner.pool_size,
            )
            self._mlx_active_rids = set()
            self._mlx_pool_initialized = False

        def finalize_mlx_result(self, launch: Any, reqs: list) -> Any:
            """Attach this step's thinker captures to the batch result.

            The thinker stream builder reads
            ``logits_output.hidden_states`` as a ``{"embed": ..., N: ...}``
            dict; the MLX runner is its sole source because this worker holds
            SGLang's zero-weight stub model and therefore no static capture.
            """

            result = super().finalize_mlx_result(launch, reqs)
            hidden = self._pop_launch_hidden_states(launch)
            if hidden is not None and result.logits_output is not None:
                result.logits_output.hidden_states = hidden
            return result

        def _pop_launch_hidden_states(self, launch: Any):
            if launch.mode == "decode" and launch.decode is not None:
                return self._mlx_runner.pop_hidden_states(launch.decode)
            if launch.mode == "extend" and launch.prefills:
                return self._mlx_runner.pop_hidden_states(launch.prefills[0])
            return None

        def get_tp_group(self):
            return self.model_runner.tp_group

        def get_attention_tp_group(self):
            return self.model_runner.attention_tp_group

        def get_attention_tp_cpu_group(self):
            return self.model_runner.attention_tp_group.cpu_group

    ps = _build_parallel_state(server_args, gpu_id=gpu_id, tp_rank=tp_rank)
    nccl_port = _resolve_nccl_port(server_args, config.nccl_port)
    _publish_scheduler_runtime_context(server_args)
    return OmniQwen3OmniThinkerMlxWorker(
        server_args=server_args,
        gpu_id=gpu_id,
        ps=ps,
        nccl_port=nccl_port,
    )


def _scheduler_runner_base() -> type:
    """The shared MLX scheduler runner.

    Resolved through a call so ``mlx_model_worker`` -- which reaches back into
    this module only through a deferred factory -- is imported exactly once and
    in one direction.
    """

    from sglang_omni.model_runner.mlx_model_worker import MlxSchedulerModelRunner

    return MlxSchedulerModelRunner


class Qwen3OmniMlxSchedulerModelRunner(_scheduler_runner_base()):  # type: ignore[misc]
    """Omni scheduler adapter for the native MLX thinker.

    Adds exactly two things to the shared MLX scheduler runner:

    * per-request MLX cache release on abort and on normal completion, so a
      request's KV cache and capture rows never outlive it;
    * an explicit guarantee that the runner-populated
      ``logits_output.hidden_states`` dictionary survives finalize and reaches
      ``SGLangOutputProcessor`` (which is constructed with ``model=None`` for
      this backend, making that dictionary the sole capture source).
    """

    def abort_request(self, request_id: str) -> None:
        """Scheduler abort callback: drop the request's MLX cache."""

        self._release_mlx_request(request_id)

    def on_request_finished(self, request_id: str, req_data: Any) -> None:
        """Normal completion: drop the request's MLX cache."""

        del req_data
        self._release_mlx_request(request_id)

    def _release_mlx_request(self, request_id: str) -> None:
        runner = getattr(self.tp_worker, "_mlx_runner", None)
        if runner is None:
            return
        runner.remove_request(request_id)

    def _finalize(
        self,
        batch_result,
        forward_batch,
        schedule_batch,
        scheduler_output,
        skip_rids: set[str] | None = None,
    ):
        logits_output = getattr(batch_result, "logits_output", None)
        hidden = getattr(logits_output, "hidden_states", None)
        output = super()._finalize(
            batch_result,
            forward_batch,
            schedule_batch,
            scheduler_output,
            skip_rids=skip_rids,
        )
        if hidden is not None and logits_output is not None:
            # Nothing downstream may consume the captures destructively: the
            # talker stream builder reads them again off the request output.
            logits_output.hidden_states = hidden
        return output
