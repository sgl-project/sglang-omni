# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni encoder adapters between v1 :class:`PipelineState` and SGLang.

For each encoder stage there is one adapter implementing
``build_batch`` / ``run_feature`` / ``slice_results``:

- ``build_batch`` materializes a :class:`BatchPlan` from a flat list of
  :class:`IncomingMessage`. Skip requests (preprocessor-stamped
  ``_skip``) contribute no upstream items; active requests yield one or
  more :class:`MultimodalDataItem` per request, with per-request
  bookkeeping captured in :class:`RequestSpan`.
- ``run_feature`` calls the loaded upstream encoder submodule directly
  — ``Qwen3OmniMoeVisionEncoder`` for image/video, ``Qwen3OmniMoeAudioEncoder``
  for audio — instead of the full ``thinker.get_*_feature`` entry
  points. The encoder runner partial-loads ONLY these submodules (see
  ``EncoderModuleSpec``), so ``model.thinker`` is never instantiated and
  the full Qwen3-Omni LLM weights never reach the encoder GPU.
  Empty plans short-circuit and never call any submodule.
- ``slice_results`` un-batches the upstream output back to per-request
  ``encoder_outs`` dicts using the captured spans.
- ``lookup_cached_output`` / ``store_cached_output`` mirror the local encoder
  ``StageOutputCache`` behavior on the TP entry rank. Cache hits are resolved
  before tensor lifting and TP broadcast, so all ranks still see the same
  miss-only forward plan.
"""
from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Any, Callable, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoder,
    Qwen3OmniMoeVisionEncoder,
    _get_feat_extract_output_lengths,
)

from sglang_omni.models.qwen3_omni.payload_types import PipelineState
from sglang_omni.models.qwen3_omni.request_builders import (
    apply_encoder_result,
    build_encoder_request,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import IncomingMessage

if TYPE_CHECKING:
    from sglang_omni.models.qwen3_omni.request_builders import EncoderRequestData

logger = logging.getLogger(__name__)

IMAGE_STAGE = "image_encoder"
AUDIO_STAGE = "audio_encoder"

# Sourced from sglang_omni.models.qwen3_omni.pipeline.visual_budget — duplicated
# here to avoid pulling that module's heavyweight import chain (preprocessor,
# transformers video utils) into the lean adapter module.
QWEN3_IMAGE_ENCODER_ACTIVATION_MULTIPLIER = 5
QWEN3_AUDIO_ENCODER_ACTIVATION_MULTIPLIER = 5


def _per_rank_activation_cost(
    *,
    replicated_bytes: int,
    sharded_bytes: int,
    multiplier: int,
    tp_size: int,
) -> int:
    """Estimate one rank's activation cost.

    Raw multimodal inputs are broadcast to every TP rank, while the
    hidden-state activation proxy is sharded by tensor parallelism.
    """
    replicated = int(replicated_bytes) * int(multiplier)
    sharded = int(sharded_bytes) * int(multiplier)
    return replicated + (sharded + int(tp_size) - 1) // int(tp_size)


# ---------------------------------------------------------------------------
# EncoderModuleSpec — declares which upstream encoder submodules a stage
# loads and which checkpoint prefixes feed them. The shared SGLang encoder
# runner uses these specs to partial-load only the relevant submodule
# weights, avoiding the ~57 GB of LLM/talker weights that
# ``Qwen3OmniMoeForConditionalGeneration`` would otherwise pull in.
# See ``docs/developer_reference/encoder_tp_path_b_design.md`` →
# "Upstream Reuse Boundary" + "EncoderModuleSpec".
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class EncoderModuleSpec:
    """Declaration of an upstream encoder submodule the runner should load.

    Args:
        name: Local attribute name the submodule will be exposed as on
            ``EncoderModuleContainer`` (e.g. ``"visual"``,
            ``"audio_tower"``). Adapters reference this name when they
            access the submodule from ``model.<name>`` in
            ``run_feature``.
        build_module: Callable receiving the HF config and a
            ``QuantizationConfig | None`` and returning the concrete
            ``nn.Module`` instance.
        checkpoint_prefixes: Tuple of checkpoint key prefixes to accept
            for this submodule. Any (name, tensor) pair in the
            checkpoint stream whose name does NOT start with one of
            these prefixes is dropped without allocation.
        checkpoint_rewrites: Tuple of ``(old, new)`` pairs applied
            sequentially via ``str.replace`` to map upstream checkpoint
            keys onto the local submodule's parameter namespace. Order
            matters; rewrites are applied left-to-right on each match.
        stacked_params_mapping: ``(target_substring, source_substring,
            shard_id)`` tuples describing fused-shard params (e.g.
            ``q_proj`` + ``k_proj`` + ``v_proj`` → ``qkv_proj``). When
            an incoming key contains ``source_substring``, the container
            replaces it with ``target_substring`` and dispatches to
            ``param.weight_loader(param, weight, shard_id)``. Only used
            when the submodule does not provide its own ``load_weights``;
            if the submodule has ``load_weights``, the container delegates
            to it and this field is ignored.
    """

    name: str
    build_module: Callable[[Any, Any], nn.Module]
    checkpoint_prefixes: tuple[str, ...]
    checkpoint_rewrites: tuple[tuple[str, str], ...] = ()
    stacked_params_mapping: tuple[tuple[str, str, str], ...] = ()


def _build_qwen3_omni_visual(hf_config: Any, quant_config: Any) -> nn.Module:
    """Build ``Qwen3OmniMoeVisionEncoder`` from a thinker-aware HF config."""
    thinker = getattr(hf_config, "thinker_config", hf_config)
    return Qwen3OmniMoeVisionEncoder(
        thinker.vision_config,
        quant_config=quant_config,
        norm_eps=getattr(thinker, "rms_norm_eps", 1e-6),
        prefix="visual",
    )


def _build_qwen3_omni_audio(hf_config: Any, quant_config: Any) -> nn.Module:
    """Build ``Qwen3OmniMoeAudioEncoder``."""
    del quant_config  # the upstream audio encoder constructor does not take one
    thinker = getattr(hf_config, "thinker_config", hf_config)
    return Qwen3OmniMoeAudioEncoder(thinker.audio_config)


QWEN3_OMNI_VISUAL_SPEC = EncoderModuleSpec(
    name="visual",
    build_module=_build_qwen3_omni_visual,
    # Upstream Qwen3-Omni checkpoints place visual weights under either
    # ``model.visual.*`` or ``thinker.visual.*`` depending on the dump
    # source; some older shards keep them at the top-level ``visual.*``.
    checkpoint_prefixes=("model.visual.", "thinker.visual.", "visual."),
    checkpoint_rewrites=(
        ("model.visual.", "visual."),
        ("thinker.visual.", "visual."),
        # Upstream key names use ``attn.qkv``/``attn.out_proj``; the
        # SGLang submodule renames these to ``attn.qkv_proj``/``attn.proj``
        # after fusing QKV. Apply rewrites so the loader's
        # ``param.weight_loader`` sees the right destination param name.
        ("attn.qkv.", "attn.qkv_proj."),
        ("attn.out_proj.", "attn.proj."),
    ),
)

QWEN3_OMNI_AUDIO_SPEC = EncoderModuleSpec(
    name="audio_tower",
    build_module=_build_qwen3_omni_audio,
    checkpoint_prefixes=("audio_tower.", "thinker.audio_tower."),
    checkpoint_rewrites=(
        ("thinker.audio_tower.", "audio_tower."),
        # Upstream rename: checkpoint stores ``self_attn.out_proj`` but the
        # SGLang module exposes ``self_attn.proj``. Mirrors the rename in
        # ``Qwen3OmniMoeForConditionalGeneration.load_weights`` so the
        # output projection actually loads.
        (".self_attn.out_proj", ".self_attn.proj"),
    ),
    stacked_params_mapping=(
        # Audio tower's ``VisionAttention`` fuses Q/K/V into one
        # ``QKVParallelLinear`` named ``qkv_proj``. The checkpoint stores
        # ``q_proj``/``k_proj``/``v_proj`` separately; container's loader
        # rewrites the source name onto the fused param and dispatches
        # via ``weight_loader(param, weight, shard_id)``. Without this
        # we silently drop ~half the audio attention weights.
        (".self_attn.qkv_proj", ".self_attn.q_proj", "q"),
        (".self_attn.qkv_proj", ".self_attn.k_proj", "k"),
        (".self_attn.qkv_proj", ".self_attn.v_proj", "v"),
    ),
)


# ---------------------------------------------------------------------------
# BatchPlan / RequestSpan
# ---------------------------------------------------------------------------


@dataclasses.dataclass(slots=True)
class RequestSpan:
    """Per-request slice of one BatchPlan.

    Exactly one of ``skip_result`` or the active fields below is populated.
    """

    request_id: str
    skip_result: dict | None = None
    image_rows: int = 0
    video_rows: int = 0
    audio_rows: int = 0
    image_token_count: int = 0
    video_token_count: int = 0
    # Audio splitting: keep both the unpadded per-request feature lengths
    # (passed back to merge_for_thinker as audio_feature_lengths) and the
    # downsampled output lengths used to slice the encoder output along
    # the token axis.
    audio_feature_lengths: torch.Tensor | None = None
    audio_output_lengths: torch.Tensor | None = None


@dataclasses.dataclass(slots=True)
class BatchPlan:
    """Bundle passed from ``build_batch`` to ``encode_batch`` to ``slice_results``."""

    adapter: "EncoderAdapter"
    image_items: list[MultimodalDataItem]
    video_items: list[MultimodalDataItem]
    audio_items: list[MultimodalDataItem]
    spans: list[RequestSpan]

    @property
    def is_empty(self) -> bool:
        return not (self.image_items or self.video_items or self.audio_items)


class EncoderAdapter(Protocol):
    """Adapter contract.

    Implementers should keep ``build_batch`` and ``slice_results`` cheap
    (no GPU work) — they run on every rank.
    """

    stage_name: str

    def build_batch(self, messages: list[IncomingMessage]) -> BatchPlan: ...

    def run_feature(
        self, model: Any, plan: BatchPlan
    ) -> dict[str, torch.Tensor | None]: ...

    def slice_results(
        self,
        raw: dict[str, torch.Tensor | None],
        plan: BatchPlan,
        messages: list[IncomingMessage],
    ) -> list[StagePayload]: ...


# ---------------------------------------------------------------------------
# Shared helpers (moved from stages.py with one device-aware fix)
# ---------------------------------------------------------------------------


def _payload_with_state(payload: StagePayload, state: PipelineState) -> StagePayload:
    return StagePayload(
        request_id=payload.request_id,
        request=payload.request,
        data=state.to_dict(),
    )


def _lookup_cached_payload(
    msg: IncomingMessage,
    cache: Any,
    *,
    stage_name: str,
) -> StagePayload | None:
    payload = msg.data
    if not isinstance(payload, StagePayload):
        return None
    state = PipelineState.from_dict(payload.data)
    request = build_encoder_request(state, stage_name=stage_name)
    if request.skip_result is not None or request.cache_key is None:
        return None
    cached = cache.get(request.cache_key)
    if cached is None:
        return None
    apply_encoder_result(state, stage_name=stage_name, result=cached)
    return _payload_with_state(payload, state)


def _store_cached_payload(
    msg: IncomingMessage,
    output: StagePayload,
    cache: Any,
    *,
    stage_name: str,
) -> None:
    payload = msg.data
    if not isinstance(payload, StagePayload):
        return
    input_state = PipelineState.from_dict(payload.data)
    request = build_encoder_request(input_state, stage_name=stage_name)
    if request.skip_result is not None or request.cache_key is None:
        return
    output_state = PipelineState.from_dict(output.data)
    result = output_state.encoder_outs.get(stage_name)
    if result is None:
        return
    cache.put(request.cache_key, result)


def _tensor_bytes(value: Any) -> int:
    if not isinstance(value, torch.Tensor):
        return 0
    return int(value.numel() * value.element_size())


def _grid_visual_tokens(grid: Any, merge: int) -> int:
    if not isinstance(grid, torch.Tensor) or grid.numel() == 0:
        return 0
    return int((grid.to(dtype=torch.long).prod(dim=-1) // merge).sum().item())


def _split_visual_features(
    tensor: torch.Tensor | None,
    *,
    start: int,
    end: int,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor[start:end]


def _split_visual_multiscale(
    tensors: list[torch.Tensor] | None,
    *,
    start: int,
    end: int,
) -> list[torch.Tensor] | None:
    if tensors is None:
        return None
    return [tensor[start:end] for tensor in tensors]


def _normalize_audio_request_tensors(
    request: "EncoderRequestData",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(features [B, mel, time], mask [B, time] bool, lengths [B] long)``.

    The synthesized fallback mask uses ``arange`` on ``lengths.device``
    so that this helper is safe in both the local v1 path (CPU) and the
    SGLang Plan B path (cuda:0 after ``_strip_and_lift``). See RFC
    "Audio adapter / mandatory device-aware fix".
    """
    input_dict = request.model_inputs
    features = input_dict["input_features"]
    if features.ndim == 2:
        features = features.unsqueeze(0)

    lengths = input_dict.get("audio_feature_lengths")
    mask = input_dict.get("feature_attention_mask")
    if isinstance(lengths, torch.Tensor):
        lengths = lengths.to(dtype=torch.long).view(-1)
    else:
        raise ValueError("audio_feature_lengths is required for SGLang audio encoder")

    time_dim = features.shape[-1]
    if isinstance(mask, torch.Tensor):
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        mask = mask.to(dtype=torch.bool)
    else:
        steps = torch.arange(
            time_dim,
            dtype=torch.long,
            device=lengths.device,
        ).unsqueeze(0)
        mask = steps < lengths.unsqueeze(1)

    return features, mask, lengths


def _pad_audio_features(features: torch.Tensor, target_time: int) -> torch.Tensor:
    pad = target_time - int(features.shape[-1])
    if pad <= 0:
        return features
    return F.pad(features, (0, pad))


def _pad_audio_mask(mask: torch.Tensor, target_time: int) -> torch.Tensor:
    pad = target_time - int(mask.shape[-1])
    if pad <= 0:
        return mask
    return F.pad(mask, (0, pad), value=False)


# ---------------------------------------------------------------------------
# Image / video adapter
# ---------------------------------------------------------------------------


class Qwen3OmniImageEncoderAdapter:
    """Adapter for the visual stage (images and videos)."""

    stage_name = IMAGE_STAGE
    # Declares the upstream submodules the encoder runner must
    # partial-load. The factory passes this through to
    # ``SGLangEncoderRunner`` so only ``Qwen3OmniMoeVisionEncoder``
    # weights end up on the encoder GPU — the LLM, talker, and audio
    # tower are never instantiated.
    encoder_specs: tuple[EncoderModuleSpec, ...] = (QWEN3_OMNI_VISUAL_SPEC,)

    def __init__(
        self,
        *,
        hf_config: Any,
        dtype: torch.dtype,
        tp_size: int = 1,
    ) -> None:
        thinker_cfg = getattr(hf_config, "thinker_config", hf_config)
        vision_cfg = thinker_cfg.vision_config
        self._merge = int(vision_cfg.spatial_merge_size) ** 2
        self._base_hidden = int(vision_cfg.out_hidden_size)
        self._output_layers = 1 + len(vision_cfg.deepstack_visual_indexes)
        self._dtype_bytes = torch.empty((), dtype=dtype).element_size()
        self._tp_size = max(int(tp_size), 1)

    # -- request_cost_fn ------------------------------------------------

    def request_cost_fn(self, payload: StagePayload) -> int:
        """Cheap byte estimator used for batch-cost admission control.

        Mirrors the local-path arithmetic at ``stages.py:144-166``:
        ``(raw_input_bytes + token_count * base_hidden * dtype_bytes *
        output_layers) * activation_multiplier``. Reads metadata from
        the HF ``vision_config`` rather than the SGLang model wrapper to
        avoid the deepstack double-count trap (the wrapper already folds
        deepstack into ``out_hidden_size``; the local config does not).
        """
        state = PipelineState.from_dict(payload.data)
        request = build_encoder_request(state, stage_name=IMAGE_STAGE)
        if request.skip_result is not None:
            return 0
        inputs = request.model_inputs
        raw_bytes = _tensor_bytes(inputs.get("pixel_values"))
        raw_bytes += _tensor_bytes(inputs.get("pixel_values_videos"))
        visual_tokens = _grid_visual_tokens(inputs.get("image_grid_thw"), self._merge)
        visual_tokens += _grid_visual_tokens(inputs.get("video_grid_thw"), self._merge)
        output_bytes = (
            visual_tokens * self._base_hidden * self._dtype_bytes * self._output_layers
        )
        return _per_rank_activation_cost(
            replicated_bytes=raw_bytes,
            sharded_bytes=output_bytes,
            multiplier=QWEN3_IMAGE_ENCODER_ACTIVATION_MULTIPLIER,
            tp_size=self._tp_size,
        )

    def lookup_cached_output(
        self, msg: IncomingMessage, cache: Any
    ) -> StagePayload | None:
        return _lookup_cached_payload(msg, cache, stage_name=IMAGE_STAGE)

    def store_cached_output(
        self,
        msg: IncomingMessage,
        output: StagePayload,
        cache: Any,
    ) -> None:
        _store_cached_payload(msg, output, cache, stage_name=IMAGE_STAGE)

    # -- build / run / slice -------------------------------------------

    def build_batch(self, messages: list[IncomingMessage]) -> BatchPlan:
        images: list[MultimodalDataItem] = []
        videos: list[MultimodalDataItem] = []
        spans: list[RequestSpan] = []

        for msg in messages:
            payload = msg.data
            state = PipelineState.from_dict(payload.data)
            request = build_encoder_request(state, stage_name=IMAGE_STAGE)

            if request.skip_result is not None:
                spans.append(
                    RequestSpan(
                        request_id=msg.request_id,
                        skip_result=request.skip_result,
                    )
                )
                continue

            inputs = request.model_inputs
            n_img = n_vid = 0
            img_tokens = vid_tokens = 0
            if isinstance(inputs.get("pixel_values"), torch.Tensor):
                # _strip_and_lift moved every tensor in the payload tree to
                # cuda:0. For grid_thw that is wrong: upstream
                # ``compute_cu_seqlens_from_grid_numpy`` asserts a CPU tensor
                # (sglang/python/sglang/srt/models/utils.py:154). Move it
                # back to CPU so the upstream model never sees a CUDA grid.
                grid = inputs["image_grid_thw"].to(dtype=torch.long, device="cpu")
                it = MultimodalDataItem(
                    modality=Modality.IMAGE,
                    feature=inputs["pixel_values"],
                )
                it.image_grid_thw = grid
                images.append(it)
                n_img = int(grid.shape[0])
                img_tokens = int((grid.prod(-1) // self._merge).sum().item())
            if isinstance(inputs.get("pixel_values_videos"), torch.Tensor):
                grid = inputs["video_grid_thw"].to(dtype=torch.long, device="cpu")
                v = MultimodalDataItem(
                    modality=Modality.VIDEO,
                    feature=inputs["pixel_values_videos"],
                )
                v.video_grid_thw = grid
                videos.append(v)
                n_vid = int(grid.shape[0])
                vid_tokens = int((grid.prod(-1) // self._merge).sum().item())
            spans.append(
                RequestSpan(
                    request_id=msg.request_id,
                    image_rows=n_img,
                    video_rows=n_vid,
                    image_token_count=img_tokens,
                    video_token_count=vid_tokens,
                )
            )
        return BatchPlan(self, images, videos, [], spans)

    def run_feature(
        self,
        model: Any,
        plan: BatchPlan,
    ) -> dict[str, torch.Tensor | None]:
        if plan.is_empty:
            return {"image": None, "video": None, "audio": None}
        # ``model`` is the EncoderModuleContainer the runner built from
        # ``self.encoder_specs``; it exposes the visual submodule under the
        # name declared in QWEN3_OMNI_VISUAL_SPEC (``visual``). We mirror
        # the simple no-chunking path of upstream
        # ``Qwen3OmniMoeThinkerForConditionalGeneration.get_{image,video}_feature``
        # without going through ``model.thinker`` (which doesn't exist —
        # the container only holds the encoder submodules).
        visual = model.visual
        image_embed = (
            self._get_visual_feature(visual, plan.image_items, kind="image")
            if plan.image_items
            else None
        )
        video_embed = (
            self._get_visual_feature(visual, plan.video_items, kind="video")
            if plan.video_items
            else None
        )
        return {"image": image_embed, "video": video_embed, "audio": None}

    @staticmethod
    def _get_visual_feature(
        visual: nn.Module,
        items: list[MultimodalDataItem],
        *,
        kind: str,
    ) -> torch.Tensor:
        """Mirror of ``thinker.get_{image,video}_feature`` simple path.

        Concats per-item pixel/grid tensors and calls the loaded
        submodule directly. Upstream's chunking branch (gated on
        ``SGLANG_VLM_MAX_PATCHES_PER_VIT`` / ``SGLANG_VLM_MAX_IMAGES_PER_VIT``)
        is not mirrored here; that upstream chunking mode needs a separate
        parity lane before it is enabled.
        """
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(
            visual.dtype
        )
        grid_attr = "image_grid_thw" if kind == "image" else "video_grid_thw"
        grid = torch.cat([getattr(item, grid_attr) for item in items], dim=0)
        assert pixel_values.dim() == 2, pixel_values.dim()
        assert grid.dim() == 2, grid.dim()
        return visual(pixel_values, grid_thw=grid)

    def slice_results(
        self,
        raw: dict[str, torch.Tensor | None],
        plan: BatchPlan,
        messages: list[IncomingMessage],
    ) -> list[StagePayload]:
        out: list[StagePayload] = []
        img_row = img_tok = vid_row = vid_tok = 0
        image_feat = raw.get("image")
        video_feat = raw.get("video")

        for span, msg in zip(plan.spans, messages):
            payload = msg.data
            state = PipelineState.from_dict(payload.data)

            if span.skip_result is not None:
                apply_encoder_result(
                    state, stage_name=IMAGE_STAGE, result=span.skip_result
                )
                out.append(_payload_with_state(payload, state))
                continue

            result: dict[str, Any] = {}
            if span.image_rows > 0:
                if image_feat is None:
                    raise ValueError(
                        "image encoder produced no image embeddings for an active "
                        f"request {span.request_id!r}"
                    )
                tok_end = img_tok + span.image_token_count
                base, deepstack = _split_with_deepstack(
                    image_feat, img_tok, tok_end, self._base_hidden
                )
                result["image_embeds"] = base
                result["deepstack_visual_embeds_image"] = deepstack
                # image_grid_thw / token counts come from the original inputs
                request = build_encoder_request(state, stage_name=IMAGE_STAGE)
                grid = request.model_inputs.get("image_grid_thw")
                if isinstance(grid, torch.Tensor):
                    result["image_grid_thw"] = grid.to(dtype=torch.long)
                    result["image_token_counts"] = (
                        grid.to(dtype=torch.long).prod(-1) // self._merge
                    )
                img_row += span.image_rows
                img_tok = tok_end

            if span.video_rows > 0:
                if video_feat is None:
                    raise ValueError(
                        "image encoder produced no video embeddings for an active "
                        f"request {span.request_id!r}"
                    )
                tok_end = vid_tok + span.video_token_count
                base, deepstack = _split_with_deepstack(
                    video_feat, vid_tok, tok_end, self._base_hidden
                )
                result["video_embeds"] = base
                result["deepstack_visual_embeds_video"] = deepstack
                request = build_encoder_request(state, stage_name=IMAGE_STAGE)
                grid = request.model_inputs.get("video_grid_thw")
                if isinstance(grid, torch.Tensor):
                    result["video_grid_thw"] = grid.to(dtype=torch.long)
                    result["video_token_counts"] = (
                        grid.to(dtype=torch.long).prod(-1) // self._merge
                    )
                vid_row += span.video_rows
                vid_tok = tok_end

            apply_encoder_result(state, stage_name=IMAGE_STAGE, result=result)
            out.append(_payload_with_state(payload, state))
        return out


def _split_with_deepstack(
    feature: torch.Tensor,
    start: int,
    end: int,
    base_hidden: int,
) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    """Split the upstream visual return tensor into base + deepstack parts.

    Upstream visual return tensor has last-dim
    ``vision_config.out_hidden_size * (1 + len(deepstack_visual_indexes))``
    (confirmed via ``disaggregation/encode_server.py:_infer_embedding_dims``).
    """
    if end > int(feature.shape[0]):
        raise ValueError(
            "visual encoder output is shorter than requested slice: "
            f"end={end} rows={int(feature.shape[0])}"
        )
    if int(feature.shape[-1]) % int(base_hidden) != 0:
        raise ValueError(
            "visual encoder hidden dimension is not divisible by base hidden "
            f"size: hidden={int(feature.shape[-1])} base_hidden={int(base_hidden)}"
        )
    sliced = feature[start:end]
    parts = list(torch.split(sliced, base_hidden, dim=-1))
    base = parts[0]
    deepstack = parts[1:] if len(parts) > 1 else None
    return base, deepstack


# ---------------------------------------------------------------------------
# Audio adapter
# ---------------------------------------------------------------------------


class Qwen3OmniAudioEncoderAdapter:
    """Adapter for the audio stage."""

    stage_name = AUDIO_STAGE
    encoder_specs: tuple[EncoderModuleSpec, ...] = (QWEN3_OMNI_AUDIO_SPEC,)

    def __init__(
        self,
        *,
        hf_config: Any,
        dtype: torch.dtype,
        tp_size: int = 1,
    ) -> None:
        thinker_cfg = getattr(hf_config, "thinker_config", hf_config)
        audio_cfg = thinker_cfg.audio_config
        self._num_mel_bins = int(audio_cfg.num_mel_bins)
        self._output_dim = int(audio_cfg.output_dim)
        self._output_dtype_bytes = torch.empty((), dtype=dtype).element_size()
        self._tp_size = max(int(tp_size), 1)

    def request_cost_fn(self, payload: StagePayload) -> int:
        return self.batch_cost_fn([payload])

    def lookup_cached_output(
        self, msg: IncomingMessage, cache: Any
    ) -> StagePayload | None:
        return _lookup_cached_payload(msg, cache, stage_name=AUDIO_STAGE)

    def store_cached_output(
        self,
        msg: IncomingMessage,
        output: StagePayload,
        cache: Any,
    ) -> None:
        _store_cached_payload(msg, output, cache, stage_name=AUDIO_STAGE)

    def batch_cost_fn(self, payloads: list[StagePayload]) -> int:
        total_rows = 0
        max_time = 0
        output_tokens = 0
        max_input_dtype_bytes = 0
        max_mask_dtype_bytes = torch.empty((), dtype=torch.bool).element_size()

        for payload in payloads:
            state = PipelineState.from_dict(payload.data)
            request = build_encoder_request(state, stage_name=AUDIO_STAGE)
            if request.skip_result is not None:
                continue

            inputs = request.model_inputs
            features = inputs["input_features"]
            if features.ndim == 2:
                rows = 1
                time_dim = int(features.shape[-1])
            else:
                rows = int(features.shape[0])
                time_dim = int(features.shape[-1])
            lengths = inputs.get("audio_feature_lengths")
            if not isinstance(lengths, torch.Tensor):
                raise ValueError(
                    "audio_feature_lengths is required for SGLang audio admission"
                )
            lengths = lengths.to(dtype=torch.long).view(-1)
            if int(lengths.numel()) != rows:
                raise ValueError(
                    "audio_feature_lengths row count does not match input_features"
                )

            total_rows += rows
            max_time = max(max_time, time_dim)
            output_tokens += int(_get_feat_extract_output_lengths(lengths).sum().item())
            max_input_dtype_bytes = max(max_input_dtype_bytes, features.element_size())

            mask = inputs.get("feature_attention_mask")
            if isinstance(mask, torch.Tensor):
                max_mask_dtype_bytes = max(max_mask_dtype_bytes, mask.element_size())

        if total_rows == 0:
            return 0

        padded_input_bytes = (
            total_rows * self._num_mel_bins * max_time * max_input_dtype_bytes
        )
        padded_mask_bytes = total_rows * max_time * max_mask_dtype_bytes
        output_bytes = output_tokens * self._output_dim * self._output_dtype_bytes
        return _per_rank_activation_cost(
            replicated_bytes=padded_input_bytes + padded_mask_bytes,
            sharded_bytes=output_bytes,
            multiplier=QWEN3_AUDIO_ENCODER_ACTIVATION_MULTIPLIER,
            tp_size=self._tp_size,
        )

    def build_batch(self, messages: list[IncomingMessage]) -> BatchPlan:
        spans: list[RequestSpan] = []
        normalized: list[dict[str, torch.Tensor]] = []

        for msg in messages:
            payload = msg.data
            state = PipelineState.from_dict(payload.data)
            request = build_encoder_request(state, stage_name=AUDIO_STAGE)

            if request.skip_result is not None:
                spans.append(
                    RequestSpan(
                        request_id=msg.request_id,
                        skip_result=request.skip_result,
                    )
                )
                continue

            features, mask, lengths = _normalize_audio_request_tensors(request)
            out_lens = _get_feat_extract_output_lengths(lengths)
            spans.append(
                RequestSpan(
                    request_id=msg.request_id,
                    audio_rows=int(lengths.shape[0]),
                    audio_feature_lengths=lengths,
                    audio_output_lengths=out_lens,
                )
            )
            normalized.append({"features": features, "mask": mask, "lengths": lengths})

        if not normalized:
            return BatchPlan(self, [], [], [], spans)

        max_time = max(int(item["features"].shape[-1]) for item in normalized)
        audios: list[MultimodalDataItem] = []
        for item in normalized:
            feat = _pad_audio_features(item["features"], max_time)
            m = _pad_audio_mask(item["mask"], max_time)
            mm = MultimodalDataItem(
                modality=Modality.AUDIO,
                feature=feat,
            )
            mm.feature_attention_mask = m
            audios.append(mm)
        return BatchPlan(self, [], [], audios, spans)

    def run_feature(
        self,
        model: Any,
        plan: BatchPlan,
    ) -> dict[str, torch.Tensor | None]:
        if plan.is_empty:
            return {"image": None, "video": None, "audio": None}
        # ``model`` is the EncoderModuleContainer; ``audio_tower`` is the
        # name declared in QWEN3_OMNI_AUDIO_SPEC.
        embed = self._get_audio_feature(model.audio_tower, plan.audio_items)
        return {"image": None, "video": None, "audio": embed}

    @staticmethod
    def _get_audio_feature(
        audio_tower: nn.Module,
        items: list[MultimodalDataItem],
    ) -> torch.Tensor:
        """Mirror of ``thinker.get_audio_feature`` against the loaded submodule."""
        feature_attention_mask = torch.cat(
            [item.feature_attention_mask for item in items], dim=0
        ).type(torch.long)
        input_features = (
            torch.cat([item.feature for item in items])
            .type(audio_tower.dtype)
            .to(next(audio_tower.parameters()).device)
        )
        audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
        # Drop padded positions before the conv stack — same as upstream.
        input_features = input_features.permute(0, 2, 1)[
            feature_attention_mask.bool()
        ].permute(1, 0)
        outputs = audio_tower(input_features, feature_lens=audio_feature_lengths)
        hidden = outputs.last_hidden_state
        if hidden.dim() == 3:
            output_lengths = _get_feat_extract_output_lengths(audio_feature_lengths)
            pieces = [
                hidden[row, : int(length.item())]
                for row, length in enumerate(output_lengths)
            ]
            if pieces:
                return torch.cat(pieces, dim=0)
            return hidden.reshape(0, hidden.shape[-1])
        return hidden

    def slice_results(
        self,
        raw: dict[str, torch.Tensor | None],
        plan: BatchPlan,
        messages: list[IncomingMessage],
    ) -> list[StagePayload]:
        out: list[StagePayload] = []
        tok = 0
        audio_feat = raw.get("audio")

        for span, msg in zip(plan.spans, messages):
            payload = msg.data
            state = PipelineState.from_dict(payload.data)

            if span.skip_result is not None:
                apply_encoder_result(
                    state, stage_name=AUDIO_STAGE, result=span.skip_result
                )
                out.append(_payload_with_state(payload, state))
                continue

            if audio_feat is None:
                raise ValueError(
                    "audio encoder produced no embeddings for an active "
                    f"request {span.request_id!r}"
                )
            if span.audio_output_lengths is None:
                raise ValueError(
                    f"audio output lengths missing for active request {span.request_id!r}"
                )
            if span.audio_feature_lengths is None:
                raise ValueError(
                    f"audio feature lengths missing for active request {span.request_id!r}"
                )
            tok_end = tok + int(span.audio_output_lengths.sum().item())
            if tok_end > int(audio_feat.shape[0]):
                raise ValueError(
                    "audio encoder output is shorter than requested slice: "
                    f"end={tok_end} rows={int(audio_feat.shape[0])}"
                )
            result = {
                "audio_embeds": audio_feat[tok:tok_end],
                # Unpadded lengths preserved at build_batch time.
                "audio_feature_lengths": span.audio_feature_lengths,
                "audio_output_lengths": span.audio_output_lengths,
            }
            apply_encoder_result(state, stage_name=AUDIO_STAGE, result=result)
            out.append(_payload_with_state(payload, state))
            tok = tok_end
        return out
