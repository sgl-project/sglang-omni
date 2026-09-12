# SPDX-License-Identifier: Apache-2.0
"""Eager Torch MPS thinker and talker with strict, stage-local HF loading."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors import safe_open

from sglang_omni.models.qwen3_omni.components.common import load_thinker_config
from sglang_omni.models.qwen3_omni.components.thinker import (
    TEXT_MODEL_CLASS,
    Qwen3OmniSplitThinker,
    _build_thinker_shell,
)
from sglang_omni.models.weight_loader import resolve_dtype
from sglang_omni.utils import instantiate_module

logger = logging.getLogger(__name__)

__all__ = [
    "Qwen3OmniTorchMpsTalker",
    "Qwen3OmniTorchMpsThinker",
    "TorchMpsTalkerPrefillShim",
    "TorchMpsTalkerStep",
    "TorchMpsThinkerOutput",
    "build_deepstack_visual_inputs",
    "build_suppress_mask",
    "load_torch_mps_talker",
    "load_torch_mps_thinker",
    "mask_suppressed_logits",
    "merge_multimodal_rows",
    "read_talker_state_dict",
    "read_thinker_text_state_dict",
    "restore_placeholder_token_ids",
]

OFFICIAL_TEXT_PREFIX = "thinker.model."
OFFICIAL_LM_HEAD_PREFIX = "thinker.lm_head."
LOCAL_TEXT_PREFIX = "model."
LOCAL_LM_HEAD_PREFIX = "lm_head."
COMPONENT_DIRECTORIES = ("thinker", "talker", "code2wav")
_OTHER_COMPONENT_PREFIXES = ("talker.", "code2wav.")

_EXPERT_KEY = re.compile(
    r"^(?P<prefix>(?:[\w.]+\.)?layers\.\d+\.mlp\.experts)\.(?P<expert>\d+)\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\.weight$"
)
_MODALITIES = ("image", "video", "audio")
_CPU = torch.device("cpu")


class UnsupportedCheckpointLayout(ValueError):
    """The checkpoint layout cannot be loaded by the eager Torch MPS stages."""


# Historical name kept for the thinker's own call sites and error handling.
UnsupportedThinkerCheckpointLayout = UnsupportedCheckpointLayout


@dataclass
class TorchMpsThinkerOutput:
    """One thinker step; logits contain only the sampled final row [1, 1, vocab]."""

    logits: torch.Tensor
    past_key_values: Any = None
    hidden_states: tuple[torch.Tensor, ...] | None = None


# ---------------------------------------------------------------------------
# Checkpoint reading
# ---------------------------------------------------------------------------


def _resolve_checkpoint_directory(model_path: str | Path) -> Path:
    path = Path(model_path).expanduser()
    if path.is_dir():
        return path.resolve()

    from huggingface_hub import snapshot_download

    return Path(snapshot_download(str(model_path)))


def _thinker_text_shards(directory: Path) -> list[Path]:
    """Root-level safetensors shards for the thinker text stack."""

    return _published_shards(directory, stage="thinker")


def _published_shards(directory: Path, *, stage: str) -> list[Path]:
    """Read root shards; reject nested exports with ambiguous component ownership."""

    shards = sorted(directory.glob("*.safetensors"))
    nested = sorted(
        path
        for path in directory.rglob("*.safetensors")
        if path.parent != directory
        and path.parent.name in COMPONENT_DIRECTORIES
        and path.parent.parent == directory
    )
    if nested:
        raise UnsupportedCheckpointLayout(
            f"Qwen3-Omni checkpoint {directory} uses an unsupported component-local "
            f"layout for the eager Torch MPS {stage}: component subdirectories "
            f"{sorted({path.parent.name for path in nested})} carry prefix-stripped "
            "keys that the thinker and the talker share. Use the published "
            "single-namespace checkpoint, or select the MLX backend."
        )
    if not shards:
        raise UnsupportedCheckpointLayout(
            f"Qwen3-Omni checkpoint {directory} exposes no safetensors shard for "
            f"the eager Torch MPS {stage}"
        )
    return shards


def _select_thinker_text_key(key: str, *, official: bool) -> tuple[str, str] | None:
    """Map a checkpoint key onto ``(component, local_key)``, or drop it."""

    if official:
        if key.startswith(OFFICIAL_TEXT_PREFIX):
            return "model", key[len(OFFICIAL_TEXT_PREFIX) :]
        if key.startswith(OFFICIAL_LM_HEAD_PREFIX):
            return "lm_head", key[len(OFFICIAL_LM_HEAD_PREFIX) :]
        return None
    if key.startswith(_OTHER_COMPONENT_PREFIXES) or key.startswith("thinker."):
        return None
    if key.startswith(LOCAL_TEXT_PREFIX):
        return "model", key[len(LOCAL_TEXT_PREFIX) :]
    if key.startswith(LOCAL_LM_HEAD_PREFIX):
        return "lm_head", key[len(LOCAL_LM_HEAD_PREFIX) :]
    return None


def _shard_keys(shard: Path) -> list[str]:
    with safe_open(str(shard), framework="pt", device="cpu") as handle:
        return list(handle.keys())


def read_thinker_text_state_dict(
    model_path: str | Path,
    *,
    dtype: torch.dtype | str | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    """Read only model/lm_head weights, stripping stage prefixes and fusing experts."""

    directory = _resolve_checkpoint_directory(model_path)
    shards = _thinker_text_shards(directory)
    torch_dtype = resolve_dtype(dtype)

    shard_keys = {shard: _shard_keys(shard) for shard in shards}
    official = any(
        key.startswith((OFFICIAL_TEXT_PREFIX, OFFICIAL_LM_HEAD_PREFIX))
        for keys in shard_keys.values()
        for key in keys
    )

    collected: dict[str, dict[str, torch.Tensor]] = {"model": {}, "lm_head": {}}
    sources: dict[tuple[str, str], Path] = {}
    for shard, keys in shard_keys.items():
        wanted = {}
        for key in keys:
            selected = _select_thinker_text_key(key, official=official)
            if selected is None:
                continue
            component, local_key = selected
            previous = sources.get((component, local_key))
            if previous is not None:
                raise ValueError(
                    "Qwen3-Omni thinker text checkpoint carries duplicate weight "
                    f"{key!r}: it appears in both {previous.name} and {shard.name}"
                )
            sources[(component, local_key)] = shard
            wanted[key] = (component, local_key)
        if not wanted:
            continue
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for key, (component, local_key) in wanted.items():
                tensor = handle.get_tensor(key)
                if torch_dtype is not None and tensor.is_floating_point():
                    tensor = tensor.to(torch_dtype)
                collected[component][local_key] = tensor

    if not collected["model"]:
        raise UnsupportedThinkerCheckpointLayout(
            f"Qwen3-Omni checkpoint {directory} carries no thinker text weights "
            f"under {OFFICIAL_TEXT_PREFIX!r} or {LOCAL_TEXT_PREFIX!r}"
        )
    collected["model"] = fuse_moe_expert_weights(collected["model"])
    return collected


def fuse_moe_expert_weights(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Stack experts on axis 0 and concatenate gate/up rows on axis 1, as HF does."""

    grouped: dict[tuple[str, str], dict[int, torch.Tensor]] = {}
    fused: dict[str, torch.Tensor] = {}
    for key in list(state_dict):
        match = _EXPERT_KEY.match(key)
        if match is None:
            fused[key] = state_dict[key]
            continue
        grouped.setdefault((match.group("prefix"), match.group("projection")), {})[
            int(match.group("expert"))
        ] = state_dict.pop(key)

    stacked: dict[tuple[str, str], torch.Tensor] = {}
    for (prefix, projection), experts in grouped.items():
        indices = sorted(experts)
        if indices != list(range(len(indices))):
            raise ValueError(
                f"Qwen3-Omni thinker experts for {prefix}.{projection} are not a "
                f"contiguous range: {indices[:8]}"
            )
        stacked[(prefix, projection)] = torch.stack(
            [experts[index] for index in indices], dim=0
        )
        experts.clear()

    prefixes = {prefix for prefix, _ in stacked}
    for prefix in sorted(prefixes):
        down = stacked.pop((prefix, "down_proj"), None)
        if down is not None:
            fused[f"{prefix}.down_proj"] = down
        gate = stacked.pop((prefix, "gate_proj"), None)
        up = stacked.pop((prefix, "up_proj"), None)
        if gate is None and up is None:
            continue
        if gate is None or up is None:
            raise ValueError(
                f"Qwen3-Omni thinker experts for {prefix} carry "
                f"{'up_proj' if gate is None else 'gate_proj'} without its pair"
            )
        # Concatenate and drop the sources immediately: for the published 30B
        # checkpoint the expert stacks dominate the load's peak memory.
        fused[f"{prefix}.gate_up_proj"] = torch.cat([gate, up], dim=1)
        del gate, up
    return fused


def _assign_state_dict(
    module: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    *,
    component: str,
) -> None:
    """Assign safetensors directly, failing loudly on any discrepancy."""

    expected = dict(module.state_dict())
    missing = sorted(set(expected) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(expected))
    mismatched = [
        f"{key}: checkpoint {tuple(tensor.shape)} != module "
        f"{tuple(expected[key].shape)}"
        for key, tensor in state_dict.items()
        if key in expected and tuple(tensor.shape) != tuple(expected[key].shape)
    ]
    problems = []
    if missing:
        problems.append(f"missing {missing[:8]}")
    if unexpected:
        problems.append(f"unexpected {unexpected[:8]}")
    if mismatched:
        problems.append(f"shape-mismatched {mismatched[:8]}")
    if problems:
        raise ValueError(
            f"Qwen3-Omni Torch MPS {component} weights do not match the model: "
            + "; ".join(problems)
        )
    module.load_state_dict(dict(state_dict), strict=True, assign=True)


class Qwen3OmniTorchMpsThinker(Qwen3OmniSplitThinker):
    """Eager split thinker; encoder towers stay on meta and run in separate stages."""

    def __init__(
        self,
        model_path: str,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        nn.Module.__init__(self)
        self._device = torch.device(device)
        torch_dtype = resolve_dtype(dtype)
        thinker_config = load_thinker_config(model_path)
        text_config = thinker_config.text_config

        from sglang_omni.models.qwen3_omni.apple_runtime import (
            get_qwen3_omni_mps_quantization,
        )

        bits = get_qwen3_omni_mps_quantization(model_path)
        started = time.perf_counter()
        if bits is not None:
            from accelerate import init_empty_weights

            from sglang_omni.models.qwen3_omni.torch_mps_checkpoint import (
                load_quantized_mps_module,
            )

            with init_empty_weights():
                component = nn.Module()
                component.model = instantiate_module(TEXT_MODEL_CLASS, text_config)
                if not text_config.tie_word_embeddings:
                    component.lm_head = nn.Linear(
                        text_config.hidden_size, text_config.vocab_size, bias=False
                    )
            component = load_quantized_mps_module(
                component,
                model_path,
                prefix="thinker.",
                bits=bits,
                dtype=torch_dtype,
                device=self._device,
            )
            text_model = component.model
            if text_config.tie_word_embeddings:
                with init_empty_weights():
                    lm_head = nn.Linear(
                        text_config.hidden_size, text_config.vocab_size, bias=False
                    )
                lm_head.weight = text_model.embed_tokens.weight
            else:
                lm_head = component.lm_head
        else:
            state = read_thinker_text_state_dict(model_path, dtype=torch_dtype)
            text_model = instantiate_module(TEXT_MODEL_CLASS, text_config)
            _assign_state_dict(
                text_model, state["model"], component="thinker text model"
            )
            lm_head = nn.Linear(
                text_config.hidden_size, text_config.vocab_size, bias=False
            )
            if text_config.tie_word_embeddings:
                if state["lm_head"]:
                    raise ValueError(
                        "Qwen3-Omni Torch MPS thinker ties its LM head to the token "
                        "embeddings, but the checkpoint also carries "
                        f"{OFFICIAL_LM_HEAD_PREFIX!r} weights"
                    )
                lm_head.weight = text_model.embed_tokens.weight
            else:
                _assign_state_dict(
                    lm_head, state["lm_head"], component="thinker lm_head"
                )
            state.clear()

        self.thinker = _build_thinker_shell(thinker_config)
        # Only the text stack becomes resident; the towers stay on meta.
        if bits is None:
            text_model = text_model.to(device=self._device, dtype=torch_dtype)
            lm_head = lm_head.to(device=self._device, dtype=torch_dtype)
        self.thinker.model = text_model.eval()
        self.thinker.lm_head = lm_head.eval()
        self.eval()

        logger.info(
            "Loaded eager Torch MPS Qwen3-Omni thinker in %.2fs (%d layers, %s, %s)",
            time.perf_counter() - started,
            text_config.num_hidden_layers,
            self._device,
            torch_dtype,
        )

    # -- forward -----------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self.thinker.model.embed_tokens.weight.dtype

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.thinker.model.embed_tokens(input_ids.to(self._device))

    def forward(  # type: ignore[override]
        self,
        input_ids: torch.Tensor | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Any = None,
        use_cache: bool = True,
        output_hidden_states: bool = False,
        deepstack_visual_embeds: Sequence[torch.Tensor] | None = None,
        visual_pos_masks: torch.Tensor | None = None,
    ) -> TorchMpsThinkerOutput:
        """Bypass the HF wrapper, which would overwrite stage-supplied DeepStack rows."""

        text_model = self.thinker.model
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError(
                "Qwen3-Omni Torch MPS thinker needs exactly one of input_ids or "
                "inputs_embeds"
            )
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        inputs_embeds = inputs_embeds.to(device=self._device, dtype=self.dtype)

        deepstack = None
        if deepstack_visual_embeds is not None:
            if visual_pos_masks is None:
                raise ValueError(
                    "Qwen3-Omni Torch MPS thinker requires visual_pos_masks "
                    "alongside deepstack_visual_embeds"
                )
            deepstack = [
                layer.to(device=self._device, dtype=self.dtype)
                for layer in deepstack_visual_embeds
            ]
            visual_pos_masks = visual_pos_masks.to(self._device)

        outputs = text_model(
            inputs_embeds=inputs_embeds,
            position_ids=(
                None if position_ids is None else position_ids.to(self._device)
            ),
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            deepstack_visual_embeds=deepstack,
            visual_pos_masks=visual_pos_masks,
        )
        logits = self.thinker.lm_head(outputs.last_hidden_state[:, -1:, :])
        return TorchMpsThinkerOutput(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )


def load_torch_mps_thinker(
    model_path: str,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> nn.Module:
    """Load the thinker text stack for eager Torch execution on ``device``."""

    return Qwen3OmniTorchMpsThinker(model_path, dtype=dtype, device=device)


# ---------------------------------------------------------------------------
# Stage-side input assembly (pure helpers)
# ---------------------------------------------------------------------------


def restore_placeholder_token_ids(
    token_ids: Sequence[int],
    *,
    positions: Mapping[str, Any] | None,
    pad_values: Mapping[str, int] | None,
    placeholder_token_ids: Mapping[str, int],
    vocab_size: int,
) -> list[int]:
    """Restore hashed media IDs using absolute positions or the pad_values map."""

    restored = [int(token_id) for token_id in token_ids]
    if positions:
        for modality, modality_positions in positions.items():
            token_id = placeholder_token_ids.get(modality)
            if token_id is None or modality_positions is None:
                continue
            for position in torch.as_tensor(modality_positions).reshape(-1).tolist():
                index = int(position)
                if 0 <= index < len(restored):
                    restored[index] = int(token_id)
    elif pad_values:
        inverse = {
            int(pad): int(placeholder_token_ids[modality])
            for modality, pad in pad_values.items()
            if modality in placeholder_token_ids
        }
        if inverse:
            restored = [inverse.get(token, token) for token in restored]

    out_of_range = [token for token in restored if not 0 <= token < vocab_size]
    if out_of_range:
        raise ValueError(
            "Qwen3-Omni Torch MPS prefill still holds non-embeddable token ids "
            f"{out_of_range[:4]} after placeholder restoration; the request's "
            "modality positions or pad_values are incomplete"
        )
    return restored


def merge_multimodal_rows(
    inputs_embeds: torch.Tensor,
    *,
    modality_rows: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """Scatter each modality's (absolute positions, encoder rows) into embeddings."""

    if inputs_embeds.ndim != 3 or inputs_embeds.shape[0] != 1:
        raise ValueError(
            "Qwen3-Omni Torch MPS thinker merges one request at a time; expected "
            f"[1, sequence, hidden] embeddings, got {tuple(inputs_embeds.shape)}"
        )
    merged = inputs_embeds
    for modality in _MODALITIES:
        entry = modality_rows.get(modality)
        if entry is None:
            continue
        positions, rows = entry
        positions = torch.as_tensor(positions, dtype=torch.long).reshape(-1)
        if rows is None or positions.numel() == 0:
            if rows is not None and rows.shape[0]:
                raise ValueError(
                    f"Qwen3-Omni Torch MPS prefill has {rows.shape[0]} {modality} "
                    "rows but no placeholder positions"
                )
            continue
        if int(rows.shape[0]) != int(positions.numel()):
            raise ValueError(
                f"Qwen3-Omni Torch MPS {modality} placeholder count mismatch: "
                f"positions={int(positions.numel())} rows={int(rows.shape[0])}"
            )
        merged = merged.index_copy(
            1,
            positions.to(merged.device),
            rows.to(device=merged.device, dtype=merged.dtype).unsqueeze(0),
        )
    return merged


def build_deepstack_visual_inputs(
    *,
    sequence_length: int,
    image_positions: torch.Tensor | None,
    video_positions: torch.Tensor | None,
    image_layers: Sequence[torch.Tensor] | None,
    video_layers: Sequence[torch.Tensor] | None,
    merged_layers: Sequence[torch.Tensor] | None,
) -> tuple[list[torch.Tensor] | None, torch.Tensor | None]:
    """Interleave image/video DeepStack rows in prompt order and build their mask."""

    image_positions = _as_positions(image_positions)
    video_positions = _as_positions(video_positions)
    visual_positions = torch.cat([image_positions, video_positions])
    if merged_layers is None and not image_layers and not video_layers:
        return None, None
    if visual_positions.numel() == 0:
        return None, None

    order = torch.argsort(visual_positions)
    ordered_positions = visual_positions[order]

    if merged_layers is not None:
        layers = [layer for layer in merged_layers]
    elif image_layers and video_layers:
        slots = torch.empty_like(order)
        slots[order] = torch.arange(order.numel(), dtype=order.dtype)
        image_slots = slots[: image_positions.numel()]
        video_slots = slots[image_positions.numel() :]
        layers = []
        for image_layer, video_layer in zip(image_layers, video_layers):
            joint = image_layer.new_zeros(
                (order.numel(), image_layer.shape[-1]), dtype=image_layer.dtype
            )
            joint[image_slots] = image_layer
            joint[video_slots] = video_layer
            layers.append(joint)
    elif image_layers:
        layers = [layer for layer in image_layers]
    else:
        layers = [layer for layer in video_layers or ()]

    for index, layer in enumerate(layers):
        if int(layer.shape[0]) != int(ordered_positions.numel()):
            raise ValueError(
                f"Qwen3-Omni Torch MPS DeepStack layer {index} carries "
                f"{int(layer.shape[0])} rows for "
                f"{int(ordered_positions.numel())} visual placeholders"
            )

    mask = torch.zeros(1, sequence_length, dtype=torch.bool)
    mask[0, ordered_positions] = True
    return layers, mask


def _as_positions(positions: torch.Tensor | None) -> torch.Tensor:
    if positions is None:
        return torch.zeros(0, dtype=torch.long)
    return torch.as_tensor(positions, dtype=torch.long).reshape(-1)


# ---------------------------------------------------------------------------
# Talker: strict split loading
# ---------------------------------------------------------------------------

OFFICIAL_TALKER_PREFIX = "talker."
TALKER_MODEL_CLASS = (
    "transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe."
    "Qwen3OmniMoeTalkerForConditionalGeneration"
)


def load_talker_config(model_path: str) -> Any:
    """The published ``talker_config`` node of the Omni config."""

    from sglang_omni.utils import load_hf_config

    return load_hf_config(
        model_path, trust_remote_code=True, local_files_only=True
    ).talker_config


def read_talker_state_dict(
    model_path: str | Path,
    *,
    dtype: torch.dtype | str | None = None,
) -> dict[str, torch.Tensor]:
    """Read only talker.* weights, strip the prefix and fuse per-expert linears."""

    directory = _resolve_checkpoint_directory(model_path)
    shards = _published_shards(directory, stage="talker")
    torch_dtype = resolve_dtype(dtype)

    collected: dict[str, torch.Tensor] = {}
    sources: dict[str, Path] = {}
    for shard in shards:
        wanted: dict[str, str] = {}
        for key in _shard_keys(shard):
            if not key.startswith(OFFICIAL_TALKER_PREFIX):
                continue
            local_key = key[len(OFFICIAL_TALKER_PREFIX) :]
            previous = sources.get(local_key)
            if previous is not None:
                raise ValueError(
                    "Qwen3-Omni talker checkpoint carries duplicate weight "
                    f"{key!r}: it appears in both {previous.name} and {shard.name}"
                )
            sources[local_key] = shard
            wanted[key] = local_key
        if not wanted:
            continue
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for key, local_key in wanted.items():
                tensor = handle.get_tensor(key)
                if torch_dtype is not None and tensor.is_floating_point():
                    tensor = tensor.to(torch_dtype)
                collected[local_key] = tensor

    if not collected:
        raise UnsupportedCheckpointLayout(
            f"Qwen3-Omni checkpoint {directory} carries no talker weights under "
            f"{OFFICIAL_TALKER_PREFIX!r}"
        )
    return fuse_moe_expert_weights(collected)


def _build_talker_shell(talker_config: Any) -> nn.Module:
    """Build meta talker parameters, leaving non-checkpoint rotary buffers real."""

    from importlib import import_module

    from accelerate import init_empty_weights

    module_path, _, class_name = TALKER_MODEL_CLASS.rpartition(".")
    factory = getattr(import_module(module_path), class_name)
    with init_empty_weights():
        return factory._from_config(talker_config)


# ---------------------------------------------------------------------------
# Talker: codec suppression
# ---------------------------------------------------------------------------


def build_suppress_mask(
    vocab_size: int,
    suppress_tokens: Sequence[int],
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build an additive vocabulary mask: -inf for suppressed IDs, zero elsewhere."""

    unique = sorted({int(token_id) for token_id in suppress_tokens})
    invalid = [token_id for token_id in unique if not 0 <= token_id < vocab_size]
    if invalid:
        raise ValueError(f"suppress token ids {invalid} are outside [0, {vocab_size})")
    if len(unique) >= vocab_size:
        raise ValueError("refusing to suppress every token of the codec vocabulary")
    mask = torch.zeros(vocab_size, dtype=dtype, device=device)
    if unique:
        mask[torch.tensor(unique, dtype=torch.long, device=device)] = float("-inf")
    return mask


def mask_suppressed_logits(
    logits: torch.Tensor,
    suppress: Sequence[int] | torch.Tensor | None,
) -> torch.Tensor:
    """Suppress codec logits using token IDs or a prebuilt additive mask."""

    if suppress is None:
        return logits
    vocab_size = int(logits.shape[-1])
    if isinstance(suppress, torch.Tensor):
        if tuple(suppress.shape) != (vocab_size,):
            raise ValueError(
                f"suppress mask shape {tuple(suppress.shape)} does not match the "
                f"codec vocabulary ({vocab_size},)"
            )
        mask = suppress
    else:
        if len(suppress) == 0:
            return logits
        mask = build_suppress_mask(
            vocab_size, suppress, device=logits.device, dtype=logits.dtype
        )
    return logits + mask.to(device=logits.device, dtype=logits.dtype)


# ---------------------------------------------------------------------------
# Talker: one step
# ---------------------------------------------------------------------------


@dataclass
class TorchMpsTalkerStep:
    """Ordered codec groups, summed next-step feedback, and the outer talker cache."""

    layer0_token: torch.Tensor
    hidden: torch.Tensor
    codes: torch.Tensor
    feedback: torch.Tensor
    past_key_values: Any


class Qwen3OmniTorchMpsTalker(nn.Module):
    """Eager split talker; call the inner HF model to expose the predictor's hidden row."""

    def __init__(
        self,
        model_path: str,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self._device = torch.device(device)
        torch_dtype = resolve_dtype(dtype) or torch.float32
        talker_config = load_talker_config(model_path)

        started = time.perf_counter()
        talker = _build_talker_shell(talker_config)
        state = read_talker_state_dict(model_path, dtype=torch_dtype)
        _assign_state_dict(talker, state, component="talker")
        state.clear()

        talker = talker.to(device=self._device, dtype=torch_dtype)
        self.talker = talker.eval()
        self.config = talker_config
        self._num_code_groups = int(talker_config.num_code_groups)
        self._codec_vocab_size = int(talker_config.text_config.vocab_size)
        self._hidden_size = int(talker_config.text_config.hidden_size)
        self.eval()
        self.requires_grad_(False)

        logger.info(
            "Loaded eager Torch MPS Qwen3-Omni talker in %.2fs (%d layers, "
            "%d code groups, %s, %s)",
            time.perf_counter() - started,
            int(talker_config.text_config.num_hidden_layers),
            self._num_code_groups,
            self._device,
            torch_dtype,
        )

    # -- accessors ---------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self.talker.model.codec_embedding.weight.dtype

    @property
    def num_code_groups(self) -> int:
        return self._num_code_groups

    @property
    def codec_vocab_size(self) -> int:
        return self._codec_vocab_size

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    # -- forward -----------------------------------------------------------

    def _as_step_rows(self, rows: torch.Tensor, *, name: str) -> torch.Tensor:
        """Normalise talker-space rows onto ``[1, sequence, hidden]`` on device."""

        if rows.ndim == 1:
            rows = rows.reshape(1, 1, -1)
        elif rows.ndim == 2:
            rows = rows.unsqueeze(0)
        if rows.ndim != 3 or rows.shape[0] != 1:
            raise ValueError(
                "the Torch MPS talker serves one request at a time; expected "
                f"[1, sequence, hidden] {name}, got {tuple(rows.shape)}"
            )
        if int(rows.shape[-1]) != self._hidden_size:
            raise ValueError(
                f"{name} must carry the talker hidden size {self._hidden_size}, "
                f"got {int(rows.shape[-1])}"
            )
        return rows.to(device=self._device, dtype=self.dtype)

    def _as_positions(self, positions: torch.Tensor, *, length: int) -> torch.Tensor:
        rows = torch.as_tensor(positions)
        if rows.ndim == 2:
            rows = rows.unsqueeze(1)
        if rows.ndim != 3 or int(rows.shape[0]) != 3 or int(rows.shape[1]) != 1:
            raise ValueError(
                "Qwen3-Omni talker M-RoPE positions must be [3, 1, sequence], got "
                f"{tuple(rows.shape)}"
            )
        if int(rows.shape[2]) != length:
            raise ValueError(
                f"Qwen3-Omni talker M-RoPE positions cover {int(rows.shape[2])} "
                f"tokens but the step holds {length}"
            )
        return rows.to(device=self._device, dtype=torch.long)

    def step(
        self,
        rows: torch.Tensor,
        *,
        mrope_positions: torch.Tensor,
        past_key_values: Any = None,
        suppress_tokens: Sequence[int] | torch.Tensor | None = None,
    ) -> TorchMpsTalkerStep:
        """Expand a code frame from projected prompt rows or one feedback+text row."""

        embeds = self._as_step_rows(rows, name="talker rows")
        positions = self._as_positions(mrope_positions, length=int(embeds.shape[1]))
        outputs = self.talker.model(
            inputs_embeds=embeds,
            position_ids=positions,
            past_key_values=past_key_values,
            use_cache=True,
        )
        hidden = outputs.last_hidden_state[:, -1:, :]
        logits = mask_suppressed_logits(self.talker.codec_head(hidden), suppress_tokens)
        layer0 = logits[:, -1, :].argmax(dim=-1)
        codes, feedback = self.predict_codes(layer0=layer0, talker_hidden=hidden)
        return TorchMpsTalkerStep(
            layer0_token=layer0,
            hidden=hidden,
            codes=codes,
            feedback=feedback,
            past_key_values=outputs.past_key_values,
        )

    def predict_codes(
        self,
        *,
        layer0: torch.Tensor,
        talker_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Expand ordered codec groups with a fresh cache; sum embeddings as feedback."""

        from transformers.cache_utils import DynamicCache

        predictor = self.talker.code_predictor
        if layer0.reshape(-1).numel() != 1:
            raise ValueError(
                f"layer0 must hold one codec token, got shape {tuple(layer0.shape)}"
            )
        layer0 = layer0.reshape(1, 1).to(device=self._device, dtype=torch.long)
        hidden = self._as_step_rows(talker_hidden, name="talker hidden")
        if int(hidden.shape[1]) != 1:
            raise ValueError(
                "the code predictor consumes exactly one talker hidden row, got "
                f"{int(hidden.shape[1])}"
            )

        layer0_embed = self.talker.model.codec_embedding(layer0)
        cache = DynamicCache(config=predictor.config)
        output = predictor(
            inputs_embeds=torch.cat([hidden, layer0_embed.to(hidden.dtype)], dim=1),
            past_key_values=cache,
            use_cache=True,
        )

        tables = predictor.model.get_input_embeddings()
        codes = [layer0]
        feedback = layer0_embed[:, 0, :]
        for group in range(self._num_code_groups - 1):
            code = output.logits[:, -1, :].argmax(dim=-1).reshape(1, 1)
            codes.append(code)
            embed = tables[group](code)
            feedback = feedback + embed[:, 0, :]
            if group < self._num_code_groups - 2:
                output = predictor(
                    input_ids=code,
                    past_key_values=cache,
                    use_cache=True,
                    generation_steps=group + 1,
                )
        return torch.cat(codes, dim=1), feedback

    def forward(self, *args: Any, **kwargs: Any):  # pragma: no cover - explicit
        raise RuntimeError(
            "the Torch MPS talker is driven through step()/predict_codes(); the "
            "conditional-generation wrapper's forward is bypassed on purpose"
        )


def load_torch_mps_talker(
    model_path: str,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> nn.Module:
    """Load the talker stack for eager Torch execution on ``device``."""

    return Qwen3OmniTorchMpsTalker(model_path, dtype=dtype, device=device)


# ---------------------------------------------------------------------------
# Talker: CPU float32 prompt-building surface
# ---------------------------------------------------------------------------


class _TalkerPrefillTextModel(nn.Module):
    """Holds the codec embedding table under its production attribute name."""

    def __init__(self, codec_embedding: nn.Embedding) -> None:
        super().__init__()
        self.codec_embedding = codec_embedding


class TorchMpsTalkerPrefillShim(nn.Module):
    """Copy the loaded talker's prompt surface to CPU float32 for host queues."""

    def __init__(
        self,
        *,
        codec_embedding_weight: torch.Tensor,
        text_projection: nn.Module,
        hidden_projection: nn.Module,
        config: Any,
    ) -> None:
        super().__init__()
        weight = codec_embedding_weight.detach().to(device=_CPU, dtype=torch.float32)
        embedding = nn.Embedding(int(weight.shape[0]), int(weight.shape[1]))
        with torch.no_grad():
            embedding.weight.copy_(weight)
        self.model = _TalkerPrefillTextModel(embedding)
        self.text_projection = text_projection
        self.hidden_projection = hidden_projection
        self.activation_dtype = torch.float32
        self.config = config
        self.eval()
        self.requires_grad_(False)

    @classmethod
    def from_talker(
        cls, talker: Qwen3OmniTorchMpsTalker
    ) -> "TorchMpsTalkerPrefillShim":
        """Snapshot a *loaded* talker's prompt surface into CPU float32 Torch."""

        import copy

        inner = talker.talker
        return cls(
            codec_embedding_weight=inner.model.codec_embedding.weight,
            text_projection=copy.deepcopy(inner.text_projection).to(
                device=_CPU, dtype=torch.float32
            ),
            hidden_projection=copy.deepcopy(inner.hidden_projection).to(
                device=_CPU, dtype=torch.float32
            ),
            config=talker.config,
        )

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.codec_embedding

    def forward(self, *args: Any, **kwargs: Any):  # pragma: no cover - never run
        raise RuntimeError(
            "TorchMpsTalkerPrefillShim only mirrors the talker's prompt-building "
            "surface; the forward pass belongs to the Torch MPS talker"
        )
