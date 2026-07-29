# SPDX-License-Identifier: Apache-2.0
"""Native SGLang model for the MiniCPM-o 4.5 understanding backbone.

The official checkpoint stores the understanding and speech-generation
networks in one repository.  This class deliberately owns only the
understanding path (Qwen3 + vision + audio encoder).  The same stage loads TTS
as a side component, while the Qwen3 forward path remains an ordinary SGLang
AR model and uses SGLang's paged KV cache.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

import torch
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import MultiModalityDataPaddingPatternTokenPairs
from sglang.srt.managers.schedule_batch import (
    MultimodalDataItem,
    MultimodalInputs,
    flatten_nested_list,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.minicpmo import MiniCPMO as SGLangMiniCPMO
from sglang.srt.models.minicpmo import (
    MiniCPMWhisperEncoder,
    MiniCPMWhisperEncoderLayer,
    MultiModalProjector,
)
from sglang.srt.models.minicpmv import MiniCPMV4_5
from torch import nn
from transformers import PretrainedConfig
from transformers.cache_utils import DynamicCache, EncoderDecoderCache
from transformers.modeling_outputs import BaseModelOutputWithPast

logger = logging.getLogger(__name__)

_WEIGHT_COMPONENT_PREFIXES = (
    "llm.",
    "vpm.",
    "resampler.",
    "apm.",
    "audio_projection_layer.",
)


def _route_weight_name(name: str) -> tuple[str, str] | None:
    """Return ``(component, local_name)`` for a MiniCPM-o 4.5 tensor.

    Keeping this routing explicit prevents the combined checkpoint's ``tts.``
    tensors from being accidentally materialized in the understanding stage.
    ``local_name`` retains the component prefix because that is how parameters
    are named on this module; the component is useful for component-specific
    packed-weight conversion.
    """

    if name.startswith("tts."):
        return None
    for prefix in _WEIGHT_COMPONENT_PREFIXES:
        if name.startswith(prefix):
            return prefix[:-1], name
    raise ValueError(f"unsupported MiniCPM-o checkpoint tensor prefix: {name!r}")


def _packed_weight_target(component: str, name: str) -> tuple[str, str | int] | None:
    """Map HF split projections to the packed SGLang parameter name."""

    if component == "llm":
        for split_name, shard_id in (
            ("q_proj", "q"),
            ("k_proj", "k"),
            ("v_proj", "v"),
            ("gate_proj", 0),
            ("up_proj", 1),
        ):
            marker = f".{split_name}."
            if marker in name:
                packed_name = "qkv_proj" if split_name[0] in "qkv" else "gate_up_proj"
                return name.replace(marker, f".{packed_name}.", 1), shard_id

    if component == "vpm":
        for split_name, shard_id in (
            ("q_proj", "q"),
            ("k_proj", "k"),
            ("v_proj", "v"),
        ):
            marker = f".{split_name}."
            if marker in name:
                return name.replace(marker, ".qkv_proj.", 1), shard_id

    return None


def _required_packed_shards(
    component: str,
    parameter_name: str,
) -> frozenset[str | int] | None:
    if component in {"llm", "vpm"} and ".qkv_proj." in parameter_name:
        return frozenset(("q", "k", "v"))
    if component == "llm" and ".gate_up_proj." in parameter_name:
        return frozenset((0, 1))
    return None


def _validate_loaded_understanding_weights(
    parameter_names: set[str],
    loaded_parameters: set[str],
    packed_shards: dict[str, set[str | int]],
) -> None:
    """Fail closed when a combined checkpoint only partially loads.

    SGLang's outer loader does not consume the set returned by ``load_weights``.
    Validation therefore belongs here, at the point where both the local module
    manifest and every split checkpoint shard are still visible.
    """

    missing_parameters = sorted(parameter_names - loaded_parameters)
    incomplete_packed: list[str] = []
    for parameter_name in sorted(parameter_names):
        component = parameter_name.split(".", 1)[0]
        required = _required_packed_shards(component, parameter_name)
        if required is None:
            continue
        missing = required - packed_shards.get(parameter_name, set())
        if missing:
            incomplete_packed.append(
                f"{parameter_name} missing shards "
                + ", ".join(sorted(map(str, missing)))
            )

    if not missing_parameters and not incomplete_packed:
        return

    details: list[str] = []
    if missing_parameters:
        preview = ", ".join(missing_parameters[:8])
        if len(missing_parameters) > 8:
            preview += f", ... ({len(missing_parameters)} total)"
        details.append(f"missing parameters: {preview}")
    if incomplete_packed:
        preview = "; ".join(incomplete_packed[:8])
        if len(incomplete_packed) > 8:
            preview += f"; ... ({len(incomplete_packed)} total)"
        details.append(f"incomplete packed parameters: {preview}")
    raise RuntimeError(
        "incomplete MiniCPM-o understanding checkpoint: " + "; ".join(details)
    )


class MiniCPMO45WhisperEncoderLayer(MiniCPMWhisperEncoderLayer):
    """Transformers 5.x-compatible cached Whisper encoder layer.

    SGLang's inherited layer still passes ``past_key_value`` (singular), which
    Transformers 5.x silently accepts through ``**kwargs`` but does not use.
    The checkpoint therefore appears to run while every unit ignores its
    session cache.  Keep the SGLang parameter layout and call the current
    WhisperAttention API with ``past_key_values`` (plural).
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor | None,
        output_attentions: bool = False,
        past_key_values: EncoderDecoderCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            past_key_values=past_key_values,
        )
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(
            hidden_states,
            p=self.activation_dropout,
            training=self.training,
        )
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value
            )

        outputs: tuple[torch.Tensor, ...] = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        if use_cache:
            # WhisperAttention updates this cache in place.
            outputs += (past_key_values,)
        return outputs


class MiniCPMO45WhisperEncoder(MiniCPMWhisperEncoder):
    """Whisper encoder with the checkpoint's exact streaming CNN trimming.

    SGLang's existing MiniCPM-o encoder already implements the attention KV
    cache but predates the 4.5 processor's redundant Mel-frame contract.  The
    4.5 processor supplies two Mel frames of CNN context on every steady-state
    chunk; those frames must be used by conv1/conv2 and removed before the
    Transformer/cache update.
    """

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        # Preserve checkpoint key names (``apm.layers.N.*``) while replacing
        # only the version-sensitive attention call.
        self.layers = nn.ModuleList(
            [
                MiniCPMO45WhisperEncoderLayer(config, layer_idx=index)
                for index in range(config.encoder_layers)
            ]
        )

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        past_key_values: EncoderDecoderCache | None = None,
        use_cache: bool | None = None,
        use_extra_context: bool = False,
        prefix_extra_frames: int = 1,
        suffix_extra_frames: int = 1,
        cnn_min_length: int | None = None,
    ) -> BaseModelOutputWithPast | tuple[torch.Tensor, ...]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        input_features = input_features.to(
            dtype=self.conv1.weight.dtype,
            device=self.conv1.weight.device,
        )
        original_length = int(input_features.shape[2])
        padded_for_cnn = bool(
            cnn_min_length is not None and original_length < int(cnn_min_length)
        )
        if padded_for_cnn:
            padded = input_features.new_zeros(
                input_features.shape[0],
                input_features.shape[1],
                int(cnn_min_length),
            )
            padded[:, :, :original_length] = input_features
            input_features = padded

        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))
        if padded_for_cnn:
            inputs_embeds = inputs_embeds[:, :, : (original_length + 1) // 2]

        if use_extra_context:
            prefix_remove = (
                (int(prefix_extra_frames) + 1) // 2
                if int(prefix_extra_frames) > 0
                else 0
            )
            suffix_remove = (
                (int(suffix_extra_frames) + 1) // 2
                if int(suffix_extra_frames) > 0
                else 0
            )
            if prefix_remove:
                inputs_embeds = inputs_embeds[:, :, prefix_remove:]
            if 0 < suffix_remove < inputs_embeds.shape[2]:
                inputs_embeds = inputs_embeds[:, :, :-suffix_remove]

        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        embed_pos = self.embed_positions.weight
        past_length = 0
        if use_cache:
            if past_key_values is None:
                past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())
            elif isinstance(past_key_values, list):
                past_key_values = EncoderDecoderCache(
                    DynamicCache.from_legacy_cache(past_key_values), DynamicCache()
                )
            elif isinstance(past_key_values, DynamicCache):
                past_key_values = EncoderDecoderCache(past_key_values, DynamicCache())
            # Transformers 5.x DynamicCache removed get_usable_length(); the
            # Whisper encoder has no per-layer sliding window, so its current
            # sequence length is the usable history length.
            past_length = past_key_values.self_attention_cache.get_seq_length()
            if past_length + inputs_embeds.shape[1] > embed_pos.shape[0]:
                # This is a defensive fallback. ``encode_audio_streaming``
                # normally resets the session-owned Whisper cache before this
                # branch, matching the official streaming implementation.
                available = embed_pos[past_length:]
                repeated = embed_pos[-1:].expand(
                    inputs_embeds.shape[1] - available.shape[0], -1
                )
                embed_pos = torch.cat((available, repeated), dim=0)
            else:
                embed_pos = embed_pos[
                    past_length : past_length + inputs_embeds.shape[1]
                ]
        else:
            embed_pos = embed_pos[: inputs_embeds.shape[1]]

        hidden_states = inputs_embeds + embed_pos
        hidden_states = nn.functional.dropout(
            hidden_states,
            p=self.dropout,
            training=self.training,
        )
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        if head_mask is not None and head_mask.size(0) != len(self.layers):
            raise ValueError(
                f"head_mask has {head_mask.size(0)} layers, expected {len(self.layers)}"
            )

        next_encoder_cache = None
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states += (hidden_states,)
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask,
                layer_head_mask=head_mask[idx] if head_mask is not None else None,
                output_attentions=output_attentions,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]
            if use_cache:
                next_encoder_cache = layer_outputs[2 if output_attentions else 1]
            if output_attentions:
                all_attentions += (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            encoder_states += (hidden_states,)
        if not return_dict:
            return tuple(
                value
                for value in (hidden_states, encoder_states, all_attentions)
                if value is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
            past_key_values=next_encoder_cache,
        )


class MiniCPMO45ForCausalLM(MiniCPMV4_5):
    """MiniCPM-o 4.5 Qwen3 understanding model on SGLang primitives.

    ``MiniCPMV4_5`` supplies the native Qwen3, MiniCPM-V 4.5 vision path,
    resampler, multimodal embedding merge, and normal SGLang forward contract.
    We add the MiniCPM-o Whisper encoder and projector without constructing a
    second LLM or vision tower.

    Streaming audio cache ownership intentionally stays outside this module.
    A model instance is shared by batched requests, so a single mutable
    ``audio_past_key_values`` field would mix sessions.  The duplex scheduler
    owns per-session audio cache.  This class exposes the stateless audio
    embedding path used by ordinary multimodal prefill; the session-aware
    streaming encoder is a separate pipeline concern.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        # This creates exactly one Qwen3 LLM, one vision tower, and one
        # resampler, all with their checkpoint-native top-level prefixes.
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)

        if not getattr(config, "init_audio", True):
            raise ValueError("MiniCPM-o 4.5 requires init_audio=True")

        self.apm = self.init_audio_module()
        audio_output_dim = int(self.apm.config.encoder_ffn_dim // 4)
        self.audio_avg_pooler = nn.AvgPool1d(
            config.audio_pool_step,
            stride=config.audio_pool_step,
        )
        self.audio_projection_layer = MultiModalProjector(
            in_dim=audio_output_dim,
            out_dim=self.embed_dim,
        )
        self.audio_encoder_layer = -1

    def init_audio_module(self) -> MiniCPMO45WhisperEncoder:
        # FlashAttention's unpadded interface is not compatible with the
        # Whisper streaming-cache path.  Match the official model's eager/SDPA
        # choice while keeping the LLM attention implementation untouched.
        attention_impl = getattr(self.config, "_attn_implementation", "eager")
        self.config.audio_config._attn_implementation = (
            "eager" if attention_impl == "eager" else "sdpa"
        )
        return MiniCPMO45WhisperEncoder(self.config.audio_config)

    # Reuse SGLang's MiniCPM-o audio math.  The regular SGLang multimodal
    # forward calls the non-streaming path below.  Do not alias upstream's
    # model-global streaming cache here: it cannot safely represent multiple
    # concurrent duplex sessions.
    _get_feat_extract_output_lengths = SGLangMiniCPMO._get_feat_extract_output_lengths
    subsequent_chunk_mask = SGLangMiniCPMO.subsequent_chunk_mask
    get_audio_embedding = SGLangMiniCPMO.get_audio_embedding

    def get_audio_feature(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        embeddings = self.get_audio_embedding(
            items,
            chunk_length=getattr(self.config, "audio_chunk_length", -1),
        )
        flattened = flatten_nested_list(embeddings)
        if flattened:
            return torch.cat(flattened, dim=0)
        parameter = next(self.audio_projection_layer.parameters())
        return torch.empty(
            (0, self.embed_dim),
            dtype=parameter.dtype,
            device=parameter.device,
        )

    @torch.no_grad()
    def encode_audio_streaming(
        self,
        data: Any,
        *,
        past_key_values: EncoderDecoderCache | None,
        use_extra_context: bool,
        prefix_extra_frames: int,
        suffix_extra_frames: int,
        cnn_min_length: int | None = None,
    ) -> tuple[list[list[torch.Tensor]], EncoderDecoderCache | None]:
        """Encode one session-owned Mel chunk without model-global cache state."""

        wavforms = data.get("audio_features", [])
        feature_lens_raw = data.get("audio_feature_lens", [])
        if len(wavforms) == 0:
            return [], past_key_values
        if not isinstance(wavforms, torch.Tensor) or wavforms.shape[0] != 1:
            raise ValueError("MiniCPM-o streaming audio supports batch_size=1")

        flattened_lens: list[torch.Tensor] = []
        for group in feature_lens_raw:
            if isinstance(group, torch.Tensor):
                flattened_lens.append(group.reshape(-1))
            else:
                flattened_lens.extend(
                    (
                        value.reshape(-1)
                        if isinstance(value, torch.Tensor)
                        else torch.tensor([value])
                    )
                    for value in group
                )
        if not flattened_lens:
            raise ValueError("streaming audio feature lengths are missing")
        feature_lens = torch.hstack(flattened_lens).to(wavforms.device)

        current_cnn_len = (int(wavforms.shape[-1]) - 1) // 2 + 1
        prefix_remove = (
            (int(prefix_extra_frames) + 1) // 2
            if use_extra_context and int(prefix_extra_frames) > 0
            else 0
        )
        suffix_remove = (
            (int(suffix_extra_frames) + 1) // 2
            if use_extra_context and int(suffix_extra_frames) > 0
            else 0
        )
        current_seq_len = current_cnn_len - prefix_remove - suffix_remove
        if current_seq_len <= 0:
            raise ValueError("streaming audio chunk has no stable CNN frames")

        if past_key_values is None:
            past_len = 0
        else:
            past_len = past_key_values.self_attention_cache.get_seq_length()
            # Whisper learned 1,500 encoder positions (about 30 seconds). The
            # official 4.5 streaming path periodically drops only the Whisper
            # cache; the Qwen paged KV still retains all prior audio embeddings.
            if past_len + current_cnn_len >= self.apm.embed_positions.num_embeddings:
                logger.warning(
                    "MiniCPM-o Whisper cache reached %d positions; resetting "
                    "the audio-encoder cache while preserving the LLM session",
                    past_len + current_cnn_len,
                )
                past_key_values = None
                past_len = 0
        attention_mask = torch.zeros(
            (1, 1, current_seq_len, past_len + current_seq_len),
            dtype=self.apm.conv1.weight.dtype,
            device=wavforms.device,
        )
        output = self.apm(
            wavforms,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            attention_mask=attention_mask,
            use_extra_context=use_extra_context,
            prefix_extra_frames=prefix_extra_frames,
            suffix_extra_frames=suffix_extra_frames,
            cnn_min_length=cnn_min_length,
        )
        audio_states = output.hidden_states[self.audio_encoder_layer]
        audio_embeds = self.audio_projection_layer(audio_states)
        audio_embeds = self.audio_avg_pooler(audio_embeds.transpose(1, 2)).transpose(
            1, 2
        )
        _, pooled_lens = self._get_feat_extract_output_lengths(feature_lens)

        nested: list[list[torch.Tensor]] = []
        row = 0
        for group in feature_lens_raw:
            count = (
                1 if isinstance(group, torch.Tensor) and group.ndim == 0 else len(group)
            )
            current: list[torch.Tensor] = []
            for _ in range(count):
                length = min(int(pooled_lens[row].item()), int(audio_embeds.shape[1]))
                current.append(audio_embeds[row, :length])
                row += 1
            nested.append(current)
        return nested, output.past_key_values

    def pad_input_ids(
        self,
        input_ids: list[int],
        mm_inputs: MultimodalInputs,
    ):
        media_token_pairs = [
            (mm_inputs.im_start_id, mm_inputs.im_end_id),
            (mm_inputs.slice_start_id, mm_inputs.slice_end_id),
            (mm_inputs.audio_start_id, mm_inputs.audio_end_id),
        ]
        pattern = MultiModalityDataPaddingPatternTokenPairs(
            data_token_pairs=media_token_pairs,
            data_start_token_ids=[mm_inputs.im_start_id, mm_inputs.audio_start_id],
        )
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.llm.get_input_embeddings()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs: Any,
    ) -> torch.Tensor:
        if bool(getattr(forward_batch, "minicpmo_projected_input_embeds", False)):
            if forward_batch.input_embeds is None:
                raise RuntimeError("projected MiniCPM-o prefill embeddings are missing")
            return self.llm(
                input_ids=None,
                positions=positions,
                forward_batch=forward_batch,
                input_embeds=forward_batch.input_embeds,
            )
        # Ordinary/offline requests retain MiniCPMV4_5's generic multimodal
        # path.  Duplex requests use the explicit ledger branch above.
        return super().forward(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            **kwargs,
        )

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        """Load the understanding prefixes and leave ``tts.`` to its stage.

        Loading is single-pass so checkpoint tensors are not retained in a
        component-sized temporary list.  Qwen3 and vision split projections
        are packed into SGLang's tensor-parallel parameters as they arrive.
        """

        params = dict(self.named_parameters(remove_duplicate=False))
        loaded: set[str] = set()
        packed_shards: dict[str, set[str | int]] = {}

        for checkpoint_name, loaded_weight in weights:
            routed = _route_weight_name(checkpoint_name)
            if routed is None:
                continue
            component, target_name = routed

            if component == "llm" and (
                "rotary_emb.inv_freq" in target_name
                or "rotary_emb.cos_cached" in target_name
                or "rotary_emb.sin_cached" in target_name
            ):
                continue

            if component == "vpm":
                target_name = target_name.replace(
                    ".self_attn.out_proj.",
                    ".self_attn.proj.",
                    1,
                )

            packed = _packed_weight_target(component, target_name)
            if packed is not None:
                packed_name, shard_id = packed
                param = params.get(packed_name)
                if param is None:
                    raise RuntimeError(
                        "MiniCPM-o checkpoint tensor has no local packed target: "
                        f"{target_name!r} -> {packed_name!r}"
                    )
                weight_loader = getattr(param, "weight_loader", None)
                if weight_loader is None:
                    raise RuntimeError(
                        f"packed MiniCPM-o parameter {packed_name!r} has no weight_loader"
                    )
                weight_loader(param, loaded_weight, shard_id)
                loaded.add(packed_name)
                packed_shards.setdefault(packed_name, set()).add(shard_id)
                continue

            param = params.get(target_name)
            if param is None:
                raise RuntimeError(
                    "MiniCPM-o checkpoint tensor has no local target: "
                    f"{target_name!r}"
                )
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded.add(target_name)

        _validate_loaded_understanding_weights(set(params), loaded, packed_shards)
        return loaded


EntryClass = MiniCPMO45ForCausalLM

__all__ = [
    "EntryClass",
    "MiniCPMO45ForCausalLM",
    "MiniCPMO45WhisperEncoder",
]
