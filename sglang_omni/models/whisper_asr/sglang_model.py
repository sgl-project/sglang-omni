# SPDX-License-Identifier: Apache-2.0
"""SGLang-native Whisper ASR model.

The Whisper encoder runs as the encoder side of an encoder-decoder SGLang
request. The decoder uses RadixAttention for both autoregressive self-attention
and cached cross-attention over encoder states, so normal SGLang KV cache,
CUDA Graph, and torch.compile paths apply to decode.
"""

from __future__ import annotations

from typing import Any, Iterable, Tuple

import torch
import torch.nn.functional as F
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from torch import nn
from transformers import WhisperConfig
from transformers.activations import ACT2FN


class WhisperEncoderAttention(nn.Module):
    def __init__(self, config: WhisperConfig) -> None:
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = states.shape
        return states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        query = self._shape(self.q_proj(hidden_states))
        key = self._shape(self.k_proj(hidden_states))
        value = self._shape(self.v_proj(hidden_states))
        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=False,
        )
        attn_output = attn_output.transpose(1, 2).reshape(
            hidden_states.shape[0],
            hidden_states.shape[1],
            self.embed_dim,
        )
        return self.out_proj(attn_output)


class WhisperEncoderLayer(nn.Module):
    def __init__(self, config: WhisperConfig) -> None:
        super().__init__()
        self.self_attn = WhisperEncoderAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.fc1 = nn.Linear(config.d_model, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, config.d_model)
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        self.activation_fn = ACT2FN[config.activation_function]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc2(self.activation_fn(self.fc1(hidden_states)))
        return residual + hidden_states


class WhisperEncoder(nn.Module):
    def __init__(self, config: WhisperConfig) -> None:
        super().__init__()
        self.config = config
        self.conv1 = nn.Conv1d(
            config.num_mel_bins,
            config.d_model,
            kernel_size=3,
            padding=1,
        )
        self.conv2 = nn.Conv1d(
            config.d_model,
            config.d_model,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.embed_positions = nn.Embedding(config.max_source_positions, config.d_model)
        self.layers = nn.ModuleList(
            [WhisperEncoderLayer(config) for _ in range(config.encoder_layers)]
        )
        self.layer_norm = nn.LayerNorm(config.d_model)

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        # Note:(Chenchen Hong) move input_features to the conv weight's device
        # (not just dtype), else the CUDA conv1 raises a device-mismatch error.
        hidden_states = input_features.to(
            device=self.conv1.weight.device, dtype=self.conv1.weight.dtype
        )
        hidden_states = F.gelu(self.conv1(hidden_states))
        hidden_states = F.gelu(self.conv2(hidden_states))
        hidden_states = hidden_states.permute(0, 2, 1)

        embed_pos = self.embed_positions.weight[: hidden_states.shape[1]]
        hidden_states = hidden_states + embed_pos.to(hidden_states.device)

        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.layer_norm(hidden_states)


class WhisperSGLangSelfAttention(nn.Module):
    def __init__(
        self,
        config: WhisperConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        del quant_config, prefix
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.decoder_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            scaling=self.scaling,
            num_kv_heads=self.num_heads,
            layer_id=layer_id,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        query = self.q_proj(hidden_states).view(-1, self.num_heads, self.head_dim)
        key = self.k_proj(hidden_states).view(-1, self.num_heads, self.head_dim)
        value = self.v_proj(hidden_states).view(-1, self.num_heads, self.head_dim)
        attn_output = self.attn(query, key, value, forward_batch)
        return self.out_proj(attn_output)


class WhisperSGLangCrossAttention(nn.Module):
    def __init__(
        self,
        config: WhisperConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        del quant_config, prefix
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.decoder_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            scaling=self.scaling,
            num_kv_heads=self.num_heads,
            layer_id=layer_id,
            is_cross_attention=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: torch.Tensor | None,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        query = self.q_proj(hidden_states).view(-1, self.num_heads, self.head_dim)
        if cross_attention_states is None:
            key = value = None
        else:
            key = self.k_proj(cross_attention_states).view(
                -1, self.num_heads, self.head_dim
            )
            value = self.v_proj(cross_attention_states).view(
                -1, self.num_heads, self.head_dim
            )
        attn_output = self.attn(query, key, value, forward_batch)
        return self.out_proj(attn_output)


class WhisperDecoderLayer(nn.Module):
    def __init__(
        self,
        config: WhisperConfig,
        layer_idx: int,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        num_decoder_layers = int(config.decoder_layers)
        self.self_attn = WhisperSGLangSelfAttention(
            config,
            layer_id=layer_idx,
            quant_config=quant_config,
        )
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.encoder_attn = WhisperSGLangCrossAttention(
            config,
            layer_id=num_decoder_layers + layer_idx,
            quant_config=quant_config,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.fc1 = nn.Linear(config.d_model, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, config.d_model)
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        self.activation_fn = ACT2FN[config.activation_function]

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: torch.Tensor | None,
        forward_batch: ForwardBatch,
        skip_cross_attention: bool,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, forward_batch)
        hidden_states = residual + hidden_states

        if not skip_cross_attention:
            residual = hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)
            hidden_states = self.encoder_attn(
                hidden_states,
                cross_attention_states,
                forward_batch,
            )
            hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc2(self.activation_fn(self.fc1(hidden_states)))
        return residual + hidden_states


class WhisperDecoder(nn.Module):
    def __init__(
        self,
        config: WhisperConfig,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.embed_positions = nn.Embedding(
            config.max_target_positions,
            config.d_model,
        )
        self.layers = nn.ModuleList(
            [
                WhisperDecoderLayer(config, layer_idx=i, quant_config=quant_config)
                for i in range(config.decoder_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        cross_attention_states: torch.Tensor | None,
        forward_batch: ForwardBatch,
        skip_cross_attention: bool,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = hidden_states + self.embed_positions(positions).to(
            hidden_states.device
        )
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                cross_attention_states,
                forward_batch,
                skip_cross_attention,
            )
        return self.layer_norm(hidden_states)


class WhisperModel(nn.Module):
    def __init__(
        self,
        config: WhisperConfig,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        self.encoder = WhisperEncoder(config)
        self.decoder = WhisperDecoder(config, quant_config=quant_config)


class WhisperForConditionalGeneration(nn.Module):
    def __init__(
        self,
        config: WhisperConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        del prefix
        super().__init__()
        self.config = config
        self.model = WhisperModel(config, quant_config=quant_config)
        self.proj_out = self.model.decoder.embed_tokens
        self.lm_head = self.proj_out
        self.logits_processor = LogitsProcessor(config)
        self.start_layer = 0
        self.end_layer = int(config.decoder_layers) * 2
        # Opt-in cross-request encoder-output cache. Off by default so behavior
        # is unchanged; when enabled, identical audio features (e.g. repeated
        # reference clips) reuse a cached encoder forward instead of recomputing.
        self._encoder_cache_enabled = bool(getattr(config, "enable_encoder_cache", False))
        self._encoder_cache = None
        if self._encoder_cache_enabled:
            from sglang_omni.models.whisper_asr.whisper_encoder_cache import (
                EncoderOutputCache,
            )

            self._encoder_cache = EncoderOutputCache(
                getattr(config, "_name_or_path", "whisper"),
                max_size=getattr(config, "encoder_cache_max_items", 256),
                max_bytes=getattr(config, "encoder_cache_max_bytes", 256 * 1024 * 1024),
            )

    def _batch_audio_inputs(
        self,
        forward_batch: ForwardBatch,
    ) -> tuple[torch.Tensor | None, list[int] | None]:
        if forward_batch.forward_mode.is_decode() or all(forward_batch.encoder_cached):
            return None, None

        features: list[torch.Tensor] = []
        encoder_lens: list[int] = []
        for index, mm_input in enumerate(forward_batch.mm_inputs):
            if forward_batch.encoder_cached[index] or mm_input is None:
                continue
            item_features = [
                item.feature for item in mm_input.mm_items if item.feature is not None
            ]
            if not item_features:
                continue
            features.append(torch.cat(item_features, dim=0))
            encoder_lens.append(int(forward_batch.encoder_lens[index].item()))

        if not features:
            return None, None
        return torch.cat(features, dim=0), encoder_lens

    @staticmethod
    def _flat_encoder_result(
        encoder_states: torch.Tensor,
        encoder_lens: list[int],
    ) -> torch.Tensor:
        # Vectorized equivalent of the previous per-request Python loop: gather
        # each request's valid encoder slice [index, :encoder_len] and concat
        # them in one kernel. Avoids the Python-level loop and the explicit
        # empty-tensor allocation; numerically identical to the old copy.
        if not encoder_lens:
            hidden_size = encoder_states.shape[-1]
            return torch.empty(
                0, hidden_size, device=encoder_states.device, dtype=encoder_states.dtype
            )
        slices = [
            encoder_states[index, :encoder_len]
            for index, encoder_len in enumerate(encoder_lens)
        ]
        return torch.cat(slices, dim=0)

    def _maybe_cached_encoder_forward(
        self,
        forward_batch: ForwardBatch,
        audio_features: torch.Tensor,
        encoder_lens: list[int],
    ) -> torch.Tensor:
        """Run the encoder, reusing a cached result when the audio is identical.

        The Whisper encoder is a deterministic, stateless forward pass: the same
        audio feature map always yields the same ``encoder_states``. We
        content-address the encoder input (shape + bytes of every request's
        feature in this batch, order-independent) and, on a hit, skip the encoder
        entirely. This is an opt-in, batch-granularity cache gated by
        ``self._encoder_cache_enabled`` (set from ``config.enable_encoder_cache``);
        when disabled this is a plain ``self.model.encoder`` call with no overhead.
        """
        if not self._encoder_cache_enabled or self._encoder_cache is None:
            return self.model.encoder(audio_features)

        features = []
        for mm_input in forward_batch.mm_inputs:
            if mm_input is None:
                continue
            for item in mm_input.mm_items:
                if getattr(item, "feature", None) is not None:
                    features.append(item.feature)

        device = audio_features.device
        return self._encoder_cache.get_or_encode(
            features,
            device,
            encode_fn=lambda: self.model.encoder(audio_features),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs: Any,
    ) -> Any:
        del kwargs
        from sglang.srt.model_executor.runner_utils.capture_mode import (
            get_is_capture_mode,
        )

        audio_features, encoder_lens = self._batch_audio_inputs(forward_batch)
        cross_attention_states = None

        if get_is_capture_mode():
            skip_cross_attention = False
        else:
            skip_cross_attention = forward_batch.encoder_lens.max() == 0

        if audio_features is not None and encoder_lens is not None:
            encoder_states = self._maybe_cached_encoder_forward(
                forward_batch, audio_features, encoder_lens
            )
            cross_attention_states = self._flat_encoder_result(
                encoder_states,
                encoder_lens,
            )

        hidden_states = self.model.decoder(
            input_ids=input_ids,
            positions=positions,
            cross_attention_states=cross_attention_states,
            forward_batch=forward_batch,
            skip_cross_attention=skip_cross_attention,
        )
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if name == "proj_out.weight":
                name = "model.decoder.embed_tokens.weight"
            if name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)


EntryClass = WhisperForConditionalGeneration
