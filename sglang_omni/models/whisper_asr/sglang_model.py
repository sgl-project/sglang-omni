# SPDX-License-Identifier: Apache-2.0
"""SGLang-native Whisper ASR model.

The Whisper encoder runs as the encoder side of an encoder-decoder SGLang
request. The decoder uses RadixAttention for both autoregressive self-attention
and cached cross-attention over encoder states, so normal SGLang KV cache,
CUDA Graph, and torch.compile paths apply to decode.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from typing import Any

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

from sglang_omni.models.whisper_asr.encoder_cuda_graph import (
    WhisperEncoderCudaGraphRunner,
)

try:
    from flashinfer.norm import layernorm as flashinfer_layer_norm
except (ImportError, AttributeError):
    flashinfer_layer_norm = None

logger = logging.getLogger(__name__)

_QKV_SHARDS = {"q_proj": 0, "k_proj": 1, "v_proj": 2}
_KV_SHARDS = {"k_proj": 0, "v_proj": 1}


class WhisperDecoderLayerNorm(nn.LayerNorm):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if (
            flashinfer_layer_norm is not None
            and os.environ.get("FLASHINFER_USE_CUDA_NORM") != "1"
            and hidden_states.is_cuda
            and hidden_states.dtype == torch.float16
            and self.weight is not None
            and self.bias is not None
            and self.weight.dtype == hidden_states.dtype
            and self.bias.dtype == hidden_states.dtype
        ):
            return flashinfer_layer_norm(
                hidden_states, self.weight, self.bias, self.eps
            )
        return super().forward(hidden_states)


def _load_projection_shard(
    param: torch.Tensor,
    loaded_weight: torch.Tensor,
    *,
    shard: int,
    shard_size: int,
) -> None:
    target = param.data.narrow(0, shard * shard_size, shard_size)
    default_weight_loader(target, loaded_weight)


class WhisperEncoderAttention(nn.Module):
    def __init__(self, config: WhisperConfig) -> None:
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.qkv_proj = nn.Linear(self.embed_dim, 3 * self.embed_dim)
        # Whisper K projections have no bias. The zero K shard preserves that
        # checkpoint structure while issuing one GEMM for all three projections.
        with torch.no_grad():
            self.qkv_proj.bias[self.embed_dim : 2 * self.embed_dim].zero_()
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = states.shape
        return states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        query, key, value = self.qkv_proj(hidden_states).chunk(3, dim=-1)
        query = self._shape(query)
        key = self._shape(key)
        value = self._shape(value)
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
        self.qkv_proj = nn.Linear(self.embed_dim, 3 * self.embed_dim)
        with torch.no_grad():
            self.qkv_proj.bias[self.embed_dim : 2 * self.embed_dim].zero_()
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
        query, key, value = self.qkv_proj(hidden_states).chunk(3, dim=-1)
        query = query.view(-1, self.num_heads, self.head_dim)
        key = key.view(-1, self.num_heads, self.head_dim)
        value = value.view(-1, self.num_heads, self.head_dim)
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
        self.kv_proj = nn.Linear(self.embed_dim, 2 * self.embed_dim)
        with torch.no_grad():
            self.kv_proj.bias[: self.embed_dim].zero_()
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
            key, value = self.kv_proj(cross_attention_states).chunk(2, dim=-1)
            key = key.view(-1, self.num_heads, self.head_dim)
            value = value.view(-1, self.num_heads, self.head_dim)
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
        self.self_attn_layer_norm = WhisperDecoderLayerNorm(config.d_model)
        self.encoder_attn = WhisperSGLangCrossAttention(
            config,
            layer_id=num_decoder_layers + layer_idx,
            quant_config=quant_config,
        )
        self.encoder_attn_layer_norm = WhisperDecoderLayerNorm(config.d_model)
        self.fc1 = nn.Linear(config.d_model, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, config.d_model)
        self.final_layer_norm = WhisperDecoderLayerNorm(config.d_model)
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
        self.layer_norm = WhisperDecoderLayerNorm(config.d_model)

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
        self._encoder_graph_runner: WhisperEncoderCudaGraphRunner | None = None

    def init_encoder_graphs(
        self,
        batch_buckets: list[int] | tuple[int, ...],
        input_feature_len: int,
    ) -> None:
        """Capture fixed-shape Whisper encoder batches after model setup."""
        if not batch_buckets:
            return
        self._encoder_graph_runner = WhisperEncoderCudaGraphRunner(
            self.model.encoder,
            num_mel_bins=int(self.config.num_mel_bins),
            input_feature_len=int(input_feature_len),
        )
        self._encoder_graph_runner.capture(batch_buckets)

    def _run_encoder(self, audio_features: torch.Tensor) -> torch.Tensor:
        """Run the Whisper encoder with CUDA-graph replay when available."""
        if self._encoder_graph_runner is not None:
            try:
                return self._encoder_graph_runner.run(audio_features)
            except Exception:
                logger.exception(
                    "Whisper encoder CUDA graph replay failed; falling back to eager"
                )
        return self.model.encoder(audio_features)

    def encode_audio_features(self, items: list[Any]) -> torch.Tensor:
        """Batch-encode mel features into encoder states [B, T, H]."""
        if not items:
            raise ValueError(
                "Whisper encode_audio_features requires at least one audio item"
            )
        features: list[torch.Tensor] = []
        reference = next(self.model.encoder.parameters())
        for item in items:
            feature = item.feature
            if feature is None:
                raise RuntimeError(
                    "Whisper audio item is missing mel features; cannot encode"
                )
            if not isinstance(feature, torch.Tensor):
                feature = torch.as_tensor(feature)
            features.append(feature.to(device=reference.device, dtype=reference.dtype))
        return self._run_encoder(torch.cat(features, dim=0))

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

    def _batch_precomputed_encoder_states(
        self,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor | None:
        """Collect pre-LM encoder hiddens into a flat cross-attn tensor."""
        if forward_batch.forward_mode.is_decode() or all(forward_batch.encoder_cached):
            return None

        parts: list[torch.Tensor] = []
        for index, mm_input in enumerate(forward_batch.mm_inputs):
            if forward_batch.encoder_cached[index] or mm_input is None:
                continue
            encoder_len = int(forward_batch.encoder_lens[index].item())
            for item in mm_input.mm_items:
                embedding = item.precomputed_embeddings
                if embedding is None:
                    continue
                if not isinstance(embedding, torch.Tensor):
                    embedding = torch.as_tensor(embedding)
                if embedding.dim() != 2 or embedding.shape[0] < encoder_len:
                    raise RuntimeError(
                        "Whisper precomputed encoder states "
                        f"{tuple(embedding.shape)} incompatible with "
                        f"encoder_len={encoder_len}"
                    )
                parts.append(embedding[:encoder_len])
        if not parts:
            return None
        return torch.cat(parts, dim=0)

    @staticmethod
    def _flat_encoder_result(
        encoder_states: torch.Tensor,
        encoder_lens: list[int],
    ) -> torch.Tensor:
        hidden_size = encoder_states.shape[-1]
        total_encoder_len = sum(encoder_lens)
        flat = torch.empty(
            total_encoder_len,
            hidden_size,
            device=encoder_states.device,
            dtype=encoder_states.dtype,
        )
        dst_start = 0
        for index, encoder_len in enumerate(encoder_lens):
            dst_end = dst_start + encoder_len
            flat[dst_start:dst_end] = encoder_states[index, :encoder_len]
            dst_start = dst_end
        return flat

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

        cross_attention_states = self._batch_precomputed_encoder_states(forward_batch)
        if cross_attention_states is None:
            audio_features, encoder_lens = self._batch_audio_inputs(forward_batch)
            if audio_features is not None and encoder_lens is not None:
                encoder_states = self._run_encoder(audio_features)
                cross_attention_states = self._flat_encoder_result(
                    encoder_states,
                    encoder_lens,
                )

        if get_is_capture_mode():
            skip_cross_attention = False
        else:
            skip_cross_attention = forward_batch.encoder_lens.max() == 0

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
            projection = name.rsplit(".", 2)[-2]
            if ".self_attn." in name and projection in _QKV_SHARDS:
                target_name = name.replace(f".{projection}.", ".qkv_proj.", 1)
                param = params_dict[target_name]
                _load_projection_shard(
                    param,
                    loaded_weight,
                    shard=_QKV_SHARDS[projection],
                    shard_size=self.config.d_model,
                )
                continue
            if ".encoder_attn." in name and projection in _KV_SHARDS:
                target_name = name.replace(f".{projection}.", ".kv_proj.", 1)
                param = params_dict[target_name]
                _load_projection_shard(
                    param,
                    loaded_weight,
                    shard=_KV_SHARDS[projection],
                    shard_size=self.config.d_model,
                )
                continue
            if name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)


EntryClass = WhisperForConditionalGeneration
