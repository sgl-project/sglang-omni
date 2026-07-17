# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import math
from typing import Any, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3 import Qwen3ForCausalLM
from sglang.srt.utils import add_prefix
from transformers.activations import ACT2FN

from .configuration_fun_asr import FunAsrNanoConfig
from .tool_funcs.audio_lengths import fun_asr_low_frame_rate_length

logger = logging.getLogger(__name__)


class SinusoidalPositionEncoder(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, timesteps, input_dim = x.size()
        positions = torch.arange(1, timesteps + 1, device=x.device, dtype=x.dtype)
        log_timescale_increment = math.log(10000.0) / (input_dim / 2 - 1)
        inv_timescales = torch.exp(
            torch.arange(input_dim / 2, device=x.device, dtype=x.dtype)
            * (-log_timescale_increment)
        )
        scaled_time = positions.view(1, -1, 1) * inv_timescales.view(1, 1, -1)
        encoding = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=2)
        return x + encoding


class MultiHeadedAttentionSANM(nn.Module):

    def __init__(
        self,
        n_head: int,
        in_feat: int,
        n_feat: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head
        self.h = n_head
        self.q_proj = nn.Linear(in_feat, n_feat)
        self.k_proj = nn.Linear(in_feat, n_feat)
        self.v_proj = nn.Linear(in_feat, n_feat)
        self.out_proj = nn.Linear(n_feat, n_feat)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.size()
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q_h = q.view(b, t, self.h, self.d_k).transpose(1, 2)  # (b, h, t, dk)
        k_h = k.view(b, t, self.h, self.d_k).transpose(1, 2)
        v_h = v.view(b, t, self.h, self.d_k).transpose(1, 2)

        q_h = q_h * (self.d_k**-0.5)
        scores = torch.matmul(q_h, k_h.transpose(-2, -1))  # (b, h, t, t)
        attn = torch.softmax(scores, dim=-1)
        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, v_h)  # (b, h, t, dk)
        x = x.transpose(1, 2).contiguous().view(b, -1, self.h * self.d_k)
        return self.out_proj(x)


class FunAsrNanoFSMN(nn.Module):

    def __init__(self, size: int, kernel_size: int, dropout_rate: float) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            size,
            size,
            kernel_size,
            stride=1,
            padding=0,
            groups=size,
            bias=False,
        )
        left_padding = (kernel_size - 1) // 2
        right_padding = kernel_size - 1 - left_padding
        self.pad = nn.ConstantPad1d((left_padding, right_padding), 0.0)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, value_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv(self.pad(value_states.transpose(1, 2)))
        hidden_states = hidden_states.transpose(1, 2) + value_states
        return self.dropout(hidden_states)


class EncoderLayerSANM(nn.Module):

    def __init__(
        self,
        in_size: int,
        size: int,
        attention_heads: int,
        linear_units: int,
        kernel_size: int,
        dropout_rate: float,
        attention_dropout_rate: float,
        activation_dropout_rate: float,
        activation_function: str,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadedAttentionSANM(
            attention_heads, in_size, size, attention_dropout_rate
        )
        self.self_attn_layer_norm = nn.LayerNorm(in_size, eps=1e-5)
        self.final_layer_norm = nn.LayerNorm(size, eps=1e-5)
        self.fc1 = nn.Linear(size, linear_units)
        self.fc2 = nn.Linear(linear_units, size)
        self.fsmn = FunAsrNanoFSMN(size, kernel_size, attention_dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation_dropout = nn.Dropout(activation_dropout_rate)
        self.activation = ACT2FN[activation_function]
        self.in_size = in_size
        self.size = size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.self_attn_layer_norm(x)
        value_states = self.self_attn.v_proj(x)
        x = self.dropout(self.self_attn(x) + self.fsmn(value_states))
        if self.in_size == self.size:
            x = residual + x
        residual = x
        x = self.final_layer_norm(x)
        x = self.activation_dropout(self.activation(self.fc1(x)))
        x = residual + self.dropout(self.fc2(x))
        if x.dtype == torch.float16:
            clamp_value = torch.finfo(x.dtype).max - 1000
            x = torch.clamp(x, min=-clamp_value, max=clamp_value)
        return x


class FunAsrNanoAudioEncoder(nn.Module):

    def __init__(
        self,
        input_size: int = 560,
        output_size: int = 512,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 50,
        tp_blocks: int = 20,
        kernel_size: int = 11,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        activation_dropout_rate: float = 0.1,
        activation_function: str = "relu",
    ) -> None:
        super().__init__()
        self._output_size = output_size
        self.embed = SinusoidalPositionEncoder()

        def make_layer(in_size: int) -> EncoderLayerSANM:
            return EncoderLayerSANM(
                in_size,
                output_size,
                attention_heads,
                linear_units,
                kernel_size,
                dropout_rate,
                attention_dropout_rate,
                activation_dropout_rate,
                activation_function,
            )

        self.stem = make_layer(input_size)
        self.layers = nn.ModuleList(
            [make_layer(output_size) for _ in range(num_blocks - 1)]
        )
        self.layer_norm = nn.LayerNorm(output_size, eps=1e-5)
        self.timestamp_prediction_layers = nn.ModuleList(
            [make_layer(output_size) for _ in range(tp_blocks)]
        )
        self.timestamp_prediction_layer_norm = nn.LayerNorm(output_size, eps=1e-5)

    def output_size(self) -> int:
        return self._output_size

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        xs = xs * (self._output_size**0.5)
        xs = self.embed(xs)
        xs = self.stem(xs)
        for layer in self.layers:
            xs = layer(xs)
        xs = self.layer_norm(xs)
        for layer in self.timestamp_prediction_layers:
            xs = layer(xs)
        xs = self.timestamp_prediction_layer_norm(xs)
        return xs


class MultiHeadedAttention(nn.Module):

    def __init__(
        self,
        n_head: int,
        n_feat: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head
        self.h = n_head
        self.q_proj = nn.Linear(n_feat, n_feat)
        self.k_proj = nn.Linear(n_feat, n_feat)
        self.v_proj = nn.Linear(n_feat, n_feat)
        self.out_proj = nn.Linear(n_feat, n_feat)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.size()
        q_h = self.q_proj(x).view(b, t, self.h, self.d_k).transpose(1, 2)
        k_h = self.k_proj(x).view(b, t, self.h, self.d_k).transpose(1, 2)
        v_h = self.v_proj(x).view(b, t, self.h, self.d_k).transpose(1, 2)
        q_h = q_h * (self.d_k**-0.5)
        scores = torch.matmul(q_h, k_h.transpose(-2, -1))
        attn = torch.softmax(scores, dim=-1)
        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, v_h)
        x = x.transpose(1, 2).contiguous().view(b, -1, self.h * self.d_k)
        return self.out_proj(x)


class AdaptorEncoderLayer(nn.Module):

    def __init__(
        self,
        size: int,
        self_attn: MultiHeadedAttention,
        feed_forward_dim: int,
        dropout_rate: float,
        activation_function: str,
    ) -> None:
        super().__init__()
        self.self_attn = self_attn
        self.self_attn_layer_norm = nn.LayerNorm(size, eps=1e-5)
        self.final_layer_norm = nn.LayerNorm(size, eps=1e-5)
        self.fc1 = nn.Linear(size, feed_forward_dim)
        self.fc2 = nn.Linear(feed_forward_dim, size)
        self.activation = ACT2FN[activation_function]
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.self_attn_layer_norm(x)
        x = residual + self.dropout(self.self_attn(x))
        residual = x
        x = self.final_layer_norm(x)
        x = residual + self.dropout(self.fc2(self.activation(self.fc1(x))))
        return x


class FunAsrNanoAdaptor(nn.Module):

    def __init__(
        self,
        encoder_dim: int = 512,
        llm_dim: int = 1024,
        ffn_dim: int = 2048,
        num_layers: int = 2,
        attention_heads: int = 8,
        dropout_rate: float = 0.0,
        activation_function: str = "relu",
    ) -> None:
        super().__init__()
        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim
        self.linear_1 = nn.Linear(encoder_dim, ffn_dim)
        self.act = ACT2FN[activation_function]
        self.linear_2 = nn.Linear(ffn_dim, llm_dim)

        ffn_hidden = llm_dim // 4
        self.blocks = nn.ModuleList(
            [
                AdaptorEncoderLayer(
                    llm_dim,
                    MultiHeadedAttention(attention_heads, llm_dim, dropout_rate),
                    ffn_hidden,
                    dropout_rate,
                    activation_function,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        for block in self.blocks:
            x = block(x)
        return x


class FunAsrNanoForConditionalGeneration(nn.Module):

    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    bitsandbytes_stacked_params_mapping = {
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: FunAsrNanoConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        enc_cfg = config.encoder_config

        self.audio_tower = FunAsrNanoAudioEncoder(
            input_size=enc_cfg.input_size,
            output_size=enc_cfg.d_model,
            attention_heads=enc_cfg.encoder_attention_heads,
            linear_units=enc_cfg.encoder_ffn_dim,
            num_blocks=enc_cfg.encoder_layers,
            tp_blocks=enc_cfg.num_timestamp_prediction_blocks,
            kernel_size=enc_cfg.kernel_size,
            dropout_rate=enc_cfg.dropout,
            attention_dropout_rate=enc_cfg.attention_dropout,
            activation_dropout_rate=enc_cfg.activation_dropout,
            activation_function=enc_cfg.activation_function,
        )
        self.multi_modal_projector = FunAsrNanoAdaptor(
            encoder_dim=enc_cfg.d_model,
            llm_dim=config.text_config.hidden_size,
            ffn_dim=config.adaptor_intermediate_size,
            num_layers=config.adaptor_num_hidden_layers,
            attention_heads=config.adaptor_num_attention_heads,
            dropout_rate=0.0,
            activation_function=config.activation_function,
        )
        self.language_model = Qwen3ForCausalLM(
            config.text_config,
            quant_config,
            prefix=add_prefix("language_model", prefix),
        )
        self.pattern = MultiModalityDataPaddingPatternMultimodalTokens()

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        return self.pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_audio_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:

        device = next(self.audio_tower.parameters()).device
        dtype = next(self.audio_tower.parameters()).dtype

        embeddings: List[torch.Tensor] = []
        for item in items:
            feature = item.feature.to(device=device, dtype=dtype)
            # feature: [1, input_size=560, T_padded] (LFR-stacked).
            mask = getattr(item, "feature_attention_mask", None)
            if mask is not None:
                mask = mask.to(device=device)
                valid = int(mask.sum().item())
                # Single audio per item; take the first row's valid frames.
                feature = feature[:, :, :valid]
            # [1, 560, T] → [1, T, 560] (encoder expects [B, T, D]).
            xs = feature.permute(0, 2, 1).contiguous()
            enc_out = self.audio_tower(xs)  # [1, T, 512]
            adp_out = self.multi_modal_projector(enc_out)  # [1, T, 1024]
            t_lfr = adp_out.shape[1]
            num_tokens = int(fun_asr_low_frame_rate_length(t_lfr))
            num_tokens = max(num_tokens, 1)
            embeddings.append(adp_out[0, :num_tokens, :])  # [num_tokens, 1024]

        return torch.cat(embeddings, dim=0)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs: Any,
    ) -> torch.Tensor:
        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            data_embedding_funcs={
                Modality.AUDIO: self.get_audio_feature,
            },
            positions=positions,
        )
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # Qwen3 LLM: q/k/v → qkv_proj, gate/up → gate_up_proj (sglang stacked).
        llm_stacked_params = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))

        for name, loaded_weight in weights:
            checkpoint_name = name
            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue

            if getattr(self.config.text_config, "tie_word_embeddings", False) and (
                name == "lm_head.weight" or name.endswith(".lm_head.weight")
            ):
                continue

            strict_multimodal = False
            if name.startswith("model.audio_tower."):
                name = name.replace("model.", "", 1)
                is_llm = False
                strict_multimodal = True
            elif name.startswith("model.multi_modal_projector."):
                name = name.replace("model.", "", 1)
                is_llm = False
                strict_multimodal = True
            elif name.startswith("model.language_model."):
                name = name.replace("model.language_model.", "language_model.model.", 1)
                is_llm = True
            elif name == "lm_head.weight":
                name = "language_model.lm_head.weight"
                is_llm = True
            else:
                is_llm = False

            if is_llm:
                stacked = False
                for param_name, weight_name, shard_id in llm_stacked_params:
                    if weight_name not in name:
                        continue
                    name_tmp = name.replace(weight_name, param_name)
                    if name_tmp.endswith(".bias") and name_tmp not in params_dict:
                        continue
                    if name_tmp not in params_dict:
                        continue
                    param = params_dict[name_tmp]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)
                    stacked = True
                    break
                if stacked:
                    continue

            if name.endswith(".bias") and name not in params_dict:
                continue
            if name not in params_dict:
                if strict_multimodal:
                    raise ValueError(
                        f"Fun-ASR checkpoint weight {checkpoint_name} has no matching "
                        f"model parameter ({name})"
                    )
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)


EntryClass = FunAsrNanoForConditionalGeneration
