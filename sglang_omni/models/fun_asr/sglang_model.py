# SPDX-License-Identifier: Apache-2.0
# Author:
# PoTaTo-Mika: https://github.com/PoTaTo-Mika

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

from .configuration_fun_asr import FunAsrNanoConfig
from .tool_funcs.audio_lengths import fun_asr_low_frame_rate_length

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Positional encoding (funasr SinusoidalPositionEncoder, verbatim)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# SANM self-attention (funasr MultiHeadedAttentionSANM, verbatim)
# ---------------------------------------------------------------------------


class MultiHeadedAttentionSANM(nn.Module):

    def __init__(
        self,
        n_head: int,
        in_feat: int,
        n_feat: int,
        dropout_rate: float,
        kernel_size: int,
        sanm_shfit: int = 0,
    ) -> None:
        super().__init__()
        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.linear_q_k_v = nn.Linear(in_feat, n_feat * 3)
        self.fsmn_block = nn.Conv1d(
            n_feat, n_feat, kernel_size, stride=1, padding=0, groups=n_feat, bias=False
        )
        left_padding = (kernel_size - 1) // 2
        if sanm_shfit > 0:
            left_padding = left_padding + sanm_shfit
        right_padding = kernel_size - 1 - left_padding
        self.pad_fn = nn.ConstantPad1d((left_padding, right_padding), 0.0)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_fsmn(self, v: torch.Tensor) -> torch.Tensor:
        x = v.transpose(1, 2)  # (b, d, t)
        x = self.pad_fn(x)
        x = self.fsmn_block(x)
        x = x.transpose(1, 2)  # (b, t, d)
        x = x + v  # residual
        x = self.dropout(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.size()
        q_k_v = self.linear_q_k_v(x)
        q, k, v = torch.split(q_k_v, int(self.h * self.d_k), dim=-1)
        q_h = q.view(b, t, self.h, self.d_k).transpose(1, 2)  # (b, h, t, dk)
        k_h = k.view(b, t, self.h, self.d_k).transpose(1, 2)
        v_h = v.view(b, t, self.h, self.d_k).transpose(1, 2)

        fsmn_memory = self.forward_fsmn(v)
        q_h = q_h * (self.d_k**-0.5)
        scores = torch.matmul(q_h, k_h.transpose(-2, -1))  # (b, h, t, t)
        attn = torch.softmax(scores, dim=-1)
        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, v_h)  # (b, h, t, dk)
        x = x.transpose(1, 2).contiguous().view(b, -1, self.h * self.d_k)
        return self.linear_out(x) + fsmn_memory


# ---------------------------------------------------------------------------
# Encoder layer (funasr EncoderLayerSANM, verbatim)
# ---------------------------------------------------------------------------


class EncoderLayerSANM(nn.Module):

    def __init__(
        self,
        in_size: int,
        size: int,
        self_attn: MultiHeadedAttentionSANM,
        feed_forward: "PositionwiseFeedForward",
        dropout_rate: float,
    ) -> None:
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = nn.LayerNorm(in_size, eps=1e-5)
        self.norm2 = nn.LayerNorm(size, eps=1e-5)
        self.dropout = nn.Dropout(dropout_rate)
        self.in_size = in_size
        self.size = size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        if self.in_size == self.size:
            x = residual + self.dropout(self.self_attn(x))
        else:
            # Input projection block (560→512): no residual on attention.
            x = self.dropout(self.self_attn(x))
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.feed_forward(x))
        return x


class PositionwiseFeedForward(nn.Module):
    """funasr PositionwiseFeedForward: w_2(relu(w_1(x)))."""

    def __init__(self, idim: int, hidden_units: int, dropout_rate: float) -> None:
        super().__init__()
        self.w_1 = nn.Linear(idim, hidden_units)
        self.w_2 = nn.Linear(hidden_units, idim)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


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
        sanm_shfit: int = 0,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self._output_size = output_size
        self.embed = SinusoidalPositionEncoder()

        def make_attn(in_feat: int) -> MultiHeadedAttentionSANM:
            return MultiHeadedAttentionSANM(
                attention_heads,
                in_feat,
                output_size,
                attention_dropout_rate,
                kernel_size,
                sanm_shfit,
            )

        def make_ffn() -> PositionwiseFeedForward:
            return PositionwiseFeedForward(output_size, linear_units, dropout_rate)

        # encoders0: 1 block, in_size=input_size (560) → output_size (512).
        self.encoders0 = nn.ModuleList(
            [
                EncoderLayerSANM(
                    input_size,
                    output_size,
                    make_attn(input_size),
                    make_ffn(),
                    dropout_rate,
                )
                for _ in range(1)
            ]
        )
        # encoders: num_blocks-1 blocks, 512 → 512.
        self.encoders = nn.ModuleList(
            [
                EncoderLayerSANM(
                    output_size,
                    output_size,
                    make_attn(output_size),
                    make_ffn(),
                    dropout_rate,
                )
                for _ in range(num_blocks - 1)
            ]
        )
        self.tp_encoders = nn.ModuleList(
            [
                EncoderLayerSANM(
                    output_size,
                    output_size,
                    make_attn(output_size),
                    make_ffn(),
                    dropout_rate,
                )
                for _ in range(tp_blocks)
            ]
        )
        self.after_norm = nn.LayerNorm(output_size, eps=1e-5)
        self.tp_norm = nn.LayerNorm(output_size, eps=1e-5)

    def output_size(self) -> int:
        return self._output_size

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        # xs: [B, T, input_size]. Scale by sqrt(output_size) then add sinusoidal PE.
        xs = xs * (self._output_size**0.5)
        xs = self.embed(xs)
        for layer in self.encoders0:
            xs = layer(xs)
        for layer in self.encoders:
            xs = layer(xs)
        xs = self.after_norm(xs)
        for layer in self.tp_encoders:
            xs = layer(xs)
        xs = self.tp_norm(xs)
        return xs


# ---------------------------------------------------------------------------
# Adaptor (funasr Transformer adaptor, downsample_rate=1)
# ---------------------------------------------------------------------------


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
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.size()
        q_h = self.linear_q(x).view(b, t, self.h, self.d_k).transpose(1, 2)
        k_h = self.linear_k(x).view(b, t, self.h, self.d_k).transpose(1, 2)
        v_h = self.linear_v(x).view(b, t, self.h, self.d_k).transpose(1, 2)
        q_h = q_h * (self.d_k**-0.5)
        scores = torch.matmul(q_h, k_h.transpose(-2, -1))
        attn = torch.softmax(scores, dim=-1)
        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, v_h)
        x = x.transpose(1, 2).contiguous().view(b, -1, self.h * self.d_k)
        return self.linear_out(x)


class AdaptorEncoderLayer(nn.Module):

    def __init__(
        self,
        size: int,
        self_attn: MultiHeadedAttention,
        feed_forward: PositionwiseFeedForward,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = nn.LayerNorm(size, eps=1e-5)
        self.norm2 = nn.LayerNorm(size, eps=1e-5)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = residual + self.dropout(self.self_attn(x))
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.feed_forward(x))
        return x


class FunAsrNanoAdaptor(nn.Module):

    def __init__(
        self,
        encoder_dim: int = 512,
        llm_dim: int = 1024,
        ffn_dim: int = 2048,
        num_layers: int = 2,
        attention_heads: int = 8,
        downsample_rate: int = 1,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim
        self.k = downsample_rate
        self.linear1 = nn.Linear(encoder_dim * self.k, ffn_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(ffn_dim, llm_dim)

        ffn_hidden = llm_dim // 4
        self.blocks = nn.ModuleList(
            [
                AdaptorEncoderLayer(
                    llm_dim,
                    MultiHeadedAttention(attention_heads, llm_dim, dropout_rate),
                    PositionwiseFeedForward(llm_dim, ffn_hidden, dropout_rate),
                    dropout_rate,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        for block in self.blocks:
            x = block(x)
        return x


# Final Model


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
        enc_cfg = config.audio_encoder_config
        adp_cfg = config.adaptor_config

        self.audio_encoder = FunAsrNanoAudioEncoder(
            input_size=enc_cfg.input_size,
            output_size=enc_cfg.output_size,
            attention_heads=enc_cfg.attention_heads,
            linear_units=enc_cfg.linear_units,
            num_blocks=enc_cfg.num_blocks,
            tp_blocks=enc_cfg.tp_blocks,
            kernel_size=enc_cfg.kernel_size,
            sanm_shfit=enc_cfg.sanm_shift,
            dropout_rate=enc_cfg.dropout_rate,
            attention_dropout_rate=enc_cfg.attention_dropout_rate,
        )
        self.audio_adaptor = FunAsrNanoAdaptor(
            encoder_dim=adp_cfg.encoder_dim,
            llm_dim=adp_cfg.llm_dim,
            ffn_dim=adp_cfg.ffn_dim,
            num_layers=adp_cfg.num_layers,
            attention_heads=adp_cfg.attention_heads,
            downsample_rate=adp_cfg.downsample_rate,
            dropout_rate=adp_cfg.dropout_rate,
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

        device = next(self.audio_encoder.parameters()).device
        dtype = next(self.audio_encoder.parameters()).dtype

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
            enc_out = self.audio_encoder(xs)  # [1, T, 512]
            adp_out = self.audio_adaptor(enc_out)  # [1, T, 1024]
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
            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue

            if getattr(self.config.text_config, "tie_word_embeddings", False) and (
                name == "lm_head.weight" or name.endswith(".lm_head.weight")
            ):
                continue

            if name.startswith("model.audio_encoder."):
                name = name.replace("model.audio_encoder.", "audio_encoder.", 1)
                is_llm = False
            elif name.startswith("model.audio_adaptor."):
                name = name.replace("model.audio_adaptor.", "audio_adaptor.", 1)
                is_llm = False
            elif name.startswith("model.language_model."):
                name = name.replace("model.language_model.", "language_model.model.", 1)
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
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)


EntryClass = FunAsrNanoForConditionalGeneration
