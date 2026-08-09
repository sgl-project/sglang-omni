# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import math
from typing import Any, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
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


def _sanm_mask_from_lengths(
    lengths: torch.Tensor, max_len: int, *, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    # note (guozhihao): SenseVoice pad mask [B, 1, T], 1=valid.
    idx = torch.arange(max_len, device=device).unsqueeze(0)
    return (idx < lengths.unsqueeze(1)).to(dtype=dtype).unsqueeze(1)


def _apply_time_mask(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return x
    return x * mask.transpose(1, 2)


def _additive_key_pad_mask(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    # note (guozhihao): SenseVoice [B, 1, T] (1=valid) -> SDPA additive [B, 1, 1, T].
    return torch.zeros(
        mask.shape[0], 1, 1, mask.shape[-1], device=mask.device, dtype=dtype
    ).masked_fill(mask.unsqueeze(1).eq(0), torch.finfo(dtype).min)


def _fused_qkv_project(
    x: torch.Tensor,
    q_proj: nn.Linear,
    k_proj: nn.Linear,
    v_proj: nn.Linear,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # note (guozhihao): keep separate q/k/v Linears for HF checkpoint names;
    # fuse into one GEMM at runtime.
    weight = torch.cat([q_proj.weight, k_proj.weight, v_proj.weight], dim=0)
    bias = None
    if q_proj.bias is not None:
        bias = torch.cat([q_proj.bias, k_proj.bias, v_proj.bias], dim=0)
    return F.linear(x, weight, bias).chunk(3, dim=-1)


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
        self.attn_dropout_p = float(dropout_rate)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Returns (attn_out, v) so FSMN can reuse the same value projection.
        b, t, _ = x.size()
        q, k, v = _fused_qkv_project(x, self.q_proj, self.k_proj, self.v_proj)
        q_h = q.view(b, t, self.h, self.d_k).transpose(1, 2)
        k_h = k.view(b, t, self.h, self.d_k).transpose(1, 2)
        v_h = v.view(b, t, self.h, self.d_k).transpose(1, 2)

        attn_mask = None if mask is None else _additive_key_pad_mask(mask, q.dtype)
        dropout_p = self.attn_dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q_h,
            k_h,
            v_h,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=False,
        )
        out = out.transpose(1, 2).contiguous().view(b, t, self.h * self.d_k)
        return self.out_proj(out), v


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

    def forward(
        self, value_states: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # note (guozhihao): zero pad frames before/after the depthwise conv so
        # kernel_size windows cannot leak padded values into valid frames.
        value_states = _apply_time_mask(value_states, mask)
        hidden_states = self.conv(self.pad(value_states.transpose(1, 2)))
        hidden_states = hidden_states.transpose(1, 2) + value_states
        hidden_states = self.dropout(hidden_states)
        return _apply_time_mask(hidden_states, mask)


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

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        residual = x
        x = self.self_attn_layer_norm(x)
        # note (guozhihao): attn returns v so FSMN does not recompute v_proj.
        attn_out, value_states = self.self_attn(x, mask)
        x = self.dropout(attn_out + self.fsmn(value_states, mask))
        x = _apply_time_mask(x, mask)
        if self.in_size == self.size:
            x = residual + x
        residual = x
        x = self.final_layer_norm(x)
        x = self.activation_dropout(self.activation(self.fc1(x)))
        x = residual + self.dropout(self.fc2(x))
        x = _apply_time_mask(x, mask)
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

    def forward(
        self, xs: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        xs = xs * (self._output_size**0.5)
        xs = self.embed(xs)
        xs = self.stem(xs, mask)
        for layer in self.layers:
            xs = layer(xs, mask)
        xs = self.layer_norm(xs)
        xs = _apply_time_mask(xs, mask)
        for layer in self.timestamp_prediction_layers:
            xs = layer(xs, mask)
        xs = self.timestamp_prediction_layer_norm(xs)
        xs = _apply_time_mask(xs, mask)
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
        self.attn_dropout_p = float(dropout_rate)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        b, t, _ = x.size()
        q, k, v = _fused_qkv_project(x, self.q_proj, self.k_proj, self.v_proj)
        q_h = q.view(b, t, self.h, self.d_k).transpose(1, 2)
        k_h = k.view(b, t, self.h, self.d_k).transpose(1, 2)
        v_h = v.view(b, t, self.h, self.d_k).transpose(1, 2)

        attn_mask = None if mask is None else _additive_key_pad_mask(mask, q.dtype)
        dropout_p = self.attn_dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q_h,
            k_h,
            v_h,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=False,
        )
        out = out.transpose(1, 2).contiguous().view(b, t, self.h * self.d_k)
        return self.out_proj(out)


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

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        residual = x
        x = self.self_attn_layer_norm(x)
        x = residual + self.dropout(self.self_attn(x, mask))
        x = _apply_time_mask(x, mask)
        residual = x
        x = self.final_layer_norm(x)
        x = residual + self.dropout(self.fc2(self.activation(self.fc1(x))))
        return _apply_time_mask(x, mask)


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

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        x = _apply_time_mask(x, mask)
        for block in self.blocks:
            x = block(x, mask)
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
        # note (guozhihao): pad+mask batch-encode to [sum_tokens, llm_dim] in order.
        if not items:
            raise ValueError(
                "Fun-ASR get_audio_feature requires at least one audio item"
            )
        device = next(self.audio_tower.parameters()).device
        dtype = next(self.audio_tower.parameters()).dtype

        feats: List[torch.Tensor] = []
        lengths: List[int] = []
        for item in items:
            if item.feature is None:
                raise ValueError(
                    "Fun-ASR audio item is missing feature (input_features); "
                    "cannot encode"
                )
            feature = item.feature
            if feature.ndim != 3 or feature.shape[0] != 1:
                raise ValueError(
                    "Fun-ASR expects item.feature shaped [1, input_size, T], "
                    f"got {tuple(feature.shape)}"
                )
            mask = getattr(item, "feature_attention_mask", None)
            if mask is not None:
                valid = int(mask.sum().item())
            else:
                valid = int(feature.shape[-1])
            valid = max(valid, 1)
            feats.append(feature[:, :, :valid])
            lengths.append(valid)

        batch_size = len(feats)
        feat_dim = feats[0].shape[1]
        t_max = max(lengths)
        batched = feats[0].new_zeros(batch_size, feat_dim, t_max)
        for i, (feat, length) in enumerate(zip(feats, lengths)):
            batched[i, :, :length] = feat[0, :, :length]

        xs = (
            batched.permute(0, 2, 1)
            .contiguous()
            .to(device=device, dtype=dtype, non_blocking=True)
        )

        adp_out: Optional[torch.Tensor] = None
        graph_runner = getattr(self, "encoder_cuda_graph_runner", None)
        if graph_runner is not None:
            # Bucketed capture/replay; always masked. Returns None when no
            # bucket fits (falls through to the eager path below).
            adp_out = graph_runner.run(xs, lengths)

        if adp_out is None:
            # note (guozhihao): skip masking for the common B=1 unpadded path
            # so it stays numerically equivalent to the unmasked encoder
            # forward.
            if batch_size == 1 and lengths[0] == t_max:
                sanm_mask: Optional[torch.Tensor] = None
            else:
                ilens = torch.tensor(lengths, device=device, dtype=torch.long)
                sanm_mask = _sanm_mask_from_lengths(
                    ilens, t_max, dtype=xs.dtype, device=device
                )

            enc_out = self.audio_tower(xs, sanm_mask)
            adp_out = self.multi_modal_projector(enc_out, sanm_mask)

        embeddings: List[torch.Tensor] = []
        for b, length in enumerate(lengths):
            num_tokens = max(int(fun_asr_low_frame_rate_length(length)), 1)
            embeddings.append(adp_out[b, :num_tokens, :])
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
