# SPDX-License-Identifier: Apache-2.0
"""SGLang-native Voxtral realtime ASR model.

This model mirrors the vLLM ``VoxtralRealtimeGeneration`` implementation:
- causal Whisper encoder (mel -> causal convs -> causal transformer)
- audio-language adapter
- learned time embedding for delay conditioning
- additive fusion of audio and text embeddings into a Mistral backbone

The encoder is run eagerly inside ``get_audio_feature`` so that the text
backbone can use SGLang's RadixAttention / CUDA graph paths unchanged.
Streaming KV-cache for the encoder is left as a future optimization; the
initial version processes each audio chunk in one full pass, which is
sufficient for offline/batched ASR and for realtime chunks that fit in a
single forward.
"""

from __future__ import annotations

import logging
import math
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from mistral_common.audio import mel_filter_bank
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import MultiModalityDataPaddingPatternMultimodalTokens
from sglang.srt.managers.schedule_batch import MultimodalDataItem, MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from torch import nn

from sglang_omni.models.voxtral_asr.model_config import (
    VoxtralAudioConfig,
    VoxtralRealtimeConfig,
)
from sglang_omni.models.voxtral_tts.sglang_model import VoxtralSGLangTextModel
from sglang_omni.models.voxtral_tts.voxtral_tts_audio_generation import (
    _interleave_qk_weight,
)

logger = logging.getLogger(__name__)


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding used for delay conditioning."""

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = torch.exp(
            -math.log(self.theta)
            * torch.arange(self.dim // 2).float()
            / (self.dim // 2)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t[..., None]
        inv_freq = self.inv_freq.to(device=t.device, dtype=t.dtype)
        emb = t * inv_freq
        return torch.cat((emb.cos(), emb.sin()), dim=-1)


class AudioLanguageAdapter(nn.Module):
    def __init__(self, hidden_size: int, dim: int) -> None:
        super().__init__()
        self.w_in = nn.Linear(hidden_size, dim, bias=False)
        self.gelu = nn.GELU()
        self.w_out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_out(self.gelu(self.w_in(x)))


class WhisperCausalConv1d(nn.Conv1d):
    """Causal 1-D convolution with asymmetric padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self._stride = self.stride[0]
        self._effective_kernel_size = (kernel_size - 1) * self.dilation[0] + 1
        self._padding_total = self._effective_kernel_size - self._stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_frames = (
            x.shape[-1] - self._effective_kernel_size + self._padding_total
        ) / self._stride + 1
        target_length = (math.ceil(n_frames) - 1) * self._stride + (
            self._effective_kernel_size - self._padding_total
        )
        extra_padding = target_length - x.shape[-1]
        x = F.pad(x, (self._padding_total, extra_padding), mode="constant")
        return super().forward(x)


def _make_causal_sliding_window_mask(
    seq_len: int,
    sliding_window: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Lower-triangular causal mask with a sliding window."""
    idx = torch.arange(seq_len, device=device)
    # causal mask
    mask = idx[None, :] <= idx[:, None]
    # sliding window mask
    if sliding_window is not None and sliding_window > 0:
        window_mask = idx[None, :] >= idx[:, None] - sliding_window
        mask = mask & window_mask
    # scaled_dot_product_attention expects a bool mask with True == attend.
    # (A float mask would be treated as an additive bias, silently disabling
    # the masking.)
    return mask


class WhisperCausalSelfAttention(nn.Module):
    """Causal self-attention with RoPE and optional sliding window.

    This is a simplified offline implementation: it computes the full
    attention matrix per chunk.  Streaming incremental decoding would need
    a KV cache and is intentionally left for future work.
    """

    def __init__(
        self,
        cfg: VoxtralAudioConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        del prefix
        self.embed_dim = cfg.dim
        self.num_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            self.embed_dim, self.num_heads * self.head_dim, bias=True
        )
        self.k_proj = nn.Linear(
            self.embed_dim, self.num_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.embed_dim, self.num_heads * self.head_dim, bias=True
        )
        self.out_proj = nn.Linear(
            self.num_heads * self.head_dim, self.embed_dim, bias=True
        )

        # RoPE with theta 1e6, matching vLLM WhisperCausalEncoder.
        from sglang_omni.vendor.sglang.layers import get_rope

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=cfg.max_source_positions * cfg.block_pool_size,
            base=1_000_000.0,
        )
        self.sliding_window = cfg.sliding_window

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Flatten for rotary embedding (expects [num_tokens, heads, head_dim])
        q_flat = q.transpose(1, 2).reshape(bsz * seq_len, self.num_heads, self.head_dim)
        k_flat = k.transpose(1, 2).reshape(bsz * seq_len, self.num_heads, self.head_dim)
        pos_flat = positions.reshape(-1)
        q_flat, k_flat = self.rotary_emb(pos_flat, q_flat, k_flat)
        q = q_flat.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k_flat.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        mask = _make_causal_sliding_window_mask(
            seq_len,
            self.sliding_window,
            hidden_states.device,
            hidden_states.dtype,
        )
        # Expand mask for broadcast over heads and batch
        mask = mask.unsqueeze(0).unsqueeze(0)

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=0.0,
            scale=self.scaling,
        )
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(bsz, seq_len, self.num_heads * self.head_dim)
        )
        return self.out_proj(attn_output)


class WhisperCausalEncoderLayer(nn.Module):
    def __init__(self, cfg: VoxtralAudioConfig) -> None:
        super().__init__()
        from sglang_omni.vendor.sglang.layers import RMSNorm

        self.self_attn = WhisperCausalSelfAttention(cfg)
        self.self_attn_layer_norm = RMSNorm(cfg.dim, eps=1e-5)

        self.gate_proj = nn.Linear(cfg.dim, cfg.hidden_dim, bias=False)
        self.up_proj = nn.Linear(cfg.dim, cfg.hidden_dim, bias=False)
        self.down_proj = nn.Linear(cfg.hidden_dim, cfg.dim, bias=True)
        self.final_layer_norm = RMSNorm(cfg.dim, eps=1e-5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, positions)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        hidden_states = F.silu(gate) * up
        hidden_states = self.down_proj(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class VoxtralRealtimeEncoder(nn.Module):
    """Causal Whisper encoder for Voxtral realtime ASR."""

    def __init__(self, cfg: VoxtralAudioConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_mel_bins = cfg.audio_encoding_args.num_mel_bins
        self.max_source_positions = cfg.max_source_positions
        self.downsample_factor = cfg.downsample_factor
        self.block_pool_size = cfg.block_pool_size

        self.conv1 = WhisperCausalConv1d(self.num_mel_bins, cfg.dim, kernel_size=3)
        self.conv2 = WhisperCausalConv1d(cfg.dim, cfg.dim, kernel_size=3, stride=2)
        self.total_stride = self.conv1._stride * self.conv2._stride

        self.layers = nn.ModuleList(
            [WhisperCausalEncoderLayer(cfg) for _ in range(cfg.n_layers)]
        )
        from sglang_omni.vendor.sglang.layers import RMSNorm

        self.layer_norm = RMSNorm(cfg.dim, eps=1e-5)

        mel_filters = mel_filter_bank(
            num_frequency_bins=1 + cfg.audio_encoding_args.window_size // 2,
            num_mel_bins=self.num_mel_bins,
            min_frequency=0.0,
            max_frequency=8000.0,
            sampling_rate=cfg.audio_encoding_args.sampling_rate,
        )
        self.register_buffer(
            "mel_filters",
            torch.tensor(mel_filters, dtype=torch.float32),
            persistent=False,
        )

    def compute_melspec(self, audio_waveforms: torch.Tensor) -> torch.Tensor:
        input_dtype = audio_waveforms.dtype
        window = torch.hann_window(
            self.cfg.audio_encoding_args.window_size,
            device=audio_waveforms.device,
        )
        stft = torch.stft(
            audio_waveforms,
            self.cfg.audio_encoding_args.window_size,
            self.cfg.audio_encoding_args.hop_length,
            window=window,
            return_complex=True,
        )
        magnitudes = stft[..., :-1].abs() ** 2
        mel_spec = self.mel_filters.T.to(audio_waveforms.device) @ magnitudes
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()

        global_log_mel_max = self.cfg.audio_encoding_args.global_log_mel_max
        if global_log_mel_max is not None:
            log_spec_max = torch.tensor(
                global_log_mel_max,
                device=log_spec.device,
                dtype=log_spec.dtype,
            )
        else:
            log_spec_max = log_spec.max()

        log_spec = torch.maximum(log_spec, log_spec_max - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec.to(input_dtype)

    def forward_conv(self, input_features: torch.Tensor) -> torch.Tensor:
        """Run causal convolutions and return post-conv frame embeddings."""
        hidden_states = F.gelu(self.conv1(input_features))
        hidden_states = F.gelu(self.conv2(hidden_states))
        hidden_states = hidden_states.transpose(-1, -2).to(hidden_states.dtype)
        return hidden_states

    def forward(
        self,
        post_conv_embeds: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Run causal transformer on post-conv embeddings."""
        hidden_states = post_conv_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states, positions)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class VoxtralRealtimeForConditionalGeneration(nn.Module):
    """Voxtral realtime ASR with SGLang-managed text KV cache."""

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
        config: VoxtralRealtimeConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        del config, quant_config, prefix
        server_args = __import__(
            "sglang.srt.server_args", fromlist=["get_global_server_args"]
        ).get_global_server_args()
        self.model_path = server_args.model_path
        self.config = VoxtralRealtimeConfig.from_model_path(self.model_path)
        self.text_cfg = self.config.text_config
        self.audio_cfg = self.config.audio_config

        self.language_model = VoxtralSGLangTextModel(self.text_cfg)
        self.whisper_encoder = VoxtralRealtimeEncoder(self.audio_cfg)
        self.audio_language_adapter = AudioLanguageAdapter(
            hidden_size=self.audio_cfg.dim * self.audio_cfg.downsample_factor,
            dim=self.text_cfg.dim,
        )
        self.time_embedding = TimeEmbedding(dim=self.text_cfg.dim)

        # Default delay tokens (480ms / 80ms per token = 6).  This can be
        # overridden per-request via request metadata in the future.
        self.n_delay_tokens = 6

        self.pattern = MultiModalityDataPaddingPatternMultimodalTokens()

        # Per-request audio embedding table for decode-time injection.
        # vLLM realtime semantics: sequence position i consumes
        # audio_embeds[i]; audio keeps being added to text embeddings on
        # every decode step until the audio stream is exhausted.
        # sglang's ReqToTokenPool hands out slots in [1, max_running_requests]
        # (slot 0 is reserved as a harmless padding row), so allocate one extra.
        max_reqs = (server_args.max_running_requests or 16) + 1
        max_audio_len = (
            self.audio_cfg.max_source_positions * self.audio_cfg.block_pool_size
        )
        self._max_audio_len = max_audio_len
        self.register_buffer(
            "_audio_embed_table",
            torch.zeros(
                max_reqs, max_audio_len, self.text_cfg.dim, dtype=torch.bfloat16
            ),
            persistent=False,
        )
        self.register_buffer(
            "_audio_embed_lens",
            torch.zeros(max_reqs, dtype=torch.long),
            persistent=False,
        )

    def get_input_embeddings(self):
        return self.language_model.embed_tokens

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        return self.pattern.pad_input_tokens(input_ids, mm_inputs)

    def _compute_audio_features(
        self,
        audio_arrays: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Compute per-sample audio embeddings from raw waveforms."""
        with torch.inference_mode():
            return self._compute_audio_features_impl(audio_arrays)

    def _compute_audio_features_impl(
        self,
        audio_arrays: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        device = next(self.whisper_encoder.parameters()).device
        dtype = next(self.whisper_encoder.parameters()).dtype

        post_conv_per_sample: List[torch.Tensor] = []
        for audio in audio_arrays:
            # audio: [num_samples]
            mel = self.whisper_encoder.compute_melspec(audio.to(device=device))
            # mel: [num_mel_bins, num_frames]
            # make divisible by 2 for conv2 stride
            if mel.shape[-1] % 2 != 0:
                mel = mel[:, 1:]
            post_conv = self.whisper_encoder.forward_conv(mel.unsqueeze(0).to(dtype))
            post_conv = post_conv.squeeze(0)  # [num_frames, dim]
            # make divisible by block_pool_size
            pool_size = self.audio_cfg.block_pool_size
            remainder = post_conv.shape[0] % pool_size
            if remainder != 0:
                post_conv = post_conv[remainder:]
            post_conv_per_sample.append(post_conv)

        # Run the transformer in small chunks: eager attention materializes
        # a (bs, heads, seq, seq) matrix which OOMs when many requests are
        # batched.  Samples are independent (causal mask, no cross-sample
        # interaction), so chunking does not change numerics.
        seq_lens = [x.shape[0] for x in post_conv_per_sample]
        downsample = self.audio_cfg.downsample_factor
        chunk_size = 4
        results = []
        for start in range(0, len(post_conv_per_sample), chunk_size):
            chunk = post_conv_per_sample[start : start + chunk_size]
            chunk_lens = seq_lens[start : start + chunk_size]
            chunk_max = max(chunk_lens)
            padded = torch.zeros(
                len(chunk),
                chunk_max,
                self.audio_cfg.dim,
                device=device,
                dtype=dtype,
            )
            for i, pc in enumerate(chunk):
                padded[i, : chunk_lens[i]] = pc

            positions = torch.arange(chunk_max, device=device, dtype=torch.long)
            positions = positions.unsqueeze(0).expand(len(chunk), -1)

            transformer_out = self.whisper_encoder(padded, positions)

            # Downsample by downsample_factor and project.
            for i, length in enumerate(chunk_lens):
                hidden = transformer_out[i, :length]  # [length, dim]
                if hidden.shape[0] % downsample != 0:
                    pad_len = downsample - (hidden.shape[0] % downsample)
                    hidden = F.pad(hidden, (0, 0, 0, pad_len))
                hidden = hidden.view(
                    hidden.shape[0] // downsample,
                    hidden.shape[1] * downsample,
                )
                adapted = self.audio_language_adapter(hidden)
                results.append(adapted)
        return results

    def get_audio_feature(
        self,
        items: List[MultimodalDataItem],
        forward_batch: ForwardBatch,  # noqa: ARG002
    ) -> torch.Tensor:
        """Return concatenated audio embeddings for all items."""
        audio_arrays = []
        for item in items:
            feature = item.feature
            if isinstance(feature, np.ndarray):
                feature = torch.from_numpy(feature)
            audio_arrays.append(feature)
        audio_embeds = self._compute_audio_features(audio_arrays)
        return torch.cat(audio_embeds, dim=0)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
    ) -> LogitsProcessorOutput:
        if input_embeds is None:
            text_embeds = self.language_model.embed_tokens(input_ids)
        else:
            text_embeds = input_embeds

        if forward_batch.forward_mode.is_extend():
            if forward_batch.contains_audio_inputs():
                mm_inputs = forward_batch.merge_mm_inputs()
                audio_items = [item for item in mm_inputs.mm_items if item.is_audio()]
                audio_arrays = []
                for item in audio_items:
                    feature = item.feature
                    if isinstance(feature, np.ndarray):
                        feature = torch.from_numpy(feature)
                    audio_arrays.append(feature)
                per_req_embeds = self._compute_audio_features(audio_arrays)
                logger.info(
                    "Voxtral ASR prefill: text_embeds=%s per_req_audio=%s",
                    tuple(text_embeds.shape),
                    [tuple(e.shape) for e in per_req_embeds],
                )
                text_embeds = self._add_audio_to_text_embeds(
                    text_embeds, per_req_embeds, forward_batch
                )
            else:
                logger.warning(
                    "No audio inputs found in forward_batch for Voxtral ASR prefill."
                )
        else:
            # Decode: keep injecting audio embeddings at the current position.
            # vLLM realtime semantics: sequence position i consumes
            # audio_embeds[i]; positions beyond the audio length add zero.
            req_slots = forward_batch.req_pool_indices.clamp(
                max=self._audio_embed_table.shape[0] - 1
            )
            pos = positions.clamp(max=self._max_audio_len - 1)
            audio = self._audio_embed_table[req_slots, pos]
            valid = positions < self._audio_embed_lens[req_slots]
            text_embeds = text_embeds + audio.to(text_embeds.dtype) * valid.unsqueeze(
                -1
            ).to(text_embeds.dtype)

        # Add delay/time conditioning via ada RMS norm in each decoder layer.
        time_tensor = torch.full(
            (1,),
            fill_value=self.n_delay_tokens,
            device=text_embeds.device,
            dtype=text_embeds.dtype,
        )
        t_cond = self.time_embedding(time_tensor)

        return self._run_language_model_and_logits(
            input_ids, positions, text_embeds, forward_batch, t_cond=t_cond
        )

    def _add_audio_to_text_embeds(
        self,
        text_embeds: torch.Tensor,
        per_req_embeds: List[torch.Tensor],
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """Add audio embeddings over the whole prompt span (vLLM start_idx=0).

        Also persists the full per-request audio embeddings into
        ``_audio_embed_table`` so decode steps can keep adding them at their
        absolute sequence positions.
        """
        extend_seq_lens = forward_batch.extend_seq_lens
        extend_prefix_lens = forward_batch.extend_prefix_lens
        if extend_seq_lens is None:
            request_starts = [0]
            request_lens = [text_embeds.shape[0]]
            prefix_lens = [0]
        else:
            request_lens = extend_seq_lens.tolist()
            if extend_prefix_lens is not None:
                prefix_lens = extend_prefix_lens.tolist()
            else:
                prefix_lens = [0] * len(request_lens)
            request_starts = [0]
            for length in request_lens[:-1]:
                request_starts.append(request_starts[-1] + length)

        if len(per_req_embeds) != len(request_starts):
            logger.warning(
                "Audio item count (%d) != request count (%d); cannot scatter audio.",
                len(per_req_embeds),
                len(request_starts),
            )
            return text_embeds

        req_slots = forward_batch.req_pool_indices.tolist()
        for embeds, req_start, req_len, prefix_len, slot in zip(
            per_req_embeds, request_starts, request_lens, prefix_lens, req_slots
        ):
            n = min(embeds.shape[0], self._max_audio_len)
            self._audio_embed_table[slot, :n] = embeds[:n].to(
                self._audio_embed_table.dtype
            )
            self._audio_embed_lens[slot] = n

            # Add audio at ABSOLUTE positions: with prefix-cache hits or
            # chunked prefill, the extended tokens may start mid-sequence.
            abs_pos = torch.arange(
                prefix_len,
                prefix_len + req_len,
                device=embeds.device,
            )
            valid = abs_pos < n
            gather_pos = abs_pos.clamp(max=max(n - 1, 0))
            add = embeds[gather_pos].to(text_embeds.dtype) * valid.unsqueeze(-1).to(
                text_embeds.dtype
            )
            text_embeds[req_start : req_start + req_len] = (
                text_embeds[req_start : req_start + req_len] + add
            )
            logger.info(
                "Voxtral ASR scatter: slot=%d req_start=%d req_len=%d prefix=%d audio=%d",
                slot,
                req_start,
                req_len,
                prefix_len,
                n,
            )

        return text_embeds

    def _run_language_model_and_logits(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        text_embeds: torch.Tensor,
        forward_batch: ForwardBatch,
        t_cond: torch.Tensor | None = None,
    ) -> LogitsProcessorOutput:
        hidden_states = self.language_model(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=text_embeds,
            t_cond=t_cond,
        )

        # Compute next-token logits from the last position of each sequence.
        # Tied embeddings: use the embedding weight as the output projection.
        if forward_batch.forward_mode.is_extend():
            last_index = self._extend_last_index(forward_batch, hidden_states.device)
            logits_hidden = hidden_states[last_index]
        else:
            logits_hidden = hidden_states

        next_token_logits = F.linear(
            logits_hidden,
            self.language_model.embed_tokens.weight,
        )

        return LogitsProcessorOutput(
            next_token_logits=next_token_logits,
            hidden_states=hidden_states,
        )

    @staticmethod
    def _extend_last_index(
        forward_batch: ForwardBatch,
        device: torch.device,
    ) -> torch.Tensor:
        extend_seq_lens = forward_batch.extend_seq_lens
        if extend_seq_lens is None:
            return torch.tensor([forward_batch.input_ids.shape[0] - 1], device=device)
        return torch.cumsum(extend_seq_lens.to(device=device), dim=0) - 1

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        params = dict(self.named_parameters())

        # Remap Mistral-format weight names to our module names.
        mistral_remapping = [
            (r"mm_streams_embeddings\.embedding_module\.(.*)", r"\1"),
            (r"mm_whisper_embeddings\.(.*)", r"\1"),
            (
                r"audio_language_projection\.(.*)",
                r"audio_language_adapter.\1",
            ),
            (
                r"audio_language_adapter\.0\.weight",
                r"audio_language_adapter.w_in.weight",
            ),
            (
                r"audio_language_adapter\.2\.weight",
                r"audio_language_adapter.w_out.weight",
            ),
            (
                r"whisper_encoder\.conv_layers\.0\.(weight|bias)",
                r"whisper_encoder.conv1.\1",
            ),
            (
                r"whisper_encoder\.conv_layers\.1\.(weight|bias)",
                r"whisper_encoder.conv2.\1",
            ),
            (
                r"whisper_encoder\.transformer\.layers\.(\d+)\.attention\.w([qkv])\.(weight|bias)",
                r"whisper_encoder.layers.\1.self_attn.\2_proj.\3",
            ),
            (
                r"whisper_encoder\.transformer\.layers\.(\d+)\.attention\.wo\.(weight|bias)",
                r"whisper_encoder.layers.\1.self_attn.out_proj.\2",
            ),
            (
                r"whisper_encoder\.transformer\.layers\.(\d+)\.attention_norm\.(weight|bias)",
                r"whisper_encoder.layers.\1.self_attn_layer_norm.\2",
            ),
            (
                r"whisper_encoder\.transformer\.layers\.(\d+)\.feed_forward\.w1\.(weight|bias)",
                r"whisper_encoder.layers.\1.gate_proj.\2",
            ),
            (
                r"whisper_encoder\.transformer\.layers\.(\d+)\.feed_forward\.w3\.(weight|bias)",
                r"whisper_encoder.layers.\1.up_proj.\2",
            ),
            (
                r"whisper_encoder\.transformer\.layers\.(\d+)\.feed_forward\.w2\.(weight|bias)",
                r"whisper_encoder.layers.\1.down_proj.\2",
            ),
            (
                r"whisper_encoder\.transformer\.layers\.(\d+)\.ffn_norm\.(weight|bias)",
                r"whisper_encoder.layers.\1.final_layer_norm.\2",
            ),
            (
                r"whisper_encoder\.transformer\.norm\.(weight|bias)",
                r"whisper_encoder.layer_norm.\2",
            ),
        ]

        import re

        num_loaded = 0
        for name, loaded_weight in weights:
            original_name = name

            # HF-format checkpoint (model.safetensors) uses prefixed names:
            #   language_model.model.<mistral-ish names>
            #   audio_tower.<hf whisper-ish names>
            #   multi_modal_projector.linear_{1,2}.weight
            if name.startswith("language_model.model."):
                if self._load_text_weight(
                    name[len("language_model.model.") :], loaded_weight, params
                ):
                    num_loaded += 1
                else:
                    logger.warning("Skipping unmatched LLM weight: %s", original_name)
                continue
            if name.startswith("audio_tower."):
                target = "whisper_encoder." + self._remap_audio_suffix(
                    name[len("audio_tower.") :]
                )
                if self._copy_weight(target, loaded_weight, params):
                    num_loaded += 1
                else:
                    logger.warning(
                        "Skipping unmatched audio weight: %s -> %s",
                        original_name,
                        target,
                    )
                continue
            if name.startswith("multi_modal_projector."):
                target = {
                    "linear_1.weight": "audio_language_adapter.w_in.weight",
                    "linear_2.weight": "audio_language_adapter.w_out.weight",
                }.get(name[len("multi_modal_projector.") :])
                if target is not None and self._copy_weight(
                    target, loaded_weight, params
                ):
                    num_loaded += 1
                else:
                    logger.warning(
                        "Skipping unmatched projector weight: %s", original_name
                    )
                continue

            # Remap Mistral-format names to our module names first.
            for pattern, repl in mistral_remapping:
                if re.fullmatch(pattern, name):
                    name = re.sub(pattern, repl, name)
                    break

            # Language model: handle stacked qkv/gate_up via VoxtralSGLangTextModel.
            if not name.startswith("whisper_encoder") and not name.startswith(
                "audio_language_adapter"
            ):
                if self._load_text_weight(name, loaded_weight, params):
                    num_loaded += 1
                    continue

            if name.endswith(".bias") and name not in params:
                continue
            if name not in params:
                logger.warning("Skipping unmatched weight: %s", original_name)
                continue

            param = params[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            num_loaded += 1

        logger.info("Voxtral ASR load_weights: loaded %d tensors", num_loaded)

    @staticmethod
    def _remap_audio_suffix(suffix: str) -> str:
        """Map HF audio_tower names onto VoxtralRealtimeEncoder module names."""
        suffix = suffix.replace("embedder.conv1.", "conv1.")
        suffix = suffix.replace("embedder.conv2.", "conv2.")
        if suffix.startswith("norm."):
            suffix = "layer_norm." + suffix[len("norm.") :]
        suffix = suffix.replace(".self_attn.o_proj.", ".self_attn.out_proj.")
        suffix = suffix.replace(".mlp.", ".")
        return suffix

    def _load_text_weight(
        self,
        name: str,
        loaded_weight: torch.Tensor,
        params: dict[str, nn.Parameter],
    ) -> bool:
        """Load text backbone weights with Mistral->SGLang remapping."""
        import re

        if name == "norm.weight":
            return self._copy_weight(
                "language_model.norm.weight", loaded_weight, params
            )
        if name in ("tok_embeddings.weight", "embed_tokens.weight"):
            return self._copy_weight(
                "language_model.embed_tokens.weight", loaded_weight, params
            )

        match = re.match(r"^layers\.(\d+)\.(.+)$", name)
        if match is None:
            return False
        layer_idx, suffix = match.group(1), match.group(2)
        prefix = f"language_model.layers.{layer_idx}"
        # Mistral consolidated naming (requires q/k interleave for RoPE layout).
        if suffix == "attention.wq.weight":
            return self._load_qkv(prefix, "q", loaded_weight, params, interleave=True)
        if suffix == "attention.wk.weight":
            return self._load_qkv(prefix, "k", loaded_weight, params, interleave=True)
        if suffix == "attention.wv.weight":
            return self._load_qkv(prefix, "v", loaded_weight, params, interleave=False)
        # HF naming (model.safetensors): q/k already use the HF RoPE layout.
        if suffix == "self_attn.q_proj.weight":
            return self._load_qkv(prefix, "q", loaded_weight, params, interleave=False)
        if suffix == "self_attn.k_proj.weight":
            return self._load_qkv(prefix, "k", loaded_weight, params, interleave=False)
        if suffix == "self_attn.v_proj.weight":
            return self._load_qkv(prefix, "v", loaded_weight, params, interleave=False)
        mapping = {
            "attention.wo.weight": "self_attn.o_proj.weight",
            "self_attn.o_proj.weight": "self_attn.o_proj.weight",
            "attention_norm.weight": "attention_norm.weight",
            "input_layernorm.weight": "attention_norm.weight",
            "ffn_norm.weight": "ffn_norm.weight",
            "post_attention_layernorm.weight": "ffn_norm.weight",
            "feed_forward.w1.weight": ("gate_up_proj.weight", 0),
            "mlp.gate_proj.weight": ("gate_up_proj.weight", 0),
            "feed_forward.w3.weight": ("gate_up_proj.weight", 1),
            "mlp.up_proj.weight": ("gate_up_proj.weight", 1),
            "feed_forward.w2.weight": "down_proj.weight",
            "mlp.down_proj.weight": "down_proj.weight",
            "ada_rms_norm.linear1.weight": "ada_rms_norm_t_cond.0.weight",
            "ada_rms_norm.linear2.weight": "ada_rms_norm_t_cond.2.weight",
        }
        target = mapping.get(suffix)
        if target is None:
            return False
        if isinstance(target, tuple):
            target_name, shard_id = target
            param = params[f"{prefix}.{target_name}"]
            param.weight_loader(param, loaded_weight, shard_id)
            return True
        return self._copy_weight(f"{prefix}.{target}", loaded_weight, params)

    def _load_qkv(
        self,
        prefix: str,
        shard_id: str,
        loaded_weight: torch.Tensor,
        params: dict[str, nn.Parameter],
        interleave: bool,
    ) -> bool:
        param = params[f"{prefix}.self_attn.qkv_proj.weight"]
        if interleave:
            layer = self.language_model.layers[int(prefix.split(".")[-1])]
            if shard_id == "q":
                loaded_weight = _interleave_qk_weight(
                    loaded_weight,
                    layer.self_attn.num_heads,
                    layer.self_attn.head_dim,
                )
            elif shard_id == "k":
                loaded_weight = _interleave_qk_weight(
                    loaded_weight,
                    layer.self_attn.num_kv_heads,
                    layer.self_attn.head_dim,
                )
        param.weight_loader(param, loaded_weight, shard_id)
        return True

    @staticmethod
    def _copy_weight(
        name: str,
        loaded_weight: torch.Tensor,
        params: dict[str, nn.Parameter],
    ) -> bool:
        if name not in params:
            return False
        param = params[name]
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, loaded_weight)
        return True


EntryClass = VoxtralRealtimeForConditionalGeneration
