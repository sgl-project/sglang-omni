"""VoxtralTTSAudioTokenizer (decoder-only) for Voxtral TTS.
"""

from __future__ import annotations

import logging
import math
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from sglang_omni_v1.models.voxtral_tts.acoustic_transformer import (
    FeedForward,
    MultimodalAudioModelArgs,
    from_nested_dict,
)

try:
    from flash_attn import flash_attn_func

    HAS_FLASH_ATTN = True
except ImportError:
    flash_attn_func = None
    HAS_FLASH_ATTN = False

logger = logging.getLogger(__name__)
rms_norm = torch.nn.RMSNorm
weight_norm = torch.nn.utils.parametrizations.weight_norm

CODEC_NORM_EPS = 1e-2


@dataclass
class AudioTokenizerArgs:
    channels: int = 1
    sampling_rate: int = 24000
    pretransform_patch_size: int = 240
    patch_proj_kernel_size: int = 7
    semantic_codebook_size: int = 8192
    semantic_dim: int = 256
    acoustic_codebook_size: int = 21
    acoustic_dim: int = 36
    conv_weight_norm: bool = True
    causal: bool = True
    attn_sliding_window_size: int = 16
    half_attn_window_upon_downsampling: bool = True
    dim: int = 1024
    hidden_dim: int = 4096
    head_dim: int = 128
    n_heads: int = 8
    n_kv_heads: int = 8
    qk_norm_eps: float = 1e-6
    qk_norm: bool = True
    use_biases: bool = False
    norm_eps: float = 1e-2
    layer_scale: bool = True
    layer_scale_init: float | None = None
    encoder_transformer_lengths_str: str = "2,2,2,2"
    encoder_convs_kernels_str: str = "4,4,4,3"
    encoder_convs_strides_str: str = "2,2,2,1"
    decoder_transformer_lengths_str: str = "2,2,2,2"
    decoder_convs_kernels_str: str = "3,4,4,4"
    decoder_convs_strides_str: str = "1,2,2,2"

    def __post_init__(self) -> None:
        assert (
            len(self.encoder_transformer_lengths)
            == len(self.encoder_convs_kernels)
            == len(self.encoder_convs_strides)
        )
        assert (
            len(self.decoder_transformer_lengths)
            == len(self.decoder_convs_kernels)
            == len(self.decoder_convs_strides)
        )

    def _str2list(self, s: str) -> tuple[int, ...]:
        return tuple(int(i) for i in s.split(","))

    @property
    def encoder_transformer_lengths(self) -> tuple[int, ...]:
        return self._str2list(self.encoder_transformer_lengths_str)

    @property
    def encoder_convs_kernels(self) -> tuple[int, ...]:
        return self._str2list(self.encoder_convs_kernels_str)

    @property
    def encoder_convs_strides(self) -> tuple[int, ...]:
        return self._str2list(self.encoder_convs_strides_str)

    @property
    def decoder_transformer_lengths(self) -> tuple[int, ...]:
        return self._str2list(self.decoder_transformer_lengths_str)

    @property
    def decoder_convs_kernels(self) -> tuple[int, ...]:
        return self._str2list(self.decoder_convs_kernels_str)

    @property
    def decoder_convs_strides(self) -> tuple[int, ...]:
        return self._str2list(self.decoder_convs_strides_str)

    @property
    def frame_rate(self) -> float:
        return self.sampling_rate / (
            self.pretransform_patch_size * math.prod(self.encoder_convs_strides)
        )


# ---- Codebooks ----


class SemanticCodebook(nn.Module):
    def __init__(self, codebook_size: int, codebook_dim: int) -> None:
        super().__init__()
        self.epsilon = 1e-5
        self.register_buffer("cluster_usage", torch.ones(codebook_size))
        self.register_buffer("embedding_sum", torch.zeros(codebook_size, codebook_dim))
        self.register_buffer("_embedding", None, persistent=False)

    @property
    def embedding(self) -> torch.Tensor:
        if self._embedding is None:
            embedding = (
                self.embedding_sum / self.cluster_usage.clamp(min=self.epsilon)[:, None]
            )
            self.register_buffer("_embedding", embedding, persistent=False)
            return embedding
        return self._embedding

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        codes = codes.squeeze(1)
        embedding = self.embedding.to(codes.device)
        quantized = F.embedding(codes, embedding)
        return rearrange(quantized, "b t d -> b d t")

    @property
    def num_codebooks(self) -> int:
        return 1

    @property
    def codebook_sizes(self) -> list[int]:
        return [self.embedding_sum.shape[0]]


class AcousticCodebook(nn.Module):
    def __init__(self, codebook_size: int, codebook_dim: int) -> None:
        super().__init__()
        self.dim = codebook_dim
        self.n_levels = codebook_size
        self.num_codebooks = codebook_dim

    def _rescale(self, x: torch.Tensor, levels: int) -> torch.Tensor:
        return (x * 2 / (levels - 1)) - 1

    def decode(
        self, codes: torch.Tensor, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        quantized = self._rescale(codes, self.n_levels).to(dtype)
        return quantized


class MistralAudioCodebook(nn.Module):
    def __init__(self, audio_tokenizer_args: AudioTokenizerArgs) -> None:
        super().__init__()
        self.semantic_codebook = SemanticCodebook(
            audio_tokenizer_args.semantic_codebook_size,
            audio_tokenizer_args.semantic_dim,
        )
        self.acoustic_codebook = AcousticCodebook(
            audio_tokenizer_args.acoustic_codebook_size,
            audio_tokenizer_args.acoustic_dim,
        )
        self.semantic_dim = audio_tokenizer_args.semantic_dim
        self.acoustic_dim = audio_tokenizer_args.acoustic_dim

    @property
    def num_codebooks(self) -> int:
        return (
            self.semantic_codebook.num_codebooks + self.acoustic_codebook.num_codebooks
        )

    def decode(
        self, codes: torch.Tensor, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        semantic_codes = codes[:, : self.semantic_codebook.num_codebooks, :]
        acoustic_codes = codes[:, self.semantic_codebook.num_codebooks :, :]
        semantic_emb = self.semantic_codebook.decode(semantic_codes).to(dtype)
        acoustic_emb = self.acoustic_codebook.decode(acoustic_codes).to(dtype)
        return torch.cat([semantic_emb, acoustic_emb], dim=1)


# ---- Conv layers ----


def pad1d(
    x: torch.Tensor,
    paddings: tuple[int, int],
    mode: str = "constant",
    value: float = 0.0,
) -> torch.Tensor:
    length = x.shape[-1]
    padding_left, padding_right = paddings
    if mode == "reflect":
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    return F.pad(x, paddings, mode, value)


class CausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        pad_mode: str = "reflect",
        use_weight_norm: bool = True,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            bias=use_bias,
        )
        self.conv = weight_norm(conv) if use_weight_norm else conv
        self.pad_mode = pad_mode
        self._stride = self.conv.stride[0]
        self._effective_kernel_size = (kernel_size - 1) * self.conv.dilation[0] + 1
        self._padding_total = self._effective_kernel_size - self._stride
        self.stride = self.conv.stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_frames = (
            x.shape[-1] - self._effective_kernel_size + self._padding_total
        ) / self._stride + 1
        target_length = (math.ceil(n_frames) - 1) * self._stride + (
            self._effective_kernel_size - self._padding_total
        )
        extra_padding = target_length - x.shape[-1]
        x = pad1d(x, (self._padding_total, extra_padding), mode=self.pad_mode)
        return self.conv(x)


class CausalConvTranspose1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        trim_ratio: float = 1.0,
        use_weight_norm: bool = True,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        conv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            groups=groups,
            bias=use_bias,
        )
        self.conv = weight_norm(conv) if use_weight_norm else conv
        self.trim_ratio = trim_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel_size = self.conv.kernel_size[0]
        stride = self.conv.stride[0]
        total_padding = kernel_size - stride
        out = self.conv(x)
        right_padding = math.ceil(total_padding * self.trim_ratio)
        left_padding = total_padding - right_padding
        return out[..., left_padding : out.shape[-1] - right_padding]


# ---- Attention ----


class Attention(nn.Module):
    def __init__(self, args: AudioTokenizerArgs, layer_id: int) -> None:
        super().__init__()
        self.args = args
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = args.n_kv_heads
        self.repeats = self.n_local_heads // self.n_local_kv_heads
        self.sliding_window = args.attn_sliding_window_size

        def get_alibi_slopes(n_heads: int) -> torch.Tensor:
            def slopes_power_of_2(n: int) -> torch.Tensor:
                r = 2.0 ** (-8.0 / n)
                return torch.tensor([r**i for i in range(n)], dtype=torch.float32)

            if math.log2(n_heads).is_integer():
                return slopes_power_of_2(n_heads)
            m = 2 ** math.floor(math.log2(n_heads))
            return torch.cat(
                [slopes_power_of_2(m), slopes_power_of_2(2 * m)[::2][: n_heads - m]]
            )

        self.register_buffer(
            "alibi_slopes", get_alibi_slopes(self.n_local_heads), persistent=False
        )

        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wo = nn.Linear(
            args.n_heads * args.head_dim, args.dim, bias=args.use_biases
        )

        if args.qk_norm:
            self.q_norm = rms_norm(args.n_heads * args.head_dim, eps=args.qk_norm_eps)
            self.k_norm = rms_norm(
                args.n_kv_heads * args.head_dim, eps=args.qk_norm_eps
            )

    def _native_attention(
        self, xq: torch.Tensor, xk: torch.Tensor, xv: torch.Tensor
    ) -> torch.Tensor:
        B, S, H, D = xq.shape
        Hkv = xk.shape[2]
        q = xq.transpose(1, 2)
        k = xk.transpose(1, 2)
        v = xv.transpose(1, 2)
        if H != Hkv:
            repeats = H // Hkv
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)

        positions = torch.arange(S, device=xq.device)
        rel_pos = positions.unsqueeze(0) - positions.unsqueeze(1)
        alibi_slopes = self.alibi_slopes.to(dtype=xq.dtype, device=xq.device)
        attn_bias = alibi_slopes.view(H, 1, 1) * rel_pos.unsqueeze(0).to(xq.dtype)
        if self.args.causal:
            attn_bias = attn_bias.masked_fill(rel_pos.unsqueeze(0) > 0, float("-inf"))
        window_left = self.sliding_window
        window_right = 0 if self.args.causal else self.sliding_window
        outside_window = (rel_pos < -window_left) | (rel_pos > window_right)
        attn_bias = attn_bias.masked_fill(outside_window.unsqueeze(0), float("-inf"))
        output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias.unsqueeze(0)
        )
        return output.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            bsz, seqlen = 1, x.shape[0]
            _ = x.shape[1]
        else:
            bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        if self.args.qk_norm:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.args.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.args.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.args.head_dim)

        if HAS_FLASH_ATTN:
            alibi_slopes = self.alibi_slopes.to(torch.float32)
            output = flash_attn_func(
                xq,
                xk,
                xv,
                causal=self.args.causal,
                window_size=(
                    self.sliding_window,
                    0 if self.args.causal else self.sliding_window,
                ),
                alibi_slopes=alibi_slopes,
            )
        else:
            output = self._native_attention(xq, xk, xv)

        output = output.view(bsz, seqlen, self.n_local_heads * self.args.head_dim)
        return self.wo(output).squeeze(0)


# ---- Transformer blocks ----


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: AudioTokenizerArgs) -> None:
        super().__init__()
        self._layer_id = layer_id
        self.attention = Attention(args, layer_id=layer_id)
        self.feed_forward = FeedForward(args.dim, args.hidden_dim, args.use_biases)
        self.attention_norm = rms_norm(args.dim, eps=args.norm_eps)
        self.ffn_norm = rms_norm(args.dim, eps=args.norm_eps)
        self.args = args
        self.layer_scale = args.layer_scale
        if self.layer_scale:
            if args.layer_scale_init is None:
                if layer_id < 18:
                    init_scale = 0.1
                elif layer_id <= 24:
                    init_scale = 1e-5
                else:
                    init_scale = 1e-6
            else:
                init_scale = args.layer_scale_init
            self.attention_scale = nn.Parameter(
                torch.full((args.dim,), init_scale, requires_grad=True)
            )
            self.ffn_scale = nn.Parameter(
                torch.full((args.dim,), init_scale, requires_grad=True)
            )

    @property
    def layer_id(self) -> int:
        return self._layer_id

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.attention(self.attention_norm(x))
        if self.layer_scale:
            r = self.attention_scale * r
        h = x + r
        r = self.feed_forward(self.ffn_norm(h))
        if self.layer_scale:
            r = self.ffn_scale * r
        return h + r


class Transformer(nn.Module):
    def __init__(self, args: AudioTokenizerArgs, n_layers: int) -> None:
        super().__init__()
        self.layers_ids = list(range(n_layers))
        self.layers = nn.ModuleDict()
        for layer_id in self.layers_ids:
            self.layers[str(layer_id)] = TransformerBlock(layer_id=layer_id, args=args)

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        h = input_ids
        for layer_id in self.layers_ids:
            h = self.layers[str(layer_id)](h)
        return h


# ---- MultiVocabEmbeddings ----


class MultiVocabEmbeddings(nn.Module):
    def __init__(self, audio_model_args: dict, embedding_dim: int) -> None:
        super().__init__()
        self.model_args = from_nested_dict(MultimodalAudioModelArgs, audio_model_args)
        self.codebook_sizes = list(
            self.model_args.get_codebook_sizes(pad_to_multiple=None)
        )
        self.offsets = torch.from_numpy(np.cumsum([0] + self.codebook_sizes[:-1]))
        self.total_vocab_size = sum(self.codebook_sizes)
        padded_size = 128 * ((self.total_vocab_size + 127) // 128)
        self.embeddings = nn.Embedding(padded_size, embedding_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.offsets = self.offsets.to(input_ids.device)
        input_ids = input_ids + self.offsets[torch.newaxis, :, torch.newaxis]
        return self.embeddings(input_ids)


# ---- Main Audio Tokenizer ----


def prepare_for_attention(x: torch.Tensor, time_last: bool = True) -> torch.Tensor:
    if time_last:
        return rearrange(x, "b d t -> (b t) d")
    return rearrange(x, "b t d -> (b t) d")


class VoxtralTTSAudioTokenizer(nn.Module):
    def __init__(self, audio_tokenizer_args: dict, audio_config: dict) -> None:
        super().__init__()
        args = from_nested_dict(AudioTokenizerArgs, audio_tokenizer_args)
        self.args = args

        self.patch_size = args.pretransform_patch_size
        self.latent_dim = args.semantic_dim + args.acoustic_dim

        # Decoder only
        decoder_blocks: list[nn.Module] = []
        decoder_convs_kernels = args.decoder_convs_kernels
        decoder_convs_strides = args.decoder_convs_strides
        decoder_transformer_lengths = args.decoder_transformer_lengths

        cur_window_size = args.attn_sliding_window_size
        for s in args.encoder_convs_strides:
            if args.half_attn_window_upon_downsampling and s > 1:
                cur_window_size = cur_window_size // 2

        decoder_blocks.append(
            CausalConv1d(
                self.latent_dim,
                args.dim,
                kernel_size=decoder_convs_kernels[0],
                stride=decoder_convs_strides[0],
                pad_mode="replicate",
                use_bias=False,
            )
        )
        if args.half_attn_window_upon_downsampling and decoder_convs_strides[0] > 1:
            cur_window_size = cur_window_size * 2

        for idx, n_layers in enumerate(decoder_transformer_lengths):
            layer_args = deepcopy(args)
            layer_args.attn_sliding_window_size = cur_window_size
            decoder_blocks.append(Transformer(args=layer_args, n_layers=n_layers))
            if (idx + 1 != len(decoder_transformer_lengths)) and (
                (decoder_convs_kernels[idx + 1] != 1)
                or (decoder_convs_strides[idx + 1] != 1)
            ):
                decoder_blocks.append(
                    CausalConvTranspose1d(
                        args.dim,
                        args.dim,
                        kernel_size=decoder_convs_kernels[idx + 1],
                        stride=decoder_convs_strides[idx + 1],
                        use_bias=False,
                    )
                )
                if (
                    args.half_attn_window_upon_downsampling
                    and decoder_convs_strides[idx + 1] > 1
                ):
                    cur_window_size = cur_window_size * 2

        self.decoder_blocks = nn.ModuleList(decoder_blocks)
        self.quantizer = MistralAudioCodebook(args)
        self.output_proj = CausalConv1d(
            args.dim,
            args.pretransform_patch_size,
            kernel_size=args.patch_proj_kernel_size,
            use_weight_norm=args.conv_weight_norm,
            use_bias=False,
        )

        scale_factor = math.prod(args.encoder_convs_strides)
        self._frame_rate = args.sampling_rate / (self.patch_size * scale_factor)
        self._sampling_rate = args.sampling_rate

    @property
    def sampling_rate(self) -> int:
        return self._sampling_rate

    @property
    def downsample_factor(self) -> int:
        return int(self._sampling_rate / self._frame_rate)

    @property
    def num_codebooks(self) -> int:
        return self.quantizer.num_codebooks

    def _forward_decoder(self, emb: torch.Tensor) -> torch.Tensor:
        emb = rearrange(emb, "b d t -> b t d").contiguous()
        for block in self.decoder_blocks:
            if isinstance(block, (CausalConvTranspose1d, CausalConv1d)):
                emb = rearrange(emb, "b t d -> b d t")
                emb = block(emb)
                emb = rearrange(emb, "b d t -> b t d")
            else:
                emb = block(emb)
        emb = rearrange(emb, "b t d -> b d t")
        emb = self.output_proj(emb)
        return rearrange(emb, "b (c h) t -> b c (t h)", h=self.patch_size)

    def decode(
        self, codes: torch.Tensor, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        emb = self.quantizer.decode(codes, dtype)
        return self._forward_decoder(emb)

    def decode_helper_batch_async(
        self, codes_list: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """Batch decode a list of code tensors to waveform.

        Args:
            codes_list: list of [T_i, K] int tensors.
        Returns:
            list of 1-D float tensors.
        """
        chunk_size = 375

        processed = []
        for codes in codes_list:
            eoa_mask = codes[:, 0] == 1
            eoa_indices = eoa_mask.nonzero(as_tuple=False)
            cutting_point = (
                eoa_indices[0].item() if len(eoa_indices) > 0 else len(codes)
            )
            audio_tokens = codes[:cutting_point] - 2
            processed.append(audio_tokens)

        results: list[torch.Tensor | None] = [None] * len(processed)
        non_empty: list[tuple[int, torch.Tensor]] = []
        for idx, tokens in enumerate(processed):
            if len(tokens) == 0:
                results[idx] = torch.tensor([], dtype=torch.float32)
            else:
                non_empty.append((idx, tokens))

        if not non_empty:
            return results

        all_chunks: list[torch.Tensor] = []
        chunk_lengths: list[int] = []
        chunk_map: list[tuple[int, list[int]]] = []
        for orig_idx, tokens in non_empty:
            req_chunk_indices = []
            for i in range(0, len(tokens), chunk_size):
                chunk = tokens[i : i + chunk_size]
                req_chunk_indices.append(len(all_chunks))
                chunk_lengths.append(len(chunk))
                all_chunks.append(chunk)
            chunk_map.append((orig_idx, req_chunk_indices))

        max_chunk_len = max(chunk_lengths)
        K = all_chunks[0].shape[1]
        padded = torch.zeros(
            len(all_chunks),
            max_chunk_len,
            K,
            dtype=all_chunks[0].dtype,
            device=all_chunks[0].device,
        )
        for i, chunk in enumerate(all_chunks):
            padded[i, : len(chunk)] = chunk

        device = next(self.parameters()).device
        audio_codes = padded.to(device=device)
        audio_values = self.decode(audio_codes.transpose(1, 2), dtype=torch.bfloat16)
        audio_values = audio_values.detach().cpu().float().squeeze(1)

        for orig_idx, chunk_indices in chunk_map:
            audio_parts = []
            for ci in chunk_indices:
                expected_samples = chunk_lengths[ci] * self.downsample_factor
                audio_parts.append(audio_values[ci, :expected_samples])
            results[orig_idx] = (
                torch.cat(audio_parts, dim=0)
                if len(audio_parts) > 1
                else audio_parts[0]
            )

        return results

    def load_weight(self, weight: tuple[str, torch.Tensor]) -> str:
        params_dict = dict(self.named_parameters())
        name, loaded_weight = weight
        if name not in params_dict:
            if name == "quantizer.semantic_codebook.cluster_usage":
                self.quantizer.semantic_codebook.cluster_usage = loaded_weight
                return name
            elif name == "quantizer.semantic_codebook.embedding_sum":
                self.quantizer.semantic_codebook.embedding_sum = loaded_weight
                return name
            else:
                logger.warning("Weight %s not found in audio tokenizer", name)
                return name
        param = params_dict[name]
        param.data.copy_(loaded_weight)
        return name
