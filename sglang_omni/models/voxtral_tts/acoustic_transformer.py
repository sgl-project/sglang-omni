"""FlowMatchingAudioTransformer for Voxtral TTS.
"""

import math
import types
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Any, Optional, Union, get_args, get_origin

import torch
import torch.nn as nn
import torch.nn.functional as F

rms_norm = torch.nn.RMSNorm


class AudioSpecialTokens(str, Enum):
    """Special tokens predicted by audio codebook heads."""

    empty_audio = "[EMPTY_AUDIO]"
    end_audio = "[END_AUDIO]"

    @staticmethod
    def all_special_tokens() -> list["AudioSpecialTokens"]:
        return [token for token in AudioSpecialTokens]

    @staticmethod
    def id(token: "AudioSpecialTokens") -> int:
        return AudioSpecialTokens.all_special_tokens().index(token)


@dataclass
class AcousticTransformerArgs:
    input_dim: int = 3072
    dim: int = 768
    n_layers: int = 3
    head_dim: int = 128
    hidden_dim: int = 2048
    n_heads: int = 6
    n_kv_heads: int = 2
    use_biases: bool = False
    norm_eps: float = 1e-5
    sigma: float = 1e-5


@dataclass
class MultimodalAudioModelArgs:
    semantic_codebook_size: int = 8192
    acoustic_codebook_size: int = 21
    n_acoustic_codebook: int = 36
    acoustic_transformer_args: Optional[AcousticTransformerArgs] = None

    @property
    def codebook_sizes(self) -> list[int]:
        return [
            self.semantic_codebook_size,
            *[self.acoustic_codebook_size for _ in range(self.n_acoustic_codebook)],
        ]

    def get_codebook_sizes(
        self, pad_to_multiple: Optional[int] = 128, include_special_tokens: bool = True
    ) -> list[int]:
        def _round_up(n: int, multiple: int) -> int:
            return multiple * ((n + multiple - 1) // multiple)

        result = []
        for cb_size in self.codebook_sizes:
            if include_special_tokens:
                cb_size += len(AudioSpecialTokens.all_special_tokens())
            if pad_to_multiple is not None:
                cb_size = _round_up(cb_size, pad_to_multiple)
            result.append(cb_size)
        return result


def from_nested_dict(cls: type, d: dict) -> Any:
    """Recursively instantiate dataclasses from nested dicts."""
    if not is_dataclass(cls):
        return d
    kwargs = {}
    for f in fields(cls):
        value = d.get(f.name, getattr(cls, f.name, None))
        field_type = f.type
        origin = get_origin(field_type)
        if origin is Union or origin is types.UnionType:
            args = get_args(field_type)
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                field_type = non_none[0]
        if is_dataclass(field_type) and isinstance(value, dict):
            value = from_nested_dict(field_type, value)
        kwargs[f.name] = value
    return cls(**kwargs)


def _repeat_interleave(t: torch.Tensor, repeats: int) -> torch.Tensor:
    return t.unsqueeze(3).expand([-1, -1, -1, repeats, -1]).flatten(2, 3)


def repeat_kv(
    keys: torch.Tensor, values: torch.Tensor, repeats: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if repeats > 1:
        keys = _repeat_interleave(keys, repeats=repeats)
        values = _repeat_interleave(values, repeats=repeats)
    return keys, values


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, use_biases: bool) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=use_biases)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class BidirectionalAttention(nn.Module):
    def __init__(self, args: AcousticTransformerArgs, layer_id: int) -> None:
        super().__init__()
        self.args = args
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = args.n_kv_heads
        self.head_dim = args.head_dim
        self.repeats = self.n_local_heads // self.n_local_kv_heads

        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=args.use_biases)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=args.use_biases)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=args.use_biases)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            bsz, seqlen = 1, x.shape[0]
            _ = x.shape[1]
        else:
            bsz, seqlen, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        keys, values = repeat_kv(xk, xv, repeats=self.repeats)

        # Native attention
        scale = 1.0 / xq.shape[-1] ** 0.5
        q = (xq * scale).transpose(1, 2)
        k = keys.transpose(1, 2)
        v = values.transpose(1, 2)
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(-1)
        output = (attn @ v).transpose(1, 2).contiguous()

        output = output.view(bsz, seqlen, -1)
        return self.wo(output).squeeze(0)


class AcousticTransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: AcousticTransformerArgs) -> None:
        super().__init__()
        self._layer_id = layer_id
        self.attention = BidirectionalAttention(args, layer_id=layer_id)
        self.feed_forward = FeedForward(args.dim, args.hidden_dim, args.use_biases)
        self.attention_norm = rms_norm(args.dim, eps=args.norm_eps)
        self.ffn_norm = rms_norm(args.dim, eps=args.norm_eps)

    @property
    def layer_id(self) -> int:
        return self._layer_id

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.attention(self.attention_norm(x))
        h = x + r
        r = self.feed_forward(self.ffn_norm(h))
        return h + r


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = torch.exp(
            -math.log(theta) * torch.arange(dim // 2).float() / (dim // 2)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=True)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = torch.einsum("bi, j -> bj", t, self.inv_freq)
        return torch.cat((emb.cos(), emb.sin()), dim=-1)


class FlowMatchingAudioTransformer(nn.Module):
    def __init__(self, audio_model_args: dict) -> None:
        super().__init__()
        if "codebook_sizes" in audio_model_args:
            codebook_sizes = [
                int(c) for c in audio_model_args.pop("codebook_sizes").split(",")
            ]
            audio_model_args.update(
                {
                    "semantic_codebook_size": codebook_sizes[0],
                    "acoustic_codebook_size": codebook_sizes[1],
                    "n_acoustic_codebook": len(codebook_sizes) - 1,
                }
            )
        self.model_args: MultimodalAudioModelArgs = from_nested_dict(
            MultimodalAudioModelArgs, audio_model_args
        )
        args = self.model_args.acoustic_transformer_args
        self.acoustic_transformer_args = args

        self.num_non_acoustic_embeddings = 1
        self.num_acoustic_codebooks = (
            len(self.model_args.get_codebook_sizes()) - self.num_non_acoustic_embeddings
        )

        self.sigma = args.sigma

        acoustic_codebook_sizes = self.model_args.get_codebook_sizes(
            pad_to_multiple=None, include_special_tokens=False
        )[1:]
        self.acoustic_embeddings_levels = acoustic_codebook_sizes[0]
        self.acoustic_embeddings_dim = len(acoustic_codebook_sizes)

        self._init_audio_embeddings_layer()
        self._init_output_layer()
        self._init_layers()

        self._end_audio_token_id = AudioSpecialTokens.id(AudioSpecialTokens.end_audio)
        self._empty_audio_token_id = AudioSpecialTokens.id(
            AudioSpecialTokens.empty_audio
        )

        self._acoustic_decode_iters = 16
        self._cfg_alpha = 1.2
        self._noise_scale = 1.0
        self.register_buffer(
            "_timesteps",
            torch.linspace(0, 1, self._acoustic_decode_iters),
            persistent=False,
        )

    def load_weight(self, weight: tuple[str, torch.Tensor]) -> str:
        params_dict = dict(self.named_parameters())
        name, loaded_weight = weight
        if name not in params_dict:
            return name
        param = params_dict[name]
        param.data.copy_(loaded_weight)
        return name

    def _init_audio_embeddings_layer(self) -> None:
        args = self.acoustic_transformer_args
        self.time_embedding = TimeEmbedding(args.dim)
        input_dim = self.acoustic_embeddings_dim
        self.input_projection = nn.Linear(input_dim, args.dim, bias=False)
        self.time_projection = nn.Linear(args.dim, args.dim, bias=False)
        self.llm_projection = nn.Linear(args.input_dim, args.dim, bias=False)

    def _init_output_layer(self) -> None:
        padded_codebook_sizes = self.model_args.get_codebook_sizes(
            pad_to_multiple=128
        )
        args = self.acoustic_transformer_args
        self.semantic_codebook_output = nn.Linear(
            args.dim, padded_codebook_sizes[0], args.use_biases
        )
        self.acoustic_codebook_output = nn.Linear(
            in_features=args.dim,
            out_features=self.model_args.n_acoustic_codebook,
            bias=False,
        )

    def _init_layers(self) -> None:
        args = self.acoustic_transformer_args
        self.layers_ids = list(range(args.n_layers))
        self.layers = nn.ModuleDict()
        for layer_id in self.layers_ids:
            self.layers[str(layer_id)] = AcousticTransformerBlock(
                layer_id=layer_id, args=args
            )
        self.norm = rms_norm(args.dim, args.norm_eps)

    def forward_attention_layers(self, h: torch.Tensor) -> torch.Tensor:
        for layer_id in self.layers_ids:
            h = self.layers[str(layer_id)](h)
        return h

    def decode_one_frame(
        self, semantic_code: torch.Tensor, llm_hidden: torch.Tensor
    ) -> torch.Tensor:
        B = semantic_code.shape[0]
        should_decode = semantic_code != self._end_audio_token_id

        x_0 = torch.randn(B, self.model_args.n_acoustic_codebook).to(
            dtype=llm_hidden.dtype, device=llm_hidden.device
        )
        x_0 = self._noise_scale * x_0

        timesteps = self._timesteps.to(dtype=llm_hidden.dtype)
        llm_hidden_zero = torch.zeros_like(llm_hidden)

        sampled = x_0
        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            dt = timesteps[i + 1] - timesteps[i]
            t_emb = self.time_embedding(t.view(-1, 1).repeat(B, 1)).to(
                llm_hidden.dtype
            )

            x_batched = torch.cat([sampled, sampled], dim=0)
            llm_batched = torch.cat([llm_hidden, llm_hidden_zero], dim=0)
            t_emb_batched = torch.cat([t_emb, t_emb], dim=0)

            v_all = self._predict_velocity(
                x_t=x_batched, llm_output=llm_batched, t_emb=t_emb_batched
            )
            v_t, uncond_v_t = v_all[:B], v_all[B:]
            v_t = self._cfg_alpha * v_t + (1 - self._cfg_alpha) * uncond_v_t
            sampled = sampled + v_t * dt

        sampled = torch.clamp(sampled, -1, 1)
        scaled_x = ((sampled + 1) / 2) * (self.acoustic_embeddings_levels - 1)
        output_codes = scaled_x.round().long()
        output_codes[~should_decode] = self._empty_audio_token_id
        return output_codes + len(AudioSpecialTokens)

    def _predict_velocity(
        self,
        x_t: torch.Tensor,
        llm_output: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> torch.Tensor:
        x_t = x_t.to(llm_output.dtype)
        t_emb = self.time_projection(t_emb)
        llm_output = self.llm_projection(llm_output)

        acoustic_and_semantic_embeddings = [
            self.input_projection(x_t.unsqueeze(1)),
            t_emb.unsqueeze(1),
            llm_output.unsqueeze(1),
        ]
        acoustic_transformer_inputs = torch.concatenate(
            acoustic_and_semantic_embeddings, dim=1
        )

        attn_output = self.forward_attention_layers(acoustic_transformer_inputs)
        final_hidden = self.norm(attn_output)
        final_hidden = final_hidden.view(
            -1, acoustic_transformer_inputs.shape[1], final_hidden.shape[-1]
        )
        return self.acoustic_codebook_output(final_hidden[:, 0, :])

    def forward(self, llm_hidden: torch.Tensor) -> torch.Tensor:
        semantic_logit = self.semantic_codebook_output(llm_hidden).float()
        semantic_logit[:, self._empty_audio_token_id] = -float("inf")
        semantic_logit[
            :,
            (len(AudioSpecialTokens) + self.model_args.semantic_codebook_size) :,
        ] = -float("inf")

        semantic_code = semantic_logit.argmax(dim=-1, keepdim=True)

        acoustic_codes = self.decode_one_frame(
            semantic_code.squeeze(1), llm_hidden
        )

        return torch.concatenate([semantic_code, acoustic_codes], dim=1)
