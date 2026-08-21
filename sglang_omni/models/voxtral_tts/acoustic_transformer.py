# SPDX-License-Identifier: Apache-2.0
"""Acoustic transformer components for Voxtral TTS.

Contains AudioSpecialTokens, FlowMatchingAudioTransformer, and all supporting
sub-modules (AcousticTransformerBlock, BidirectionalAttention, FeedForward,
TimeEmbedding, etc.).

"""

import logging
import math
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Union, get_args, get_origin

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from apex.normalization import FusedRMSNorm

    rms_norm = FusedRMSNorm
except ImportError:
    from torch.nn import RMSNorm as RMSNorm

    rms_norm = RMSNorm

from sglang_omni.cuda_graph import KeyedGraphCache, normalize_batch_sizes
from sglang_omni.models.weight_loader import default_weight_loader

logger = logging.getLogger(__name__)

VOXTRAL_FRAME_GRAPH_ENV = "SGLANG_OMNI_VOXTRAL_FRAME_GRAPH"


# ---------------------------------------------------------------------------
# Audio special tokens
# ---------------------------------------------------------------------------


class AudioSpecialTokens(str, Enum):
    """Special tokens predicted by audio codebook heads.

    These tokens are inserted by ``audio_tokens_with_pattern``.  They are not
    part of the text vocabulary.  We offset the output audio tokens from the
    quantizer by ``len(all_special_tokens)`` to avoid conflicts with text
    tokens.
    """

    empty_audio = "[EMPTY_AUDIO]"
    end_audio = "[END_AUDIO]"

    @staticmethod
    def all_special_tokens() -> list["AudioSpecialTokens"]:
        return [token for token in AudioSpecialTokens]

    @staticmethod
    def id(token: "AudioSpecialTokens") -> int:
        return AudioSpecialTokens.all_special_tokens().index(token)


# ---------------------------------------------------------------------------
# Model argument dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AcousticTransformerArgs:
    input_dim: int
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
    # comma-separated list of codebook sizes.
    # The first token in a codebook should always be reserved to indicate
    # absence.  The codebook size should be inclusive of this.
    semantic_codebook_size: int
    acoustic_codebook_size: int
    n_acoustic_codebook: int
    acoustic_transformer_args: AcousticTransformerArgs

    @property
    def codebook_sizes(self) -> list[int]:
        return [
            self.semantic_codebook_size,
            *[self.acoustic_codebook_size for _ in range(self.n_acoustic_codebook)],
        ]

    def get_codebook_sizes(
        self,
        pad_to_multiple: int | None = 128,
        include_special_tokens: bool = True,
    ) -> list[int]:
        def _round_up(n: int, multiple: int) -> int:
            return multiple * ((n + multiple - 1) // multiple)

        result: list[int] = []
        for cb_size in self.codebook_sizes:
            if include_special_tokens:
                cb_size += len(AudioSpecialTokens.all_special_tokens())
            if pad_to_multiple is not None:
                cb_size = _round_up(cb_size, pad_to_multiple)
            result.append(cb_size)
        return result


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _repeat_interleave(t: torch.Tensor, repeats: int) -> torch.Tensor:
    return t.unsqueeze(3).expand([-1, -1, -1, repeats, -1]).flatten(2, 3)


def repeat_kv(
    keys: torch.Tensor, values: torch.Tensor, repeats: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if repeats > 1:
        keys = _repeat_interleave(keys, repeats=repeats)
        values = _repeat_interleave(values, repeats=repeats)
    return keys, values


def from_nested_dict(cls, d):
    """Recursively instantiate dataclasses from nested dicts."""
    if not is_dataclass(cls):
        return d

    kwargs = {}
    for f in fields(cls):
        value = d.get(f.name, getattr(cls, f.name, None))
        field_type = f.type

        origin = get_origin(field_type)
        if origin is Union:
            args = get_args(field_type)
            # Filter out NoneType from Union args (e.g. Optional[X] = Union[X, None])
            non_none = [a for a in args if a is not type(None)]  # noqa: E721
            if len(non_none) == 1:
                field_type = non_none[0]

        if is_dataclass(field_type) and isinstance(value, dict):
            value = from_nested_dict(field_type, value)

        kwargs[f.name] = value
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, use_biases: bool) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=use_biases)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class BidirectionalAttention(nn.Module):
    """Attention layer (without RoPE embeddings)."""

    def __init__(self, args: AcousticTransformerArgs, layer_id: int) -> None:
        super().__init__()
        self.args = args
        self.n_local_heads: int = args.n_heads
        self.n_local_kv_heads: int = args.n_kv_heads
        self.layer_id = layer_id
        self.head_dim = args.head_dim

        self.wq = nn.Linear(
            args.dim, args.n_heads * args.head_dim, bias=args.use_biases
        )
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(
            args.dim, args.n_kv_heads * args.head_dim, bias=args.use_biases
        )
        self.wo = nn.Linear(
            args.n_heads * args.head_dim, args.dim, bias=args.use_biases
        )

        self.softmax_scale: float = args.head_dim**-0.5
        self.repeats = self.n_local_heads // self.n_local_kv_heads

    def _native_attention(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        scale = 1.0 / query.shape[-1] ** 0.5
        query = query * scale
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        attn = query @ key.transpose(-2, -1)
        attn = attn.softmax(-1)
        attn = attn @ value
        return attn.transpose(1, 2).contiguous()

    def _forward_attention(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        key, value = repeat_kv(key, value, repeats=self.repeats)
        bsz, seqlen, _, _ = query.shape
        output = self._native_attention(query, key, value)
        return output.view(bsz, seqlen, -1)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if x.dim() == 2:
            bsz, (seqlen, _) = 1, x.shape
        else:
            bsz, seqlen, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        output = self._forward_attention(query=xq, key=xk, value=xv, **kwargs)
        output = output.view(bsz, seqlen, self.n_local_heads * self.head_dim)
        return self.wo(output).squeeze(0)


class AcousticTransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: AcousticTransformerArgs) -> None:
        super().__init__()
        self._layer_id = layer_id
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.attention = BidirectionalAttention(args, layer_id=layer_id)
        self.feed_forward = FeedForward(args.dim, args.hidden_dim, args.use_biases)
        self.attention_norm = rms_norm(args.dim, eps=args.norm_eps)
        self.ffn_norm = rms_norm(args.dim, eps=args.norm_eps)
        self.args = args

    @property
    def layer_id(self) -> int:
        return self._layer_id

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.attention.forward(self.attention_norm(x))
        h = x + r
        r = self.feed_forward.forward(self.ffn_norm(h))
        return h + r


# ---------------------------------------------------------------------------
# Flow Matching Acoustic Transformer
# ---------------------------------------------------------------------------


class TimeEmbedding(nn.Module):
    """Sinusoidal embedding for encoding time."""

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = torch.exp(
            -math.log(theta) * torch.arange(dim // 2).float() / (dim // 2)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=True)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = torch.einsum("bi, j -> bj", t, self.inv_freq)
        return torch.cat((emb.cos(), emb.sin()), dim=-1)


class _FrameDecodeGraph:
    """CUDA graph over the whole per-frame flow-matching chain for one bucket.

    The chain is the 7 step ODE loop plus its CFG batch-doubled acoustic stack,
    so a replay covers the frame's few hundred launches with one. Per-frame
    inputs, including the noise drawn on the host RNG, reach the captured region
    through persistent device buffers written with device-side copies.
    """

    def __init__(
        self,
        model: "FlowMatchingAudioTransformer",
        batch_size: int,
        *,
        input_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self.semantic_code = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.llm_hidden = torch.zeros(batch_size, input_dim, dtype=dtype, device=device)
        self.noise = torch.zeros(
            batch_size,
            model.model_args.n_acoustic_codebook,
            dtype=dtype,
            device=device,
        )
        self.graph = torch.cuda.CUDAGraph()
        self.output_codes: torch.Tensor | None = None
        try:
            self._capture()
        except Exception:
            # Note: (Jiaxin Deng) release the graph's private memory pool
            # eagerly; the raising object may linger on traceback frames.
            try:
                self.graph.reset()
            except Exception:
                pass
            raise

    @torch.no_grad()
    def _capture(self) -> None:
        model = self.model
        device = self.llm_hidden.device
        # Note: (Jiaxin Deng) prime the dtype-cast timestep cache outside the
        # captured region; a tensor first allocated inside it would be owned by
        # the graph pool but then read by the eager path too.
        model._frame_timesteps(self.llm_hidden.dtype, device)
        with torch.cuda.device(device):
            current_stream = torch.cuda.current_stream(device=device)
            warmup_stream = torch.cuda.Stream(device=device)
            warmup_stream.wait_stream(current_stream)
            with torch.cuda.stream(warmup_stream):
                for _ in range(2):
                    model._decode_one_frame_chain(
                        self.semantic_code, self.llm_hidden, self.noise
                    )
            current_stream.wait_stream(warmup_stream)

            capture_stream = torch.cuda.Stream(device=device)
            capture_stream.wait_stream(current_stream)
            with torch.cuda.graph(
                self.graph,
                pool=model._frame_graph_cache.memory_pool(),
                stream=capture_stream,
                capture_error_mode="thread_local",
            ):
                self.output_codes = model._decode_one_frame_chain(
                    self.semantic_code, self.llm_hidden, self.noise
                )
            current_stream.wait_stream(capture_stream)

        if self.output_codes is None:
            raise RuntimeError("Voxtral-TTS frame CUDA graph captured no outputs")

    @torch.no_grad()
    def replay(
        self,
        semantic_code: torch.Tensor,
        llm_hidden: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        live = semantic_code.shape[0]
        if live > self.batch_size:
            raise ValueError(
                "Voxtral-TTS frame CUDA graph bucket is too small: "
                f"bucket={self.batch_size}, live={live}"
            )
        with torch.cuda.device(self.llm_hidden.device):
            self.semantic_code[:live].copy_(semantic_code.reshape(live))
            self.llm_hidden[:live].copy_(llm_hidden)
            self.noise[:live].copy_(noise)
            if live < self.batch_size:
                self.semantic_code[live:].zero_()
                self.llm_hidden[live:].zero_()
                self.noise[live:].zero_()
            self.graph.replay()
        return self.output_codes[:live]


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
        assert isinstance(self.model_args, MultimodalAudioModelArgs)
        args = self.model_args.acoustic_transformer_args
        self.acoustic_transformer_args = args
        assert isinstance(self.acoustic_transformer_args, AcousticTransformerArgs)

        # currently assuming always 1 semantic codebook + N acoustic codebooks
        self.num_non_acoustic_embeddings = 1
        self.num_acoustic_codebooks = (
            len(self.model_args.get_codebook_sizes()) - self.num_non_acoustic_embeddings
        )

        # flow matching utils
        self.sigma = args.sigma

        # codebook sizes
        acoustic_codebook_sizes = self.model_args.get_codebook_sizes(
            pad_to_multiple=None, include_special_tokens=False
        )[1:]
        assert (
            len(set(acoustic_codebook_sizes)) == 1
        ), "only 1 size for acoustic codebooks supported"
        self.acoustic_embeddings_levels = acoustic_codebook_sizes[0]
        self.acoustic_embeddings_dim = len(acoustic_codebook_sizes)

        self._init_audio_embeddings_layer()
        self._init_output_layer()
        self._init_layers()

        self._end_audio_token_id = AudioSpecialTokens.id(AudioSpecialTokens.end_audio)
        self._empty_audio_token_id = AudioSpecialTokens.id(
            AudioSpecialTokens.empty_audio
        )

        # Flow matching constants
        self._acoustic_decode_iters = 8
        self._cfg_alpha = 1.2
        self._noise_scale = 1.0
        self.register_buffer(
            "_timesteps",
            torch.linspace(0, 1, self._acoustic_decode_iters),
            persistent=False,
        )

        self._frame_timesteps_cache: dict[tuple, torch.Tensor] = {}
        self._frame_graph_cache: KeyedGraphCache | None = None
        self._frame_graph_runtime_checked = True

    def enable_frame_graph(
        self,
        *,
        max_batch_size: int,
        cuda_graph_bs: list[int] | None = None,
    ) -> None:
        """Arm per-frame CUDA graph capture for the serving decode path."""
        self._frame_graph_cache = KeyedGraphCache(
            name="Voxtral-TTS frame",
            batch_sizes=normalize_batch_sizes(
                cuda_graph_bs, max_batch_size=max_batch_size
            ),
            env_var=VOXTRAL_FRAME_GRAPH_ENV,
        )
        # Note: (Jiaxin Deng) runtime gates resolve at decode time; bootstrap
        # forces disable_cuda_graph on during init (deferred capture).
        self._frame_graph_runtime_checked = False

    def load_weight(self, weight: tuple[str, torch.Tensor]) -> str:
        params_dict = dict(self.named_parameters())
        name, loaded_weight = weight
        if name not in params_dict:
            logger.warning(f"{name} not found in FlowMatchingAudioTransformer (UNUSED)")
            return name
        param = params_dict[name]
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, loaded_weight)
        return name

    # -- Initialization helpers ---------------------------------------------

    def _init_audio_embeddings_layer(self) -> None:
        self.time_embedding = TimeEmbedding(self.acoustic_transformer_args.dim)
        input_dim = self.acoustic_embeddings_dim
        self.input_projection = nn.Linear(
            input_dim, self.acoustic_transformer_args.dim, bias=False
        )
        self.time_projection = nn.Linear(
            self.acoustic_transformer_args.dim,
            self.acoustic_transformer_args.dim,
            bias=False,
        )
        self.llm_projection = nn.Linear(
            self.acoustic_transformer_args.input_dim,
            self.acoustic_transformer_args.dim,
            bias=False,
        )

    def _init_output_layer(self) -> None:
        padded_codebook_sizes = self.model_args.get_codebook_sizes(pad_to_multiple=128)
        self.semantic_codebook_output = nn.Linear(
            self.acoustic_transformer_args.dim,
            padded_codebook_sizes[0],
            self.acoustic_transformer_args.use_biases,
        )
        self.acoustic_codebook_output = nn.Linear(
            in_features=self.acoustic_transformer_args.dim,
            out_features=self.model_args.n_acoustic_codebook,
            bias=False,
        )

    def _init_layers(self) -> None:
        self.layers_ids: list[int] = list(
            range(self.acoustic_transformer_args.n_layers)
        )
        self.layers = nn.ModuleDict()
        for layer_id in self.layers_ids:
            self.layers[str(layer_id)] = AcousticTransformerBlock(
                layer_id=layer_id, args=self.acoustic_transformer_args
            )
        self.norm = rms_norm(
            self.acoustic_transformer_args.dim,
            self.acoustic_transformer_args.norm_eps,
        )

    # -- Forward path -------------------------------------------------------

    def forward_attention_layers(self, h: torch.Tensor) -> torch.Tensor:
        for layer_id in self.layers_ids:
            h = self.layers[str(layer_id)](h)
        return h

    def _frame_timesteps(
        self, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        key = (dtype, device)
        timesteps = self._frame_timesteps_cache.get(key)
        if timesteps is None:
            timesteps = self._timesteps.to(dtype=dtype, device=device)
            self._frame_timesteps_cache[key] = timesteps
        return timesteps

    def _draw_frame_noise(
        self, batch_size: int, llm_hidden: torch.Tensor
    ) -> torch.Tensor:
        # Note: (Jiaxin Deng) the draw stays on the host RNG and outside any
        # captured region, so a graphed frame consumes the same seeded stream
        # as the eager one; only the resulting values enter the graph.
        x_0 = torch.randn(
            batch_size, self.model_args.n_acoustic_codebook, device="cpu"
        ).to(dtype=llm_hidden.dtype, device=llm_hidden.device)
        return self._noise_scale * x_0

    def decode_one_frame(
        self, semantic_code: torch.Tensor, llm_hidden: torch.Tensor
    ) -> torch.Tensor:
        x_0 = self._draw_frame_noise(semantic_code.shape[0], llm_hidden)
        graphed = self._decode_one_frame_graphed(semantic_code, llm_hidden, x_0)
        if graphed is not None:
            return graphed
        return self._decode_one_frame_chain(semantic_code, llm_hidden, x_0)

    def _check_frame_graph_runtime(self) -> None:
        """Resolve the server-arg gates once, on the first decoded frame."""
        if self._frame_graph_runtime_checked:
            return
        self._frame_graph_runtime_checked = True
        from sglang.srt.server_args import get_global_server_args

        server_args = get_global_server_args()
        if bool(server_args.disable_cuda_graph):
            self._frame_graph_cache.disable("disable_cuda_graph is set")
        elif int(getattr(server_args, "tp_size", 1) or 1) != 1:
            # Note: (Jiaxin Deng) capture under TP would record collectives; the
            # graphed chain is only validated single-rank, so TP stays eager.
            self._frame_graph_cache.disable("tp_size > 1")

    def _decode_one_frame_graphed(
        self,
        semantic_code: torch.Tensor,
        llm_hidden: torch.Tensor,
        x_0: torch.Tensor,
    ) -> torch.Tensor | None:
        cache = self._frame_graph_cache
        if cache is None:
            return None
        self._check_frame_graph_runtime()
        if not cache.enabled:
            return None
        batch_size = semantic_code.shape[0]
        if batch_size == 0 or semantic_code.ndim != 1 or llm_hidden.ndim != 2:
            return None
        if semantic_code.dtype not in (torch.int, torch.long):
            return None
        if not semantic_code.is_cuda or not llm_hidden.is_cuda:
            return None
        if torch.cuda.is_current_stream_capturing():
            return None
        bucket_size = cache.bucket_for(batch_size)
        if bucket_size is None:
            return None
        input_dim = int(llm_hidden.shape[-1])
        graph = cache.get_or_capture(
            (bucket_size, str(llm_hidden.dtype), input_dim),
            lambda: _FrameDecodeGraph(
                self,
                bucket_size,
                input_dim=input_dim,
                dtype=llm_hidden.dtype,
                device=llm_hidden.device,
            ),
        )
        if graph is None:
            return None
        return graph.replay(semantic_code, llm_hidden, x_0)

    def _decode_one_frame_chain(
        self,
        semantic_code: torch.Tensor,
        llm_hidden: torch.Tensor,
        x_0: torch.Tensor,
    ) -> torch.Tensor:
        B = semantic_code.shape[0]
        should_decode = (semantic_code != self._end_audio_token_id).view(-1, 1)

        timesteps = self._frame_timesteps(llm_hidden.dtype, llm_hidden.device)
        llm_batched = torch.cat([llm_hidden, torch.zeros_like(llm_hidden)], dim=0)

        sampled = x_0
        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            dt = timesteps[i + 1] - timesteps[i]

            t_emb = self.time_embedding(t.view(-1, 1).repeat(B, 1)).to(llm_hidden.dtype)

            x_batched = torch.cat([sampled, sampled], dim=0)
            t_emb_batched = torch.cat([t_emb, t_emb], dim=0)

            v_all = self._predict_velocity(
                x_t=x_batched, llm_output=llm_batched, t_emb=t_emb_batched
            )
            v_t, uncond_v_t = v_all[:B], v_all[B:]
            v_t = self._cfg_alpha * v_t + (1 - self._cfg_alpha) * uncond_v_t

            sampled = sampled + v_t * dt

        sampled = torch.clamp(sampled, -1, 1)
        # Scale from [-1, 1] to [0, levels-1] for quantization
        quantized_levels = ((sampled + 1) / 2) * (self.acoustic_embeddings_levels - 1)
        output_codes = quantized_levels.round().long()
        # Note: (Jiaxin Deng) branchless mask; the boolean index_put_ this
        # replaces takes a data-dependent shape and cannot be captured.
        output_codes = torch.where(
            should_decode, output_codes, self._empty_audio_token_id
        )
        # Offset by the number of special tokens to avoid ID conflicts
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
            :, (len(AudioSpecialTokens) + self.model_args.semantic_codebook_size) :
        ] = -float("inf")

        semantic_code = semantic_logit.argmax(dim=-1, keepdim=True)

        acoustic_codes = self.decode_one_frame(semantic_code.squeeze(1), llm_hidden)

        return torch.concatenate([semantic_code, acoustic_codes], dim=1)
