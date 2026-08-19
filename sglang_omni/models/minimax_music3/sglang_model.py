# SPDX-License-Identifier: Apache-2.0
"""MiniMax Music 3 audio modules bolted onto SGLang's Qwen3 backbone."""

from __future__ import annotations

import logging
from types import MethodType
from typing import Any

import torch
from torch import nn

from .checkpoint import load_audio_state, load_json, resolve_checkpoint
from .constants import AR_CFG_SCALE, AR_CFG_TOP_K
from .prompt import AUDIO_CODE_OFFSET, SPECIAL_TOKEN_IDS
from .rvq_decoder import RVQDepthDecoder, sample_topk_seeded

logger = logging.getLogger(__name__)

_AUDIO_EMBEDDING_KEY = "model.audio_extra_embedding.weight"
_C0_VOCAB_SIZE = 16384


def attach_minimax_modules(model: Any, checkpoint_root: str) -> None:
    """Install the MiniMax audio modules on a loaded Qwen3 backbone."""

    paths = resolve_checkpoint(checkpoint_root)
    config = load_json(paths.config_path)
    audio_state = load_audio_state(paths)

    parameter = next(model.parameters())
    device, dtype = parameter.device, parameter.dtype
    hidden_size = int(config["hidden_size"])
    audio_vocab_size = int(config["audio_vocab_size"])
    num_codebooks = int(config["audio_num_codebooks"])

    expected = (audio_vocab_size * (num_codebooks - 1), hidden_size)
    weight = audio_state.get(_AUDIO_EMBEDDING_KEY)
    if weight is None or tuple(weight.shape) != expected:
        raise ValueError(
            f"MiniMax Music 3 audio embedding mismatch: expected {expected}, "
            f"got {None if weight is None else tuple(weight.shape)}"
        )
    audio_embeddings = nn.Embedding(*expected).to(device=device, dtype=dtype)
    audio_embeddings.weight.data.copy_(weight.to(device=device, dtype=dtype))

    rvq_decoder = (
        RVQDepthDecoder(
            hidden_size=hidden_size,
            num_layers=int(config["decoder_num_layers"]),
            num_heads=int(config["decoder_num_heads"]),
            intermediate_size=int(config["decoder_intermediate_size"]),
            audio_vocab_size=audio_vocab_size,
            num_codebooks=num_codebooks,
        )
        .to(device=device, dtype=dtype)
        .eval()
    )
    rvq_decoder.load_checkpoint_state(audio_state)

    model.audio_embeddings = audio_embeddings
    model.rvq_decoder = rvq_decoder
    model.audio_vocab_size = audio_vocab_size
    model.num_codebooks = num_codebooks
    model.mel_token_offset = AUDIO_CODE_OFFSET
    model.stop_mel_token = SPECIAL_TOKEN_IDS["<|audio_end|>"]
    model.cfg_token = SPECIAL_TOKEN_IDS["<|audio_cfg|>"]
    model.audio_embedding_offsets = (
        torch.arange(num_codebooks - 1, device=device) * audio_vocab_size
    ).unsqueeze(0)
    model.frame_embedding_scale = num_codebooks**-0.5
    model.c0_logit_ids = _build_c0_logit_ids(model.stop_mel_token, device)
    model.graph_feedback_buffer = None

    _use_unfused_qk_norm(model)
    _route_forward_batch_input_embeds(model)
    logger.info(
        f"MiniMax Music 3 audio modules attached device={device} dtype={dtype} codebooks={num_codebooks} audio_vocab_size={audio_vocab_size} rvq_parameters={sum((p.numel() for p in rvq_decoder.parameters()))}"
    )


def embed_audio_frames(model: Any, codes: torch.Tensor) -> torch.Tensor:
    """Compose one conditioning embedding per row from all eight codebooks.

    Args:
        codes: [rows, num_codebooks] codebook indices (c0 is 0-based).

    Returns:
        [rows, hidden_size] in model dtype.
    """
    embed_tokens = model.get_input_embeddings()
    c0 = embed_tokens(codes[:, 0] + model.mel_token_offset)
    extra = model.audio_embeddings(codes[:, 1:] + model.audio_embedding_offsets).sum(
        dim=1
    )
    return (c0 + extra.to(c0.dtype)) * model.frame_embedding_scale


def apply_depth_cfg(logits: torch.Tensor) -> torch.Tensor:
    """Guide paired depth-head rows. Rows are [cond, uncond] interleaved."""
    cond, uncond = logits[0::2].float(), logits[1::2].float()
    return uncond + (cond - uncond) * AR_CFG_SCALE


def apply_cfg(cond: torch.Tensor, uncond: torch.Tensor) -> torch.Tensor:
    """Guide cond by uncond, restricted to cond's own top candidates.

    note (chenyang): the mask is a second, separate restriction from the top-k
    the sampler applies afterwards, and dropping it widens the distribution the
    reference drew from.
    """
    guided = uncond + (cond - uncond) * AR_CFG_SCALE
    threshold = torch.topk(cond, AR_CFG_TOP_K, dim=-1).values[..., -1, None]
    return guided.masked_fill(cond < threshold, -float("inf"))


def select_c0_logits(model: Any, logits: torch.Tensor) -> torch.Tensor:
    """Narrow backbone logits to the only ids c0 sampling can ever return.

    Returns [rows, 1 + 16384] in vocabulary order: <|audio_end|> first,
    then the c0 code range. Column 0 therefore means stop and column code+1
    means code.
    """
    return logits.index_select(1, model.c0_logit_ids).float()


def enable_rvq_depth_cuda_graph(model: Any, buckets: list[int]) -> None:
    """Capture the RVQ depth pass so it stops paying per-launch dispatch."""
    from .rvq_cuda_graph import RVQDepthCudaGraphRunner

    parameter = next(model.parameters())
    model.rvq_depth_graph = RVQDepthCudaGraphRunner(
        forward=lambda hidden, c0, seeds, positions, forced, replay: (
            _depth_decode_eager(model, hidden, c0, seeds, positions, forced, replay)
        ),
        device=parameter.device,
        dtype=parameter.dtype,
        hidden_size=model.rvq_decoder.hidden_size,
        num_codebooks=model.num_codebooks,
        buckets=buckets,
    )


def depth_decode(
    model: Any,
    hidden: torch.Tensor,
    c0: torch.Tensor,
    seeds: torch.Tensor,
    frame_positions: torch.Tensor,
    forced_codes: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the RVQ depth decoder for one audio frame, batched over rows."""
    rows = hidden.shape[0]
    if forced_codes is None:
        forced_codes = c0.new_zeros((rows, model.num_codebooks))
        replay = c0.new_zeros((), dtype=torch.bool)
    else:
        replay = c0.new_ones((), dtype=torch.bool)
    graph = getattr(model, "rvq_depth_graph", None)
    if graph is not None:
        replayed = graph(hidden, c0, seeds, frame_positions, forced_codes, replay)
        if replayed is not None:
            return replayed
    return _depth_decode_eager(
        model, hidden, c0, seeds, frame_positions, forced_codes, replay
    )


def _depth_decode_eager(
    model: Any,
    hidden: torch.Tensor,
    c0: torch.Tensor,
    seeds: torch.Tensor,
    frame_positions: torch.Tensor,
    forced_codes: torch.Tensor,
    replay: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode c1..c7 for rows interleaved as cond, uncond, cond, uncond.

    todo (chenyang): teach RVQDepthCudaGraphRunner which of its inputs and
    outputs are per-pair rather than per-row, so the seeds, positions, forced
    codes and returned tensors stop being duplicated across each pair and the
    captured buffers are sized for the halves that are actually read.
    """
    decoder = model.rvq_decoder
    embed_tokens = model.get_input_embeddings()
    cond_seeds = seeds[0::2]
    cond_positions = frame_positions[0::2]
    cond_forced = forced_codes[0::2]
    c0_embed = embed_tokens(c0 + model.mel_token_offset)
    seq = [
        decoder.projection(hidden).unsqueeze(1),
        decoder.projection(c0_embed).unsqueeze(1),
    ]
    codes = [c0]
    hidden_parts: list[torch.Tensor] = []
    ranks: list[torch.Tensor] = []
    for index in range(1, model.num_codebooks):
        out = decoder(torch.cat(seq, dim=1))[:, -1]
        hidden_parts.append(out.detach())
        guided = apply_depth_cfg(decoder.audio_heads[index - 1](out))
        code = sample_topk_seeded(guided, cond_seeds, cond_positions + index)
        reference = cond_forced[:, index].unsqueeze(1)
        ranks.append((guided > guided.gather(1, reference)).sum(dim=1))
        code = torch.where(replay, cond_forced[:, index], code)
        paired = code.repeat_interleave(2)
        codes.append(paired)
        if index < model.num_codebooks - 1:
            embedding = model.audio_embeddings(
                paired + (index - 1) * model.audio_vocab_size
            )
            seq.append(decoder.projection(embedding).unsqueeze(1))
    return (
        torch.stack(codes, dim=1),
        torch.cat(hidden_parts, dim=-1),
        torch.stack(ranks, dim=1).repeat_interleave(2, dim=0),
    )


def _build_c0_logit_ids(stop_token: int, device: torch.device) -> torch.Tensor:
    """Return the legal c0 token ids in vocabulary order, stop token first."""
    assert stop_token < AUDIO_CODE_OFFSET, "stop token must precede the code range"
    codes = torch.arange(
        AUDIO_CODE_OFFSET, AUDIO_CODE_OFFSET + _C0_VOCAB_SIZE, device=device
    )
    return torch.cat((torch.tensor([stop_token], device=device), codes))


def _forward_prepare_unfused_qk_norm(self, positions, hidden_states):
    from sglang.srt.models.utils import apply_qk_norm

    qkv, _ = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    q, k = apply_qk_norm(
        q=q,
        k=k,
        q_norm=self.q_norm,
        k_norm=self.k_norm,
        head_dim=self.head_dim,
        alt_stream=self.alt_stream,
        allow_inplace=False,
    )
    q, k = self.rotary_emb(positions, q, k)
    return q, k, v


def _use_unfused_qk_norm(model: Any) -> None:
    """Keep QK norm, but off flashinfer's fused in-place kernel."""
    for layer in model.model.layers:
        attention = getattr(layer, "self_attn", None)
        if attention is None:
            continue
        if attention.q_norm is None or attention.k_norm is None:
            raise RuntimeError(
                "MiniMax Music 3 expects Qwen3 QK norm layers to be present and loaded"
            )
        attention.forward_prepare_native = MethodType(
            _forward_prepare_unfused_qk_norm, attention
        )


def enable_graph_feedback(model: Any, max_batch_size: int) -> None:
    """Route decode conditioning through a fixed-address buffer."""
    if max_batch_size <= 0:
        raise ValueError("MiniMax Music 3 graph feedback buffer needs a positive size")
    parameter = next(model.parameters())
    model.graph_feedback_buffer = torch.zeros(
        (int(max_batch_size), int(model.config.hidden_size)),
        device=parameter.device,
        dtype=parameter.dtype,
    )


def _route_forward_batch_input_embeds(model: Any) -> None:
    """Feed decode conditioning that SGLang would otherwise drop."""
    backbone_forward = model.forward

    def forward(input_ids, positions, forward_batch, input_embeds=None, **kwargs):
        # todo (chenyang): remove getattr here
        buffer = getattr(model, "graph_feedback_buffer", None)
        if buffer is not None and forward_batch.forward_mode.is_decode():
            input_embeds = buffer[: input_ids.shape[0]]
        elif input_embeds is None:
            input_embeds = forward_batch.input_embeds
        return backbone_forward(
            input_ids, positions, forward_batch, input_embeds=input_embeds, **kwargs
        )

    model.forward = forward


__all__ = [
    "attach_minimax_modules",
    "enable_graph_feedback",
    "enable_rvq_depth_cuda_graph",
    "depth_decode",
    "embed_audio_frames",
    "select_c0_logits",
]
