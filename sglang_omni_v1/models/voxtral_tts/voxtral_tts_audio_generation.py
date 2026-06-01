# SPDX-License-Identifier: Apache-2.0
"""Voxtral TTS audio generation model.

Standalone pure-PyTorch implementation with a Llama-family causal decoder
backbone, the FlowMatchingAudioTransformer for acoustic code prediction,
and MultiVocabEmbeddings for audio token embedding.

Architecture overview::

    VoxtralTTSAudioGeneration
    ├── CausalLlamaDecoder          (text LLM backbone)
    │   ├── DecoderBlock × N
    │   │   ├── CausalSelfAttention (GQA via SDPA + RoPE)
    │   │   └── GatedFFN            (SiLU gate)
    │   └── RMSNorm                 (final norm)
    ├── FlowMatchingAudioTransformer (imported from acoustic_transformer)
    └── MultiVocabEmbeddings         (multi-codebook offset embedding)
"""

import logging
import math
import os
import re
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang_omni_v1.models.voxtral_tts.acoustic_transformer import (
    FlowMatchingAudioTransformer,
    MultimodalAudioModelArgs,
    from_nested_dict,
)

logger = logging.getLogger(__name__)

SUPPORTED_LANGS = {
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "ar": "Arabic",
    "hi": "Hindi",
    "it": "Italian",
    "pt": "Portuguese",
}


# ---------------------------------------------------------------------------
# Multi-codebook audio token embeddings
# ---------------------------------------------------------------------------


class MultiVocabEmbeddings(nn.Module):
    """Embed audio tokens from multiple codebooks into a shared space.

    Each codebook's token IDs are offset by the cumulative size of the
    preceding codebooks so that a single ``nn.Embedding`` table serves all
    codebooks.  The table size is rounded up to a multiple of 128 for
    efficient GPU memory alignment.
    """

    def __init__(self, audio_model_args: dict, embedding_dim: int) -> None:
        super().__init__()
        self.model_args = from_nested_dict(MultimodalAudioModelArgs, audio_model_args)
        self.codebook_sizes = list(
            self.model_args.get_codebook_sizes(pad_to_multiple=None)
        )
        # Cumulative offsets: codebook *i* tokens are shifted by sum(sizes[:i])
        offsets = [0]
        for sz in self.codebook_sizes[:-1]:
            offsets.append(offsets[-1] + sz)
        self.register_buffer(
            "offsets", torch.tensor(offsets, dtype=torch.long), persistent=False
        )
        total_vocab = sum(self.codebook_sizes)
        aligned_size = 128 * ((total_vocab + 127) // 128)
        self.embeddings = nn.Embedding(aligned_size, embedding_dim)

    def _rebuild_offsets(self) -> None:
        """Recompute the per-codebook offsets buffer.

        Must be called after ``to_empty()`` (meta-device init) because
        ``register_buffer`` values are replaced with uninitialised tensors.
        """
        vals = [0]
        for sz in self.codebook_sizes[:-1]:
            vals.append(vals[-1] + sz)
        self.offsets.copy_(torch.tensor(vals, dtype=torch.long))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [batch, n_codebooks, seq_len]
        shifted = input_ids + self.offsets[None, :, None]
        return self.embeddings(shifted)


# ---------------------------------------------------------------------------
# Llama-family causal decoder backbone
# ---------------------------------------------------------------------------


@dataclass
class LlamaConfig:
    """Minimal configuration for the Llama causal decoder."""

    vocab_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    max_position_embeddings: int
    rope_theta: float
    rms_norm_eps: float


class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation.

    When *residual* is supplied the residual connection is fused into the
    normalisation step to avoid an extra memory read/write pass.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            x = x + residual
        pre_norm = x  # keep the un-normalised value for the next residual
        out = self._norm(x).to(x.dtype) * self.scale
        return out, pre_norm


class RotaryPositionEmbedding(nn.Module):
    """Precomputed rotary position embedding (complex-exponential form).

    Builds a ``[max_positions, head_dim]`` cos/sin cache at init time.  At
    forward time the cache is sliced by the position indices and applied in
    the standard Neox-rotate-half convention.
    """

    def __init__(
        self,
        head_dim: int,
        max_positions: int,
        theta: float,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.max_positions = max_positions
        self.theta = theta
        self._dtype = dtype
        self._materialise_cache()

    def _materialise_cache(self) -> None:
        half = self.head_dim // 2
        freq_exponents = torch.arange(half, dtype=torch.float32) / half
        inv_freq = 1.0 / (self.theta**freq_exponents)
        positions = torch.arange(self.max_positions, dtype=torch.float32)
        # outer product → [max_positions, half]
        angles = torch.outer(positions, inv_freq)
        cos_cache = angles.cos().to(self._dtype)
        sin_cache = angles.sin().to(self._dtype)
        self.register_buffer("_cos", cos_cache, persistent=False)
        self.register_buffer("_sin", sin_cache, persistent=False)

    @staticmethod
    def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply Neox-style rotation: split in half, rotate, concatenate."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    def forward(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = positions.numel()
        cos = self._cos[positions.flatten()].unsqueeze(1).to(q.dtype)  # [S, 1, D/2]
        sin = self._sin[positions.flatten()].unsqueeze(1).to(q.dtype)

        q = self._rotate(q.view(seq_len, -1, self.head_dim), cos, sin).flatten(1)
        k = self._rotate(k.view(seq_len, -1, self.head_dim), cos, sin).flatten(1)
        return q, k


class CausalSelfAttention(nn.Module):
    """Grouped-query self-attention with rotary position embeddings.

    Uses ``F.scaled_dot_product_attention`` with ``enable_gqa=True`` so the
    KV heads are broadcast-expanded to the query head count inside the SDPA
    kernel rather than via an explicit ``repeat_interleave``.
    """

    def __init__(self, cfg: LlamaConfig) -> None:
        super().__init__()
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.head_dim
        q_dim = cfg.num_heads * cfg.head_dim
        kv_dim = cfg.num_kv_heads * cfg.head_dim

        self.q_proj = nn.Linear(cfg.hidden_size, q_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, kv_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, kv_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, cfg.hidden_size, bias=False)

        self.rotary_emb = RotaryPositionEmbedding(
            cfg.head_dim, cfg.max_position_embeddings, cfg.rope_theta
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        n = hidden.shape[0]
        q = self.q_proj(hidden)
        k = self.k_proj(hidden)
        v = self.v_proj(hidden)

        q, k = self.rotary_emb(positions, q, k)

        q = q.view(n, self.num_heads, self.head_dim)
        k = k.view(n, self.num_kv_heads, self.head_dim)
        v = v.view(n, self.num_kv_heads, self.head_dim)

        # Prepend cached KV pairs when decoding
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=0)
            v = torch.cat([kv_cache[1], v], dim=0)
        updated_kv = (k, v)

        # SDPA expects [B, H, S, D]; batch dim is always 1 here
        q = q.unsqueeze(0).transpose(1, 2)
        k = k.unsqueeze(0).transpose(1, 2)
        v = v.unsqueeze(0).transpose(1, 2)

        causal = kv_cache is None and n > 1
        out = F.scaled_dot_product_attention(q, k, v, is_causal=causal, enable_gqa=True)
        out = out.transpose(1, 2).squeeze(0).reshape(n, -1)
        return self.o_proj(out), updated_kv


class GatedFFN(nn.Module):
    """SiLU-gated feed-forward network (SwiGLU variant)."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class DecoderBlock(nn.Module):
    """Single Llama-style decoder block: pre-norm attention + pre-norm FFN."""

    def __init__(self, cfg: LlamaConfig) -> None:
        super().__init__()
        self.attn = CausalSelfAttention(cfg)
        self.ffn = GatedFFN(cfg.hidden_size, cfg.intermediate_size)
        self.norm1 = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.norm2 = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # Fused residual + pre-norm before attention
        if residual is None:
            residual = hidden
            hidden, _ = self.norm1(hidden)
        else:
            hidden, residual = self.norm1(hidden, residual)

        hidden, new_kv = self.attn(positions, hidden, kv_cache)

        # Fused residual + pre-norm before FFN
        hidden, residual = self.norm2(hidden, residual)
        hidden = self.ffn(hidden)
        return hidden, residual, new_kv


class CausalLlamaDecoder(nn.Module):
    """Llama-family causal decoder (embedding + N blocks + final norm)."""

    def __init__(self, cfg: LlamaConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.blocks = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.num_layers)])
        self.final_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        positions: torch.Tensor,
        past_kv: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        hidden = inputs_embeds
        residual: torch.Tensor | None = None
        new_kv_list: list[tuple[torch.Tensor, torch.Tensor]] = []
        for idx, block in enumerate(self.blocks):
            cached = past_kv[idx] if past_kv is not None else None
            hidden, residual, kv = block(positions, hidden, residual, cached)
            new_kv_list.append(kv)
        hidden, _ = self.final_norm(hidden, residual)
        return hidden, new_kv_list


# ---------------------------------------------------------------------------
# Weight key mapping: Mistral checkpoint → our parameter names
# ---------------------------------------------------------------------------

# Per-layer Mistral key suffix → our DecoderBlock key suffix
_LAYER_KEY_MAP: dict[str, str] = {
    "attention.wq.weight": "attn.q_proj.weight",
    "attention.wk.weight": "attn.k_proj.weight",
    "attention.wv.weight": "attn.v_proj.weight",
    "attention.wo.weight": "attn.o_proj.weight",
    "attention_norm.weight": "norm1.scale",
    "feed_forward.w1.weight": "ffn.gate.weight",
    "feed_forward.w2.weight": "ffn.down.weight",
    "feed_forward.w3.weight": "ffn.up.weight",
    "ffn_norm.weight": "norm2.scale",
}

# Global (non-layer) Mistral key → our parameter name
_GLOBAL_KEY_MAP: dict[str, str] = {
    "norm.weight": "final_norm.scale",
    "mm_audio_embeddings.tok_embeddings.weight": "embed_tokens.weight",
}


def _remap_checkpoint_key(name: str) -> str | None:
    """Map a Mistral-format checkpoint key to our parameter name.

    Returns ``None`` if the key does not belong to the LLM backbone.
    """
    if name in _GLOBAL_KEY_MAP:
        return _GLOBAL_KEY_MAP[name]

    # layers.<N>.<suffix> → blocks.<N>.<mapped_suffix>
    m = re.match(r"^layers\.(\d+)\.(.+)$", name)
    if m is not None:
        layer_idx, suffix = m.group(1), m.group(2)
        mapped = _LAYER_KEY_MAP.get(suffix)
        if mapped is not None:
            return f"blocks.{layer_idx}.{mapped}"
    return None


def _interleave_qk_weight(w: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
    """Re-order Q/K weight rows from Mistral's grouped layout to the
    interleaved layout expected by the Neox-style rotary convention.

    Mistral stores each head's half-dimensions contiguously as
    ``[head, half_dim, 2, hidden]``; we transpose to ``[head, 2, half_dim, hidden]``
    which interleaves the even/odd dimensions the way our RoPE expects.
    """
    total = n_heads * head_dim
    half = head_dim // 2
    return w.view(n_heads, half, 2, -1).transpose(1, 2).reshape(total, -1)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class VoxtralTTSAudioGeneration(nn.Module):
    """Voxtral TTS audio generation model.

    Combines a Llama causal decoder for autoregressive text-to-semantic-code
    generation with a FlowMatchingAudioTransformer for acoustic code
    prediction and a MultiVocabEmbeddings layer for audio token lookup.
    """

    def __init__(
        self,
        text_config,
        audio_model_args: dict,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.n_heads = text_config.n_heads
        self.n_kv_heads = text_config.n_kv_heads
        self.head_dim = text_config.head_dim
        self.hidden_size = text_config.dim

        cfg = LlamaConfig(
            vocab_size=text_config.vocab_size,
            hidden_size=text_config.dim,
            num_layers=text_config.n_layers,
            num_heads=text_config.n_heads,
            num_kv_heads=text_config.n_kv_heads,
            head_dim=text_config.head_dim,
            intermediate_size=text_config.hidden_dim,
            max_position_embeddings=text_config.max_seq_len,
            rope_theta=text_config.rope_theta,
            rms_norm_eps=text_config.norm_eps,
        )
        self.language_model = CausalLlamaDecoder(cfg)
        self.acoustic_transformer = FlowMatchingAudioTransformer(audio_model_args)
        self.audio_token_embedding = MultiVocabEmbeddings(
            audio_model_args=audio_model_args,
            embedding_dim=embedding_dim,
        )

    # ---- forward helpers --------------------------------------------------

    def forward_llm(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values=None,
        use_cache: bool = True,
        do_layer_debug: bool = False,
    ) -> tuple[torch.Tensor, list | None]:
        """Run the LLM backbone.

        *inputs_embeds* may be ``[B, S, D]`` or ``[S, D]``; the batch
        dimension is squeezed to ``[S, D]`` for the unbatched decoder.

        Returns ``(hidden_states, kv_cache)`` where *hidden_states* has
        a leading batch dimension added back for caller convenience.
        """
        embeds = inputs_embeds.squeeze(0) if inputs_embeds.dim() == 3 else inputs_embeds
        positions = position_ids.flatten()

        if do_layer_debug and past_key_values is None:
            hidden, kv = self._debug_forward(embeds, positions)
        else:
            hidden, kv = self.language_model(embeds, positions, past_key_values)

        return hidden.unsqueeze(0), kv if use_cache else None

    @torch.no_grad()
    def _debug_forward(
        self,
        embeds: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Layer-by-layer forward pass for debugging / profiling."""
        model = self.language_model
        hidden, residual = embeds, None
        kv_list: list[tuple[torch.Tensor, torch.Tensor]] = []
        for block in model.blocks:
            hidden, residual, kv = block(positions, hidden, residual)
            kv_list.append(kv)
        hidden, _ = model.final_norm(hidden, residual)
        return hidden, kv_list

    # ---- weight loading ---------------------------------------------------

    def load_weights(self, checkpoint_dir: str) -> None:
        """Load weights from a Mistral-format safetensors checkpoint."""
        import glob

        from safetensors import safe_open

        shard_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
        if not shard_paths:
            raise FileNotFoundError(f"No .safetensors files found in {checkpoint_dir}")

        n_heads = self.n_heads
        n_kv_heads = self.n_kv_heads
        head_dim = self.head_dim

        llm_state: dict[str, torch.Tensor] = {}
        counters = {"llm": 0, "acoustic": 0, "embedding": False}

        for path in shard_paths:
            with safe_open(path, framework="pt", device="cpu") as fp:
                for ckpt_key in fp.keys():
                    tensor = fp.get_tensor(ckpt_key)
                    self._dispatch_weight(
                        ckpt_key,
                        tensor,
                        llm_state,
                        counters,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                    )

        missing, unexpected = self.language_model.load_state_dict(
            llm_state, strict=False
        )
        logger.info(
            "LLM: %d loaded, %d missing, %d unexpected",
            counters["llm"],
            len(missing),
            len(unexpected),
        )
        if missing:
            logger.warning("Missing keys (first 5): %s", missing[:5])
        if unexpected:
            logger.warning("Unexpected keys (first 5): %s", unexpected[:5])
        logger.info(
            "Acoustic transformer: %d loaded | Audio embedding: %s",
            counters["acoustic"],
            counters["embedding"],
        )

    def _dispatch_weight(
        self,
        ckpt_key: str,
        tensor: torch.Tensor,
        llm_state: dict[str, torch.Tensor],
        counters: dict,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
    ) -> None:
        """Route a single checkpoint tensor to the correct sub-module."""
        # --- LLM backbone ---
        mapped = _remap_checkpoint_key(ckpt_key)
        if mapped is not None:
            if "attention.wq." in ckpt_key:
                tensor = _interleave_qk_weight(tensor, n_heads, head_dim)
            elif "attention.wk." in ckpt_key:
                tensor = _interleave_qk_weight(tensor, n_kv_heads, head_dim)
            llm_state[mapped] = tensor
            counters["llm"] += 1
            return

        # --- Acoustic transformer ---
        prefix = "acoustic_transformer."
        if ckpt_key.startswith(prefix):
            param_name = ckpt_key[len(prefix) :]
            self.acoustic_transformer.load_weight((param_name, tensor))
            counters["acoustic"] += 1
            return

        # --- Audio codebook embedding ---
        if ckpt_key == (
            "mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight"
        ):
            self.audio_token_embedding.embeddings.weight.data.copy_(tensor)
            counters["embedding"] = True

    # ---- factory ----------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls, checkpoint_dir: str, device: str = "cuda:0"
    ) -> tuple["VoxtralTTSAudioGeneration", dict[str, torch.Tensor], object]:
        """Build and load the full model from a Mistral-format checkpoint."""
        from dataclasses import asdict

        from sglang_omni_v1.models.voxtral_tts.model_config import VoxtralModelConfig

        config = VoxtralModelConfig.from_model_path(checkpoint_dir)
        audio_args = asdict(config.audio_model_args)

        logger.info("Loading model from %s …", checkpoint_dir)
        t0 = time.perf_counter()
        mem0 = torch.cuda.memory_allocated(device) if device.startswith("cuda") else 0

        # Fast-init on meta device, then materialise on CPU
        with torch.device("meta"):
            model = cls(
                text_config=config.text_config,
                audio_model_args=audio_args,
                embedding_dim=config.text_config.dim,
            )
        model = model.to_empty(device="cpu")

        # Rebuild non-persistent buffers lost during meta init
        for block in model.language_model.blocks:
            block.attn.rotary_emb._materialise_cache()

        model.audio_token_embedding._rebuild_offsets()

        at = model.acoustic_transformer
        at._timesteps = torch.linspace(0, 1, at._acoustic_decode_iters)
        dim = at.acoustic_transformer_args.dim
        at.time_embedding.inv_freq = torch.exp(
            -math.log(10_000.0) * torch.arange(dim // 2).float() / (dim // 2)
        )

        model.load_weights(checkpoint_dir)
        logger.info("Weight loading: %.2f s", time.perf_counter() - t0)

        model = model.to(dtype=torch.bfloat16, device=device).eval()

        mem1 = torch.cuda.memory_allocated(device) if device.startswith("cuda") else 0
        logger.info(
            "Model ready: %.2f GiB GPU, %.2f s total",
            (mem1 - mem0) / (1 << 30),
            time.perf_counter() - t0,
        )

        # Voice embeddings (speaker conditioning vectors)
        voice_embeddings: dict[str, torch.Tensor] = {}
        voice_dir = os.path.join(checkpoint_dir, "voice_embedding")
        if os.path.isdir(voice_dir):
            for fname in sorted(os.listdir(voice_dir)):
                if fname.endswith(".pt"):
                    name = fname.removesuffix(".pt")
                    emb = torch.load(
                        os.path.join(voice_dir, fname),
                        map_location=device,
                        weights_only=True,
                    )
                    voice_embeddings[name] = emb.to(dtype=torch.bfloat16)
            logger.info(
                "Loaded %d voice embeddings: %s",
                len(voice_embeddings),
                list(voice_embeddings.keys()),
            )

        return model, voice_embeddings, config
