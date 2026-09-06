# SPDX-License-Identifier: Apache-2.0
"""Breeze's Llama-style depth decoder, using the pinned Transformers primitives.

Forward/token layout follows breezeblue-ai/breeze-tts (Apache-2.0), revision
43e2ea1595297c4059477e2e4a300653761c759b. Depth KV state lasts one audio frame,
not one speech request; no mutable cache is stored on the module.
"""

import torch
from torch import nn
from transformers import LlamaConfig, LlamaModel, StaticCache

from .sampling import BatchedSampling, SamplingConfig, sample_logits_batched


class BreezeDepthDecoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        cfg = LlamaConfig(**config)
        cfg._attn_implementation = "sdpa"
        self.config = cfg
        self.num_codebooks = config["num_codebooks"]
        self.audio_embed_size = config["audio_embed_size"]
        self.model = LlamaModel(cfg)
        self.model.embed_tokens = nn.Embedding(
            self.num_codebooks * cfg.vocab_size, self.audio_embed_size
        )
        self.model.inputs_embeds_projector = nn.Linear(
            self.audio_embed_size, cfg.hidden_size, bias=False
        )
        backbone_hidden_size = config["backbone_hidden_size"]
        if backbone_hidden_size != self.audio_embed_size:
            raise ValueError(
                "Breeze-TTS-2 requires matching backbone/audio embedding sizes"
            )
        self.codebooks_head = nn.Module()
        self.codebooks_head.weight = nn.Parameter(
            torch.empty(self.num_codebooks - 1, cfg.hidden_size, cfg.vocab_size)
        )
        self._head_fp32: torch.Tensor | None = None
        self.graphs: "BreezeDepthGraphs | None" = None

    def _fp32_head(self) -> torch.Tensor:
        """The codebook heads in fp32, materialized once instead of per step."""
        weight = self.codebooks_head.weight
        cached = self._head_fp32
        if cached is None or cached.device != weight.device:
            cached = weight.float()
            self._head_fp32 = cached
        return cached

    def embed_frames(self, codes: torch.Tensor) -> torch.Tensor:
        offsets = (
            torch.arange(self.num_codebooks, device=codes.device)
            * self.config.vocab_size
        )
        return self.model.embed_tokens(codes.long() + offsets).sum(dim=-2)

    @torch.no_grad()
    def decode_frames(
        self,
        hidden: torch.Tensor,
        first_codes: torch.Tensor,
        params: list[SamplingConfig],
        frames: torch.Tensor,
        *,
        codebook_size: int = 2048,
        sampling: BatchedSampling | None = None,
    ) -> torch.Tensor:
        """Decode one complete codec frame per logical request in one batch."""
        batch_size = len(params)
        if (
            batch_size == 0
            or frames.numel() != batch_size
            or first_codes.numel() != batch_size
            or hidden.ndim != 2
            or hidden.shape[0] != 2 * batch_size
        ):
            raise ValueError(
                "Breeze depth decode requires two hidden rows and one frame index "
                "per logical request"
            )
        if sampling is None:
            sampling = BatchedSampling(params, hidden.device)
        if self.graphs is not None:
            replayed = self.graphs.decode(
                hidden, first_codes, frames, sampling, codebook_size=codebook_size
            )
            if replayed is not None:
                return replayed
        return self._decode_frames_eager(
            hidden, first_codes, frames, sampling, codebook_size=codebook_size
        )

    def _decode_frames_eager(
        self,
        hidden: torch.Tensor,
        first_codes: torch.Tensor,
        frames: torch.Tensor,
        sampling: BatchedSampling,
        *,
        codebook_size: int,
        cache: StaticCache | None = None,
    ) -> torch.Tensor:
        """The fifteen depth steps, written so the whole loop can be captured."""
        if cache is not None:
            cache.reset()
        # [backbone hidden, c0] prefill predicts c1 with head 0. Every later
        # position embeds c(k) from its own codebook before predicting c(k+1).
        first_codes = first_codes.reshape(sampling.size)
        paired_codes = first_codes.repeat_interleave(2)
        embeds = torch.stack((hidden, self.model.embed_tokens(paired_codes)), dim=1)
        head = self._fp32_head()
        codes = [first_codes]
        for codebook in range(1, self.num_codebooks):
            extra = (
                {} if cache is None else {"cache_position": self._positions(codebook)}
            )
            output = self.model(
                inputs_embeds=self.model.inputs_embeds_projector(embeds),
                past_key_values=cache,
                use_cache=True,
                **extra,
            )
            if cache is None:
                cache = output.past_key_values
            logits = output.last_hidden_state[:, -1].float() @ head[codebook - 1]
            code = sample_logits_batched(
                logits,
                sampling,
                sampling.positions(frames, codebook),
                codebook_size=codebook_size,
            )
            codes.append(code)
            embeds = self.model.embed_tokens(
                code.repeat_interleave(2) + codebook * self.config.vocab_size
            ).unsqueeze(1)
        return torch.stack(codes, dim=1)

    def _positions(self, codebook: int) -> torch.Tensor:
        """Cache slots written by one depth step; step 1 writes both prefill rows."""
        cached = getattr(self, "_cache_positions", None)
        if cached is None:
            device = self.codebooks_head.weight.device
            cached = [None, torch.tensor([0, 1], device=device, dtype=torch.long)]
            cached += [
                torch.tensor([index + 1], device=device, dtype=torch.long)
                for index in range(1, self.num_codebooks)
            ]
            self._cache_positions = cached
        return cached[codebook]

    @torch.no_grad()
    def decode_frame(
        self,
        hidden: torch.Tensor,
        first_code: torch.Tensor,
        params: SamplingConfig,
        frame: int = 0,
        *,
        codebook_size: int = 2048,
    ) -> torch.Tensor:
        return self.decode_frames(
            hidden,
            first_code.reshape(1),
            [params],
            torch.tensor([frame], device=hidden.device, dtype=torch.long),
            codebook_size=codebook_size,
        )[0]


class BreezeDepthGraphs:
    """CUDA-graph replays of the whole depth loop, one graph per batch bucket.

    Every depth step has static shapes -- ``2 x batch`` rows, one position, a
    fixed fifteen iterations -- so the entire loop can be captured. That removes
    the per-step host dispatch and the gaps between the loop's small kernels,
    which is what keeps eager depth decoding above the frame budget. Rows never
    interact, so a short batch is padded up to its bucket and the padding rows
    are discarded.
    """

    def __init__(
        self,
        decoder: BreezeDepthDecoder,
        buckets: list[int],
        *,
        codebook_size: int,
    ) -> None:
        if not buckets or any(bucket < 1 for bucket in buckets):
            raise ValueError("Breeze depth graph buckets must be positive")
        self._decoder = decoder
        self._codebook_size = codebook_size
        self._buckets = sorted(set(buckets))
        self._captured: dict[int, dict] = {}
        self._pool = None

    @property
    def buckets(self) -> list[int]:
        return list(self._buckets)

    def capture(self) -> None:
        decoder = self._decoder
        device = decoder.codebooks_head.weight.device
        dtype = decoder.codebooks_head.weight.dtype
        hidden_size = decoder.audio_embed_size
        decoder._fp32_head()
        for bucket in self._buckets:
            params = [SamplingConfig() for _ in range(bucket)]
            state = {
                "hidden": torch.zeros(
                    2 * bucket, hidden_size, device=device, dtype=dtype
                ),
                "first": torch.zeros(bucket, device=device, dtype=torch.long),
                "frames": torch.zeros(bucket, device=device, dtype=torch.long),
                "sampling": BatchedSampling(params, device),
                "cache": StaticCache(
                    decoder.config, max_cache_len=decoder.num_codebooks + 1
                ),
            }
            run = lambda state=state: decoder._decode_frames_eager(
                state["hidden"],
                state["first"],
                state["frames"],
                state["sampling"],
                codebook_size=self._codebook_size,
                cache=state["cache"],
            )
            with torch.no_grad():
                # Warm up so lazy cache allocation and any one-time kernel setup
                # happen before capture rather than inside it.
                for _ in range(3):
                    run()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=self._pool):
                    state["codes"] = run()
                self._pool = graph.pool()
            state["graph"] = graph
            self._captured[bucket] = state

    def decode(
        self,
        hidden: torch.Tensor,
        first_codes: torch.Tensor,
        frames: torch.Tensor,
        sampling: BatchedSampling,
        *,
        codebook_size: int,
    ) -> torch.Tensor | None:
        """Replay a captured loop, or return None when no bucket fits."""
        if codebook_size != self._codebook_size:
            return None
        bucket = next((size for size in self._buckets if size >= sampling.size), None)
        state = self._captured.get(bucket)
        if state is None:
            return None

        rows = sampling.size
        # Padding repeats the first request so the padded rows stay in range;
        # their tokens are dropped and cannot influence a real row.
        self._fill(state["hidden"], hidden, 2 * rows)
        self._fill(state["first"], first_codes.reshape(rows), rows)
        self._fill(state["frames"], frames, rows)
        target = state["sampling"]
        for name in (
            "cfg_scale",
            "temperature",
            "top_p",
            "repetition_penalty",
            "top_k",
            "seeds",
        ):
            self._fill(getattr(target, name), getattr(sampling, name), rows)
        state["graph"].replay()
        return state["codes"][:rows].clone()

    @staticmethod
    def _fill(buffer: torch.Tensor, values: torch.Tensor, rows: int) -> None:
        buffer[:rows].copy_(values)
        if buffer.shape[0] > rows:
            buffer[rows:].copy_(values[:1].expand_as(buffer[rows:]))
