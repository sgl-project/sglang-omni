# SPDX-License-Identifier: Apache-2.0
"""YuE2 AR -> NAR -> VAE synthesis."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .batched import run_batched_ar
from .cuda_graph import default_session_pool
from .nar import synthesize as nar_synthesize
from .nar_fast import synthesize_batched, synthesize_from_session
from .protocol import (
    CODEC_OFFSET,
    CONTEXT,
    GenerationConfig,
    Sampling,
    SongRequest,
    negative_prefix,
    token_prefixes,
)
from .runtime import Yue2Runtime
from .song import generate_song_tokens, song_token_budget


def default_vae_dir(model_path: str | Path) -> Path:
    env = os.environ.get("SGLANG_YUE2_VAE_DIR")
    if env:
        return Path(env)
    else:
        pass
    return Path(model_path).parent / "YuE2-Vae"


def build_generation_config(state) -> GenerationConfig:
    abc_max = int(state.abc_max_tokens)
    semantic_max = int(state.semantic_max_tokens)
    return GenerationConfig(
        abc=Sampling(0.7, 0.9, 30, 1.005, 100, min(32, abc_max), abc_max),
        semantic=Sampling(1.0, 0.95, 100, 1.2, 50, min(200, semantic_max), semantic_max),
        ode_steps=int(state.ode_steps),
    )


@dataclass
class BatchItem:
    """Minimal Req-shaped adapter for ``batched.run_batched_ar``."""

    extra: dict = field(default_factory=dict)
    sampling_params: object = None


class Yue2Synthesizer:
    """Loaded AR-NAR model + VAE that renders one or several requests.

    Uses the SGLang-YuE2 optimizations: one pooled ``GraphAR`` session with the
    fused decode+sample step graph for AR, fused NAR on the borrowed KV
    (``synthesize_from_session``), and tiled VAE decode. ``synthesize_batch``
    additionally batches concurrent requests through ``batched.run_batched_ar``
    + ``nar_fast.synthesize_batched`` + same-frame VAE decode; it falls back to
    per-request synthesis for a mixed batch the batched path cannot serve.
    """

    def __init__(
        self,
        model_path: str | Path,
        vae_dir: str | Path | None = None,
        device: str | torch.device | None = None,
    ):
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        else:
            device = torch.device(device)
        self.model_path = str(model_path)
        self.runtime = Yue2Runtime.from_paths(
            model_dir=model_path,
            vae_dir=vae_dir if vae_dir is not None else default_vae_dir(model_path),
            device=device,
        )

    @property
    def device(self) -> torch.device:
        return self.runtime.device

    def requests_for_states(self, states) -> list[SongRequest]:
        return [
            SongRequest(
                style=state.style,
                lyrics=state.lyrics,
                cot=state.cot,
                seed=int(state.seed),
                abc=state.abc,
                cfg_scale=float(state.cfg_scale),
            )
            for state in states
        ]

    @torch.inference_mode()
    def decode_audio(self, latents_list, states) -> list[torch.Tensor]:
        """Tiled VAE decode, batching rows that share a latent frame count."""
        runtime = self.runtime
        buckets: dict[tuple, list[int]] = {}
        for index, latents in enumerate(latents_list):
            state = states[index]
            key = (int(latents.shape[0]), int(state.vae_core_frames),
                   int(state.vae_halo_frames))
            buckets.setdefault(key, []).append(index)

        audio: list[torch.Tensor] = [None] * len(latents_list)
        for (frames, core, halo), indices in buckets.items():
            stacked = torch.cat(
                [latents_list[i].unsqueeze(0).transpose(1, 2) for i in indices], dim=0
            ).to(device=runtime.device, dtype=torch.float32)
            decoded = runtime.vae.decode_tiled(
                stacked, core_frames=core, halo_frames=halo, output_device="cpu")
            for row, index in enumerate(indices):
                audio[index] = decoded[row]
        return audio

    @torch.inference_mode()
    def synthesize(self, state):
        runtime = self.runtime
        request = self.requests_for_states([state])[0]
        config = build_generation_config(state)
        prefix = token_prefixes(request, runtime.tokenizer)
        negative = negative_prefix(request, runtime.tokenizer) if float(state.cfg_scale) != 1.0 else None
        prefixes = [prefix, negative] if negative is not None else [prefix]

        started = time.perf_counter()
        session = default_session_pool.acquire(
            runtime.model, prefixes, song_token_budget(config.abc, config.semantic))
        try:
            abc_ids, semantic_ids, _timing, fed_codec = generate_song_tokens(
                runtime.model, session, runtime.tokenizer, request,
                abc_sampling=config.abc, semantic_sampling=config.semantic,
                cfg_scale=float(state.cfg_scale), negative_prefix_ids=negative)
            semantic_prefix = token_prefixes(request, runtime.tokenizer, abc_ids=abc_ids)
            codec = [int(token) - CODEC_OFFSET for token in semantic_ids]
            try:
                latents = synthesize_from_session(
                    runtime.model, session, prefix_len=len(semantic_prefix),
                    codec=codec, seed=request.seed, fed_codec=int(fed_codec),
                    steps=config.ode_steps, context=CONTEXT)
            except ValueError:
                latents = nar_synthesize(
                    runtime.model, prefix=semantic_prefix, codec=codec,
                    seed=request.seed, steps=config.ode_steps, context=CONTEXT,
                    attention="sdpa", offload_ar=False, query_chunk_size=None)
        finally:
            default_session_pool.release(session)

        state.finish_reason = "stop"
        return self.decode_audio([latents], [state])[0], time.perf_counter() - started

    @torch.inference_mode()
    def synthesize_batch(self, states):
        """Render a whole batch; returns ``[(waveform, seconds)]`` per state."""
        if len(states) <= 1:
            return [self.synthesize(states[0])]
        else:
            pass

        runtime = self.runtime
        configs = [build_generation_config(state) for state in states]
        requests = self.requests_for_states(states)
        items = [
            BatchItem(extra={"yue2_request": request,
                             "yue2_prefix": list(token_prefixes(request, runtime.tokenizer))})
            for request in requests
        ]

        started = time.perf_counter()
        try:
            run_batched_ar(runtime.model, runtime.tokenizer, items, lambda _sp: configs[0])
        except Exception:
            return [self.synthesize(state) for state in states]

        session = items[0].extra["yue2_session"]
        branches = [int(item.extra["yue2_session_branch"]) for item in items]
        prefix_lens = [len(item.extra["yue2_semantic_prefix"]) for item in items]
        codecs = [
            [int(token) - CODEC_OFFSET for token in item.extra["yue2_semantic_ids"]]
            for item in items
        ]
        seeds = [request.seed for request in requests]
        try:
            latents_list = synthesize_batched(
                runtime.model, session, branches, prefix_lens, codecs, seeds,
                configs[0].ode_steps)
        finally:
            lease = items[0].extra.get("yue2_session_lease")
            if lease is not None:
                for _ in items:
                    lease.release()
            else:
                pass

        audio = self.decode_audio(latents_list, states)
        elapsed = time.perf_counter() - started
        for state in states:
            state.finish_reason = "stop"
        return [(waveform, elapsed) for waveform in audio]


__all__ = ["BatchItem", "Yue2Synthesizer", "build_generation_config", "default_vae_dir"]
