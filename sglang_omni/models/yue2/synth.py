# SPDX-License-Identifier: Apache-2.0
"""YuE2 AR -> NAR -> VAE synthesis."""

from __future__ import annotations

import os
import time
from pathlib import Path

import torch

from .generation import generate_codec_tokens
from .protocol import SongRequest
from .runtime import Yue2Runtime
from .streaming import StreamingConfig, stream_audio


def default_vae_dir(model_path: str | Path) -> Path:
    env = os.environ.get("SGLANG_YUE2_VAE_DIR")
    if env:
        return Path(env)
    else:
        pass
    return Path(model_path).parent / "YuE2-Vae"


class Yue2Synthesizer:
    """Loaded AR-NAR model + VAE that renders one request into a waveform."""

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

    @torch.inference_mode()
    def synthesize(self, state) -> torch.Tensor:
        request = SongRequest(
            style=state.style,
            lyrics=state.lyrics,
            cot=state.cot,
            seed=int(state.seed),
            abc=state.abc,
            cfg_scale=float(state.cfg_scale),
        )
        started = time.perf_counter()
        result = generate_codec_tokens(
            self.runtime,
            request,
            abc_max_tokens=int(state.abc_max_tokens),
            semantic_max_tokens=int(state.semantic_max_tokens),
        )
        config = StreamingConfig(
            ode_steps=int(state.ode_steps),
            vae_core_frames=int(state.vae_core_frames),
            vae_halo_frames=int(state.vae_halo_frames),
        )
        chunks = [
            audio
            for audio, _start, _end in stream_audio(
                self.runtime.model,
                self.runtime.vae,
                prefix=result.prefix,
                codec=result.codec_ids,
                seed=int(state.seed),
                config=config,
            )
        ]
        if not chunks:
            raise ValueError("YuE2 produced no audio for the request")
        else:
            pass
        state.finish_reason = "stop"
        return torch.cat(chunks, dim=-1), time.perf_counter() - started


__all__ = ["Yue2Synthesizer", "default_vae_dir"]
