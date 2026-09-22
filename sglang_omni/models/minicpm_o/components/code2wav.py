# SPDX-License-Identifier: Apache-2.0
"""Vocode MiniCPM-o codec tokens with a cached speaker reference."""

from __future__ import annotations

import io
import logging
import os
import threading
import time
from collections import OrderedDict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from sglang_omni.models.weight_loader import resolve_dtype, resolve_model_path
from sglang_omni.preprocessing.cache_key import hash_bytes, reference_path_cache_key

FLOW_DTYPES = (torch.float32, torch.float16, torch.bfloat16)

OUTPUT_SAMPLE_RATE = 24000
CODEC_TOKEN_RATE = 25
SAMPLES_PER_CODEC_TOKEN = OUTPUT_SAMPLE_RATE // CODEC_TOKEN_RATE
DEFAULT_PROMPT_CACHE_CAPACITY = 32
DEFAULT_REFERENCE_WORKERS = 8

logger = logging.getLogger(__name__)


def positive_int_env(name: str, default: int) -> int:
    """Read a positive integer configuration value."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    else:
        try:
            value = int(raw)
        except ValueError as error:
            raise ValueError(f"{name} must be an integer, got {raw!r}") from error
        if value < 1:
            raise ValueError(f"{name} must be positive, got {value}")
        else:
            return value


class MiniCPMOCode2Wav(nn.Module):
    """Convert codec tokens into a float32 waveform with Token2wav."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda",
        dtype: str | torch.dtype | None = None,
        n_timesteps: int = 10,
        prompt_wav: str | None = None,
        enable_flow_variable_length: bool = False,
        enable_dit_torch_compile: bool = True,
        enable_flow_cuda_graph: bool = True,
        flow_cuda_graph_capture_shapes: tuple[tuple[int, int], ...] | None = None,
    ) -> None:
        super().__init__()
        self.prompt_cache_capacity: int = positive_int_env(
            "MINICPMO_PROMPT_CACHE_CAPACITY", DEFAULT_PROMPT_CACHE_CAPACITY
        )
        self.reference_workers: int = positive_int_env(
            "MINICPMO_REF_WORKERS", DEFAULT_REFERENCE_WORKERS
        )
        self.reference_lock: threading.Lock = threading.Lock()
        self.prompt_cache_lock: threading.Lock = threading.Lock()
        self.reference_executor: ThreadPoolExecutor | None = None
        self.references_closed: bool = False
        self.reference_hits: int = 0
        self.reference_misses: int = 0
        self.reference_evictions: int = 0
        from sglang_omni.models.minicpm_o.components.token2wav.vocoder import Token2Wav

        dev = torch.device(device)
        if dev.type != "cuda":
            raise ValueError(f"Token2wav requires a CUDA device, got {device}")
        else:
            pass
        self.device_context = torch.cuda.device(dev.index or 0)

        model_dir = str(resolve_model_path(model_path))
        asset_dir = os.path.join(model_dir, "assets", "token2wav")
        if not os.path.isdir(asset_dir):
            raise FileNotFoundError(
                f"token2wav assets not found at {asset_dir}; copy the "
                "checkpoint's assets/token2wav directory next to the weights"
            )
        else:
            pass
        if dtype is None:
            torch_dtype = torch.float32
        elif isinstance(dtype, torch.dtype):
            torch_dtype = dtype
        else:
            torch_dtype = resolve_dtype(dtype)
        if torch_dtype not in FLOW_DTYPES:
            raise ValueError(
                f"Code2Wav dtype must be float32, float16, or bfloat16, got {dtype}"
            )
        else:
            pass
        with self.device_context:
            self.token2wav = Token2Wav(
                Path(asset_dir),
                device=dev,
                dtype=torch_dtype,
                n_timesteps=n_timesteps,
                enable_flow_variable_length=enable_flow_variable_length,
                enable_dit_torch_compile=enable_dit_torch_compile,
                enable_flow_cuda_graph=enable_flow_cuda_graph,
                flow_cuda_graph_capture_shapes=flow_cuda_graph_capture_shapes,
            )

        if prompt_wav is None:
            default_wav = os.path.join(model_dir, "assets", "HT_ref_audio.wav")
            prompt_wav = default_wav if os.path.isfile(default_wav) else None
        else:
            pass
        self.default_prompt_wav = prompt_wav
        # Keyed by reference so a mixed-reference batch never thrashes one slot.
        self.prompt_cache: OrderedDict[
            str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = OrderedDict()
        self.sample_rate = OUTPUT_SAMPLE_RATE
        self.eval()

    @torch.inference_mode()
    def forward(
        self,
        *,
        codec_tokens: torch.Tensor,
        prompt_wav: str | bytes | None = None,
        **_: object,
    ) -> dict[str, object]:
        """Vocode EOS-stripped codec tokens using the supplied or default reference."""
        tokens = codec_tokens.reshape(-1).tolist()
        if not tokens:
            waveform = np.zeros(0, dtype=np.float32)
        else:
            with self.device_context:
                reference = self.resolve_prompt_wav(prompt_wav)
                waveform = self.vocode([tokens], reference)[0]
        return {"waveform": waveform, "sample_rate": OUTPUT_SAMPLE_RATE}

    def resolve_prompt_wav(self, prompt_wav: str | bytes | None) -> str | bytes:
        if prompt_wav is not None:
            resolved = prompt_wav
        elif self.default_prompt_wav is None:
            raise ValueError("No speaker-reference audio supplied or default available")
        else:
            resolved = self.default_prompt_wav
        return resolved

    @staticmethod
    def prompt_key(prompt_wav: str | bytes) -> str:
        if isinstance(prompt_wav, bytes):
            return f"bytes:{hash_bytes(prompt_wav)}"
        else:
            return reference_path_cache_key(prompt_wav) or f"path:{prompt_wav}"

    def speaker_prompt(
        self, prompt_wav: str | bytes | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        prompt_wav = self.resolve_prompt_wav(prompt_wav)
        prompt_key = self.prompt_key(prompt_wav)
        with self.prompt_cache_lock:
            cached = self.prompt_cache.get(prompt_key)
            if cached is not None:
                self.prompt_cache.move_to_end(prompt_key)
                self.reference_hits += 1
                return cached
            else:
                self.reference_misses += 1
        # Bytes references decode in memory; they are not spilled to a temp file.
        source = io.BytesIO(prompt_wav) if isinstance(prompt_wav, bytes) else prompt_wav
        prompt = self.token2wav.prepare_prompt(source)
        with self.prompt_cache_lock:
            self.prompt_cache[prompt_key] = prompt
            if len(self.prompt_cache) > self.prompt_cache_capacity:
                self.prompt_cache.popitem(last=False)
                self.reference_evictions += 1
            else:
                pass
        return prompt

    def prepare_references(
        self, references: Sequence[str | bytes | None]
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Prepare unique speaker references and restore the requested row order."""
        with self.reference_lock:
            if self.references_closed:
                raise RuntimeError("Code2Wav reference preparation is closed")
            else:
                pass
            resolved = [self.resolve_prompt_wav(reference) for reference in references]
            keys = [self.prompt_key(reference) for reference in resolved]
            unique = dict(zip(keys, resolved, strict=True))
            started = time.perf_counter()
            if len(unique) > 1 and self.reference_workers > 1:
                if self.reference_executor is None:
                    self.reference_executor = ThreadPoolExecutor(
                        max_workers=self.reference_workers,
                        thread_name_prefix="minicpmo-ref",
                    )
                else:
                    pass
                futures = [
                    self.reference_executor.submit(self.speaker_prompt, reference)
                    for reference in unique.values()
                ]
                try:
                    prepared = [future.result() for future in futures]
                finally:
                    # note (MayDomine): failed batches must drain GPU preparation too.
                    wait(futures)
            else:
                prepared = [
                    self.speaker_prompt(reference) for reference in unique.values()
                ]
            by_key = dict(zip(unique, prepared, strict=True))
            logger.debug(
                f"minicpm_code2wav_ref_prep rows={len(references)} "
                f"unique={len(unique)} workers={self.reference_workers} "
                f"wall_ms={(time.perf_counter() - started) * 1000.0:.1f} "
                f"hits={self.reference_hits} misses={self.reference_misses} "
                f"evictions={self.reference_evictions}"
            )
            return [by_key[key] for key in keys]

    def close_reference_pool(self) -> None:
        """Drain reference preparation and permanently close the worker pool."""
        with self.reference_lock:
            self.references_closed = True
            if self.reference_executor is not None:
                self.reference_executor.shutdown(wait=True)
                self.reference_executor = None
            else:
                pass

    def vocode(
        self,
        token_sequences: Sequence[Sequence[int]],
        prompt_wav: str | bytes | Sequence[str | bytes] | None = None,
    ) -> list[np.ndarray]:
        """Batch flow across references and preserve each HiFT sequence boundary."""
        if not token_sequences:
            return []
        else:
            pass
        if any(len(tokens) == 0 for tokens in token_sequences):
            raise ValueError("codec token sequences must be non-empty")
        else:
            pass

        batch_size = len(token_sequences)
        if isinstance(prompt_wav, (list, tuple)):
            if len(prompt_wav) != batch_size:
                raise ValueError(
                    f"prompt_wav count {len(prompt_wav)} does not match "
                    f"token sequence count {batch_size}"
                )
            else:
                pass
            references = list(prompt_wav)
        else:
            references = [prompt_wav] * batch_size

        token_lens = [len(tokens) for tokens in token_sequences]
        device = self.token2wav.device
        speech_tokens = pad_sequence(
            [
                torch.tensor(tokens, dtype=torch.int32, device=device)
                for tokens in token_sequences
            ],
            batch_first=True,
        )
        speech_tokens_lens = torch.tensor(token_lens, dtype=torch.int32, device=device)

        # Stack one conditioning row per reference; a shared reference broadcasts
        # instead of copying, and mixed references concatenate along the batch.
        # References of different lengths pad to a common token width here; the
        # flow re-derives each row's real width from prompt_speech_tokens_lens.
        prompts = self.prepare_references(references)
        if len({id(prompt) for prompt in prompts}) == 1:
            (
                prompt_speech_tokens,
                prompt_speech_tokens_lens,
                speaker_embedding,
                prompt_mels,
            ) = prompts[0]
            prompt_speech_tokens = prompt_speech_tokens.expand(
                batch_size, -1
            ).contiguous()
            prompt_speech_tokens_lens = prompt_speech_tokens_lens.expand(
                batch_size
            ).contiguous()
            speaker_embedding = speaker_embedding.expand(batch_size, -1).contiguous()
            prompt_mels = prompt_mels.expand(batch_size, -1, -1).contiguous()
        else:
            # References of different lengths pad to a common token width; the
            # flow re-derives each row's real width from prompt_speech_tokens_lens.
            token_width = max(prompt[0].numel() for prompt in prompts)
            prompt_speech_tokens = torch.cat(
                [
                    torch.nn.functional.pad(
                        prompt[0].reshape(1, -1), (0, token_width - prompt[0].numel())
                    )
                    for prompt in prompts
                ],
                dim=0,
            )
            prompt_speech_tokens_lens = torch.cat(
                [prompt[1] for prompt in prompts], dim=0
            )
            speaker_embedding = torch.cat([prompt[2] for prompt in prompts], dim=0)
            mel_frames = max(prompt[3].shape[1] for prompt in prompts)
            prompt_mels = torch.cat(
                [
                    torch.nn.functional.pad(
                        prompt[3], (0, 0, 0, mel_frames - prompt[3].shape[1])
                    )
                    for prompt in prompts
                ],
                dim=0,
            )

        with torch.amp.autocast(
            "cuda",
            dtype=self.token2wav.dtype,
            enabled=self.token2wav.dtype != torch.float32,
        ):
            mel = self.token2wav.flow.inference(
                speech_tokens,
                speech_tokens_lens,
                prompt_speech_tokens,
                prompt_speech_tokens_lens,
                prompt_mels,
                speaker_embedding,
                self.token2wav.n_timesteps,
            )

        up_rate = self.token2wav.flow.up_rate
        length_groups: dict[int, list[int]] = {}
        for idx, token_len in enumerate(token_lens):
            length_groups.setdefault(token_len, []).append(idx)

        waveform_rows: dict[int, torch.Tensor] = {}
        # note (MayDomine): padding changes HiFT's noncausal convolution boundaries.
        for token_len, indices in length_groups.items():
            speech_feat = mel[indices, :, : token_len * up_rate].float().contiguous()
            wav, _ = self.token2wav.hift(speech_feat=speech_feat)
            for row, idx in enumerate(indices):
                waveform_rows[idx] = wav[row].reshape(-1)[
                    : token_len * SAMPLES_PER_CODEC_TOKEN
                ]
        wav = (
            pad_sequence(
                [waveform_rows[idx] for idx in range(batch_size)], batch_first=True
            )
            .float()
            .cpu()
        )
        return [
            wav[idx, : token_len * SAMPLES_PER_CODEC_TOKEN].numpy()
            for idx, token_len in enumerate(token_lens)
        ]
