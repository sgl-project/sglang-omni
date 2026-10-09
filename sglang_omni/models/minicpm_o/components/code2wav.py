# SPDX-License-Identifier: Apache-2.0
"""Vocode MiniCPM-o codec tokens with cached speaker references."""

from __future__ import annotations

import io
import os
import threading
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from sglang_omni.models.minicpm_o.components.token2wav.vocoder import (
    SpeakerPrompt,
    Token2Wav,
)
from sglang_omni.models.weight_loader import resolve_dtype, resolve_model_path
from sglang_omni.preprocessing.cache_key import hash_bytes, reference_path_cache_key

FLOW_DTYPES = (torch.float32, torch.float16, torch.bfloat16)

OUTPUT_SAMPLE_RATE = 24000
CODEC_TOKEN_RATE = 25
SAMPLES_PER_CODEC_TOKEN = OUTPUT_SAMPLE_RATE // CODEC_TOKEN_RATE
FLOW_WARMUP_TOKENS = 32


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
        enable_dit_torch_compile: bool = False,
        enable_hift_torch_compile: bool = False,
        enable_flow_variable_length: bool,
        reference_workers: int,
        prompt_cache_capacity: int,
        decode_stream_priority: int,
        enable_flow_block_compile: bool,
        enable_flow_cuda_graph: bool,
        flow_cuda_graph_capture_shapes: tuple[tuple[int, int], ...],
    ) -> None:
        super().__init__()
        resolved_device = torch.device(device)
        if resolved_device.type not in {"cuda", "xpu"}:
            raise ValueError(f"Token2wav requires a CUDA or XPU device, got {device}")
        elif reference_workers < 1 or prompt_cache_capacity < 1:
            raise ValueError(
                "reference_workers and prompt_cache_capacity must be positive, got "
                f"{reference_workers} and {prompt_cache_capacity}"
            )
        elif (
            enable_dit_torch_compile or enable_hift_torch_compile
        ) and resolved_device.type != "cuda":
            raise ValueError(
                f"Code2Wav torch.compile is validated on CUDA only, got {device}; "
                "set enable_dit_torch_compile and enable_hift_torch_compile to false"
            )
        else:
            pass
        self.device_context = torch.get_device_module(resolved_device).device(
            resolved_device.index or 0
        )

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
                device=resolved_device,
                dtype=torch_dtype,
                n_timesteps=n_timesteps,
            )
        self.token2wav.flow.decoder.estimator.enable_variable_length = (
            enable_flow_variable_length
        )
        if enable_flow_block_compile and self.token2wav.device.type == "cuda":
            for flow_block in self.token2wav.flow.decoder.estimator.blocks:
                flow_block.forward_packed = torch.compile(
                    flow_block.forward_packed,
                    dynamic=True,
                    fullgraph=True,
                    options={"emulate_precision_casts": True},
                )
        else:
            pass
        with self.device_context:
            device_module = torch.get_device_module(self.token2wav.device)
            self.decode_stream: torch.Stream = device_module.Stream(
                priority=decode_stream_priority,
            )
            # note (zhaochenyang20): weights are published on the load stream.
            self.decode_stream.wait_stream(device_module.current_stream())

        if prompt_wav is None:
            default_wav = os.path.join(model_dir, "assets", "HT_ref_audio.wav")
            prompt_wav = default_wav if os.path.isfile(default_wav) else None
        else:
            pass
        self.default_prompt_wav = prompt_wav
        self.prompt_cache_capacity = prompt_cache_capacity
        self.prompt_cache: OrderedDict[str, SpeakerPrompt] = OrderedDict()
        self.pending_references: dict[str, Future[SpeakerPrompt]] = {}
        self.reserved_reference_keys_by_request: dict[str, str] = {}
        self.reference_reservations: Counter[str] = Counter()
        self.reference_lock = threading.RLock()
        self.reference_executor = ThreadPoolExecutor(
            max_workers=reference_workers, thread_name_prefix="minicpmo-reference"
        )
        self.sample_rate = OUTPUT_SAMPLE_RATE
        self.eval()
        if enable_dit_torch_compile:
            flow = self.token2wav.flow
            # note (Dayuxiaoshui): compile the dense blocks only; the packed path
            # slices by data-dependent lengths and would recompile per length.
            for block in flow.decoder.estimator.blocks:
                block.forward = torch.compile(
                    block.forward, dynamic=True, options={"triton.cudagraphs": False}
                )
            warmup_prompt = SpeakerPrompt(
                prompt_tokens=torch.zeros(
                    1, FLOW_WARMUP_TOKENS, dtype=torch.int32, device=resolved_device
                ),
                prompt_token_lengths=torch.tensor(
                    [FLOW_WARMUP_TOKENS], dtype=torch.int32, device=resolved_device
                ),
                speaker_embedding=torch.zeros(
                    1,
                    flow.speaker_embedding_projection.in_features,
                    device=resolved_device,
                ),
                prompt_mel=torch.zeros(
                    1,
                    FLOW_WARMUP_TOKENS * flow.up_rate,
                    flow.output_size,
                    device=resolved_device,
                ),
            )
            # note (Dayuxiaoshui): trace once at startup so no request pays the compile.
            with self.device_context, torch.inference_mode():
                self.flow_mel(
                    warmup_prompt.prompt_tokens,
                    warmup_prompt.prompt_token_lengths,
                    [warmup_prompt],
                )
        else:
            pass
        if enable_hift_torch_compile:
            hift = self.token2wav.hift
            # note (0xtoward): STFT and iSTFT stay eager; the real-valued body compiles.
            hift.decode_body = torch.compile(
                hift.decode_body, dynamic=True, fullgraph=True
            )
            mel_bins = self.token2wav.flow.output_size
            mel_frames = FLOW_WARMUP_TOKENS * self.token2wav.flow.up_rate
            stft_frames = (
                mel_frames * int(hift.f0_upsamp.scale_factor)
            ) // hift.istft_params["hop_len"] + 1
            # note (0xtoward): warm the compiled body for batch 1 and batch >= 2 with
            # the shapes decode feeds it, without running the F0 and source path. The
            # mel is made outside inference mode like the one vocode passes, and the
            # source spectrum inside it like the one HiFT computes, so serving meets
            # the same guards.
            with self.device_context:
                for batch_size in (1, 2):
                    mel = torch.zeros(
                        batch_size, mel_bins, mel_frames, device=resolved_device
                    )
                    with torch.inference_mode():
                        hift.decode_body(
                            mel,
                            torch.zeros(
                                batch_size,
                                hift.istft_params["n_fft"] + 2,
                                stft_frames,
                                device=resolved_device,
                            ),
                        )
        else:
            pass

        if enable_flow_cuda_graph and resolved_device.type == "cuda":
            with self.device_context:
                self.token2wav.capture_flow_graphs(
                    flow_cuda_graph_capture_shapes,
                )
        else:
            pass

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
                waveform = self.vocode([tokens], [prompt_wav])[0]
        return {"waveform": waveform, "sample_rate": OUTPUT_SAMPLE_RATE}

    def resolve_prompt_wav(self, prompt_wav: str | bytes | None) -> str | bytes:
        if prompt_wav is not None:
            resolved = prompt_wav
        elif self.default_prompt_wav is None:
            raise ValueError("No speaker-reference audio supplied or default available")
        else:
            resolved = self.default_prompt_wav
        return resolved

    def resolve_reference_key(
        self, reference: str | bytes | None
    ) -> tuple[str, str | bytes]:
        resolved = self.resolve_prompt_wav(reference)
        if isinstance(resolved, bytes):
            reference_key = f"bytes:{hash_bytes(resolved)}"
        else:
            reference_key = reference_path_cache_key(resolved) or f"path:{resolved}"
        return reference_key, resolved

    def submit_reference(
        self, reference_key: str, reference: str | bytes
    ) -> Future[SpeakerPrompt]:
        """Start preparing one reference; the caller holds reference_lock."""
        source = io.BytesIO(reference) if isinstance(reference, bytes) else reference

        def prepare_reference() -> SpeakerPrompt:
            device_module = torch.get_device_module(self.token2wav.device)
            # note (zhaochenyang20): cached prompts must share the decoder's stream.
            with device_module.stream(self.decode_stream):
                return self.token2wav.prepare_prompt(source)

        future = self.reference_executor.submit(prepare_reference)
        self.pending_references[reference_key] = future
        future.add_done_callback(
            lambda completed: self.store_reference(reference_key, completed)
        )
        return future

    def store_reference(
        self, reference_key: str, future: Future[SpeakerPrompt]
    ) -> None:
        with self.reference_lock:
            if self.pending_references.get(reference_key) is future:
                del self.pending_references[reference_key]
            else:
                pass
            if future.exception() is None:
                self.prompt_cache[reference_key] = future.result()
                self.prompt_cache.move_to_end(reference_key)
            else:
                pass
            self.evict_unreserved_references()

    def evict_unreserved_references(self) -> None:
        overflow = len(self.prompt_cache) - self.prompt_cache_capacity
        if overflow <= 0:
            return
        else:
            pass
        evictable_reference_keys = [
            reference_key
            for reference_key in self.prompt_cache
            if reference_key not in self.reference_reservations
        ][:overflow]
        for reference_key in evictable_reference_keys:
            del self.prompt_cache[reference_key]

    def prefetch_reference(
        self, request_id: str, reference: str | bytes | None
    ) -> None:
        """Start preparing a queued request's reference and pin it until release."""
        reference_key, resolved = self.resolve_reference_key(reference)
        with self.reference_lock:
            self.reserved_reference_keys_by_request[request_id] = reference_key
            self.reference_reservations[reference_key] += 1
            if reference_key in self.prompt_cache:
                self.prompt_cache.move_to_end(reference_key)
            elif reference_key not in self.pending_references:
                self.submit_reference(reference_key, resolved)
            else:
                pass

    def release_reference(self, request_id: str) -> None:
        """Unpin a request's prompt once its batch consumed it or it was aborted."""
        with self.reference_lock:
            reference_key = self.reserved_reference_keys_by_request.pop(
                request_id, None
            )
            if reference_key is None:
                return
            else:
                pass
            self.reference_reservations[reference_key] -= 1
            if self.reference_reservations[reference_key] == 0:
                del self.reference_reservations[reference_key]
            else:
                pass
            self.evict_unreserved_references()

    def prepare_references(
        self, references: Sequence[str | bytes | None]
    ) -> list[SpeakerPrompt]:
        """Prepare each distinct reference once, in parallel, and keep row order."""
        row_reference_keys: list[str] = []
        references_by_reference_key: dict[str, str | bytes] = {}
        for reference in references:
            reference_key, resolved = self.resolve_reference_key(reference)
            row_reference_keys.append(reference_key)
            references_by_reference_key[reference_key] = resolved

        prompts_by_reference_key: dict[str, SpeakerPrompt] = {}
        futures_by_reference_key: dict[str, Future[SpeakerPrompt]] = {}
        with self.reference_lock:
            for reference_key, reference in references_by_reference_key.items():
                if reference_key in self.prompt_cache:
                    self.prompt_cache.move_to_end(reference_key)
                    prompts_by_reference_key[reference_key] = self.prompt_cache[
                        reference_key
                    ]
                elif reference_key in self.pending_references:
                    futures_by_reference_key[reference_key] = self.pending_references[
                        reference_key
                    ]
                else:
                    futures_by_reference_key[reference_key] = self.submit_reference(
                        reference_key, reference
                    )
        # note (MayDomine): failed batches must drain GPU preparation too.
        wait(futures_by_reference_key.values())
        for reference_key, future in futures_by_reference_key.items():
            prompts_by_reference_key[reference_key] = future.result()
        return [
            prompts_by_reference_key[reference_key]
            for reference_key in row_reference_keys
        ]

    def close_reference_pool(self) -> None:
        """Drain reference preparation and reject later submissions."""
        self.reference_executor.shutdown(wait=True)

    def flow_mel(
        self,
        speech_tokens: torch.Tensor,
        speech_token_lengths: torch.Tensor,
        speaker_prompts: Sequence[SpeakerPrompt],
    ) -> torch.Tensor:
        """Run the flow on padded codec tokens, one speaker prompt per row."""
        with torch.amp.autocast(
            self.token2wav.device.type,
            dtype=self.token2wav.dtype,
            enabled=self.token2wav.dtype != torch.float32,
        ):
            return self.token2wav.flow.inference(
                speech_tokens,
                speech_token_lengths,
                pad_sequence(
                    [prompt.prompt_tokens[0] for prompt in speaker_prompts],
                    batch_first=True,
                ),
                torch.cat([prompt.prompt_token_lengths for prompt in speaker_prompts]),
                pad_sequence(
                    [prompt.prompt_mel[0] for prompt in speaker_prompts],
                    batch_first=True,
                ),
                torch.cat([prompt.speaker_embedding for prompt in speaker_prompts]),
                self.token2wav.n_timesteps,
            )

    def vocode(
        self,
        token_sequences: Sequence[Sequence[int]],
        references: Sequence[str | bytes | None],
    ) -> list[np.ndarray]:
        """Batch flow across references and keep each HiFT sequence boundary exact."""
        assert len(references) == len(token_sequences)
        if not token_sequences:
            return []
        elif any(len(tokens) == 0 for tokens in token_sequences):
            raise ValueError("codec token sequences must be non-empty")
        else:
            pass

        speaker_prompts = self.prepare_references(references)
        device_module = torch.get_device_module(self.token2wav.device)
        with device_module.stream(self.decode_stream):
            return self.decode_waveforms(token_sequences, speaker_prompts)

    def decode_waveforms(
        self,
        token_sequences: Sequence[Sequence[int]],
        speaker_prompts: Sequence[SpeakerPrompt],
    ) -> list[np.ndarray]:
        """Run flow and HiFT. The caller selects the decode stream."""
        token2wav_device = self.token2wav.device
        token_lengths = [len(tokens) for tokens in token_sequences]
        speech_tokens = pad_sequence(
            [torch.tensor(tokens, dtype=torch.int32) for tokens in token_sequences],
            batch_first=True,
        ).to(token2wav_device)
        speech_token_lengths = torch.tensor(
            token_lengths, dtype=torch.int32, device=token2wav_device
        )
        mel = self.flow_mel(speech_tokens, speech_token_lengths, speaker_prompts)

        mel_upsample_rate = self.token2wav.flow.up_rate
        rows_by_token_length: defaultdict[int, list[int]] = defaultdict(list)
        for row, token_length in enumerate(token_lengths):
            rows_by_token_length[token_length].append(row)
        waveforms_by_row: dict[int, torch.Tensor] = {}
        # note (MayDomine): padding changes HiFT's noncausal convolution boundaries.
        for token_length, row_indices in rows_by_token_length.items():
            speech_mel = (
                mel[row_indices, :, : token_length * mel_upsample_rate]
                .float()
                .contiguous()
            )
            group_waveforms, _ = self.token2wav.hift(speech_feat=speech_mel)
            for group_row, row in enumerate(row_indices):
                waveforms_by_row[row] = group_waveforms[group_row].reshape(-1)[
                    : token_length * SAMPLES_PER_CODEC_TOKEN
                ]
        host_waveforms = torch.cat(
            [waveforms_by_row[row] for row in range(len(token_lengths))]
        ).cpu()
        sample_counts = [length * SAMPLES_PER_CODEC_TOKEN for length in token_lengths]
        return [waveform.numpy() for waveform in host_waveforms.split(sample_counts)]
