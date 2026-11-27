# SPDX-License-Identifier: Apache-2.0
"""Nemotron Unified ASR on the Nemotron 3.5 ASR stage.

The checkpoint is a Transformers ``Nemotron3_5AsrForRNNT`` with a non-causal
``parakeet_encoder``. Offline transcription is the inherited ``generate()``
path. Streaming follows NeMo's buffered RNN-T inference
(``speech_to_text_streaming_infer_rnnt.py``, chunk 0.56 s, left 5.6 s, right
0.32 s): hop ``k`` encodes ``[kC - L, kC + C + R)`` with chunked attention and
decodes the ``C`` chunk frames. The last hop keeps the trailing ``L + C + R``
samples, like NeMo's ``StreamingBatchedAudioBuffer``, and decodes every frame
from ``kC`` on. The encoder keeps no state between hops.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch.nn.utils.rnn import pad_sequence
from transformers import Nemotron3_5AsrForRNNT, Nemotron3_5AsrProcessor

from sglang_omni.models.nemotron3_5_asr.decoder import (
    Nemotron3_5ASRDecodeState,
    decode_streaming_batch,
)
from sglang_omni.models.nemotron3_5_asr.encoder_state_pool import EncoderPoolLayout
from sglang_omni.models.nemotron3_5_asr.model_runner import (
    Nemotron3_5ASRModelRunner,
    Nemotron3_5ASRStreamingBatchResult,
)
from sglang_omni.models.nemotron3_5_asr.nemo_checkpoint import load_nemo_archive
from sglang_omni.models.nemotron3_5_asr.request_builders import NEMOTRON_ASR_SAMPLE_RATE
from sglang_omni.models.nemotron3_5_asr.streaming import (
    Nemotron3_5ASRAudioWindow,
    Nemotron3_5ASRStreamState,
)
from sglang_omni.utils.device import supports_device_streams
from sglang_omni.utils.hf import (
    NEMOTRON_UNIFIED_ASR_ARCHIVE,
    load_raw_config,
    try_resolve_arch_from_nemotron_unified_asr_layout,
)

FRAME_SAMPLES = 1280  # one encoder frame: 8 mel frames of 10 ms
LEFT_FRAMES, CHUNK_FRAMES, RIGHT_FRAMES = 70, 7, 4
CHUNK_SAMPLES = CHUNK_FRAMES * FRAME_SAMPLES
LOOKAHEAD_SAMPLES = (CHUNK_FRAMES + RIGHT_FRAMES) * FRAME_SAMPLES
WINDOW_SAMPLES = (LEFT_FRAMES + CHUNK_FRAMES + RIGHT_FRAMES) * FRAME_SAMPLES
MIN_AUDIO_SAMPLES = 320  # shorter audio has no encoder frame; the final is empty


def is_unified_checkpoint(model_path: str) -> bool:
    repo_id, separator, revision = model_path.partition("@")
    if Path(model_path).is_dir() or not separator:
        repo_id, revision = model_path, ""
    else:
        pass
    encoder = (load_raw_config(repo_id, revision or None) or {}).get("encoder_config")
    return (
        isinstance(encoder, dict) and encoder.get("model_type") == "parakeet_encoder"
    ) or try_resolve_arch_from_nemotron_unified_asr_layout(
        repo_id, revision or None
    ) is not None


@dataclass(slots=True)
class UnifiedHop:
    waveform: NDArray[np.float32]
    left_frames: int
    is_last: bool
    prompt_id: int


@dataclass(slots=True)
class NemotronUnifiedASRStreamState(Nemotron3_5ASRStreamState):
    def window_layout(self, index: int) -> tuple[int, int, bool]:
        """Sample bounds of hop ``index`` and whether it is the last hop."""
        start = index * CHUNK_SAMPLES
        if self.is_input_done and start + LOOKAHEAD_SAMPLES >= self.total_samples:
            return max(0, self.total_samples - WINDOW_SAMPLES), self.total_samples, True
        else:
            begin = max(0, start - LEFT_FRAMES * FRAME_SAMPLES)
            return begin, start + LOOKAHEAD_SAMPLES, False

    def next_window_bounds(self) -> tuple[int, int]:
        return self.window_layout(self.model_chunk_index)[:2]

    @property
    def is_drained(self) -> bool:
        return self.has_reached_decode_limit or (
            self.is_input_done and self.covered_audio_end >= self.total_samples
        )

    def has_ready_window(self) -> bool:
        if self.is_drained:
            return False
        elif self.is_input_done:
            return self.total_samples >= MIN_AUDIO_SAMPLES
        else:
            # Audio ending exactly on the right edge is the last hop, so a
            # regular hop waits for one sample past it.
            return self.total_samples > self.next_window_bounds()[1]

    def trim_pcm(self) -> None:
        # The last buffer can start one chunk before the next regular window.
        start = self.next_window_bounds()[0] - CHUNK_SAMPLES
        keep_from = self.total_samples if self.is_drained else max(start, 0)
        trim_bytes = (min(keep_from, self.total_samples) - self.pcm_base_sample) * 2
        if trim_bytes > 0:
            self.pcm_bytes = self.pcm_bytes[trim_bytes:]
            self.pcm_base_sample += trim_bytes // 2
        else:
            pass


class NemotronUnifiedASRModelRunner(Nemotron3_5ASRModelRunner):
    """Stock Transformers model on the inherited offline and session paths."""

    stream_state_type = NemotronUnifiedASRStreamState

    def __init__(self, model_path: str, **kwargs) -> None:
        # Each hop re-encodes its window, so there is no encoder graph.
        super().__init__(model_path, **{**kwargs, "enable_encoder_cuda_graph": False})
        self.chunked_attention = False
        if self.model.generation_config.eos_token_id is None:
            # note: generate masks finished rows of a padded batch only with an
            # EOS criterion; RNN-T has no end token, so use an id never emitted.
            self.model.generation_config.eos_token_id = self.model.config.vocab_size
        else:
            pass
        for layer in self.model.encoder.layers:
            layer.register_forward_pre_hook(self.limit_attention, with_kwargs=True)

    def load_model(
        self, checkpoint: str, dtype: torch.dtype, num_lookahead_tokens: int
    ) -> tuple[Nemotron3_5AsrProcessor, Nemotron3_5AsrForRNNT]:
        # The vendored Nemotron 3.5 classes build only the cache-aware encoder.
        archive = Path(checkpoint, NEMOTRON_UNIFIED_ASR_ARCHIVE)
        if archive.is_file() and not Path(checkpoint, "config.json").is_file():
            return load_nemo_archive(archive, dtype)
        else:
            pass
        return Nemotron3_5AsrProcessor.from_pretrained(
            checkpoint, local_files_only=True
        ), Nemotron3_5AsrForRNNT.from_pretrained(
            checkpoint, dtype=dtype, local_files_only=True
        )

    def new_encoder_pool_layout(self) -> EncoderPoolLayout:
        # Slots carry no encoder state; the pool still bounds open sessions.
        return EncoderPoolLayout(
            attention_shape=(0, 0, 0, 0),
            convolution_shapes={},
            dtype=self.dtype,
            device=self.device,
        )

    def limit_attention(
        self, module: torch.nn.Module, args: tuple, kwargs: dict
    ) -> tuple[tuple, dict]:
        """On hops, narrow the encoder mask to chunked_limited_with_rc.

        Frame ``i`` sees keys in ``[s - L, s + C - 1 + R]`` with
        ``s = (i // C) * C`` counted from the window start.
        """
        mask = kwargs.get("attention_mask")
        if self.chunked_attention and mask is not None:
            index = torch.arange(mask.shape[-1], device=mask.device)
            start = index // CHUNK_FRAMES * CHUNK_FRAMES
            keys = index[None, :] - start[:, None]
            kwargs["attention_mask"] = (
                mask & (keys >= -LEFT_FRAMES) & (keys < CHUNK_FRAMES + RIGHT_FRAMES)
            )
        else:
            pass
        return args, kwargs

    @property
    def streaming_chunk_spec(self) -> dict[str, int]:
        extractor = self.processor.feature_extractor
        return {
            "sample_rate": NEMOTRON_ASR_SAMPLE_RATE,
            "first_samples": LOOKAHEAD_SAMPLES,
            "subsequent_samples": CHUNK_SAMPLES,
            "first_frames": LOOKAHEAD_SAMPLES // extractor.hop_length,
            "subsequent_frames": CHUNK_SAMPLES // extractor.hop_length,
            "hop_length": extractor.hop_length,
            "n_fft": extractor.n_fft,
            "streaming_latency_ms": LOOKAHEAD_SAMPLES // 16,
        }

    def prepare_stream_window(
        self, state: NemotronUnifiedASRStreamState, window: Nemotron3_5ASRAudioWindow
    ) -> UnifiedHop:
        begin, _, is_last = state.window_layout(window.model_chunk_index)
        prompt_id = self.processor.prompt_dictionary.get(state.language)
        if prompt_id is None:
            raise ValueError(f"Unknown Nemotron language {state.language!r}")
        else:
            pass
        left_frames = (
            window.model_chunk_index * CHUNK_SAMPLES - begin
        ) // FRAME_SAMPLES
        return UnifiedHop(window.waveform, left_frames, is_last, prompt_id)

    def run_streaming_batch(
        self,
        states: Sequence[Nemotron3_5ASRDecodeState],
        chunks: Sequence[UnifiedHop],
        *,
        requested_languages: Sequence[str],
        max_new_tokens: Sequence[int | None] | None = None,
    ) -> Nemotron3_5ASRStreamingBatchResult:
        with self.model_lock, torch.inference_mode():
            started_at_s = time.perf_counter()
            features = self.processor.feature_extractor(
                [chunk.waveform for chunk in chunks],
                sampling_rate=NEMOTRON_ASR_SAMPLE_RATE,
                padding="longest",
                return_tensors="pt",
            ).to(self.device)
            self.chunked_attention = True
            try:
                encoded = self.model.get_audio_features(
                    input_features=features["input_features"].to(self.dtype),
                    attention_mask=features["attention_mask"],
                    prompt_ids=torch.tensor(
                        [chunk.prompt_id for chunk in chunks], device=self.device
                    ),
                )
            finally:
                self.chunked_attention = False
            # A regular hop decodes the chunk; the last hop decodes the rest.
            frames = [
                encoded.pooler_output[row, chunk.left_frames : valid][
                    : None if chunk.is_last else CHUNK_FRAMES
                ]
                for row, (chunk, valid) in enumerate(
                    zip(chunks, encoded.attention_mask.sum(-1).tolist())
                )
            ]
            decode_streaming_batch(
                self.model,
                states,
                pad_sequence(frames, batch_first=True),
                [None] * len(states) if max_new_tokens is None else max_new_tokens,
                [len(row) for row in frames],
            )
            if supports_device_streams(self.device):
                self.device_module.synchronize(self.device)
            else:
                pass
            elapsed_s = time.perf_counter() - started_at_s
        return self.decode_stream_texts(states, requested_languages, elapsed_s)


__all__ = [
    "NemotronUnifiedASRModelRunner",
    "NemotronUnifiedASRStreamState",
    "is_unified_checkpoint",
]
