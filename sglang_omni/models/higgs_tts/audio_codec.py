"""Thin facade over Transformers' Higgs Audio V2 tokenizer.

The codec weights are bundled inside the Higgs TTS checkpoint under the
prefix ``tied.embedding.modality_embeddings.0.model.*`` (acoustic encoder,
acoustic decoder, quantizer, semantic model, etc.).
:meth:`HiggsAudioCodec.from_pretrained` extracts those keys directly from
the TTS ``model.safetensors`` so a single checkpoint serves both the AR
engine and the codec.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import HiggsAudioV2TokenizerConfig, HiggsAudioV2TokenizerModel

WaveformInput = torch.Tensor | np.ndarray
logger = logging.getLogger(__name__)

# Codec weights live under this prefix inside the TTS checkpoint.
_CODEC_IN_TTS_CKPT_PREFIX = "tied.embedding.modality_embeddings.0.model."

# Bundled codec config (taken byte-for-byte from
# https://huggingface.co/bosonai/higgs-audio-v2-tokenizer/blob/main/config.json).
_BUNDLED_CODEC_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),
    "configs",
    "higgs_audio_v2_tokenizer.json",
)


@dataclass
class _DecodeCudaGraph:
    graph: torch.cuda.CUDAGraph
    input_codes: torch.Tensor
    output_audio: torch.Tensor


def _capture_safe_quantizer_decode(quantizer: Any, codes: torch.Tensor) -> torch.Tensor:
    """Equivalent RVQ decode without a pageable CPU-to-GPU scalar copy.

    Transformers 5.6 initializes the accumulator with
    ``torch.tensor(0.0, device=codes.device)``.  During raw CUDA graph capture
    that construction is a pageable host-to-device copy and invalidates the
    capture.  Starting from the first quantizer output is numerically
    equivalent and entirely device-side.
    """
    if int(codes.shape[0]) == 0:
        raise ValueError("Higgs codec decode requires at least one quantizer")
    quantized_out = quantizer.quantizers[0].decode(codes[0])
    for i, indices in enumerate(codes[1:], start=1):
        quantized_out = quantized_out + quantizer.quantizers[i].decode(indices)
    return quantized_out


def _to_mono_3d(waveform: WaveformInput) -> torch.Tensor:
    """Normalise (Tensor | ndarray) waveform to mono ``[1, 1, L]``."""
    if isinstance(waveform, np.ndarray):
        wav = torch.from_numpy(np.ascontiguousarray(waveform))
    elif isinstance(waveform, torch.Tensor):
        wav = waveform
    else:
        raise TypeError(
            f"waveform must be Tensor or ndarray, got {type(waveform).__name__}"
        )

    if wav.ndim == 1:
        return wav.view(1, 1, -1)
    if wav.ndim == 2:
        return wav[:1].unsqueeze(0)
    if wav.ndim == 3:
        if wav.shape[1] != 1:
            raise ValueError(f"audio must be mono, got shape {tuple(wav.shape)}")
        return wav
    raise ValueError(f"waveform must be 1-, 2- or 3-D, got {wav.ndim}-D")


def _resolve_ckpt_dir(model_path: str | Path) -> str:
    """Local dir or HF repo id → local snapshot dir."""
    p = str(model_path)
    if os.path.isdir(p):
        return p
    return snapshot_download(p)


def _load_codec_state_dict(tts_ckpt_dir: str) -> dict[str, torch.Tensor]:
    """Pull all ``tied.embedding.modality_embeddings.0.model.*`` tensors out of
    the TTS checkpoint's shards into a state dict keyed by the codec's own
    parameter names (prefix stripped)."""
    index_path = os.path.join(tts_ckpt_dir, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        shards: dict[str, list[str]] = {}
        for full_name, shard in weight_map.items():
            if full_name.startswith(_CODEC_IN_TTS_CKPT_PREFIX):
                shards.setdefault(shard, []).append(full_name)
    else:
        shards = {"model.safetensors": None}  # single-shard layout

    state: dict[str, torch.Tensor] = {}
    for shard, names in shards.items():
        shard_path = os.path.join(tts_ckpt_dir, shard)
        with safe_open(shard_path, framework="pt") as f:
            keys = (
                names
                if names is not None
                else [k for k in f.keys() if k.startswith(_CODEC_IN_TTS_CKPT_PREFIX)]
            )
            for full_name in keys:
                state[full_name[len(_CODEC_IN_TTS_CKPT_PREFIX) :]] = f.get_tensor(
                    full_name
                )
    return state


class HiggsAudioCodec:
    """Frozen encode/decode wrapper around :class:`HiggsAudioV2TokenizerModel`."""

    SAMPLE_RATE: int = 24_000

    def __init__(
        self, model: HiggsAudioV2TokenizerModel, *, device: torch.device
    ) -> None:
        self.model = model
        self.device = device
        self._dtype = next(model.parameters()).dtype
        self._decode_cuda_graphs: dict[int, _DecodeCudaGraph] = {}
        self._decode_cuda_graph_hits = 0
        self._decode_cuda_graph_misses = 0
        self._decode_cuda_graph_missed_shapes: set[int] = set()
        # Graphs for different frame counts share one private CUDA memory pool.
        # Their temporary/output storage may therefore alias and decode must
        # remain single-flight until the selected output has reached the host.
        self._decode_single_flight_lock = threading.Lock()

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "HiggsAudioCodec":
        """Load codec from a Higgs TTS checkpoint (local dir or HF repo id).

        Codec weights are bundled inside the TTS ``model.safetensors`` under
        the ``tied.embedding.modality_embeddings.0.model.*`` prefix; we use
        the bundled codec config for the codec architecture and load the codec
        tensors out of the TTS shards directly.

        ``dtype`` defaults to fp32; decode ConvTranspose is unstable in bf16
        — opt in only if you've validated quality at your sample rate.
        """
        device = torch.device(device)
        ckpt_dir = _resolve_ckpt_dir(model_path)

        config = HiggsAudioV2TokenizerConfig.from_json_file(_BUNDLED_CODEC_CONFIG_PATH)
        model = HiggsAudioV2TokenizerModel(config).to(dtype=dtype).eval()

        state = _load_codec_state_dict(ckpt_dir)
        if not state:
            raise FileNotFoundError(
                f"No codec weights found under {_CODEC_IN_TTS_CKPT_PREFIX!r} in "
                f"{ckpt_dir}; this checkpoint doesn't bundle the audio codec."
            )

        missing, unexpected = model.load_state_dict(state, strict=False)
        # Some upstream init keys (e.g. `weight_g`/`weight_v` weight-norm parts)
        # are regenerated by post_init; the substantive codec tensors all load.
        if len(missing) > len(state) // 2:
            raise RuntimeError(
                f"Codec weight load is too sparse: {len(missing)} missing / "
                f"{len(state)} loaded; bundled codec config may be incompatible "
                "with the installed Transformers version."
            )
        model = model.to(device=device)
        for p in model.parameters():
            p.requires_grad_(False)
        return cls(model, device=device)

    @torch.no_grad()
    def capture_decode_cuda_graphs(self, frame_counts: tuple[int, ...]) -> None:
        """Capture and publish decode graphs while decode is quiescent."""
        with self._decode_single_flight_lock:
            self._capture_decode_cuda_graphs_locked(frame_counts)

    def _capture_decode_cuda_graphs_locked(self, frame_counts: tuple[int, ...]) -> None:
        """Capture fixed-shape single-item codec decode graphs at startup.

        Higgs' streaming cursor has a finite set of single-item decode-window
        shapes. Capturing the configured domain removes hundreds of eager
        kernel launches from both steady and final chunks. Unlisted shapes
        still use eager decode. Capture finishes before the stage becomes
        ready, so it cannot race SGLang's independently owned AR graphs.
        """
        if self.device.type != "cuda":
            raise RuntimeError("Higgs codec CUDA graphs require a CUDA device")
        # Descending so the shared mempool is sized once from the largest shape
        # rather than grown across captures: 130 vs 146 MiB reserved on the
        # default 1..150 domain.
        frames = tuple(sorted({int(value) for value in frame_counts}, reverse=True))
        if not frames or any(value <= 0 for value in frames):
            raise ValueError("decode CUDA graph frame counts must be positive")

        num_quantizers = int(self.model.config.num_quantizers)
        current_stream = torch.cuda.current_stream(self.device)
        capture_stream = torch.cuda.Stream(device=self.device)
        capture_stream.wait_stream(current_stream)
        original_quantizer_decode = self.model.quantizer.decode
        self.model.quantizer.decode = types.MethodType(
            _capture_safe_quantizer_decode,
            self.model.quantizer,
        )
        captured: dict[int, _DecodeCudaGraph] = {}
        try:
            # Bind the device: CUDAGraph and graph_pool_handle act on the
            # current device, which need not be the codec's.
            with torch.cuda.device(self.device):
                # Initialize shape-specialized CUDA-library state outside
                # capture. One eager pass per shape is sufficient after model
                # startup; doing three passes made comprehensive tail-shape
                # capture needlessly expensive.
                with torch.cuda.stream(capture_stream):
                    for frame_count in frames:
                        warm_codes = torch.zeros(
                            (1, num_quantizers, frame_count),
                            dtype=torch.long,
                            device=self.device,
                        )
                        self.model.decode(warm_codes).audio_values
                capture_stream.synchronize()

                # One mempool for all shapes: a private pool per graph reserves
                # the sum of every shape's peak instead of the largest, 6558 vs
                # 130 MiB on the default 1..150 domain. Aliasing across shapes
                # is safe because decode() replays one graph at a time and
                # copies the output to host before returning.
                graph_pool = torch.cuda.graph_pool_handle()
                for frame_count in frames:
                    # Outside the capture, so the static input is not drawn from
                    # the shared pool.
                    input_codes = torch.zeros(
                        (1, num_quantizers, frame_count),
                        dtype=torch.long,
                        device=self.device,
                    )
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(
                        graph, pool=graph_pool, stream=capture_stream
                    ):
                        output_audio = self.model.decode(input_codes).audio_values
                    captured[frame_count] = _DecodeCudaGraph(
                        graph=graph,
                        input_codes=input_codes,
                        output_audio=output_audio,
                    )
        finally:
            self.model.quantizer.decode = original_quantizer_decode

        current_stream.wait_stream(capture_stream)
        torch.cuda.synchronize(self.device)
        self._decode_cuda_graphs = captured
        logger.info(
            f"Captured {len(captured)} Higgs codec decode CUDA graphs for "
            f"frame counts {min(captured)}..{max(captured)}"
        )

    @torch.no_grad()
    def encode_reference(
        self,
        waveform: WaveformInput,
        *,
        sample_rate: int | None = None,
    ) -> torch.Tensor:
        """Reference clip → ``[T, num_codebooks]`` int64 codes (CPU).

        Accepts 1-D ``[L]``, 2-D ``[C, L]``, or 3-D ``[1, 1, L]`` float32 in
        ``[-1, 1]``. Resamples to 24 kHz; clips < 1 s are zero-padded since
        the encoder errors otherwise.
        """
        wav = _to_mono_3d(waveform).to(torch.float32)
        sr = sample_rate or self.SAMPLE_RATE
        if sr != self.SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, self.SAMPLE_RATE)
        if wav.shape[-1] < self.SAMPLE_RATE:
            wav = F.pad(wav, (0, self.SAMPLE_RATE - wav.shape[-1]))

        wav = wav.to(device=self.device, dtype=self._dtype)
        codes_BNT = self.model.encode(wav).audio_codes
        return codes_BNT.squeeze(0).transpose(0, 1).to(torch.long).cpu()

    def _bucketed_batch(
        self,
        items: list[torch.Tensor],
        *,
        bucket_key_fn,
        single_fn,
        batch_fn,
        error_label: str,
    ) -> list[torch.Tensor]:
        """Run single_fn on singleton buckets and batch_fn on multi-item buckets."""
        if not items:
            return []
        if len(items) == 1:
            return [single_fn(items[0])]

        buckets: dict[int, list[int]] = {}
        for i, item in enumerate(items):
            buckets.setdefault(bucket_key_fn(item), []).append(i)

        results: list[torch.Tensor | None] = [None] * len(items)
        for indices in buckets.values():
            if len(indices) == 1:
                results[indices[0]] = single_fn(items[indices[0]])
            else:
                batch_results = batch_fn([items[i] for i in indices])
                for idx, result in zip(indices, batch_results):
                    results[idx] = result

        out: list[torch.Tensor] = []
        for i, result in enumerate(results):
            if result is None:
                raise RuntimeError(f"{error_label} did not produce result for item {i}")
            out.append(result)
        return out

    @torch.no_grad()
    def encode_batch(self, waveforms: list[torch.Tensor]) -> list[torch.Tensor]:
        # Offline/bulk encoding utility.
        padded: list[torch.Tensor] = []
        for w in waveforms:
            wav = _to_mono_3d(w).to(torch.float32)
            if wav.shape[-1] < self.SAMPLE_RATE:
                wav = F.pad(wav, (0, self.SAMPLE_RATE - wav.shape[-1]))
            padded.append(wav)

        def _batch_fn(batch_items: list[torch.Tensor]) -> list[torch.Tensor]:
            batch = torch.cat(batch_items).to(device=self.device, dtype=self._dtype)
            codes_BNT = self.model.encode(batch).audio_codes.to(torch.long).cpu()
            return [codes_BNT[j].transpose(0, 1) for j in range(len(batch_items))]

        return self._bucketed_batch(
            padded,
            bucket_key_fn=lambda w: w.shape[-1],
            single_fn=self.encode_reference,
            batch_fn=_batch_fn,
            error_label="encode_batch",
        )

    @torch.no_grad()
    def decode(self, codes_TN: torch.Tensor) -> torch.Tensor:
        """``[T, num_codebooks]`` → mono waveform ``[L]``."""
        with self._decode_single_flight_lock:
            return self._decode_one(codes_TN)

    def _decode_one(self, codes_TN: torch.Tensor) -> torch.Tensor:
        if codes_TN.ndim != 2:
            raise ValueError(
                f"codes must be 2-D [T, num_codebooks], got {tuple(codes_TN.shape)}"
            )
        frame_count = int(codes_TN.shape[0])
        decode_graphs = self._decode_cuda_graphs
        decode_graph = decode_graphs.get(frame_count)
        codes_BNT = codes_TN.transpose(0, 1).unsqueeze(0)
        if decode_graph is not None:
            self._decode_cuda_graph_hits += 1
            if int(codes_BNT.shape[1]) != int(decode_graph.input_codes.shape[1]):
                raise ValueError(
                    f"codes have {int(codes_BNT.shape[1])} quantizers, expected "
                    f"{int(decode_graph.input_codes.shape[1])}"
                )
            decode_graph.input_codes.copy_(
                codes_BNT.to(dtype=torch.long), non_blocking=True
            )
            decode_graph.graph.replay()
            audio = decode_graph.output_audio
        else:
            if decode_graphs:
                self._decode_cuda_graph_misses += 1
                if frame_count not in self._decode_cuda_graph_missed_shapes:
                    self._decode_cuda_graph_missed_shapes.add(frame_count)
                    logger.warning(
                        f"Higgs codec decode CUDA graph miss for frame count "
                        f"{frame_count}; using eager decode"
                    )
            audio = self.model.decode(
                codes_BNT.to(device=self.device, dtype=torch.long)
            ).audio_values
        total_graph_lookups = (
            self._decode_cuda_graph_hits + self._decode_cuda_graph_misses
        )
        if decode_graphs and total_graph_lookups % 1024 == 0:
            logger.info(
                f"Higgs codec decode CUDA graph stats: "
                f"hits={self._decode_cuda_graph_hits} "
                f"misses={self._decode_cuda_graph_misses}"
            )
        return audio.squeeze(0).squeeze(0).cpu()

    @torch.no_grad()
    def decode_batch(self, codes_list: list[torch.Tensor]) -> list[torch.Tensor]:
        """Batch-decode variable-length ``[T_i, N]`` tensors into ``[L_i]`` waveforms."""
        if not codes_list:
            return []

        def _batch_fn(batch_items: list[torch.Tensor]) -> list[torch.Tensor]:
            stacked = torch.stack(batch_items)
            codes_BNT = stacked.transpose(1, 2).to(device=self.device, dtype=torch.long)
            audio = self.model.decode(codes_BNT).audio_values.cpu()
            return [audio[j, 0] for j in range(len(batch_items))]

        with self._decode_single_flight_lock:
            return self._bucketed_batch(
                codes_list,
                bucket_key_fn=lambda c: c.shape[0],
                single_fn=self._decode_one,
                batch_fn=_batch_fn,
                error_label="decode_batch",
            )


__all__ = ["HiggsAudioCodec"]
