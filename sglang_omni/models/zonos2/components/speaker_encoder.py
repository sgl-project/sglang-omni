# SPDX-License-Identifier: Apache-2.0
"""ZONOS2 speaker encoder producing the raw 2048-d voice embedding.

Port of the reference ``Qwen3SpeakerEmbedding`` plus the server audio loader.
LDA (2048->1024) and ``speaker_projection`` (1024->2048) live in modeling.py.
"""

from __future__ import annotations

import hashlib
import io
import subprocess
import threading
import wave
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import AutoModel

from sglang_omni.scheduling.reference_encoder import (
    ReferenceEncodeService,
    TensorReferenceEncodeHook,
)

SPEAKER_EMBEDDING_DIM = 2048


class Qwen3SpeakerEmbedding(nn.Module):
    """Qwen3 voice embedding extractor for 2048-d speaker-conditioned checkpoints.

    Model id, mel parameters and preprocessing match the reference exactly so the
    raw embedding is bit-for-bit identical.
    """

    MODEL_NAME = "marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B"
    TARGET_SAMPLE_RATE = 24_000
    N_FFT = 1024
    HOP_LENGTH = 256
    WIN_LENGTH = 1024
    N_MELS = 128
    F_MIN = 0.0
    F_MAX = 12_000.0

    def __init__(self, device: str = "cuda", compile_forward: bool = False):
        super().__init__()
        self.device = device
        self._compile_forward = compile_forward
        self._compiled = None
        self.model = AutoModel.from_pretrained(
            self.MODEL_NAME,
            trust_remote_code=True,
        )
        self.model.to(device)
        self.model.eval()

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.TARGET_SAMPLE_RATE,
            n_fft=self.N_FFT,
            win_length=self.WIN_LENGTH,
            hop_length=self.HOP_LENGTH,
            f_min=self.F_MIN,
            f_max=self.F_MAX,
            n_mels=self.N_MELS,
            power=1.0,
            center=False,
            norm="slaney",
            mel_scale="slaney",
        ).to(device)

        self.requires_grad_(False).eval()

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    @cache
    def _get_resampler(self, orig_sample_rate: int):
        return torchaudio.transforms.Resample(
            orig_sample_rate, self.TARGET_SAMPLE_RATE
        ).to(self.device)

    def prepare_input(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        assert wav.ndim < 3
        if wav.ndim == 2:
            wav = wav.mean(0, keepdim=True)
        wav = wav.to(self.device, torch.float32)
        if sample_rate != self.TARGET_SAMPLE_RATE:
            wav = self._get_resampler(sample_rate)(wav)
        return wav

    def _make_mel(self, wav: torch.Tensor) -> torch.Tensor:
        pad = (self.N_FFT - self.HOP_LENGTH) // 2
        wav = F.pad(wav.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
        mel = self.mel_transform(wav)
        mel = torch.log(torch.clamp(mel, min=1e-5))
        return mel.transpose(1, 2)

    def _forward_impl(self, wav: torch.Tensor, sample_rate: int):
        wav = self.prepare_input(wav, sample_rate)
        mel = self._make_mel(wav)
        if self._compile_forward:
            # note (Yue Yin): mark the mel time dim symbolic so dynamic=True yields
            # one graph across variable audio lengths instead of per-length recompiles.
            torch._dynamo.mark_dynamic(mel, 1)
        return self.model(input_values=mel).last_hidden_state.to(torch.float32)

    def forward(self, wav: torch.Tensor, sample_rate: int):
        if not self._compile_forward:
            return self._forward_impl(wav, sample_rate)
        if self._compiled is None:
            self._compiled = torch.compile(self._forward_impl, dynamic=True)
        return self._compiled(wav, sample_rate)


def _transcode_audio_bytes_to_wav(audio_bytes: bytes) -> bytes:
    """Transcode arbitrary audio bytes to mono 16-bit PCM WAV via ffmpeg."""
    if not audio_bytes:
        raise ValueError("Reference file is empty.")

    try:
        proc = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                "pipe:0",
                "-map",
                "0:a:0",
                "-vn",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                "-f",
                "wav",
                "pipe:1",
            ],
            input=audio_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError as exc:
        raise ValueError(
            "Reference file must contain an audio stream supported by ffmpeg."
        ) from exc

    if proc.returncode != 0 or not proc.stdout:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        suffix = f" ffmpeg said: {stderr}" if stderr else ""
        raise ValueError(
            "Reference file must contain an audio stream supported by ffmpeg." + suffix
        )
    return proc.stdout


def _decode_wav_bytes(wav_bytes: bytes) -> tuple[torch.Tensor, int]:
    """Decode PCM WAV bytes to a ``[channels, T]`` float32 waveform in [-1, 1]."""
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            n_frames = wav_file.getnframes()
            pcm = wav_file.readframes(n_frames)
    except Exception as exc:
        raise ValueError("Reference audio must be a valid PCM WAV file.") from exc

    if len(pcm) == 0:
        raise ValueError("Reference audio is empty.")

    pcm_view = memoryview(bytearray(pcm))

    if sample_width == 1:
        audio = torch.frombuffer(pcm_view, dtype=torch.uint8).to(torch.float32)
        audio = (audio - 128.0) / 128.0
    elif sample_width == 2:
        audio = torch.frombuffer(pcm_view, dtype=torch.int16).to(torch.float32)
        audio = audio / 32768.0
    elif sample_width == 4:
        audio = torch.frombuffer(pcm_view, dtype=torch.int32).to(torch.float32)
        audio = audio / 2147483648.0
    else:
        raise ValueError("Unsupported WAV bit depth. Use 8/16/32-bit PCM WAV.")

    if channels > 1:
        audio = audio.view(-1, channels).transpose(0, 1).contiguous()
    else:
        audio = audio.view(1, -1)

    return audio, sample_rate


# Stable tag for the frozen encoder config; bump if the mel/model params change so
# stale cached embeddings are not reused.
_ENCODER_CONFIG_HASH = hashlib.sha256(
    "|".join(
        str(v)
        for v in (
            Qwen3SpeakerEmbedding.MODEL_NAME,
            Qwen3SpeakerEmbedding.TARGET_SAMPLE_RATE,
            Qwen3SpeakerEmbedding.N_FFT,
            Qwen3SpeakerEmbedding.HOP_LENGTH,
            Qwen3SpeakerEmbedding.WIN_LENGTH,
            Qwen3SpeakerEmbedding.N_MELS,
            Qwen3SpeakerEmbedding.F_MIN,
            Qwen3SpeakerEmbedding.F_MAX,
        )
    ).encode()
).hexdigest()[:16]


@dataclass
class _Zonos2RefInput:
    """Normalized reference input: raw file bytes, or an already-decoded waveform."""

    kind: str  # "raw" | "wav"
    input_key: str
    raw: bytes | None = None
    wav: torch.Tensor | None = None
    sr: int | None = None


class SpeakerEncoder(TensorReferenceEncodeHook[_Zonos2RefInput]):
    """Turns reference audio into the raw 2048-d ZONOS2 speaker embedding.

    The Qwen3 model is loaded lazily on first :meth:`encode`. Caching and
    single-flight dedup run on the shared :class:`ReferenceEncodeService` keyed by
    a content hash, so a repeated reference audio costs at most one forward.
    """

    model_id = "zonos2"
    model_revision = "v1"
    encoder_id = Qwen3SpeakerEmbedding.MODEL_NAME
    encoder_config_hash = _ENCODER_CONFIG_HASH
    artifact_kind = "zonos2_speaker_embedding"
    storage_dtype = torch.float32

    def __init__(
        self,
        device: str = "cuda",
        cache_max_items: int = 256,
        compile_forward: bool = False,
    ):
        if int(cache_max_items) < 1:
            raise ValueError(f"cache_max_items must be >= 1, got {cache_max_items}")
        self.device = device
        self._embedder: Qwen3SpeakerEmbedding | None = None
        self._embedder_lock = threading.Lock()
        # note (Yue Yin): opt-in compile kill-switch (default OFF for bit-for-bit
        # parity); driven by the speaker_encode stage's spk_compile factory arg.
        self._compile = compile_forward
        # note (Yue Yin): [2048] embeddings are fixed-size, so the count cap is the
        # budget (no byte cap). The service holds the content-hash LRU + single-flight.
        self._service: ReferenceEncodeService[
            _Zonos2RefInput, torch.Tensor, torch.Tensor
        ] = ReferenceEncodeService(
            self,
            max_items=int(cache_max_items),
            max_bytes=None,
            log_prefix="ZONOS2 speaker cache",
        )

    def _get_embedder(self) -> Qwen3SpeakerEmbedding:
        if self._embedder is None:
            with self._embedder_lock:
                if self._embedder is None:
                    self._embedder = Qwen3SpeakerEmbedding(
                        device=self.device, compile_forward=self._compile
                    )
        return self._embedder

    def encode(self, ref_audio: Any, sample_rate: int | None = None) -> torch.Tensor:
        """Encode reference audio into a raw ``[2048]`` CPU float32 embedding.

        ``ref_audio`` may be a file path, raw audio bytes, or a
        ``(waveform, sample_rate)`` pair.
        """
        embedding, _ = self.encode_with_fingerprint(ref_audio, sample_rate)
        return embedding

    def encode_with_fingerprint(
        self, ref_audio: Any, sample_rate: int | None = None
    ) -> tuple[torch.Tensor, str]:
        """Return an embedding and the fingerprint of the same normalized input."""
        item = self.normalize_input((ref_audio, sample_rate))
        return self._service.get_or_encode(item), item.input_key

    # ---- ReferenceEncodeHook ----

    def normalize_input(self, raw_input: Any) -> _Zonos2RefInput:
        if isinstance(raw_input, _Zonos2RefInput):
            return raw_input
        ref_audio, sample_rate = raw_input

        if isinstance(ref_audio, (tuple, list)) and len(ref_audio) == 2:
            wav, sr = ref_audio
            wav = torch.as_tensor(wav, dtype=torch.float32)
            sr = int(sr)
            return _Zonos2RefInput(
                kind="wav", wav=wav, sr=sr, input_key=self._hash_waveform(wav, sr)
            )

        if isinstance(ref_audio, torch.Tensor) or hasattr(
            ref_audio, "__array_interface__"
        ):
            if sample_rate is None:
                raise ValueError(
                    "sample_rate is required when ref_audio is a bare waveform."
                )
            wav = torch.as_tensor(ref_audio, dtype=torch.float32)
            sr = int(sample_rate)
            return _Zonos2RefInput(
                kind="wav", wav=wav, sr=sr, input_key=self._hash_waveform(wav, sr)
            )

        raw = self._to_bytes(ref_audio)
        return _Zonos2RefInput(
            kind="raw", raw=raw, input_key="raw:" + hashlib.sha256(raw).hexdigest()
        )

    def input_key(self, item: _Zonos2RefInput) -> str:
        return item.input_key

    def encode_one(self, item: _Zonos2RefInput) -> torch.Tensor:
        if item.kind == "raw":
            wav, sr = _decode_wav_bytes(_transcode_audio_bytes_to_wav(item.raw))
        else:
            wav, sr = item.wav, item.sr
        embedder = self._get_embedder()
        with torch.inference_mode():
            output = embedder(wav, sr)
        return self._select_embedding(output)

    @staticmethod
    def _to_bytes(ref_audio: Any) -> bytes:
        """Raw bytes of a path/bytes input, for content hashing."""
        if isinstance(ref_audio, (bytes, bytearray, memoryview)):
            return bytes(ref_audio)
        if isinstance(ref_audio, (str, Path)):
            return Path(ref_audio).read_bytes()
        raise TypeError(f"Unsupported ref_audio type: {type(ref_audio)!r}")

    @staticmethod
    def _hash_waveform(wav: torch.Tensor, sr: int) -> str:
        contig = wav.detach().to("cpu", torch.float32).contiguous()
        h = hashlib.sha256()
        h.update(str(tuple(contig.shape)).encode())
        h.update(str(int(sr)).encode())
        h.update(contig.numpy().tobytes())
        return "wav:" + h.hexdigest()

    @staticmethod
    def _select_embedding(output: Any) -> torch.Tensor:
        """Reduce the model output to a ``[2048]`` CPU float32 vector.

        Squeeze the batch dim and pick the candidate with 2048 elements.
        """
        if isinstance(output, tuple):
            candidates = [t.squeeze(0).to(torch.float32).cpu() for t in output]
        else:
            candidates = [output.squeeze(0).to(torch.float32).cpu()]

        for candidate in candidates:
            if candidate.numel() == SPEAKER_EMBEDDING_DIM:
                return candidate.reshape(SPEAKER_EMBEDDING_DIM).contiguous()

        raise ValueError(
            f"Reference embedding dimension mismatch. Model expects "
            f"{SPEAKER_EMBEDDING_DIM}, but speaker encoder produced "
            f"{', '.join(str(c.numel()) for c in candidates)}."
        )
