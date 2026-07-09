# SPDX-License-Identifier: Apache-2.0
"""Owned MOSS audio-tokenizer wrapper for MOSS-TTS Delay."""

from __future__ import annotations

import base64
import io
import logging
import os
from typing import Any

import torch

logger = logging.getLogger(__name__)

DEFAULT_MOSS_AUDIO_TOKENIZER = "OpenMOSS-Team/MOSS-Audio-Tokenizer"


def _resolve_checkpoint(checkpoint: str) -> str:
    if os.path.isdir(checkpoint):
        return checkpoint
    from huggingface_hub import snapshot_download

    return snapshot_download(
        checkpoint,
        allow_patterns=(
            "config.json",
            "*.safetensors",
            "*.safetensors.index.json",
        ),
    )


def _torch_dtype(dtype: str | torch.dtype) -> torch.dtype:
    return getattr(torch, dtype) if isinstance(dtype, str) else dtype


class MossAudioTokenizer:
    """Narrow encode/decode API around the upstream MOSS audio-tokenizer model."""

    def __init__(
        self,
        model: Any,
        *,
        model_path: str,
        checkpoint_dir: str,
        device: str,
        dtype: str | torch.dtype,
    ) -> None:
        self.model = model
        self.model_path = model_path
        self.checkpoint_dir = checkpoint_dir
        self.device = str(device)
        self.dtype = str(dtype)
        try:
            config = model.config
        except AttributeError:
            config = None
        try:
            self.sample_rate = int(config.sampling_rate)
        except AttributeError:
            self.sample_rate = 24000
        try:
            quantizer_kwargs = config.quantizer_kwargs or {}
        except AttributeError:
            quantizer_kwargs = {}
        self.n_vq = int(quantizer_kwargs.get("num_quantizers", 32) or 32)

    def encode_wavs(
        self,
        wav_list: list[torch.Tensor] | torch.Tensor,
        sample_rate: int,
        n_vq: int | None = None,
    ) -> list[torch.Tensor]:
        if isinstance(wav_list, torch.Tensor):
            wav_list = [wav_list]
        if not wav_list:
            raise ValueError("wav_list must contain at least one waveform")

        prepared: list[torch.Tensor] = []
        target_sr = int(self.sample_rate)
        resample = int(sample_rate) != target_sr
        for wav in wav_list:
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if resample:
                import torchaudio

                wav = torchaudio.functional.resample(
                    waveform=wav,
                    orig_freq=int(sample_rate),
                    new_freq=target_sr,
                )
            prepared.append(self._loudness_normalize(wav.squeeze(0)).to(self.device))

        try:
            batch_encode = self.model.batch_encode
        except AttributeError:
            batch_encode = None
        with torch.inference_mode():
            if batch_encode is not None:
                enc = batch_encode(prepared, num_quantizers=n_vq)
                audio_codes = enc.audio_codes
                audio_codes_lengths = enc.audio_codes_lengths
            else:
                max_len = max(int(wav.shape[-1]) for wav in prepared)
                input_values = torch.zeros(
                    len(prepared), 1, max_len, device=self.device, dtype=torch.float32
                )
                padding_mask = torch.zeros(
                    len(prepared), max_len, device=self.device, dtype=torch.bool
                )
                for idx, wav in enumerate(prepared):
                    this_len = int(wav.shape[-1])
                    input_values[idx, 0, :this_len] = wav
                    padding_mask[idx, :this_len] = True
                enc = self.model.encode(
                    input_values,
                    padding_mask=padding_mask,
                    num_quantizers=n_vq,
                    return_dict=True,
                )
                audio_codes = enc.audio_codes
                audio_codes_lengths = enc.audio_codes_lengths

        if audio_codes is None or audio_codes_lengths is None:
            raise RuntimeError(
                "MOSS audio tokenizer encode returned empty audio_codes/audio_codes_lengths"
            )

        out: list[torch.Tensor] = []
        for idx in range(int(audio_codes.shape[1])):
            length = int(audio_codes_lengths[idx].item())
            out.append(
                audio_codes[:, idx, :length]
                .transpose(0, 1)
                .contiguous()
                .to(torch.long)
                .cpu()
            )
        return out

    def encode_paths(
        self,
        paths: list[str] | str,
        n_vq: int | None = None,
    ) -> list[torch.Tensor]:
        if isinstance(paths, str):
            paths = [paths]
        if not paths:
            raise ValueError("paths must contain at least one audio path")

        import torchaudio

        wavs: list[torch.Tensor] = []
        target_sr = int(self.sample_rate)
        for path in paths:
            wav, sample_rate = torchaudio.load(path)
            if int(sample_rate) != target_sr:
                wav = torchaudio.functional.resample(
                    waveform=wav,
                    orig_freq=int(sample_rate),
                    new_freq=target_sr,
                )
            wavs.append(wav)
        return self.encode_wavs(wavs, target_sr, n_vq=n_vq)

    def encode_data_uri(self, ref_audio: str, n_vq: int | None = None) -> torch.Tensor:
        from sglang_omni.models.moss_tts.request_builders import _DATA_URI_RE

        match = _DATA_URI_RE.match(ref_audio)
        if match is None:
            raise ValueError(f"not a MOSS-TTS audio data URI: {ref_audio[:40]!r}")
        try:
            import soundfile as sf
        except ImportError as exc:
            raise RuntimeError(
                "MOSS-TTS base64 reference audio requires soundfile to decode the data URI"
            ) from exc

        raw = base64.b64decode(match.group("data"))
        audio, sample_rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        wav = torch.from_numpy(audio.T)
        return self.encode_wavs([wav], int(sample_rate), n_vq=n_vq)[0]

    def decode_codes(
        self,
        codes_list: list[torch.Tensor] | torch.Tensor,
    ) -> list[torch.Tensor]:
        if isinstance(codes_list, torch.Tensor):
            codes_list = [codes_list]
        if not codes_list:
            return []

        transposed = [
            codes.transpose(0, 1).contiguous().to(device=self.device, dtype=torch.long)
            for codes in codes_list
        ]
        n_vq = int(transposed[0].shape[0])
        max_t = max(int(codes.shape[1]) for codes in transposed)
        audio_codes = torch.zeros(
            n_vq, len(transposed), max_t, device=self.device, dtype=torch.long
        )
        padding_mask = torch.zeros(
            len(transposed), max_t, device=self.device, dtype=torch.bool
        )
        for idx, codes in enumerate(transposed):
            length = int(codes.shape[1])
            audio_codes[:, idx, :length] = codes
            padding_mask[idx, :length] = True

        with torch.inference_mode():
            decoded = self.model.decode(
                audio_codes,
                padding_mask=padding_mask,
                return_dict=True,
                chunk_duration=8,
            )
        audio = decoded.audio
        audio_lengths = decoded.audio_lengths
        if audio is None or audio_lengths is None:
            raise RuntimeError(
                "MOSS audio tokenizer decode returned empty audio/audio_lengths"
            )

        wavs: list[torch.Tensor] = []
        for idx in range(int(audio.shape[0])):
            length = int(audio_lengths[idx].item())
            wavs.append(audio[idx, 0, :length].contiguous().to(torch.float32).cpu())
        return wavs

    @staticmethod
    def _loudness_normalize(
        wav: torch.Tensor,
        target_dbfs: float = -20,
        gain_range: tuple[float, float] = (-3.0, 3.0),
    ) -> torch.Tensor:
        wav = wav.to(torch.float32)
        if wav.numel() == 0:
            return wav
        current_dbfs = 10.0 * torch.log10(torch.mean(wav**2) + 1e-9)
        gain = float(target_dbfs - current_dbfs)
        gain = max(gain_range[0], min(gain, gain_range[1]))
        return wav * (10.0 ** (gain / 20.0))


def load_moss_audio_tokenizer(
    model_path: str = DEFAULT_MOSS_AUDIO_TOKENIZER,
    *,
    device: str = "cuda:0",
    dtype: str | torch.dtype = "float32",
) -> MossAudioTokenizer:
    checkpoint_dir = _resolve_checkpoint(model_path)
    logger.info(
        "Loading MOSS audio tokenizer from %s on %s dtype=%s",
        checkpoint_dir,
        device,
        dtype,
    )
    from sglang_omni.models.moss_tts.configuration_moss_audio_tokenizer import (
        MossAudioTokenizerConfig,
    )
    from sglang_omni.models.moss_tts.modeling_moss_audio_tokenizer import (
        MossAudioTokenizerModel,
    )

    config = MossAudioTokenizerConfig.from_pretrained(checkpoint_dir)
    load_kwargs: dict[str, Any] = {
        "config": config,
        "local_files_only": True,
    }
    if str(device) != "cpu":
        load_kwargs["torch_dtype"] = _torch_dtype(dtype)
    model = MossAudioTokenizerModel.from_pretrained(checkpoint_dir, **load_kwargs)
    try:
        model.eval()
    except AttributeError:
        raise RuntimeError("MOSS audio tokenizer model does not implement eval()")
    try:
        move_model = model.to
    except AttributeError:
        raise RuntimeError("MOSS audio tokenizer model does not implement to()")
    else:
        kwargs: dict[str, Any] = {"device": device}
        if str(device) != "cpu":
            kwargs["dtype"] = _torch_dtype(dtype)
        move_model(**kwargs)
    return MossAudioTokenizer(
        model,
        model_path=model_path,
        checkpoint_dir=checkpoint_dir,
        device=device,
        dtype=dtype,
    )
