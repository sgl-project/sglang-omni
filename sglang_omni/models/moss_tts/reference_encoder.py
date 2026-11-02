# SPDX-License-Identifier: Apache-2.0
"""Shared reference-audio execution for MOSS-TTS models."""

from __future__ import annotations

import concurrent.futures
import os
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx
import torch

from sglang_omni.models.moss_tts.audio_tokenizer import MossAudioEncoder
from sglang_omni.preprocessing.cache_key import hash_bytes, reference_path_cache_key
from sglang_omni.scheduling.reference_encoder import (
    ReferenceEncodeKey,
    ReferenceEncodeService,
    TensorReferenceEncodeHook,
)
from sglang_omni.utils.audio import (
    audio_request_timeout,
    decode_audio_data_uri,
    load_audio,
)

MAX_REFERENCE_SECONDS = 100.0


@dataclass(frozen=True, kw_only=True)
class MossAudioReference:
    source: str | bytes
    content_key: str | None


class MossReferenceEncodeHook(
    TensorReferenceEncodeHook[MossAudioReference, str | os.PathLike[str]]
):
    model_id = "moss_tts"
    encoder_id = "moss_audio_tokenizer"
    artifact_kind = "moss_reference_codes"
    storage_dtype = torch.int32
    output_dtype = torch.long

    def __init__(
        self,
        audio_encoder: MossAudioEncoder,
        *,
        codec_model_path: str,
        n_vq: int,
        cache_enabled: bool,
    ) -> None:
        self.audio_encoder = audio_encoder
        self.n_vq = n_vq
        self.cache_enabled = cache_enabled
        self.model_revision = codec_model_path
        config = (
            f"n_vq:{n_vq}|sample_rate:{audio_encoder.sample_rate}|"
            f"channels:{audio_encoder.number_channels}|device:{audio_encoder.device}|"
            f"dtype:{audio_encoder.model.encoder_dtype}"
        )
        self.encoder_config_hash = hash_bytes(config.encode("utf-8"))
        self.stream: torch.cuda.Stream | None = None
        device = torch.device(audio_encoder.device)
        if device.type == "cuda":
            self.stream = torch.cuda.Stream(device=device)
            self.stream.wait_stream(torch.cuda.current_stream(device))
        else:
            pass

    def normalize_input(self, raw_input: str | os.PathLike[str]) -> MossAudioReference:
        source = os.fsdecode(raw_input)
        raw_audio = decode_audio_data_uri(source)
        if raw_audio is not None:
            return MossAudioReference(
                source=raw_audio, content_key=f"bytes:{hash_bytes(raw_audio)}"
            )
        elif source.startswith(("http://", "https://")):
            response = httpx.get(
                source, timeout=audio_request_timeout(), follow_redirects=True
            )
            response.raise_for_status()
            raw_audio = response.content
            return MossAudioReference(
                source=raw_audio, content_key=f"bytes:{hash_bytes(raw_audio)}"
            )
        elif source.startswith("file://"):
            source = unquote(urlparse(source).path)
        else:
            pass
        source = str(Path(source).expanduser())
        return MossAudioReference(
            source=source, content_key=reference_path_cache_key(source)
        )

    def input_key(self, item: MossAudioReference) -> str | None:
        return item.content_key if self.cache_enabled else None

    def revalidate(self, item: MossAudioReference, key: ReferenceEncodeKey) -> bool:
        return isinstance(item.source, bytes) or (
            reference_path_cache_key(item.source) == key.input_key
        )

    def can_encode_batch(self) -> bool:
        return True

    def encode_one(self, item: MossAudioReference) -> torch.Tensor:
        return self.encode_batch([item])[0]

    def encode_batch(self, items: list[MossAudioReference]) -> list[torch.Tensor]:
        identifiers = [
            item.content_key if item.content_key is not None else item.source
            for item in items
        ]
        unique = dict(zip(identifiers, items, strict=True))
        with torch.cuda.stream(self.stream):
            waveforms: list[tuple[torch.Tensor, int]] = []
            for item in unique.values():
                waveform = load_audio(
                    item.source,
                    source_name="MOSS-TTS reference",
                    target_sample_rate=self.audio_encoder.sample_rate,
                    mono=self.audio_encoder.number_channels == 1,
                )
                duration_seconds = waveform.shape[-1] / self.audio_encoder.sample_rate
                if duration_seconds > MAX_REFERENCE_SECONDS:
                    raise ValueError(
                        f"reference audio is {duration_seconds:.1f}s long, "
                        f"limit is {MAX_REFERENCE_SECONDS:.0f}s"
                    )
                else:
                    pass
                waveforms.append(
                    (torch.from_numpy(waveform), self.audio_encoder.sample_rate)
                )
            encoded = self.audio_encoder.encode_waveforms(
                waveforms, num_quantizers=self.n_vq
            )
        codes_by_input = dict(zip(unique, encoded, strict=True))
        return [codes_by_input[identifier] for identifier in identifiers]


class MossReferenceEncoder(
    ReferenceEncodeService[
        MossAudioReference, torch.Tensor, torch.Tensor, str | os.PathLike[str]
    ]
):
    def __init__(
        self,
        audio_encoder: MossAudioEncoder,
        *,
        codec_model_path: str,
        n_vq: int,
        max_batch_size: int,
        max_batch_wait_ms: int,
        cache_enabled: bool,
        max_items: int,
        max_bytes: int,
    ) -> None:
        super().__init__(
            MossReferenceEncodeHook(
                audio_encoder,
                codec_model_path=codec_model_path,
                n_vq=n_vq,
                cache_enabled=cache_enabled,
            ),
            max_items=max_items,
            max_bytes=max_bytes,
            log_prefix="MOSS-TTS ref cache",
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
            batch_worker_name="moss-tts-ref-encode",
        )
        self.closed = False
        if self.batch_queue is None:
            self.batch_queue = queue.Queue()
            self.batch_thread = threading.Thread(
                target=self.batch_worker, name="moss-tts-ref-encode", daemon=True
            )
            self.batch_thread.start()
        else:
            pass

    def encode_leader(self, item: MossAudioReference) -> torch.Tensor:
        assert self.batch_queue is not None
        future: concurrent.futures.Future[torch.Tensor] = concurrent.futures.Future()
        with self.lock:
            if self.closed:
                raise RuntimeError("MOSS-TTS reference encoder is closed")
            else:
                self.batch_queue.put((item, future))
        return future.result(timeout=self.timeout_s)

    def encode(self, source: str | os.PathLike[str]) -> torch.Tensor:
        with self.lock:
            if self.closed:
                raise RuntimeError("MOSS-TTS reference encoder is closed")
            else:
                pass
        return self.get_or_encode(source, desc="MOSS-TTS reference")

    def close(self) -> None:
        with self.lock:
            if self.closed:
                return
            else:
                self.closed = True
        super().close()
