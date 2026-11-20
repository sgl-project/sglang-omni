# SPDX-License-Identifier: Apache-2.0
"""Lower a /v1/audio/speech request into Zonos3 prompt frame rows."""

from __future__ import annotations

import bisect
import math
import threading
from collections.abc import Mapping

import torch
from tokenizers import Tokenizer

from sglang_omni.models.zonos3.components.codec import (
    EXTERNAL_SAMPLES_PER_FRAME,
    SAMPLE_RATE_HZ,
)
from sglang_omni.models.zonos3.components.codec_encoder import Zonos3CodecEncoder
from sglang_omni.models.zonos3.hf_config import Zonos3Config
from sglang_omni.models.zonos3.request_builders import (
    REQUEST_ERROR_PREFIX,
    Zonos3GenerationInput,
)
from sglang_omni.preprocessing.cache_key import hash_bytes
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.reference_encoder import (
    ReferenceEncodeService,
    TensorReferenceEncodeHook,
)
from sglang_omni.utils.audio import load_audio
from sglang_omni.utils.audio_payload import audio_data_uri_from_reference

# Sampling defaults of the model's reference server.
DEFAULT_TEMPERATURE = 0.75
DEFAULT_TOP_K = 80
DEFAULT_TOP_P = 0.69
DEFAULT_REPETITION_PENALTY = 2.4
MIN_REFERENCE_SECONDS = 0.5
REFERENCE_AUDIO_FIELDS = ("audio_path", "ref_audio", "audio")
REFERENCE_CODE_CACHE_MAX_ITEMS = 256


def build_prompt_rows(
    text_token_ids: list[int],
    reference_codes: torch.Tensor | None,
    config: Zonos3Config,
) -> tuple[torch.Tensor, int]:
    """Prompt rows [rows, frame_width] and the index of the first condition row.

    Voice cloning prepends the reference codes [codebooks, frames] sheared so codebook c lags c rows, with the tail flushed by audio padding; one all-pad row per condition table follows, then the text rows.
    """
    condition_rows = torch.full(
        (len(config.condition_bucket_counts), config.frame_width), config.audio_pad_id
    )
    condition_rows[:, -1] = config.text_pad_id
    text_rows = torch.full(
        (len(text_token_ids), config.frame_width), config.audio_pad_id
    )
    text_rows[:, -1] = torch.tensor(text_token_ids)
    if reference_codes is None:
        reference_rows = torch.empty((0, config.frame_width), dtype=torch.int64)
    else:
        frame_count = reference_codes.shape[1]
        reference_rows = torch.full(
            (frame_count + config.codebook_count - 1, config.frame_width),
            config.audio_pad_id,
        )
        reference_rows[:, -1] = config.text_pad_id
        for codebook_index in range(config.codebook_count):
            reference_rows[
                codebook_index : codebook_index + frame_count, codebook_index
            ] = reference_codes[codebook_index]
    return (
        torch.cat([reference_rows, condition_rows, text_rows]),
        reference_rows.shape[0],
    )


def resolve_condition_buckets(
    config: Zonos3Config,
    buckets: Mapping[str, object] | None,
    values: Mapping[str, object] | None,
) -> tuple[int, ...]:
    """One bucket id per condition table: an explicit id wins, a raw value maps through the edges, else unknown."""
    for mapping, field_name in (
        (buckets, "condition_buckets"),
        (values, "condition_values"),
    ):
        if mapping is None:
            continue
        else:
            pass
        unknown_names = sorted(set(mapping) - set(config.condition_names))
        if unknown_names:
            raise ValueError(
                f"{REQUEST_ERROR_PREFIX} {field_name} names {unknown_names}; "
                f"the conditions are {config.condition_names}"
            )
        else:
            pass
    resolved: list[int] = []
    for name, bucket_count in zip(
        config.condition_names, config.condition_bucket_counts, strict=True
    ):
        bucket = None if buckets is None else buckets.get(name)
        value = None if values is None else values.get(name)
        if bucket is not None:
            if isinstance(bucket, bool) or not isinstance(bucket, int):
                raise ValueError(
                    f"{REQUEST_ERROR_PREFIX} condition_buckets[{name}] must be an integer"
                )
            elif not 0 <= bucket <= bucket_count:
                raise ValueError(
                    f"{REQUEST_ERROR_PREFIX} condition_buckets[{name}] must be in "
                    f"[0, {bucket_count - 1}] or {bucket_count} (unknown)"
                )
            else:
                resolved.append(bucket)
        elif value is not None:
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(
                    f"{REQUEST_ERROR_PREFIX} condition_values[{name}] must be a number"
                )
            elif not math.isfinite(value):
                raise ValueError(
                    f"{REQUEST_ERROR_PREFIX} condition_values[{name}] must be finite"
                )
            elif name not in config.condition_edges:
                raise ValueError(
                    f"{REQUEST_ERROR_PREFIX} the checkpoint has no edges for {name}; "
                    f"send condition_buckets[{name}]"
                )
            else:
                resolved.append(
                    bisect.bisect_right(config.condition_edges[name], float(value))
                )
        else:
            resolved.append(bucket_count)
    return tuple(resolved)


class Zonos3ReferenceCodeHook(TensorReferenceEncodeHook[torch.Tensor]):
    """Encodes a (1, samples) 44.1 kHz reference waveform, cached by its content."""

    model_id = "zonos3"
    model_revision = ""
    encoder_id = "zonos3_dual_codec"
    encoder_config_hash = f"sr{SAMPLE_RATE_HZ}"
    artifact_kind = "reference_codes"
    storage_dtype = torch.int16
    output_dtype = torch.int64

    def __init__(self, encoder: Zonos3CodecEncoder) -> None:
        self.encoder = encoder
        self.encode_stream: torch.cuda.Stream | None = None
        self.encode_lock: threading.Lock = threading.Lock()

    def input_key(self, waveform: torch.Tensor) -> str:
        return hash_bytes(waveform.contiguous().numpy().tobytes())

    def encode_one(self, waveform: torch.Tensor) -> torch.Tensor:
        """Host codes; the CPU resampling runs outside the lock, and on CUDA the encoder runs on its own stream so the engine's stream is never synchronized."""
        lanes = self.encoder.reference_lanes(waveform)
        device = self.encoder.input_conv.weight.device
        if device.type != "cuda":
            return self.encoder.encode_lanes(lanes)
        else:
            with self.encode_lock:
                if self.encode_stream is None:
                    self.encode_stream = torch.cuda.Stream(device=device)
                    self.encode_stream.wait_stream(torch.cuda.current_stream(device))
                else:
                    pass
                with torch.cuda.stream(self.encode_stream):
                    return self.encoder.encode_lanes(lanes).cpu()


def number_parameter(
    request_params: Mapping[str, object], field_name: str, default: float
) -> float:
    """The number request_params holds under field_name, or default when it is absent."""
    value = request_params.get(field_name)
    if value is None:
        return default
    elif isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{REQUEST_ERROR_PREFIX} {field_name} must be a number")
    else:
        return float(value)


def mapping_parameter(
    tts_params: Mapping[str, object], field_name: str
) -> Mapping[str, object] | None:
    """The object tts_params holds under field_name, or None when it is absent."""
    value = tts_params.get(field_name)
    if value is None:
        return None
    elif not isinstance(value, Mapping):
        raise ValueError(f"{REQUEST_ERROR_PREFIX} {field_name} must be an object")
    else:
        return value


class Zonos3SpeechPreprocessor:
    """Turns a speech payload into one carrying Zonos3GenerationInput."""

    def __init__(
        self,
        *,
        tokenizer: Tokenizer,
        text_token_offset: int,
        begin_of_text_id: int,
        end_of_text_id: int,
        config: Zonos3Config,
        reference_encoder: Zonos3CodecEncoder,
        silence_frame: torch.Tensor,
        silence_prefill_frames: int,
        max_new_tokens: int,
        max_reference_seconds: float,
    ) -> None:
        self.tokenizer = tokenizer
        self.text_token_offset = text_token_offset
        self.begin_of_text_id = begin_of_text_id
        self.end_of_text_id = end_of_text_id
        self.config = config
        self.reference_service = ReferenceEncodeService(
            Zonos3ReferenceCodeHook(reference_encoder),
            max_items=REFERENCE_CODE_CACHE_MAX_ITEMS,
            log_prefix="Zonos3 reference code cache",
        )
        self.silence_frame = silence_frame
        self.silence_prefill_frames = silence_prefill_frames
        self.max_new_tokens = max_new_tokens
        self.max_reference_frames = math.floor(
            max_reference_seconds * SAMPLE_RATE_HZ / EXTERNAL_SAMPLES_PER_FRAME
        )

    def reference_codes(self, reference: Mapping[str, object]) -> torch.Tensor:
        """Codes [codebooks, frames] of a reference: given vq_codes, or its audio encoded."""
        vq_codes = reference.get("vq_codes")
        if vq_codes is not None:
            codes = torch.tensor(vq_codes, dtype=torch.int64)
            if codes.dim() != 2 or codes.shape[0] != self.config.codebook_count:
                raise ValueError(
                    f"{REQUEST_ERROR_PREFIX} vq_codes must be "
                    f"[{self.config.codebook_count}, frames]"
                )
            else:
                return codes.clamp(0, self.config.codebook_size - 1)
        else:
            pass
        audio_sources = [
            reference[field_name]
            for field_name in REFERENCE_AUDIO_FIELDS
            if reference.get(field_name) is not None
        ]
        data_uri = audio_data_uri_from_reference(reference)
        if data_uri is not None:
            audio_sources.append(data_uri)
        else:
            pass
        if not audio_sources:
            raise ValueError(
                f"{REQUEST_ERROR_PREFIX} a reference needs audio or vq_codes"
            )
        else:
            pass
        waveform = torch.from_numpy(
            load_audio(audio_sources[0], target_sample_rate=SAMPLE_RATE_HZ, mono=True)
        ).reshape(1, -1)
        if waveform.shape[1] < MIN_REFERENCE_SECONDS * SAMPLE_RATE_HZ:
            raise ValueError(
                f"{REQUEST_ERROR_PREFIX} reference audio is shorter than "
                f"{MIN_REFERENCE_SECONDS:g} s"
            )
        else:
            pass
        peak = float(waveform.abs().amax())
        if peak > 1.0:
            waveform = waveform / peak
        else:
            pass
        # Longer clips keep their first whole frames up to the limit.
        waveform = waveform[:, : self.max_reference_frames * EXTERNAL_SAMPLES_PER_FRAME]
        return self.reference_service.get_or_encode(waveform.contiguous())

    def __call__(self, payload: StagePayload) -> StagePayload:
        request = payload.request
        inputs = request.inputs
        if isinstance(inputs, str):
            text, references = inputs, []
        elif isinstance(inputs, Mapping) and isinstance(inputs.get("text"), str):
            text, references = inputs["text"], list(inputs.get("references") or [])
        else:
            raise ValueError(
                f"{REQUEST_ERROR_PREFIX} the speech input must be text or an object "
                f"with text and references"
            )
        request_params = request.params
        tts_params = request.metadata.get("tts_params")
        if tts_params is None:
            tts_params = {}
        elif not isinstance(tts_params, Mapping):
            raise ValueError(f"{REQUEST_ERROR_PREFIX} tts_params must be an object")
        else:
            pass
        seed = request_params.get("seed")
        prepend_silence = tts_params.get("prepend_silence")
        if len(references) > 1:
            raise ValueError(
                f"{REQUEST_ERROR_PREFIX} at most one reference is supported"
            )
        elif references and not isinstance(references[0], Mapping):
            raise ValueError(f"{REQUEST_ERROR_PREFIX} a reference must be an object")
        elif seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValueError(f"{REQUEST_ERROR_PREFIX} seed must be an integer")
        elif prepend_silence is not None and not isinstance(prepend_silence, bool):
            raise ValueError(
                f"{REQUEST_ERROR_PREFIX} prepend_silence must be a boolean"
            )
        else:
            pass
        reference_codes = self.reference_codes(references[0]) if references else None
        text_token_ids = (
            [self.begin_of_text_id]
            + [
                token_id + self.text_token_offset
                for token_id in self.tokenizer.encode(
                    text, add_special_tokens=False
                ).ids
            ]
            + [self.end_of_text_id]
        )
        prompt_rows, condition_row_index = build_prompt_rows(
            text_token_ids, reference_codes, self.config
        )
        # The prompt ends with the first rows of a codec-silence stream, sheared like generated rows; the output never contains them.
        silence_rows = torch.full(
            (
                0 if prepend_silence is False else self.silence_prefill_frames,
                self.config.frame_width,
            ),
            self.config.audio_pad_id,
        )
        silence_rows[:, -1] = self.config.text_pad_id
        for codebook_index in range(self.config.codebook_count):
            silence_rows[codebook_index:, codebook_index] = self.silence_frame[
                codebook_index
            ]
        return StagePayload(
            request_id=payload.request_id,
            request=request,
            data=Zonos3GenerationInput(
                prompt_rows=torch.cat([prompt_rows, silence_rows]),
                condition_row_index=condition_row_index,
                condition_buckets=resolve_condition_buckets(
                    self.config,
                    mapping_parameter(tts_params, "condition_buckets"),
                    mapping_parameter(tts_params, "condition_values"),
                ),
                max_new_tokens=int(
                    number_parameter(
                        request_params, "max_new_tokens", self.max_new_tokens
                    )
                ),
                temperature=number_parameter(
                    request_params, "temperature", DEFAULT_TEMPERATURE
                ),
                top_k=int(number_parameter(request_params, "top_k", DEFAULT_TOP_K)),
                top_p=number_parameter(request_params, "top_p", DEFAULT_TOP_P),
                repetition_penalty=number_parameter(
                    request_params, "repetition_penalty", DEFAULT_REPETITION_PENALTY
                ),
                seed=seed,
            ),
        )
