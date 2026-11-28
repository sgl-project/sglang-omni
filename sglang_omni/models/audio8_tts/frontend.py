# SPDX-License-Identifier: Apache-2.0
"""Pack Audio8 prompts and decode predicted audio codebooks."""

import re
from pathlib import Path

import torch
from tokenizers import Tokenizer

from sglang_omni.models.audio8_tts.codec.model import load_audio8_codec
from sglang_omni.models.audio8_tts.codec.quantization import (
    CODEBOOK_COUNT,
    SEMANTIC_CODEBOOK_SIZE,
)
from sglang_omni.models.audio8_tts.hf_config import (
    FRAME_SAMPLES,
    SAMPLE_RATE_HZ,
    ArkttsConfig,
)
from sglang_omni.models.audio8_tts.payload_types import Audio8State
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.pipeline_state import store_state
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.audio import AudioDecodeError, load_audio
from sglang_omni.utils.audio_payload import (
    audio_data_uri_from_reference,
    audio_waveform_payload,
)
from sglang_omni.utils.checkpoint import resolve_checkpoint
from sglang_omni.utils.device import resolve_concrete_device

SPEAKER_TAG = re.compile(r"<\|speaker:\d+\|>")


def tokenize_prompt_segments(tokenizer: Tokenizer, segments: list[str]) -> torch.Tensor:
    return torch.tensor(
        [
            token_id
            for segment in segments
            for token_id in tokenizer.encode(segment, add_special_tokens=False).ids
        ],
        dtype=torch.long,
        device="cpu",
    )


def create_preprocessing_executor(
    model_path: str, device: str | None = None, gpu_id: int | None = None
) -> SimpleScheduler[StagePayload, StagePayload]:
    checkpoint = Path(resolve_checkpoint(model_path)).resolve()
    config = ArkttsConfig.from_pretrained(checkpoint)
    if (
        config.num_codebooks != CODEBOOK_COUNT
        or config.codebook_size != SEMANTIC_CODEBOOK_SIZE
        or config.codec_sample_rate != SAMPLE_RATE_HZ
        or config.codec_frame_size != FRAME_SAMPLES
    ):
        raise ValueError(
            "Audio8 checkpoint uses unsupported codebook or audio dimensions"
        )
    else:
        pass
    tokenizer = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
    codec_device = resolve_concrete_device(device, gpu_id)

    def preprocess(payload: StagePayload) -> StagePayload:
        inputs = payload.request.inputs
        if isinstance(inputs, str):
            synthesis_text = inputs
            references = []
        elif isinstance(inputs, dict):
            synthesis_text = inputs.get("text", inputs.get("input"))
            references = inputs.get("references", [])
        else:
            raise ValueError("Audio8 expects synthesis text and an optional reference")
        if not isinstance(synthesis_text, str) or not synthesis_text.strip():
            raise ValueError("Audio8 requires nonempty synthesis text")
        elif not isinstance(references, list) or len(references) > 1:
            raise ValueError("Audio8 accepts at most one reference")
        else:
            pass

        system_start_segment = "<|im_start|>system\n"
        target_turn_segments = [
            "<|im_end|>\n",
            "<|im_start|>user\n",
            " ".join(synthesis_text.split()),
            "<|im_end|>\n",
            "<|im_start|>assistant\n<|voice|>",
        ]
        system_instruction = "convert the provided text to speech"
        waveform = None
        if references:
            reference = references[0]
            if not isinstance(reference, dict):
                raise ValueError("Audio8 references must be objects")
            else:
                transcript = reference.get("text")
            if not isinstance(transcript, str) or not transcript.strip():
                raise ValueError("Audio8 requires a transcript for reference audio")
            else:
                transcript = " ".join(transcript.split())
            reference_source = (
                reference.get("audio_path")
                or reference.get("ref_audio")
                or reference.get("audio")
                or reference.get("bytes")
                or audio_data_uri_from_reference(reference)
            )
            if reference_source is None:
                raise ValueError("Audio8 reference has no audio")
            else:
                pass
            if SPEAKER_TAG.search(transcript) is None:
                transcript = f"<|speaker:0|>{transcript}"
            else:
                pass
            prefix_segments = [
                system_start_segment,
                f"{system_instruction} reference to the following:\n\nText:\n",
                transcript,
                "\n\nSpeech:\n",
            ]
            suffix_segments = target_turn_segments
            waveform = torch.from_numpy(
                load_audio(
                    reference_source,
                    source_name="Audio8 reference",
                    target_sample_rate=config.codec_sample_rate,
                )
            )
            if not waveform.numel():
                raise AudioDecodeError("Could not decode Audio8 reference audio input")
            else:
                pass
        else:
            prefix_segments = [
                system_start_segment,
                system_instruction,
                *target_turn_segments,
            ]
            suffix_segments = []

        prefix_token_ids = tokenize_prompt_segments(tokenizer, prefix_segments)
        suffix_token_ids = tokenize_prompt_segments(tokenizer, suffix_segments)
        reference_frame_count = (
            (waveform.numel() + config.codec_frame_size - 1) // config.codec_frame_size
            if waveform is not None
            else 0
        )
        reference_start_position = prefix_token_ids.numel()
        reference_end_position = reference_start_position + reference_frame_count
        prompt_position_count = reference_end_position + suffix_token_ids.numel()
        if prompt_position_count >= config.max_seq_len:
            raise ValueError(
                f"Audio8 prompt has {prompt_position_count} positions and is longer than the model's "
                f"context length after reserving one generation position ({config.max_seq_len})"
            )
        else:
            pass
        prompt_codes = torch.zeros(
            (prompt_position_count, config.num_codebooks + 1),
            dtype=torch.long,
            device="cpu",
        )
        prompt_codes[:reference_start_position, 0] = prefix_token_ids
        prompt_codes[reference_end_position:, 0] = suffix_token_ids
        if waveform is None:
            pass
        else:
            reference_codes = (
                load_audio8_codec(checkpoint, codec_device).encode(waveform).cpu()
            )
            assert reference_codes.shape == (
                config.num_codebooks,
                reference_frame_count,
            )
            prompt_codes[reference_start_position:reference_end_position, 0] = (
                reference_codes[0] + config.semantic_begin_id
            )
            prompt_codes[reference_start_position:reference_end_position, 1:] = (
                reference_codes.transpose(0, 1)
            )

        return store_state(payload, Audio8State(prompt_codes=prompt_codes))

    return SimpleScheduler(preprocess)


def create_vocoder_executor(
    model_path: str, device: str | None = None, gpu_id: int | None = None
) -> SimpleScheduler[StagePayload, StagePayload]:
    checkpoint = Path(resolve_checkpoint(model_path)).resolve()
    codec = load_audio8_codec(checkpoint, resolve_concrete_device(device, gpu_id))

    def decode(payload: StagePayload) -> StagePayload:
        generated = Audio8State.from_dict(payload.data)
        audio_codes = generated.audio_codes
        assert isinstance(audio_codes, torch.Tensor)
        assert audio_codes.ndim == 2 and audio_codes.dtype == torch.long
        assert audio_codes.shape[1] > 0
        waveform = codec.decode(audio_codes)
        generated.audio_codes = None
        payload.data = generated.to_dict() | audio_waveform_payload(
            waveform,
            sample_rate=generated.sample_rate,
            modality="audio",
            source_hint="Audio8",
        )
        return payload

    return SimpleScheduler(decode)
