# SPDX-License-Identifier: Apache-2.0
"""Stage executors of the LFM3-Audio pipeline: preprocessing, generation, decode."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from sglang_omni.models.lfm3_audio.io import Lfm3AudioState
from sglang_omni.preprocessing.audio import AudioMediaIO
from sglang_omni.preprocessing.base import is_url
from sglang_omni.preprocessing.resource_connector import get_global_resource_connector
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import load_state, store_state
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.audio_payload import audio_waveform_payload
from sglang_omni.utils.checkpoint import resolve_checkpoint
from sglang_omni.utils.device import resolve_concrete_device

if TYPE_CHECKING:
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler
else:
    pass

logger = logging.getLogger(__name__)

INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
AUDIO_START = "<|audio_start|>"
MIXED_START = "<|mixed_start|>"
# An empty reasoning block: text answers come without the model's thinking.
NO_THINKING = "<think></think>"
DEFAULT_TTS_LANGUAGE = "American English"
DEFAULT_TTS_VOICE = "female"
TTS_VOICES = frozenset({"female", "male"})
INTERLEAVED_SYSTEM_PROMPT = "Respond with interleaved text and audio."
AUDIO_PART_TYPES = frozenset({"audio", "audio_url", "input_audio"})


def load_processor(model_path: str):
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(
        resolve_checkpoint(model_path), trust_remote_code=True
    )


def tts_system_prompt(tts_params: Mapping[str, object]) -> str:
    """The checkpoint's TTS instruction; `instructions` replaces it verbatim."""
    instructions = tts_params.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        return instructions.strip()
    else:
        pass
    voice = str(tts_params.get("voice") or DEFAULT_TTS_VOICE).lower()
    if voice in ("", "default"):
        voice = DEFAULT_TTS_VOICE
    elif voice not in TTS_VOICES:
        raise ValueError(f"LFM3-Audio voices are {sorted(TTS_VOICES)}, got {voice!r}")
    else:
        pass
    language = str(tts_params.get("language") or DEFAULT_TTS_LANGUAGE)
    return f"Perform TTS in {language} using a {voice} voice."


def as_messages(inputs: object) -> list[dict]:
    if isinstance(inputs, str):
        return [{"role": "user", "content": inputs}]
    elif isinstance(inputs, list):
        return [dict(message) for message in inputs]
    elif isinstance(inputs, dict) and "messages" in inputs:
        return [dict(message) for message in inputs["messages"]]
    else:
        raise ValueError(
            f"LFM3-Audio cannot read chat inputs of type {type(inputs).__name__}"
        )


def audio_source(part: Mapping[str, object]) -> object:
    kind = part.get("type")
    if kind == "input_audio":
        audio = part["input_audio"]
        return f"data:audio/{audio['format']};base64,{audio['data']}"
    elif kind == "audio_url":
        url = part["audio_url"]
        return url["url"] if isinstance(url, Mapping) else url
    else:
        return part.get("audio")


async def load_waveform(source: object, media_io: AudioMediaIO) -> np.ndarray:
    if isinstance(source, np.ndarray):
        return source.astype(np.float32)
    elif isinstance(source, str) and is_url(source):
        waveform, _ = await get_global_resource_connector().load_resource_async(
            source, media_io
        )
        return waveform
    elif isinstance(source, (str, Path)):
        waveform, _ = media_io.load_file(Path(source))
        return waveform
    else:
        raise ValueError(f"Unsupported LFM3-Audio audio input {type(source).__name__}")


def audio_sampling(params: Mapping[str, object], is_tts: bool) -> dict[str, object]:
    """Speech-frame sampling; a TTS request's temperature, top_k and repetition_penalty set it."""
    if is_tts:
        names = {
            "audio_temperature": "temperature",
            "audio_top_k": "top_k",
            "audio_repetition_penalty": "repetition_penalty",
        }
    else:
        names = {
            name: name
            for name in ("audio_temperature", "audio_top_k", "audio_repetition_penalty")
        }
    return {field: params.get(source) for field, source in names.items()}


class Lfm3AudioPreprocessor:
    """Builds the prompt ids and the input clips' log-mel features."""

    def __init__(self, model_path: str) -> None:
        self.processor = load_processor(model_path)
        self.tokenizer = self.processor.tokenizer
        self.media_io = AudioMediaIO(target_sr=INPUT_SAMPLE_RATE)

    async def __call__(self, payload: StagePayload) -> StagePayload:
        request = payload.request
        params = request.params or {}
        metadata = request.metadata or {}
        tts_params = metadata.get("tts_params") or {}
        is_tts = metadata.get("task") == "tts"
        waveforms: list[np.ndarray] = []
        if is_tts:
            inputs = request.inputs
            text = inputs if isinstance(inputs, str) else (inputs or {}).get("text", "")
            messages = [
                {"role": "system", "content": tts_system_prompt(tts_params)},
                {"role": "user", "content": text},
            ]
            marker = AUDIO_START
            text_only = False
        else:
            messages = as_messages(request.inputs)
            for message in messages:
                content = message.get("content")
                if isinstance(content, list):
                    parts = []
                    for part in content:
                        part = dict(part)
                        if part.get("type") in AUDIO_PART_TYPES:
                            waveforms.append(
                                await load_waveform(audio_source(part), self.media_io)
                            )
                            part = {"type": "audio"}
                        else:
                            pass
                        parts.append(part)
                    message["content"] = parts
                else:
                    pass
            extra_audio = (
                request.inputs.get("audio") or request.inputs.get("audios")
                if isinstance(request.inputs, dict)
                else None
            )
            for source in extra_audio or []:
                waveforms.append(await load_waveform(source, self.media_io))
                messages[-1]["content"] = (
                    [{"type": "text", "text": messages[-1]["content"]}]
                    if isinstance(messages[-1]["content"], str)
                    else list(messages[-1]["content"])
                ) + [{"type": "audio"}]
            speaks = "audio" in (metadata.get("output_modalities") or [])
            if speaks and not any(
                message.get("role") == "system" for message in messages
            ):
                messages.insert(
                    0, {"role": "system", "content": INTERLEAVED_SYSTEM_PROMPT}
                )
            else:
                pass
            marker = MIXED_START if speaks else NO_THINKING
            text_only = not speaks
        prompt = (
            self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            + marker
        )
        encoded = self.processor(
            text=prompt,
            audio=waveforms or None,
            sampling_rate=INPUT_SAMPLE_RATE,
            return_tensors="pt",
        )
        state = Lfm3AudioState(
            input_ids=encoded["input_ids"][0].tolist(),
            input_features=encoded.get("input_features"),
            feature_attention_mask=encoded.get("feature_attention_mask"),
            text_only=text_only,
            max_new_tokens=int(params.get("max_new_tokens") or 2048),
            temperature=0.0 if is_tts else float(params.get("temperature") or 0.0),
            top_p=1.0 if is_tts else float(params.get("top_p") or 1.0),
            top_k=-1 if is_tts else int(params.get("top_k") or -1),
            seed=params.get("seed"),
            **audio_sampling(params, is_tts),
        )
        return store_state(payload, state)


def create_preprocessing_executor(model_path: str) -> SimpleScheduler:
    return SimpleScheduler(Lfm3AudioPreprocessor(model_path))


def create_generation_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    server_args_overrides: Mapping[str, object] | None = None,
) -> OmniScheduler:
    from sglang_omni.models.lfm3_audio.engine_builder import Lfm3AudioEngineBuilder

    return Lfm3AudioEngineBuilder().build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        server_args_overrides=server_args_overrides,
    )


class Lfm3AudioDecoder:
    """Detokenizes the text and decodes the speech frames with LiquidCodec (24 kHz)."""

    def __init__(self, model_path: str, device: str) -> None:
        processor = load_processor(model_path)
        self.tokenizer = processor.tokenizer
        self.codec = processor.audio_tokenizer.to(device=device).eval()
        self.device = device

    @torch.inference_mode()
    def __call__(self, payload: StagePayload) -> StagePayload:
        state = load_state(payload, Lfm3AudioState)
        text = self.tokenizer.decode(
            state.generated_ids, skip_special_tokens=True
        ).strip()
        data: dict[str, object] = state.to_dict()
        data["text"] = text
        codes = state.audio_codes
        if codes is not None and codes.numel():
            waveform = self.codec.decode(
                codes.T[None].to(self.device, dtype=torch.long)
            ).audio_values[0]
            data.update(
                audio_waveform_payload(waveform, sample_rate=OUTPUT_SAMPLE_RATE)
            )
        else:
            pass
        payload.data = data
        return payload


def create_decode_executor(
    model_path: str, *, device: str | None = None, gpu_id: int | None = None
) -> SimpleScheduler:
    decoder = Lfm3AudioDecoder(model_path, str(resolve_concrete_device(device, gpu_id)))
    return SimpleScheduler(decoder)
