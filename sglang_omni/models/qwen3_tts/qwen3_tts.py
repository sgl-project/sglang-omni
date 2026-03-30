# SPDX-License-Identifier: Apache-2.0
"""Qwen3-TTS pipeline: components, engine I/O, and stage executors."""

from __future__ import annotations

import base64
import io
import logging
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlparse

import librosa
import numpy as np
import soundfile as sf
import torch
from torch import nn
from transformers import AutoTokenizer
from transformers.utils.hub import cached_file

from sglang_omni.config import ExecutorConfig, PipelineConfig, RelayConfig, StageConfig
from sglang_omni.engines.omni import create_ar_engine, create_encoder_engine
from sglang_omni.engines.omni.model_runner import ModelRunner
from sglang_omni.engines.omni.runtime.ar import (
    ARBatchPlanner,
    ARInputPreparer,
    AROutputProcessor,
    ARRequestData,
    ARResourceManager,
)
from sglang_omni.engines.omni.runtime.common import EosIterationController
from sglang_omni.engines.omni.runtime.encoder import EncoderRequestData
from sglang_omni.engines.omni.scheduler import Scheduler
from sglang_omni.executors import EngineExecutor, PreprocessingExecutor
from sglang_omni.models.qwen3_tts.configuration_qwen3_tts import Qwen3TTSConfig
from sglang_omni.models.qwen3_tts.modeling_qwen3_tts import (
    Qwen3TTSSpeakerEncoder,
    Qwen3TTSTalkerForConditionalGeneration,
    mel_spectrogram,
)
from sglang_omni.models.qwen3_tts.qwen3_tts_tokenizer import Qwen3TTSSpeechTokenizer
from sglang_omni.models.utils.hf import load_hf_config
from sglang_omni.models.weight_loader import load_module
from sglang_omni.proto import StagePayload

logger = logging.getLogger(__name__)


@dataclass
class TTSPipelineState:
    """Per-request TTS pipeline state (msgpack-safe via to_dict/from_dict)."""

    task_type: str = "custom_voice"
    text: str = ""
    speaker: str | None = None
    language: str = "auto"
    non_streaming_mode: bool = False

    input_ids: Any = None
    instruct_ids: Any = None
    ref_ids: Any = None
    voice_clone_prompt: dict[str, Any] | None = None

    speaker_embedding: Any = None
    speech_codes: Any = None
    talker_hidden_states: Any = None

    audio_waveform: Any = None
    sample_rate: int = 24000

    engine_outputs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> TTSPipelineState:
        data = data or {}
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


TTSEventType = Literal[
    "audio_chunk",
    "audio_final",
    "debug",
    "final",
]


@dataclass
class TTSEvent:
    """Streaming-friendly event emitted by TTS decode stage."""

    type: TTSEventType
    payload: dict[str, Any]
    is_final: bool = False


def load_tts_config(model_id: str) -> Qwen3TTSConfig:
    return load_hf_config(model_id, trust_remote_code=True, local_files_only=False)


def load_state(payload: StagePayload) -> TTSPipelineState:
    return TTSPipelineState.from_dict(payload.data)


def store_state(payload: StagePayload, state: TTSPipelineState) -> StagePayload:
    payload.data = state.to_dict()
    return payload


FRONTEND_STAGE = "tts_frontend"
SPEAKER_ENCODER_STAGE = "tts_speaker_encoder"
TALKER_STAGE = "tts_talker"
CODE2WAV_STAGE = "tts_code2wav"
DECODE_STAGE = "tts_decode"


def frontend_next(request_id: str, output: Any) -> str:
    del request_id
    if isinstance(output, StagePayload):
        state = TTSPipelineState.from_dict(output.data)
        if state.task_type == "base":
            return SPEAKER_ENCODER_STAGE
    return TALKER_STAGE


def speaker_encoder_next(request_id: str, output: Any) -> str:
    del request_id, output
    return TALKER_STAGE


def talker_next(request_id: str, output: Any) -> str:
    del request_id, output
    return CODE2WAV_STAGE


def code2wav_next(request_id: str, output: Any) -> str:
    del request_id, output
    return DECODE_STAGE


def decode_next(request_id: str, output: Any) -> None:
    del request_id, output
    return None


def _extract_text_from_messages(messages: list[dict]) -> str:
    """Extract text from the last user message in OpenAI chat format."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(
                    p["text"]
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
    return ""


class Qwen3TTSFrontend:
    """CPU preprocessing: tokenize text and populate TTSPipelineState."""

    def __init__(self, model_id: str):
        self.config = load_tts_config(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    def _tokenize(self, text: str):
        return self.tokenizer(
            f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n",
            return_tensors="pt",
        )

    def _tokenize_ref(self, ref_text: str):
        return self.tokenizer(
            f"<|im_start|>assistant\n{ref_text}<|im_end|>\n",
            return_tensors="pt",
        )

    def __call__(self, payload: StagePayload) -> StagePayload:
        state = load_state(payload)
        req = payload.request
        params = req.params if req else {}
        inputs = req.inputs if req else None

        # Resolve text from params, inputs dict, or chat messages.
        text = params.get("text")
        if not text:
            if isinstance(inputs, str):
                text = inputs
            elif isinstance(inputs, dict):
                text = inputs.get("text") or inputs.get("prompt") or ""
                if not text and isinstance(inputs.get("messages"), list):
                    text = _extract_text_from_messages(inputs["messages"])
            elif isinstance(inputs, list):
                text = _extract_text_from_messages(inputs)

        state.text = text or state.text
        state.speaker = params.get("speaker", state.speaker)
        state.language = params.get("language", state.language or "auto")
        state.task_type = params.get("task_type", state.task_type or "custom_voice")
        state.non_streaming_mode = bool(
            params.get("non_streaming_mode", state.non_streaming_mode)
        )

        if not state.text or not state.text.strip():
            raise ValueError("TTS text is empty.")

        state.input_ids = self._tokenize(state.text)["input_ids"]

        instruct_text = params.get("instruct_text")
        if instruct_text and state.task_type == "voice_design":
            state.instruct_ids = self._tokenize(instruct_text)["input_ids"]

        ref_audio = params.get("ref_audio")
        if ref_audio is not None:
            vcp = state.voice_clone_prompt or {}
            vcp["ref_audio"] = ref_audio
            vcp["ref_sr"] = int(params.get("ref_sr", 24000))
            state.voice_clone_prompt = vcp

        if state.task_type == "base":
            ref_text = params.get("ref_text", "")
            if ref_text:
                state.ref_ids = self._tokenize_ref(ref_text)["input_ids"]
            if ref_audio is None:
                raise ValueError("task_type='base' requires ref_audio.")

        return store_state(payload, state)


def _load_audio(audio: Any, sr: int | None, target_sr: int = 24000) -> torch.Tensor:
    """Load audio from tensor, ndarray, file path, URL, or base64 string."""
    if isinstance(audio, torch.Tensor):
        wav = audio.detach().float().cpu().numpy()
        src_sr = sr or target_sr
    elif isinstance(audio, np.ndarray):
        wav = audio.astype(np.float32, copy=False)
        src_sr = sr or target_sr
    elif isinstance(audio, str):
        parsed = urlparse(audio)
        if parsed.scheme in ("http", "https"):
            with urllib.request.urlopen(audio) as resp:
                wav, src_sr = sf.read(io.BytesIO(resp.read()), dtype="float32")
        elif audio.startswith("data:audio") or (len(audio) > 256 and "/" not in audio):
            raw = audio.split(",", 1)[-1] if "," in audio else audio
            wav, src_sr = sf.read(io.BytesIO(base64.b64decode(raw)), dtype="float32")
        else:
            wav, src_sr = librosa.load(audio, sr=None, mono=True)
    else:
        raise TypeError(f"Unsupported audio type: {type(audio)}")

    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    if src_sr != target_sr:
        wav = librosa.resample(y=wav, orig_sr=src_sr, target_sr=target_sr)
    return torch.from_numpy(wav)


class Qwen3TTSSpeakerEncoderComponent(nn.Module):
    """GPU wrapper around the ECAPA-TDNN speaker encoder."""

    def __init__(self, model_id: str, device: str = "cuda", dtype: str | None = None):
        super().__init__()
        config = load_tts_config(model_id)
        self.speaker_encoder = Qwen3TTSSpeakerEncoder(config.speaker_encoder_config)

        torch_dtype = getattr(torch, dtype) if dtype else torch.float32
        load_module(
            self.speaker_encoder,
            model_id,
            prefix=("speaker_encoder.",),
            device=device,
            dtype=torch_dtype,
        )
        self.to(device=device, dtype=torch_dtype)

    @torch.inference_mode()
    def forward(self, audio: torch.Tensor) -> dict[str, torch.Tensor]:
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        mels = mel_spectrogram(
            audio,
            n_fft=1024,
            num_mels=128,
            sampling_rate=24000,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=12000,
        ).transpose(1, 2)

        dev = next(self.parameters())
        mels = mels.to(device=dev.device, dtype=dev.dtype)
        return {"speaker_embedding": self.speaker_encoder(mels)}


def _resolve_speech_tokenizer_source(model_id: str) -> str:
    """Resolve local path to the speech tokenizer config."""
    local_subdir = os.path.join(model_id, "speech_tokenizer", "config.json")
    if os.path.isfile(local_subdir):
        return os.path.dirname(local_subdir)
    if os.path.isfile(os.path.join(model_id, "config.json")):
        return model_id
    cfg_file = cached_file(model_id, "speech_tokenizer/config.json")
    return os.path.dirname(cfg_file)


def _to_numpy(value: Any) -> Any:
    """Convert tensor/ndarray to float32 numpy; pass strings through."""
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().numpy()
    if isinstance(value, np.ndarray):
        value = value.astype(np.float32, copy=False)
        if value.ndim > 1:
            value = value.mean(axis=-1)
    return value


def _unwrap(value: Any) -> Any:
    return value[0] if isinstance(value, list) and len(value) == 1 else value


class Qwen3TTSCode2WavComponent(nn.Module):
    """GPU wrapper: speech codes -> waveform via the speech tokenizer."""

    def __init__(self, model_id: str, device: str = "cuda", dtype: str | None = None):
        super().__init__()
        torch_dtype = getattr(torch, dtype) if dtype else torch.float32
        source = _resolve_speech_tokenizer_source(model_id)
        self.speech_tokenizer = Qwen3TTSSpeechTokenizer.from_pretrained(
            source,
            device_map=device,
            torch_dtype=torch_dtype,
        )
        self.tokenizer_model_type = self.speech_tokenizer.get_model_type()

    @torch.inference_mode()
    def forward(
        self,
        speech_codes: torch.Tensor,
        ref_audio: Any = None,
        ref_sr: int = 24000,
        xvectors: Any = None,
        ref_mels: Any = None,
    ) -> dict[str, Any]:
        encoded: dict[str, Any] = {"audio_codes": speech_codes}

        if self.tokenizer_model_type == "qwen3_tts_tokenizer_25hz":
            ref_audio, ref_sr = _unwrap(ref_audio), _unwrap(ref_sr)
            xvectors, ref_mels = _unwrap(xvectors), _unwrap(ref_mels)

            if xvectors is None or ref_mels is None:
                ref_encoded = self.speech_tokenizer.encode(
                    _to_numpy(ref_audio),
                    sr=int(ref_sr or 24000),
                    return_dict=True,
                )
                xvectors = getattr(ref_encoded, "xvectors", None)
                ref_mels = getattr(ref_encoded, "ref_mels", None)

            encoded["xvectors"] = xvectors
            encoded["ref_mels"] = ref_mels

        wavs, sample_rate = self.speech_tokenizer.decode(encoded)
        return {"audio_waveform": wavs, "sample_rate": sample_rate}


def _resolve_speaker_embed(
    talker: Qwen3TTSTalkerForConditionalGeneration,
    config: Qwen3TTSConfig,
    speaker: str | None,
    speaker_embedding: torch.Tensor | None,
    voice_clone_prompt: dict[str, Any] | None,
    input_ids: torch.Tensor,
) -> torch.Tensor | None:
    device = next(talker.parameters()).device
    dtype = next(talker.parameters()).dtype
    tc = config.talker_config

    if speaker_embedding is not None:
        return speaker_embedding.to(device=device, dtype=dtype)

    if voice_clone_prompt is not None:
        ref_embeds = voice_clone_prompt.get("ref_spk_embedding")
        if ref_embeds:
            x_mode = voice_clone_prompt.get("x_vector_only_mode", [False])[0]
            icl_mode = voice_clone_prompt.get("icl_mode", [False])[0]
            if x_mode or icl_mode:
                return ref_embeds[0].to(device=device, dtype=dtype)
        return None

    if not speaker:
        return None

    spk_ids = (tc.spk_id or {}).get(speaker.lower())
    if spk_ids is None:
        raise ValueError(f"Unknown speaker: {speaker!r}")
    spk_id = int(spk_ids[0]) if isinstance(spk_ids, (list, tuple)) else int(spk_ids)
    return talker.get_input_embeddings()(
        torch.tensor([[spk_id]], device=device, dtype=input_ids.dtype)
    ).squeeze(1)


def _resolve_language_id(tc: Any, speaker: str | None, language: str) -> int | None:
    lang = language.lower()
    lang_map = tc.codec_language_id or {}

    language_id = lang_map.get(lang) if lang != "auto" else None

    # Dialect override for Chinese speakers.
    if lang in ("chinese", "auto") and speaker:
        dialect = (tc.spk_is_dialect or {}).get(speaker.lower())
        if dialect:
            language_id = lang_map[dialect]

    return language_id


def _build_icl_prompt(
    talker: Qwen3TTSTalkerForConditionalGeneration,
    config: Qwen3TTSConfig,
    text_id: torch.Tensor,
    ref_id: torch.Tensor,
    ref_code: torch.Tensor,
    tts_pad_embed: torch.Tensor,
    tts_eos_embed: torch.Tensor,
    non_streaming_mode: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = text_id.device
    tc = config.talker_config

    text_embed = talker.text_projection(
        talker.get_text_embeddings()(torch.cat([ref_id, text_id], dim=-1))
    )
    text_embed = torch.cat([text_embed, tts_eos_embed], dim=1)

    codec_parts = [talker.get_input_embeddings()(ref_code[:, :1])]
    for i in range(1, tc.num_code_groups):
        codec_parts.append(
            talker.code_predictor.get_input_embeddings()[i - 1](ref_code[:, i : i + 1])
        )
    codec_embed = torch.cat(codec_parts, dim=1).sum(1).unsqueeze(0)
    bos = talker.get_input_embeddings()(
        torch.tensor([[tc.codec_bos_id]], device=device, dtype=text_id.dtype)
    )
    codec_embed = torch.cat([bos, codec_embed], dim=1)

    text_len = text_embed.shape[1]
    codec_len = codec_embed.shape[1]

    if non_streaming_mode:
        pad_ids = torch.full(
            (1, text_len), tc.codec_pad_id, device=device, dtype=text_id.dtype
        )
        embed = text_embed + talker.get_input_embeddings()(pad_ids)
        return torch.cat([embed, codec_embed + tts_pad_embed], dim=1), tts_pad_embed

    if text_len > codec_len:
        return text_embed[:, :codec_len] + codec_embed, text_embed[:, codec_len:]

    padding = torch.cat([tts_pad_embed] * (codec_len - text_len), dim=1)
    text_embed = torch.cat([text_embed, padding], dim=1)
    return text_embed + codec_embed, tts_pad_embed


@torch.inference_mode()
def build_tts_prompt(
    talker: Qwen3TTSTalkerForConditionalGeneration,
    config: Qwen3TTSConfig,
    *,
    input_ids: torch.Tensor,
    speaker: str | None,
    language: str,
    non_streaming_mode: bool = False,
    instruct_ids: torch.Tensor | None = None,
    ref_ids: torch.Tensor | None = None,
    voice_clone_prompt: dict[str, Any] | None = None,
    speaker_embedding: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Build Talker prefill embeddings, attention mask, and generation metadata."""
    device = next(talker.parameters()).device
    tc = config.talker_config

    input_ids = input_ids.to(device)
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)

    # Instruct embedding (voice design mode).
    instruct_embed = None
    if instruct_ids is not None:
        instruct_ids = instruct_ids.to(device)
        if instruct_ids.ndim == 1:
            instruct_ids = instruct_ids.unsqueeze(0)
        instruct_embed = talker.text_projection(
            talker.get_text_embeddings()(instruct_ids)
        )

    speaker_embed = _resolve_speaker_embed(
        talker,
        config,
        speaker,
        speaker_embedding,
        voice_clone_prompt,
        input_ids,
    )
    language_id = _resolve_language_id(tc, speaker, language)

    # Special token embeddings.
    special_ids = torch.tensor(
        [[config.tts_bos_token_id, config.tts_eos_token_id, config.tts_pad_token_id]],
        device=device,
        dtype=input_ids.dtype,
    )
    tts_bos_embed, tts_eos_embed, tts_pad_embed = talker.text_projection(
        talker.get_text_embeddings()(special_ids)
    ).chunk(3, dim=1)

    # Codec prefill: think tokens + optional language.
    if language_id is None:
        prefill = [[tc.codec_nothink_id, tc.codec_think_bos_id, tc.codec_think_eos_id]]
    else:
        prefill = [
            [
                tc.codec_think_id,
                tc.codec_think_bos_id,
                language_id,
                tc.codec_think_eos_id,
            ]
        ]

    embed_fn = talker.get_input_embeddings()
    codec_embed_0 = embed_fn(
        torch.tensor(prefill, device=device, dtype=input_ids.dtype)
    )
    codec_embed_1 = embed_fn(
        torch.tensor(
            [[tc.codec_pad_id, tc.codec_bos_id]], device=device, dtype=input_ids.dtype
        )
    )

    parts = [codec_embed_0]
    if speaker_embed is not None:
        parts.append(speaker_embed.view(1, 1, -1))
    parts.append(codec_embed_1)
    codec_input_embedding = torch.cat(parts, dim=1)

    role_embed = talker.text_projection(talker.get_text_embeddings()(input_ids[:, :3]))
    pad_expanded = tts_pad_embed.expand(-1, codec_input_embedding.shape[1] - 2, -1)
    interleave = (
        torch.cat((pad_expanded, tts_bos_embed), dim=1) + codec_input_embedding[:, :-1]
    )
    talker_input_embed = torch.cat((role_embed, interleave), dim=1)

    # ICL (in-context learning) mode for voice cloning.
    icl_mode = (
        voice_clone_prompt is not None
        and voice_clone_prompt.get("ref_code") is not None
        and voice_clone_prompt.get("icl_mode", [False])[0]
    )

    if icl_mode:
        ref_ids_t = ref_ids.to(device)
        if ref_ids_t.ndim == 1:
            ref_ids_t = ref_ids_t.unsqueeze(0)
        icl_embed, trailing_text_hidden = _build_icl_prompt(
            talker,
            config,
            text_id=input_ids[:, 3:-5],
            ref_id=ref_ids_t[:, 3:-2],
            ref_code=voice_clone_prompt["ref_code"][0].to(device),
            tts_pad_embed=tts_pad_embed,
            tts_eos_embed=tts_eos_embed,
            non_streaming_mode=non_streaming_mode,
        )
        talker_input_embed = torch.cat([talker_input_embed, icl_embed], dim=1)
    else:
        first_text_embed = (
            talker.text_projection(talker.get_text_embeddings()(input_ids[:, 3:4]))
            + codec_input_embedding[:, -1:]
        )
        talker_input_embed = torch.cat([talker_input_embed, first_text_embed], dim=1)

        if non_streaming_mode:
            talker_input_embed = talker_input_embed[:, :-1]
            text_content = input_ids[:, 3:-5]
            text_embed = talker.text_projection(
                talker.get_text_embeddings()(text_content)
            )
            text_plus_eos = torch.cat((text_embed, tts_eos_embed), dim=1)
            n_pad = text_content.shape[1] + 1
            pad_ids = torch.full(
                (1, n_pad), tc.codec_pad_id, device=device, dtype=input_ids.dtype
            )
            text_plus_eos = text_plus_eos + embed_fn(pad_ids)

            bos_embed = tts_pad_embed + embed_fn(
                torch.tensor([[tc.codec_bos_id]], device=device, dtype=input_ids.dtype)
            )
            talker_input_embed = torch.cat(
                [talker_input_embed, text_plus_eos, bos_embed],
                dim=1,
            )
            trailing_text_hidden = tts_pad_embed
        else:
            trailing_text_hidden = torch.cat(
                (
                    talker.text_projection(
                        talker.get_text_embeddings()(input_ids[:, 4:-5])
                    ),
                    tts_eos_embed,
                ),
                dim=1,
            )

    if instruct_embed is not None:
        talker_input_embed = torch.cat([instruct_embed, talker_input_embed], dim=1)

    seq_len = talker_input_embed.shape[1]
    suppress_tokens = [
        i
        for i in range(tc.vocab_size - 1024, tc.vocab_size)
        if i != tc.codec_eos_token_id
    ]

    return {
        "inputs_embeds": talker_input_embed,
        "attention_mask": torch.ones(1, seq_len, dtype=torch.long, device=device),
        "trailing_text_hidden": trailing_text_hidden,
        "tts_pad_embed": tts_pad_embed,
        "suppress_tokens": suppress_tokens,
        "codec_eos_token_id": tc.codec_eos_token_id,
    }


class _SubtalkerStepper:
    """Synchronous AR loop for sub-codebook generation."""

    def __init__(self, model: nn.Module, device: str, max_seq_len: int) -> None:
        self._idx = 0
        self._scheduler = Scheduler(
            batch_planner=ARBatchPlanner(),
            resource_manager=ARResourceManager(max_count=1),
            iteration_controller=EosIterationController(
                eos_token_id=-1, max_length=max_seq_len
            ),
        )
        self._runner = ModelRunner(
            model=model,
            input_preparer=ARInputPreparer(),
            output_processor=AROutputProcessor(),
            device=device,
        )

    @staticmethod
    def _step_hook(data: ARRequestData, model_output: Any, token: int) -> None:
        del token
        gen_step = getattr(model_output, "generation_steps", None)
        if gen_step is not None:
            data.persistent_inputs["generation_steps"] = int(gen_step)

    @torch.inference_mode()
    def generate(
        self,
        prefill_embeds: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> torch.Tensor:
        request_id = f"sub-{self._idx}"
        self._idx += 1

        req = ARRequestData(
            input_ids=torch.zeros(1, dtype=torch.long, device=prefill_embeds.device),
            inputs_embeds=prefill_embeds,
            attention_mask=torch.ones(
                1,
                prefill_embeds.shape[1],
                dtype=torch.long,
                device=prefill_embeds.device,
            ),
            prefill_seq_len=prefill_embeds.shape[1],
            persistent_inputs={"generation_steps": 0},
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            step_hook=self._step_hook,
        )

        self._scheduler.add_request(request_id, req)
        max_iters = (max_new_tokens + 2) * 4
        for _ in range(max_iters):
            sched_out = self._scheduler.schedule()
            if sched_out is None:
                continue
            model_out = self._runner.execute(sched_out)
            finished = self._scheduler.update(sched_out, model_out)
            if finished:
                result: ARRequestData = finished[0].data
                self._scheduler.cleanup_finished(request_id)
                return torch.tensor(
                    [result.output_ids],
                    dtype=torch.long,
                    device=prefill_embeds.device,
                )

        self._scheduler.cleanup_finished(request_id)
        raise RuntimeError(f"Subtalker exceeded {max_iters} iterations ({request_id})")


class Qwen3TTSTalkerComponent(nn.Module):
    """GPU wrapper around the Talker for AR execution."""

    def __init__(self, model_id: str, device: str = "cuda", dtype: str | None = None):
        super().__init__()
        self.device_str = device
        torch_dtype = getattr(torch, dtype) if dtype else torch.bfloat16

        self.config = load_tts_config(model_id)
        self.talker = Qwen3TTSTalkerForConditionalGeneration(self.config.talker_config)
        load_module(
            self.talker, model_id, prefix=("talker.",), device=device, dtype=torch_dtype
        )
        self.talker.eval()

        tc = self.config.talker_config
        self._subtalker = _SubtalkerStepper(
            model=self.talker.code_predictor,
            device=device,
            max_seq_len=tc.num_code_groups + 8,
        )

    @torch.inference_mode()
    def generate_sub_sequences(
        self,
        past_hidden: torch.Tensor,
        token_id: int,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """Generate sub-codebook tokens via the sglang AR runtime."""
        token = torch.tensor([[token_id]], dtype=torch.long, device=past_hidden.device)
        last_hidden = self.talker.get_input_embeddings()(token)
        prefill = self.talker.code_predictor.small_to_mtp_projection(
            torch.cat((past_hidden, last_hidden), dim=1)
        )
        return self._subtalker.generate(
            prefill,
            max_new_tokens=self.config.talker_config.num_code_groups - 1,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    @torch.inference_mode()
    def forward(self, **kwargs: Any):
        return self.talker(**kwargs)


def make_tts_step_hook(talker: Qwen3TTSTalkerComponent):
    """Create a per-request step hook for state propagation and sub-code generation."""

    def hook(data: ARRequestData, model_output: Any, token: int) -> None:
        pi = data.persistent_inputs
        past_hidden = getattr(model_output, "past_hidden", None)

        if past_hidden is not None:
            pi["past_hidden"] = past_hidden

        gen_step = getattr(model_output, "generation_step", None)
        if gen_step is not None:
            pi["generation_step"] = int(gen_step)

        if past_hidden is not None:
            pi["sub_sequences"] = talker.generate_sub_sequences(
                past_hidden,
                int(token),
                temperature=float(pi.get("subtalker_temperature", 0.9)),
                top_k=int(pi.get("subtalker_top_k", 50)),
                top_p=float(pi.get("subtalker_top_p", 1.0)),
            )

        hs = getattr(model_output, "hidden_states", None)
        if isinstance(hs, tuple) and len(hs) == 2 and hs[1] is not None:
            data.extra_model_outputs.setdefault("all_codec_ids", []).append(
                hs[1].detach()
            )

    return hook


def build_talker_request(
    state: TTSPipelineState,
    *,
    talker: Qwen3TTSTalkerComponent,
    params: dict[str, Any],
    max_new_tokens: int = 4096,
) -> ARRequestData:
    """Build an ARRequestData from pipeline state for the Talker engine."""
    prompt = build_tts_prompt(
        talker=talker.talker,
        config=talker.config,
        input_ids=state.input_ids,
        speaker=state.speaker,
        language=state.language or "auto",
        non_streaming_mode=state.non_streaming_mode,
        instruct_ids=state.instruct_ids,
        ref_ids=state.ref_ids,
        voice_clone_prompt=state.voice_clone_prompt,
        speaker_embedding=state.speaker_embedding,
    )

    embeds = prompt["inputs_embeds"]
    return ARRequestData(
        input_ids=torch.zeros(1, dtype=torch.long),
        inputs_embeds=embeds,
        attention_mask=prompt["attention_mask"],
        prefill_seq_len=embeds.shape[1],
        persistent_inputs={
            "trailing_text_hidden": prompt["trailing_text_hidden"],
            "tts_pad_embed": prompt["tts_pad_embed"],
            "generation_step": -1,
            "subtalker_temperature": float(params.get("subtalker_temperature", 0.9)),
            "subtalker_top_k": int(params.get("subtalker_top_k", 50)),
            "subtalker_top_p": float(params.get("subtalker_top_p", 1.0)),
            "output_hidden_states": False,
        },
        temperature=float(params.get("temperature", 0.9)),
        top_k=int(params.get("top_k", 50)),
        top_p=float(params.get("top_p", 1.0)),
        repetition_penalty=float(params.get("repetition_penalty", 1.05)),
        suppress_tokens=prompt["suppress_tokens"],
        max_new_tokens=int(params.get("max_new_tokens", max_new_tokens)),
        eos_token_ids=[prompt["codec_eos_token_id"]],
        step_hook=make_tts_step_hook(talker),
    )


def apply_talker_result(
    state: TTSPipelineState,
    result: Any,
    codec_eos_token_id: int | None = None,
) -> None:
    """Extract speech codes from the completed ARRequestData."""
    if not isinstance(result, ARRequestData):
        state.speech_codes = None
        return

    codec_ids_list = result.extra_model_outputs.get("all_codec_ids", [])
    if not codec_ids_list:
        state.speech_codes = None
        return

    codec_ids = torch.cat(codec_ids_list, dim=0)

    eos_id = codec_eos_token_id or (
        result.eos_token_ids[0] if result.eos_token_ids else None
    )
    if eos_id is not None:
        eos_mask = codec_ids[:, 0] == eos_id
        if eos_mask.any():
            codec_ids = codec_ids[: eos_mask.nonzero()[0, 0]]

    state.speech_codes = codec_ids


def create_tts_frontend_executor(model_id: str) -> PreprocessingExecutor:
    frontend = Qwen3TTSFrontend(model_id=model_id)
    return PreprocessingExecutor(frontend)


def create_tts_speaker_encoder_executor(
    model_id: str,
    device: str = "cuda",
    dtype: str | None = None,
) -> EngineExecutor:
    model = Qwen3TTSSpeakerEncoderComponent(
        model_id=model_id, device=device, dtype=dtype
    )

    def request_builder(payload: StagePayload) -> EncoderRequestData:
        state = load_state(payload)
        vcp = state.voice_clone_prompt or {}
        ref_audio = vcp.get("ref_audio")
        if ref_audio is None:
            return EncoderRequestData(
                input_dict={"_skip": True, "_result": {"speaker_embedding": None}}
            )
        wav = _load_audio(ref_audio, sr=vcp.get("ref_sr"), target_sr=24000)
        return EncoderRequestData(input_dict={"audio": wav})

    def result_builder(payload: StagePayload, result: Any) -> StagePayload:
        state = load_state(payload)
        if isinstance(result, EncoderRequestData):
            d = result.output_dict or {}
            state.speaker_embedding = d.get("speaker_embedding") or result.embeddings
        elif isinstance(result, dict):
            state.speaker_embedding = result.get("speaker_embedding")
        else:
            state.speaker_embedding = result
        return store_state(payload, state)

    engine = create_encoder_engine(model, device=device, max_batch_size=1)
    return EngineExecutor(
        engine=engine, request_builder=request_builder, result_builder=result_builder
    )


def create_tts_talker_executor(
    model_id: str,
    device: str = "cuda",
    dtype: str | None = None,
    max_new_tokens: int = 4096,
) -> EngineExecutor:
    model = Qwen3TTSTalkerComponent(model_id=model_id, device=device, dtype=dtype)

    def request_builder(payload: StagePayload) -> ARRequestData:
        state = load_state(payload)
        params = dict(payload.request.params) if payload.request else {}
        talker_overrides = (params.get("stage_params") or {}).get("tts_talker")
        if talker_overrides:
            params.update(talker_overrides)
        return build_talker_request(
            state, talker=model, params=params, max_new_tokens=max_new_tokens
        )

    def result_builder(payload: StagePayload, result: Any) -> StagePayload:
        state = load_state(payload)
        apply_talker_result(state, result)
        return store_state(payload, state)

    def stream_builder(payload: StagePayload | None, item: Any) -> Any:
        return {"codec_token": item} if isinstance(item, int) else {"data": item}

    engine = create_ar_engine(
        model=model, max_seq_len=max_new_tokens + 512, device=device
    )
    return EngineExecutor(
        engine=engine,
        request_builder=request_builder,
        result_builder=result_builder,
        stream_builder=stream_builder,
    )


def create_tts_code2wav_executor(
    model_id: str,
    device: str = "cuda",
    dtype: str | None = None,
) -> EngineExecutor:
    model = Qwen3TTSCode2WavComponent(model_id=model_id, device=device, dtype=dtype)

    def request_builder(payload: StagePayload) -> EncoderRequestData:
        state = load_state(payload)
        if state.speech_codes is None:
            return EncoderRequestData(
                input_dict={
                    "_skip": True,
                    "_result": {"audio_waveform": [], "sample_rate": 24000},
                }
            )
        vcp = state.voice_clone_prompt or {}
        return EncoderRequestData(
            input_dict={
                "speech_codes": state.speech_codes,
                "ref_audio": vcp.get("ref_audio"),
                "ref_sr": int(vcp.get("ref_sr", 24000)),
                "xvectors": vcp.get("xvectors"),
                "ref_mels": vcp.get("ref_mels"),
            }
        )

    def result_builder(payload: StagePayload, result: Any) -> StagePayload:
        state = load_state(payload)
        d = (
            result.output_dict
            if isinstance(result, EncoderRequestData)
            else (result if isinstance(result, dict) else {})
        )
        wav = d.get("audio_waveform")
        state.audio_waveform = wav[0] if isinstance(wav, list) and wav else wav
        state.sample_rate = int(d.get("sample_rate", 24000))
        return store_state(payload, state)

    engine = create_encoder_engine(model, device=device, max_batch_size=1)
    return EngineExecutor(
        engine=engine, request_builder=request_builder, result_builder=result_builder
    )


def create_tts_decode_executor() -> PreprocessingExecutor:
    def decode(payload: StagePayload) -> StagePayload:
        state = load_state(payload)
        payload.data = {
            "type": "audio_final",
            "modality": "audio",
            "audio": state.audio_waveform,
            "sample_rate": state.sample_rate,
            "is_final": True,
        }
        return payload

    return PreprocessingExecutor(decode)


_MODULE = "sglang_omni.models.qwen3_tts.qwen3_tts"


def create_tts_pipeline_config(
    *,
    model_id: str,
    name: str = "qwen3_tts",
    frontend_device: str = "cpu",
    speaker_encoder_device: str = "cuda:0",
    talker_device: str = "cuda:0",
    code2wav_device: str = "cuda:0",
    dtype: str | None = None,
    relay_type: str = "shm",
    max_new_tokens: int = 4096,
    fused_stages: list[list[str]] | None = None,
) -> PipelineConfig:
    """Build a 5-stage TTS pipeline configuration."""

    def _relay(device: str) -> RelayConfig:
        return RelayConfig(type=relay_type, device=device)

    def _stage(
        name: str, factory: str, args: dict, next_fn: str, device: str
    ) -> StageConfig:
        return StageConfig(
            name=name,
            executor=ExecutorConfig(factory=f"{_MODULE}.{factory}", args=args),
            get_next=f"{_MODULE}.{next_fn}",
            relay=_relay(device),
        )

    return PipelineConfig(
        name=name,
        entry_stage=FRONTEND_STAGE,
        fused_stages=fused_stages or [],
        stages=[
            _stage(
                FRONTEND_STAGE,
                "create_tts_frontend_executor",
                {"model_id": model_id},
                "frontend_next",
                frontend_device,
            ),
            _stage(
                SPEAKER_ENCODER_STAGE,
                "create_tts_speaker_encoder_executor",
                {
                    "model_id": model_id,
                    "device": speaker_encoder_device,
                    "dtype": dtype,
                },
                "speaker_encoder_next",
                speaker_encoder_device,
            ),
            _stage(
                TALKER_STAGE,
                "create_tts_talker_executor",
                {
                    "model_id": model_id,
                    "device": talker_device,
                    "dtype": dtype,
                    "max_new_tokens": max_new_tokens,
                },
                "talker_next",
                talker_device,
            ),
            _stage(
                CODE2WAV_STAGE,
                "create_tts_code2wav_executor",
                {"model_id": model_id, "device": code2wav_device, "dtype": dtype},
                "code2wav_next",
                code2wav_device,
            ),
            _stage(
                DECODE_STAGE, "create_tts_decode_executor", {}, "decode_next", "cpu"
            ),
        ],
    )
