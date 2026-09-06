from __future__ import annotations

import json
from pathlib import Path

import torch
from einops import rearrange
from torch import nn
from transformers import AutoTokenizer

from sglang_omni.models.nemotron_voicechat.code2wav_stream import (
    NemotronCode2WavScheduler,
)
from sglang_omni.models.nemotron_voicechat.codec import RVQVAEDecoder
from sglang_omni.models.nemotron_voicechat.conformer import AudioPerception
from sglang_omni.models.nemotron_voicechat.engine_builder import (
    NemotronVoiceChatEngineBuilder,
    NemotronVoiceChatTalkerEngineBuilder,
)
from sglang_omni.models.nemotron_voicechat.payload_types import (
    OUTPUT_SAMPLE_RATE,
    NemotronVoiceChatState,
)
from sglang_omni.models.weight_loader import (
    load_module,
    load_weights_by_prefix,
    resolve_dtype,
    resolve_model_path,
)
from sglang_omni.preprocessing.transcription import resolve_audio_source
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.audio import load_audio
from sglang_omni.utils.audio_payload import audio_waveform_payload
from sglang_omni.utils.device import resolve_device_spec

PERCEPTION_PREFIX = "stt_model.perception."
SAMPLES_PER_FRAME = 1_280
INPUT_SAMPLE_RATE = 16_000


def _perception_config(model_path: str) -> dict:
    config_path = Path(resolve_model_path(model_path)) / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return config["model"]["stt"]["model"]["perception"]


def create_preprocessing_executor(model_path: str, **_):
    del model_path

    def preprocess(payload: StagePayload) -> StagePayload:
        # Channel 0, not a downmix: this model speaks the other side of the
        # conversation, so a two-party recording carries the agent on channel 1.
        channels = load_audio(
            resolve_audio_source(payload),
            source_name="VoiceChat",
            target_sample_rate=INPUT_SAMPLE_RATE,
            mono=False,
        )
        waveform = torch.as_tensor(channels[0], dtype=torch.float32)
        remainder = waveform.shape[-1] % SAMPLES_PER_FRAME
        if remainder:
            waveform = nn.functional.pad(waveform, (0, SAMPLES_PER_FRAME - remainder))

        state = NemotronVoiceChatState.from_dict(payload.data)
        state.waveform = waveform
        state.num_frames = waveform.shape[-1] // SAMPLES_PER_FRAME
        payload.data = state.to_dict()
        return payload

    return SimpleScheduler(preprocess)


def create_perception_executor(model_path: str, *, dtype=None, device=None):
    device = resolve_device_spec(device)
    module = AudioPerception(_perception_config(model_path))
    load_module(
        module,
        model_path,
        prefix=PERCEPTION_PREFIX,
        dtype=resolve_dtype(dtype),
        device=device,
        strict=True,
    )
    module.eval()
    parameter_dtype = module.proj.weight.dtype

    @torch.inference_mode()
    def encode(payload: StagePayload) -> StagePayload:
        state = NemotronVoiceChatState.from_dict(payload.data)
        waveform = state.waveform
        waveform_1S = (
            rearrange(waveform, "s -> 1 s") if waveform.ndim == 1 else waveform
        )

        frames = module(waveform_1S.to(device=device, dtype=parameter_dtype))
        assert frames.shape[1] == state.num_frames + 1, (
            f"Perception returned {frames.shape[1]} rows for {state.num_frames} "
            "frames of audio; expected one more than the frame count."
        )

        state.acoustic_frames = frames[0]
        payload.data = state.to_dict()
        return payload

    return SimpleScheduler(encode)


def create_thinker_executor(
    model_path, *, dtype=None, device=None, gpu_id=None, **overrides
):
    builder = NemotronVoiceChatEngineBuilder(max_running_requests=1)
    return builder.build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        dtype=dtype or "float32",
        server_args_overrides=overrides or None,
    )


def _speech_generation_config(model_path: str) -> dict:
    config_path = Path(resolve_model_path(model_path)) / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return config["model"]["speech_generation"]["model"]


def create_talker_executor(
    model_path,
    *,
    dtype=None,
    device=None,
    gpu_id=None,
    context_length=None,
    **overrides,
):
    builder = NemotronVoiceChatTalkerEngineBuilder(
        max_running_requests=1,
        context_length=context_length,
    )
    return builder.build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        dtype=dtype or "bfloat16",
        server_args_overrides=overrides or None,
    )


def create_code2wav_executor(model_path, *, dtype=None, device=None):
    device = resolve_device_spec(device)
    generation = _speech_generation_config(model_path)
    weights = load_weights_by_prefix(model_path, prefix=("tts_model.audio_codec.",))
    markers = {
        name: load_weights_by_prefix(model_path, prefix=f"tts_model.{name}")[""]
        for name in ("_control_codes", "codec_silence_tokens")
    }
    decoder_weights = {
        key: value
        for key, value in weights.items()
        if key.startswith("decoder.") or key.startswith("prvq.mus_list.")
    }
    decoder = RVQVAEDecoder(generation["codec_config"])
    decoder.load_state_dict(decoder_weights, strict=True)
    # What counts as a control code, and the silence that stands in for it,
    # are the checkpoint's to say.
    decoder.control_codes.copy_(markers["_control_codes"].reshape(-1))
    decoder.silence_codes.copy_(markers["codec_silence_tokens"].reshape(-1))
    decoder = decoder.to(device=device, dtype=resolve_dtype(dtype)).eval()

    @torch.inference_mode()
    def decode(payload: StagePayload) -> StagePayload:
        """Render a whole code stack at once, for a reply that never streamed."""
        state = NemotronVoiceChatState.from_dict(payload.data)
        waveform = decoder(state.codes.to(device)).float().cpu()
        payload.data = audio_waveform_payload(
            waveform,
            sample_rate=OUTPUT_SAMPLE_RATE,
            modality="audio",
            source_hint="NemotronVoiceChat",
        )
        return payload

    return NemotronCode2WavScheduler(decoder, device, compute_fn=decode)


def create_decode_executor(model_path, **_):
    """Turn the thinker's frame-locked token timeline into the reply text.

    Most frames carry a marker rather than a word — the model is listening, or
    punctuating a turn — so only the ids that spell something are detokenized.
    """
    speech = _speech_generation_config(model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        speech["tts_config"]["cas_config"]["pretrained_tokenizer_name"]
    )
    silent_ids = set(tokenizer.all_special_ids) | {
        tokenizer.convert_tokens_to_ids(token) for token in ("<s>", "</s>")
    }

    def detokenize(payload: StagePayload) -> StagePayload:
        state = NemotronVoiceChatState.from_dict(payload.data)
        spoken = [int(i) for i in state.text_ids if int(i) not in silent_ids]
        payload.data = {"text": tokenizer.decode(spoken) if spoken else ""}
        return payload

    return SimpleScheduler(detokenize)
