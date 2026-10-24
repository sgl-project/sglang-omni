# SPDX-License-Identifier: Apache-2.0
"""Stage factories for Ming 3 text-to-speech."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Literal, TypedDict

import onnxruntime
import torch
from pydantic import BaseModel, ConfigDict, JsonValue
from safetensors import safe_open
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.models.registry import ModelRegistry
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.utils.hf_transformers.tokenizer import get_tokenizer
from transformers import AutoConfig, PreTrainedTokenizerBase

from sglang_omni.models.ming3_tts.dense_prompt import (
    AUDIO_PATCH_TOKEN_ID,
    DENSE_MAXIMUM_DECODE_STEPS,
    DENSE_MINIMUM_DECODE_STEPS,
    DENSE_PATCH_SIZE,
    DENSE_SAMPLING_TEMPERATURE,
    DENSE_STOP_THRESHOLD,
    STOP_TOKEN_ID,
    build_dense_speech_prompt,
    validate_dense_tokenizer,
)
from sglang_omni.models.ming3_tts.model_runner import (
    Ming3TTSModelRunner,
    Ming3TTSRequestData,
    StopPolicy,
)
from sglang_omni.models.ming3_tts.sglang_model import (
    VOCODER_WEIGHT_PREFIX,
    Ming3DenseTTSConfig,
    Ming3DenseTTSModel,
    Ming3TTSConfig,
    Ming3TTSModel,
)
from sglang_omni.modules.ming_audio.vae.configuration_audio_vae import AudioVAEconfig
from sglang_omni.modules.ming_audio.vae.modeling_audio_vae import AudioVAE
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.bootstrap import create_sglang_infrastructure
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.scheduling.sglang_backend.server_args_builder import (
    build_sglang_server_args,
)
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.audio import load_audio
from sglang_omni.utils.audio_features import cached_fbank
from sglang_omni.utils.audio_payload import audio_waveform_payload
from sglang_omni.utils.checkpoint import resolve_checkpoint
from sglang_omni.utils.device import resolve_concrete_device

INSTRUCTION_PREFIX = (
    "<role>HUMAN</role>Please generate speech based on the following description."
    "\n  speaker_1:"
)
TEXT_PREFIX = "\n Text input:\n"
ASSISTANT_PREFIX = "<role>ASSISTANT</role>"
SPEAKER_ENCODER_FILE = "campplus.onnx"
SPEAKER_SAMPLE_RATE = 16000
SPEAKER_MEL_BINS = 80
SPEAKER_ENCODER_THREADS = 2


class SpeechConfig(BaseModel):
    """The tts_config subset this pipeline implements; other contracts fail at load."""

    model_config = ConfigDict(extra="ignore", frozen=True)
    prompt_format: Literal["legacy_antmmf"]
    with_rml: Literal[True]
    conditioning_mode: Literal["speaker_only"]
    decode_alignment: Literal["old_shift"]
    patch_size: Literal[1]
    history_size: Literal[1]
    latent_dim: int
    speaker_embedding_dim: int
    sampling_temperature: float
    stop_threshold: float
    minimum_decode_steps: int
    maximum_decode_steps: int
    audio_patch_token_id: int
    audio_start_token_id: int
    audio_end_token_id: int
    speaker_start_token_id: int
    speaker_end_token_id: int


class SpeechPrompt(TypedDict):
    input_ids: torch.Tensor
    speaker_position: int
    speaker_embedding: torch.Tensor
    zero_speaker: bool
    prompt_latents: torch.Tensor | None
    latent_position: int


class DenseSpeechInputs(TypedDict):
    text: str
    speaker_embedding: torch.Tensor
    zero_speaker: bool
    reference_waveform: torch.Tensor | None
    reference_text: str | None


def load_checkpoint_config(checkpoint: Path) -> dict[str, JsonValue]:
    return json.loads((checkpoint / "config.json").read_text())


def is_dense_checkpoint(configuration: dict[str, JsonValue]) -> bool:
    return configuration.get("model_type") == "dense"


def audio_section(configuration: dict[str, JsonValue]) -> dict[str, JsonValue]:
    if is_dense_checkpoint(configuration):
        return configuration["audio_tokenizer_config"]
    else:
        return configuration["audio_config"]


def load_audio_vae(checkpoint: Path, device: torch.device) -> tuple[AudioVAE, int]:
    """Load the AudioVAE. Tiny checkpoints are sharded; 0.5B is one file."""
    vae_configuration = audio_section(load_checkpoint_config(checkpoint))
    for codec_section in ("enc_kwargs", "dec_kwargs"):
        vae_configuration[codec_section]["backbone"]["_attn_implementation"] = "sdpa"
    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.is_file():
        weight_index = json.loads(index_path.read_text())
        shard_names = {
            shard_name
            for weight_name, shard_name in weight_index["weight_map"].items()
            if weight_name.startswith(VOCODER_WEIGHT_PREFIX)
        }
    else:
        shard_names = {"model.safetensors"}
    vocoder_weights: dict[str, torch.Tensor] = {}
    for shard_name in sorted(shard_names):
        with safe_open(str(checkpoint / shard_name), framework="pt") as shard:
            for weight_name in shard.keys():
                if weight_name.startswith(VOCODER_WEIGHT_PREFIX):
                    vocoder_weights[weight_name.removeprefix(VOCODER_WEIGHT_PREFIX)] = (
                        shard.get_tensor(weight_name)
                    )
                else:
                    pass
    vae = AudioVAE(AudioVAEconfig(**vae_configuration)).eval()
    vae.load_state_dict(vocoder_weights)
    vae.to(device=device, dtype=torch.bfloat16)
    return vae, int(vae_configuration["sample_rate"])


def encode_reference_latents(
    vae: AudioVAE, waveform: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Pad a prompt on the left and return its VAE latents, shape [steps, dim]."""
    align = (
        vae.encoder.hop_size * max(1, int(vae.encoder.patch_size)) * DENSE_PATCH_SIZE
    )
    samples = waveform.reshape(1, -1)
    remainder = int(samples.shape[-1] % align)
    if remainder:
        padded = samples.new_zeros((1, int(samples.shape[-1]) + align - remainder))
        padded[:, -samples.shape[-1] :] = samples
        samples = padded
    else:
        pass
    samples = samples.to(device=device, dtype=torch.bfloat16)
    length = torch.tensor([samples.shape[-1]], dtype=torch.long, device=device)
    latents, _frame_count = vae.encode_latent(samples, length)
    return latents[0]


def materialize_dense_prompt(
    tokenizer: PreTrainedTokenizerBase,
    prompt_vae: AudioVAE,
    device: torch.device,
    inputs: DenseSpeechInputs,
    seed: int | None,
) -> SpeechPrompt:
    """Build the dense token ids after encoding a reference, when one is present."""
    reference_text = inputs["reference_text"]
    waveform = inputs["reference_waveform"]
    if waveform is not None and reference_text:
        if seed is not None:
            torch.manual_seed(seed)
        else:
            pass
        prompt_latents = encode_reference_latents(prompt_vae, waveform, device).cpu()
        latent_steps = int(prompt_latents.shape[0])
    else:
        prompt_latents = None
        latent_steps = 0
    input_ids, speaker_position, latent_position = build_dense_speech_prompt(
        tokenizer,
        inputs["text"],
        prompt_text=reference_text,
        prompt_latent_steps=latent_steps,
        include_speaker=not inputs["zero_speaker"],
    )
    return SpeechPrompt(
        input_ids=torch.tensor(input_ids, dtype=torch.long),
        speaker_position=speaker_position,
        speaker_embedding=inputs["speaker_embedding"],
        zero_speaker=inputs["zero_speaker"],
        prompt_latents=prompt_latents,
        latent_position=latent_position,
    )


def build_speech_prompt(
    tokenizer: PreTrainedTokenizerBase, speech_config: SpeechConfig, text: str
) -> tuple[list[int], int]:
    """Return the prompt token ids and the index the speaker embedding replaces."""

    def encode(segment: str) -> list[int]:
        return tokenizer.encode(segment, add_special_tokens=False)

    prefix_ids = encode(INSTRUCTION_PREFIX) + [speech_config.speaker_start_token_id]
    input_ids = (
        prefix_ids
        + [speech_config.audio_patch_token_id, speech_config.speaker_end_token_id]
        + encode(TEXT_PREFIX)
        + encode(text)
        + encode(ASSISTANT_PREFIX)
        + [speech_config.audio_start_token_id]
    )
    return input_ids, len(prefix_ids)


def create_preprocessing_executor(model_path: str) -> SimpleScheduler:
    checkpoint = Path(resolve_checkpoint(model_path))
    configuration = load_checkpoint_config(checkpoint)
    dense = is_dense_checkpoint(configuration)
    speech_config = (
        None if dense else SpeechConfig.model_validate(configuration["tts_config"])
    )
    sample_rate = int(audio_section(configuration)["sample_rate"])
    speaker_embedding_dim = 192 if dense else speech_config.speaker_embedding_dim
    tokenizer = get_tokenizer(str(checkpoint), trust_remote_code=False)
    if dense:
        validate_dense_tokenizer(tokenizer)
    else:
        pass
    session_options = onnxruntime.SessionOptions()
    session_options.intra_op_num_threads = SPEAKER_ENCODER_THREADS
    speaker_encoder = onnxruntime.InferenceSession(
        str(checkpoint / SPEAKER_ENCODER_FILE),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    speaker_input_name = speaker_encoder.get_inputs()[0].name

    def preprocess(payload: StagePayload) -> StagePayload:
        inputs = payload.request.inputs
        if isinstance(inputs, str):
            text, references = inputs, []
        else:
            text, references = inputs["text"], inputs.get("references", [])
        if not text.strip():
            raise ValueError("Ming 3 TTS requires non-empty input text")
        elif len(references) > 1:
            raise ValueError("Ming 3 TTS accepts at most one reference audio")
        else:
            pass
        if not references:
            speaker_embedding = torch.zeros(speaker_embedding_dim)
            reference_waveform = None
            reference_text = None
        else:
            reference = references[0]
            if "data" in reference:
                reference_source = base64.b64decode(reference["data"], validate=True)
            else:
                reference_source = reference["audio_path"]
            waveform = torch.from_numpy(
                load_audio(reference_source, target_sample_rate=SPEAKER_SAMPLE_RATE)
            )
            filterbank = cached_fbank(
                waveform[None],
                num_mel_bins=SPEAKER_MEL_BINS,
                sample_frequency=SPEAKER_SAMPLE_RATE,
            )
            filterbank = filterbank - filterbank.mean(dim=0, keepdim=True)
            speaker_vector = speaker_encoder.run(
                None, {speaker_input_name: filterbank[None].numpy()}
            )[0]
            speaker_embedding = torch.from_numpy(speaker_vector).reshape(-1).float()
            reference_text = reference.get("text") or None
            if dense and reference_text:
                reference_waveform = torch.from_numpy(
                    load_audio(reference_source, target_sample_rate=sample_rate)
                )
            else:
                reference_waveform = None
        if dense:
            payload.data = DenseSpeechInputs(
                text=text,
                speaker_embedding=speaker_embedding,
                zero_speaker=not references,
                reference_waveform=reference_waveform,
                reference_text=reference_text,
            )
        else:
            input_ids, speaker_position = build_speech_prompt(
                tokenizer, speech_config, text
            )
            payload.data = SpeechPrompt(
                input_ids=torch.tensor(input_ids, dtype=torch.long),
                speaker_position=speaker_position,
                speaker_embedding=speaker_embedding,
                zero_speaker=False,
                prompt_latents=None,
                latent_position=-1,
            )
        return payload

    return SimpleScheduler(preprocess)


def create_tts_engine_executor(
    model_path: str,
    *,
    max_seq_len: int,
    enable_acoustic_graph: bool,
    device: str | None = None,
    gpu_id: int | None = None,
    tp_rank: int = 0,
    tp_size: int = 1,
    nccl_port: int | None = None,
    server_args_overrides: dict[str, JsonValue] | None = None,
    total_gpu_memory_fraction: float | None = None,
) -> OmniScheduler:
    resolved_device = resolve_concrete_device(device, gpu_id)
    if tp_size != 1:
        raise ValueError(f"Ming 3 TTS supports tp_size=1, got {tp_size}")
    else:
        pass
    checkpoint = resolve_checkpoint(model_path)
    configuration = load_checkpoint_config(Path(checkpoint))
    dense = is_dense_checkpoint(configuration)
    if dense:
        AutoConfig.register(
            Ming3DenseTTSConfig.model_type, Ming3DenseTTSConfig, exist_ok=True
        )
        ModelRegistry.models["Qwen2ForCausalLM"] = Ming3DenseTTSModel
        backbone_architecture = "Qwen2ForCausalLM"
        encoder_config = configuration["audio_tokenizer_config"]["enc_kwargs"]
        latent_dim = int(encoder_config["latent_dim"])
        sampling_temperature = DENSE_SAMPLING_TEMPERATURE
        stop_threshold = DENSE_STOP_THRESHOLD
        minimum_decode_steps = DENSE_MINIMUM_DECODE_STEPS
        maximum_decode_steps_limit = DENSE_MAXIMUM_DECODE_STEPS
        audio_patch_token_id = AUDIO_PATCH_TOKEN_ID
        audio_end_token_id = STOP_TOKEN_ID
        prompt_vae, _sample_rate = load_audio_vae(Path(checkpoint), resolved_device)
    else:
        speech_config = SpeechConfig.model_validate(configuration["tts_config"])
        AutoConfig.register(Ming3TTSConfig.model_type, Ming3TTSConfig, exist_ok=True)
        ModelRegistry.models["BailingMoeV3ForCausalLM"] = Ming3TTSModel
        backbone_architecture = "BailingMoeV3ForCausalLM"
        latent_dim = speech_config.latent_dim
        sampling_temperature = speech_config.sampling_temperature
        stop_threshold = speech_config.stop_threshold
        minimum_decode_steps = speech_config.minimum_decode_steps
        maximum_decode_steps_limit = speech_config.maximum_decode_steps
        audio_patch_token_id = speech_config.audio_patch_token_id
        audio_end_token_id = speech_config.audio_end_token_id
        prompt_vae = None
    overrides = dict(server_args_overrides or {})
    overrides.update(
        tp_size=tp_size,
        trust_remote_code=False,
        json_model_override_args=json.dumps({"architectures": [backbone_architecture]}),
        disable_radix_cache=True,
        chunked_prefill_size=-1,
    )
    if dense:
        # The checkpoint is bfloat16. SGLang otherwise downcasts a missing
        # dtype to float16, and this backbone overflows to NaN.
        overrides["dtype"] = "bfloat16"
    else:
        pass
    server_args = build_sglang_server_args(
        checkpoint, context_length=max_seq_len, **overrides
    )
    worker, tree_cache, request_pool, token_pool, model_config = (
        create_sglang_infrastructure(
            server_args,
            resolved_device.index or 0,
            tp_rank=tp_rank,
            nccl_port=nccl_port,
            total_gpu_memory_fraction=total_gpu_memory_fraction,
        )
    )
    tokenizer = get_tokenizer(checkpoint, trust_remote_code=False)

    def build_request(payload: StagePayload) -> Ming3TTSRequestData:
        parameters = payload.request.params
        seed = parameters.get("seed")
        if dense:
            prompt = materialize_dense_prompt(
                tokenizer, prompt_vae, resolved_device, payload.data, seed
            )
        else:
            prompt = payload.data
        requested_steps = parameters.get("max_new_tokens")
        if requested_steps is None:
            max_decode_steps = maximum_decode_steps_limit
        else:
            max_decode_steps = min(requested_steps, maximum_decode_steps_limit)
        if dense:
            room = max_seq_len - len(prompt["input_ids"])
            if room < 1:
                raise ValueError("Ming 3 TTS prompt exceeds the context length")
            else:
                max_decode_steps = min(max_decode_steps, room)
        else:
            pass
        temperature = parameters.get("temperature")
        if temperature is None:
            temperature = sampling_temperature
        else:
            pass
        noise_generator = torch.Generator(device=resolved_device)
        if seed is None:
            noise_generator.seed()
        else:
            noise_generator.manual_seed(seed)
        sampling_params = SamplingParams(
            temperature=0.0,
            max_new_tokens=max_decode_steps,
            stop_token_ids=[audio_end_token_id],
        )
        sampling_params.normalize(tokenizer)
        sampling_params.verify(model_config.vocab_size)
        request = Req(
            rid=payload.request_id,
            origin_input_text="",
            origin_input_ids=prompt["input_ids"].tolist(),
            sampling_params=sampling_params,
            vocab_size=model_config.vocab_size,
            eos_token_ids={audio_end_token_id},
        )
        request.tokenizer = tokenizer
        if prompt["prompt_latents"] is not None:
            latent_history = prompt["prompt_latents"][-1:].to(resolved_device)
        else:
            latent_history = torch.zeros((1, latent_dim), device=resolved_device)
        return Ming3TTSRequestData(
            input_ids=prompt["input_ids"],
            req=request,
            max_new_tokens=max_decode_steps,
            temperature=0.0,
            output_ids=request.output_ids,
            stage_payload=payload,
            speaker_embedding=prompt["speaker_embedding"],
            speaker_position=prompt["speaker_position"],
            sampling_temperature=float(temperature),
            noise_generator=noise_generator,
            latent_history=latent_history,
            zero_speaker=prompt["zero_speaker"],
            prompt_latents=prompt["prompt_latents"],
            latent_position=prompt["latent_position"],
        )

    def build_result(request: Ming3TTSRequestData) -> StagePayload:
        payload = request.stage_payload
        payload.data = {
            "latents": torch.stack(request.latents).float().cpu(),
            "usage": {
                "prompt_tokens": len(request.input_ids),
                "completion_tokens": len(request.output_ids),
            },
        }
        return payload

    stop_policy = StopPolicy(
        stop_probability_threshold=stop_threshold,
        minimum_decode_steps=minimum_decode_steps,
        audio_patch_token_id=audio_patch_token_id,
        audio_end_token_id=audio_end_token_id,
    )
    if enable_acoustic_graph:
        acoustic_graph_max_batch_size = server_args.max_running_requests
    else:
        acoustic_graph_max_batch_size = None
    model_runner = Ming3TTSModelRunner(
        worker, SGLangOutputProcessor(), stop_policy, acoustic_graph_max_batch_size
    )
    return OmniScheduler(
        tp_worker=worker,
        tree_cache=tree_cache,
        req_to_token_pool=request_pool,
        token_to_kv_pool_allocator=token_pool,
        server_args=server_args,
        model_config=model_config,
        model_runner=model_runner,
        request_builder=build_request,
        result_adapter=build_result,
        enable_async_decode=False,
    )


def create_vocoder_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
) -> SimpleScheduler:
    resolved_device = resolve_concrete_device(device, gpu_id)
    checkpoint = Path(resolve_checkpoint(model_path))
    vae, sample_rate = load_audio_vae(checkpoint, resolved_device)

    @torch.inference_mode()
    def decode(payload: StagePayload) -> StagePayload:
        latents = payload.data["latents"].to(resolved_device, dtype=torch.bfloat16)
        waveform, _, _ = vae.decode(latents.reshape(1, -1, latents.shape[-1]))
        usage = payload.data["usage"]
        payload.data = audio_waveform_payload(
            waveform.float().cpu(),
            sample_rate=sample_rate,
            modality="audio",
            source_hint="Ming-3.0-TTS",
        )
        payload.data["usage"] = usage
        return payload

    return SimpleScheduler(decode)
