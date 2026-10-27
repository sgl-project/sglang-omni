# SPDX-License-Identifier: Apache-2.0
"""Stage factories for Ming 3 text-to-speech."""

from __future__ import annotations

import base64
import copy
import json
import logging
import queue
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Protocol, TypedDict
from urllib.parse import unquote, urlparse

import onnxruntime
import torch
import xxhash
from pydantic import JsonValue
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.models.registry import ModelRegistry
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.utils.hf_transformers.tokenizer import get_tokenizer
from transformers import AutoConfig, PreTrainedTokenizerBase

from sglang_omni.models.ming3_tts.checkpoint import (
    SpeechConfig,
    audio_section,
    is_dense_checkpoint,
    load_audio_vae,
    load_checkpoint_config,
    load_prefixed_weights,
)
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
    build_stream_output,
)
from sglang_omni.models.ming3_tts.sglang_model import (
    AGGREGATOR_WEIGHT_PREFIX,
    Ming3DenseTTSConfig,
    Ming3DenseTTSModel,
    Ming3TTSConfig,
    Ming3TTSModel,
    build_audio_aggregator,
)
from sglang_omni.models.ming3_tts.vocoder import create_vocoder_executor
from sglang_omni.modules.ming_audio.flow.aggregator import Aggregator
from sglang_omni.modules.ming_audio.vae.modeling_audio_vae import AudioVAE
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.bootstrap import create_sglang_infrastructure
from sglang_omni.scheduling.generation_batch_policy import (
    CudaGraphBackend,
    operator_selected_prefill_backend,
)
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.reference_encoder import (
    KeyedReferenceEncodeHook,
    ReferenceEncodeService,
)
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.scheduling.sglang_backend.server_args_builder import (
    build_sglang_server_args,
)
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.audio import decode_audio_data_uri, load_audio
from sglang_omni.utils.audio_features import cached_fbank
from sglang_omni.utils.checkpoint import resolve_checkpoint
from sglang_omni.utils.device import resolve_concrete_device

logger = logging.getLogger(__name__)

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


class SpeechPrompt(TypedDict):
    input_ids: torch.Tensor
    speaker_position: int
    speaker_embedding: torch.Tensor
    zero_speaker: bool
    prompt_latents: torch.Tensor | None
    prompt_rows: torch.Tensor | None
    latent_position: int


class PromptLatentInput(TypedDict):
    reference_key: str
    reference_audio: bytes
    seed: int


class PromptConditioning(TypedDict):
    """Reference VAE latents and their aggregator rows, both [steps, dim] on CPU."""

    prompt_latents: torch.Tensor
    prompt_rows: torch.Tensor


class SpeakerEncoder(Protocol):
    def __call__(self, reference_audio: bytes | str) -> torch.Tensor:
        pass


class PromptLatentEncoder(Protocol):
    def __call__(
        self, reference_audio: bytes | str, seed: int | None
    ) -> PromptConditioning:
        pass


class SpeakerEmbeddingHook(
    KeyedReferenceEncodeHook[bytes | str, torch.Tensor, torch.Tensor, bytes | str]
):
    """Share one reference's speaker embedding across requests by audio content."""

    model_revision = ""
    encoder_id = "campplus"
    artifact_kind = "speaker_embedding"

    def __init__(self, encoder: SpeakerEncoder, *, model_identity: str) -> None:
        self.encoder = encoder
        self.model_id = model_identity
        self.encoder_config_hash = ""

    def input_key(self, item: bytes | str) -> str | None:
        if isinstance(item, bytes):
            return xxhash.xxh3_128_hexdigest(item)
        else:
            return None

    def encode_one(self, item: bytes | str) -> torch.Tensor:
        return self.encoder(item)

    def store_artifact(self, artifact: torch.Tensor) -> torch.Tensor:
        return artifact

    def load_artifact(self, stored: torch.Tensor) -> torch.Tensor:
        return stored


class PromptLatentHook(
    KeyedReferenceEncodeHook[
        PromptLatentInput, PromptConditioning, PromptConditioning, PromptLatentInput
    ]
):
    """Share a reference's VAE prompt latents across requests with the same seed."""

    model_revision = ""
    encoder_id = "audio_vae_prompt_latents"
    artifact_kind = "prompt_latents"

    def __init__(self, encoder: PromptLatentEncoder, *, model_identity: str) -> None:
        self.encoder = encoder
        self.model_id = model_identity
        self.encoder_config_hash = ""

    def input_key(self, item: PromptLatentInput) -> str:
        return item["reference_key"]

    def options_key(self, item: PromptLatentInput) -> str:
        return str(item["seed"])

    def encode_one(self, item: PromptLatentInput) -> PromptConditioning:
        return self.encoder(item["reference_audio"], item["seed"])

    def store_artifact(self, artifact: PromptConditioning) -> PromptConditioning:
        return artifact

    def load_artifact(self, stored: PromptConditioning) -> PromptConditioning:
        return stored


class DenseSpeechInputs(TypedDict):
    text: str
    speaker_embedding: torch.Tensor
    zero_speaker: bool
    prompt: PromptConditioning | None
    reference_text: str | None


def load_prompt_aggregator(
    checkpoint: Path, configuration: dict[str, JsonValue], device: torch.device
) -> Aggregator:
    """Load the dense aggregator alone, built like the engine builds its copy."""
    latent_dim = int(
        configuration["audio_tokenizer_config"]["enc_kwargs"]["latent_dim"]
    )
    with set_default_torch_dtype(torch.bfloat16), device:
        aggregator = build_audio_aggregator(
            int(configuration["llm_config"]["hidden_size"]),
            latent_dim,
            configuration["aggregator_config"],
        )
    aggregator.load_state_dict(
        load_prefixed_weights(checkpoint, AGGREGATOR_WEIGHT_PREFIX)
    )
    return aggregator.eval()


def encode_reference_latents(
    vae: AudioVAE,
    waveform: torch.Tensor,
    device: torch.device,
    generator: torch.Generator | None,
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
    latents, _frame_count = vae.encode_latent(samples, length, generator=generator)
    return latents[0]


def materialize_dense_prompt(
    tokenizer: PreTrainedTokenizerBase, inputs: DenseSpeechInputs
) -> SpeechPrompt:
    """Build the dense token ids around the reference's prompt latents, if any."""
    prompt = inputs["prompt"]
    latent_steps = 0 if prompt is None else int(prompt["prompt_latents"].shape[0])
    input_ids, speaker_position, latent_position = build_dense_speech_prompt(
        tokenizer,
        inputs["text"],
        prompt_text=inputs["reference_text"],
        prompt_latent_steps=latent_steps,
        include_speaker=not inputs["zero_speaker"],
    )
    return SpeechPrompt(
        input_ids=torch.tensor(input_ids, dtype=torch.long),
        speaker_position=speaker_position,
        speaker_embedding=inputs["speaker_embedding"],
        zero_speaker=inputs["zero_speaker"],
        prompt_latents=None if prompt is None else prompt["prompt_latents"],
        prompt_rows=None if prompt is None else prompt["prompt_rows"],
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


def read_reference_audio(reference: dict[str, JsonValue]) -> bytes | str:
    """Return the reference bytes; a remote URL stays for load_audio to fetch."""
    if "data" in reference:
        return base64.b64decode(reference["data"], validate=True)
    else:
        source = reference["audio_path"]
    decoded = decode_audio_data_uri(source)
    if decoded is not None:
        return decoded
    elif source.startswith(("http://", "https://")):
        return source
    elif source.startswith("file://"):
        return Path(unquote(urlparse(source).path)).read_bytes()
    else:
        return Path(source).read_bytes()


def create_preprocessing_executor(
    model_path: str,
    *,
    max_concurrency: int = 1,
    reference_cache_entries: int = 0,
    reference_cache_bytes: int = 64 * 1024 * 1024,
    max_reference_seconds: float | None = None,
    device: str | None = None,
    gpu_id: int | None = None,
) -> SimpleScheduler:
    if (
        reference_cache_entries < 0
        or reference_cache_bytes < 1
        or (max_reference_seconds is not None and max_reference_seconds <= 0)
    ):
        raise ValueError("Ming 3 TTS reference cache and duration limits are invalid")
    elif max_concurrency < 1:
        raise ValueError("Ming 3 TTS preprocessing max_concurrency must be positive")
    else:
        pass
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
        preprocessing_device = "cpu" if device is None and gpu_id is None else device
        resolved_device = resolve_concrete_device(preprocessing_device, gpu_id)
        prompt_vae, _sample_rate = load_audio_vae(
            checkpoint, resolved_device, component="encoder"
        )
        prompt_aggregator = load_prompt_aggregator(
            checkpoint, configuration, resolved_device
        )
        # note (rsi): each new reference length builds a cuDNN attention plan
        # for tens of milliseconds; memory-efficient attention has no such cost.
        torch.backends.cuda.enable_cudnn_sdp(False)
    else:
        resolved_device = None
        prompt_vae = None
        prompt_aggregator = None
    session_options = onnxruntime.SessionOptions()
    session_options.intra_op_num_threads = SPEAKER_ENCODER_THREADS
    # note (rsi): one session per worker keeps every run on its own two threads.
    speaker_sessions = [
        onnxruntime.InferenceSession(
            str(checkpoint / SPEAKER_ENCODER_FILE),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        for _ in range(max_concurrency)
    ]
    speaker_input_name = speaker_sessions[0].get_inputs()[0].name
    speaker_encoders: queue.SimpleQueue[onnxruntime.InferenceSession] = (
        queue.SimpleQueue()
    )
    for speaker_session in speaker_sessions:
        speaker_encoders.put(speaker_session)

    def encode_speaker(reference_audio: bytes | str) -> torch.Tensor:
        waveform = torch.from_numpy(
            load_audio(reference_audio, target_sample_rate=SPEAKER_SAMPLE_RATE)
        )
        if (
            max_reference_seconds is not None
            and waveform.numel() > max_reference_seconds * SPEAKER_SAMPLE_RATE
        ):
            raise ValueError(
                f"Ming 3 TTS reference exceeds {max_reference_seconds:g} seconds"
            )
        else:
            pass
        filterbank = cached_fbank(
            waveform[None],
            num_mel_bins=SPEAKER_MEL_BINS,
            sample_frequency=SPEAKER_SAMPLE_RATE,
        )
        filterbank = filterbank - filterbank.mean(dim=0, keepdim=True)
        speaker_encoder = speaker_encoders.get()
        try:
            speaker_vector = speaker_encoder.run(
                None, {speaker_input_name: filterbank[None].numpy()}
            )[0]
        finally:
            speaker_encoders.put(speaker_encoder)
        return torch.from_numpy(speaker_vector).reshape(-1).float()

    @torch.inference_mode()
    def encode_prompt_latents(
        reference_audio: bytes | str, seed: int | None
    ) -> PromptConditioning:
        prompt_waveform = torch.from_numpy(
            load_audio(reference_audio, target_sample_rate=sample_rate)
        )
        if (
            max_reference_seconds is not None
            and prompt_waveform.numel() > max_reference_seconds * sample_rate
        ):
            raise ValueError(
                f"Ming 3 TTS reference exceeds {max_reference_seconds:g} seconds"
            )
        else:
            pass
        # note (rsi): a private generator seeded like torch.manual_seed draws the
        # same posterior noise without touching the process-wide RNG.
        if seed is None:
            generator = None
        else:
            generator = torch.Generator(device=resolved_device).manual_seed(seed)
        prompt_latents = encode_reference_latents(
            prompt_vae, prompt_waveform, resolved_device, generator
        )
        prompt_rows = prompt_aggregator(
            prompt_latents.reshape(-1, 1, prompt_latents.shape[-1])
        ).reshape(prompt_latents.shape[0], -1)
        return PromptConditioning(
            prompt_latents=prompt_latents.cpu(), prompt_rows=prompt_rows.cpu()
        )

    speaker_service = ReferenceEncodeService(
        SpeakerEmbeddingHook(encode_speaker, model_identity=str(checkpoint)),
        max_items=reference_cache_entries or 1,
        max_bytes=reference_cache_bytes,
        log_prefix="Ming 3 TTS speaker cache",
    )
    latent_service = ReferenceEncodeService(
        PromptLatentHook(encode_prompt_latents, model_identity=str(checkpoint)),
        max_items=reference_cache_entries or 1,
        max_bytes=reference_cache_bytes,
        log_prefix="Ming 3 TTS prompt latent cache",
    )
    # note (rsi): the CPU speaker encoder overlaps the GPU prompt-latent encode.
    speaker_executor = ThreadPoolExecutor(
        max_workers=max_concurrency, thread_name_prefix="ming3-speaker"
    )

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
        prompt = None
        if not references:
            speaker_embedding = torch.zeros(speaker_embedding_dim)
            reference_text = None
        else:
            reference = references[0]
            reference_audio = read_reference_audio(reference)
            reference_text = reference.get("text") or None
            if dense and reference_text:
                speaker_job = speaker_executor.submit(
                    (
                        speaker_service.get_or_encode
                        if reference_cache_entries
                        else encode_speaker
                    ),
                    reference_audio,
                )
                seed = payload.request.params.get("seed")
                if (
                    not reference_cache_entries
                    or seed is None
                    or not isinstance(reference_audio, bytes)
                ):
                    prompt = encode_prompt_latents(reference_audio, seed)
                else:
                    prompt = latent_service.get_or_encode(
                        PromptLatentInput(
                            reference_key=xxhash.xxh3_128_hexdigest(reference_audio),
                            reference_audio=reference_audio,
                            seed=seed,
                        )
                    )
                speaker_embedding = speaker_job.result()
            else:
                speaker_embedding = (
                    speaker_service.get_or_encode(reference_audio)
                    if reference_cache_entries
                    else encode_speaker(reference_audio)
                )
        if dense:
            payload.data = DenseSpeechInputs(
                text=text,
                speaker_embedding=speaker_embedding,
                zero_speaker=not references,
                prompt=prompt,
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
                prompt_rows=None,
                latent_position=-1,
            )
        return payload

    def shutdown() -> None:
        speaker_executor.shutdown(wait=True)
        speaker_service.close()
        latent_service.close()
        logger.info(
            f"Ming 3 TTS speaker final reference encode stats: {speaker_service.stats()}"
        )
        logger.info(
            f"Ming 3 TTS prompt latent final reference encode stats: {latent_service.stats()}"
        )

    return SimpleScheduler(
        preprocess, max_concurrency=max_concurrency, shutdown_callback=shutdown
    )


def create_tts_engine_executor(
    model_path: str,
    *,
    max_seq_len: int,
    enable_acoustic_graph: bool,
    prefill_graph_token_counts: list[int] | None = None,
    prefill_graph_max_requests: int = 32,
    enable_async_decode: bool = False,
    enable_fused_flow_head: bool = False,
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
    overrides = copy.deepcopy(server_args_overrides or {})
    overrides.update(
        tp_size=tp_size,
        trust_remote_code=False,
        json_model_override_args=json.dumps({"architectures": [backbone_architecture]}),
        disable_radix_cache=True,
    )
    overrides.setdefault("chunked_prefill_size", max_seq_len if dense else -1)
    overrides.setdefault("enable_mixed_chunk", dense)
    if dense:
        # The checkpoint is bfloat16. SGLang otherwise downcasts a missing
        # dtype to float16, and this backbone overflows to NaN.
        overrides["dtype"] = "bfloat16"
        # note (rsi): prefill embeds through staged input rows, so it replays
        # one full CUDA graph per token bucket like the decode step.
        disables_prefill = overrides.get("disable_cuda_graph") or overrides.get(
            "disable_prefill_cuda_graph"
        )
        if not disables_prefill and prefill_graph_token_counts:
            if not operator_selected_prefill_backend(overrides):
                overrides.setdefault(
                    "cuda_graph_backend_prefill", CudaGraphBackend.FULL
                )
            else:
                pass
            overrides.setdefault("cuda_graph_bs_prefill", prefill_graph_token_counts)
            graph_config = overrides.setdefault("cuda_graph_config", {})
            prefill_config = graph_config.setdefault("prefill", {})
            prefill_config.setdefault(
                "full_prefill_max_req", prefill_graph_max_requests
            )
        else:
            pass
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
        if dense and parameters.get("stream", False):
            raise ValueError("Ming 3 0.5B streaming requires the companion dense PR")
        else:
            pass
        seed = parameters.get("seed")
        if dense:
            prompt = materialize_dense_prompt(tokenizer, payload.data)
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
        # note (rsi): the history moves to the device with the prompt rows in
        # the prefill, so admission never waits on the in-flight decode step.
        if prompt["prompt_latents"] is not None:
            latent_history = prompt["prompt_latents"][-1:]
        else:
            latent_history = torch.zeros((1, latent_dim))
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
            prompt_rows=prompt["prompt_rows"],
            latent_position=prompt["latent_position"],
        )

    device_module = torch.get_device_module(resolved_device)
    result_stream = device_module.Stream(device=resolved_device)

    def build_result(request: Ming3TTSRequestData) -> StagePayload:
        payload = request.stage_payload
        # note (rsi): a finished request's latents are complete once its step
        # resolved, so the copy runs on a side stream instead of waiting for
        # the lookahead step already queued on the loop's stream.
        if not payload.request.params.get("stream", False):
            with device_module.stream(result_stream):
                latents = torch.stack(request.latents).float().cpu()
        else:
            latents = None
        payload.data = {
            "latents": latents,
            "finish_reason": request.finish_reason,
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
        worker,
        SGLangOutputProcessor(),
        stop_policy,
        acoustic_graph_max_batch_size,
        enable_fused_flow_head=enable_fused_flow_head and dense,
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
        stream_output_builder=build_stream_output,
        # note (rsi): a decode step launches before the previous one's host
        # bookkeeping, which then runs while the GPU works.
        enable_async_decode=enable_async_decode,
    )


__all__ = [
    "create_preprocessing_executor",
    "create_tts_engine_executor",
    "create_vocoder_executor",
]
