# SPDX-License-Identifier: Apache-2.0
"""Stage executor factories for the S2-Pro TTS pipeline."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import torch

from sglang_omni.executors import EngineExecutor, PreprocessingExecutor
from sglang_omni.models.fishaudio_s2_pro.io import S2ProState
from sglang_omni.models.fishaudio_s2_pro.pipeline.engine_io import (
    apply_tts_result,
    build_sglang_tts_request,
)
from sglang_omni.models.fishaudio_s2_pro.pipeline.state_io import (
    load_state,
    store_state,
)
from sglang_omni.models.fishaudio_s2_pro.pipeline.streaming_vocoder import (
    build_stream_vocoder_chunk,
    flush_stream_vocoder_chunk,
    resolve_stream_overlap_tokens,
)
from sglang_omni.proto import StagePayload

logger = logging.getLogger(__name__)

_VOCODER_BYTES_PER_TOKEN = int(5.3 * 1024 * 1024)

# Used only if ``_measure_vocoder_peak_bytes`` itself fails
# (e.g. probe OOMs on a tiny GPU or the codec is unreadable).
_VOCODER_RESERVE_FALLBACK_BYTES = int(2.0 * 1024**3)
_LOW_MEM_FRACTION_WARN_THRESHOLD = 0.4


def _resolve_checkpoint(checkpoint: str) -> str:
    if os.path.isdir(checkpoint):
        return checkpoint
    from huggingface_hub import snapshot_download

    return snapshot_download(checkpoint)


def _load_audio_decoder(checkpoint: str, device: str):
    from transformers import PreTrainedTokenizerFast

    from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.text2semantic.configuration import (
        FishQwen3OmniConfig,
    )
    from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.text2semantic.modeling import (
        FishQwen3OmniForCausalLM,
    )

    checkpoint = _resolve_checkpoint(checkpoint)
    logger.info("Loading S2-Pro model from %s …", checkpoint)
    t0 = time.perf_counter()

    config = FishQwen3OmniConfig.from_pretrained(checkpoint)
    model = FishQwen3OmniForCausalLM.from_pretrained(checkpoint, config=config)
    model = model.to(dtype=torch.bfloat16).eval()

    audio_decoder = model.audio_decoder
    audio_decoder.to(device=device)
    num_codebooks = config.audio_decoder_config.num_codebooks
    codebook_size = config.audio_decoder_config.vocab_size

    del model
    torch.cuda.empty_cache()
    logger.info("Audio decoder loaded in %.2fs", time.perf_counter() - t0)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(checkpoint)
    return audio_decoder, num_codebooks, codebook_size, tokenizer, checkpoint


def _load_codec(checkpoint_dir: str, device: str):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval, replace=True)

    codec_path = os.path.join(checkpoint_dir, "codec.pth")
    logger.info("Loading DAC codec from %s …", codec_path)
    t0 = time.perf_counter()

    import sglang_omni.models.fishaudio_s2_pro.fish_speech.models.dac.modded_dac as _dac_mod

    configs_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(_dac_mod.__file__))),
        "configs",
    )
    cfg = OmegaConf.load(os.path.join(configs_dir, "modded_dac_vq.yaml"))
    codec = instantiate(cfg)

    state_dict = torch.load(
        codec_path, map_location=device, mmap=True, weights_only=True
    )
    codec.load_state_dict(state_dict, strict=False, assign=True)
    codec.eval()
    codec.to(device)
    logger.info("DAC codec loaded in %.2fs", time.perf_counter() - t0)
    return codec


def _warmup_codec(codec: Any, *, num_codebooks: int, device: str) -> None:
    """Pre-load mmap'd codec weights into RAM with a short dummy decode."""
    logger.info("Warming up stream codec on %s …", device)
    t0 = time.perf_counter()
    # Use a tiny 4-token sequence so the first real decode is fast.
    dummy = torch.zeros(1, num_codebooks - 1, 4, dtype=torch.long, device=device)
    with torch.no_grad():
        codec.from_indices(dummy)
    logger.info("Stream codec warmup done in %.2fs", time.perf_counter() - t0)


def _measure_vocoder_peak_bytes(
    checkpoint_dir: str,
    device: str,
    num_codebooks: int,
    probe_tokens: int,
) -> int:
    """Probe-load the vocoder codec to measure its real GPU peak.

    Loads the codec on ``device``, runs one dummy decode at the largest
    per-call shape the vocoder will see in production (``probe_tokens``,
    typically the streaming-chunk size), and returns
    ``max_memory_allocated - pre_probe_baseline``. The codec is freed
    before returning so the probe leaves no permanent allocation.

    The returned value is the reserve SGLang must keep off-limits so the
    vocoder stage can co-exist on the same GPU.
    """
    gpu_id = int(device.split(":")[-1]) if ":" in device else 0
    torch.cuda.empty_cache()
    torch.cuda.synchronize(gpu_id)
    pre = torch.cuda.memory_allocated(gpu_id)
    torch.cuda.reset_peak_memory_stats(gpu_id)

    t0 = time.perf_counter()
    codec = _load_codec(checkpoint_dir, device)
    dummy = torch.zeros(
        1, num_codebooks - 1, probe_tokens, dtype=torch.long, device=device
    )
    with torch.no_grad():
        codec.from_indices(dummy)
    torch.cuda.synchronize(gpu_id)

    peak = torch.cuda.max_memory_allocated(gpu_id) - pre

    del codec, dummy
    torch.cuda.empty_cache()
    torch.cuda.synchronize(gpu_id)

    logger.info(
        "Vocoder probe: peak=%.2f GiB on %s (took %.2fs, dummy=%d tokens)",
        peak / (1024**3),
        device,
        time.perf_counter() - t0,
        probe_tokens,
    )
    return peak


def _compute_auto_mem_fraction(device: str, vocoder_reserve_bytes: int) -> float:
    """Size SGLang's static pool from free GPU memory minus the measured vocoder reserve."""
    gpu_id = int(device.split(":")[-1]) if ":" in device else 0
    total_bytes = torch.cuda.get_device_properties(gpu_id).total_memory
    free_bytes = torch.cuda.mem_get_info(gpu_id)[0]
    target = max(0, free_bytes - vocoder_reserve_bytes)
    fraction = round(min(0.95, target / total_bytes), 3)
    log = logger.warning if fraction < _LOW_MEM_FRACTION_WARN_THRESHOLD else logger.info
    log(
        "Auto-sized mem_fraction_static=%.3f (free=%.1f GiB, total=%.1f GiB, "
        "vocoder_reserve=%.2f GiB)",
        fraction,
        free_bytes / (1024**3),
        total_bytes / (1024**3),
        vocoder_reserve_bytes / (1024**3),
    )
    return fraction


def create_preprocessing_executor(model_path: str) -> PreprocessingExecutor:
    checkpoint_dir = _resolve_checkpoint(model_path)

    from transformers import PreTrainedTokenizerFast

    from sglang_omni.models.fishaudio_s2_pro.tokenizer import (
        Reference,
        S2ProTokenizerAdapter,
    )

    tokenizer = PreTrainedTokenizerFast.from_pretrained(checkpoint_dir)
    adapter = S2ProTokenizerAdapter(tokenizer)

    codec = _load_codec(checkpoint_dir, "cpu")

    def _encode_reference_audio(audio_path: str, device: str = "cpu") -> torch.Tensor:
        import io

        import httpx
        import torchaudio

        if audio_path.startswith(("http://", "https://")):
            resp = httpx.get(audio_path, follow_redirects=True, timeout=30)
            resp.raise_for_status()
            audio, sr = torchaudio.load(io.BytesIO(resp.content))
        else:
            audio, sr = torchaudio.load(audio_path)
        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdim=True)
        audio = torchaudio.functional.resample(audio, sr, codec.sample_rate)
        # s2-pro-alpha codec expects [B, T] (adds channel dim internally)
        audios = audio.squeeze(0).unsqueeze(0).to(device)  # [1, T]
        audio_lengths = torch.tensor([audios.shape[1]], device=device, dtype=torch.long)
        with torch.no_grad():
            indices, _ = codec.encode(audios, audio_lengths)
            if indices.ndim == 3:
                indices = indices[0]
        return indices.cpu()

    def _preprocess(payload: StagePayload) -> StagePayload:
        inputs = payload.request.inputs or {}
        params = payload.request.params or {}

        # Speech endpoint sends prompt as a plain string
        if isinstance(inputs, str):
            inputs = {"text": inputs}

        text = inputs.get("text", "")
        num_codebooks = inputs.get("num_codebooks", 10)
        codebook_size = inputs.get("codebook_size", 4096)

        # Build voice-cloning references
        references: list[Reference] | None = None
        raw_refs = inputs.get("references")
        if raw_refs:
            references = []
            for ref_data in raw_refs:
                vq_codes = ref_data.get("vq_codes")
                if vq_codes is not None and not isinstance(vq_codes, torch.Tensor):
                    vq_codes = torch.tensor(vq_codes)

                if vq_codes is None and ref_data.get("audio_path"):
                    vq_codes = _encode_reference_audio(ref_data["audio_path"])

                references.append(
                    Reference(
                        audio_bytes=b"",
                        text=ref_data.get("text", ""),
                        vq_codes=vq_codes,
                    )
                )

        prompt_data = adapter.build_prompt(
            text=text,
            references=references,
            num_codebooks=num_codebooks,
        )

        state = S2ProState(
            input_ids=prompt_data["input_ids"],
            vq_mask_tokens=prompt_data["vq_mask_tokens"],
            vq_parts=prompt_data["vq_parts"],
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            max_new_tokens=params.get("max_new_tokens", 1024),
            temperature=params.get("temperature", 0.8),
            top_p=params.get("top_p", 0.8),
            top_k=params.get("top_k", 30),
            repetition_penalty=params.get("repetition_penalty", 1.1),
        )
        return store_state(payload, state)

    return PreprocessingExecutor(_preprocess)


def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    device: str = "cuda",
    max_new_tokens: int = 2048,
    top_k: int = 30,
    stream_stride: int = 5,
    stream_followup_stride: int = 100,
    stream_overlap_tokens: int | None = None,
    stream_crossfade_samples: int = 0,
    stream_vocoder_device: str | None = None,
    warmup_stream_codec_on_startup: bool = True,
    server_args_overrides: dict[str, Any] | None = None,
) -> EngineExecutor:
    """Factory for the S2-Pro TTS engine stage."""
    from sglang.srt.server_args import ServerArgs

    from sglang_omni.models.fishaudio_s2_pro.factory import (
        _patch_fish_config_for_sglang,
        create_s2pro_sglang_engine,
    )

    if stream_vocoder_device is None:
        stream_vocoder_device = "cpu"

    audio_decoder, num_codebooks, codebook_size, tokenizer, checkpoint_dir = (
        _load_audio_decoder(model_path, device)
    )

    # TODO (Chenyang): If multi-threaded access becomes
    # possible in the future, add threading.Lock protection
    # at that point.
    _stream_codec: Any = None
    _stream_overlap_tokens: int | None = None

    def _get_stream_codec_bundle() -> tuple[Any, int]:
        nonlocal _stream_codec, _stream_overlap_tokens
        if _stream_codec is None:
            codec = _load_codec(checkpoint_dir, stream_vocoder_device)
            _warmup_codec(
                codec, num_codebooks=num_codebooks, device=stream_vocoder_device
            )
            _stream_codec = codec
            _stream_overlap_tokens = resolve_stream_overlap_tokens(
                codec, stream_overlap_tokens
            )
            logger.info(
                "Streaming codec overlap resolved to %d tokens (delay=%d samples, device=%s)",
                _stream_overlap_tokens,
                int(codec.delay),
                stream_vocoder_device,
            )
        return _stream_codec, int(_stream_overlap_tokens)

    if warmup_stream_codec_on_startup:
        # Load and warm the stream codec during executor creation so the first
        # streaming request is not dominated by codec initialization.
        _get_stream_codec_bundle()

    _patch_fish_config_for_sglang(checkpoint_dir)

    # Probe the vocoder codec to measure how much GPU memory the vocoder stage
    # will need (weights + activation peak + workspace), then size SGLang from
    # free VRAM minus that measurement. The probe shape matches the largest
    # per-call workload the vocoder actually sees in production: the streaming
    # chunk size (``stream_followup_stride``). If the probe itself fails
    # (e.g. tiny GPU, missing checkpoint shards), fall back to a conservative
    # constant. Power users can still bypass with ``server_args_overrides``.
    probe_tokens = max(stream_stride, stream_followup_stride)
    try:
        vocoder_reserve_bytes = _measure_vocoder_peak_bytes(
            checkpoint_dir, device, num_codebooks, probe_tokens
        )
    except Exception as exc:
        logger.warning(
            "Vocoder probe failed (%s: %s); using fallback reserve %.1f GiB",
            type(exc).__name__,
            exc,
            _VOCODER_RESERVE_FALLBACK_BYTES / (1024**3),
        )
        vocoder_reserve_bytes = _VOCODER_RESERVE_FALLBACK_BYTES
    auto_mem_fraction = _compute_auto_mem_fraction(device, vocoder_reserve_bytes)

    server_args_kwargs: dict[str, Any] = dict(
        model_path=checkpoint_dir,
        tp_size=1,
        dtype="bfloat16",
        mem_fraction_static=auto_mem_fraction,
        max_running_requests=64,  # int required; omni scheduler does arithmetic on it
        disable_cuda_graph=False,
    )
    if server_args_overrides:
        server_args_kwargs.update(server_args_overrides)
    server_args = ServerArgs(**server_args_kwargs)

    engine = create_s2pro_sglang_engine(
        server_args=server_args,
        audio_decoder=audio_decoder,
        tokenizer=tokenizer,
        gpu_id=int(device.split(":")[-1]) if ":" in device else 0,
        num_codebooks=num_codebooks,
        codebook_size=codebook_size,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
    )

    # Note (Xuesong, Chenyang):
    # SGLang engine pre-allocates ~85% of total VRAM for model weights
    # and KV cache. The remaining ~15% is shared by runtime activations
    # and the vocoder (DAC decoder).

    # Unlike the KV cache, the vocoder has no pre-allocated memory pool —
    # it allocates dynamically during codec.from_indices() on each request.
    # If the AR model produces an oversized codebook sequence, DAC conv1d
    # layers need ~5.3 MB per token (float32, measured empirically on H100),
    # easily exceeding the remaining free VRAM.

    # To prevent this, we snapshot free GPU memory at engine startup and
    # compute the maximum token count the vocoder can safely handle.
    # Requests whose max_new_tokens exceed this limit are clamped and raise
    # warnings.

    # Caveat: PyTorch's caching allocator reserves memory for intermediate
    # tensors across requests (~5.5 GB over ~100 diverse requests on H100).
    # This memory is NOT lost — the vocoder allocates through the same
    # allocator and reuses cached blocks. The allocator also evicts old
    # blocks when large contiguous allocations are needed.

    # reference: https://github.com/sgl-project/sglang-omni/pull/267

    gpu_id_int = int(device.split(":")[-1]) if ":" in device else 0
    free_mem = torch.cuda.mem_get_info(gpu_id_int)[0]
    max_vocoder_tokens = int(free_mem / _VOCODER_BYTES_PER_TOKEN)
    logger.info(
        f"Vocoder memory guard: GPU free {free_mem / 1e9:.1f} GB, max_vocoder_tokens={max_vocoder_tokens}"
    )

    def _request_builder(payload: StagePayload):
        state = load_state(payload)
        if state.max_new_tokens > max_vocoder_tokens:
            logger.warning(
                f"Request {payload.request_id}: max_new_tokens={state.max_new_tokens} exceeds vocoder limit {max_vocoder_tokens}, clamping."
            )
            state.max_new_tokens = max_vocoder_tokens
        return build_sglang_tts_request(state, tokenizer, request_id=payload.request_id)

    def _result_builder(payload: StagePayload, result: Any) -> StagePayload:
        state = load_state(payload)
        apply_tts_result(state, result)
        payload = store_state(payload, state)
        usage = {
            "prompt_tokens": state.prompt_tokens,
            "completion_tokens": state.completion_tokens,
            "total_tokens": state.prompt_tokens + state.completion_tokens,
        }
        engine_time_s = payload.data.get("engine_time_s")
        if engine_time_s is not None:
            usage["engine_time_s"] = round(float(engine_time_s), 6)
        payload.data["usage"] = usage
        return payload

    def _stream_builder(
        payload: StagePayload | None, item: Any
    ) -> dict[str, Any] | None:
        if payload is None:
            return None
        # Note (Chenyang): Hot path optimization: skip expensive
        # GPU→CPU transfer for non-streaming requests.
        if not payload.request.params.get("stream"):
            return None
        codec, overlap_tokens = _get_stream_codec_bundle()
        return build_stream_vocoder_chunk(
            payload,
            item,
            codec=codec,
            device=stream_vocoder_device,
            stream_stride=stream_stride,
            stream_followup_stride=stream_followup_stride,
            stream_overlap_tokens=overlap_tokens,
            stream_crossfade_samples=stream_crossfade_samples,
        )

    def _flush_stream_builder(payload: StagePayload | None) -> dict[str, Any] | None:
        if payload is None:
            return None
        codec, overlap_tokens = _get_stream_codec_bundle()
        return flush_stream_vocoder_chunk(
            payload,
            codec=codec,
            device=stream_vocoder_device,
            stream_overlap_tokens=overlap_tokens,
            stream_crossfade_samples=stream_crossfade_samples,
        )

    _stream_builder.flush = _flush_stream_builder

    return EngineExecutor(
        engine=engine,
        request_builder=_request_builder,
        result_builder=_result_builder,
        stream_builder=_stream_builder,
    )


def create_vocoder_executor(
    model_path: str,
    *,
    device: str = "cuda",
) -> PreprocessingExecutor:
    """Factory for the vocoder stage."""
    checkpoint_dir = _resolve_checkpoint(model_path)
    codec = _load_codec(checkpoint_dir, device)

    def _vocode(payload: StagePayload) -> StagePayload:
        state = load_state(payload)
        output_codes = state.output_codes

        codebook_codes = output_codes[1:].to(device)

        with torch.no_grad():
            audio = codec.from_indices(codebook_codes[None])

        audio_np = audio[0, 0].float().cpu()
        state.audio_samples = audio_np
        state.sample_rate = codec.sample_rate
        payload = store_state(payload, state)

        payload.data["audio_data"] = audio_np.tolist()
        payload.data["sample_rate"] = codec.sample_rate
        payload.data["modality"] = "audio"
        if state.prompt_tokens or state.completion_tokens:
            usage = {
                "prompt_tokens": state.prompt_tokens,
                "completion_tokens": state.completion_tokens,
                "total_tokens": state.prompt_tokens + state.completion_tokens,
            }
            if state.engine_time_s:
                usage["engine_time_s"] = round(state.engine_time_s, 6)
            payload.data["usage"] = usage
        return payload

    return PreprocessingExecutor(_vocode)
