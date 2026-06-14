# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the ZONOS2 TTS pipeline.

Three stages:
1. Preprocessing: text normalization, byte tokenization, speaker embedding
2. TTS Engine: MoE AR backbone generating multi-codebook DAC codes
3. Vocoder: DAC decode -> PCM @ 44.1kHz
"""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace
from typing import Any

import torch

from sglang_omni.models.zonos2_tts.hf_config import (
    ensure_zonos2_hf_layout,
    load_zonos2_params,
    register_zonos2_hf_config,
)
from sglang_omni.models.zonos2_tts.payload_types import (
    ZONOS2_SAMPLE_RATE,
    Zonos2TTSState,
)
from sglang_omni.models.zonos2_tts.request_builders import (
    cleanup_prepared_zonos2_request,
    encode_zonos2_speaker_payload,
    make_zonos2_scheduler_adapters,
    preprocess_zonos2_tts_payload,
    set_zonos2_preprocessing_context,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.audio_payload import audio_waveform_payload

logger = logging.getLogger(__name__)

_ZONOS2_INSTALL_HINT = (
    "ZONOS2 TTS requires the 'dac' package for audio decoding. "
    "Install with: pip install descript-audio-codec==1.0.0"
)


def load_state(payload: StagePayload) -> Zonos2TTSState:
    return Zonos2TTSState.from_dict(payload.data)


def store_state(payload: StagePayload, state: Zonos2TTSState) -> StagePayload:
    payload.data = state.to_dict()
    return payload


def _resolve_checkpoint(checkpoint: str) -> str:
    if os.path.isdir(checkpoint):
        return checkpoint
    from huggingface_hub import snapshot_download

    return snapshot_download(checkpoint)


def _torch_dtype(dtype: str | torch.dtype) -> torch.dtype:
    return getattr(torch, dtype) if isinstance(dtype, str) else dtype


def _load_zonos2_model_config(checkpoint_dir: str) -> Any:
    """Load ZONOS2 model config from checkpoint directory."""
    import json

    params_path = os.path.join(checkpoint_dir, "params.json")
    if os.path.isfile(params_path):
        return SimpleNamespace(**load_zonos2_params(checkpoint_dir))

    config_path = os.path.join(checkpoint_dir, "config.json")
    if os.path.isfile(config_path):
        with open(config_path, encoding="utf-8") as f:
            config_data = json.load(f)
        return SimpleNamespace(**config_data)

    logger.warning("No config.json or params.json found in %s", checkpoint_dir)
    return None


def _load_speaker_model(device: str = "cpu") -> Any:
    """Load the speaker embedding model (Qwen3-voice-embedding).

    This is optional and only needed for voice cloning.
    Returns None if the model cannot be loaded.
    """
    try:
        from zonos2.models.speaker_cloning import Qwen3SpeakerEmbedding

        model = Qwen3SpeakerEmbedding(device=device)
        model.eval()
        logger.info("Loaded speaker embedding model on %s", device)
        return model
    except ImportError:
        logger.info(
            "zonos2 package is not importable; voice-clone speaker embedding "
            "extraction is disabled unless a speaker_embedding is supplied"
        )
        return None
    except Exception as exc:
        logger.info(
            "Speaker embedding model not available (voice cloning disabled): %s", exc
        )
        return None


def _build_usage(state: Zonos2TTSState) -> dict[str, Any] | None:
    if not (state.prompt_tokens or state.completion_tokens or state.engine_time_s):
        return None
    usage = {
        "prompt_tokens": int(state.prompt_tokens),
        "completion_tokens": int(state.completion_tokens),
        "total_tokens": int(state.prompt_tokens + state.completion_tokens),
    }
    if state.engine_time_s:
        usage["engine_time_s"] = round(float(state.engine_time_s), 6)
    return usage


# ============================================================================
# Stage 1: Preprocessing
# ============================================================================


def create_preprocessing_executor(
    model_path: str,
    *,
    device: str = "cpu",
    max_concurrency: int = 8,
    load_speaker_model: bool = False,
) -> SimpleScheduler:
    """Create the preprocessing stage executor.

    Loads model config and optionally the speaker embedding model.
    """
    checkpoint_dir = _resolve_checkpoint(model_path)
    model_config = _load_zonos2_model_config(checkpoint_dir)

    del device, load_speaker_model

    set_zonos2_preprocessing_context(
        model_config=model_config,
        speaker_model=None,
    )

    return SimpleScheduler(
        preprocess_zonos2_tts_payload,
        abort_callback=cleanup_prepared_zonos2_request,
        max_concurrency=max_concurrency,
    )


# ============================================================================
# Stage 2: Speaker Encoding
# ============================================================================


def create_speaker_encode_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    max_concurrency: int = 4,
    load_speaker_model: bool = True,
    **_: Any,
) -> SimpleScheduler:
    """Create the optional speaker embedding stage."""

    checkpoint_dir = _resolve_checkpoint(model_path)
    model_config = _load_zonos2_model_config(checkpoint_dir)
    speaker_model = _load_speaker_model(device) if load_speaker_model else None

    set_zonos2_preprocessing_context(
        model_config=model_config,
        speaker_model=None,
    )

    def _speaker(payload: StagePayload) -> StagePayload:
        return encode_zonos2_speaker_payload(payload, speaker_model=speaker_model)

    return SimpleScheduler(
        _speaker,
        abort_callback=cleanup_prepared_zonos2_request,
        max_concurrency=max_concurrency,
    )


# ============================================================================
# Stage 3: TTS Engine (MoE AR backbone)
# ============================================================================


def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    gpu_id: int | None = None,
    tp_rank: int = 0,
    tp_size: int = 1,
    nccl_port: int | None = None,
    dtype: str = "bfloat16",
    server_args_overrides: dict[str, Any] | None = None,
    total_gpu_memory_fraction: float | None = None,
    enable_async_decode: bool = False,
    async_decode_min_batch_size: int = 2,
) -> Any:
    """Create the SGLang-based TTS engine executor.

    Uses OmniScheduler with CUDA graphs for efficient batched inference.
    """
    from sglang_omni.models.zonos2_tts.model_runner import Zonos2TTSModelRunner
    from sglang_omni.scheduling.bootstrap import create_sglang_infrastructure
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler
    from sglang_omni.scheduling.sglang_backend import (
        SGLangOutputProcessor,
        build_sglang_server_args,
    )

    register_zonos2_hf_config()
    checkpoint_dir = _resolve_checkpoint(model_path)
    sglang_checkpoint_dir = ensure_zonos2_hf_layout(checkpoint_dir)
    zonos2_config = _load_zonos2_model_config(checkpoint_dir)
    if gpu_id is not None:
        device = f"cuda:{gpu_id}"
    gpu_id = int(device.split(":")[-1]) if ":" in device else 0
    context_length = int(getattr(zonos2_config, "max_seqlen", 6144) or 6144)
    cuda_graph_max_bs = 256

    overrides: dict[str, Any] = {
        "dtype": dtype,
        "cuda_graph_bs": [1, 2, 4, *range(8, cuda_graph_max_bs + 1, 8)],
        "cuda_graph_max_bs": cuda_graph_max_bs,
        "disable_cuda_graph": False,
        "disable_overlap_schedule": True,
        "enable_torch_compile": False,
        "max_prefill_tokens": 8192,
        "max_running_requests": 256,
        "sampling_backend": "pytorch",
        "torch_compile_max_bs": 32,
        "trust_remote_code": True,
    }
    if server_args_overrides:
        overrides.update(server_args_overrides)
    overrides["tp_size"] = int(tp_size)

    server_args = build_sglang_server_args(
        sglang_checkpoint_dir,
        context_length=context_length,
        **overrides,
    )

    want_cuda_graph = not bool(getattr(server_args, "disable_cuda_graph", False))
    if want_cuda_graph:
        server_args.disable_cuda_graph = True

    (
        model_worker,
        tree_cache,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        prefill_mgr,
        decode_mgr,
        model_config,
    ) = create_sglang_infrastructure(
        server_args,
        gpu_id,
        tp_rank=tp_rank,
        nccl_port=nccl_port,
        model_arch_override="Zonos2SGLangModel",
        total_gpu_memory_fraction=total_gpu_memory_fraction,
    )

    if want_cuda_graph:
        server_args.disable_cuda_graph = False

    model = model_worker.model_runner.model
    if want_cuda_graph:
        model_worker.model_runner.init_device_graphs()

    output_proc = SGLangOutputProcessor(
        capture_hidden=False,
        capture_hidden_layers=None,
        model=model,
    )
    request_builder, result_adapter = make_zonos2_scheduler_adapters(model=model)

    def abort_request(request_id: str) -> None:
        cleanup_prepared_zonos2_request(request_id)
        reset_request = getattr(model, "reset_request", None)
        if reset_request is not None:
            reset_request(request_id)

    return OmniScheduler(
        tp_worker=model_worker,
        tree_cache=tree_cache,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        server_args=server_args,
        model_config=model_config,
        prefill_manager=prefill_mgr,
        decode_manager=decode_mgr,
        model_runner=Zonos2TTSModelRunner(model_worker, output_proc),
        request_builder=request_builder,
        result_adapter=result_adapter,
        abort_callback=abort_request,
        enable_async_decode=enable_async_decode,
        async_decode_min_batch_size=async_decode_min_batch_size,
    )


# ============================================================================
# Stage 4: Vocoder (DAC decode -> PCM @ 44.1kHz)
# ============================================================================


def shear_up(x: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Remove delay pattern: column j shifted up by j rows.

    This is the inverse of shear() - it removes the delay pattern applied
    during generation to align all codebook outputs for DAC decoding.
    """
    H, W = x.shape[-2:]
    out = x.new_full(x.shape, pad_id)
    for j in range(W):
        if H > j:
            out[..., : H - j, j] = x[..., j:, j]
    return out


def prepare_dac_codes_for_decode(
    audio_codes: torch.Tensor,
    *,
    n_codebooks: int,
    audio_pad_id: int,
    codebook_size: int,
    eos_frame: int | None,
) -> torch.Tensor | None:
    codes = torch.as_tensor(audio_codes, dtype=torch.long)
    if codes.ndim != 2:
        raise RuntimeError(
            f"ZONOS2 audio_codes must be [T, C], got {tuple(codes.shape)}"
        )

    codes = shear_up(codes, int(audio_pad_id))
    valid_frames = max(0, int(codes.shape[0]) - (int(n_codebooks) - 1))
    if eos_frame is not None and int(eos_frame) >= 0:
        valid_frames = min(valid_frames, max(0, int(eos_frame)))
    if valid_frames <= 0:
        return None

    codes = codes[:valid_frames]
    return torch.clamp(codes, max=int(codebook_size) - 1)


def _resolve_codec_device(device: str | None, gpu_id: int | None) -> str:
    """Pick the DAC device.

    Stage placement can inject ``gpu_id`` after CUDA_VISIBLE_DEVICES is scoped
    for the process, so it takes precedence over the static config device. A
    static CUDA device still falls back to cuda:0 when only one local device is
    visible.
    """
    if gpu_id is not None:
        return f"cuda:{int(gpu_id)}"
    if device:
        if device.startswith("cuda:") and torch.cuda.is_available():
            try:
                index = int(device.split(":", 1)[1])
            except ValueError:
                return device
            if index >= torch.cuda.device_count():
                logger.info(
                    "Requested ZONOS2 DAC device %s is not visible; using cuda:0",
                    device,
                )
                return "cuda:0"
        return device
    return "cuda:0"


def _slice_batched_dac_waveforms(
    wavs: torch.Tensor,
    frame_lengths: list[int],
) -> list[torch.Tensor]:
    """Trim padded batched DAC output back to each request's code length."""
    if wavs.ndim == 1:
        wavs = wavs.view(1, -1)
    elif wavs.ndim == 3 and wavs.shape[1] == 1:
        wavs = wavs[:, 0, :]
    elif wavs.ndim != 2:
        wavs = wavs.reshape(wavs.shape[0], -1)

    if not frame_lengths:
        return []
    max_frames = max(max(int(length), 0) for length in frame_lengths)
    if max_frames <= 0:
        return [torch.zeros(1, dtype=torch.float32) for _ in frame_lengths]

    max_samples = int(wavs.shape[-1])
    trimmed: list[torch.Tensor] = []
    for wav, frames in zip(wavs, frame_lengths):
        frames = max(int(frames), 0)
        samples = max(1, round(max_samples * frames / max_frames))
        trimmed.append(wav[:samples].contiguous())
    return trimmed


def create_vocoder_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    gpu_id: int | None = None,
    dtype: str = "float32",
    max_batch_size: int = 8,
    max_batch_wait_ms: int = 2,
    max_batch_frames: int | None = None,
) -> SimpleScheduler:
    """Create the DAC vocoder stage executor.

    Decodes multi-codebook audio codes to PCM waveform at 44.1kHz.
    """
    device = _resolve_codec_device(device, gpu_id)

    # Lazy-load DAC model
    _dac_model = None

    def _get_dac():
        nonlocal _dac_model
        if _dac_model is None:
            try:
                import dac as dac_module

                _dac_model = (
                    dac_module.DAC.load(
                        dac_module.utils.download(model_type="44khz")
                    )
                    .eval()
                    .to(device)
                )
                logger.info("Loaded DAC 44kHz vocoder on %s", device)
            except ImportError as exc:
                raise RuntimeError(_ZONOS2_INSTALL_HINT) from exc
        return _dac_model

    # Pre-load DAC at startup
    _get_dac()

    def _prepare_vocoder_item(
        payload: StagePayload,
    ) -> tuple[Zonos2TTSState, torch.Tensor | None]:
        state = load_state(payload)
        audio_codes = state.audio_codes
        if audio_codes is None:
            raise RuntimeError("ZONOS2 vocoder requires audio_codes")

        if not isinstance(audio_codes, torch.Tensor):
            audio_codes = torch.as_tensor(audio_codes, dtype=torch.long)

        if audio_codes.numel() == 0:
            raise RuntimeError("ZONOS2 generated no audio codes")

        return (
            state,
            prepare_dac_codes_for_decode(
                audio_codes,
                n_codebooks=state.n_codebooks,
                audio_pad_id=state.audio_pad_id,
                codebook_size=state.codebook_size,
                eos_frame=state.eos_frame,
            ),
        )

    def _vocoder_request_cost(payload: StagePayload) -> int:
        """Use generated frame count as DAC batch cost to limit pad waste."""
        try:
            state = load_state(payload)
            audio_codes = state.audio_codes
            if audio_codes is None:
                return 0
            if isinstance(audio_codes, torch.Tensor):
                return int(audio_codes.shape[0]) if audio_codes.ndim > 0 else 0
            return int(len(audio_codes))
        except Exception:
            return 0

    def _store_vocoder_result(
        payload: StagePayload,
        state: Zonos2TTSState,
        wav: torch.Tensor,
    ) -> StagePayload:
        # Build output payload
        audio_payload = audio_waveform_payload(wav, source_hint="ZONOS2")
        state.audio_codes = None  # Free memory
        state.sample_rate = ZONOS2_SAMPLE_RATE
        payload = store_state(payload, state)
        payload.data.update(audio_payload)
        payload.data["sample_rate"] = ZONOS2_SAMPLE_RATE
        payload.data["modality"] = "audio"
        usage = _build_usage(state)
        if usage is not None:
            payload.data["usage"] = usage
        return payload

    def _vocode_batch(payloads: list[StagePayload]) -> list[StagePayload]:
        prepared = [_prepare_vocoder_item(payload) for payload in payloads]
        non_empty = [
            (payload, state, codes)
            for payload, (state, codes) in zip(payloads, prepared)
            if codes is not None
        ]
        decoded_by_id: dict[int, torch.Tensor] = {}
        if non_empty:
            frame_lengths = [int(codes.shape[0]) for _, _, codes in non_empty]
            max_frames = max(frame_lengths)
            n_codebooks = max(int(codes.shape[1]) for _, _, codes in non_empty)
            batch_codes = torch.zeros(
                (len(non_empty), n_codebooks, max_frames),
                dtype=torch.long,
                device=device,
            )
            for idx, (_, _, codes) in enumerate(non_empty):
                frames, codebooks = int(codes.shape[0]), int(codes.shape[1])
                batch_codes[idx, :codebooks, :frames] = codes.T.to(
                    device=device,
                    dtype=torch.long,
                    non_blocking=True,
                )

            dac = _get_dac()
            with torch.no_grad(), torch.inference_mode():
                z = dac.quantizer.from_codes(batch_codes)[0]
                wavs = dac.decode(z).float().detach().cpu()
            trimmed = _slice_batched_dac_waveforms(wavs, frame_lengths)
            decoded_by_id = {
                id(payload): wav
                for (payload, _, _), wav in zip(non_empty, trimmed)
            }

        results = []
        for payload, (state, codes) in zip(payloads, prepared):
            if codes is None:
                wav = torch.zeros(1, dtype=torch.float32)
            else:
                wav = decoded_by_id[id(payload)]
            results.append(_store_vocoder_result(payload, state, wav))
        return results

    def _vocode(payload: StagePayload) -> StagePayload:
        return _vocode_batch([payload])[0]

    return SimpleScheduler(
        _vocode,
        batch_compute_fn=_vocode_batch,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
        request_cost_fn=_vocoder_request_cost,
        max_batch_cost=max_batch_frames,
    )
