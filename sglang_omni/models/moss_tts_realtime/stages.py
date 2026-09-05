# SPDX-License-Identifier: Apache-2.0
"""Stage factories for MOSS-TTS-Realtime."""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Iterator, Literal

import torch

from sglang_omni.models.moss_tts.hf_loading import (
    moss_transformers_processor_compat as _shared_moss_transformers_processor_compat,
)
from sglang_omni.models.moss_tts_realtime.config import (
    DEFAULT_MOSS_TTS_REALTIME_CODEC_MODEL,
    MossTTSRealtimeResourceLimits,
)
from sglang_omni.models.moss_tts_realtime.reference_encoder import (
    BatchedMossTTSRealtimeAudioEncoder,
    MossTTSRealtimeAudioEncoder,
    MossTTSRealtimeReferenceEncoder,
)
from sglang_omni.models.moss_tts_realtime.request_builders import (
    cleanup_prepared_moss_tts_realtime_request,
    preprocess_moss_tts_realtime_payload,
    set_moss_tts_realtime_preprocessing_context,
)
from sglang_omni.models.moss_tts_realtime.streaming_vocoder import (
    MossTTSRealtimeStreamingVocoderScheduler,
)
from sglang_omni.models.moss_tts_realtime.vocoder_decoder import (
    configure_moss_tts_realtime_vocoder_decoder,
)
from sglang_omni.models.weight_loader import load_module, resolve_dtype
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.checkpoint import resolve_checkpoint

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "MOSS-TTS-Realtime requires the trusted custom code published with the "
    "model and OpenMOSS-Team/MOSS-Audio-Tokenizer."
)
_DEFAULT_LIMITS = MossTTSRealtimeResourceLimits()


@contextmanager
def moss_transformers_processor_compat() -> Iterator[None]:
    """Handle Transformers API drift and HF-cache source symlinks locally."""

    from transformers import dynamic_module_utils

    original_compute_hash = dynamic_module_utils._compute_local_source_files_hash

    def compute_local_source_files_hash(
        pretrained_model_name_or_path: str | os.PathLike[str],
        resolved_module_file: str | os.PathLike[str],
    ) -> str:
        # Transformers 5.12 resolves the module symlink before discovering
        # relative imports. For an HF snapshot this moves discovery under
        # ``blobs/<hash>``, where sibling source files no longer have their
        # logical module names. Discover through the snapshot path, but hash
        # the physical file contents.
        model_path = Path(os.path.abspath(pretrained_model_name_or_path))
        logical_module_file = Path(os.path.abspath(resolved_module_file))

        def source_entry(source_file: str | os.PathLike[str]) -> tuple[str, Path]:
            logical_path = Path(os.path.abspath(source_file))
            try:
                relative_path = logical_path.relative_to(model_path).as_posix()
            except ValueError:
                relative_path = logical_path.as_posix()
            return relative_path, logical_path.resolve()

        source_files = [source_entry(logical_module_file)]
        source_files.extend(
            source_entry(source_file)
            for source_file in dynamic_module_utils.get_relative_import_files(
                logical_module_file
            )
        )
        source_hash = hashlib.sha256()
        for relative_path, physical_path in sorted(
            source_files,
            key=lambda entry: entry[0],
        ):
            source_hash.update(relative_path.encode("utf-8"))
            source_hash.update(physical_path.read_bytes())
        return source_hash.hexdigest()[:16]

    dynamic_module_utils._compute_local_source_files_hash = (
        compute_local_source_files_hash
    )
    try:
        with _shared_moss_transformers_processor_compat():
            yield
    finally:
        dynamic_module_utils._compute_local_source_files_hash = original_compute_hash


def _resolve_codec_device(device: str | None, gpu_id: int | None) -> str:
    if device:
        return device
    if gpu_id is not None:
        return f"cuda:{int(gpu_id)}"
    return "cuda:0"


def _create_moss_tts_realtime_codec_shell(checkpoint_dir: str) -> Any:
    """Build the codec structure without materializing its parameters."""
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModel

    with moss_transformers_processor_compat():
        config = AutoConfig.from_pretrained(
            checkpoint_dir,
            trust_remote_code=True,
        )
        with init_empty_weights(include_buffers=False):
            codec = AutoModel.from_config(config, trust_remote_code=True)

    for name in ("encoder", "decoder", "quantizer"):
        if not isinstance(getattr(codec, name, None), torch.nn.Module):
            raise RuntimeError(
                f"MOSS-TTS-Realtime codec does not expose a {name} module"
            )
    return codec


def _load_moss_tts_realtime_codec_component(
    checkpoint_dir: str,
    *,
    component: Literal["encoder", "decoder"],
    device: str,
    dtype: torch.dtype | None = None,
) -> Any:
    codec = _create_moss_tts_realtime_codec_shell(checkpoint_dir)
    unused_component = "decoder" if component == "encoder" else "encoder"
    setattr(codec, unused_component, torch.nn.ModuleList())

    loaded_component = load_module(
        getattr(codec, component),
        checkpoint_dir,
        prefix=f"{component}.",
        dtype=dtype,
        device=device,
        strict=True,
        local_files_only=True,
    )
    setattr(codec, component, loaded_component)
    codec.quantizer = load_module(
        codec.quantizer,
        checkpoint_dir,
        prefix="quantizer.",
        device=device,
        strict=True,
        local_files_only=True,
    )

    remaining_meta = [
        name
        for name, value in codec.state_dict().items()
        if value.device.type == "meta"
    ]
    if remaining_meta:
        raise RuntimeError(
            "MOSS-TTS-Realtime codec component remained on meta device: "
            f"{remaining_meta[:8]}"
        )
    codec.eval()
    loaded_bytes = sum(
        value.numel() * value.element_size() for value in codec.state_dict().values()
    )
    logger.info(
        "Loaded MOSS-TTS-Realtime codec %s component from %s on %s (%.2f GiB)",
        component,
        checkpoint_dir,
        device,
        loaded_bytes / (1024**3),
    )
    return codec


def _module_tensor_bytes(module: torch.nn.Module) -> int:
    seen: set[int] = set()
    total = 0
    for tensors in (module.parameters(), module.buffers()):
        for tensor in tensors:
            identity = id(tensor)
            if identity in seen:
                continue
            seen.add(identity)
            total += tensor.numel() * tensor.element_size()
    return total


def _tensor_tree_bytes(value: Any, seen: set[int]) -> int:
    if value is None:
        return 0
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)

    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if is_dataclass(value) and not isinstance(value, type):
        return sum(
            _tensor_tree_bytes(getattr(value, field.name), seen)
            for field in fields(value)
        )
    if isinstance(value, Mapping):
        return sum(_tensor_tree_bytes(item, seen) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return sum(_tensor_tree_bytes(item, seen) for item in value)
    if hasattr(value, "__dict__"):
        return sum(_tensor_tree_bytes(item, seen) for item in vars(value).values())
    return 0


def _streaming_state_bytes(codec: torch.nn.Module) -> int:
    seen: set[int] = set()
    return sum(
        _tensor_tree_bytes(getattr(module, "_streaming_state", None), seen)
        for module in codec.modules()
    )


def estimate_moss_tts_realtime_codec_memory(
    model_path: str = DEFAULT_MOSS_TTS_REALTIME_CODEC_MODEL,
    *,
    stream_slots: int,
) -> tuple[int, int]:
    """Return decoder-component and fixed-slot streaming-state bytes.

    ``stream_slots`` is the fixed width of the ``codec.streaming()`` context --
    since slots are session-scoped, callers pass the session-held slot total
    (held sessions plus active turns), not the per-turn decode width.
    """
    if stream_slots < 1:
        raise ValueError("stream_slots must be positive")

    checkpoint_dir = str(resolve_checkpoint(model_path))
    try:
        codec = _create_moss_tts_realtime_codec_shell(checkpoint_dir)
        codec.encoder = torch.nn.ModuleList()
        decoder_component_bytes = _module_tensor_bytes(codec)
        with torch.no_grad(), codec.streaming(stream_slots):
            streaming_state_bytes = _streaming_state_bytes(codec)
    except Exception as exc:
        raise RuntimeError(_INSTALL_HINT) from exc
    return decoder_component_bytes, streaming_state_bytes


def load_moss_tts_realtime_codec(
    model_path: str = DEFAULT_MOSS_TTS_REALTIME_CODEC_MODEL,
    *,
    component: Literal["encoder", "decoder"],
    device: str = "cuda:0",
    dtype: torch.dtype | None = None,
) -> Any:
    if component not in ("encoder", "decoder"):
        raise ValueError(f"unsupported MOSS-TTS-Realtime codec component: {component}")
    checkpoint_dir = str(resolve_checkpoint(model_path))
    resolved_device = str(torch.device(device))
    logger.info(
        "Loading MOSS-TTS-Realtime codec %s component from %s on %s",
        component,
        checkpoint_dir,
        resolved_device,
    )
    return _load_moss_tts_realtime_codec_component(
        checkpoint_dir,
        component=component,
        device=resolved_device,
        dtype=dtype,
    )


def _strict_processor_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def bind_moss_tts_realtime_processor_config(config: Any, processor: Any) -> Any:
    """Validate processor-owned metadata and attach it to the HF config."""
    expected = {
        "channels": int(config.rvq),
        "audio_channel_pad": int(config.audio_pad_token),
        "audio_pad_token_id": int(config.reference_audio_pad),
        "text_pad_token_id": int(config.text_pad),
    }
    for name, expected_value in expected.items():
        actual = _strict_processor_int(
            getattr(processor, name, None),
            f"processor.{name}",
        )
        if actual != expected_value:
            raise ValueError(
                f"processor.{name} must match model config "
                f"{expected_value}, got {actual}"
            )

    audio_bos_token = _strict_processor_int(
        getattr(processor, "audio_bos_token", None),
        "processor.audio_bos_token",
    )
    audio_eos_token = _strict_processor_int(
        getattr(processor, "audio_eos_token", None),
        "processor.audio_eos_token",
    )
    delay_tokens_len = _strict_processor_int(
        getattr(processor, "delay_tokens_len", None),
        "processor.delay_tokens_len",
        minimum=1,
    )
    special_audio_tokens = {
        int(config.audio_pad_token),
        audio_bos_token,
        audio_eos_token,
    }
    if len(special_audio_tokens) != 3:
        raise ValueError("processor audio pad, BOS, and EOS tokens must be distinct")
    if max(special_audio_tokens) >= int(config.audio_vocab_size):
        raise ValueError("processor audio special tokens exceed audio_vocab_size")

    config.audio_bos_token = audio_bos_token
    config.audio_eos_token = audio_eos_token
    config.delay_tokens_len = delay_tokens_len
    processor.model_config = config
    return config


def load_moss_tts_realtime_processor(model_path: str) -> Any:
    """Load the model config and processor through the checkpoint auto map."""

    checkpoint_dir = resolve_checkpoint(model_path)
    logger.info("Loading MOSS-TTS-Realtime processor from %s", checkpoint_dir)
    try:
        from transformers import AutoConfig, AutoProcessor

        with moss_transformers_processor_compat():
            model_config = AutoConfig.from_pretrained(
                checkpoint_dir,
                trust_remote_code=True,
            )
            processor = AutoProcessor.from_pretrained(
                checkpoint_dir,
                trust_remote_code=True,
            )
            bind_moss_tts_realtime_processor_config(model_config, processor)
    except Exception as exc:
        raise RuntimeError(_INSTALL_HINT) from exc
    return processor


def create_preprocessing_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    codec_model_path: str | None = None,
    max_concurrency: int = 8,
    encode_batch_size: int = 8,
    encode_batch_wait_ms: int = 4,
    ref_audio_cache: bool = True,
    ref_audio_cache_max_items: int = 8192,
    ref_audio_cache_max_bytes: int = 64 * 1024 * 1024,
) -> SimpleScheduler:
    # Ops kill switch / A-B toggle; unset preserves the config value.
    env_toggle = os.environ.get("MOSS_REF_AUDIO_CACHE")
    if env_toggle is not None:
        ref_audio_cache = env_toggle.strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
            "",
        )
    resolved_device = _resolve_codec_device(device, gpu_id)
    processor = load_moss_tts_realtime_processor(model_path)
    codec_source = codec_model_path or DEFAULT_MOSS_TTS_REALTIME_CODEC_MODEL
    codec_checkpoint = str(resolve_checkpoint(codec_source))
    codec = load_moss_tts_realtime_codec(
        codec_checkpoint,
        component="encoder",
        device=resolved_device,
    )
    num_quantizers = int(processor.model_config.rvq)
    codec_encoder = MossTTSRealtimeAudioEncoder(
        codec,
        device=resolved_device,
        num_quantizers=num_quantizers,
    )
    audio_encoder = BatchedMossTTSRealtimeAudioEncoder(
        codec_encoder,
        max_batch_size=encode_batch_size,
        max_batch_wait_ms=encode_batch_wait_ms,
    )
    reference_encoder: Any = audio_encoder
    if ref_audio_cache:
        reference_encoder = MossTTSRealtimeReferenceEncoder(
            audio_encoder,
            model_revision=codec_checkpoint,
            num_quantizers=num_quantizers,
            max_items=ref_audio_cache_max_items,
            max_bytes=ref_audio_cache_max_bytes,
        )
    set_moss_tts_realtime_preprocessing_context(
        processor=processor,
        audio_encoder=audio_encoder,
        reference_encoder=reference_encoder,
    )
    return SimpleScheduler(
        preprocess_moss_tts_realtime_payload,
        abort_callback=cleanup_prepared_moss_tts_realtime_request,
        max_concurrency=max_concurrency,
    )


def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    server_args_overrides: dict[str, Any] | None = None,
    max_seq_len: int | None = None,
    total_gpu_memory_fraction: float | None = None,
    codec_model_path: str | None = None,
    max_sessions: int = _DEFAULT_LIMITS.max_sessions,
    max_held_sessions: int = _DEFAULT_LIMITS.max_held_sessions,
    max_active_turns: int = _DEFAULT_LIMITS.max_active_turns,
    max_pending_text_tokens: int = _DEFAULT_LIMITS.max_pending_text_tokens,
    max_pending_text_bytes: int = _DEFAULT_LIMITS.max_pending_text_bytes,
    max_input_updates: int = _DEFAULT_LIMITS.max_input_updates,
    terminal_tombstone_limit: int = _DEFAULT_LIMITS.terminal_tombstone_limit,
    input_idle_timeout_s: float = _DEFAULT_LIMITS.input_idle_timeout_s,
    turn_timeout_s: float = _DEFAULT_LIMITS.turn_timeout_s,
    session_idle_ttl_s: float = _DEFAULT_LIMITS.session_idle_ttl_s,
) -> Any:
    from sglang_omni.models.moss_tts_realtime.engine_builder import (
        MossTTSRealtimeEngineBuilder,
    )

    return MossTTSRealtimeEngineBuilder(
        max_seq_len=max_seq_len,
        total_gpu_memory_fraction=total_gpu_memory_fraction,
        codec_model_path=(codec_model_path or DEFAULT_MOSS_TTS_REALTIME_CODEC_MODEL),
        max_sessions=max_sessions,
        max_held_sessions=max_held_sessions,
        max_active_turns=max_active_turns,
        max_pending_text_tokens=max_pending_text_tokens,
        max_pending_text_bytes=max_pending_text_bytes,
        max_input_updates=max_input_updates,
        terminal_tombstone_limit=terminal_tombstone_limit,
        input_idle_timeout_s=input_idle_timeout_s,
        turn_timeout_s=turn_timeout_s,
        session_idle_ttl_s=session_idle_ttl_s,
    ).build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        dtype=dtype,
        server_args_overrides=server_args_overrides,
    )


create_tts_engine_executor = create_sglang_tts_engine_executor


def create_vocoder_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    codec_model_path: str | None = None,
    stream_slots: int = 16,
    max_batch_size: int = 8,
    max_batch_wait_ms: int = 2,
    cuda_graph: bool = True,
    cuda_graph_frames: list[int] | None = None,
    cuda_graph_min_free_gb: float = 3.0,
    dtype: str | torch.dtype = "bfloat16",
    session_idle_ttl_s: float = 300.0,
) -> MossTTSRealtimeStreamingVocoderScheduler:
    resolved_device = _resolve_codec_device(device, gpu_id)
    decoder_dtype = resolve_dtype(dtype)
    if decoder_dtype is None:
        raise ValueError("MOSS-TTS-Realtime vocoder dtype must be explicit")
    processor = load_moss_tts_realtime_processor(model_path)
    codec = load_moss_tts_realtime_codec(
        codec_model_path or DEFAULT_MOSS_TTS_REALTIME_CODEC_MODEL,
        component="decoder",
        device=resolved_device,
        dtype=decoder_dtype,
    )
    if decoder_dtype != torch.float32:
        configure_moss_tts_realtime_vocoder_decoder(
            codec,
            dtype=decoder_dtype,
        )
    scheduler = MossTTSRealtimeStreamingVocoderScheduler(
        codec,
        n_vq=int(processor.model_config.rvq),
        stream_slots=stream_slots,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
        cuda_graph=cuda_graph,
        cuda_graph_frames=cuda_graph_frames,
        cuda_graph_min_free_gb=cuda_graph_min_free_gb,
        session_idle_ttl_s=session_idle_ttl_s,
    )
    scheduler.warmup_now()
    return scheduler
