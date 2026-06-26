# SPDX-License-Identifier: Apache-2.0
"""Stage factories for Fish Audio S2-Pro TTS pipeline.

Each factory returns a callable (for SimpleScheduler) or an OmniScheduler.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch

from sglang_omni.models.fishaudio_s2_pro.payload_types import S2ProState
from sglang_omni.models.fishaudio_s2_pro.request_builders import (
    make_tts_scheduler_adapters,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.generation_batch_policy import (
    build_generation_batch_overrides,
    validate_generation_batch_policy,
)

logger = logging.getLogger(__name__)


def _compile_s2pro_codebook_decoder(model: Any, *, max_batch_size: int) -> None:
    """Compile Fast AR decoder layers while leaving sampling and loop control eager."""
    from sglang.srt.model_executor.cuda_graph_runner import set_torch_compile_config

    if max_batch_size < 1:
        raise ValueError("max_batch_size must be >= 1")

    set_torch_compile_config()
    compile_mode = os.environ.get(
        "SGLANG_TORCH_COMPILE_MODE",
        "max-autotune-no-cudagraphs",
    )
    audio_decoder = model._audio_decoder
    compiled_forward_kvcached_layers = [
        torch.compile(layer.forward_kvcached, mode=compile_mode)
        for layer in audio_decoder.layers
    ]
    audio_decoder.set_compiled_forward_kvcached_layers(
        compiled_forward_kvcached_layers,
        max_batch_size=max_batch_size,
    )
    logger.info(
        "Compiled %d Fast AR decoder layers (mode=%s, max_batch_size=%d)",
        len(compiled_forward_kvcached_layers),
        compile_mode,
        max_batch_size,
    )


def _resolve_s2pro_model_buffer_bs(model: Any) -> int:
    return min(
        int(model.vq_decode_max_batch_size),
        int(model._audio_decoder.kv_cache_max_batch_size),
    )


def _resolve_checkpoint(checkpoint: str) -> str:
    if os.path.isdir(checkpoint):
        return checkpoint
    from huggingface_hub import snapshot_download

    return snapshot_download(checkpoint)


def _load_codec(checkpoint_dir: str, device: str):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval, replace=True)
    codec_path = os.path.join(checkpoint_dir, "codec.pth")
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
    codec.eval().to(device)
    return codec


def load_state(payload: StagePayload) -> S2ProState:
    return S2ProState.from_dict(payload.data)


def store_state(payload: StagePayload, state: S2ProState) -> StagePayload:
    payload.data = state.to_dict()
    return payload


# ---------------------------------------------------------------------------
# Preprocessing — returns callable
# ---------------------------------------------------------------------------


def create_preprocessing_executor(
    model_path: str,
    *,
    max_concurrency: int = 8,
):
    """Returns a threaded scheduler for CPU-heavy preprocessing."""
    from sglang_omni.scheduling.threaded_simple_scheduler import ThreadedSimpleScheduler

    checkpoint_dir = _resolve_checkpoint(model_path)

    from transformers import PreTrainedTokenizerFast

    from sglang_omni.models.fishaudio_s2_pro.tokenizer import (
        Reference,
        S2ProTokenizerAdapter,
    )

    tokenizer = PreTrainedTokenizerFast.from_pretrained(checkpoint_dir)
    adapter = S2ProTokenizerAdapter(tokenizer)
    codec = _load_codec(checkpoint_dir, "cpu")

    def _encode_reference_waveform(audio: torch.Tensor, sr: int) -> torch.Tensor:
        import torchaudio

        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdim=True)
        audio = torchaudio.functional.resample(audio, sr, codec.sample_rate)
        audios = audio.squeeze(0).unsqueeze(0)
        audio_lengths = torch.tensor([audios.shape[1]], dtype=torch.long)
        with torch.no_grad():
            indices, _ = codec.encode(audios, audio_lengths)
            if indices.ndim == 3:
                indices = indices[0]
        return indices.cpu()

    def _encode_reference_audio(audio_path: str) -> torch.Tensor:
        import torchaudio

        audio, sr = torchaudio.load(audio_path)
        return _encode_reference_waveform(audio, int(sr))

    def _encode_reference_data(ref_data: dict[str, Any]) -> torch.Tensor | None:
        data = ref_data.get("base64") or ref_data.get("data")
        if data is None and ref_data.get("bytes") is None:
            return None

        from sglang_omni.preprocessing.audio import AudioMediaIO

        audio_io = AudioMediaIO(target_sr=codec.sample_rate)
        if ref_data.get("bytes") is not None:
            audio, sr = audio_io.load_bytes(ref_data["bytes"])
        else:
            audio, sr = audio_io.load_base64(
                ref_data.get("media_type") or "audio/wav", data
            )
        audio_tensor = torch.from_numpy(audio).float().reshape(1, -1)
        return _encode_reference_waveform(audio_tensor, int(sr))

    def _preprocess(payload: StagePayload) -> StagePayload:
        inputs = payload.request.inputs or {}
        params = payload.request.params or {}
        if isinstance(inputs, str):
            inputs = {"text": inputs}

        text = inputs.get("text", "")
        num_codebooks = inputs.get("num_codebooks", 10)
        codebook_size = inputs.get("codebook_size", 4096)

        references = None
        raw_refs = inputs.get("references")
        if raw_refs:
            references = []
            for ref_data in raw_refs:
                vq_codes = ref_data.get("vq_codes")
                if vq_codes is not None and not isinstance(vq_codes, torch.Tensor):
                    vq_codes = torch.tensor(vq_codes)
                if vq_codes is None and ref_data.get("audio_path"):
                    vq_codes = _encode_reference_audio(ref_data["audio_path"])
                if vq_codes is None:
                    vq_codes = _encode_reference_data(ref_data)
                references.append(
                    Reference(
                        audio_bytes=b"",
                        text=ref_data.get("text", ""),
                        vq_codes=vq_codes,
                    )
                )

        prompt_data = adapter.build_prompt(
            text=text, references=references, num_codebooks=num_codebooks
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
            seed=params.get("seed"),
        )
        return store_state(payload, state)

    return ThreadedSimpleScheduler(_preprocess, max_concurrency=max_concurrency)


# ---------------------------------------------------------------------------
# TTS Engine — returns OmniScheduler
# ---------------------------------------------------------------------------


def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    device: str = "cuda",
    max_new_tokens: int = 2048,
    top_k: int = 30,
    ras_window: int = 16,
    server_args_overrides: dict[str, Any] | None = None,
):
    """Returns OmniScheduler for the Fish TTS AR engine."""
    from sglang_omni.models.fishaudio_s2_pro.bootstrap import (
        bootstrap_text_model_for_decode,
        load_audio_decoder,
        patch_fish_config_for_sglang,
        truncate_rope_to_bf16,
    )
    from sglang_omni.models.fishaudio_s2_pro.fish_scheduler import FishScheduler
    from sglang_omni.models.fishaudio_s2_pro.model_runner import FishS2ProModelRunner
    from sglang_omni.models.fishaudio_s2_pro.tokenizer import S2ProTokenizerAdapter
    from sglang_omni.scheduling.bootstrap import (
        create_sglang_infrastructure_defer_cuda_graph,
    )
    from sglang_omni.scheduling.sglang_backend import (
        SGLangOutputProcessor,
        build_sglang_server_args,
    )

    checkpoint_dir = _resolve_checkpoint(model_path)
    gpu_id = int(device.split(":")[-1]) if ":" in device else 0

    patch_fish_config_for_sglang()

    overrides = build_generation_batch_overrides(
        max_running_requests=64,
        server_args_overrides=server_args_overrides,
        disable_cuda_graph=False,
        mem_fraction_static=0.85,
        chunked_prefill_size=8192,
        dtype="bfloat16",
        enable_torch_compile=True,
        random_seed=int.from_bytes(os.urandom(4), "little") & 0x7FFFFFFF,
    )

    server_args = build_sglang_server_args(
        checkpoint_dir,
        context_length=4096,
        **overrides,
    )
    server_args.disable_overlap_schedule = True
    if getattr(server_args, "attention_backend", None) is None:
        server_args.attention_backend = "fa3"

    want_cuda_graph, (
        model_worker,
        tree_cache,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        prefill_mgr,
        decode_mgr,
        model_config,
    ) = create_sglang_infrastructure_defer_cuda_graph(server_args, gpu_id)

    truncate_rope_to_bf16(model_worker.model_runner.model)

    audio_decoder, num_codebooks, codebook_size, tokenizer = load_audio_decoder(
        checkpoint_dir,
        device=device,
    )
    adapter = S2ProTokenizerAdapter(tokenizer)
    bootstrap_text_model_for_decode(
        text_model=model_worker.model_runner.model,
        audio_decoder=audio_decoder,
        semantic_begin_id=adapter.semantic_begin_id,
        semantic_end_id=adapter.semantic_end_id,
        im_end_token_id=adapter.eos_token_ids[0],
        max_batch_size=server_args.max_running_requests,
        num_codebooks=num_codebooks,
        codebook_size=codebook_size,
        ras_window=ras_window,
    )
    validate_generation_batch_policy(
        model_name="FishAudio S2-Pro",
        server_args=server_args,
        model_buffer_bs=_resolve_s2pro_model_buffer_bs(model_worker.model_runner.model),
    )

    if bool(getattr(server_args, "enable_torch_compile", False)):
        _compile_s2pro_codebook_decoder(
            model_worker.model_runner.model,
            max_batch_size=server_args.torch_compile_max_bs,
        )
        server_args.enable_torch_compile = False

    if want_cuda_graph:
        model_worker.model_runner.init_device_graphs()

    output_proc = SGLangOutputProcessor(
        capture_hidden=False,
        capture_hidden_layers=None,
        model=model_worker.model_runner.model,
    )
    request_builder, result_adapter = make_tts_scheduler_adapters(tokenizer=tokenizer)

    scheduler = FishScheduler(
        tree_cache=tree_cache,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        prefill_manager=prefill_mgr,
        decode_manager=decode_mgr,
        server_args=server_args,
        model_runner=FishS2ProModelRunner(model_worker, output_proc),
        request_builder=request_builder,
        result_adapter=result_adapter,
        im_end_token_id=adapter.eos_token_ids[0],
        max_new_tokens=max_new_tokens,
    )
    return scheduler


# ---------------------------------------------------------------------------
# Vocoder — returns callable
# ---------------------------------------------------------------------------


def create_vocoder_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    max_batch_size: int = 8,
    max_batch_wait_ms: int = 2,
    stream_stride: int = 10,
    stream_followup_stride: int = 90,
    stream_overlap_tokens: int | None = 20,
    stream_crossfade_samples: int = 512,
):
    from sglang_omni.models.fishaudio_s2_pro.streaming_vocoder import (
        S2ProVocoderScheduler,
    )

    if device is None:
        device = f"cuda:{gpu_id}" if gpu_id is not None else "cpu"
    checkpoint_dir = _resolve_checkpoint(model_path)
    codec = _load_codec(checkpoint_dir, device)

    return S2ProVocoderScheduler(
        codec,
        device=device,
        stream_stride=stream_stride,
        stream_followup_stride=stream_followup_stride,
        stream_overlap_tokens=stream_overlap_tokens,
        stream_crossfade_samples=stream_crossfade_samples,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
    )
