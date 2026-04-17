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

logger = logging.getLogger(__name__)


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


def create_preprocessing_executor(model_path: str):
    """Returns SimpleScheduler for preprocessing stage."""
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    checkpoint_dir = _resolve_checkpoint(model_path)

    from transformers import PreTrainedTokenizerFast

    from sglang_omni.models.fishaudio_s2_pro.tokenizer import (
        Reference,
        S2ProTokenizerAdapter,
    )

    tokenizer = PreTrainedTokenizerFast.from_pretrained(checkpoint_dir)
    adapter = S2ProTokenizerAdapter(tokenizer)
    codec = _load_codec(checkpoint_dir, "cpu")

    def _encode_reference_audio(audio_path: str) -> torch.Tensor:
        import torchaudio

        audio, sr = torchaudio.load(audio_path)
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
        )
        return store_state(payload, state)

    return SimpleScheduler(_preprocess)


# ---------------------------------------------------------------------------
# TTS Engine — returns OmniScheduler
# ---------------------------------------------------------------------------


def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    device: str = "cuda",
    max_new_tokens: int = 2048,
    top_k: int = 30,
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
    from sglang_omni.scheduling.factory import _create_sglang_infrastructure
    from sglang_omni.scheduling.sglang_backend import (
        SGLangOutputProcessor,
        build_sglang_server_args,
    )

    checkpoint_dir = _resolve_checkpoint(model_path)
    gpu_id = int(device.split(":")[-1]) if ":" in device else 0

    patch_fish_config_for_sglang()

    overrides: dict[str, Any] = {
        "disable_cuda_graph": False,
        "mem_fraction_static": 0.85,
        "max_running_requests": 64,
        "chunked_prefill_size": 8192,
        "dtype": "bfloat16",
        "random_seed": int.from_bytes(os.urandom(4), "little") & 0x7FFFFFFF,
    }
    if server_args_overrides:
        overrides.update(server_args_overrides)

    server_args = build_sglang_server_args(
        checkpoint_dir,
        context_length=4096,
        **overrides,
    )
    server_args.disable_overlap_schedule = True
    if getattr(server_args, "attention_backend", None) is None:
        server_args.attention_backend = "fa3"

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
    ) = _create_sglang_infrastructure(server_args, gpu_id)

    if want_cuda_graph:
        server_args.disable_cuda_graph = False

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
        im_end_id=adapter.eos_token_ids[0],
        max_batch_size=server_args.max_running_requests,
        num_codebooks=num_codebooks,
        codebook_size=codebook_size,
    )

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
):
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    if device is None:
        device = f"cuda:{gpu_id}" if gpu_id is not None else "cpu"
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
        return payload

    return SimpleScheduler(_vocode)
