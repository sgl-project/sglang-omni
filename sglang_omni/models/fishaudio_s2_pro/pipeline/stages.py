# SPDX-License-Identifier: Apache-2.0
"""Stage executor factories for the FishAudio S2-Pro TTS pipeline.

Uses the ``FishQwen3OmniForCausalLM`` model via the ``qwen3.py`` inference
module, which provides a complete generation loop with:
- HuggingFace ``AutoModel`` loading
- Built-in audio decoder for codebook generation
- Repetition Aware Sampling (RAS)
- Constrained semantic decoding
- Proper KV cache management via ``model.reset_caches()``
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import torch

from sglang_omni.executors import PreprocessingExecutor
from sglang_omni.models.fishaudio_s2_pro.io import S2ProState
from sglang_omni.proto import StagePayload

logger = logging.getLogger(__name__)


def _resolve_checkpoint(checkpoint: str) -> str:
    if os.path.isdir(checkpoint):
        return checkpoint
    from huggingface_hub import snapshot_download

    return snapshot_download(checkpoint)


def _load_state(payload: StagePayload) -> S2ProState:
    return S2ProState.from_dict(payload.data.get("s2pro_state", {}))


def _store_state(payload: StagePayload, state: S2ProState) -> StagePayload:
    payload.data["s2pro_state"] = state.to_dict()
    return payload


# ---------------------------------------------------------------------------
# Stage 1: Preprocessing
# ---------------------------------------------------------------------------


def create_preprocessing_executor(
    model_id: str, *, vqgan_device: str = "cpu"
) -> PreprocessingExecutor:
    """Build the Qwen3 chat-format prompt with reference audio.

    Loads the HuggingFace tokenizer and DAC VQGAN codec, then for each
    request encodes reference audio and builds the conversation prompt.
    """
    checkpoint_dir = _resolve_checkpoint(model_id)

    from transformers import PreTrainedTokenizerFast

    from sglang_omni.models.fishaudio_s2_pro.tokenizer import (
        Reference,
        S2ProTokenizerAdapter,
    )

    tokenizer = PreTrainedTokenizerFast.from_pretrained(checkpoint_dir)
    adapter = S2ProTokenizerAdapter(tokenizer)

    _vqgan_cache: dict[str, Any] = {}

    def _get_vqgan():
        if "model" not in _vqgan_cache:
            from fish_speech.models.dac.vqgan import load_model as load_vqgan_model

            codec_path = os.path.join(checkpoint_dir, "codec.pth")
            logger.info("Loading VQGAN codec from %s on %s", codec_path, vqgan_device)
            t0 = time.perf_counter()
            model = load_vqgan_model(
                config_name="modded_dac_vq",
                checkpoint_path=codec_path,
                device=vqgan_device,
            )
            _vqgan_cache["model"] = model
            logger.info("VQGAN loaded in %.2fs", time.perf_counter() - t0)
        return _vqgan_cache["model"]

    def _encode_reference_audio(audio_path: str) -> torch.Tensor:
        from fish_speech.models.dac.vqgan import batch_encode as vqgan_encode

        vqgan_model = _get_vqgan()
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        features = vqgan_encode(vqgan_model, [audio_bytes])
        return torch.cat(features, dim=1)

    async def _preprocess(payload: StagePayload) -> StagePayload:
        inputs = payload.request.inputs or {}
        params = payload.request.params or {}

        if isinstance(inputs, str):
            inputs = {"text": inputs}

        text = inputs.get("text", "")

        references: list[Reference] | None = None
        raw_refs = inputs.get("references")

        if not raw_refs:
            metadata = payload.request.metadata or {}
            tts_params = metadata.get("tts_params", {})
            ref_audio = tts_params.get("ref_audio")
            if ref_audio:
                raw_refs = [
                    {"audio_path": ref_audio, "text": tts_params.get("ref_text", "")}
                ]

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

        input_ids, vq_parts, vq_mask_tokens = adapter.build_prompt(
            text=text,
            references=references,
        )

        state = S2ProState(
            input_ids=input_ids,
            vq_parts=vq_parts,
            vq_mask_tokens=vq_mask_tokens,
            max_new_tokens=params.get("max_new_tokens", 2048),
            temperature=params.get("temperature", 1.0),
            top_p=params.get("top_p", 0.9),
            top_k=params.get("top_k", 30),
            seed=params.get("seed"),
        )
        return _store_state(payload, state)

    return PreprocessingExecutor(_preprocess)


# ---------------------------------------------------------------------------
# Stage 2: TTS Engine (FishQwen3OmniForCausalLM)
# ---------------------------------------------------------------------------


def create_tts_engine_executor(
    model_id: str,
    *,
    device: str = "cuda:0",
    max_new_tokens: int = 2048,
    max_seq_len: int = 4096,
    use_compile: bool = True,
    use_radix_cache: bool = False,
) -> PreprocessingExecutor:
    """TTS engine using ``FishQwen3OmniForCausalLM`` with ``qwen3.generate``.

    Loads the model once via ``qwen3.load_model`` (which handles
    ``AutoModel.from_pretrained``, KV cache setup, and optional
    ``torch.compile``), then runs the complete generation loop for each
    request.
    """
    from fish_speech.models.text2semantic.qwen3 import (
        generate,
        load_model,
    )

    checkpoint_dir = _resolve_checkpoint(model_id)

    logger.info("Loading FishQwen3OmniForCausalLM from %s", checkpoint_dir)
    t0 = time.perf_counter()
    model, tokenizer, decode_one_token_fn = load_model(
        checkpoint_path=checkpoint_dir,
        device=device,
        dtype=torch.bfloat16,
        max_seq_len=max_seq_len,
        max_batch_size=1,
        use_cuda_graph=False,
        use_torch_compile=use_compile,
    )
    logger.info(
        "FishQwen3OmniForCausalLM loaded in %.2fs", time.perf_counter() - t0
    )

    async def _generate(payload: StagePayload) -> StagePayload:
        state = _load_state(payload)

        input_ids = state.input_ids
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids)
        input_ids = input_ids.to(device)

        vq_parts_for_embed = None
        if state.vq_parts is not None:
            vp = state.vq_parts
            if not isinstance(vp, torch.Tensor):
                vp = torch.tensor(vp)
            vq_parts_for_embed = vp.to(device)

        vq_mask = None
        if state.vq_mask_tokens is not None:
            vm = state.vq_mask_tokens
            if not isinstance(vm, torch.Tensor):
                vm = torch.tensor(vm, dtype=torch.bool)
            vq_mask = vm.to(device)

        result = generate(
            model=model,
            input_ids=input_ids,
            max_new_tokens=state.max_new_tokens,
            decode_one_token_fn=decode_one_token_fn,
            temperature=state.temperature,
            top_p=state.top_p,
            top_k=state.top_k,
            num_samples=1,
            constrain_to_semantic=True,
            vq_parts=vq_parts_for_embed,
            vq_mask_tokens=vq_mask,
            seed=state.seed,
        )

        sample = result.samples[0]
        codebook_tokens = sample.vq_parts

        if codebook_tokens is not None and codebook_tokens.numel() > 0:
            state.output_codes = codebook_tokens.cpu()
            state.num_semantic_tokens = sample.vq_mask_tokens.sum().item()
        else:
            state.output_codes = None
            state.num_semantic_tokens = 0

        return _store_state(payload, state)

    return PreprocessingExecutor(_generate)  # async → runs in event loop


# ---------------------------------------------------------------------------
# Stage 3: Vocoder (DAC VQGAN decode)
# ---------------------------------------------------------------------------


def create_vocoder_executor(
    model_id: str,
    *,
    device: str = "cuda:0",
) -> PreprocessingExecutor:
    """Decode VQ codes to audio waveform using the DAC VQGAN codec."""
    checkpoint_dir = _resolve_checkpoint(model_id)

    from fish_speech.models.dac.vqgan import (
        decode as vqgan_decode,
        load_model as load_vqgan_model,
    )

    codec_path = os.path.join(checkpoint_dir, "codec.pth")
    logger.info("Loading VQGAN codec for vocoder from %s", codec_path)
    vqgan_model = load_vqgan_model(
        config_name="modded_dac_vq",
        checkpoint_path=codec_path,
        device=device,
    )

    async def _vocode(payload: StagePayload) -> StagePayload:
        state = _load_state(payload)

        output_codes = state.output_codes
        if output_codes is None:
            state.audio_samples = None
            return _store_state(payload, state)

        if not isinstance(output_codes, torch.Tensor):
            output_codes = torch.tensor(output_codes)

        # output_codes from qwen3.generate: [num_semantic, num_codebooks]
        # vqgan_decode expects list of [num_codebooks, seq_len] tensors
        codes_for_decode = [output_codes.mT.to(device)]

        with torch.no_grad():
            audios = vqgan_decode(vqgan_model, codes_for_decode)

        audio = audios[0].float().cpu()
        if audio.ndim > 1:
            audio = audio.squeeze()

        state.audio_samples = audio
        state.sample_rate = vqgan_model.sample_rate
        payload = _store_state(payload, state)

        payload.data["audio_data"] = audio.tolist()
        payload.data["modality"] = "audio"
        return payload

    return PreprocessingExecutor(_vocode)
