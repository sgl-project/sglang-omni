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
    build_tts_request,
)
from sglang_omni.models.fishaudio_s2_pro.pipeline.state_io import (
    load_state,
    store_state,
)
from sglang_omni.proto import StagePayload

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers (model loading)
# ---------------------------------------------------------------------------


def _resolve_checkpoint(checkpoint: str) -> str:
    """Resolve an HF model ID to a local snapshot path, or return as-is."""
    if os.path.isdir(checkpoint):
        return checkpoint
    from huggingface_hub import snapshot_download

    return snapshot_download(checkpoint)


def _load_s2pro_model(checkpoint: str, device: str):
    """Load FishQwen3OmniForCausalLM and HF tokenizer from a checkpoint."""
    from fish_speech.models.text2semantic.configuration import FishQwen3OmniConfig
    from fish_speech.models.text2semantic.modeling import FishQwen3OmniForCausalLM
    from transformers import PreTrainedTokenizerFast

    checkpoint = _resolve_checkpoint(checkpoint)
    logger.info("Loading S2-Pro model from %s …", checkpoint)
    t0 = time.perf_counter()

    config = FishQwen3OmniConfig.from_pretrained(checkpoint)
    model = FishQwen3OmniForCausalLM.from_pretrained(checkpoint, config=config)
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    logger.info("S2-Pro model loaded in %.2fs", time.perf_counter() - t0)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(checkpoint)
    return model, tokenizer, checkpoint


def _load_codec(checkpoint_dir: str, device: str):
    """Load the DAC codec from codec.pth inside the checkpoint directory."""
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval, replace=True)

    codec_path = os.path.join(checkpoint_dir, "codec.pth")
    logger.info("Loading DAC codec from %s …", codec_path)
    t0 = time.perf_counter()

    import fish_speech.models.dac.modded_dac as _dac_mod

    configs_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(_dac_mod.__file__))),
        "configs",
    )
    cfg = OmegaConf.load(os.path.join(configs_dir, "modded_dac_vq.yaml"))
    codec = instantiate(cfg)

    state_dict = torch.load(
        codec_path, map_location=device, mmap=True, weights_only=True
    )
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if any("generator" in k for k in state_dict):
        state_dict = {
            k.replace("generator.", ""): v
            for k, v in state_dict.items()
            if "generator." in k
        }
    codec.load_state_dict(state_dict, strict=False, assign=True)
    codec.eval()
    codec.to(device)
    logger.info("DAC codec loaded in %.2fs", time.perf_counter() - t0)
    return codec


# ---------------------------------------------------------------------------
# Stage 1: Preprocessing
# ---------------------------------------------------------------------------


def create_preprocessing_executor(model_id: str) -> PreprocessingExecutor:
    """Factory for the S2-Pro preprocessing stage.

    Loads HF tokenizer and DAC codec. Tokenizes text, encodes reference audio,
    and builds the S2-Pro prompt using ContentSequence.encode().
    """
    checkpoint_dir = _resolve_checkpoint(model_id)

    from transformers import PreTrainedTokenizerFast

    from sglang_omni.models.fishaudio_s2_pro.tokenizer import (
        Reference,
        S2ProTokenizerAdapter,
    )

    tokenizer = PreTrainedTokenizerFast.from_pretrained(checkpoint_dir)
    adapter = S2ProTokenizerAdapter(tokenizer)

    # Lazy-loaded codec
    _codec_cache: dict[str, Any] = {}

    def _get_codec(device: str = "cpu"):
        if "codec" not in _codec_cache:
            _codec_cache["codec"] = _load_codec(checkpoint_dir, device)
        return _codec_cache["codec"]

    def _encode_reference_audio(audio_path: str, device: str = "cpu") -> torch.Tensor:
        import torchaudio

        codec = _get_codec(device)
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

        if isinstance(inputs, str):
            inputs = {"text": inputs}

        text = inputs.get("text", "")
        num_codebooks = inputs.get("num_codebooks", 10)
        codebook_size = inputs.get("codebook_size", 4096)

        # Build voice-cloning references
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


# ---------------------------------------------------------------------------
# Stage 2: TTS Engine (S2-Pro)
# ---------------------------------------------------------------------------


def create_tts_engine_executor(
    model_id: str,
    *,
    device: str = "cuda:0",
    max_new_tokens: int = 2048,
    max_seq_len: int = 4096,
    use_compile: bool = False,
    use_radix_cache: bool = False,
) -> EngineExecutor:
    """Factory for the S2-Pro TTS engine stage.

    Loads FishQwen3OmniForCausalLM and creates an OmniEngine via
    create_s2pro_engine.
    """
    from sglang_omni.models.fishaudio_s2_pro.factory import create_s2pro_engine

    model, tokenizer, _checkpoint_dir = _load_s2pro_model(model_id, device)

    # Get codebook config from audio_decoder
    num_codebooks = model.config.audio_decoder_config.num_codebooks
    codebook_size = model.config.audio_decoder_config.vocab_size

    engine = create_s2pro_engine(
        model=model,
        tokenizer=tokenizer,
        num_codebooks=num_codebooks,
        codebook_size=codebook_size,
        max_new_tokens=max_new_tokens,
        max_seq_len=max_seq_len,
        device=device,
    )

    def _request_builder(payload: StagePayload):
        state = load_state(payload)
        return build_tts_request(state)

    def _result_builder(payload: StagePayload, result: Any) -> StagePayload:
        state = load_state(payload)
        apply_tts_result(state, result)
        return store_state(payload, state)

    return EngineExecutor(
        engine=engine,
        request_builder=_request_builder,
        result_builder=_result_builder,
    )


# ---------------------------------------------------------------------------
# Stage 3: Vocoder (DAC codec decode)
# ---------------------------------------------------------------------------


def create_vocoder_executor(
    model_id: str,
    *,
    device: str = "cuda:0",
) -> PreprocessingExecutor:
    """Factory for the vocoder stage. Same codec as S1."""
    checkpoint_dir = _resolve_checkpoint(model_id)
    codec = _load_codec(checkpoint_dir, device)

    def _vocode(payload: StagePayload) -> StagePayload:
        state = load_state(payload)

        output_codes = state.output_codes
        if output_codes is None:
            state.audio_samples = None
            return store_state(payload, state)

        if not isinstance(output_codes, torch.Tensor):
            output_codes = torch.tensor(output_codes)

        # output_codes: [num_codebooks+1, T] — rows 1..N are codebook indices
        codebook_codes = output_codes[1:].to(device)  # [num_codebooks, T]

        with torch.no_grad():
            # s2-pro-alpha codec: from_indices([B, N, T]) -> [B, 1, samples]
            audio = codec.from_indices(codebook_codes[None])

        audio_np = audio[0, 0].float().cpu()
        state.audio_samples = audio_np
        state.sample_rate = codec.sample_rate
        payload = store_state(payload, state)

        payload.data["audio_data"] = audio_np.tolist()
        payload.data["modality"] = "audio"
        return payload

    return PreprocessingExecutor(_vocode)
