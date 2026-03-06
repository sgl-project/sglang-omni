# SPDX-License-Identifier: Apache-2.0
"""Stage executor factories for the FishAudio-S1 TTS pipeline."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import torch

from sglang_omni.executors import EngineExecutor, PreprocessingExecutor
from sglang_omni.models.fishaudio_s1.io import FishAudioState
from sglang_omni.models.fishaudio_s1.pipeline.engine_io import (
    apply_tts_result,
    build_tts_request,
)
from sglang_omni.models.fishaudio_s1.pipeline.state_io import load_state, store_state
from sglang_omni.proto import StagePayload

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers (model loading — mirrors the original run_fishaudio_e2e.py logic)
# ---------------------------------------------------------------------------


def _resolve_checkpoint(checkpoint: str) -> str:
    """Resolve an HF model ID to a local snapshot path, or return as-is."""
    if os.path.isdir(checkpoint):
        return checkpoint
    from huggingface_hub import snapshot_download

    return snapshot_download(checkpoint)


def _load_model_and_tokenizer(checkpoint: str, device: str):
    """Load DualARTransformer and FishTokenizer from a checkpoint."""
    from fish_speech.models.text2semantic.llama import DualARTransformer
    from fish_speech.tokenizer import FishTokenizer

    checkpoint = _resolve_checkpoint(checkpoint)
    logger.info("Loading DualAR model from %s …", checkpoint)
    t0 = time.perf_counter()
    model = DualARTransformer.from_pretrained(checkpoint, load_weights=True)
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    logger.info("DualAR model loaded in %.2fs", time.perf_counter() - t0)

    tokenizer = FishTokenizer.from_pretrained(checkpoint)
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
    """Factory for the preprocessing stage.

    Loads ``FishTokenizer`` and DAC codec once.  For each request, tokenizes
    text, encodes reference audio (if provided), and builds the DualAR prompt.
    """
    checkpoint_dir = _resolve_checkpoint(model_id)

    from fish_speech.tokenizer import FishTokenizer

    from sglang_omni.models.fishaudio_s1.tokenizer import (
        FishTokenizerAdapter,
        Reference,
    )

    tokenizer = FishTokenizer.from_pretrained(checkpoint_dir)
    adapter = FishTokenizerAdapter(tokenizer)

    # Lazy-loaded codec (only when reference audio encoding is needed)
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
        audios = audio[None].to(device)
        audio_lengths = torch.tensor([audios.shape[2]], device=device, dtype=torch.long)
        with torch.no_grad():
            indices, _ = codec.encode(audios, audio_lengths)
            if indices.ndim == 3:
                indices = indices[0]
        return indices.cpu()

    def _preprocess(payload: StagePayload) -> StagePayload:
        inputs = payload.request.inputs or {}
        params = payload.request.params or {}

        # Support raw string inputs from the speech API
        if isinstance(inputs, str):
            inputs = {"text": inputs}

        text = inputs.get("text", "")
        num_codebooks = inputs.get("num_codebooks", 10)
        codebook_size = inputs.get("codebook_size", 4096)

        # Build voice-cloning references
        references: list[Reference] | None = None
        raw_refs = inputs.get("references")

        # Fall back to metadata tts_params for voice cloning via speech API
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

                # Encode from audio path if no pre-encoded codes
                if vq_codes is None and ref_data.get("audio_path"):
                    vq_codes = _encode_reference_audio(ref_data["audio_path"])

                references.append(
                    Reference(
                        audio_bytes=b"",
                        text=ref_data.get("text", ""),
                        vq_codes=vq_codes,
                    )
                )

        input_values, audio_masks, audio_parts = adapter.build_prompt(
            text=text,
            references=references,
            num_codebooks=num_codebooks,
        )

        state = FishAudioState(
            input_values=input_values,
            audio_masks=audio_masks,
            audio_parts=audio_parts,
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            max_new_tokens=params.get("max_new_tokens", 1024),
            temperature=params.get("temperature", 0.8),
            top_p=params.get("top_p", 0.8),
            repetition_penalty=params.get("repetition_penalty", 1.1),
        )
        return store_state(payload, state)

    return PreprocessingExecutor(_preprocess)


# ---------------------------------------------------------------------------
# Stage 2: TTS Engine (DualAR)
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
    """Factory for the TTS engine stage.

    Loads the DualARTransformer and creates an OmniEngine via
    :func:`create_dual_ar_engine`.
    """
    from sglang_omni.models.fishaudio_s1.factory import create_dual_ar_engine

    model, tokenizer, _checkpoint_dir = _load_model_and_tokenizer(model_id, device)
    num_codebooks = model.config.num_codebooks
    codebook_size = model.config.codebook_size

    engine = create_dual_ar_engine(
        model=model,
        tokenizer=tokenizer,
        num_codebooks=num_codebooks,
        codebook_size=codebook_size,
        max_new_tokens=max_new_tokens,
        max_seq_len=max_seq_len,
        device=device,
        use_radix_cache=use_radix_cache,
        use_compile=use_compile,
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
    """Factory for the vocoder stage.

    Loads the DAC codec once and decodes VQ codes to audio waveform.
    """
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
        feature_lengths = torch.tensor([codebook_codes.shape[1]], device=device)

        with torch.no_grad():
            audio, _ = codec.decode(codebook_codes[None], feature_lengths)

        audio_np = audio[0, 0].float().cpu()
        state.audio_samples = audio_np
        state.sample_rate = codec.sample_rate
        payload = store_state(payload, state)

        # Add keys expected by the generic Client result builder.
        # Must be serialisable (no Tensor) since this goes via msgpack.
        payload.data["audio_data"] = audio_np.tolist()
        payload.data["modality"] = "audio"
        return payload

    return PreprocessingExecutor(_vocode)
