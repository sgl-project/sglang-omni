# SPDX-License-Identifier: Apache-2.0
"""Talker executor for Ming-Omni.

Wraps MingOmniTalker as a pipeline Executor stage. The talker
receives decoded text from the thinker and independently generates speech audio
using its own internal LLM + CFM + DiT + AudioVAE pipeline.

The talker is a self-contained TTS system that:
1. Tokenizes input text with its own tokenizer
2. Runs its own Qwen2 LLM with StaticCache + CUDA graphs
3. Uses CFM (Conditional Flow Matching) + DiT for diffusion-based audio synthesis
4. Decodes audio latents to waveform via AudioVAE
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path

import torch

from sglang_omni.executors.interface import Executor
from sglang_omni.proto import StagePayload

logger = logging.getLogger(__name__)

DEFAULT_VOICE = "DB30"


class MingTalkerExecutor(Executor):
    """Executor that wraps MingOmniTalker for speech generation."""

    def __init__(
        self,
        model_path: str,
        talker_model_path: str | None = None,
        device: str = "cuda",
        voice: str = DEFAULT_VOICE,
    ):
        self._model_path = model_path
        self._talker_model_path = talker_model_path or str(Path(model_path) / "talker")
        self._device = device
        self._voice = voice
        self._results: asyncio.Queue[StagePayload] = asyncio.Queue()
        self._aborted: set[str] = set()

        self._talker = None
        self._vae = None
        self._thinker_tokenizer = None

    async def start(self) -> None:
        """Initialize the talker model and AudioVAE."""
        logger.info("Loading Ming talker from %s", self._talker_model_path)
        await asyncio.to_thread(self._load_models)
        logger.info("Ming talker loaded and initialized")

    def _load_models(self) -> None:
        """Load talker model and VAE (runs in thread pool)."""
        from transformers import AutoTokenizer

        from sglang_omni.models.ming_omni.talker import (
            MingOmniTalker,
            MingOmniTalkerConfig,
            SpkembExtractor,
        )
        from sglang_omni.models.ming_omni.talker.audio_vae.modeling_audio_vae import (
            AudioVAE,
        )
        from sglang_omni.models.weight_loader import load_weights_by_prefix

        logger.info(
            "[TALKER] Loading MingOmniTalker from %s (device=%s)",
            self._talker_model_path,
            self._device,
        )

        # 1. Load config from checkpoint
        t0 = time.time()
        config = MingOmniTalkerConfig.from_pretrained_dir(self._talker_model_path)

        # 2. Create model (no weights yet)
        self._talker = MingOmniTalker(config)
        self._talker.eval()

        # 3. Stream weights, then move to device with bf16
        weights = load_weights_by_prefix(self._talker_model_path, prefix="")
        self._talker.load_weights(weights.items())
        self._talker.to(device=self._device, dtype=torch.bfloat16)
        logger.info("[TALKER] MingOmniTalker loaded in %.1fs", time.time() - t0)

        # 4. Load tokenizer externally
        tokenizer = AutoTokenizer.from_pretrained(
            str(Path(self._talker_model_path) / "llm")
        )
        self._talker.set_tokenizer(tokenizer)

        # 5. Load voice presets
        voice_json_path = os.path.join(
            self._talker_model_path, "data", "voice_name.json"
        )
        if os.path.exists(voice_json_path):
            with open(voice_json_path, "r") as f:
                voice_dict = json.load(f)
            for key in voice_dict:
                voice_dict[key]["prompt_wav_path"] = os.path.join(
                    self._talker_model_path,
                    voice_dict[key]["prompt_wav_path"],
                )
            self._talker.set_voice_presets(voice_dict)
        else:
            logger.warning("[TALKER] voice_name.json not found at %s", voice_json_path)

        # 6. Load speaker embedding extractor (optional)
        campplus_path = os.path.join(self._talker_model_path, "campplus.onnx")
        try:
            extractor = SpkembExtractor(campplus_path)
            self._talker.set_spkemb_extractor(extractor)
        except (ImportError, Exception) as e:
            logger.warning("[TALKER] SpkembExtractor not available: %s", e)

        # 7. Load text normalizer (optional)
        try:
            from talker_tn.talker_tn import TalkerTN

            self._talker.set_normalizer(TalkerTN())
        except ImportError:
            logger.warning(
                "[TALKER] TalkerTN (pynini) not available — using identity normalizer"
            )

        # 8. Load AudioVAE
        vae_path = str(Path(self._talker_model_path) / "vae")
        if Path(vae_path).exists():
            t0v = time.time()
            self._vae = AudioVAE.from_pretrained(vae_path, dtype=torch.bfloat16)
            self._vae.to(self._device)
            self._vae.eval()
            logger.info("[TALKER] AudioVAE loaded in %.1fs", time.time() - t0v)
        else:
            logger.warning("[TALKER] AudioVAE not found at %s", vae_path)

        # 9. Load thinker tokenizer for decoding output_ids
        try:
            from sglang_omni.models.ming_omni.components.common import (
                load_ming_tokenizer,
            )

            self._thinker_tokenizer = load_ming_tokenizer(self._model_path)
            logger.info(
                "[TALKER] Thinker tokenizer loaded: %s",
                type(self._thinker_tokenizer).__name__,
            )
        except Exception as e:
            logger.warning("[TALKER] Could not load thinker tokenizer: %s", e)

        # 10. Initialize CUDA graphs
        logger.info("[TALKER] Initializing CUDA graphs...")
        t0g = time.time()
        self._talker.initial_graph()
        logger.info("[TALKER] CUDA graphs initialized in %.1fs", time.time() - t0g)

    async def add_request(self, payload: StagePayload) -> None:
        """Process a TTS request."""
        request_id = payload.request_id
        if request_id in self._aborted:
            return

        text = self._extract_text(payload)
        logger.info(
            "[TALKER] Extracted text (len=%d): %r",
            len(text) if text else 0,
            text[:200] if text else "",
        )
        if not text:
            result = StagePayload(
                request_id=request_id,
                request=payload.request,
                data={"audio_waveform": None, "sample_rate": 44100, "duration": 0.0},
            )
            await self._results.put(result)
            return

        t0 = time.time()
        logger.debug(f"[TALKER] Starting TTS generation for {len(text)} chars...")
        waveform, sample_rate, duration = await asyncio.to_thread(
            self._generate_speech, text
        )
        logger.debug(
            f"[TALKER] TTS done in {time.time() - t0:.1f}s, audio={duration:.2f}s"
        )

        # Serialize tensor to bytes for cross-process msgpack transport
        if waveform is not None:
            waveform_np = waveform.cpu().float().numpy()
            waveform_data = waveform_np.tobytes()
            waveform_dtype = str(waveform_np.dtype)
            waveform_shape = list(waveform_np.shape)
        else:
            waveform_data = None
            waveform_dtype = "float32"
            waveform_shape = []

        result = StagePayload(
            request_id=request_id,
            request=payload.request,
            data={
                "audio_waveform": waveform_data,
                "audio_waveform_dtype": waveform_dtype,
                "audio_waveform_shape": waveform_shape,
                "sample_rate": sample_rate,
                "duration": duration,
            },
        )
        await self._results.put(result)

    async def get_result(self) -> StagePayload:
        while True:
            result = await self._results.get()
            if result.request_id in self._aborted:
                continue
            return result

    async def abort(self, request_id: str) -> None:
        self._aborted.add(request_id)

    def _extract_text(self, payload: StagePayload) -> str:
        """Extract generated text from the thinker output in the payload."""
        data = payload.data
        if not isinstance(data, dict):
            return ""

        # Check thinker_out field
        thinker_out = data.get("thinker_out", {})
        if isinstance(thinker_out, dict):
            output_ids = thinker_out.get("output_ids", [])
            if output_ids:
                tokenizer = self._thinker_tokenizer
                if tokenizer is None and hasattr(self._talker, "tokenizer"):
                    tokenizer = self._talker.tokenizer
                if tokenizer is not None:
                    return tokenizer.decode(output_ids, skip_special_tokens=True)

        # Fallback: pre-decoded text
        text = data.get("generated_text", "")
        if text:
            return text

        # Check stream_state
        stream_state = data.get("stream_state", {})
        return stream_state.get("accumulated_text", "")

    @torch.no_grad()
    def _generate_speech(self, text: str) -> tuple[torch.Tensor | None, int, float]:
        """Generate speech from text using MingOmniTalker.

        Returns:
            Tuple of (waveform tensor, sample_rate, duration in seconds).

        Note (Xuesong): the (None, 44100, 0.0) returns below for "no supported
        generation method" / "no waveforms produced" are pre-existing soft
        failures, kept out of #300's OOM-propagation scope. Tracked in #188.
        """
        if self._talker is None:
            raise RuntimeError("Talker model not loaded")

        all_wavs = []

        if hasattr(self._talker, "omni_audio_generation"):
            for tts_speech, _, _, _ in self._talker.omni_audio_generation(
                tts_text=text,
                voice_name=self._voice,
                audio_detokenizer=self._vae,
                stream=False,
            ):
                if tts_speech is not None:
                    all_wavs.append(tts_speech)
        elif hasattr(self._talker, "instruct_audio_generation"):
            prompt = "Please generate speech based on the following description.\n"
            for tts_speech, _, _, _ in self._talker.instruct_audio_generation(
                prompt=prompt,
                text=text,
                audio_detokenizer=self._vae,
                stream=False,
            ):
                if tts_speech is not None:
                    all_wavs.append(tts_speech)
        else:
            logger.error("Talker has no supported generation method")
            return None, 44100, 0.0

        if not all_wavs:
            return None, 44100, 0.0

        waveform = torch.cat(all_wavs, dim=-1)
        sample_rate = 44100
        if self._vae is not None and hasattr(self._vae, "config"):
            sample_rate = getattr(self._vae.config, "sample_rate", 44100)
        duration = waveform.shape[-1] / sample_rate

        return waveform, sample_rate, duration
