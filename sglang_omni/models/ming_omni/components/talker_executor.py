# SPDX-License-Identifier: Apache-2.0
"""Talker executor for Ming-Omni.

Wraps BailingTalker2 as a pipeline Executor stage. The talker receives
decoded text from the thinker and independently generates speech audio
using its own internal LLM + CFM + DiT + AudioVAE pipeline.

The talker is a self-contained TTS system that:
1. Tokenizes input text with its own tokenizer
2. Runs its own LLM (Qwen2 or BailingMoe) with StaticCache + CUDA graphs
3. Uses CFM (Conditional Flow Matching) + DiT for diffusion-based audio synthesis
4. Decodes audio latents to waveform via AudioVAE
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import torch

from sglang_omni.executors.interface import Executor
from sglang_omni.proto import StagePayload

logger = logging.getLogger(__name__)

DEFAULT_VOICE = "DB30"


class MingTalkerExecutor(Executor):
    """Executor that wraps BailingTalker2 for speech generation.

    This executor:
    - Receives decoded text from the thinker stage
    - Runs the Ming TTS pipeline (own LLM + CFM + AudioVAE)
    - Returns audio waveform as the stage output
    """

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

        # Will be initialized in start()
        self._talker = None
        self._vae = None

    async def start(self) -> None:
        """Initialize the talker model and AudioVAE."""
        logger.info("Loading Ming talker from %s", self._talker_model_path)

        # Add Ming source directory to path for imports
        ming_dir = str(Path(self._model_path).parent)
        if ming_dir not in sys.path:
            sys.path.insert(0, ming_dir)

        await asyncio.to_thread(self._load_models)
        logger.info("Ming talker loaded and initialized")

    def _load_models(self) -> None:
        """Load talker model and VAE (runs in thread pool)."""
        from transformers import AutoConfig, AutoModel

        # Load talker model
        talker_config = AutoConfig.from_pretrained(
            self._talker_model_path, trust_remote_code=True
        )
        self._talker = AutoModel.from_pretrained(
            self._talker_model_path,
            config=talker_config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self._talker.to(self._device)
        self._talker.eval()

        # Load AudioVAE
        vae_path = str(Path(self._talker_model_path) / "vae")
        if Path(vae_path).exists():
            vae_config = AutoConfig.from_pretrained(vae_path, trust_remote_code=True)
            self._vae = AutoModel.from_pretrained(
                vae_path,
                config=vae_config,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            self._vae.to(self._device)
            self._vae.eval()
        else:
            logger.warning("AudioVAE not found at %s, talker may fail", vae_path)

        # Initialize CUDA graphs
        if hasattr(self._talker, "initial_graph"):
            self._talker.initial_graph()

    async def add_request(self, payload: StagePayload) -> None:
        """Process a TTS request."""
        request_id = payload.request_id
        if request_id in self._aborted:
            return

        # Extract text from thinker output
        text = self._extract_text(payload)
        if not text:
            # No text to synthesize, return empty audio
            result = StagePayload(
                request_id=request_id,
                request=payload.request,
                data={
                    "audio_waveform": None,
                    "sample_rate": 24000,
                    "duration": 0.0,
                },
            )
            await self._results.put(result)
            return

        # Run TTS in thread pool (blocking CUDA operations)
        try:
            waveform, duration = await asyncio.to_thread(self._generate_speech, text)
        except Exception as e:
            logger.error("Talker generation failed: %s", e, exc_info=True)
            waveform = None
            duration = 0.0

        result = StagePayload(
            request_id=request_id,
            request=payload.request,
            data={
                "audio_waveform": waveform,
                "sample_rate": 24000,
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
        if isinstance(data, dict):
            # Check thinker_out field
            thinker_out = data.get("thinker_out", {})
            if isinstance(thinker_out, dict):
                output_ids = thinker_out.get("output_ids", [])
                if output_ids:
                    # Decode token IDs to text
                    if hasattr(self._talker, "tokenizer"):
                        return self._talker.tokenizer.decode(
                            output_ids, skip_special_tokens=True
                        )

            # Fallback: check for pre-decoded text
            text = data.get("generated_text", "")
            if text:
                return text

            # Check stream_state for accumulated text
            stream_state = data.get("stream_state", {})
            text = stream_state.get("accumulated_text", "")
            if text:
                return text

        return ""

    @torch.no_grad()
    def _generate_speech(self, text: str) -> tuple[torch.Tensor | None, float]:
        """Generate speech from text using BailingTalker2.

        Returns:
            Tuple of (waveform tensor, duration in seconds).
        """
        if self._talker is None:
            raise RuntimeError("Talker model not loaded")

        all_wavs = []

        # Use the talker's omni_audio_generation method
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
            return None, 0.0

        if not all_wavs:
            return None, 0.0

        waveform = torch.cat(all_wavs, dim=-1)
        sample_rate = 24000  # Ming TTS default sample rate
        duration = waveform.shape[-1] / sample_rate

        return waveform, duration
