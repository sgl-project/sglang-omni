# SPDX-License-Identifier: Apache-2.0
"""Image generation executor for Ming-Omni.

Wraps a DiffusionBackend (SD3 or Z-Image) as a pipeline Executor stage.
The executor receives decoded text from the thinker and generates images
using its own self-contained diffusion pipeline (text encoder + DiT + VAE).

This is the Phase 1 implementation using text conditioning.
Phase 2 will replace text input with LLM hidden-state conditioning.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import time

import torch

from sglang_omni.executors.interface import Executor
from sglang_omni.models.ming_omni.diffusion.backend import DiffusionBackend, ImageGenParams
from sglang_omni.proto import StagePayload

logger = logging.getLogger(__name__)


def _create_backend(dit_type: str) -> DiffusionBackend:
    """Instantiate the appropriate diffusion backend."""
    if dit_type == "sd3":
        from sglang_omni.models.ming_omni.diffusion.sd3_backend import SD3Backend

        return SD3Backend()
    elif dit_type == "zimage":
        from sglang_omni.models.ming_omni.diffusion.zimage_backend import ZImageBackend

        return ZImageBackend()
    else:
        raise ValueError(f"Unknown dit_type: {dit_type!r}. Must be 'sd3' or 'zimage'.")


class MingImageGenExecutor(Executor):
    """Executor that generates images via SD3 or Z-Image diffusion."""

    def __init__(
        self,
        model_path: str,
        dit_type: str = "zimage",
        dit_model_path: str | None = None,
        device: str = "cuda",
    ):
        self._model_path = model_path
        self._dit_type = dit_type
        self._dit_model_path = dit_model_path or model_path
        self._device = device

        self._backend: DiffusionBackend | None = None
        self._thinker_tokenizer = None
        self._results: asyncio.Queue[StagePayload] = asyncio.Queue()
        self._aborted: set[str] = set()

    async def start(self) -> None:
        """Load diffusion models and thinker tokenizer."""
        logger.info(
            "[IMG_GEN] Loading %s backend from %s (device=%s)",
            self._dit_type,
            self._dit_model_path,
            self._device,
        )
        await asyncio.to_thread(self._load_models)
        logger.info("[IMG_GEN] Backend loaded and ready")

    def _load_models(self) -> None:
        """Load diffusion backend + thinker tokenizer (runs in thread pool)."""
        t0 = time.time()
        self._backend = _create_backend(self._dit_type)
        self._backend.load_models(self._dit_model_path, torch.device(self._device))
        logger.info("[IMG_GEN] Diffusion backend loaded in %.1fs", time.time() - t0)

        # Load thinker tokenizer for decoding output_ids → text prompt
        try:
            from sglang_omni.models.ming_omni.components.common import load_ming_tokenizer

            self._thinker_tokenizer = load_ming_tokenizer(self._model_path)
            logger.info(
                "[IMG_GEN] Thinker tokenizer loaded: %s",
                type(self._thinker_tokenizer).__name__,
            )
        except Exception as e:
            logger.warning("[IMG_GEN] Could not load thinker tokenizer: %s", e)

    async def add_request(self, payload: StagePayload) -> None:
        """Process an image generation request."""
        request_id = payload.request_id
        if request_id in self._aborted:
            return

        text, params = self._extract_input(payload)
        logger.info(
            "[IMG_GEN] prompt (len=%d): %r, size=%dx%d, steps=%d",
            len(text) if text else 0,
            text[:200] if text else "",
            params.width,
            params.height,
            params.num_inference_steps,
        )

        if not text:
            result = StagePayload(
                request_id=request_id,
                request=payload.request,
                data={"image_data": None, "modality": "image"},
            )
            await self._results.put(result)
            return

        t0 = time.time()
        logger.info("[IMG_GEN] Starting image generation...")
        try:
            image = await asyncio.to_thread(self._generate_image, text, params)
            elapsed = time.time() - t0
            logger.info(
                "[IMG_GEN] Image generated in %.1fs (%dx%d)",
                elapsed,
                image.width,
                image.height,
            )
        except Exception as e:
            logger.error(
                "[IMG_GEN] ERROR after %.1fs: %s", time.time() - t0, e, exc_info=True
            )
            result = StagePayload(
                request_id=request_id,
                request=payload.request,
                data={"image_data": None, "modality": "image", "error": str(e)},
            )
            await self._results.put(result)
            return

        # Serialize image to PNG bytes for cross-process msgpack transport
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        image_bytes = buf.getvalue()
        image_b64 = base64.b64encode(image_bytes).decode("ascii")

        result = StagePayload(
            request_id=request_id,
            request=payload.request,
            data={
                "image_data": image_b64,
                "image_format": "png",
                "image_width": image.width,
                "image_height": image.height,
                "modality": "image",
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

    async def stop(self) -> None:
        if self._backend is not None:
            self._backend.unload()
            self._backend = None

    def _extract_input(self, payload: StagePayload) -> tuple[str, ImageGenParams]:
        """Extract text prompt and image generation params from payload."""
        data = payload.data
        if not isinstance(data, dict):
            return "", ImageGenParams()

        # 1. Decode thinker output_ids to text prompt
        text = ""
        thinker_out = data.get("thinker_out", {})
        if isinstance(thinker_out, dict):
            output_ids = thinker_out.get("output_ids", [])
            if output_ids and self._thinker_tokenizer is not None:
                text = self._thinker_tokenizer.decode(output_ids, skip_special_tokens=True)

        # Fallback: pre-decoded text
        if not text:
            text = data.get("generated_text", "")
        if not text:
            stream_state = data.get("stream_state", {})
            text = stream_state.get("accumulated_text", "")

        # 2. Extract image_generation params from request metadata
        raw_inputs = data.get("raw_inputs")
        img_params_dict: dict = {}
        if isinstance(raw_inputs, dict):
            img_params_dict = raw_inputs.get("image_generation", {})

        # Also check the request's metadata
        if not img_params_dict and payload.request is not None:
            metadata = getattr(payload.request, "metadata", {}) or {}
            img_params_dict = metadata.get("image_generation", {})

        # Parse size string like "1024x1024"
        width = img_params_dict.get("width", 1024)
        height = img_params_dict.get("height", 1024)
        size = img_params_dict.get("size")
        if isinstance(size, str) and "x" in size:
            parts = size.split("x")
            try:
                width, height = int(parts[0]), int(parts[1])
            except ValueError:
                pass

        params = ImageGenParams(
            width=width,
            height=height,
            num_inference_steps=img_params_dict.get("num_inference_steps", 28),
            guidance_scale=img_params_dict.get("guidance_scale", 7.0),
            seed=img_params_dict.get("seed"),
            negative_prompt=img_params_dict.get("negative_prompt", ""),
        )
        return text, params

    @torch.no_grad()
    def _generate_image(self, text: str, params: ImageGenParams):
        """Run the diffusion pipeline (called in thread pool)."""
        if self._backend is None:
            raise RuntimeError("Diffusion backend not loaded")
        return self._backend.generate(text, params)
