# SPDX-License-Identifier: Apache-2.0
"""Image generation executor for Ming-Omni.

Wraps a DiffusionBackend (SD3 or Z-Image) as a pipeline Executor stage.
Supports two conditioning modes:

1. **Hidden-state conditioning** (Phase 2) -- when the thinker provides
   hidden_states + gen_mask, a :class:`SemanticConditioner` projects them
   into condition embeddings for the diffusion model.
2. **Text-only conditioning** (Phase 1 fallback) -- the executor decodes
   thinker output_ids to text and uses the backend's built-in text encoder.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import time

import torch

from sglang_omni.executors.interface import Executor
from sglang_omni.models.ming_omni.diffusion.backend import (
    DiffusionBackend,
    ImageGenParams,
)
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
        conditioner=None,  # SemanticConditioner instance (or None for text-only)
        skip_semantic_encoder: bool = False,
    ):
        self._model_path = model_path
        self._dit_type = dit_type
        self._dit_model_path = dit_model_path or model_path
        self._device = device
        self._conditioner = conditioner
        self._skip_semantic_encoder = skip_semantic_encoder

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

        # skip_semantic_encoder is only supported by ZImageBackend.  Other
        # backends (e.g. SD3) use a simpler load_models(path, device) API.
        if self._dit_type == "zimage" and self._skip_semantic_encoder:
            self._backend.load_models(
                self._dit_model_path,
                torch.device(self._device),
                skip_semantic_encoder=True,
            )
        else:
            self._backend.load_models(self._dit_model_path, torch.device(self._device))
        logger.info("[IMG_GEN] Diffusion backend loaded in %.1fs", time.time() - t0)

        # Load thinker tokenizer for decoding output_ids → text prompt
        try:
            from sglang_omni.models.ming_omni.components.common import (
                load_ming_tokenizer,
            )

            self._thinker_tokenizer = load_ming_tokenizer(self._model_path)
            logger.info(
                "[IMG_GEN] Thinker tokenizer loaded: %s",
                type(self._thinker_tokenizer).__name__,
            )
        except Exception as e:
            logger.warning("[IMG_GEN] Could not load thinker tokenizer: %s", e)

    async def add_request(self, payload: StagePayload) -> None:
        """Process an image generation request.

        Two conditioning paths are attempted in order:

        1. **Hidden-state conditioning** -- if a :class:`SemanticConditioner`
           is configured and the payload contains ``hidden_states`` +
           ``gen_mask`` from the thinker, project them into condition
           embeddings and pass directly to the diffusion backend.
        2. **Text-only conditioning** (fallback) -- decode thinker
           ``output_ids`` to text and let the backend's built-in text
           encoder produce the condition embeddings.
        """
        request_id = payload.request_id
        if request_id in self._aborted:
            return

        data = payload.data
        if not isinstance(data, dict):
            data = {}

        # ------------------------------------------------------------------
        # Try hidden-state conditioning (Phase 2)
        # ------------------------------------------------------------------
        condition_embeds = None
        negative_embeds = None

        if self._conditioner is not None:
            condition_embeds, negative_embeds = self._try_condition_from_hidden_states(
                data
            )

        if condition_embeds is not None:
            # Hidden-state conditioning path
            params = self._extract_params(data, payload.request)
            prompt_text = self._extract_text_for_byt5(data)
            logger.info(
                "[IMG_GEN] Using hidden-state conditioning, size=%dx%d, steps=%d",
                params.width,
                params.height,
                params.num_inference_steps,
            )

            t0 = time.time()
            try:
                image = await asyncio.to_thread(
                    self._generate_with_condition_embeds,
                    prompt_text,
                    params,
                    condition_embeds,
                    negative_embeds,
                )
                elapsed = time.time() - t0
                logger.info(
                    "[IMG_GEN] Image generated in %.1fs (%dx%d)",
                    elapsed,
                    image.width,
                    image.height,
                )
            except Exception as e:
                logger.error(
                    "[IMG_GEN] ERROR after %.1fs: %s",
                    time.time() - t0,
                    e,
                    exc_info=True,
                )
                result = StagePayload(
                    request_id=request_id,
                    request=payload.request,
                    data={"image_data": None, "modality": "image", "error": str(e)},
                )
                await self._results.put(result)
                return
        else:
            # ------------------------------------------------------------------
            # Fallback: text-only conditioning (Phase 1)
            # ------------------------------------------------------------------
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
                    "[IMG_GEN] ERROR after %.1fs: %s",
                    time.time() - t0,
                    e,
                    exc_info=True,
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
                text = self._thinker_tokenizer.decode(
                    output_ids, skip_special_tokens=True
                )

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

    # ------------------------------------------------------------------
    # Hidden-state conditioning helpers
    # ------------------------------------------------------------------

    def _try_condition_from_hidden_states(
        self, data: dict
    ) -> tuple[list[torch.Tensor] | None, list[torch.Tensor] | None]:
        """Try to build condition embeddings from thinker hidden states.

        Returns ``(condition_embeds, negative_embeds)`` as lists of tensors
        (one per batch element), or ``(None, None)`` if the required fields
        are missing from *data*.
        """
        thinker_out = data.get("thinker_out", {})
        if not isinstance(thinker_out, dict):
            return None, None

        extra = thinker_out.get("extra_model_outputs", {})
        if not isinstance(extra, dict):
            return None, None

        # Hidden states captured by SGLang during the thinker forward pass
        hidden_states = extra.get("hidden_states")
        if hidden_states is None:
            return None, None

        # gen_mask from mm_inputs (set by the preprocessor)
        mm_inputs = data.get("mm_inputs", {})
        image_gen = mm_inputs.get("image_gen", {})
        gen_mask_list = image_gen.get("gen_mask")
        if gen_mask_list is None:
            return None, None

        # Resolve hidden states to a single tensor
        if isinstance(hidden_states, dict):
            # Side-channel capture: pick the last (highest) layer
            numeric_keys = [
                k
                for k in hidden_states
                if isinstance(k, int) or (isinstance(k, str) and k.isdigit())
            ]
            if not numeric_keys:
                return None, None
            last_key = max(numeric_keys, key=lambda k: int(k))
            hs = hidden_states[last_key]
        elif isinstance(hidden_states, torch.Tensor):
            hs = hidden_states
        else:
            return None, None

        # Extract query token positions using gen_mask
        gen_mask = torch.tensor(gen_mask_list, dtype=torch.bool, device=hs.device)
        if hs.dim() == 2:
            # [seq_len, hidden_dim] -> [1, num_query, hidden_dim]
            query_hidden = hs[gen_mask].unsqueeze(0)
        elif hs.dim() == 3:
            # [batch, seq_len, hidden_dim]
            query_hidden = hs[:, gen_mask, :]
        else:
            logger.warning(
                "[IMG_GEN] Unexpected hidden_states dim=%d, skipping", hs.dim()
            )
            return None, None

        logger.info(
            "[IMG_GEN] Projecting hidden states %s through conditioner",
            list(query_hidden.shape),
        )

        # Project through the conditioner: [B, N, 4096] -> [B, N, 2560]
        condition_embeds = self._conditioner.project(query_hidden)
        negative_embeds = condition_embeds * 0.0

        # Convert to list format expected by DiffusionBackend.generate()
        pos_list = list(condition_embeds.unbind(dim=0))
        neg_list = list(negative_embeds.unbind(dim=0))

        return pos_list, neg_list

    def _extract_params(self, data: dict, request) -> ImageGenParams:
        """Extract image generation parameters without text.

        Used by the hidden-state conditioning path where text is not the
        primary conditioning signal.
        """
        raw_inputs = data.get("raw_inputs")
        img_params_dict: dict = {}
        if isinstance(raw_inputs, dict):
            img_params_dict = raw_inputs.get("image_generation", {})

        # Also check mm_inputs (set by the preprocessor)
        if not img_params_dict:
            mm_inputs = data.get("mm_inputs", {})
            image_gen = mm_inputs.get("image_gen", {})
            img_params_dict = image_gen.get("image_gen_params", {})

        if not img_params_dict and request is not None:
            metadata = getattr(request, "metadata", {}) or {}
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

        return ImageGenParams(
            width=width,
            height=height,
            num_inference_steps=img_params_dict.get("num_inference_steps", 28),
            guidance_scale=img_params_dict.get("guidance_scale", 7.0),
            seed=img_params_dict.get("seed"),
            negative_prompt=img_params_dict.get("negative_prompt", ""),
        )

    def _extract_text_for_byt5(self, data: dict) -> str:
        """Extract original prompt text for ByT5 text rendering.

        When using hidden-state conditioning, the prompt text is still
        passed to generate() so that ZImageBackend can extract quoted
        text for ByT5 encoding if available.
        """
        prompt = data.get("prompt", {})
        if isinstance(prompt, dict):
            return prompt.get("prompt_text", "")
        return ""

    @torch.no_grad()
    def _generate_with_condition_embeds(
        self,
        prompt_text: str,
        params: ImageGenParams,
        condition_embeds: list[torch.Tensor],
        negative_embeds: list[torch.Tensor],
    ):
        """Run the diffusion pipeline with pre-computed condition embeddings."""
        if self._backend is None:
            raise RuntimeError("Diffusion backend not loaded")
        return self._backend.generate(
            prompt_text or "",
            params,
            condition_embeds=condition_embeds,
            negative_condition_embeds=negative_embeds,
        )

    # ------------------------------------------------------------------
    # Text-only generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _generate_image(self, text: str, params: ImageGenParams):
        """Run the diffusion pipeline (called in thread pool)."""
        if self._backend is None:
            raise RuntimeError("Diffusion backend not loaded")
        return self._backend.generate(text, params)
