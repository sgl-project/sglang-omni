# SPDX-License-Identifier: Apache-2.0
"""LLaDA2-Uni text-to-image token generation via the upstream HF API."""

from __future__ import annotations

import gc
import logging
import random
from typing import Any

import torch

from sglang_omni.models.llada2_uni.config import THINKER_STAGE
from sglang_omni.models.llada2_uni.payload_types import LLaDA2UniPipelineState
from sglang_omni.models.llada2_uni.request_builders import apply_dllm_thinker_result
from sglang_omni.models.weight_loader import resolve_dtype
from sglang_omni.proto import StagePayload

logger = logging.getLogger(__name__)


def _device_type(device: str | torch.device) -> str:
    return torch.device(device).type


def _resolve_image_generator_dtype(
    dtype: str | torch.dtype | None,
    *,
    device: str | torch.device,
) -> torch.dtype:
    resolved = resolve_dtype(dtype)
    if dtype is None and _device_type(device) == "cpu":
        return torch.float32
    return resolved or torch.bfloat16


def seed_image_generation(seed: int | None) -> None:
    """Seed Python and torch RNGs for deterministic HF image-token generation."""
    if seed is None:
        return
    seed = int(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LLaDA2ImageTokenGenerator:
    """Generate LLaDA2-Uni VQ image tokens with the checkpoint's remote code.

    The upstream LLaDA2-Uni checkpoint exposes ``model.generate_image``. Reusing
    that method keeps image-token generation aligned with the reference
    implementation while SGLang's generic dLLM loop remains available for text.
    """

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda",
        dtype: str | torch.dtype | None = None,
        local_files_only: bool = False,
        attn_implementation: str | None = None,
    ) -> None:
        self.model_path = model_path
        self.device = torch.device(device)
        self.dtype = _resolve_image_generator_dtype(dtype, device=self.device)
        self.local_files_only = bool(local_files_only)
        self.attn_implementation = attn_implementation

        self._tokenizer: Any | None = None
        self._model: Any | None = None

    def _load(self) -> tuple[Any, Any]:
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer

        from transformers import AutoModelForCausalLM, AutoTokenizer

        common_kwargs = {
            "trust_remote_code": True,
            "local_files_only": self.local_files_only,
        }
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, **common_kwargs)

        model_kwargs: dict[str, Any] = {
            **common_kwargs,
            "low_cpu_mem_usage": True,
            "torch_dtype": self.dtype,
        }
        if self.device.type == "cuda":
            model_kwargs["device_map"] = {"": str(self.device)}
        else:
            model_kwargs["device_map"] = {"": "cpu"}
            model_kwargs.setdefault("attn_implementation", "sdpa")
        if self.attn_implementation is not None:
            model_kwargs["attn_implementation"] = self.attn_implementation

        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            **model_kwargs,
        ).eval()
        model.tokenizer = tokenizer

        self._model = model
        self._tokenizer = tokenizer
        logger.info(
            "Loaded LLaDA2-Uni HF image token generator from %s on %s dtype=%s",
            self.model_path,
            self.device,
            self.dtype,
        )
        return model, tokenizer

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __call__(self, payload: StagePayload) -> StagePayload:
        state = LLaDA2UniPipelineState.from_dict(payload.data)
        generation = state.generation
        if generation.get("type") != "image":
            return payload

        prompt = generation.get("text_prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("LLaDA2-Uni image generation requires text_prompt.")

        seed_image_generation(generation.get("seed"))
        model, tokenizer = self._load()

        result = model.generate_image(
            prompt,
            tokenizer=tokenizer,
            image_h=int(generation["height"]),
            image_w=int(generation["width"]),
            steps=int(generation.get("steps") or 16),
            block_length=int(generation.get("block_length") or 32),
            cfg_scale=float(generation.get("cfg_scale", 4.0)),
            gen_length=int(generation.get("gen_length") or 1088),
            use_sprint=bool(generation.get("use_sprint", False)),
            remasking=str(generation.get("remasking", "low_confidence")),
            keep_ratio=float(generation.get("keep_ratio", 0.7)),
            cache_warmup_steps=int(generation.get("cache_warmup_steps", 2)),
            confidence_alpha=float(generation.get("confidence_alpha", 0.5)),
            image_keep_ratio=generation.get("image_keep_ratio"),
            text_keep_ratio=generation.get("text_keep_ratio"),
            mode=str(generation.get("mode", "normal")),
        )

        token_ids = [int(t) for t in result["token_ids"]]
        token_grid_h = int(result.get("h", generation["token_grid_h"]))
        token_grid_w = int(result.get("w", generation["token_grid_w"]))
        expected = token_grid_h * token_grid_w
        if len(token_ids) != expected:
            raise ValueError(
                "LLaDA2-Uni image generator returned "
                f"{len(token_ids)} tokens for grid {token_grid_h}x{token_grid_w}."
            )

        generation["token_grid_h"] = token_grid_h
        generation["token_grid_w"] = token_grid_w
        generation["num_image_tokens"] = expected
        if "thinking" in result:
            generation["thinking"] = result["thinking"]

        offset = int(generation.get("image_token_offset", 157184))
        output_ids = [token_id + offset for token_id in token_ids]
        apply_dllm_thinker_result(
            state,
            stage_name=THINKER_STAGE,
            output_ids=output_ids,
            finish_reason="length",
        )

        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data=state.to_dict(),
        )
