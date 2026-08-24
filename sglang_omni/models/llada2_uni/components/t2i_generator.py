# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

from sglang_omni.models.llada2_uni.components.common import resolve_local_model_dir

DEFAULT_T2I_STEPS = 16
DEFAULT_T2I_CFG_SCALE = 4.0
DEFAULT_T2I_IMAGE_SIZE = 512


def extract_prompt_text(inputs: Any) -> str:
    if isinstance(inputs, dict):
        messages = inputs.get("messages", [])
    else:
        messages = inputs or []
    for message in reversed(messages):
        if message.get("role", "user") != "user":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        parts = [
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        ]
        return " ".join(part for part in parts if part)
    return ""


class LLaDA2T2IGenerator:
    def __init__(self, model_path: str, device: str = "cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_dir = resolve_local_model_dir(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, device_map={"": device}, trust_remote_code=True
        )
        # A blanket .to(bfloat16) would upcast FP8 expert weights and double memory.
        for param in model.parameters():
            if param.dtype != torch.float8_e4m3fn:
                param.data = param.data.to(torch.bfloat16)
        model.tokenizer = tokenizer
        self._model = model.eval()

    def __call__(self, payload):
        params = payload.request.params or {}
        prompt = extract_prompt_text(payload.request.inputs)
        result = self._model.generate_image(
            prompt,
            image_h=int(params.get("image_h", DEFAULT_T2I_IMAGE_SIZE)),
            image_w=int(params.get("image_w", DEFAULT_T2I_IMAGE_SIZE)),
            steps=int(params.get("t2i_steps", DEFAULT_T2I_STEPS)),
            cfg_scale=float(params.get("cfg_scale", DEFAULT_T2I_CFG_SCALE)),
        )
        token_ids = result["token_ids"]
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        payload.data = {
            "vq_token_ids": [int(t) for t in token_ids],
            "grid_h": int(result["h"]),
            "grid_w": int(result["w"]),
        }
        return payload
