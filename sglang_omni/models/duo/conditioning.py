"""Incremental Intent layer-50 extraction with a persistent transformer cache."""

import torch
from peft import PeftModel
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen3VLForConditionalGeneration,
    Qwen3VLProcessor,
)
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config


def load_intent(
    model_path: str,
    adapter_path: str,
    *,
    device: str = "cuda:0",
    tensor_parallel_size: int = 1,
) -> tuple[Qwen3VLProcessor, PeftModel]:
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, local_files_only=True)
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    processor.tokenizer = tokenizer
    if tensor_parallel_size < 1:
        raise ValueError("Intent tensor parallel size must be positive")
    elif tensor_parallel_size > 1:
        parallel_plan = {
            f"model.language_model.{name}": style
            for name, style in Qwen3Config.base_model_tp_plan.items()
        }
        device_map = None
    else:
        parallel_plan = None
        device_map = {"": device}
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation="sdpa",
        device_map=device_map,
        tp_plan=parallel_plan,
        tp_size=tensor_parallel_size if parallel_plan is not None else None,
    )
    if base.get_input_embeddings().num_embeddings != len(tokenizer):
        base.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    else:
        pass
    model = (
        PeftModel.from_pretrained(
            base, adapter_path, is_trainable=False, autocast_adapter_dtype=False
        )
        .eval()
        .requires_grad_(False)
    )
    return processor, model
