# SPDX-License-Identifier: Apache-2.0
"""Tiny SenseNova model and tokenizer that run generation and editing on CPU."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import pytest
import torch
from sglang.kernels.fused_op import get_fused_op_backend, set_fused_op_backend
from sglang.kernels.spec import KernelBackend
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

from sglang_omni.vendor.sensenova_u1.configuration_neo_chat import NEOChatConfig
from sglang_omni.vendor.sensenova_u1.modeling_neo_chat import NEOChatModel


def tiny_sensenova_config(num_hidden_layers: int = 1) -> NEOChatConfig:
    return NEOChatConfig(
        vision_config={
            "hidden_size": 32,
            "llm_hidden_size": 32,
            "patch_size": 16,
            "max_position_embeddings_vision": 64,
        },
        llm_config={
            "architectures": ["Qwen3ForCausalLM"],
            "vocab_size": 16,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": num_hidden_layers,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "max_position_embeddings": 2048,
            "max_position_embeddings_hw": 64,
        },
        template="neo1_0",
        use_pixel_head=False,
        fm_head_layers=2,
        concat_time_token_num=0,
        noise_scale=1.0,
        noise_scale_mode="constant",
        noise_scale_base_image_seq_len=1,
        add_noise_scale_embedding=True,
        noise_scale_max_value=10.0,
    )


def tiny_sensenova_model(num_hidden_layers: int = 1) -> NEOChatModel:
    """Build the same weights on every call, so separate models can be compared."""
    torch.manual_seed(0)
    return NEOChatModel(tiny_sensenova_config(num_hidden_layers)).eval()


def tiny_sensenova_tokenizer() -> PreTrainedTokenizerFast:
    vocabulary = {"[PAD]": 0, "[UNK]": 1, "<img>": 2, "</img>": 3, "<IMG_CONTEXT>": 4}
    tokenizer_backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer_backend.pre_tokenizer = WhitespaceSplit()
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        additional_special_tokens=["<img>", "</img>", "<IMG_CONTEXT>"],
    )


@contextmanager
def torch_fused_op_backend() -> Iterator[None]:
    """Force native PyTorch ops for CPU tests when running on an accelerator host."""
    previous_backend = get_fused_op_backend()
    set_fused_op_backend(KernelBackend.TORCH)
    try:
        yield
    finally:
        set_fused_op_backend(previous_backend)


def mock_cpu_device_memory(
    monkeypatch: pytest.MonkeyPatch, free_memory_bytes: int = 4 * 1024**3
) -> None:
    """Supply memory budget queries when SenseNova editing tests run on CPU."""
    original_get_device_module = torch.get_device_module
    cpu_module = SimpleNamespace(
        mem_get_info=lambda device: (free_memory_bytes, 0),
        memory_reserved=lambda device: 0,
        memory_allocated=lambda device: 0,
    )

    def resolve_device_module(
        device: torch.device | str | None = None,
    ) -> ModuleType | SimpleNamespace:
        if device is not None and torch.device(device).type == "cpu":
            return cpu_module
        else:
            return original_get_device_module(device)

    monkeypatch.setattr(torch, "get_device_module", resolve_device_module)
