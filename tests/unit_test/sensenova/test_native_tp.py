# SPDX-License-Identifier: Apache-2.0
"""Distributed parity for the vendored SenseNova Qwen3 attention paths."""

import copy
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from sglang_omni.models.sensenova_u1.neo_unify.configuration_neo_chat import (
    NEOLLMConfig,
)
from sglang_omni.models.sensenova_u1.neo_unify.modeling_qwen3 import Qwen3ForCausalLM
from sglang_omni.models.sensenova_u1.tensor_parallel import apply_tensor_parallel


def native_parity(rank, rendezvous, dtype):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        torch.manual_seed(29)
        config = NEOLLMConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=8,
            max_position_embeddings=128,
            attention_dropout=0.0,
            attention_bias=True,
            rms_norm_eps=1e-6,
        )
        config.layer_types = ["full_attention"] * 2
        config._attn_implementation = (  # noqa: leading-underscore -- Transformers attention backend field
            "eager"
        )
        reference = nn.Module()
        reference.language_model = Qwen3ForCausalLM(config).eval().to(dtype)
        sharded = copy.deepcopy(reference)
        apply_tensor_parallel(sharded, rank, 2, dist.group.WORLD)
        with torch.inference_mode():
            for batch in (1, 2):
                inputs = torch.randn(batch, 5, 32, dtype=dtype)
                indexes = torch.arange(5).expand(3, -1)
                kwargs = {
                    "inputs_embeds": inputs,
                    "indexes": indexes,
                    "attention_mask": {"full_attention": None},
                    "use_cache": True,
                }
                full = reference.language_model.model(**kwargs)
                local = sharded.language_model.model(**kwargs)
                tolerance = 0.03 if dtype == torch.bfloat16 else 2e-5
                torch.testing.assert_close(
                    local.last_hidden_state,
                    full.last_hidden_state,
                    atol=tolerance,
                    rtol=tolerance,
                )
                assert local.past_key_values.layers[0].keys.shape[1] == 2
                for _ in range(3):
                    kwargs = {
                        "inputs_embeds": torch.randn(batch, 3, 32, dtype=dtype),
                        "indexes": torch.arange(5, 8).expand(3, -1),
                        "attention_mask": {"full_attention": None},
                        "image_only": True,
                        "use_cache": True,
                        "update_cache": False,
                    }
                    full_step = reference.language_model.model(
                        past_key_values=full.past_key_values, **kwargs
                    )
                    local_step = sharded.language_model.model(
                        past_key_values=local.past_key_values, **kwargs
                    )
                    torch.testing.assert_close(
                        local_step.last_hidden_state,
                        full_step.last_hidden_state,
                        atol=tolerance,
                        rtol=tolerance,
                    )
                    assert local.past_key_values.get_seq_length() == 5
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="requires Gloo")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_qwen3_prefix_and_image_denoising_tp_parity(tmp_path, dtype):
    mp.spawn(
        native_parity,
        args=((tmp_path / "rendezvous").as_uri(), dtype),
        nprocs=2,
        join=True,
    )
