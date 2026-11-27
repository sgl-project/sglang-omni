# SPDX-License-Identifier: Apache-2.0
"""Native weight-loading, numerical, and dual-tail attention contracts."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from sglang.srt.configs.model_config import ModelConfig
from transformers import Qwen3Config
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm, Qwen3RotaryEmbedding

from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.models.moss_speech.hf_config import (
    MossSpeechConfig,
    ensure_moss_speech_config_registered,
)
from sglang_omni.models.moss_speech.model_layers import (
    MossSpeechRMSNorm,
    reference_rope_frequencies,
)
from sglang_omni.models.moss_speech.model_runner import MossSpeechModelRunner
from sglang_omni.models.moss_speech.request_builders import MossSpeechSGLangRequestData
from sglang_omni.models.moss_speech.sglang_model import MossSpeechSGLangModel
from sglang_omni.scheduling.types import SchedulerRequest
from tests.unit_test.fixtures.accelerator import require_cuda
from tests.unit_test.moss_speech.model_helpers import (
    checkpoint_weights,
    decode,
    native_runtime,
    prefill,
    tiny_model,
)

NATIVE_CPU = pytest.mark.skipif(
    os.environ.get("SGLANG_USE_CPU_ENGINE") != "1",
    reason="Native CPU construction requires SGLANG_USE_CPU_ENGINE=1 before imports",
)


@NATIVE_CPU
def test_decode_packs_both_channels_in_current_batch_order() -> None:
    """Changing decode order must move token rows rather than stale embeddings."""
    with native_runtime(torch.device("cpu")), torch.no_grad():
        model = tiny_model(torch.device("cpu"))
        runner = object.__new__(MossSpeechModelRunner)
        runner.model = model
        rows = [(3, 17), (model.config.modality_pad_token_id, 4)]
        requests = [
            SchedulerRequest(
                request_id=str(index),
                data=MossSpeechSGLangRequestData(output_rows=[row]),
            )
            for index, row in enumerate(rows)
        ]
        batch = type(
            "DecodeBatch", (), {"input_ids": torch.zeros(2, dtype=torch.long)}
        )()
        for order in (requests, requests[::-1]):
            runner.before_decode(batch, None, order)
            unpacked = torch.stack(
                (
                    batch.input_ids // model.audio_vocab_size,
                    batch.input_ids % model.audio_vocab_size,
                ),
                dim=1,
            )
            assert unpacked.tolist() == [
                list(request.data.output_rows[-1]) for request in order
            ]
            expected = torch.stack(
                [
                    (
                        model.audio_embed.weight[row[1]]
                        if row[0] == model.config.modality_pad_token_id
                        else model.embed_tokens.weight[row[0]]
                    )
                    for row in unpacked.tolist()
                ]
            )
            torch.testing.assert_close(
                model.rows_to_embeddings(unpacked), expected, rtol=0, atol=0
            )


@NATIVE_CPU
def test_actual_worker_allocates_both_modality_tails_without_rewriting_hf_layers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configuration = MossSpeechConfig(
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        vocab_size=256,
        audio_vocab_size=64,
        max_position_embeddings=16,
        architectures=["MossSpeechForCausalLM"],
    )
    configuration.save_pretrained(tmp_path)
    monkeypatch.setitem(
        CONFIG_MAPPING._extra_content, "moss_speech", MossSpeechConfig
    )  # noqa: leading-underscore  # Transformers registry spelling.
    ensure_moss_speech_config_registered()
    worker_configuration = ModelConfig(
        str(tmp_path), trust_remote_code=False, dtype="bfloat16"
    )
    ModelWorker.apply_arch_override(worker_configuration, "MossSpeechForCausalLM")
    assert worker_configuration.num_attention_layers == 40
    assert worker_configuration.num_hidden_layers == 36
    assert worker_configuration.hf_config.num_hidden_layers == 36
    with native_runtime(torch.device("cpu")):
        model = MossSpeechSGLangModel(configuration)
        weights = checkpoint_weights(configuration)
        assert len(weights) == 446
        model.load_weights(weights)
        assert model.attention_layer_ids() == list(range(40))


@NATIVE_CPU
def test_checkpoint_sentinels_reach_every_independent_parameter() -> None:
    with native_runtime(torch.device("cpu")), torch.no_grad():
        model = tiny_model(torch.device("cpu"))
        weights = dict(checkpoint_weights(model.config))
        model.load_weights(weights.items())
        for parameter, source in (
            (model.embed_tokens.weight, "model.embed_tokens.weight"),
            (model.audio_embed.weight, "model.audio_embed.weight"),
            (model.text_norm.weight, "model.text_norm.weight"),
            (model.audio_norm.weight, "model.audio_norm.weight"),
            (model.text_lm_head.weight, "text_lm_head.weight"),
            (model.audio_lm_head.weight, "audio_lm_head.weight"),
        ):
            torch.testing.assert_close(parameter, weights[source], rtol=0, atol=0)
        for layers, source_block in (
            (model.layers, "shared_block"),
            (model.text_block, "text_block"),
            (model.audio_block, "audio_block"),
        ):
            for layer_index, layer in enumerate(layers):
                prefix = f"model.{source_block}.layers.{layer_index}."
                expected_qkv = torch.cat(
                    [
                        weights[prefix + f"self_attn.{projection}_proj.weight"]
                        for projection in ("q", "k", "v")
                    ]
                )
                expected_gate_up = torch.cat(
                    [
                        weights[prefix + f"mlp.{projection}_proj.weight"]
                        for projection in ("gate", "up")
                    ]
                )
                torch.testing.assert_close(
                    layer.self_attn.qkv_proj.weight, expected_qkv, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    layer.mlp.gate_up_proj.weight, expected_gate_up, rtol=0, atol=0
                )
                for parameter, leaf in (
                    (layer.self_attn.o_proj.weight, "self_attn.o_proj.weight"),
                    (layer.self_attn.q_norm.weight, "self_attn.q_norm.weight"),
                    (layer.self_attn.k_norm.weight, "self_attn.k_norm.weight"),
                    (layer.mlp.down_proj.weight, "mlp.down_proj.weight"),
                    (layer.input_layernorm.weight, "input_layernorm.weight"),
                    (
                        layer.post_attention_layernorm.weight,
                        "post_attention_layernorm.weight",
                    ),
                ):
                    torch.testing.assert_close(
                        parameter, weights[prefix + leaf], rtol=0, atol=0
                    )


@NATIVE_CPU
@pytest.mark.parametrize(
    "defect,message",
    [
        ("missing", "cover"),
        ("unexpected", "Unexpected"),
        ("duplicate", "Duplicate"),
        ("broadcast", "shape"),
        ("shape", "shape"),
    ],
)
def test_incomplete_duplicate_and_malformed_checkpoints_fail(
    defect: str, message: str
) -> None:
    with native_runtime(torch.device("cpu")):
        model = tiny_model(torch.device("cpu"))
        weights = checkpoint_weights(model.config)
        if defect == "missing":
            weights.pop()
        elif defect == "unexpected":
            weights.append(("model.lm_head.weight", torch.zeros(1)))
        elif defect == "duplicate":
            weights.append(weights[0])
        elif defect == "broadcast":
            source_name, tensor = weights[6]
            weights[6] = (source_name, tensor[:1])
        else:
            source_name, tensor = weights[0]
            weights[0] = (
                source_name,
                torch.zeros(tuple(length + 3 for length in tensor.shape)),
            )
        with pytest.raises(RuntimeError, match=message):
            model.load_weights(weights)


def test_bf16_normalization_rounds_residual_and_normalized_value() -> None:
    """Combining FP32 residual addition or weight multiplication changes checkpoint rounding."""
    generator = torch.Generator().manual_seed(921)
    hidden_states = torch.randn(3, 128, generator=generator).bfloat16()
    residual = torch.randn(3, 128, generator=generator).bfloat16()
    original_hidden = hidden_states.clone()
    with torch.random.fork_rng(devices=[]):
        normalization = MossSpeechRMSNorm(128, eps=1e-6).bfloat16()
    with torch.no_grad():
        normalization.weight.copy_(torch.randn(128, generator=generator).bfloat16())
    rounded_residual = hidden_states + residual
    reference = Qwen3RMSNorm(128, eps=1e-6).bfloat16()
    with torch.no_grad():
        reference.weight.copy_(normalization.weight)
    expected = reference(rounded_residual)
    actual, actual_residual = normalization(hidden_states, residual)
    assert torch.equal(actual, expected)
    assert torch.equal(actual_residual, rounded_residual)
    assert torch.equal(hidden_states, original_hidden)


@pytest.mark.parametrize("head_dim,theta", [(8, 10000.0), (128, 1_000_000.0)])
def test_rope_frequencies_keep_cpu_float32_reference_values(
    head_dim: int, theta: float
) -> None:
    frequencies = reference_rope_frequencies(head_dim, theta)
    assert frequencies.device.type == "cpu"
    assert frequencies.dtype == torch.float32
    with torch.device("cpu"):
        reference = Qwen3RotaryEmbedding(
            Qwen3Config(head_dim=head_dim, rope_theta=theta)
        )
    torch.testing.assert_close(frequencies, reference.inv_freq, rtol=0, atol=0)


def assert_native_request_isolation(device: torch.device) -> None:
    with native_runtime(device), torch.no_grad():
        generator = torch.Generator(device=device).manual_seed(183)
        model = tiny_model(device)
        prompt_embeddings = [
            torch.randn((length, 128), generator=generator, device=device)
            for length in (2, 3)
        ]
        feedback_rows = torch.tensor(
            [[3, 17], [model.config.modality_pad_token_id, 4]], device=device
        )
        packed_prefill, packed_backend = prefill(
            model, prompt_embeddings, ["first", "second"]
        )
        packed_decode = decode(
            model, packed_backend, [2, 3], feedback_rows, ["first", "second"]
        )
        for request_index, prompt in enumerate(prompt_embeddings):
            isolated_prefill, isolated_backend = prefill(model, [prompt], ["isolated"])
            isolated_decode = decode(
                model,
                isolated_backend,
                [len(prompt)],
                feedback_rows[request_index : request_index + 1],
                ["isolated"],
            )
            for packed, isolated in (
                (packed_prefill.text_logits, isolated_prefill.text_logits),
                (packed_prefill.audio_logits, isolated_prefill.audio_logits),
                (packed_decode.text_logits, isolated_decode.text_logits),
                (packed_decode.audio_logits, isolated_decode.audio_logits),
            ):
                assert packed is not None and isolated is not None
                torch.testing.assert_close(
                    packed[request_index], isolated[0], rtol=0, atol=0
                )
        first_tail = model.config.num_shared_layers
        second_tail = first_tail + model.config.num_modality_layers
        text_keys = packed_backend.token_to_kv_pool.get_key_buffer(first_tail)
        audio_keys = packed_backend.token_to_kv_pool.get_key_buffer(second_tail)
        assert text_keys.data_ptr() != audio_keys.data_ptr()
        assert torch.count_nonzero(text_keys[1:8]) > 0
        assert torch.count_nonzero(audio_keys[1:8]) > 0
        assert not torch.equal(text_keys[1:8], audio_keys[1:8])


@NATIVE_CPU
def test_native_cpu_prefill_decode_keep_requests_and_tail_kv_independent() -> None:
    assert_native_request_isolation(torch.device("cpu"))


@pytest.mark.accelerator
def test_native_cuda_prefill_decode_keep_requests_and_tail_kv_independent() -> None:
    require_cuda()
    if os.environ.get("SGLANG_USE_CPU_ENGINE") == "1":
        pytest.skip("Run the accelerator lane without the CPU engine override")
    else:
        assert_native_request_isolation(torch.device("cuda", 0))


@NATIVE_CPU
def test_prefill_projects_only_one_hidden_state_per_request() -> None:
    """Prefill allocated vocabulary logits for positions discarded before sampling."""
    with native_runtime(torch.device("cpu")), torch.no_grad():
        model = tiny_model(torch.device("cpu"))
        prompts = [
            model.rows_to_embeddings(torch.tensor([[3, 0], [4, 0], [5, 0]])),
            model.rows_to_embeddings(torch.tensor([[6, 0], [7, 0]])),
        ]
        with patch(
            "torch.nn.functional.linear",
            wraps=torch.nn.functional.linear,
        ) as projection:
            output, _ = prefill(model, prompts, ["long", "short"])
        assert output.text_logits.shape == (2, model.text_vocab_size)
        assert output.audio_logits.shape == (2, model.audio_vocab_size)
        head_calls = [
            call
            for call in projection.call_args_list
            if call.args[1] is model.text_lm_head.weight
            or call.args[1] is model.audio_lm_head.weight
        ]
        assert len(head_calls) == 2
        assert all(call.args[0].shape[0] == len(prompts) for call in head_calls)
