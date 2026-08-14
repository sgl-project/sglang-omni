# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
import inspect
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

try:
    _has_sglang = importlib.util.find_spec("sglang") is not None
except ValueError:
    _has_sglang = False

if not _has_sglang:
    # Keep this focused unit test runnable without the optional SGLang runtime.
    # Integration tests exercise the same subclass against the pinned SGLang
    # dependency in CI.
    module_names = (
        "sglang",
        "sglang.srt",
        "sglang.srt.dllm",
        "sglang.srt.dllm.algorithm",
        "sglang.srt.dllm.algorithm.base",
        "sglang.srt.dllm.config",
        "sglang.srt.model_executor",
        "sglang.srt.model_executor.forward_batch_info",
    )
    for module_name in module_names:
        sys.modules[module_name] = ModuleType(module_name)

    class _DllmAlgorithm:
        def __init__(self, config: SimpleNamespace) -> None:
            self.block_size = config.block_size
            self.mask_id = config.mask_id
            self.fdfo = config.first_done_first_out_mode

        def run(self, model_runner, forward_batch, algo_states=None):
            del algo_states
            return model_runner.forward(forward_batch)

    sys.modules["sglang.srt.dllm.algorithm.base"].DllmAlgorithm = _DllmAlgorithm
    sys.modules["sglang.srt.dllm.config"].DllmConfig = object
    sys.modules["sglang.srt.model_executor.forward_batch_info"].ForwardBatch = object

from sglang_omni.models.llada2_uni.algorithm.low_confidence_cfg import (
    LowConfidenceCFG,
)


def _algorithm(*, block_size: int = 2, mask_id: int = 99) -> LowConfidenceCFG:
    return LowConfidenceCFG(
        SimpleNamespace(
            block_size=block_size,
            mask_id=mask_id,
            first_done_first_out_mode=False,
            algorithm_config={"threshold": 2.0},
        )
    )


def test_cfg_step_uses_group_metadata_and_updates_all_rows_in_one_batch() -> None:
    algorithm = _algorithm()
    forward_batch = SimpleNamespace(
        batch_size=2,
        input_ids=torch.tensor([99, 99, 99, 99]),
        omni_dllm_group=SimpleNamespace(
            roles=("conditional", "unconditional"),
            algorithm_args={
                "cfg_scale": 2.0,
                "cfg_rescale": 0.0,
                "force_image_only": True,
                "image_token_offset": 3,
            },
        ),
    )
    # Conditional logits prefer image token 4, while unconditional logits
    # prefer text token 1. Guidance must choose token 4 and image-only masking
    # must prevent any token below the checkpoint-provided offset.
    logits = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 5.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 4.0, 0.0],
            [0.0, 5.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 4.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    done = algorithm.step(forward_batch, logits, [None, None])

    assert done == [False, False]
    updated = forward_batch.input_ids.view(2, 2)
    assert torch.equal(updated[0], updated[1])
    assert bool(torch.all(updated >= 3))


def test_cfg_step_rejects_malformed_group_roles() -> None:
    algorithm = _algorithm()
    forward_batch = SimpleNamespace(
        batch_size=2,
        input_ids=torch.tensor([99, 99, 99, 99]),
        omni_dllm_group=SimpleNamespace(
            roles=("conditional", "conditional"),
            algorithm_args={"cfg_scale": 4.0},
        ),
    )

    with pytest.raises(RuntimeError, match="roles"):
        algorithm.step(forward_batch, torch.zeros(4, 6), [None, None])


def test_cfg_step_does_not_validate_alignment_with_cuda_scalar_sync() -> None:
    source = inspect.getsource(LowConfidenceCFG._step_cfg)
    before_required_done_result = source.rsplit("return", maxsplit=1)[0]

    assert "torch.equal" not in source
    assert ".item(" not in source
    assert ".tolist(" not in before_required_done_result


def test_independent_step_applies_batched_native_image_vocab_boundaries() -> None:
    algorithm = _algorithm(block_size=1)
    forward_batch = SimpleNamespace(
        batch_size=2,
        input_ids=torch.tensor([99, 99]),
        omni_dllm_group=None,
        omni_dllm_image_token_offsets=torch.tensor([3, 0]),
    )
    logits = torch.tensor(
        [
            [0.0, 9.0, 0.0, 0.0, 8.0],
            [0.0, 9.0, 0.0, 0.0, 8.0],
        ]
    )

    algorithm.step(forward_batch, logits, [None, None])

    assert forward_batch.input_ids.tolist() == [4, 1]


def test_independent_image_step_allows_only_explicit_low_vocab_stop_tokens() -> None:
    algorithm = _algorithm(block_size=1)
    forward_batch = SimpleNamespace(
        batch_size=1,
        input_ids=torch.tensor([99]),
        omni_dllm_group=None,
        omni_dllm_image_token_offsets=torch.tensor([3]),
        omni_dllm_allowed_stop_token_ids=((1,),),
    )
    logits = torch.tensor([[0.0, 10.0, 9.0, 0.0, 8.0]])

    algorithm.step(forward_batch, logits, [None])

    assert forward_batch.input_ids.tolist() == [1]


def test_grouped_image_step_allows_only_explicit_low_vocab_stop_tokens() -> None:
    algorithm = _algorithm(block_size=1)
    forward_batch = SimpleNamespace(
        batch_size=2,
        input_ids=torch.tensor([99, 99]),
        omni_dllm_group=SimpleNamespace(
            roles=("conditional", "unconditional"),
            algorithm_args={
                "cfg_scale": 1.0,
                "cfg_rescale": 0.0,
                "force_image_only": True,
                "image_token_offset": 3,
                "allowed_stop_token_ids": (1,),
            },
        ),
    )
    logits = torch.tensor(
        [
            [0.0, 10.0, 9.0, 0.0, 8.0],
            [0.0, 10.0, 9.0, 0.0, 8.0],
        ]
    )

    algorithm.step(forward_batch, logits, [None, None])

    assert forward_batch.input_ids.tolist() == [1, 1]


def test_in_query_left_pad_temporarily_disables_public_cuda_graph_runner() -> None:
    algorithm = _algorithm()
    graph_runner = object()
    observed = SimpleNamespace(context_set=False)

    class _AttentionBackend:
        def set_cfg_runtime_forward_batch(self, forward_batch) -> None:
            observed.context_set = forward_batch is not None

    attention_backend = _AttentionBackend()
    forward_batch = SimpleNamespace(
        batch_size=2,
        input_ids=torch.tensor([99, 99, 99, 99]),
        dllm_left_pad_lens_cpu=(0, 40),
        extend_prefix_lens_cpu=[32, 32],
        extend_seq_lens_cpu=[2, 2],
        omni_dllm_group=SimpleNamespace(
            roles=("conditional", "unconditional"),
            algorithm_args={"cfg_scale": 4.0},
        ),
    )

    def forward(observed_batch):
        assert observed_batch is forward_batch
        assert model_runner.decode_cuda_graph_runner is None
        assert observed.context_set
        return "eager-result"

    model_runner = SimpleNamespace(
        decode_cuda_graph_runner=graph_runner,
        attn_backend=attention_backend,
        decode_attn_backend=attention_backend,
        forward=forward,
    )

    assert algorithm.run(model_runner, forward_batch) == "eager-result"
    assert model_runner.decode_cuda_graph_runner is graph_runner
    assert not observed.context_set


def test_grouped_prompt_prefill_does_not_denoise_left_pad_masks() -> None:
    algorithm = _algorithm()
    graph_runner = object()
    observed = SimpleNamespace(context_set=False)

    class _AttentionBackend:
        def set_cfg_runtime_forward_batch(self, forward_batch) -> None:
            observed.context_set = forward_batch is not None

    attention_backend = _AttentionBackend()
    original_input_ids = torch.tensor([99, 10, 99, 99])
    forward_batch = SimpleNamespace(
        batch_size=2,
        input_ids=original_input_ids.clone(),
        dllm_left_pad_lens_cpu=(1, 2),
        extend_prefix_lens_cpu=[0, 0],
        extend_seq_lens_cpu=[2, 2],
        omni_dllm_group=SimpleNamespace(
            roles=("conditional", "unconditional"),
            algorithm_args={"cfg_scale": 4.0},
        ),
        omni_dllm_group_is_prefill=True,
    )
    calls = []

    def forward(observed_batch, pp_proxy_tensors=None):
        assert model_runner.decode_cuda_graph_runner is None
        assert observed.context_set
        calls.append((observed_batch, pp_proxy_tensors))
        return SimpleNamespace(logits_output="prefill-logits", can_run_graph=False)

    model_runner = SimpleNamespace(
        decode_cuda_graph_runner=graph_runner,
        attn_backend=attention_backend,
        decode_attn_backend=attention_backend,
        forward=forward,
    )

    result = algorithm.run(model_runner, forward_batch)

    assert result == (
        "prefill-logits",
        [[], []],
        None,
        None,
        False,
    )
    assert calls == [(forward_batch, None)]
    assert torch.equal(forward_batch.input_ids, original_input_ids)
    assert model_runner.decode_cuda_graph_runner is graph_runner
    assert not observed.context_set


def test_two_way_cfg_has_a_single_pytorch_guidance_path() -> None:
    algorithm = _algorithm(block_size=1)
    forward_batch = SimpleNamespace(
        batch_size=2,
        input_ids=torch.tensor([99, 99]),
        omni_dllm_group=SimpleNamespace(
            roles=("conditional", "unconditional"),
            algorithm_args={
                "cfg_scale": 2.0,
                "cfg_rescale": 0.0,
                "force_image_only": True,
                "image_token_offset": 3,
            },
        ),
    )
    logits = torch.tensor(
        [
            [0.0, 0.0, 0.0, 4.0, 1.0],
            [0.0, 9.0, 0.0, 0.0, 5.0],
        ]
    )

    algorithm.step(forward_batch, logits, states=[])

    assert forward_batch.input_ids.tolist() == [3, 3]
