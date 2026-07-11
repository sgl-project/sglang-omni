# SPDX-License-Identifier: Apache-2.0
"""Ming usage accounting tests."""

from __future__ import annotations

from types import SimpleNamespace


class _TensorLike:
    def __init__(self, n: int):
        self._n = n

    def numel(self) -> int:
        return self._n


def test_build_text_usage_counts_tensor_prompt_and_output_ids() -> None:
    from sglang_omni.models.ming_omni.pipeline.usage import build_text_usage

    state = SimpleNamespace(
        prompt={"input_ids": _TensorLike(11)},
        thinker_out={"output_ids": [101, 102, 103]},
    )

    assert build_text_usage(state) == {
        "prompt_tokens": 11,
        "completion_tokens": 3,
        "total_tokens": 14,
    }


def test_build_text_usage_counts_attribute_prompt_input_ids() -> None:
    from sglang_omni.models.ming_omni.io import MingOmniPipelineState
    from sglang_omni.models.ming_omni.pipeline.usage import build_text_usage

    class PromptObject:
        input_ids = _TensorLike(11)

    state = MingOmniPipelineState(
        prompt=PromptObject(),  # type: ignore[arg-type]
        thinker_out={"output_ids": [101, 102, 103]},
    )

    assert build_text_usage(state) == {
        "prompt_tokens": 11,
        "completion_tokens": 3,
        "total_tokens": 14,
    }


def test_build_text_usage_counts_list_prompt_and_explicit_thinker_out() -> None:
    from sglang_omni.models.ming_omni.pipeline.usage import build_text_usage

    state = {"prompt": {"input_ids": [1, 2, 3, 4]}, "thinker_out": {}}
    thinker_out = {"output_ids": [8, 9]}

    assert build_text_usage(state, thinker_out) == {
        "prompt_tokens": 4,
        "completion_tokens": 2,
        "total_tokens": 6,
    }


def test_build_text_usage_defaults_missing_ids_to_zero() -> None:
    from sglang_omni.models.ming_omni.pipeline.usage import build_text_usage

    assert build_text_usage({}) == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
