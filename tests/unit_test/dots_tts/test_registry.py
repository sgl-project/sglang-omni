# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations


def test_dots_model_and_runner_registered() -> None:
    from sglang.srt.models.registry import ModelRegistry

    from sglang_omni.model_runner.sglang_model_runner import SGLModelRunner
    from sglang_omni.models.dots_tts.sglang_model import DotsTTSSGLangModel

    SGLModelRunner._register_omni_model(SGLModelRunner)

    assert ModelRegistry.models["DotsTTSForConditionalGeneration"] is DotsTTSSGLangModel
