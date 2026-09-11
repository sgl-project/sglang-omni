from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sglang_omni.models.minicpm_o.thinker_model_runner import (
    MiniCPMOThinkerModelRunner,
)
from sglang_omni.scheduling.sglang_backend import SGLangOutputProcessor


@pytest.mark.parametrize("speech_enabled", [False, True])
@pytest.mark.parametrize("return_hidden_states", [False, True])
@pytest.mark.parametrize("phase", ["prefill", "decode"])
@pytest.mark.parametrize(
    "modalities", [[["text"]], [["text", "audio"]], [["text"], ["text", "audio"]]]
)
def test_capture_hidden_mode_matches_deployment(
    speech_enabled, return_hidden_states, phase, modalities
):
    from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

    model = SimpleNamespace(
        thinker=SimpleNamespace(model=SimpleNamespace(embed_tokens=object()))
    )
    worker = SimpleNamespace(
        gpu_id=0,
        model_runner=SimpleNamespace(
            model=model,
            server_args=SimpleNamespace(
                enable_return_hidden_states=return_hidden_states
            ),
        ),
    )
    should_emit_hidden = Mock(side_effect=AssertionError("must not gate per request"))
    output_processor = SGLangOutputProcessor(
        capture_hidden=speech_enabled, should_emit_hidden=should_emit_hidden
    )
    runner = MiniCPMOThinkerModelRunner(worker, output_processor)
    requests = [SimpleNamespace(modalities=values) for values in modalities]
    expected = (
        CaptureHiddenMode.FULL
        if speech_enabled or return_hidden_states
        else CaptureHiddenMode.NULL
    )

    requested_mode = getattr(runner, f"requested_capture_hidden_mode_{phase}")

    assert requested_mode(SimpleNamespace(), requests) == expected
    should_emit_hidden.assert_not_called()
