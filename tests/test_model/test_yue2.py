# SPDX-License-Identifier: Apache-2.0
"""Real-checkpoint generation and cancellation checks for one CUDA device."""

import os
import threading

import pytest
import torch

from sglang_omni.models.yue2.request import build_request
from sglang_omni.models.yue2.runtime import Yue2Runtime
from sglang_omni.proto.request import OmniRequest, StagePayload

pytestmark = [
    pytest.mark.accelerator,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


@pytest.fixture(scope="module")
def runtime() -> Yue2Runtime:
    return Yue2Runtime.from_pretrained(
        os.environ.get("YUE2_MODEL_PATH", "m-a-p/YuE2-3B"),
        os.environ.get("YUE2_VAE_PATH", "m-a-p/YuE2-Vae"),
        torch.device("cuda:0"),
        vae_core_frames=1024,
        vae_halo_frames=16,
    )


@pytest.mark.parametrize(
    "cot,abc,cfg_scale",
    [
        ("full", None, 1.0),
        ("melody", None, 1.5),
        ("off", None, None),
        ("melody", "X:1\nT:Morning\nM:4/4\nL:1/4\nK:C\nC D E F |", 1.5),
    ],
)
def test_real_checkpoint_generation(
    runtime: Yue2Runtime,
    cot: str,
    abc: str | None,
    cfg_scale: float | None,
) -> None:
    message = StagePayload(
        request_id="song",
        data={},
        request=OmniRequest(
            inputs="[Verse]\nMorning light is shining through",
            metadata={
                "tts_params": {
                    "instructions": "Gentle acoustic pop, warm female vocals",
                    "seed": 42,
                }
            },
            params={
                "max_new_tokens": 16,
                "stage_params": {
                    "synthesis": {
                        "cot": cot,
                        "abc": abc,
                        "cfg_scale": cfg_scale,
                        "abc_max_tokens": 8,
                        "ode_steps": 2,
                    }
                },
            },
        ),
    )
    request = build_request(message)
    song = runtime.generate(request, lambda: False)
    assert song.audio.shape[0] == 2 and song.audio.shape[1] > 0
    assert song.audio.dtype == torch.float32 and torch.isfinite(song.audio).all()
    assert song.sample_rate == 48000
    assert len(song.codec_tokens) == 16 and song.finish_reason == "length"
    aborted = threading.Event()
    aborted.set()
    with pytest.raises(InterruptedError):
        runtime.generate(request, aborted.is_set)
    aborted.clear()
    recovered = runtime.generate(request, aborted.is_set)
    assert recovered.abc_tokens == song.abc_tokens
    assert recovered.codec_tokens == song.codec_tokens
    torch.testing.assert_close(recovered.audio, song.audio, rtol=0, atol=0)
