# SPDX-License-Identifier: Apache-2.0
"""Public MiniCPM-o vocoder contracts: import, checkpoint decode, speaker ref."""

from __future__ import annotations

import base64
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from sglang_omni.models.minicpm_o.components.code2wav import MiniCPMOCode2Wav
from sglang_omni.models.minicpm_o.payload_types import MiniCPMOPipelineState
from sglang_omni.models.minicpm_o.routing import (
    code2wav_reference_audio,
    project_talker_to_code2wav,
)
from sglang_omni.proto import OmniRequest, StagePayload


def test_native_vocoder_import_does_not_require_legacy_packages() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys

class BlockLegacy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {
            "stepaudio2", "s3tokenizer", "minicpmo", "hyperpyyaml"
        }:
            raise ImportError(f"Legacy dependency requested: {fullname}")

sys.meta_path.insert(0, BlockLegacy())
from sglang_omni.models.minicpm_o.components.token2wav.vocoder import Token2Wav
""",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.accelerator
def test_native_vocoder_with_checkpoint() -> None:
    checkpoint = os.environ.get("MINICPMO_CHECKPOINT")
    if not checkpoint or not torch.cuda.is_available():
        pytest.skip("Set MINICPMO_CHECKPOINT and provide CUDA for vocoder validation")
    model = MiniCPMOCode2Wav(checkpoint, device="cuda:0", float16=False)
    output = model(codec_tokens=torch.tensor([1498, 1734, 3732, 3726, 3645]))
    waveform = output["waveform"]
    assert output["sample_rate"] == 24000
    assert waveform.dtype == np.float32
    assert waveform.shape == (4800,)
    assert np.isfinite(waveform).all()
    assert np.max(np.abs(waveform)) > 1e-5
    assert np.max(np.abs(waveform)) <= 0.99


def _data_uri(audio: bytes) -> str:
    return "data:audio/wav;base64," + base64.b64encode(audio).decode("ascii")


def _payload(
    *, params: dict | None = None, metadata: dict | None = None
) -> StagePayload:
    return StagePayload(
        request_id="test",
        request=OmniRequest(inputs=None, params=params or {}, metadata=metadata or {}),
        data=MiniCPMOPipelineState(
            engine_outputs={"talker": {"codec_tokens": torch.tensor([1, 2])}}
        ).to_dict(),
    )


def test_chat_api_forwards_reference_to_vocoder() -> None:
    from sglang_omni.client.client import _build_params
    from sglang_omni.serve.openai_api import (
        ChatCompletionRequest,
        _build_chat_generate_request,
    )

    reference = _data_uri(b"reference")
    request = ChatCompletionRequest(
        model="minicpm-o",
        messages=[{"role": "user", "content": "Hello"}],
        modalities=["text", "audio"],
        audio={"format": "wav", "ref_audio": reference},
    )
    generate_request = _build_chat_generate_request(request)
    payload = _payload(
        params=_build_params(generate_request), metadata=generate_request.metadata
    )
    assert code2wav_reference_audio(project_talker_to_code2wav(payload)) == b"reference"


def test_invalid_reference_does_not_silently_use_default() -> None:
    payload = _payload(params={"ref_audio": "/tmp/ref.wav"})
    with pytest.raises(ValueError, match="inline audio"):
        code2wav_reference_audio(payload)
