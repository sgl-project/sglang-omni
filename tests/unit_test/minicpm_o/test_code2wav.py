# SPDX-License-Identifier: Apache-2.0
"""Public MiniCPM-o vocoder contracts: import, checkpoint decode, speaker ref."""

from __future__ import annotations

import base64
import inspect
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from sglang_omni.models.minicpm_o.components.code2wav import (
    SAMPLES_PER_CODEC_TOKEN,
    MiniCPMOCode2Wav,
)
from sglang_omni.models.minicpm_o.config import MiniCPMOSpeechPipelineConfig
from sglang_omni.models.minicpm_o.payload_types import MiniCPMOPipelineState
from sglang_omni.models.minicpm_o.routing import (
    code2wav_reference_audio,
    project_talker_to_code2wav,
)
from sglang_omni.models.minicpm_o.stages import (
    CODE2WAV_BATCH_WAIT_WHEN_IDLE,
    CODE2WAV_MAX_BATCH_SIZE,
    CODE2WAV_MAX_BATCH_WAIT_MS,
    create_code2wav_executor,
    vocode_code2wav_payloads,
)
from sglang_omni.proto import OmniRequest, StagePayload

REPO_ROOT = Path(__file__).resolve().parents[3]


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


def _checkpoint_dir() -> Path | None:
    env = os.environ.get("MINICPMO_CHECKPOINT")
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    candidates = [Path(env)] if env else []
    candidates += [REPO_ROOT / "MiniCPM-o-4_6", REPO_ROOT / "MiniCPM-o-4_5"]
    for hub in (
        hf_home / "hub" / "models--openbmb--MiniCPM-o-4_5" / "snapshots",
        hf_home / "models--openbmb--MiniCPM-o-4_5" / "snapshots",
    ):
        if hub.is_dir():
            candidates.extend(sorted(hub.iterdir(), reverse=True))
    for path in candidates:
        if path is not None and (path / "assets" / "token2wav").is_dir():
            return path
    return None


@pytest.mark.accelerator
def test_native_vocoder_with_checkpoint() -> None:
    checkpoint = _checkpoint_dir()
    if checkpoint is None or not torch.cuda.is_available():
        pytest.skip("Set MINICPMO_CHECKPOINT and provide CUDA for vocoder validation")
    model = MiniCPMOCode2Wav(str(checkpoint), device="cuda:0", float16=False)
    tokens = [1498, 1734, 3732, 3726, 3645]
    output = model(codec_tokens=torch.tensor(tokens))
    waveform = output["waveform"]
    assert output["sample_rate"] == 24000
    assert waveform.dtype == np.float32
    assert waveform.shape == (len(tokens) * SAMPLES_PER_CODEC_TOKEN,)
    assert np.isfinite(waveform).all()
    assert np.max(np.abs(waveform)) > 1e-5
    assert np.max(np.abs(waveform)) <= 0.99


@pytest.mark.accelerator
def test_native_vocoder_batch_matches_single_request_shapes() -> None:
    checkpoint = _checkpoint_dir()
    if checkpoint is None or not torch.cuda.is_available():
        pytest.skip("Set MINICPMO_CHECKPOINT and provide CUDA for vocoder validation")
    model = MiniCPMOCode2Wav(str(checkpoint), device="cuda:0", float16=False)
    tokens_a = [1498, 1734, 3732, 3726, 3645]
    tokens_b = tokens_a + [3645, 3726]
    equal_rows = model.vocode_many([tokens_a, tokens_a], None)
    cosine = float(
        np.dot(equal_rows[0], equal_rows[1])
        / (np.linalg.norm(equal_rows[0]) * np.linalg.norm(equal_rows[1]))
    )
    assert cosine > 0.99
    batched = model.vocode_many([tokens_a, tokens_b], None)
    single_a = model.vocode(tokens_a, None)
    single_b = model.vocode(tokens_b, None)
    assert (
        batched[0].shape == single_a.shape == (len(tokens_a) * SAMPLES_PER_CODEC_TOKEN,)
    )
    assert (
        batched[1].shape == single_b.shape == (len(tokens_b) * SAMPLES_PER_CODEC_TOKEN,)
    )
    assert all(np.isfinite(wave).all() for wave in (*batched, single_a, single_b))


def _data_uri(audio: bytes) -> str:
    return "data:audio/wav;base64," + base64.b64encode(audio).decode("ascii")


def _payload(
    *,
    request_id: str = "test",
    tokens: list[int] | None = None,
    params: dict[str, object] | None = None,
    metadata: dict[str, object] | None = None,
) -> StagePayload:
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs=None, params=params or {}, metadata=metadata or {}),
        data=MiniCPMOPipelineState(
            engine_outputs={"talker": {"codec_tokens": torch.tensor(tokens or [1, 2])}}
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


def test_speech_pipeline_enables_code2wav_batching_by_default() -> None:
    config = MiniCPMOSpeechPipelineConfig(model_path="unused")
    code2wav = next(stage for stage in config.stages if stage.name == "code2wav")
    assert code2wav.factory.max_batch_size == 4
    assert code2wav.factory.max_batch_wait_ms == 10.0
    assert code2wav.factory.batch_wait_when_idle is False


def test_code2wav_factory_defaults_enable_batching() -> None:
    signature = inspect.signature(create_code2wav_executor)
    assert signature.parameters["max_batch_size"].default == CODE2WAV_MAX_BATCH_SIZE
    assert (
        signature.parameters["max_batch_wait_ms"].default == CODE2WAV_MAX_BATCH_WAIT_MS
    )
    assert (
        signature.parameters["batch_wait_when_idle"].default
        is CODE2WAV_BATCH_WAIT_WHEN_IDLE
    )


def test_vocode_many_slices_waveforms_to_token_lengths() -> None:
    class FakeFlow:
        up_rate = 2

        def inference(
            self,
            speech_tokens: torch.Tensor,
            speech_tokens_lens: torch.Tensor,
            *args: object,
        ) -> torch.Tensor:
            frames = speech_tokens.shape[1] * self.up_rate
            return torch.zeros(speech_tokens.shape[0], 80, frames)

    class FakeHiFT:
        def __call__(self, speech_feat: torch.Tensor) -> tuple[torch.Tensor, None]:
            samples = speech_feat.shape[-1] * (SAMPLES_PER_CODEC_TOKEN // 2)
            wav = speech_feat.new_ones(speech_feat.shape[0], 1, samples)
            return wav, None

    model = MiniCPMOCode2Wav.__new__(MiniCPMOCode2Wav)
    model.token2wav = SimpleNamespace(
        device=torch.device("cpu"),
        float16=False,
        n_timesteps=10,
        flow=FakeFlow(),
        hift=FakeHiFT(),
    )
    model.speaker_prompt = lambda prompt_wav: (
        torch.zeros(1, 1, dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        torch.zeros(1, 4),
        torch.zeros(1, 1, 80),
    )
    waveforms = model.vocode_many([[1, 2], [3, 4, 5]], b"ref")
    assert [wave.shape for wave in waveforms] == [
        (2 * SAMPLES_PER_CODEC_TOKEN,),
        (3 * SAMPLES_PER_CODEC_TOKEN,),
    ]


def test_vocode_many_rejects_empty_sequences() -> None:
    model = MiniCPMOCode2Wav.__new__(MiniCPMOCode2Wav)
    assert model.vocode_many([], b"ref") == []
    with pytest.raises(ValueError, match="non-empty"):
        model.vocode_many([[1], []], b"ref")


def _fake_code2wav_model() -> MagicMock:
    fake = MagicMock()
    fake.sample_rate = 24000
    fake.resolve_prompt_wav.side_effect = lambda reference: (
        b"default" if reference is None else reference
    )
    fake.vocode_many.side_effect = lambda sequences, reference: [
        np.full(
            len(tokens) * SAMPLES_PER_CODEC_TOKEN,
            float(len(tokens)),
            dtype=np.float32,
        )
        for tokens in sequences
    ]
    return fake


def test_vocode_payloads_uses_one_batch_path() -> None:
    fake = _fake_code2wav_model()
    output = vocode_code2wav_payloads(fake, [_payload(tokens=[7, 8, 9])])[0]
    fake.vocode_many.assert_called_once_with([[7, 8, 9]], b"default")
    assert output.data["sample_rate"] == 24000
    assert output.data["audio_waveform_shape"] == [3 * SAMPLES_PER_CODEC_TOKEN]


def test_vocode_payloads_groups_by_resolved_reference() -> None:
    fake = _fake_code2wav_model()
    outputs = vocode_code2wav_payloads(
        fake,
        [
            _payload(
                request_id="a", tokens=[1, 2], params={"ref_audio": _data_uri(b"spk-a")}
            ),
            _payload(
                request_id="b", tokens=[3], params={"ref_audio": _data_uri(b"spk-b")}
            ),
            _payload(
                request_id="c",
                tokens=[4, 5, 6],
                params={"ref_audio": _data_uri(b"spk-a")},
            ),
        ],
    )
    assert fake.vocode_many.call_count == 2
    batched_calls = {
        call.args[1]: call.args[0] for call in fake.vocode_many.call_args_list
    }
    assert batched_calls[b"spk-a"] == [[1, 2], [4, 5, 6]]
    assert batched_calls[b"spk-b"] == [[3]]
    assert [out.data["audio_waveform_shape"][0] for out in outputs] == [
        2 * SAMPLES_PER_CODEC_TOKEN,
        SAMPLES_PER_CODEC_TOKEN,
        3 * SAMPLES_PER_CODEC_TOKEN,
    ]


def test_vocode_payloads_resolves_default_reference_before_grouping() -> None:
    fake = _fake_code2wav_model()
    vocode_code2wav_payloads(
        fake,
        [_payload(request_id="a", tokens=[1]), _payload(request_id="b", tokens=[2, 3])],
    )
    fake.vocode_many.assert_called_once_with([[1], [2, 3]], b"default")
