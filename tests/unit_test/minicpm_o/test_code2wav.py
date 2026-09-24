# SPDX-License-Identifier: Apache-2.0
"""Public MiniCPM-o vocoder contracts: import, checkpoint decode, speaker ref."""

from __future__ import annotations

import base64
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf
import torch

from sglang_omni.models.minicpm_o.components.code2wav import (
    SAMPLES_PER_CODEC_TOKEN,
    MiniCPMOCode2Wav,
)
from sglang_omni.models.minicpm_o.components.token2wav.dit import DiT, TimestepEmbedder
from sglang_omni.models.minicpm_o.components.token2wav.flow import CausalConditionalCFM
from sglang_omni.models.minicpm_o.components.token2wav.flow_cuda_graph import (
    FlowCudaGraphRunner,
)
from sglang_omni.models.minicpm_o.config import MiniCPMOSpeechPipelineConfig
from sglang_omni.models.minicpm_o.payload_types import MiniCPMOPipelineState
from sglang_omni.models.minicpm_o.routing import (
    code2wav_reference_audio,
    project_talker_to_code2wav,
)
from sglang_omni.models.minicpm_o.stages import vocode_code2wav_payloads
from sglang_omni.proto import OmniRequest, StagePayload

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_timestep_frequencies_preserve_input_precision(dtype: torch.dtype) -> None:
    embedder = TimestepEmbedder(16).to(dtype=dtype)
    for input_dtype in (torch.float32, dtype):
        t = torch.tensor([0.1, 500.0, 1000.0], dtype=input_dtype)
        frequencies = torch.exp(-math.log(10000) * torch.arange(128) / 128).to(t)
        phases = t[:, None] * frequencies[None]
        expected = torch.cat([phases.cos(), phases.sin()], dim=-1)
        torch.testing.assert_close(
            embedder.timestep_embedding(t), expected, atol=0, rtol=0
        )


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_whole_flow_graph_matches_eager(dtype: torch.dtype) -> None:
    torch.manual_seed(42)
    estimator = DiT(16, 4, depth=2, num_heads=2, head_dim=8, hidden_size=16)
    for parameter in estimator.parameters():
        torch.nn.init.normal_(parameter, std=0.1)
    decoder = CausalConditionalCFM(estimator).to(device="cuda", dtype=dtype).eval()
    runner = FlowCudaGraphRunner(
        decoder,
        capture_shapes=((1, 32), (2, 32)),
        frame_bucket=16,
        n_timesteps=10,
        conditioning_dtype=torch.float32,
    )
    runner.capture_all()
    assert runner.select(1, 17) is not None
    assert runner.select(3, 17) is None
    assert runner.select(1, 33) is None
    previous = None
    saved = None
    for batch_size, frames in [(1, 32), (2, 27), (1, 17), (2, 31)]:
        mu = torch.randn(batch_size, 4, frames, device="cuda", dtype=torch.float32)
        mask = torch.ones(batch_size, 1, frames, device="cuda", dtype=torch.float32)
        mask[-1, :, frames // 2 :] = 0
        spks = torch.randn(batch_size, 4, device="cuda", dtype=dtype)
        cond = torch.randn_like(mu)
        noise = torch.randn_like(mu, dtype=dtype)
        t_span = 1 - torch.cos(
            torch.linspace(0, 1, 11, device="cuda", dtype=torch.float32)
            * 0.5
            * torch.pi
        )
        with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
            expected = decoder.solve_euler(noise, t_span, mu, mask, spks, cond)
            actual = runner.run(noise, t_span, mu, mask, spks, cond)
        assert actual is not None and actual.shape == expected.shape
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
        if previous is not None:
            torch.testing.assert_close(previous, saved, atol=0, rtol=0)
        previous, saved = actual, actual.clone()
    assert runner.run(noise, t_span[:-1], mu, mask, spks, cond) is None
    for steps in (10, 5):
        with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
            expected = decoder(mu, mask, spks, cond, n_timesteps=steps, temperature=0.5)
            decoder.cuda_graph_runner = runner
            actual = decoder(mu, mask, spks, cond, n_timesteps=steps, temperature=0.5)
            decoder.cuda_graph_runner = None
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
    with pytest.raises(RuntimeError, match="startup"):
        runner.capture_all()


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_flow_graph_capture_failure_stays_eager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decoder = (
        CausalConditionalCFM(
            DiT(16, 4, depth=1, num_heads=2, head_dim=8, hidden_size=16)
        )
        .cuda()
        .eval()
    )
    runner = FlowCudaGraphRunner(
        decoder,
        capture_shapes=((1, 16), (2, 16)),
        frame_bucket=16,
        n_timesteps=10,
        conditioning_dtype=torch.float32,
    )
    solve_euler = decoder.solve_euler
    calls = 0

    def fail_second_capture(
        x: torch.Tensor,
        t_span: torch.Tensor,
        mu: torch.Tensor,
        mask: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        nonlocal calls
        output = solve_euler(x, t_span, mu, mask, spks, cond)
        if torch.cuda.is_current_stream_capturing():
            calls += 1
            if calls == 2:
                raise RuntimeError("injected active capture failure")
        return output

    monkeypatch.setattr(decoder, "solve_euler", fail_second_capture)
    runner.capture_all()
    assert calls == 2
    assert runner.select(1, 16) is None
    assert runner.select(2, 16) is None
    mu = torch.zeros(1, 4, 16, device="cuda")
    mask = torch.ones(1, 1, 16, device="cuda")
    spks = torch.zeros(1, 4, device="cuda")
    expected = decoder(mu, mask, spks, mu)
    decoder.cuda_graph_runner = runner
    torch.testing.assert_close(decoder(mu, mask, spks, mu), expected)


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
    model = MiniCPMOCode2Wav(str(checkpoint), device="cuda:0")
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
    model = MiniCPMOCode2Wav(str(checkpoint), device="cuda:0")
    tokens_a = [1498, 1734, 3732, 3726, 3645]
    tokens_b = tokens_a + [3645, 3726]
    batched = model.vocode([tokens_a, tokens_b], None)
    single_a = model.vocode([tokens_a], None)[0]
    single_b = model.vocode([tokens_b], None)[0]
    assert (
        batched[0].shape == single_a.shape == (len(tokens_a) * SAMPLES_PER_CODEC_TOKEN,)
    )
    assert (
        batched[1].shape == single_b.shape == (len(tokens_b) * SAMPLES_PER_CODEC_TOKEN,)
    )
    assert all(np.isfinite(wave).all() for wave in (*batched, single_a, single_b))


@pytest.mark.accelerator
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_native_vocoder_graph_matches_eager(
    dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint = _checkpoint_dir()
    if checkpoint is None or not torch.cuda.is_available():
        pytest.skip("Set MINICPMO_CHECKPOINT and provide CUDA for vocoder validation")
    eager = MiniCPMOCode2Wav(str(checkpoint), device="cuda:0", dtype=dtype)
    prompt_tokens, _, _, prompt_mels = eager.speaker_prompt(None)
    reference_tokens = prompt_tokens.flatten().tolist()
    cases = {
        "b1_short": [26],
        "b1_long": [67],
        "b4_mixed": [25, 33, 49, 67],
        "b8_mixed": [25, 29, 33, 37, 41, 49, 57, 67],
    }
    quantum = 16
    capture_shapes = tuple(
        (
            len(lengths),
            (
                prompt_mels.shape[1]
                + max(lengths) * eager.token2wav.flow.up_rate
                + quantum
                - 1
            )
            // quantum
            * quantum,
        )
        for lengths in cases.values()
    )
    graphed = MiniCPMOCode2Wav(
        str(checkpoint),
        device="cuda:0",
        dtype=dtype,
        flow_cuda_graph_capture_shapes=capture_shapes,
        flow_cuda_graph_frame_bucket=quantum,
    )
    decoder = graphed.token2wav.flow.decoder
    decoder.rand_noise.copy_(eager.token2wav.flow.decoder.rand_noise)
    eager_solver = MagicMock(wraps=decoder.solve_euler)
    monkeypatch.setattr(decoder, "solve_euler", eager_solver)
    mel_features: list[torch.Tensor] = []

    def record_mel(
        module: torch.nn.Module,
        args: tuple[torch.Tensor, ...],
        kwargs: dict[str, torch.Tensor],
    ) -> None:
        mel_features.append(kwargs["speech_feat"].clone())

    eager.token2wav.hift.register_forward_pre_hook(record_mel, with_kwargs=True)
    graphed.token2wav.hift.register_forward_pre_hook(record_mel, with_kwargs=True)
    results = []
    cases["batch_miss"] = [25, 33, 49]
    cases["length_miss"] = [97]
    for name, lengths in cases.items():
        token_sequences = [
            (
                reference_tokens
                * ((length + len(reference_tokens) - 1) // len(reference_tokens))
            )[:length]
            for length in lengths
        ]
        torch.manual_seed(1234)
        mel_features.clear()
        expected = eager.vocode(token_sequences, None)
        expected_mel_count = len(mel_features)
        torch.manual_seed(1234)
        eager_solver.reset_mock()
        actual = graphed.vocode(token_sequences, None)
        assert eager_solver.call_count == (1 if name.endswith("miss") else 0)
        assert len(actual) == len(lengths)
        assert len(mel_features) == 2 * expected_mel_count
        mel_bit_exact = True
        max_mel_error = 0.0
        for reference_mel, actual_mel in zip(
            mel_features[:expected_mel_count],
            mel_features[expected_mel_count:],
            strict=True,
        ):
            torch.testing.assert_close(actual_mel, reference_mel, atol=2e-3, rtol=2e-3)
            mel_bit_exact = mel_bit_exact and torch.equal(actual_mel, reference_mel)
            max_mel_error = max(
                max_mel_error, (actual_mel - reference_mel).abs().max().item()
            )
        max_error = 0.0
        max_relative_rmse = 0.0
        for index, (wave, reference, length) in enumerate(
            zip(actual, expected, lengths, strict=True)
        ):
            assert wave.shape == reference.shape == (length * SAMPLES_PER_CODEC_TOKEN,)
            assert wave.dtype == np.float32 and np.isfinite(wave).all()
            assert 1e-5 < np.max(np.abs(wave)) <= 0.99
            # note (Codex): Bound waveform variation below -60 dB after checking mel equivalence.
            relative_rmse = float(
                np.linalg.norm(wave - reference) / np.linalg.norm(reference)
            )
            assert relative_rmse < 1e-3
            np.testing.assert_allclose(wave, reference, atol=1e-3, rtol=0)
            max_relative_rmse = max(max_relative_rmse, relative_rmse)
            max_error = max(max_error, float(np.max(np.abs(wave - reference))))
            sf.write(
                tmp_path / f"{name}_{index}_graph.wav", wave, 24000, subtype="FLOAT"
            )
            sf.write(
                tmp_path / f"{name}_{index}_eager.wav",
                reference,
                24000,
                subtype="FLOAT",
            )
        results.append(
            {
                "case": name,
                "token_lengths": lengths,
                "eager_solver_calls": eager_solver.call_count,
                "max_abs_waveform_error": max_error,
                "max_relative_waveform_rmse": max_relative_rmse,
                "mel_bit_exact": mel_bit_exact,
                "max_abs_mel_error": max_mel_error,
            }
        )
    (tmp_path / "results.json").write_text(json.dumps(results, indent=2) + "\n")


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
    from sglang_omni.client.client import build_params
    from sglang_omni.serve.openai_api import (
        ChatCompletionRequest,
        build_chat_generate_request,
    )

    reference = _data_uri(b"reference")
    request = ChatCompletionRequest(
        model="minicpm-o",
        messages=[{"role": "user", "content": "Hello"}],
        modalities=["text", "audio"],
        audio={"format": "wav", "ref_audio": reference},
    )
    generate_request = build_chat_generate_request(request)
    payload = _payload(
        params=build_params(generate_request), metadata=generate_request.metadata
    )
    assert code2wav_reference_audio(project_talker_to_code2wav(payload)) == b"reference"


def test_invalid_reference_does_not_silently_use_default() -> None:
    payload = _payload(params={"ref_audio": "/tmp/ref.wav"})
    with pytest.raises(ValueError, match="inline audio"):
        code2wav_reference_audio(payload)


def test_speech_pipeline_enables_code2wav_batching_by_default() -> None:
    config = MiniCPMOSpeechPipelineConfig(model_path="unused")
    code2wav = next(stage for stage in config.stages if stage.name == "code2wav")
    assert code2wav.factory.max_batch_size == 8
    assert code2wav.factory.max_batch_wait_ms == 0.0
    assert code2wav.factory.batch_wait_when_idle is False


def test_vocode_slices_waveforms_to_token_lengths() -> None:
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
        dtype=torch.float32,
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
    waveforms = model.vocode([[1, 2], [3, 4, 5]], b"ref")
    assert [wave.shape for wave in waveforms] == [
        (2 * SAMPLES_PER_CODEC_TOKEN,),
        (3 * SAMPLES_PER_CODEC_TOKEN,),
    ]


def test_vocode_rejects_empty_sequences() -> None:
    model = MiniCPMOCode2Wav.__new__(MiniCPMOCode2Wav)
    assert model.vocode([], b"ref") == []
    with pytest.raises(ValueError, match="non-empty"):
        model.vocode([[1], []], b"ref")


def _fake_code2wav_model() -> MagicMock:
    fake = MagicMock()
    fake.sample_rate = 24000
    fake.resolve_prompt_wav.side_effect = lambda reference: (
        b"default" if reference is None else reference
    )
    fake.vocode.side_effect = lambda sequences, reference: [
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
    fake.vocode.assert_called_once_with([[7, 8, 9]], b"default")
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
    assert fake.vocode.call_count == 2
    batched_calls = {call.args[1]: call.args[0] for call in fake.vocode.call_args_list}
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
    fake.vocode.assert_called_once_with([[1], [2, 3]], b"default")
