# SPDX-License-Identifier: Apache-2.0
"""Public MiniCPM-o vocoder contracts: import, checkpoint decode, speaker ref."""

from __future__ import annotations

import base64
import io
import os
import subprocess
import sys
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from typing import Protocol
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf
import torch

from sglang_omni.client.client import build_params
from sglang_omni.config import FactoryArgs
from sglang_omni.models.minicpm_o import stages
from sglang_omni.models.minicpm_o.components import code2wav
from sglang_omni.models.minicpm_o.components.code2wav import (
    SAMPLES_PER_CODEC_TOKEN,
    MiniCPMOCode2Wav,
)
from sglang_omni.models.minicpm_o.components.token2wav.vocoder import SpeakerPrompt
from sglang_omni.models.minicpm_o.config import MiniCPMOSpeechPipelineConfig
from sglang_omni.models.minicpm_o.payload_types import MiniCPMOPipelineState
from sglang_omni.models.minicpm_o.routing import (
    code2wav_reference_audio,
    project_talker_to_code2wav,
)
from sglang_omni.models.minicpm_o.stages import vocode_code2wav_payloads
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.message import IncomingMessage
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.serve.openai_api import (
    ChatCompletionRequest,
    build_chat_generate_request,
)
from sglang_omni.utils.device import resolve_concrete_device

REPO_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_CODEC_TOKENS = [1498, 1734, 3732, 3726, 3645]
FAKE_UP_RATE = 2
FAKE_MEL_BINS = 80
REFERENCE_ID_SCALE = 1000
CONDITIONING_TENSOR_COUNT = 3
DEFAULT_REFERENCE_AUDIO = b"default"
THREAD_WAIT_SECONDS = 5
BLOCKED_CALL_PROBE_SECONDS = 0.1


class Code2WavBuilder(Protocol):
    def __call__(
        self,
        *,
        reference_workers: int = ...,
        prompt_cache_capacity: int = ...,
        enable_flow_variable_length: bool = ...,
    ) -> MiniCPMOCode2Wav: ...


def fake_speaker_prompt(reference_audio: bytes) -> SpeakerPrompt:
    """Prompt as wide as the reference, with every tensor tagged by its first byte."""
    reference_id = reference_audio[0]
    prompt_token_count = len(reference_audio)
    return SpeakerPrompt(
        prompt_tokens=torch.full(
            (1, prompt_token_count), reference_id, dtype=torch.int32
        ),
        prompt_token_lengths=torch.tensor([prompt_token_count], dtype=torch.int32),
        speaker_embedding=torch.full((1, 4), float(reference_id)),
        prompt_mel=torch.full(
            (1, prompt_token_count * FAKE_UP_RATE, FAKE_MEL_BINS), float(reference_id)
        ),
    )


def fake_prepare_prompt(source: str | io.BytesIO) -> SpeakerPrompt:
    if isinstance(source, str):
        reference_audio = Path(source).read_bytes()
    else:
        reference_audio = source.getvalue()
    return fake_speaker_prompt(reference_audio)


def fake_flow_inference(
    speech_tokens: torch.Tensor,
    speech_token_lengths: torch.Tensor,
    prompt_tokens: torch.Tensor,
    prompt_token_lengths: torch.Tensor,
    prompt_mels: torch.Tensor,
    speaker_embeddings: torch.Tensor,
    n_timesteps: int,
) -> torch.Tensor:
    # A row that receives another row's conditioning changes this sum.
    reference_ids = (
        prompt_tokens[:, :1] + speaker_embeddings[:, :1] + prompt_mels[:, 0, :1]
    )
    frame_values = speech_tokens + REFERENCE_ID_SCALE * reference_ids
    frames = frame_values.float().repeat_interleave(FAKE_UP_RATE, dim=1)
    return frames.unsqueeze(1).expand(-1, FAKE_MEL_BINS, -1)


def fake_hift(speech_feat: torch.Tensor) -> tuple[torch.Tensor, None]:
    samples_per_frame = SAMPLES_PER_CODEC_TOKEN // FAKE_UP_RATE
    waveform = speech_feat[:, :1].repeat_interleave(samples_per_frame, dim=-1)
    return waveform, None


def boundary_sensitive_hift(speech_feat: torch.Tensor) -> tuple[torch.Tensor, None]:
    """Noncausal smoothing whose last samples change if a row is padded."""
    kernel = speech_feat.new_ones(1, 1, 3)
    hidden = torch.nn.functional.conv1d(speech_feat[:, :1], kernel, padding=1) + 1
    smoothed = torch.nn.functional.conv1d(hidden, kernel, padding=1)
    samples_per_frame = SAMPLES_PER_CODEC_TOKEN // FAKE_UP_RATE
    return smoothed.repeat_interleave(samples_per_frame, dim=-1), None


def expected_waveform(codec_tokens: list[int], reference_audio: bytes) -> np.ndarray:
    reference_value = (
        REFERENCE_ID_SCALE * CONDITIONING_TENSOR_COUNT * reference_audio[0]
    )
    sample_values = np.asarray(codec_tokens, dtype=np.float32) + reference_value
    return np.repeat(sample_values, SAMPLES_PER_CODEC_TOKEN)


@pytest.fixture
def fake_token2wav(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    (tmp_path / "assets" / "token2wav").mkdir(parents=True)
    token2wav = MagicMock()
    token2wav.device = torch.device("cpu")
    token2wav.dtype = torch.float32
    token2wav.n_timesteps = 1
    token2wav.flow.up_rate = FAKE_UP_RATE
    token2wav.flow.inference.side_effect = fake_flow_inference
    token2wav.hift.side_effect = fake_hift
    token2wav.prepare_prompt.side_effect = fake_prepare_prompt
    monkeypatch.setattr(code2wav, "Token2Wav", MagicMock(return_value=token2wav))
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    return token2wav


@pytest.fixture
def build_code2wav_model(
    tmp_path: Path, fake_token2wav: MagicMock
) -> Iterator[Code2WavBuilder]:
    default_reference_path = tmp_path / "default.wav"
    default_reference_path.write_bytes(DEFAULT_REFERENCE_AUDIO)
    built_models: list[MiniCPMOCode2Wav] = []

    def build(
        *,
        reference_workers: int = 8,
        prompt_cache_capacity: int = 32,
        enable_flow_variable_length: bool = False,
    ) -> MiniCPMOCode2Wav:
        model = MiniCPMOCode2Wav(
            str(tmp_path),
            prompt_wav=str(default_reference_path),
            enable_flow_variable_length=enable_flow_variable_length,
            reference_workers=reference_workers,
            prompt_cache_capacity=prompt_cache_capacity,
        )
        built_models.append(model)
        return model

    yield build
    for model in built_models:
        model.close_reference_pool()


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


def find_checkpoint_dir() -> Path | None:
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
        else:
            pass
    for path in candidates:
        if (path / "assets" / "token2wav").is_dir():
            return path
        else:
            pass
    return None


def require_checkpoint_dir() -> Path:
    checkpoint = find_checkpoint_dir()
    if checkpoint is None or resolve_concrete_device(None).type not in ("cuda", "xpu"):
        pytest.skip(
            "Set MINICPMO_CHECKPOINT and provide CUDA or XPU for vocoder validation"
        )
    else:
        pass
    return checkpoint


def code2wav_stage_factory() -> FactoryArgs:
    config = MiniCPMOSpeechPipelineConfig(model_path="unused")
    return next(stage for stage in config.stages if stage.name == "code2wav").factory


def build_code2wav_stage(
    model: MiniCPMOCode2Wav, monkeypatch: pytest.MonkeyPatch
) -> SimpleScheduler:
    monkeypatch.setattr(stages, "MiniCPMOCode2Wav", MagicMock(return_value=model))
    factory = code2wav_stage_factory()
    return stages.create_code2wav_executor(
        "unused",
        device=None,
        gpu_id=0,
        enable_dit_torch_compile=factory.enable_dit_torch_compile,
        enable_flow_variable_length=factory.enable_flow_variable_length,
        reference_workers=factory.reference_workers,
        prompt_cache_capacity=factory.prompt_cache_capacity,
    )


def load_checkpoint_model(
    checkpoint: Path, *, enable_flow_variable_length: bool = False
) -> MiniCPMOCode2Wav:
    factory = code2wav_stage_factory()
    return MiniCPMOCode2Wav(
        str(checkpoint),
        device=str(resolve_concrete_device(None)),
        enable_flow_variable_length=enable_flow_variable_length,
        reference_workers=factory.reference_workers,
        prompt_cache_capacity=factory.prompt_cache_capacity,
    )


def relative_rms_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return (
        ((actual - expected).square().mean() / expected.square().mean()).sqrt().item()
    )


@pytest.mark.accelerator
def test_native_vocoder_with_checkpoint() -> None:
    model = load_checkpoint_model(require_checkpoint_dir())
    output = model(codec_tokens=torch.tensor(CHECKPOINT_CODEC_TOKENS))
    waveform = output["waveform"]
    assert output["sample_rate"] == 24000
    assert waveform.dtype == np.float32
    assert waveform.shape == (len(CHECKPOINT_CODEC_TOKENS) * SAMPLES_PER_CODEC_TOKEN,)
    assert np.isfinite(waveform).all()
    assert np.max(np.abs(waveform)) > 1e-5
    assert np.max(np.abs(waveform)) <= 0.99


@pytest.mark.accelerator
def test_native_vocoder_batch_matches_single_request_shapes() -> None:
    model = load_checkpoint_model(require_checkpoint_dir())
    tokens_a = CHECKPOINT_CODEC_TOKENS
    tokens_b = tokens_a + [3645, 3726]
    batched = model.vocode([tokens_a, tokens_b], [None, None])
    single_a = model.vocode([tokens_a], [None])[0]
    single_b = model.vocode([tokens_b], [None])[0]
    assert (
        batched[0].shape == single_a.shape == (len(tokens_a) * SAMPLES_PER_CODEC_TOKEN,)
    )
    assert (
        batched[1].shape == single_b.shape == (len(tokens_b) * SAMPLES_PER_CODEC_TOKEN,)
    )
    assert all(np.isfinite(wave).all() for wave in (*batched, single_a, single_b))


@pytest.mark.accelerator
@pytest.mark.parametrize(
    ("enable_flow_variable_length", "max_relative_rms_error"),
    [(False, 1e-3), (True, 2e-2)],
)
def test_mixed_reference_batch_matches_single_row_mels(
    enable_flow_variable_length: bool,
    max_relative_rms_error: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = require_checkpoint_dir()
    model = load_checkpoint_model(
        checkpoint, enable_flow_variable_length=enable_flow_variable_length
    )
    reference_path = checkpoint / "assets" / "HT_ref_audio.wav"
    audio, sample_rate = sf.read(reference_path, dtype="float32", always_2d=True)
    short_reference = io.BytesIO()
    sf.write(short_reference, audio[: audio.shape[0] // 2], sample_rate, format="wav")
    references = [str(reference_path), short_reference.getvalue(), str(reference_path)]
    speech_tokens = model.token2wav.prepare_prompt(str(reference_path)).prompt_tokens
    codec_tokens = speech_tokens.reshape(-1).tolist()
    sequences = [codec_tokens[:60], codec_tokens[10:110], codec_tokens[40:65]]
    generated_mels: list[torch.Tensor] = []
    flow_inference = model.token2wav.flow.inference

    def record_generated_mel(*flow_inputs: torch.Tensor | int) -> torch.Tensor:
        generated_mel = flow_inference(*flow_inputs)
        generated_mels.append(generated_mel)
        return generated_mel

    monkeypatch.setattr(model.token2wav.flow, "inference", record_generated_mel)
    try:
        model.vocode(sequences, references)
        batched_mels = generated_mels.pop()
        for row, (tokens, reference) in enumerate(
            zip(sequences, references, strict=True)
        ):
            model.vocode([tokens], [reference])
            single_mel = generated_mels.pop()[0]
            batched_mel = batched_mels[row, :, : single_mel.shape[-1]]
            assert (
                relative_rms_error(batched_mel, single_mel) < max_relative_rms_error
            ), f"row {row}"
    finally:
        model.close_reference_pool()


@pytest.fixture(scope="module")
def compiled_vocoder() -> Iterator[MiniCPMOCode2Wav]:
    checkpoint = find_checkpoint_dir()
    device = resolve_concrete_device(None)
    if checkpoint is None or device.type != "cuda":
        pytest.skip("Set MINICPMO_CHECKPOINT and provide CUDA for compiled flow checks")
    else:
        pass
    factory = code2wav_stage_factory()
    model = MiniCPMOCode2Wav(
        str(checkpoint),
        device=str(device),
        dtype=factory.dtype,
        enable_dit_torch_compile=True,
        enable_flow_variable_length=factory.enable_flow_variable_length,
        reference_workers=factory.reference_workers,
        prompt_cache_capacity=factory.prompt_cache_capacity,
    )
    yield model
    model.close_reference_pool()


@pytest.mark.accelerator
def test_compiled_flow_serves_new_batches_without_recompiling(
    compiled_vocoder: MiniCPMOCode2Wav,
) -> None:
    tokens = CHECKPOINT_CODEC_TOKENS
    sequences = [tokens * 3, tokens, *([tokens * 7] * 5)]
    with torch.compiler.set_stance("fail_on_recompile"):
        waveforms = compiled_vocoder.vocode(sequences[:2], [None] * 2)
        waveforms += compiled_vocoder.vocode(sequences[2:], [None] * 5)
    assert [wave.shape for wave in waveforms] == [
        (len(sequence) * SAMPLES_PER_CODEC_TOKEN,) for sequence in sequences
    ]
    assert all(np.isfinite(wave).all() for wave in waveforms)


@pytest.mark.accelerator
def test_compiled_flow_matches_eager(compiled_vocoder: MiniCPMOCode2Wav) -> None:
    device = compiled_vocoder.token2wav.device
    tokens = torch.tensor(
        [CHECKPOINT_CODEC_TOKENS * 7], dtype=torch.int32, device=device
    )
    token_lengths = torch.tensor([tokens.shape[1]], dtype=torch.int32, device=device)
    prompts = compiled_vocoder.prepare_references([None])
    compiled = compiled_vocoder.flow_mel(tokens, token_lengths, prompts).float()
    with torch.compiler.set_stance("force_eager"):
        eager = compiled_vocoder.flow_mel(tokens, token_lengths, prompts).float()
    assert relative_rms_error(compiled, eager) < 5e-3


def test_dit_torch_compile_rejects_non_cuda_device() -> None:
    with pytest.raises(ValueError, match="CUDA only"):
        MiniCPMOCode2Wav(
            "unused",
            device="xpu:0",
            enable_dit_torch_compile=True,
            enable_flow_variable_length=True,
            reference_workers=8,
            prompt_cache_capacity=32,
        )


def wav_data_uri(audio: bytes) -> str:
    return "data:audio/wav;base64," + base64.b64encode(audio).decode("ascii")


def talker_payload(
    *,
    request_id: str = "test",
    codec_tokens: list[int] | None = None,
    params: dict[str, object] | None = None,
    metadata: dict[str, object] | None = None,
) -> StagePayload:
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs=None, params=params or {}, metadata=metadata or {}),
        data=MiniCPMOPipelineState(
            engine_outputs={
                "talker": {"codec_tokens": torch.tensor(codec_tokens or [1, 2])}
            }
        ).to_dict(),
    )


def test_chat_api_forwards_reference_to_vocoder() -> None:
    reference = wav_data_uri(b"reference")
    request = ChatCompletionRequest(
        model="minicpm-o",
        messages=[{"role": "user", "content": "Hello"}],
        modalities=["text", "audio"],
        audio={"format": "wav", "ref_audio": reference},
    )
    generate_request = build_chat_generate_request(request)
    payload = talker_payload(
        params=build_params(generate_request), metadata=generate_request.metadata
    )
    assert code2wav_reference_audio(project_talker_to_code2wav(payload)) == b"reference"


def test_invalid_reference_does_not_silently_use_default() -> None:
    payload = talker_payload(params={"ref_audio": "/tmp/ref.wav"})
    with pytest.raises(ValueError, match="inline audio"):
        code2wav_reference_audio(payload)


@pytest.mark.parametrize("enable_flow_variable_length", [False, True])
def test_variable_length_option_reaches_dit(
    build_code2wav_model: Code2WavBuilder,
    fake_token2wav: MagicMock,
    enable_flow_variable_length: bool,
) -> None:
    build_code2wav_model(enable_flow_variable_length=enable_flow_variable_length)
    estimator = fake_token2wav.flow.decoder.estimator
    assert estimator.enable_variable_length is enable_flow_variable_length


def test_speech_pipeline_enables_code2wav_batching_by_default() -> None:
    factory = code2wav_stage_factory()
    assert factory.max_batch_size == 8
    assert factory.max_batch_wait_ms == 0.0
    assert factory.batch_wait_when_idle is False
    assert factory.dtype == "float16"
    assert factory.enable_dit_torch_compile is True
    assert factory.enable_flow_variable_length is False
    assert factory.reference_workers == 8
    assert factory.prompt_cache_capacity == 32


@pytest.mark.parametrize(
    "references",
    [[b"a", b"a", b"a"], [b"a", b"a", b"spk-b"]],
    ids=["shared_reference", "mixed_references"],
)
def test_vocode_conditions_each_row_on_its_reference(
    build_code2wav_model: Code2WavBuilder, references: list[bytes]
) -> None:
    sequences = [[1, 2], [3, 4, 5], [6]]
    waveforms = build_code2wav_model().vocode(sequences, references)
    for tokens, reference, waveform in zip(
        sequences, references, waveforms, strict=True
    ):
        np.testing.assert_array_equal(waveform, expected_waveform(tokens, reference))


def test_vocode_runs_mixed_references_in_one_flow_call(
    build_code2wav_model: Code2WavBuilder, fake_token2wav: MagicMock
) -> None:
    build_code2wav_model().vocode([[1, 2], [3], [4, 5, 6]], [b"a", b"spk-b", b"c"])
    assert fake_token2wav.flow.inference.call_count == 1


def test_vocode_mixed_lengths_preserve_hift_boundaries(
    build_code2wav_model: Code2WavBuilder, fake_token2wav: MagicMock
) -> None:
    fake_token2wav.hift.side_effect = boundary_sensitive_hift
    model = build_code2wav_model()
    sequences = [[1, 2], [3, 4, 5], [6, 7]]
    batched = model.vocode(sequences, [b"a"] * len(sequences))
    for tokens, waveform in zip(sequences, batched, strict=True):
        np.testing.assert_array_equal(waveform, model.vocode([tokens], [b"a"])[0])


def test_vocode_rejects_empty_sequences(
    build_code2wav_model: Code2WavBuilder,
) -> None:
    model = build_code2wav_model()
    assert model.vocode([], []) == []
    with pytest.raises(ValueError, match="non-empty"):
        model.vocode([[1], []], [b"a", b"a"])


def test_prompt_cache_reuses_references_and_evicts_least_recent(
    build_code2wav_model: Code2WavBuilder, fake_token2wav: MagicMock
) -> None:
    model = build_code2wav_model(prompt_cache_capacity=2)
    for reference in (b"a", b"b", b"a", b"c", b"a", b"b"):
        model.prepare_references([reference])
    assert fake_token2wav.prepare_prompt.call_count == 4


@pytest.mark.parametrize("reference_workers", [1, 2])
def test_prepare_references_prepares_identical_rows_once(
    build_code2wav_model: Code2WavBuilder,
    fake_token2wav: MagicMock,
    reference_workers: int,
) -> None:
    model = build_code2wav_model(reference_workers=reference_workers)
    prompts = model.prepare_references([b"a"] * 8)
    assert [prompt.prompt_tokens[0, 0].item() for prompt in prompts] == [ord("a")] * 8
    fake_token2wav.prepare_prompt.assert_called_once()


def test_prepare_references_runs_in_parallel_and_restores_row_order(
    build_code2wav_model: Code2WavBuilder, fake_token2wav: MagicMock
) -> None:
    both_started = threading.Barrier(2, timeout=THREAD_WAIT_SECONDS)
    second_finished = threading.Event()

    def prepare_out_of_order(source: io.BytesIO) -> SpeakerPrompt:
        reference_audio = source.getvalue()
        both_started.wait()
        if reference_audio == b"a":
            assert second_finished.wait(THREAD_WAIT_SECONDS)
        else:
            second_finished.set()
        return fake_speaker_prompt(reference_audio)

    fake_token2wav.prepare_prompt.side_effect = prepare_out_of_order
    model = build_code2wav_model(reference_workers=2)
    prompts = model.prepare_references([b"a", b"b", b"a", b"b"])
    assert [prompt.prompt_tokens[0, 0].item() for prompt in prompts] == [97, 98, 97, 98]
    assert fake_token2wav.prepare_prompt.call_count == 2


def test_failed_reference_batch_drains_workers_and_can_retry(
    build_code2wav_model: Code2WavBuilder, fake_token2wav: MagicMock
) -> None:
    slow_started = threading.Event()
    failed = threading.Event()
    release_slow = threading.Event()

    def fail_while_other_runs(source: io.BytesIO) -> SpeakerPrompt:
        reference_audio = source.getvalue()
        if reference_audio == b"a":
            assert slow_started.wait(THREAD_WAIT_SECONDS)
            failed.set()
            raise ValueError("invalid reference")
        else:
            slow_started.set()
            assert release_slow.wait(THREAD_WAIT_SECONDS)
            return fake_speaker_prompt(reference_audio)

    fake_token2wav.prepare_prompt.side_effect = fail_while_other_runs
    model = build_code2wav_model(reference_workers=2)
    with ThreadPoolExecutor(max_workers=1) as caller:
        preparation = caller.submit(model.prepare_references, [b"a", b"b"])
        try:
            assert failed.wait(THREAD_WAIT_SECONDS)
            with pytest.raises(TimeoutError):
                preparation.result(timeout=BLOCKED_CALL_PROBE_SECONDS)
        finally:
            release_slow.set()
        with pytest.raises(ValueError, match="invalid reference"):
            preparation.result(timeout=THREAD_WAIT_SECONDS)

    fake_token2wav.prepare_prompt.side_effect = fake_prepare_prompt
    prompts = model.prepare_references([b"a", b"b"])
    assert [prompt.prompt_tokens[0, 0].item() for prompt in prompts] == [97, 98]


@pytest.mark.parametrize("reference_workers", [1, 2])
def test_close_reference_pool_waits_for_running_preparation(
    build_code2wav_model: Code2WavBuilder,
    fake_token2wav: MagicMock,
    reference_workers: int,
) -> None:
    started = threading.Event()
    closing = threading.Event()
    release = threading.Event()
    preparation_threads: set[threading.Thread] = set()

    def blocking_prepare(source: io.BytesIO) -> SpeakerPrompt:
        preparation_threads.add(threading.current_thread())
        started.set()
        assert release.wait(THREAD_WAIT_SECONDS)
        return fake_speaker_prompt(source.getvalue())

    fake_token2wav.prepare_prompt.side_effect = blocking_prepare
    model = build_code2wav_model(reference_workers=reference_workers)

    def close() -> None:
        closing.set()
        model.close_reference_pool()

    with ThreadPoolExecutor(max_workers=2) as callers:
        preparation = callers.submit(model.prepare_references, [b"a", b"b"])
        try:
            assert started.wait(THREAD_WAIT_SECONDS)
            shutdown = callers.submit(close)
            assert closing.wait(THREAD_WAIT_SECONDS)
            with pytest.raises(TimeoutError):
                shutdown.result(timeout=BLOCKED_CALL_PROBE_SECONDS)
        finally:
            release.set()
        assert len(preparation.result(timeout=THREAD_WAIT_SECONDS)) == 2
        shutdown.result(timeout=THREAD_WAIT_SECONDS)
    assert all(not thread.is_alive() for thread in preparation_threads)


def test_stage_stop_rejects_new_reference_preparation(
    build_code2wav_model: Code2WavBuilder,
    fake_token2wav: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = build_code2wav_model()
    scheduler = build_code2wav_stage(model, monkeypatch)
    scheduler.stop()
    with pytest.raises(RuntimeError):
        model.prepare_references([b"a", b"b"])
    fake_token2wav.prepare_prompt.assert_not_called()


@pytest.mark.parametrize(
    ("reference_workers", "prompt_cache_capacity"), [(0, 32), (8, 0)]
)
def test_reference_configuration_rejects_non_positive_values(
    build_code2wav_model: Code2WavBuilder,
    reference_workers: int,
    prompt_cache_capacity: int,
) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        build_code2wav_model(
            reference_workers=reference_workers,
            prompt_cache_capacity=prompt_cache_capacity,
        )


def test_vocode_payloads_returns_each_row_with_its_reference(
    build_code2wav_model: Code2WavBuilder,
) -> None:
    rows = [
        ("a", [1, 2], b"spk-a"),
        ("b", [3], None),
        ("c", [4, 5, 6], b"spk-a"),
    ]
    payloads = [
        talker_payload(
            request_id=request_id,
            codec_tokens=codec_tokens,
            params={} if reference is None else {"ref_audio": wav_data_uri(reference)},
        )
        for request_id, codec_tokens, reference in rows
    ]
    outputs = vocode_code2wav_payloads(build_code2wav_model(), payloads)
    for (request_id, codec_tokens, reference), output in zip(
        rows, outputs, strict=True
    ):
        assert output.request_id == request_id
        assert output.data["sample_rate"] == 24000
        waveform = np.frombuffer(output.data["audio_waveform"], dtype=np.float32)
        np.testing.assert_array_equal(
            waveform,
            expected_waveform(codec_tokens, reference or DEFAULT_REFERENCE_AUDIO),
        )


QUEUED_REFERENCES = (("req-a", b"a"), ("req-b", b"b"))


def drain_reference_worker(model: MiniCPMOCode2Wav) -> None:
    """With one worker, a no-op task runs only after earlier prompts are stored."""
    model.reference_executor.submit(int).result(timeout=THREAD_WAIT_SECONDS)


def enqueue_talker_request(
    scheduler: SimpleScheduler, request_id: str, reference_audio: str
) -> None:
    payload = talker_payload(
        request_id=request_id, params={"ref_audio": reference_audio}
    )
    scheduler.enqueue(IncomingMessage(request_id, "new_request", payload))


def test_prefetched_reference_serves_its_batch_without_second_preparation(
    build_code2wav_model: Code2WavBuilder, fake_token2wav: MagicMock
) -> None:
    model = build_code2wav_model()
    model.prefetch_reference("req-a", b"a")
    prompts = model.prepare_references([b"a"])
    assert prompts[0].prompt_tokens[0, 0].item() == ord("a")
    fake_token2wav.prepare_prompt.assert_called_once()


def test_queued_references_survive_cache_overflow(
    build_code2wav_model: Code2WavBuilder, fake_token2wav: MagicMock
) -> None:
    model = build_code2wav_model(reference_workers=1, prompt_cache_capacity=1)
    for request_id, reference in QUEUED_REFERENCES:
        model.prefetch_reference(request_id, reference)
    drain_reference_worker(model)
    model.prepare_references([reference for _, reference in QUEUED_REFERENCES])
    assert fake_token2wav.prepare_prompt.call_count == len(QUEUED_REFERENCES)


def test_released_references_become_evictable(
    build_code2wav_model: Code2WavBuilder, fake_token2wav: MagicMock
) -> None:
    model = build_code2wav_model(reference_workers=1, prompt_cache_capacity=1)
    for request_id, reference in QUEUED_REFERENCES:
        model.prefetch_reference(request_id, reference)
    drain_reference_worker(model)
    for request_id, _ in QUEUED_REFERENCES:
        model.release_reference(request_id)
    model.prepare_references([b"a"])
    assert fake_token2wav.prepare_prompt.call_count == len(QUEUED_REFERENCES) + 1


def test_failed_prefetch_is_retried_by_its_batch(
    build_code2wav_model: Code2WavBuilder, fake_token2wav: MagicMock
) -> None:
    fake_token2wav.prepare_prompt.side_effect = ValueError("invalid reference")
    model = build_code2wav_model(reference_workers=1)
    model.prefetch_reference("req-a", b"a")
    drain_reference_worker(model)
    fake_token2wav.prepare_prompt.side_effect = fake_prepare_prompt
    prompts = model.prepare_references([b"a"])
    assert prompts[0].prompt_tokens[0, 0].item() == ord("a")


def test_code2wav_stage_prepares_reference_on_arrival(
    build_code2wav_model: Code2WavBuilder,
    fake_token2wav: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = build_code2wav_model(reference_workers=1)
    scheduler = build_code2wav_stage(model, monkeypatch)
    enqueue_talker_request(scheduler, "req-a", wav_data_uri(b"a"))
    drain_reference_worker(model)
    fake_token2wav.prepare_prompt.assert_called_once()


def test_code2wav_stage_queues_invalid_reference_without_preparing_it(
    build_code2wav_model: Code2WavBuilder,
    fake_token2wav: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = build_code2wav_stage(build_code2wav_model(), monkeypatch)
    enqueue_talker_request(scheduler, "req-a", "/tmp/ref.wav")
    assert scheduler.inbox.get_nowait().request_id == "req-a"
    fake_token2wav.prepare_prompt.assert_not_called()


def test_code2wav_stage_abort_releases_queued_reference(
    build_code2wav_model: Code2WavBuilder,
    fake_token2wav: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = build_code2wav_model(reference_workers=1, prompt_cache_capacity=1)
    scheduler = build_code2wav_stage(model, monkeypatch)
    for request_id, reference in QUEUED_REFERENCES:
        enqueue_talker_request(scheduler, request_id, wav_data_uri(reference))
    drain_reference_worker(model)
    for request_id, _ in QUEUED_REFERENCES:
        scheduler.abort(request_id)
    model.prepare_references([b"a"])
    assert fake_token2wav.prepare_prompt.call_count == len(QUEUED_REFERENCES) + 1


def test_code2wav_stage_batch_releases_its_references(
    build_code2wav_model: Code2WavBuilder,
    fake_token2wav: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = build_code2wav_model(reference_workers=1, prompt_cache_capacity=1)
    scheduler = build_code2wav_stage(model, monkeypatch)
    for request_id, reference in QUEUED_REFERENCES:
        enqueue_talker_request(scheduler, request_id, wav_data_uri(reference))
    scheduler.batch_fn([scheduler.inbox.get_nowait().data for _ in QUEUED_REFERENCES])
    model.prepare_references([b"a"])
    assert fake_token2wav.prepare_prompt.call_count == len(QUEUED_REFERENCES) + 1
