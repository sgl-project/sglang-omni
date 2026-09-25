# SPDX-License-Identifier: Apache-2.0
"""Public MiniCPM-o vocoder contracts: import, checkpoint decode, speaker ref."""

from __future__ import annotations

import base64
import io
import json
import math
import os
import subprocess
import sys
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf
import torch

from sglang_omni.config.manager import ConfigManager
from sglang_omni.models.minicpm_o import stages
from sglang_omni.models.minicpm_o.components.code2wav import (
    SAMPLES_PER_CODEC_TOKEN,
    MiniCPMOCode2Wav,
)
from sglang_omni.models.minicpm_o.components.token2wav import vocoder
from sglang_omni.models.minicpm_o.components.token2wav.dit import TimestepEmbedder
from sglang_omni.models.minicpm_o.components.token2wav.flow import (
    SEEDTTS_EN_DENSE_PACKED_DIT_CUDA_GRAPH_SHAPES,
    SEEDTTS_EN_PACKED_DIT_CUDA_GRAPH_SHAPES,
    build_default_flow_cuda_graph_shapes,
)
from sglang_omni.models.minicpm_o.config import MiniCPMOSpeechPipelineConfig
from sglang_omni.models.minicpm_o.payload_types import MiniCPMOPipelineState
from sglang_omni.models.minicpm_o.routing import (
    code2wav_reference_audio,
    project_talker_to_code2wav,
)
from sglang_omni.models.minicpm_o.stages import vocode_code2wav_payloads
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.messages import IncomingMessage

REPO_ROOT = Path(__file__).resolve().parents[3]


def prompt_result(value: int) -> vocoder.SpeakerPrompt:
    return (
        torch.full((1, 2), value, dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32),
        torch.zeros(1, 4),
        torch.zeros(1, 4, 80),
    )


@pytest.fixture
def mock_token2wav(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    (tmp_path / "assets" / "token2wav").mkdir(parents=True)
    monkeypatch.delenv("MINICPMO_REF_WORKERS", raising=False)
    monkeypatch.delenv("MINICPMO_PROMPT_CACHE_CAPACITY", raising=False)
    backend = MagicMock()
    backend.prepare_prompt.side_effect = lambda source: prompt_result(
        source.getvalue()[0]
    )
    monkeypatch.setattr(vocoder, "Token2Wav", MagicMock(return_value=backend))
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    return backend


@pytest.fixture
def reference_model(
    tmp_path: Path, mock_token2wav: MagicMock
) -> Iterator[MiniCPMOCode2Wav]:
    model = MiniCPMOCode2Wav(str(tmp_path))
    try:
        yield model
    finally:
        model.close_reference_pool()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("frequency_size", [255, 256])
def test_timestep_embedding_matches_reference(
    dtype: torch.dtype, frequency_size: int
) -> None:
    model = TimestepEmbedder(16, frequency_size).to(dtype).eval()
    t = torch.linspace(0, 1, 11, dtype=dtype)
    half = frequency_size // 2
    frequencies = torch.exp(-math.log(10000) * torch.arange(half) / half).to(t)
    angles = (t * 1000)[:, None] * frequencies[None]
    embedding = torch.cat([angles.cos(), angles.sin()], dim=-1)
    if frequency_size % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    torch.testing.assert_close(model(t), model.mlp(embedding), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_timestep_embedding_autocast_preserves_frequencies(dtype: torch.dtype) -> None:
    model = TimestepEmbedder(16).to(device="cuda", dtype=dtype).eval()
    t = torch.linspace(0, 1, 11, device="cuda", dtype=torch.float32)
    frequencies = torch.exp(-math.log(10000) * torch.arange(128) / 128).to(t)
    angles = (t * 1000)[:, None] * frequencies[None]
    embedding = torch.cat([angles.cos(), angles.sin()], dim=-1)
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=dtype):
        torch.testing.assert_close(model(t), model.mlp(embedding), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_timestep_embedding_cuda_graph_replays_new_inputs() -> None:
    model = TimestepEmbedder(16).cuda().eval()
    t = torch.zeros(2, device="cuda")
    with torch.inference_mode():
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                model(t)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = model(t)
        t.fill_(0.25)
        expected = model(t)
        graph.replay()
        torch.testing.assert_close(output, expected, rtol=0, atol=0)


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
@pytest.mark.parametrize("enable_variable_length", [False, True])
def test_parallel_reference_preparation_preserves_checkpoint_waveforms(
    enable_variable_length: bool,
) -> None:
    checkpoint = _checkpoint_dir()
    if checkpoint is None or not torch.cuda.is_available():
        pytest.skip("Set MINICPMO_CHECKPOINT and provide CUDA for vocoder validation")
    reference_path = checkpoint / "assets" / "HT_ref_audio.wav"
    audio, sample_rate = sf.read(reference_path, dtype="float32", always_2d=True)
    short_reference = io.BytesIO()
    sf.write(
        short_reference,
        audio[: audio.shape[0] // 2],
        sample_rate,
        format="wav",
    )
    references = [str(reference_path), short_reference.getvalue(), str(reference_path)]
    tokens = [1498, 1734, 3732, 3726, 3645]
    sequences = [tokens, tokens + [3645, 3726], tokens[:3]]
    model = MiniCPMOCode2Wav(
        str(checkpoint),
        device="cuda:0",
        enable_flow_variable_length=enable_variable_length,
    )
    try:
        with torch.inference_mode():
            model.reference_workers = 1
            torch.manual_seed(0)
            serial = model.vocode(sequences, references)
            model.prompt_cache.clear()
            model.reference_workers = 2
            torch.manual_seed(0)
            parallel = model.vocode(sequences, references)
        for actual, expected in zip(parallel, serial, strict=True):
            assert np.isfinite(actual).all()
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
    finally:
        model.close_reference_pool()


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


@pytest.mark.parametrize("enabled", [False, True])
def test_variable_length_option_reaches_dit(
    enabled: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "assets" / "token2wav").mkdir(parents=True)
    token2wav = MagicMock()
    monkeypatch.setattr(vocoder, "Token2Wav", lambda *args, **kwargs: token2wav)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    MiniCPMOCode2Wav(str(tmp_path), enable_flow_variable_length=enabled)
    assert token2wav.flow.decoder.estimator.enable_variable_length is enabled


def test_speech_pipeline_enables_code2wav_batching_by_default() -> None:
    config = MiniCPMOSpeechPipelineConfig(model_path="unused")
    code2wav = next(stage for stage in config.stages if stage.name == "code2wav")
    assert code2wav.factory.max_batch_size == 8
    assert code2wav.factory.max_batch_wait_ms == 0.0
    assert code2wav.factory.batch_wait_when_idle is False
    assert code2wav.factory.dtype is None
    assert code2wav.factory.enable_flow_variable_length is True
    assert code2wav.factory.enable_dit_torch_compile is True
    assert code2wav.factory.enable_flow_cuda_graph is True
    assert code2wav.factory.flow_cuda_graph_capture_shapes is None
    assert code2wav.factory.packed_dit_cuda_graph_capture_shapes is None


def test_speech_pipeline_parses_flow_execution_options() -> None:
    config = MiniCPMOSpeechPipelineConfig(model_path="unused")
    merged = ConfigManager(config).merge_config(
        [
            ("code2wav.factory.enable_dit_torch_compile", "false"),
            ("code2wav.factory.enable_flow_cuda_graph", "false"),
            (
                "code2wav.factory.flow_cuda_graph_capture_shapes",
                "[[1,256],[1,512]]",
            ),
            ("code2wav.factory.packed_dit_cuda_graph_capture_shapes", "[[2,1856]]"),
        ]
    )
    code2wav = next(stage for stage in merged.stages if stage.name == "code2wav")
    assert code2wav.factory.enable_dit_torch_compile is False
    assert code2wav.factory.enable_flow_cuda_graph is False
    assert code2wav.factory.flow_cuda_graph_capture_shapes == (
        (1, 256),
        (1, 512),
    )
    assert code2wav.factory.packed_dit_cuda_graph_capture_shapes == ((2, 1856),)


def test_seedtts_en_packed_graph_table_can_be_selected_explicitly() -> None:
    assert len(SEEDTTS_EN_PACKED_DIT_CUDA_GRAPH_SHAPES) == 12
    config = MiniCPMOSpeechPipelineConfig(model_path="unused")
    merged = ConfigManager(config).merge_config(
        [
            (
                "code2wav.factory.packed_dit_cuda_graph_capture_shapes",
                json.dumps(SEEDTTS_EN_PACKED_DIT_CUDA_GRAPH_SHAPES),
            )
        ]
    )
    code2wav = next(stage for stage in merged.stages if stage.name == "code2wav")
    assert (
        code2wav.factory.packed_dit_cuda_graph_capture_shapes
        == SEEDTTS_EN_PACKED_DIT_CUDA_GRAPH_SHAPES
    )


def test_seedtts_en_dense_graph_tables_can_be_selected_explicitly() -> None:
    packed_shapes = SEEDTTS_EN_DENSE_PACKED_DIT_CUDA_GRAPH_SHAPES
    flow_shapes = build_default_flow_cuda_graph_shapes()
    assert len(flow_shapes) == 141
    assert sum(batch == 1 for batch, _ in flow_shapes) == 44
    assert len(packed_shapes) == len(set(packed_shapes)) == 46
    assert set(SEEDTTS_EN_PACKED_DIT_CUDA_GRAPH_SHAPES) <= set(packed_shapes)
    config = MiniCPMOSpeechPipelineConfig(model_path="unused")
    merged = ConfigManager(config).merge_config(
        [
            (
                "code2wav.factory.flow_cuda_graph_capture_shapes",
                json.dumps(flow_shapes),
            ),
            (
                "code2wav.factory.packed_dit_cuda_graph_capture_shapes",
                json.dumps(packed_shapes),
            ),
        ]
    )
    code2wav = next(stage for stage in merged.stages if stage.name == "code2wav")
    assert code2wav.factory.flow_cuda_graph_capture_shapes == flow_shapes
    assert code2wav.factory.packed_dit_cuda_graph_capture_shapes == packed_shapes


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
    prompt = (
        torch.zeros(1, 1, dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        torch.zeros(1, 4),
        torch.zeros(1, 1, 80),
    )
    model.prepare_references = lambda references: [prompt] * len(references)
    waveforms = model.vocode([[1, 2], [3, 4, 5]], b"ref")
    assert [wave.shape for wave in waveforms] == [
        (2 * SAMPLES_PER_CODEC_TOKEN,),
        (3 * SAMPLES_PER_CODEC_TOKEN,),
    ]


def test_vocode_rejects_mismatched_reference_count() -> None:
    model = MiniCPMOCode2Wav.__new__(MiniCPMOCode2Wav)
    with pytest.raises(ValueError, match="does not match"):
        model.vocode([[1], [2]], [b"a"])


def test_prompt_cache_reuses_references_and_evicts_least_recent(
    reference_model: MiniCPMOCode2Wav, mock_token2wav: MagicMock
) -> None:
    reference_model.prompt_cache_capacity = 2
    for reference in (b"a", b"b", b"a", b"c", b"a", b"b"):
        prompt = reference_model.prepare_references([reference])[0]
        assert prompt[0][0, 0].item() == reference[0]
    assert mock_token2wav.prepare_prompt.call_count == 4


@pytest.mark.parametrize("workers", [1, 2])
def test_prepare_references_deduplicates_identical_rows(
    reference_model: MiniCPMOCode2Wav, mock_token2wav: MagicMock, workers: int
) -> None:
    reference_model.reference_workers = workers
    prompts = reference_model.prepare_references([b"a"] * 8)
    assert len(prompts) == 8
    assert all(prompt is prompts[0] for prompt in prompts)
    assert prompts[0][0][0, 0].item() == ord("a")
    mock_token2wav.prepare_prompt.assert_called_once()


def test_prepare_references_runs_in_parallel_and_restores_row_order(
    reference_model: MiniCPMOCode2Wav, mock_token2wav: MagicMock
) -> None:
    both_started = threading.Barrier(2, timeout=5)
    second_finished = threading.Event()

    def prepare(source: io.BytesIO) -> vocoder.SpeakerPrompt:
        reference = source.getvalue()
        both_started.wait()
        if reference == b"a":
            assert second_finished.wait(5)
        else:
            second_finished.set()
        return prompt_result(reference[0])

    mock_token2wav.prepare_prompt.side_effect = prepare
    reference_model.reference_workers = 2
    reference_model.prompt_cache_capacity = 1
    prompts = reference_model.prepare_references([b"a", b"b", b"a", b"b"])
    assert [prompt[0][0, 0].item() for prompt in prompts] == [97, 98, 97, 98]
    assert prompts[0] is prompts[2]
    assert prompts[1] is prompts[3]
    assert mock_token2wav.prepare_prompt.call_count == 2


@pytest.mark.parametrize(
    ("workers", "capacity", "expected_workers", "expected_capacity"),
    [(None, None, 8, 32), ("1", "1", 1, 1), ("3", "2", 3, 2)],
)
def test_reference_configuration_is_read_by_constructor(
    tmp_path: Path,
    mock_token2wav: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    workers: str | None,
    capacity: str | None,
    expected_workers: int,
    expected_capacity: int,
) -> None:
    if workers is not None:
        monkeypatch.setenv("MINICPMO_REF_WORKERS", workers)
    if capacity is not None:
        monkeypatch.setenv("MINICPMO_PROMPT_CACHE_CAPACITY", capacity)
    model = MiniCPMOCode2Wav(str(tmp_path))
    try:
        assert model.reference_workers == expected_workers
        assert model.prompt_cache_capacity == expected_capacity
        assert vocoder.Token2Wav.call_args.kwargs["dtype"] == torch.float32
    finally:
        model.close_reference_pool()


@pytest.mark.parametrize(
    "name", ["MINICPMO_REF_WORKERS", "MINICPMO_PROMPT_CACHE_CAPACITY"]
)
@pytest.mark.parametrize("value", ["0", "-1", "abc", "1.5", ""])
def test_reference_configuration_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        MiniCPMOCode2Wav("unused")


def test_failed_reference_batch_drains_workers_and_can_retry(
    reference_model: MiniCPMOCode2Wav, mock_token2wav: MagicMock
) -> None:
    slow_started = threading.Event()
    failed = threading.Event()
    release = threading.Event()

    def prepare(source: io.BytesIO) -> vocoder.SpeakerPrompt:
        reference = source.getvalue()
        if reference == b"a":
            assert slow_started.wait(5)
            failed.set()
            raise ValueError("invalid reference")
        else:
            slow_started.set()
            assert release.wait(5)
            return prompt_result(reference[0])

    mock_token2wav.prepare_prompt.side_effect = prepare
    reference_model.reference_workers = 2
    with ThreadPoolExecutor(max_workers=1) as caller:
        preparation = caller.submit(reference_model.prepare_references, [b"a", b"b"])
        try:
            assert failed.wait(5)
            with pytest.raises(TimeoutError):
                preparation.result(timeout=0.1)
        finally:
            release.set()
        with pytest.raises(ValueError, match="invalid reference"):
            preparation.result(timeout=5)

    mock_token2wav.prepare_prompt.side_effect = lambda source: prompt_result(
        source.getvalue()[0]
    )
    prompts = reference_model.prepare_references([b"a", b"b"])
    assert [prompt[0][0, 0].item() for prompt in prompts] == [97, 98]
    assert mock_token2wav.prepare_prompt.call_count == 3


@pytest.mark.parametrize("workers", [1, 2])
def test_close_waits_for_reference_preparation_and_rejects_new_work(
    reference_model: MiniCPMOCode2Wav, mock_token2wav: MagicMock, workers: int
) -> None:
    started = threading.Event()
    closing = threading.Event()
    release = threading.Event()
    preparation_threads: set[threading.Thread] = set()

    def prepare(source: io.BytesIO) -> vocoder.SpeakerPrompt:
        preparation_threads.add(threading.current_thread())
        started.set()
        assert release.wait(5)
        return prompt_result(source.getvalue()[0])

    def close() -> None:
        closing.set()
        reference_model.close_reference_pool()

    mock_token2wav.prepare_prompt.side_effect = prepare
    reference_model.reference_workers = workers
    with ThreadPoolExecutor(max_workers=2) as callers:
        preparation = callers.submit(reference_model.prepare_references, [b"a", b"b"])
        try:
            assert started.wait(5)
            shutdown = callers.submit(close)
            assert closing.wait(5)
            with pytest.raises(TimeoutError):
                shutdown.result(timeout=0.1)
        finally:
            release.set()
        assert len(preparation.result(timeout=5)) == 2
        shutdown.result(timeout=5)
    assert all(not thread.is_alive() for thread in preparation_threads)
    reference_model.close_reference_pool()
    with pytest.raises(RuntimeError, match="closed"):
        reference_model.prepare_references([b"c", b"d"])
    assert mock_token2wav.prepare_prompt.call_count == 2


def test_stage_stop_closes_reference_preparation(
    reference_model: MiniCPMOCode2Wav, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        stages, "MiniCPMOCode2Wav", MagicMock(return_value=reference_model)
    )
    scheduler = stages.create_code2wav_executor("unused", device="cuda", gpu_id=0)
    reference_model.prepare_references([b"a", b"b"])
    scheduler.stop()
    scheduler.stop()
    with pytest.raises(RuntimeError, match="closed"):
        reference_model.prepare_references([b"a", b"b"])


def test_stage_stop_prevents_late_reference_pool_creation(
    reference_model: MiniCPMOCode2Wav,
    mock_token2wav: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = threading.Event()
    release = threading.Event()

    def delayed_vocode(
        model: MiniCPMOCode2Wav, payloads: list[StagePayload]
    ) -> list[StagePayload]:
        entered.set()
        assert release.wait(5)
        model.prepare_references([b"a", b"b"])
        return payloads

    monkeypatch.setattr(
        stages, "MiniCPMOCode2Wav", MagicMock(return_value=reference_model)
    )
    monkeypatch.setattr(stages, "vocode_code2wav_payloads", delayed_vocode)
    scheduler = stages.create_code2wav_executor("unused", device="cuda", gpu_id=0)
    scheduler.inbox.put(IncomingMessage("late", "new_request", _payload()))
    with ThreadPoolExecutor(max_workers=1) as caller:
        running = caller.submit(scheduler.start)
        try:
            assert entered.wait(5)
            scheduler.stop()
        finally:
            release.set()
            scheduler.stop()
        running.result(timeout=5)
    output = scheduler.outbox.get(timeout=5)
    assert output.type == "error"
    assert isinstance(output.data, RuntimeError)
    assert "closed" in str(output.data)
    mock_token2wav.prepare_prompt.assert_not_called()


def _batch_model() -> MiniCPMOCode2Wav:
    class FakeFlow:
        up_rate = 2

        def inference(
            self,
            speech_tokens: torch.Tensor,
            speech_tokens_lens: torch.Tensor,
            prompt_tokens: torch.Tensor,
            prompt_tokens_lens: torch.Tensor,
            prompt_mels: torch.Tensor,
            speaker_embedding: torch.Tensor,
            n_timesteps: int,
        ) -> torch.Tensor:
            frames = (speech_tokens_lens + prompt_tokens_lens).max() * self.up_rate
            return torch.zeros(speech_tokens.shape[0], 80, frames)

    class FakeHiFT:
        def __call__(self, speech_feat: torch.Tensor) -> tuple[torch.Tensor, None]:
            samples = speech_feat.shape[-1] * (SAMPLES_PER_CODEC_TOKEN // 2)
            return speech_feat.new_ones(speech_feat.shape[0], 1, samples), None

    model = MiniCPMOCode2Wav.__new__(MiniCPMOCode2Wav)
    model.token2wav = SimpleNamespace(
        device=torch.device("cpu"),
        dtype=torch.float32,
        n_timesteps=10,
        flow=FakeFlow(),
        hift=FakeHiFT(),
    )

    def speaker_prompt(prompt_wav: bytes | None) -> vocoder.SpeakerPrompt:
        prompt_len = 1 if prompt_wav in (None, b"ref") else 3
        return (
            torch.zeros(1, prompt_len, dtype=torch.int32),
            torch.tensor([prompt_len], dtype=torch.int32),
            torch.zeros(1, 4),
            torch.zeros(1, prompt_len * 2, 80),
        )

    model.prepare_references = lambda references: [
        speaker_prompt(reference) for reference in references
    ]
    return model


def test_vocode_mixed_references_and_lengths_share_one_batch() -> None:
    model = _batch_model()
    waveforms = model.vocode([[1, 2], [3, 4, 5], [6]], [b"ref", b"other", b"ref"])
    assert [wave.shape for wave in waveforms] == [
        (2 * SAMPLES_PER_CODEC_TOKEN,),
        (3 * SAMPLES_PER_CODEC_TOKEN,),
        (SAMPLES_PER_CODEC_TOKEN,),
    ]


def test_vocode_mixed_lengths_preserve_hift_boundaries() -> None:
    class BoundarySensitiveHiFT:
        def __call__(self, speech_feat: torch.Tensor) -> tuple[torch.Tensor, None]:
            kernel = speech_feat.new_ones(1, 1, 3)
            hidden = (
                torch.nn.functional.conv1d(speech_feat[:, :1], kernel, padding=1) + 1
            )
            samples = torch.nn.functional.conv1d(hidden, kernel, padding=1)
            waveform = samples.repeat_interleave(SAMPLES_PER_CODEC_TOKEN // 2, dim=-1)
            return waveform, None

    model = _batch_model()
    model.token2wav.hift = BoundarySensitiveHiFT()
    sequences = [[1, 2], [3, 4, 5], [6, 7]]
    batched = model.vocode(sequences, b"ref")
    for tokens, waveform in zip(sequences, batched, strict=True):
        reference = model.vocode([tokens], b"ref")[0]
        np.testing.assert_array_equal(waveform, reference)


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
    fake.vocode.side_effect = lambda sequences, references: [
        np.full(
            len(tokens) * SAMPLES_PER_CODEC_TOKEN,
            float(len(tokens)),
            dtype=np.float32,
        )
        for tokens in sequences
    ]
    return fake


def test_vocode_payloads_vocodes_one_reference_per_row() -> None:
    fake = _fake_code2wav_model()
    output = vocode_code2wav_payloads(fake, [_payload(tokens=[7, 8, 9])])[0]
    fake.vocode.assert_called_once_with([[7, 8, 9]], [b"default"])
    assert output.data["sample_rate"] == 24000
    assert output.data["audio_waveform_shape"] == [3 * SAMPLES_PER_CODEC_TOKEN]


def test_vocode_payloads_keeps_mixed_references_in_one_call() -> None:
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
    fake.vocode.assert_called_once_with(
        [[1, 2], [3], [4, 5, 6]], [b"spk-a", b"spk-b", b"spk-a"]
    )
    assert [out.data["audio_waveform_shape"][0] for out in outputs] == [
        2 * SAMPLES_PER_CODEC_TOKEN,
        SAMPLES_PER_CODEC_TOKEN,
        3 * SAMPLES_PER_CODEC_TOKEN,
    ]


def test_vocode_payloads_resolves_default_reference_per_row() -> None:
    fake = _fake_code2wav_model()
    vocode_code2wav_payloads(
        fake,
        [_payload(request_id="a", tokens=[1]), _payload(request_id="b", tokens=[2, 3])],
    )
    fake.vocode.assert_called_once_with([[1], [2, 3]], [b"default", b"default"])
