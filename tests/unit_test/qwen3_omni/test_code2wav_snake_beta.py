# SPDX-License-Identifier: Apache-2.0
"""Shared SnakeBeta installation and real Code2Wav PCM parity."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import SnakeBeta

from sglang_omni.models.qwen3_omni.components import code2wav_scheduler
from sglang_omni.models.qwen3_omni.components.code2wav_cuda_graph import (
    Code2WavCudaGraphRunner,
    GraphKey,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.utils import snake_beta
from tests.unit_test.fixtures.qwen_fakes import make_qwen_payload


@pytest.mark.parametrize("enabled", [False, True])
def test_factory_installs_shared_snake_before_graph_capture(
    monkeypatch: pytest.MonkeyPatch, enabled: bool
) -> None:
    original = SnakeBeta(96).eval()
    model = SimpleNamespace(
        decoder=torch.nn.Sequential(original),
        config=SimpleNamespace(num_quantizers=16),
        total_upsample=1920,
    )
    monkeypatch.setattr(
        code2wav_scheduler, "load_code2wav_model", Mock(return_value=model)
    )
    graph_runner = Mock()
    graph_runner.stats.return_value = {"enabled": True}

    def build(loaded_model: SimpleNamespace, **kwargs: object) -> Mock:
        assert loaded_model is model
        assert isinstance(model.decoder[0], snake_beta.FusedSnakeBeta) == enabled
        return graph_runner

    monkeypatch.setattr(Code2WavCudaGraphRunner, "build", build)
    scheduler = code2wav_scheduler.create_code2wav_scheduler(
        "unused",
        device="cpu",
        enable_cuda_graph=True,
        total_gpu_memory_fraction=0.1,
        fused_snake_activation=enabled,
    )
    assert scheduler.cuda_graph_runner is graph_runner
    assert model.decoder[0].alpha is original.alpha
    assert model.decoder[0].beta is original.beta


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_hf_snake_uses_shared_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    original = SnakeBeta(96).to(device="cuda", dtype=torch.bfloat16).eval()
    decoder = torch.nn.Sequential(original)
    x = torch.randn(16, 96, 37845, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        expected = decoder(x)
        assert snake_beta.fuse_vocoder_decoder(decoder) == 1
        launch = Mock(wraps=snake_beta.launch)
        monkeypatch.setattr(snake_beta, "launch", launch)
        assert torch.equal(decoder(x), expected)
        assert launch.call_count == 1


@pytest.mark.benchmark
@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_real_code2wav_pcm_equal(monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint = os.environ.get("QWEN3_OMNI_MODEL_PATH")
    if checkpoint is None:
        pytest.skip("Set QWEN3_OMNI_MODEL_PATH to run the real checkpoint gate")
    model = code2wav_scheduler.load_code2wav_model(
        checkpoint, device="cuda:0", dtype="bfloat16"
    )
    graph_keys = tuple(
        dict.fromkeys(
            code2wav_scheduler.batched_graph_keys(10, 25, 16)
            + code2wav_scheduler.batched_graph_keys(10, 25, 16, 2)
        )
    )
    shapes = sorted(
        {(key.batch_size, key.frames) for key in graph_keys}
        | {(1, frames) for frames in range(1, 37)}
        | {(1, 64)}
    )
    generator = torch.Generator(device="cuda:0").manual_seed(42)
    codes = [
        torch.randint(
            model.config.codebook_size,
            (batch, model.config.num_quantizers, frames),
            device="cuda:0",
            generator=generator,
        )
        for batch, frames in shapes
    ]
    recorded = os.environ.get("QWEN3_OMNI_CODES_PATH")
    if recorded is not None:
        for path in sorted(Path(recorded).glob("*.pt")):
            codes.append(torch.load(path, weights_only=True, map_location="cuda:0"))
        assert len(codes) > len(shapes), "No recorded Talker codes found"
    with torch.inference_mode():
        expected = [model(value).clone() for value in codes]
        chunk_input = codes[shapes.index((1, 64))]
        chunk_expected = {
            chunk_size: model.chunked_decode(chunk_input, chunk_size=chunk_size).clone()
            for chunk_size in (2, 10, 25)
        }
        eager_runner = Code2WavCudaGraphRunner.build(
            model,
            device="cuda:0",
            num_quantizers=model.config.num_quantizers,
            total_gpu_memory_fraction=0.2,
            graph_keys=graph_keys,
        )
        assert eager_runner.stats()["enabled"], eager_runner.stats()

        def decode_stream(
            runner: Code2WavCudaGraphRunner | None, overlap: bool
        ) -> list[torch.Tensor]:
            scheduler = code2wav_scheduler.Code2WavScheduler(
                model,
                device="cuda:0",
                enable_output_overlap=overlap,
                enable_cuda_graph=runner is not None,
                cuda_graph_runner=runner,
            )
            scheduler.stream_payloads["parity"] = make_qwen_payload(request_id="parity")
            state = scheduler.get_or_create_stream_state("parity")
            for index, frame in enumerate(chunk_input[0].T):
                scheduler.handle_stream_chunk(
                    "parity",
                    StreamItem(index, frame, "talker", metadata={"stream": True}),
                )
            scheduler.handle_stream_done("parity")
            return [torch.from_numpy(part.copy()) for part in state.audio_parts]

        stream_expected = {
            (graph, overlap): decode_stream(eager_runner if graph else None, overlap)
            for graph in (False, True)
            for overlap in (False, True)
        }
        graph_expected = []
        for value, pcm in zip(codes, expected):
            result = eager_runner.run(value)
            assert torch.equal(result.output, pcm)
            graph_expected.append(result.output.clone())

        assert snake_beta.fuse_vocoder_decoder(model.decoder) == 29
        original_launch = snake_beta.launch
        launch = Mock(wraps=original_launch)
        monkeypatch.setattr(snake_beta, "launch", launch)
        for value, pcm in zip(codes, expected):
            launch.reset_mock()
            assert torch.equal(model(value), pcm), tuple(value.shape)
            assert launch.call_count == 29, tuple(value.shape)
        launch.reset_mock()
        monkeypatch.setattr(snake_beta, "launch", original_launch)

        fused_runner = Code2WavCudaGraphRunner.build(
            model,
            device="cuda:0",
            num_quantizers=model.config.num_quantizers,
            total_gpu_memory_fraction=0.2,
            graph_keys=graph_keys,
        )
        assert fused_runner.stats()["enabled"], fused_runner.stats()
        for value, pcm in zip(codes, graph_expected):
            result = fused_runner.run(value)
            key = GraphKey(batch_size=value.shape[0], frames=value.shape[-1])
            if key in graph_keys:
                assert result.execution_mode == "cuda_graph", key
            assert torch.equal(result.output, pcm), tuple(value.shape)

        for chunk_size, pcm in chunk_expected.items():
            assert torch.equal(
                model.chunked_decode(chunk_input, chunk_size=chunk_size), pcm
            )
        for (graph, overlap), chunks in stream_expected.items():
            actual_chunks = decode_stream(fused_runner if graph else None, overlap)
            assert len(actual_chunks) == len(chunks) > 0
            assert all(
                torch.equal(actual, pcm) for actual, pcm in zip(actual_chunks, chunks)
            )
