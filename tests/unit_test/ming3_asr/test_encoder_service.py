# SPDX-License-Identifier: Apache-2.0
"""Ming ASR pre-LM embeddings match encoding each mel on its own."""

from __future__ import annotations

import json
import os
import threading
import time
from functools import lru_cache
from pathlib import Path

import pytest
import torch

from sglang_omni.models.ming3_asr.encoder_service import (
    EncodedMelBatch,
    Ming3ASRPreLMEncoderService,
)
from sglang_omni.modules.ming_audio.audio_encoder import (
    AudioEncoderConfig,
    Ming3AudioEncoder,
)


def tiny_encoder() -> Ming3AudioEncoder:
    config = AudioEncoderConfig(
        n_mels=4,
        n_audio_ctx=32,
        n_audio_state=8,
        n_audio_head=2,
        n_audio_layer=2,
        downsampler_kernel_size_list=[3],
        downsampler_stride_list=[2],
        whisper_chunk_size=32,
    )
    return Ming3AudioEncoder(
        config,
        hidden_size=16,
        should_upcast_layer_norm=True,
        should_batch_chunks=False,
    ).eval()


def test_pre_lm_encode_matches_separate_mels() -> None:
    torch.manual_seed(0)
    encoder = tiny_encoder()
    long_mel = torch.randn(40, encoder.config.n_mels)
    short_mel = torch.randn(9, encoder.config.n_mels)
    service = Ming3ASRPreLMEncoderService(
        encoder,
        max_batch_size=8,
        max_batch_wait_ms=2,
    )
    try:
        combined = service.encode([long_mel, short_mel])
        long_embedding = service.submit([long_mel])
        short_embedding = service.submit([short_mel])
        with torch.inference_mode():
            expected = encoder([long_mel, short_mel])
            expected_long = encoder([long_mel])
            expected_short = encoder([short_mel])
        long_result = long_embedding.result(timeout=service.ENCODE_TIMEOUT_S)
        short_result = short_embedding.result(timeout=service.ENCODE_TIMEOUT_S)
    finally:
        service.close()
    torch.testing.assert_close(combined, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(long_result, expected_long, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(short_result, expected_short, rtol=1e-5, atol=1e-5)
    with pytest.raises(RuntimeError, match="closed"):
        service.encode([short_mel])


def boundary_mels(
    encoder: Ming3AudioEncoder,
    frame_counts: list[int] | None = None,
) -> list[torch.Tensor]:
    weight = encoder.audio.conv1.weight
    if frame_counts is None:
        chunk_frames = encoder.config.whisper_chunk_size
        frame_counts = [
            chunk_frames - 1,
            chunk_frames,
            chunk_frames + 1,
            (2 * chunk_frames) - 1,
        ]
    return [
        torch.randn(
            frame_count,
            encoder.config.n_mels,
            device=weight.device,
            dtype=weight.dtype,
        )
        for frame_count in frame_counts
    ]


def separate_chunk_embeddings(
    encoder: Ming3AudioEncoder,
    mels: list[torch.Tensor],
) -> list[torch.Tensor]:
    chunk_frames = encoder.config.whisper_chunk_size
    embeddings: list[torch.Tensor] = []
    with torch.inference_mode():
        for mel in mels:
            parts = [encoder.embed_mels([part])[0] for part in mel.split(chunk_frames)]
            embeddings.append(torch.cat(parts))
    return embeddings


def test_mixed_length_batch_matches_separate_chunks() -> None:
    torch.manual_seed(0)
    encoder = tiny_encoder()
    mels = boundary_mels(encoder)
    chunk_frames = encoder.config.whisper_chunk_size
    chunks = [part for mel in mels for part in mel.split(chunk_frames)]
    with torch.inference_mode():
        batched = encoder.embed_mels(chunks)
        separate = [encoder.embed_mels([part])[0] for part in chunks]
    for got, expected in zip(batched, separate, strict=True):
        torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)


def test_pre_lm_batch_matches_separate_requests_near_chunk_boundary() -> None:
    torch.manual_seed(0)
    encoder = tiny_encoder()
    mels = boundary_mels(encoder)
    request_counts: list[int] = []
    service = Ming3ASRPreLMEncoderService(
        encoder,
        max_batch_size=len(mels),
        max_batch_wait_ms=1000,
    )
    original = service.encode_batch

    def record_batch(requests: list[list[torch.Tensor]]) -> EncodedMelBatch:
        request_counts.append(len(requests))
        return original(requests)

    service.encode_batch = record_batch
    try:
        futures = [service.submit([mel]) for mel in mels]
        batched = [future.result(timeout=5) for future in futures]
    finally:
        service.close()
    assert request_counts == [len(mels)]
    separate = separate_chunk_embeddings(encoder, mels)
    for got, expected in zip(batched, separate, strict=True):
        torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)


@lru_cache(maxsize=1)
def load_checkpoint_encoder(model_path: str) -> Ming3AudioEncoder:
    configuration = json.loads((Path(model_path) / "config.json").read_text())
    encoder = Ming3AudioEncoder(
        AudioEncoderConfig.model_validate(configuration["audio_config"]),
        configuration["text_config"]["hidden_size"],
        should_upcast_layer_norm=True,
        should_batch_chunks=False,
    )
    encoder.load_checkpoint(model_path)
    return encoder.eval().to(device="cuda", dtype=torch.bfloat16)


@pytest.mark.accelerator
@pytest.mark.parametrize(
    "frame_counts",
    [
        [2999, 3001],
        [2999, 3000, 3001, 5999],
        [2999, 3000, 3001, 5999, 1, 1500, 3002, 6000],
    ],
    ids=["batch-2", "batch-4", "batch-8"],
)
def test_checkpoint_bf16_batch_matches_separate_utterances(
    frame_counts: list[int],
) -> None:
    """Real-checkpoint BF16 embeddings match encoding each chunk on its own.

    3000 mel frames is the 30 second chunk.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    model_path = os.environ.get("MING3_ASR_MODEL_PATH")
    if model_path is None:
        pytest.skip("set MING3_ASR_MODEL_PATH to a Ming ASR checkpoint")
    encoder = load_checkpoint_encoder(model_path)
    assert encoder.config.whisper_chunk_size == 3000
    torch.manual_seed(0)
    mels = boundary_mels(encoder, frame_counts)
    request_counts: list[int] = []
    service = Ming3ASRPreLMEncoderService(
        encoder,
        max_batch_size=len(mels),
        max_batch_wait_ms=1000,
    )
    original = service.encode_batch

    def record_batch(requests: list[list[torch.Tensor]]) -> EncodedMelBatch:
        request_counts.append(len(requests))
        return original(requests)

    service.encode_batch = record_batch
    try:
        futures = [service.submit([mel]) for mel in mels]
        batched = [
            future.result(timeout=service.ENCODE_TIMEOUT_S) for future in futures
        ]
    finally:
        service.close()
    assert request_counts == [len(mels)]
    separate = separate_chunk_embeddings(encoder, mels)
    for got, expected in zip(batched, separate, strict=True):
        torch.testing.assert_close(got, expected, rtol=5e-3, atol=5e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_bf16_mixed_length_batch_matches_separate_chunks() -> None:
    torch.manual_seed(0)
    encoder = tiny_encoder().to(device="cuda", dtype=torch.bfloat16)
    mels = boundary_mels(encoder)
    chunk_frames = encoder.config.whisper_chunk_size
    chunks = [part for mel in mels for part in mel.split(chunk_frames)]
    with torch.inference_mode():
        batched = encoder.embed_mels(chunks)
        separate = [encoder.embed_mels([part])[0] for part in chunks]
    for got, expected in zip(batched, separate, strict=True):
        torch.testing.assert_close(got, expected, rtol=1e-2, atol=1e-2)


def test_batch_failure_retries_each_request() -> None:
    torch.manual_seed(0)
    encoder = tiny_encoder()
    long_mel = torch.randn(12, encoder.config.n_mels)
    short_mel = torch.randn(6, encoder.config.n_mels)
    service = Ming3ASRPreLMEncoderService(
        encoder,
        max_batch_size=2,
        max_batch_wait_ms=1000,
    )
    request_counts: list[int] = []
    original = service.encode_batch

    def fail_multi(requests: list[list[torch.Tensor]]) -> EncodedMelBatch:
        request_counts.append(len(requests))
        if len(requests) > 1:
            raise RuntimeError("batch encode failed")
        else:
            pass
        return original(requests)

    service.encode_batch = fail_multi
    try:
        futures = [service.submit([long_mel]), service.submit([short_mel])]
        results = [future.result(timeout=5) for future in futures]
    finally:
        service.close()
    assert request_counts == [2, 1, 1]
    with torch.inference_mode():
        expected = [
            encoder.embed_mels([long_mel])[0],
            encoder.embed_mels([short_mel])[0],
        ]
    for got, target in zip(results, expected, strict=True):
        torch.testing.assert_close(got, target, rtol=1e-5, atol=1e-5)


def test_close_waits_for_the_in_flight_batch() -> None:
    encoder = tiny_encoder()
    service = Ming3ASRPreLMEncoderService(
        encoder,
        max_batch_size=1,
        max_batch_wait_ms=0,
    )
    entered = threading.Event()
    release = threading.Event()
    original = service.encode_batch
    closer: threading.Thread | None = None

    def wait_for_release(requests: list[list[torch.Tensor]]) -> EncodedMelBatch:
        entered.set()
        assert release.wait(timeout=5)
        return original(requests)

    service.encode_batch = wait_for_release
    mel = torch.randn(9, encoder.config.n_mels)
    try:
        future = service.submit([mel])
        assert entered.wait(timeout=5)
        closer = threading.Thread(target=service.close)
        closer.start()
        deadline = time.monotonic() + 5
        while not service.closed and time.monotonic() < deadline:
            time.sleep(0.01)
        assert service.closed
        assert service.thread.is_alive()
        release.set()
        closer.join(timeout=5)
        assert not closer.is_alive()
        assert not service.thread.is_alive()
        with torch.inference_mode():
            expected = encoder.embed_mels([mel])[0]
        torch.testing.assert_close(
            future.result(timeout=5),
            expected,
            rtol=1e-5,
            atol=1e-5,
        )
        with pytest.raises(RuntimeError, match="closed"):
            service.encode([mel])
    finally:
        release.set()
        if closer is not None:
            closer.join(timeout=5)
        else:
            service.close()
