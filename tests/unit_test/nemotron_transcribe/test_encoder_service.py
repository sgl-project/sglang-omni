# SPDX-License-Identifier: Apache-2.0
"""Pre-LM perception encode stays off the decoder step."""

from __future__ import annotations

import threading
from collections.abc import Iterator

import pytest
import torch
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem

from sglang_omni.models.nemotron_transcribe.encoder_service import (
    NemotronTranscribePreLMEncoderService,
    clips_share_a_batch,
)

HIDDEN_SIZE = 4
SERVICES: list[NemotronTranscribePreLMEncoderService] = []


@pytest.fixture(autouse=True)
def close_services() -> Iterator[None]:
    yield
    for service in SERVICES:
        service.close()
    SERVICES.clear()


class Perception(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(2, HIDDEN_SIZE, bias=False)


class StubModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.perception = Perception()
        self.config = None
        self.encode_calls = 0
        self.encode_batch_sizes: list[int] = []
        self.encode_gate: threading.Event | None = None
        self.encode_started: threading.Event | None = None
        self.fail_next = False
        self.fail_multi_item = False

    def get_audio_feature(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        self.encode_calls += 1
        self.encode_batch_sizes.append(len(items))
        if self.encode_started is not None:
            self.encode_started.set()
        gate = self.encode_gate
        if gate is not None:
            self.encode_gate = None
            assert gate.wait(timeout=5)
        else:
            pass
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("encode failed")
        else:
            pass
        if self.fail_multi_item and len(items) > 1:
            self.fail_multi_item = False
            raise RuntimeError("batched encode failed")
        else:
            pass
        dtype = self.perception.proj.weight.dtype
        return torch.cat(
            [
                torch.full((int(item.num_audio_tokens), HIDDEN_SIZE), 1.0, dtype=dtype)
                for item in items
            ],
            dim=0,
        )


def make_service(
    model: StubModel | None = None, *, max_batch_size: int = 4
) -> NemotronTranscribePreLMEncoderService:
    service = NemotronTranscribePreLMEncoderService(
        model or StubModel(),
        cache_namespace="nemotron-test",
        cache_max_entries=8,
        cache_max_bytes=1 << 20,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=0,
    )
    SERVICES.append(service)
    return service


def make_item(fingerprint: str, tokens: int, samples: int) -> MultimodalDataItem:
    return MultimodalDataItem(
        modality=Modality.AUDIO,
        feature=torch.zeros(samples),
        model_specific_data={
            "audio_fingerprint": fingerprint,
            "num_audio_tokens": tokens,
        },
    )


def test_clips_longer_than_one_minute_do_not_share_a_batch() -> None:
    short = 1600
    assert clips_share_a_batch(short, short * 2)
    assert not clips_share_a_batch(short, short * 5)
    assert not clips_share_a_batch(61 * 16000, short)


def test_encode_attaches_embeddings_and_drops_the_waveform() -> None:
    model = StubModel()
    service = make_service(model)
    item = make_item("clip", 3, 1600)

    service.encode_item(item)

    assert item.precomputed_embeddings.shape == (3, HIDDEN_SIZE)
    assert item.precomputed_embeddings.dtype == model.perception.proj.weight.dtype
    assert item.feature is None
    assert item.format.name == "PRECOMPUTED_EMBEDDING"
    assert model.encode_calls == 1


def test_repeated_audio_uses_the_cache() -> None:
    model = StubModel()
    service = make_service(model)
    service.encode_item(make_item("clip", 3, 1600))
    second = make_item("clip", 3, 1600)

    service.encode_item(second)

    assert model.encode_calls == 1
    assert second.precomputed_embeddings.shape == (3, HIDDEN_SIZE)
    assert service.stats()["hits"] == 1


def test_short_clips_encode_together() -> None:
    model = StubModel()
    gate = threading.Event()
    started = threading.Event()
    model.encode_gate = gate
    model.encode_started = started
    service = make_service(model)
    first = service.submit_item(make_item("a", 2, 1600))
    assert started.wait(timeout=2)
    pending = [
        service.submit_item(make_item("b", 2, 1600)),
        service.submit_item(make_item("c", 2, 1800)),
    ]
    gate.set()
    first.result(timeout=2)
    for future in pending:
        future.result(timeout=2)

    assert model.encode_batch_sizes == [1, 2]


def test_long_upload_does_not_pad_a_short_clip() -> None:
    model = StubModel()
    gate = threading.Event()
    started = threading.Event()
    model.encode_gate = gate
    model.encode_started = started
    service = make_service(model)
    first = service.submit_item(make_item("a", 2, 1600))
    assert started.wait(timeout=2)
    pending = [
        service.submit_item(make_item("long", 4, 61 * 16000)),
        service.submit_item(make_item("b", 2, 1600)),
    ]
    gate.set()
    first.result(timeout=2)
    for future in pending:
        future.result(timeout=2)

    assert model.encode_batch_sizes == [1, 1, 1]


def test_batched_failure_retries_each_item() -> None:
    model = StubModel()
    gate = threading.Event()
    started = threading.Event()
    model.encode_gate = gate
    model.encode_started = started
    model.fail_multi_item = True
    service = make_service(model)
    first = service.submit_item(make_item("a", 2, 1600))
    assert started.wait(timeout=2)
    pending = [
        service.submit_item(make_item("b", 2, 1600)),
        service.submit_item(make_item("c", 2, 1600)),
    ]
    gate.set()
    first.result(timeout=2)
    for future in pending:
        future.result(timeout=2)

    assert model.encode_batch_sizes == [1, 2, 1, 1]


def test_single_item_failure_does_not_stop_the_worker() -> None:
    model = StubModel()
    model.fail_next = True
    service = make_service(model)
    with pytest.raises(RuntimeError, match="encode failed"):
        service.encode_item(make_item("bad", 2, 1600))

    item = make_item("good", 2, 1600)
    service.encode_item(item)
    assert item.precomputed_embeddings.shape == (2, HIDDEN_SIZE)
