# SPDX-License-Identifier: Apache-2.0
"""Tests for X-ASR model discovery and batched request isolation."""

from types import SimpleNamespace

import numpy as np

from sglang_omni.config.manager import resolve_config_cls_for_model_path
from sglang_omni.models.xasr.config import XASRPipelineConfig
from sglang_omni.models.xasr.stages import XASRWorker
from sglang_omni.proto.request import OmniRequest, StagePayload


class FakeModel:
    def __init__(self):
        self.closed = []
        self.batch_sizes = []

    def new_stream(self):
        return SimpleNamespace(tokens=[0, 0])

    def step_batch(self, states, features):
        self.batch_sizes.append(len(states))
        for state in states:
            if len(state.tokens) == 2:
                state.tokens.append(2)

    def close(self, state):
        self.closed.append(state)


class FakeTokenizer:
    def decode_pieces(self, pieces):
        return " ".join(pieces)


def test_local_model_architecture_is_resolved(tmp_path):
    (tmp_path / "config.json").write_text(
        '{"model_type":"xasr_zipformer_transducer",'
        '"architectures":["XASRZipformerTransducer"]}'
    )
    assert resolve_config_cls_for_model_path(str(tmp_path)) is XASRPipelineConfig


def test_invalid_audio_does_not_abort_other_batch_items(monkeypatch):
    worker = XASRWorker.__new__(XASRWorker)
    worker.model = FakeModel()
    worker.tokens = {2: "好"}
    worker.tokenizer = FakeTokenizer()
    monkeypatch.setattr(
        "sglang_omni.models.xasr.stages.load_audio",
        lambda source, **kwargs: (
            np.zeros(16000, dtype=np.float32)
            if source == b"good"
            else np.zeros(0, dtype=np.float32)
        ),
    )
    payloads = [
        StagePayload(str(index), OmniRequest(inputs={"audio_bytes": audio}), {})
        for index, audio in enumerate((b"good", b"bad", b"good"))
    ]
    results = worker.transcribe_batch(payloads)
    assert results[0].data["text"] == results[2].data["text"] == "好"
    assert isinstance(results[1], ValueError)
    assert len(worker.model.closed) == 2
    assert max(worker.model.batch_sizes) == 2
