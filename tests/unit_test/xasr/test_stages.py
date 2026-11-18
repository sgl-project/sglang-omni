# SPDX-License-Identifier: Apache-2.0
"""Tests for X-ASR model discovery and batched request isolation."""

from types import SimpleNamespace

import kaldi_native_fbank as fbank
import numpy as np
import pytest

from sglang_omni.config.manager import resolve_config_cls_for_model_path
from sglang_omni.models.xasr.config import XASRPipelineConfig
from sglang_omni.models.xasr.stages import XASRWorker
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.serve.openai_errors import generation_error_status_code


class FakeModel:
    def __init__(self):
        self.closed = []
        self.batch_sizes = []
        self.features = []

    def new_stream(self):
        return SimpleNamespace(tokens=[0, 0])

    def step_batch(self, states, features):
        self.batch_sizes.append(len(states))
        self.features.extend(features)
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


@pytest.mark.parametrize("sample_count", [1000, 16000, 24000])
def test_frontend_matches_reference_streaming_features(monkeypatch, sample_count):
    """The model sees the reference cutoff and actual silence in its tail."""
    sample_index = np.arange(sample_count, dtype=np.float32)
    waveform = np.sin(sample_index * (2 * np.pi * 7600 / 16000)).astype(np.float32)
    worker = XASRWorker.__new__(XASRWorker)
    worker.model = FakeModel()
    worker.tokens = {2: "好"}
    worker.tokenizer = FakeTokenizer()
    monkeypatch.setattr(
        "sglang_omni.models.xasr.stages.load_audio",
        lambda source, **kwargs: waveform,
    )
    payload = StagePayload("sample", OmniRequest(inputs={"audio_bytes": b"x"}), {})
    assert worker.transcribe_one(payload).data["text"] == "好"

    options = fbank.FbankOptions()
    options.frame_opts.dither = 0
    options.frame_opts.snip_edges = False
    options.frame_opts.samp_freq = 16000
    options.mel_opts.num_bins = 80
    options.mel_opts.high_freq = -400
    reference = fbank.OnlineFbank(options)
    silence_samples = max(4800, 10080 - sample_count)
    reference.accept_waveform(16000, np.pad(waveform, (0, silence_samples)).tolist())
    frames = np.stack(
        [reference.get_frame(i) for i in range(reference.num_frames_ready)]
    )
    expected = [frames[i : i + 61] for i in range(0, len(frames) - 61 + 1, 48)]
    assert len(worker.model.features) == len(expected) > 0
    for actual, reference_chunk in zip(worker.model.features, expected):
        np.testing.assert_allclose(actual, reference_chunk, atol=1e-6)


@pytest.mark.parametrize(
    "waveform",
    [
        np.zeros(0, dtype=np.float32),
        np.full(16000, np.nan, dtype=np.float32),
        np.zeros(30 * 16000 + 1, dtype=np.float32),
    ],
)
def test_invalid_waveform_is_a_bad_request(monkeypatch, waveform):
    worker = XASRWorker.__new__(XASRWorker)
    worker.model = FakeModel()
    monkeypatch.setattr(
        "sglang_omni.models.xasr.stages.load_audio",
        lambda source, **kwargs: waveform,
    )
    payload = StagePayload("invalid", OmniRequest(inputs={"audio_bytes": b"x"}), {})
    with pytest.raises(ValueError) as error:
        worker.transcribe_one(payload)
    assert generation_error_status_code(error.value) == 400
