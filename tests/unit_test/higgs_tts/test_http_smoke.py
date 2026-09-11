# SPDX-License-Identifier: Apache-2.0
"""Validate the HTTP smoke client's checks using mocked responses, not a model."""

import io
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from scripts import higgs_tts_http_smoke as probe


@pytest.mark.parametrize("same", [True, False])
def test_seeded_probe_fails_on_changed_audio(tmp_path, monkeypatch, same):
    def wav(scale):
        data = io.BytesIO()
        sf.write(data, np.full(240, scale), 24000, format="WAV")
        return data.getvalue()

    responses = iter(
        [
            SimpleNamespace(content=wav(0.1), raise_for_status=lambda: None),
            SimpleNamespace(
                content=wav(0.1 if same else 0.2), raise_for_status=lambda: None
            ),
            SimpleNamespace(status_code=400),
        ]
    )
    monkeypatch.setattr(probe.requests, "post", lambda *a, **k: next(responses))
    monkeypatch.setattr(sys, "argv", ["probe", "--output", str(tmp_path)])
    if same:
        probe.main()
    else:
        with pytest.raises(AssertionError, match="seeded outputs differ"):
            probe.main()
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["status"] == ("passed" if same else "failed")


@pytest.mark.parametrize("abort", [False, True])
def test_stream_checks_pcm_and_closes_response(tmp_path, monkeypatch, abort):
    closed = []
    chunks = [np.full(240, 1000, dtype="<i2").tobytes()] * 2

    class Response:
        headers = {
            "content-type": "audio/pcm",
            "x-sample-rate": "24000",
            "x-channels": "1",
            "x-bit-depth": "16",
        }

        def __enter__(self):
            return self

        def __exit__(self, *args):
            closed.append(True)

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield from chunks

    def post(url, **kwargs):
        assert kwargs["stream"] is True
        assert kwargs["json"]["response_format"] == "pcm"
        return Response()

    monkeypatch.setattr(probe.requests, "post", post)
    record = probe.check_stream(
        "http://unused", {}, timeout=1, output=tmp_path / "stream.wav", abort=abort
    )
    assert closed == [True]
    assert record["received_chunks"] == (1 if abort else 2)
    samples, rate = sf.read(tmp_path / "stream.wav", dtype="int16")
    assert rate == 24000 and len(samples) == (240 if abort else 480)
    assert (samples == 1000).all()


def test_reference_requires_matching_transcript(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["probe", "--reference-audio", "unused.wav"])
    with pytest.raises(SystemExit) as exc:
        probe.main()
    assert exc.value.code == 2
