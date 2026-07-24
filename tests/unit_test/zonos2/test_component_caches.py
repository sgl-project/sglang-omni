# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from types import SimpleNamespace

from sglang_omni.models.zonos2.components import audio_codec, streaming_vocoder


def test_dac_cache_reuses_device_and_reloads_on_device_change(monkeypatch) -> None:
    created = []

    class _FakeDAC:
        def eval(self):
            return self

        def to(self, device):
            self.device = device
            return self

        def modules(self):
            return []

    def _load(_checkpoint):
        model = _FakeDAC()
        created.append(model)
        return model

    dac_module = SimpleNamespace(
        DAC=SimpleNamespace(load=_load),
        utils=SimpleNamespace(download=lambda **_kwargs: "checkpoint"),
    )
    monkeypatch.setitem(sys.modules, "dac", dac_module)
    monkeypatch.setattr(audio_codec, "_dac_cache", None)

    first = audio_codec._get_dac("cuda:0")
    assert audio_codec._get_dac("cuda:0") is first
    assert audio_codec._get_dac("cuda:1") is not first
    assert [model.device for model in created] == ["cuda:0", "cuda:1"]


def test_vocoder_cache_reuses_device_and_reloads_on_device_change(monkeypatch) -> None:
    created = []

    class _FakeVocoder:
        def __init__(self, device):
            self.device = device
            created.append(self)

    monkeypatch.setattr(streaming_vocoder, "Zonos2DACVocoder", _FakeVocoder)
    monkeypatch.setattr(streaming_vocoder, "_vocoder_cache", None)

    first = streaming_vocoder._get_vocoder("cuda:0")
    assert streaming_vocoder._get_vocoder("cuda:0") is first
    assert streaming_vocoder._get_vocoder("cuda:1") is not first
    assert [vocoder.device for vocoder in created] == ["cuda:0", "cuda:1"]
