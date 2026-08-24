# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang_omni.models.whisper_asr import encoder_compile


class _Layer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.weight


class _Encoder(torch.nn.Module):
    def __init__(self, num_layers: int = 3) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(_Layer() for _ in range(num_layers))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            features = layer(features)
        return features


def test_compile_is_skipped_for_cpu_encoder(monkeypatch) -> None:
    encoder = _Encoder()
    calls: list[object] = []
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: calls.append(fn) or fn)

    assert (
        encoder_compile.compile_encoder_layers(
            encoder, num_mel_bins=2, input_feature_len=5
        )
        is False
    )
    assert calls == []
    assert torch.equal(encoder(torch.ones(1)), torch.tensor([4.0]))


def test_bind_compiles_shared_forward_once_and_rebinds_every_layer(
    monkeypatch,
) -> None:
    encoder = _Encoder()
    compiled_targets: list[tuple[object, str | None]] = []

    def fake_compile(fn, *, mode=None):
        compiled_targets.append((fn, mode))

        def wrapped(self, hidden_states):
            return fn(self, hidden_states) * 10

        return wrapped

    monkeypatch.setattr(torch, "compile", fake_compile)

    encoder_compile._bind_compiled_layers(list(encoder.layers), mode="max-autotune")

    assert compiled_targets == [(_Layer.forward, "max-autotune")]
    assert torch.equal(encoder(torch.ones(1)), torch.tensor([2110.0]))

    encoder_compile._unbind_layers(list(encoder.layers))
    assert torch.equal(encoder(torch.ones(1)), torch.tensor([4.0]))


def test_compile_failure_restores_eager_layers(monkeypatch, caplog) -> None:
    encoder = _Encoder()
    cuda_param = SimpleNamespace(device=torch.device("cuda", 0), dtype=torch.float16)
    monkeypatch.setattr(encoder, "parameters", lambda: iter([cuda_param]))

    def failing_compile(fn, **kw):
        raise RuntimeError("no triton")

    monkeypatch.setattr(torch, "compile", failing_compile)

    with caplog.at_level(
        "WARNING", logger="sglang_omni.models.whisper_asr.encoder_compile"
    ):
        result = encoder_compile.compile_encoder_layers(
            encoder, num_mel_bins=2, input_feature_len=5
        )

    assert result is False
    assert torch.equal(encoder(torch.ones(1)), torch.tensor([4.0]))
    assert any("no triton" in (record.exc_text or "") for record in caplog.records)
