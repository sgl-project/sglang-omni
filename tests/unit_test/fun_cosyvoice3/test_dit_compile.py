# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

import sglang_omni.models.fun_cosyvoice3.stages as stages


class _FakeDiTEstimator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1))

    def forward(self, x, mask, mu, t, spks=None, cond=None, streaming=False):
        del mask, mu, t, spks, cond, streaming
        return x * self.scale


class _FakeFlow(torch.nn.Module):
    def __init__(self, estimator) -> None:
        super().__init__()
        self.decoder = torch.nn.Module()
        self.decoder.estimator = estimator


class _NonModuleEstimator:
    pass


def test_compile_dit_backbone_compiles_estimator_forward_dynamic(monkeypatch) -> None:
    estimator = _FakeDiTEstimator()
    flow = _FakeFlow(estimator)
    original_forward = estimator.forward
    param_names = set(dict(estimator.named_parameters()))

    compile_calls = []
    forward_shapes = []

    def _fake_compile(fn, dynamic=None):
        compile_calls.append({"fn": fn, "dynamic": dynamic})

        def _wrapped(x, mask, mu, t, spks, cond, streaming):
            forward_shapes.append(
                (tuple(x.shape), tuple(mask.shape), tuple(mu.shape), streaming)
            )
            return fn(x, mask, mu, t, spks, cond, streaming)

        return _wrapped

    monkeypatch.setattr(torch, "compile", _fake_compile)

    assert stages._compile_dit_backbone(flow, warmup_mel_frames=16) is True

    assert [call["dynamic"] for call in compile_calls] == [True]
    assert compile_calls[0]["fn"] == original_forward
    # Bound-method compile keeps parameter names stable (no _orig_mod prefix).
    assert set(dict(estimator.named_parameters())) == param_names
    # Warmup runs the CFG [2, 80, T] signature.
    assert forward_shapes == [((2, 80, 16), (2, 1, 16), (2, 80, 16), False)] * 3


def test_compile_dit_backbone_warmup_matches_serving_grad_mode(monkeypatch) -> None:
    # flow.inference is @torch.inference_mode(); warmup must match (Dynamo
    # guards on grad mode) so the first request reuses the warmed graph.
    estimator = _FakeDiTEstimator()
    flow = _FakeFlow(estimator)
    modes = []

    def _fake_compile(fn, dynamic=None):
        def _wrapped(x, mask, mu, t, spks, cond, streaming):
            modes.append((torch.is_inference_mode_enabled(), torch.is_inference(x)))
            return fn(x, mask, mu, t, spks, cond, streaming)

        return _wrapped

    monkeypatch.setattr(torch, "compile", _fake_compile)

    stages._compile_dit_backbone(flow, warmup_mel_frames=16, warmup_steps=2)
    assert modes == [(True, True)] * 2


def test_compile_dit_backbone_skips_non_module_estimator(monkeypatch) -> None:
    flow = _FakeFlow(_NonModuleEstimator())

    def _fail_compile(fn, dynamic=None):
        raise AssertionError("torch.compile must not run for a non-module estimator")

    monkeypatch.setattr(torch, "compile", _fail_compile)

    assert stages._compile_dit_backbone(flow) is False


def test_compile_dit_backbone_falls_back_to_eager_on_compile_failure(
    monkeypatch,
) -> None:
    # torch.compile is lazy: a failure surfaces on the first warmup call and
    # must restore the eager forward.
    estimator = _FakeDiTEstimator()
    flow = _FakeFlow(estimator)
    original_forward = estimator.forward

    def _fail_compile(fn, dynamic=None):
        def _wrapped(x, mask, mu, t, spks, cond, streaming):
            raise RuntimeError("synthetic compile failure")

        return _wrapped

    monkeypatch.setattr(torch, "compile", _fail_compile)

    assert stages._compile_dit_backbone(flow, warmup_mel_frames=16) is False
    assert estimator.forward == original_forward
    # The restored eager forward still runs.
    x = torch.ones(2, 80, 16)
    assert torch.equal(estimator(x, None, None, None, None, None, False), x)


def test_compile_dit_backbone_rejects_degenerate_warmup_length(monkeypatch) -> None:
    flow = _FakeFlow(_FakeDiTEstimator())

    def _fail_compile(fn, dynamic=None):
        raise AssertionError("torch.compile must not run for invalid warmup")

    monkeypatch.setattr(torch, "compile", _fail_compile)

    try:
        stages._compile_dit_backbone(flow, warmup_mel_frames=1)
    except ValueError as exc:
        assert "warmup_mel_frames" in str(exc)
    else:
        raise AssertionError("expected ValueError for warmup_mel_frames < 2")


def test_vocoder_factory_exposes_dit_torch_compile_flag() -> None:
    import inspect

    signature = inspect.signature(stages.create_vocoder_executor)
    assert signature.parameters["enable_dit_torch_compile"].default is False
    assert signature.parameters["enable_flow_estimator_trt"].default is False
