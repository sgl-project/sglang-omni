# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from sglang_omni.models.fun_cosyvoice3.flow_estimator_trt import (
    _CFG_BATCH,
    _MEL_DIM,
    _PROFILE_MAX_TIME,
    _PROFILE_MIN_TIME,
    FlowEstimatorTRT,
    FlowEstimatorTRTModule,
    _cfg_pair_shapes,
    _dynamic_shapes,
    _require_cfg_pair_inputs,
    execute_flow_estimator,
    is_flow_estimator_trt,
    resolve_flow_estimator_onnx,
)


class _ExecuteTRT:
    def __init__(self, max_batch: int) -> None:
        self.max_batch = max_batch
        self.calls: list[torch.Tensor] = []

    def execute(self, x, mask, mu, t, spks, cond):
        del mask, mu, t, spks, cond
        self.calls.append(x.detach().clone())
        return x + 1.0


def test_resolve_flow_estimator_onnx_prefers_fp32(tmp_path: Path) -> None:
    fp32 = tmp_path / "flow.decoder.estimator.fp32.onnx"
    fp16 = tmp_path / "flow.decoder.estimator.autocast_fp16.onnx"
    fp32.write_bytes(b"onnx-fp32")
    fp16.write_bytes(b"onnx-fp16")

    assert resolve_flow_estimator_onnx(str(tmp_path)) == str(fp32)


def test_resolve_flow_estimator_onnx_falls_back_to_generic(tmp_path: Path) -> None:
    generic = tmp_path / "flow.decoder.estimator.onnx"
    generic.write_bytes(b"onnx")

    assert resolve_flow_estimator_onnx(str(tmp_path)) == str(generic)


def test_resolve_flow_estimator_onnx_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No Flow estimator ONNX"):
        resolve_flow_estimator_onnx(str(tmp_path))


def test_dynamic_shapes_keep_official_cfg_batch() -> None:
    for time in (_PROFILE_MIN_TIME, 500, _PROFILE_MAX_TIME):
        shapes = _dynamic_shapes(time)
        assert list(shapes) == ["x", "mask", "mu", "cond"]
        assert all(shape[0] == _CFG_BATCH for shape in shapes.values())
        assert shapes["x"] == (_CFG_BATCH, _MEL_DIM, time)
        assert shapes["mask"] == (_CFG_BATCH, 1, time)


def test_cfg_pair_shapes_match_official_layout() -> None:
    shapes = _cfg_pair_shapes(16)
    assert shapes["t"] == (_CFG_BATCH,)
    assert shapes["spks"] == (_CFG_BATCH, _MEL_DIM)
    assert shapes["x"] == (_CFG_BATCH, _MEL_DIM, 16)


def test_require_cfg_pair_inputs_accepts_official_layout() -> None:
    frames = 16
    x = torch.zeros(_CFG_BATCH, _MEL_DIM, frames)
    mask = torch.ones(_CFG_BATCH, 1, frames)
    mu = torch.zeros_like(x)
    t = torch.zeros(_CFG_BATCH)
    spks = torch.zeros(_CFG_BATCH, _MEL_DIM)
    cond = torch.zeros_like(x)
    assert _require_cfg_pair_inputs(x, mask, mu, t, spks, cond) == _cfg_pair_shapes(
        frames
    )


def test_require_cfg_pair_inputs_rejects_wrong_t_or_spks() -> None:
    frames = 8
    x = torch.zeros(_CFG_BATCH, _MEL_DIM, frames)
    mask = torch.ones(_CFG_BATCH, 1, frames)
    mu = torch.zeros_like(x)
    cond = torch.zeros_like(x)
    with pytest.raises(ValueError, match=r"input t has shape"):
        _require_cfg_pair_inputs(
            x, mask, mu, torch.zeros(4), torch.zeros(_CFG_BATCH, _MEL_DIM), cond
        )
    with pytest.raises(ValueError, match=r"input spks has shape"):
        _require_cfg_pair_inputs(
            x, mask, mu, torch.zeros(_CFG_BATCH), torch.zeros(4, _MEL_DIM), cond
        )


def test_canonicalize_device_equates_cuda_and_cuda0(monkeypatch) -> None:
    from sglang_omni.models.fun_cosyvoice3.flow_estimator_trt import (
        _canonicalize_device,
    )

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    assert _canonicalize_device("cuda") == torch.device("cuda:0")
    assert _canonicalize_device("cuda:0") == torch.device("cuda:0")


def test_execute_flow_estimator_rejects_odd_cfg_batch() -> None:
    x = torch.zeros(3, 4, 8)
    dummy = torch.zeros_like(x)
    t = torch.zeros(3)
    spks = torch.zeros(3, 4)
    with pytest.raises(ValueError, match="even and >= 2"):
        execute_flow_estimator(_ExecuteTRT(2), x, dummy, dummy, t, spks, dummy)


def test_execute_flow_estimator_expands_broadcast_timestep() -> None:
    estimator = _ExecuteTRT(max_batch=2)
    x = torch.tensor([[[1.0]], [[-1.0]]])
    mask = torch.ones_like(x)
    mu = torch.zeros_like(x)
    t = torch.tensor([0.3])
    spks = torch.zeros(2, 1)
    cond = torch.zeros_like(x)

    out = execute_flow_estimator(estimator, x, mask, mu, t, spks, cond)

    assert len(estimator.calls) == 1
    torch.testing.assert_close(out, x + 1.0)


def test_execute_flow_estimator_chunks_cfg_pairs_not_raw_rows() -> None:
    estimator = _ExecuteTRT(max_batch=2)
    x = torch.tensor(
        [
            [[10.0]],
            [[20.0]],
            [[30.0]],
            [[-10.0]],
            [[-20.0]],
            [[-30.0]],
        ]
    )
    mask = torch.ones_like(x)
    mu = torch.zeros_like(x)
    t = torch.zeros(6)
    spks = torch.zeros(6, 1)
    cond = torch.zeros_like(x)

    out = execute_flow_estimator(estimator, x, mask, mu, t, spks, cond)

    assert [tuple(call.reshape(-1).tolist()) for call in estimator.calls] == [
        (10.0, -10.0),
        (20.0, -20.0),
        (30.0, -30.0),
    ]
    torch.testing.assert_close(out, x + 1.0)


def test_execute_flow_estimator_skips_chunking_when_engine_fits() -> None:
    estimator = _ExecuteTRT(max_batch=8)
    x = torch.arange(8, dtype=torch.float32).reshape(8, 1, 1)
    mask = torch.ones_like(x)
    mu = torch.zeros_like(x)
    t = torch.zeros(8)
    spks = torch.zeros(8, 1)
    cond = torch.zeros_like(x)

    out = execute_flow_estimator(estimator, x, mask, mu, t, spks, cond)

    assert len(estimator.calls) == 1
    assert estimator.calls[0].shape[0] == 8
    torch.testing.assert_close(out, x + 1.0)


def test_is_flow_estimator_trt_accepts_execute_wrapper() -> None:
    class _Execute:
        def execute(self, *args, **kwargs):
            del args, kwargs
            return None

    assert is_flow_estimator_trt(_Execute()) is True
    assert is_flow_estimator_trt(object()) is False
    assert is_flow_estimator_trt(torch.nn.Linear(1, 1)) is False


class _FakeTRTEngine:
    max_batch = 2


class _FallbackDiT(torch.nn.Module):
    def forward(self, x, mask, mu, t, spks, cond, streaming=False):
        del mask, mu, t, spks, cond, streaming
        return x * 2.0


def test_is_flow_estimator_trt_accepts_module_wrapper() -> None:
    module = FlowEstimatorTRTModule(_FakeTRTEngine())
    assert is_flow_estimator_trt(module) is True
    assert isinstance(module, torch.nn.Module)


def test_flow_estimator_trt_module_forwards_in_profile(monkeypatch) -> None:
    import sglang_omni.models.fun_cosyvoice3.flow_estimator_trt as trt_mod

    seen: dict[str, object] = {}

    def fake_execute(estimator, x, mask, mu, t, spks, cond):
        seen["estimator"] = estimator
        del mask, mu, t, spks, cond
        return x + 1.0

    monkeypatch.setattr(trt_mod, "execute_flow_estimator", fake_execute)
    engine = _FakeTRTEngine()
    module = FlowEstimatorTRTModule(engine)
    frames = 16
    x = torch.zeros(_CFG_BATCH, _MEL_DIM, frames)
    mask = torch.ones(_CFG_BATCH, 1, frames)
    mu = torch.zeros_like(x)
    t = torch.zeros(_CFG_BATCH)
    spks = torch.zeros(_CFG_BATCH, _MEL_DIM)
    cond = torch.zeros_like(x)

    out = module(x, mask, mu, t, spks, cond, streaming=False)

    assert seen["estimator"] is engine
    torch.testing.assert_close(out, x + 1.0)


@pytest.mark.parametrize("cfg_batch", [2, 32])
@pytest.mark.parametrize("frames", [2, 16, _PROFILE_MAX_TIME + 1])
def test_streaming_trt_uses_causal_fallback(cfg_batch, frames) -> None:
    class _CausalDiT(torch.nn.Module):
        def forward(self, x, mask, mu, t, spks, cond, streaming=False):
            assert streaming is True
            assert x.shape[0] == cfg_batch
            return x + 2.0

    engine = _ExecuteTRT(max_batch=2)
    module = FlowEstimatorTRTModule(engine, fallback=_CausalDiT())
    x = torch.zeros(cfg_batch, 1, frames)
    out = module(
        x,
        torch.ones_like(x),
        x,
        torch.zeros(cfg_batch),
        torch.zeros(cfg_batch, 1),
        x,
        streaming=True,
    )
    assert not engine.calls
    torch.testing.assert_close(out, x + 2.0)


def test_streaming_trt_rejects_missing_causal_fallback() -> None:
    engine = _ExecuteTRT(max_batch=2)
    module = FlowEstimatorTRTModule(engine)
    x = torch.zeros(2, 1, 16)
    with pytest.raises(ValueError, match="streaming=True"):
        module(x, x, x, torch.zeros(2), torch.zeros(2, 1), x, streaming=True)
    assert not engine.calls


def test_raw_trt_rejects_streaming_before_enqueue() -> None:
    estimator = object.__new__(FlowEstimatorTRT)
    x = torch.zeros(2, 1, 16)
    with pytest.raises(ValueError, match="streaming=True"):
        estimator.execute(x, x, x, torch.zeros(2), torch.zeros(2, 1), x, streaming=True)


def test_flow_estimator_trt_module_falls_back_outside_profile() -> None:
    module = FlowEstimatorTRTModule(
        _FakeTRTEngine(),
        fallback=_FallbackDiT(),
        min_time=4,
        max_time=10,
    )
    frames = 20
    x = torch.ones(_CFG_BATCH, _MEL_DIM, frames)
    mask = torch.ones(_CFG_BATCH, 1, frames)
    mu = torch.zeros_like(x)
    t = torch.zeros(_CFG_BATCH)
    spks = torch.zeros(_CFG_BATCH, _MEL_DIM)
    cond = torch.zeros_like(x)

    out = module(x, mask, mu, t, spks, cond)

    torch.testing.assert_close(out, x * 2.0)


def test_flow_estimator_trt_module_raises_without_fallback() -> None:
    module = FlowEstimatorTRTModule(_FakeTRTEngine(), min_time=4, max_time=10)
    frames = 20
    x = torch.zeros(_CFG_BATCH, _MEL_DIM, frames)
    mask = torch.ones(_CFG_BATCH, 1, frames)
    with pytest.raises(ValueError, match="outside the engine profile"):
        module(
            x,
            mask,
            torch.zeros_like(x),
            torch.zeros(_CFG_BATCH),
            torch.zeros(_CFG_BATCH, _MEL_DIM),
            torch.zeros_like(x),
        )


def test_execute_flow_estimator_requires_max_batch() -> None:
    class _NoBatch:
        def execute(self, *args, **kwargs):
            del args, kwargs
            raise AssertionError("must fail on max_batch")

    x = torch.zeros(2, 1, 1)
    dummy = torch.zeros_like(x)
    with pytest.raises(AttributeError, match="max_batch"):
        execute_flow_estimator(
            _NoBatch(), x, dummy, dummy, torch.zeros(2), torch.zeros(2, 1), dummy
        )
