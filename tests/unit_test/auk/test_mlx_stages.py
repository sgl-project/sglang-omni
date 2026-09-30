# SPDX-License-Identifier: Apache-2.0
"""Native AuK stage dispatch, request isolation, and wire contracts."""

from threading import Thread
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core")

from sglang_omni.models.auk import stages as torch_stages
from sglang_omni.models.auk.hf_config import AuKRuntimeConfig
from sglang_omni.models.auk.mlx import stages
from sglang_omni.models.auk.payload_types import AuKState
from sglang_omni.pipeline.control_plane import deserialize_message, serialize_message
from sglang_omni.proto import CompleteMessage, OmniRequest, StagePayload
from sglang_omni.scheduling.message import IncomingMessage

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")


def test_native_schedulers_evaluate_on_their_worker_threads():
    weights = mx.arange(16, dtype=mx.float32)
    mx.eval(weights)

    def compute(payloads):
        for item in payloads:
            item.data["sum"] = float(mx.sum(weights * item.data["multiplier"]).item())
        return payloads

    runners = [stages.scheduler(compute, 2, 0) for _ in range(2)]
    threads = [Thread(target=runner.start, daemon=True) for runner in runners]
    try:
        for thread in threads:
            thread.start()
        for i, runner in enumerate(runners):
            item = payload(str(i))
            item.data["multiplier"] = i + 1
            runner.inbox.put(
                IncomingMessage(request_id=str(i), type="new_request", data=item)
            )
        for i, runner in enumerate(runners):
            result = runner.outbox.get(timeout=30)
            assert result.type == "result", result.data
            assert result.request_id == str(i)
            assert result.data.data["sum"] == 120 * (i + 1)
    finally:
        for runner in runners:
            runner.stop()
        for thread in threads:
            thread.join(timeout=5)


def payload(request_id, frames=7, seed=11):
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs="hello"),
        data=AuKState(
            instruction="Say hello",
            gen_frames=frames,
            seed=seed,
            ref_audio=np.zeros(25, dtype=np.float32),
        ).to_dict(),
    )


def test_stage_handoffs_preserve_requests_and_produce_wire_safe_audio():
    encoder = Mock()
    encoder.encode_batch.return_value = [
        (mx.zeros((3, 6, 16)), mx.ones((6,), dtype=mx.bool_)) for _ in range(3)
    ]
    vae = SimpleNamespace(
        hop_size=4,
        encode=lambda waveform, lengths, key: (
            mx.random.normal((1, 7, 4), key=key),
            lengths // 4,
        ),
        decode=Mock(
            side_effect=lambda latent: mx.full(
                (latent.shape[0], latent.shape[1] * 4), 0.25
            )
        ),
    )
    flow = Mock()
    flow.sample_batch.side_effect = lambda items, **kwargs: [
        mx.zeros((item.target_frames, 4)) for item in items
    ]
    payloads = [
        payload(str(i), frames, seed)
        for i, (frames, seed) in enumerate(((7, 11), (3, 11), (7, 12)))
    ]
    conditioned = stages.condition_batch(
        payloads, encoder, vae, (mx.zeros((2,)), mx.ones((1,)))
    )
    refs = [item.data["ref_latent"] for item in conditioned]
    assert torch.equal(refs[0], refs[1])
    assert not torch.equal(refs[0], refs[2])
    assert [item.data["ref_length"] for item in conditioned] == [6, 6, 6]
    assert all(
        isinstance(item.data["conditioning"], torch.Tensor) for item in conditioned
    )
    sampled = stages.sample_batch(
        conditioned,
        flow,
        20,
        steps=4,
        cfg_strength=0,
        sway_sampling_coef=None,
        t_grid=None,
    )
    results = stages.decode_batch(sampled, vae)
    assert [call.args[0].shape[0] for call in vae.decode.call_args_list] == [2, 1]
    for i, (frames, result) in enumerate(zip((7, 3, 7), results)):
        restored = deserialize_message(
            serialize_message(
                CompleteMessage(
                    request_id=result.request_id,
                    from_stage="decode",
                    success=True,
                    result=result.data,
                )
            )
        )
        assert restored.request_id == str(i)
        assert restored.result["usage"]["completion_tokens"] == frames
        assert restored.result["audio_waveform_shape"] == [frames * 4]
        np.testing.assert_array_equal(
            np.frombuffer(restored.result["audio_waveform"], dtype=np.float32),
            np.full(frames * 4, 0.25),
        )


@pytest.mark.parametrize(
    "factory",
    [
        "create_conditioning_executor",
        "create_auk_engine_executor",
        "create_decode_executor",
    ],
)
def test_explicit_mlx_dispatch_does_not_load_torch_models(monkeypatch, factory):
    monkeypatch.setattr("sglang.srt.hardware_backend.mlx.runtime.use_mlx", lambda: True)
    native = Mock(return_value="native")
    monkeypatch.setattr(stages, factory, native)
    monkeypatch.setattr(
        torch_stages,
        "resolve_checkpoint",
        Mock(side_effect=AssertionError("Torch loader selected")),
    )
    assert (
        getattr(torch_stages, factory)("checkpoint", device="mps", gpu_id=0) == "native"
    )
    assert native.call_args.args == ("checkpoint",)
    assert native.call_args.kwargs["device"] == "mps"


@pytest.mark.parametrize("flash", [False, True])
def test_native_engine_preserves_checkpoint_recipe(monkeypatch, flash):
    monkeypatch.setattr(stages, "validate_device", lambda *args: None)
    monkeypatch.setattr(stages, "resolve_checkpoint", lambda path: path)
    monkeypatch.setattr(
        stages,
        "make_runtime_config",
        lambda path: AuKRuntimeConfig(
            model_path=path, name="AuK-Flash" if flash else "AuK"
        ),
    )
    monkeypatch.setattr(stages, "load_flow", lambda *args: object())
    sample = Mock(return_value=["sample"])
    monkeypatch.setattr(stages, "sample_batch", sample)
    executor = stages.create_auk_engine_executor(
        "checkpoint",
        device="mps",
        gpu_id=0,
        dtype="bfloat16",
        weight_dtype="bfloat16",
        nfe=8,
        cfg_strength=3,
        sway_sampling_coef=-0.5,
        max_seconds=30,
        max_batch_size=2,
        max_batch_wait_ms=0,
    )
    assert executor.fn(payload("one")) == "sample"
    assert sample.call_args.kwargs["steps"] == (4 if flash else 8)
    assert sample.call_args.kwargs["cfg_strength"] == (0 if flash else 3)
    assert sample.call_args.kwargs["sway_sampling_coef"] == (None if flash else -0.5)
    assert (sample.call_args.kwargs["t_grid"] is not None) == flash


def test_native_engine_rejects_incompatible_precision_before_loading(monkeypatch):
    monkeypatch.setattr(stages, "validate_device", lambda *args: None)
    monkeypatch.setattr(
        stages, "resolve_checkpoint", Mock(side_effect=AssertionError("downloaded"))
    )
    with pytest.raises(ValueError, match="matching dtype and weight_dtype"):
        stages.create_auk_engine_executor(
            "checkpoint",
            device="mps",
            gpu_id=0,
            dtype="bfloat16",
            weight_dtype="float32",
            nfe=8,
            cfg_strength=3,
            sway_sampling_coef=-0.5,
            max_seconds=30,
            max_batch_size=2,
            max_batch_wait_ms=0,
        )


@pytest.mark.parametrize("stage_name", ["sample", "decode"])
def test_nonfinite_model_outputs_fail_visibly(stage_name):
    request = payload("one")
    if stage_name == "sample":
        request.data.update(
            conditioning=torch.zeros(3, 4), text_mask=torch.ones(3, dtype=torch.bool)
        )
        model = SimpleNamespace(
            sample_batch=lambda *args, **kwargs: [mx.full((7, 4), float("nan"))]
        )
        with pytest.raises(RuntimeError, match="latent contains NaN/Inf"):
            stages.sample_batch(
                [request],
                model,
                20,
                steps=4,
                cfg_strength=0,
                sway_sampling_coef=None,
                t_grid=None,
            )
    else:
        request.data["latent"] = torch.zeros(7, 4)
        model = SimpleNamespace(
            decode=lambda x: mx.full((1, 28), float("inf")),
        )
        with pytest.raises(RuntimeError, match="audio contains NaN/Inf"):
            stages.decode_batch([request], model)
