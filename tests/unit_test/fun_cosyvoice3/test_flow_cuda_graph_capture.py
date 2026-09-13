# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

import sglang_omni.models.fun_cosyvoice3.stages as stages

pytestmark = pytest.mark.accelerator


def _cuda_flow(*, channels: int = 4, max_frames: int = 512) -> object:
    """Flow stand-in whose parameters live on CUDA so FlowCudaGraphRunner
    builds static inputs on the capture device.

    ``forward_estimator`` is a pure functional map so the Euler solve stays
    capturable: vector_field = token_condition + prompt_mel.
    """
    parameter = torch.nn.Parameter(torch.zeros(1, device="cuda"))

    def forward_estimator(
        noisy_mel_cfg,
        mel_mask_cfg,
        token_condition_cfg,
        flow_time,
        speaker_embedding_cfg,
        prompt_mel_cfg,
        *,
        streaming=False,
    ):
        return token_condition_cfg + prompt_mel_cfg

    from types import SimpleNamespace

    return SimpleNamespace(
        parameters=lambda: iter((parameter,)),
        decoder=SimpleNamespace(
            t_scheduler="linear",
            inference_cfg_rate=0.0,
            rand_noise=torch.zeros(1, channels, max_frames),
            estimator=torch.nn.Identity(),
            forward_estimator=forward_estimator,
        ),
        output_size=channels,
        token_mel_ratio=1,
        spk_embed_affine_layer=torch.nn.Linear(3, 5),
        input_embedding=lambda token: torch.ones(*token.shape, channels),
        pre_lookahead_layer=lambda x, context=None: x,
        pre_lookahead_len=3,
        cuda_graph_runner=None,
    )


def test_capture_populates_graphs_and_replays_eager_equivalent() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for flow CUDA graph capture")
    flow = _cuda_flow()
    runner = stages.FlowCudaGraphRunner(
        flow, device=torch.device("cuda"), autocast_dtype=None
    )
    shape = (1, 16)
    runner.capture((shape,))

    captured = runner.graphs.get(shape)
    assert captured is not None, "capture() must install a graph for each shape"
    for static in captured.static_inputs:
        assert static.is_cuda

    # Replay must match an eager solve for the same inputs.
    noisy_mel, time_span, token_condition, mel_mask, speaker, prompt_mel = (
        runner.capture_inputs(*shape)
    )
    noisy_mel = noisy_mel + 1.0
    token_condition = token_condition + 0.5
    prompt_mel = prompt_mel + 0.25

    replayed = runner.run(
        noisy_mel.clone(),
        time_span,
        token_condition.clone(),
        mel_mask,
        speaker,
        prompt_mel.clone(),
    )
    eager = stages.solve_flow_euler(
        flow.decoder,
        noisy_mel,
        time_span,
        token_condition,
        mel_mask,
        speaker,
        prompt_mel,
    )
    assert replayed is not None
    assert replayed.shape == eager.shape
    torch.testing.assert_close(replayed, eager, rtol=1e-4, atol=1e-4)
