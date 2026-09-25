# SPDX-License-Identifier: Apache-2.0
"""Graph-vs-eager contract for the slot-pool AudioVAE step.

The fake inference below is deliberately nonlinear and state-dependent per
row, so a stale static buffer, a cross-row leak, or a wrong (B, T) lookup
changes the waveform instead of cancelling out.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.dots_tts.vocoder import DotsTTSStreamingVocoder
from sglang_omni.models.dots_tts.vocoder_cuda_graph import DotsVocoderGraphRunner
from sglang_omni.models.dots_tts.vocoder_slot_pool import DotsVocoderSlotPool

LATENT_DIM = 5
HIDDEN = 8
HOP = 2
PATCH = 4
MERGE = 4
CHUNK = PATCH * MERGE
NUM_SLOTS = 6


class NonlinearInference:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.vocoder = SimpleNamespace(hop_size=HOP)
        generator = torch.Generator().manual_seed(0)
        self.drive = torch.rand(1, 1, HIDDEN, generator=generator).to(device)

    def init_stream_state(self, *, batch_size: int, chunk_size: int):
        window = torch.zeros(batch_size, LATENT_DIM, chunk_size + 4, device=self.device)
        hidden = torch.zeros(1, batch_size, HIDDEN, device=self.device)
        return SimpleNamespace(
            lstm_hidden=(hidden, hidden.clone()),
            decoder=SimpleNamespace(window=window, chunk_size=chunk_size),
        )

    def _decoder_stream_lookahead(
        self,
    ) -> int:  # noqa: leading-underscore  # AudioVAE interface
        return 1

    def _validate_stream_latents(
        self, latents: torch.Tensor
    ) -> None:  # noqa: leading-underscore  # AudioVAE interface
        if latents.ndim != 3 or int(latents.shape[1]) != LATENT_DIM:
            raise ValueError(f"bad latents {tuple(latents.shape)}")
        else:
            pass

    def _decode_stream_latents(
        self, latents, hidden
    ):  # noqa: leading-underscore  # AudioVAE interface
        hidden_h, hidden_c = hidden
        drive = latents.mean(dim=(1, 2))
        hidden_h = torch.tanh(hidden_h * 0.9 + drive[None, :, None] * self.drive)
        hidden_c = hidden_c + latents.abs().mean(dim=(1, 2))[None, :, None]
        gain = 1.0 + hidden_h.sum(dim=(0, 2))
        return latents * gain[:, None, None], (hidden_h, hidden_c)

    def _decode_stream_window(
        self, window: torch.Tensor
    ) -> torch.Tensor:  # noqa: leading-underscore  # AudioVAE interface
        mixed = torch.tanh(window.sum(dim=1).cumsum(dim=-1))
        return mixed.repeat_interleave(HOP, dim=-1).unsqueeze(1)


def make_pool(device: torch.device, *, graph_keys=None) -> DotsVocoderSlotPool:
    inference = NonlinearInference(device)
    pool = DotsVocoderSlotPool(
        inference, num_slots=NUM_SLOTS, chunk_size=CHUNK, latent_dim=LATENT_DIM
    )
    if graph_keys is not None:
        runner = DotsVocoderGraphRunner(
            forward=pool.forward,
            new_inputs=pool.new_step_inputs,
            device=pool.device,
        )
        runner.capture(graph_keys)
        pool.graph_runner = runner
    else:
        pass
    return pool


# (slots stepped together, frames): batch sizes 1..4 with uneven row ages,
# plus single-row final drains at both intermediate frame counts.
SCENARIO = [
    ([0], PATCH),
    ([0, 1], PATCH),
    ([0, 1, 2, 3], CHUNK),
    ([1, 3], CHUNK),
    ([2], PATCH),
    ([0, 1, 2], CHUNK),
    ([3, 0, 2, 1], CHUNK),
    ([1, 2, 3], PATCH),
    ([0], 2 * PATCH),
    ([1], 3 * PATCH),
]


def scenario_inputs(device: torch.device) -> list[dict[int, torch.Tensor]]:
    generator = torch.Generator().manual_seed(1234)
    steps = []
    for slots, frames in SCENARIO:
        steps.append(
            {
                slot: torch.randn(1, frames, LATENT_DIM, generator=generator).to(device)
                for slot in slots
            }
        )
    return steps


def run_scenario(pool: DotsVocoderSlotPool, steps) -> list[dict[int, torch.Tensor]]:
    for _ in range(4):
        pool.acquire()
    outputs = [pool.step(step) for step in steps]
    outputs.append({slot: pool.flush(slot) for slot in range(4)})
    return outputs


def assert_same_outputs(reference, candidate) -> None:
    assert len(reference) == len(candidate)
    for step_index, (expected, actual) in enumerate(zip(reference, candidate)):
        assert expected.keys() == actual.keys()
        for slot in expected:
            assert torch.equal(expected[slot], actual[slot]), (
                f"step {step_index} slot {slot} differs: "
                f"max|delta|={(expected[slot] - actual[slot]).abs().max().item():.3e}"
            )


@pytest.fixture
def cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    else:
        pass
    return torch.device("cuda")


ALL_KEYS = [
    (1, 4),
    (1, 8),
    (1, 12),
    (1, 16),
    (2, 4),
    (2, 16),
    (3, 4),
    (3, 16),
    (4, 4),
    (4, 16),
]


@pytest.mark.accelerator
def test_graph_step_is_bit_identical_to_eager_across_aging_rows(
    cuda_device: torch.device,
) -> None:
    device = cuda_device
    steps = scenario_inputs(device)
    reference = run_scenario(make_pool(device), steps)

    pool = make_pool(device, graph_keys=ALL_KEYS)
    assert pool.graph_runner.captured_keys == ALL_KEYS
    assert_same_outputs(reference, run_scenario(pool, steps))
    assert pool.graph_runner.replays == len(SCENARIO)
    assert pool.graph_runner.misses == 0


@pytest.mark.accelerator
def test_uncaptured_shape_falls_back_to_eager_bit_identically(
    cuda_device: torch.device,
) -> None:
    device = cuda_device
    steps = scenario_inputs(device)
    reference = run_scenario(make_pool(device), steps)

    captured = [(2, PATCH), (4, CHUNK)]
    pool = make_pool(device, graph_keys=captured)
    assert_same_outputs(reference, run_scenario(pool, steps))
    hits = sum(1 for slots, frames in SCENARIO if (len(slots), frames) in captured)
    assert pool.graph_runner.replays == hits
    assert pool.graph_runner.misses == len(SCENARIO) - hits


def make_streaming_vocoder(
    device: torch.device, **kwargs: bool | int
) -> DotsTTSStreamingVocoder:
    codec = SimpleNamespace(
        inference=NonlinearInference(device),
        lock=threading.RLock(),
        sample_rate=48000,
        patch_size=PATCH,
        latent_dim=LATENT_DIM,
        device=device,
        hop_size=HOP,
    )
    return DotsTTSStreamingVocoder(codec, **{"optimize": True, **kwargs})


def test_step_graphs_can_be_disabled() -> None:
    vocoder = make_streaming_vocoder(
        torch.device("cpu"), enable_streaming_audio_vae_cuda_graph=False
    )
    assert vocoder.ensure_slot_pool().graph_runner is None


def test_default_step_graphs_reject_a_non_cuda_codec() -> None:
    vocoder = make_streaming_vocoder(torch.device("cpu"))
    with pytest.raises(RuntimeError, match="require a CUDA device"):
        vocoder.ensure_slot_pool()


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, ALL_KEYS),
        ({"stream_slots": 1}, [(1, 4), (1, 8), (1, 12), (1, 16)]),
        ({"max_batch_size": 2}, [(1, 4), (1, 8), (1, 12), (1, 16), (2, 4), (2, 16)]),
        ({"optimize": False}, [(1, 4), (2, 4), (3, 4), (4, 4)]),
    ],
)
def test_capture_keys_match_serving_shapes(
    kwargs: dict[str, bool | int], expected: list[tuple[int, int]]
) -> None:
    vocoder = make_streaming_vocoder(torch.device("cpu"), **kwargs)
    assert vocoder.cuda_graph_capture_keys() == expected


@pytest.mark.accelerator
def test_default_vocoder_captures_serving_shapes(cuda_device: torch.device) -> None:
    vocoder = make_streaming_vocoder(cuda_device)
    pool = vocoder.ensure_slot_pool()
    assert pool.graph_runner.captured_keys == ALL_KEYS
    assert vocoder.ensure_slot_pool() is pool


@pytest.mark.accelerator
@pytest.mark.parametrize("failure_call", [1, 3], ids=["warmup", "capture"])
def test_capture_failure_aborts_pool_setup(
    cuda_device: torch.device, monkeypatch: pytest.MonkeyPatch, failure_call: int
) -> None:
    vocoder = make_streaming_vocoder(cuda_device)
    decode_window = (
        vocoder.codec.inference._decode_stream_window
    )  # noqa: leading-underscore  # AudioVAE interface
    calls = 0

    def fail_forward(window: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        if calls == failure_call:
            raise RuntimeError("injected AudioVAE failure")
        else:
            return decode_window(window)

    monkeypatch.setattr(vocoder.codec.inference, "_decode_stream_window", fail_forward)
    with pytest.raises(RuntimeError, match="injected AudioVAE failure"):
        vocoder.ensure_slot_pool()
    assert vocoder.slot_pool is None
