# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from sglang_omni.models.ming_omni.talker.audio_vae.configuration_audio_vae import (
    AudioVAEconfig,
)
from sglang_omni.models.ming_omni.talker.audio_vae.modeling_audio_vae import AudioVAE
from sglang_omni.models.ming_tts import audio_decode as audio_decode_module
from sglang_omni.models.ming_tts.audio_decode import (
    MingAudioDecoder,
    _AudioVAEFixedStreamingOutput,
    _AudioVAEFixedStreamingTransition,
    _CapturedAudioVAEGraph,
    _MingAudioStreamingRunner,
    decode_ming_tts_audio_payload,
)
from sglang_omni.models.ming_tts.payload_types import MingTTSState
from sglang_omni.proto import OmniRequest, StagePayload


class _FakeAudioVAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.empty(()))
        self.config = SimpleNamespace(sample_rate=44100)
        self.decoder = object()
        self.decode_actions: deque[torch.Tensor | Exception] = deque()
        self.decode_calls: list[torch.Tensor] = []

    def decode(self, latent: torch.Tensor, **_kwargs: Any):
        self.decode_calls.append(latent.clone())
        action = (
            self.decode_actions.popleft()
            if self.decode_actions
            else torch.tensor([0.25, -0.5], dtype=torch.float32)
        )
        if isinstance(action, Exception):
            raise action
        return action.reshape(1, 1, -1), (None, None, None), None


class _FakeTransition:
    def __init__(
        self,
        decoder: object,
        *,
        capacity: int,
        max_step_latents: int,
    ) -> None:
        self.decoder = decoder
        self.capacity = capacity
        self.max_step_latents = max_step_latents
        self.reset_rows_calls: list[tuple[int, ...]] = []
        self.reset_all_calls = 0
        self.reset_rows_error: Exception | None = None
        self.reset_all_error: Exception | None = None

    def reset_rows(self, slot_ids) -> None:
        self.reset_rows_calls.append(tuple(slot_ids))
        if self.reset_rows_error is not None:
            raise self.reset_rows_error

    def reset_all(self) -> None:
        self.reset_all_calls += 1
        if self.reset_all_error is not None:
            raise self.reset_all_error


class _FakeRunner:
    def __init__(
        self,
        transition: _FakeTransition,
        *,
        cuda_graph_required: bool,
    ) -> None:
        self.transition = transition
        self.cuda_graph_required = cuda_graph_required
        self.actions: deque[tuple[torch.Tensor, ...] | Exception] = deque()
        self.calls = 0
        self.prepare_calls = 0
        self.close_calls = 0

    @property
    def is_ready(self) -> bool:
        return True

    def run(
        self,
        *,
        slot_ids: tuple[int, ...],
        patch_groups: tuple[tuple[torch.Tensor, ...], ...],
        terminal_flags: tuple[bool, ...],
    ) -> tuple[torch.Tensor, ...]:
        del patch_groups, terminal_flags
        self.calls += 1
        action = (
            self.actions.popleft()
            if self.actions
            else tuple(torch.empty(0) for _ in slot_ids)
        )
        if isinstance(action, Exception):
            raise action
        return action

    def prepare_cuda_graph(self) -> None:
        self.prepare_calls += 1

    def close(self) -> None:
        self.close_calls += 1


def _make_facade(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[MingAudioDecoder, _FakeAudioVAE, _FakeTransition, _FakeRunner]:
    created: dict[str, object] = {}

    class FakeTransition(_FakeTransition):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            created["transition"] = self

    class FakeRunner(_FakeRunner):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            created["runner"] = self

    monkeypatch.setattr(
        audio_decode_module,
        "_AudioVAEFixedStreamingTransition",
        FakeTransition,
    )
    monkeypatch.setattr(
        audio_decode_module,
        "_MingAudioStreamingRunner",
        FakeRunner,
    )
    audio_vae = _FakeAudioVAE()
    decoder = MingAudioDecoder(
        audio_vae,
        stream_capacity=2,
        max_stream_step_latents=8,
        streaming_cuda_graph_required=True,
    )
    return (
        decoder,
        audio_vae,
        created["transition"],
        created["runner"],
    )


def _run_one_streaming_step(decoder: MingAudioDecoder):
    return decoder.run_streaming(
        slot_ids=(0,),
        patch_groups=((torch.ones((2, 3), dtype=torch.float32),),),
        terminal_flags=(False,),
    )


def test_full_empty_latents_skip_audio_vae(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decoder, audio_vae, _transition, _runner = _make_facade(monkeypatch)

    waveform = decoder.decode_full(torch.empty((0, 2, 3), dtype=torch.float32))

    assert waveform.shape == (0,)
    assert waveform.dtype == torch.float32
    assert waveform.device.type == "cpu"
    assert audio_vae.decode_calls == []


def test_streaming_error_does_not_gate_future_calls_after_owner_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decoder, audio_vae, transition, runner = _make_facade(monkeypatch)
    latents = torch.ones((1, 2, 3), dtype=torch.float32)
    runner.actions.append(RuntimeError("stream transaction failed"))

    with pytest.raises(RuntimeError, match="stream transaction failed"):
        _run_one_streaming_step(decoder)

    decoder.reset_stream_rows((0,))
    assert transition.reset_rows_calls == [(0,)]
    runner.actions.append((torch.tensor([0.5]),))
    assert _run_one_streaming_step(decoder)[0].item() == pytest.approx(0.5)
    assert decoder.decode_full(latents).numel() == 2


def test_full_error_does_not_gate_streaming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decoder, audio_vae, _transition, runner = _make_facade(monkeypatch)
    latents = torch.ones((1, 2, 3), dtype=torch.float32)
    audio_vae.decode_actions.append(RuntimeError("full device failed"))

    with pytest.raises(RuntimeError, match="full device failed"):
        decoder.decode_full(latents)

    runner.actions.append((torch.tensor([0.5]),))
    assert _run_one_streaming_step(decoder)[0].item() == pytest.approx(0.5)


class _ScriptedGraph:
    def __init__(self) -> None:
        self.replay_error: Exception | None = None
        self.replay_calls = 0
        self.reset_calls = 0

    def replay(self) -> None:
        self.replay_calls += 1
        if self.replay_error is not None:
            raise self.replay_error

    def reset(self) -> None:
        self.reset_calls += 1


class _ScriptedCudaStream:
    def __init__(self) -> None:
        self.synchronize_error: Exception | None = None
        self.synchronize_calls = 0

    def synchronize(self) -> None:
        self.synchronize_calls += 1
        if self.synchronize_error is not None:
            error = self.synchronize_error
            self.synchronize_error = None
            raise error


class _ScriptedRunnerTransition:
    capacity = 2
    max_step_latents = 2
    latent_dim = 3
    max_output_samples = 4
    input_dtype = torch.float32
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.decode_actions: deque[_AudioVAEFixedStreamingOutput | Exception] = deque()
        self.decode_calls = 0

    def decode(self, *_args, **_kwargs) -> _AudioVAEFixedStreamingOutput:
        self.decode_calls += 1
        action = self.decode_actions.popleft()
        if isinstance(action, Exception):
            raise action
        return action


def _runner_output(
    *, sample_lengths: tuple[int, int] = (2, 1)
) -> _AudioVAEFixedStreamingOutput:
    return _AudioVAEFixedStreamingOutput(
        waveform=torch.tensor(
            [[1.0, 2.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        sample_lengths=torch.tensor(sample_lengths, dtype=torch.long),
    )


def _make_scripted_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    _MingAudioStreamingRunner,
    _ScriptedRunnerTransition,
    _ScriptedGraph,
    _ScriptedCudaStream,
]:
    transition = _ScriptedRunnerTransition()
    graph = _ScriptedGraph()
    stream = _ScriptedCudaStream()
    runner = _MingAudioStreamingRunner.__new__(_MingAudioStreamingRunner)
    runner._transition = transition
    runner._cuda_graph_required_at_startup = True
    runner._startup_prepared = True
    runner._captured_graph = _CapturedAudioVAEGraph(
        graph=graph, output=_runner_output()
    )
    runner._host_latents = torch.empty((2, 2, 3), dtype=torch.float32)
    runner._host_latent_lengths = torch.empty(2, dtype=torch.long)
    runner._host_exec_mask = torch.empty(2, dtype=torch.bool)
    runner._host_terminal_mask = torch.empty(2, dtype=torch.bool)
    runner._latents = torch.empty((2, 2, 3), dtype=torch.float32)
    runner._latent_lengths = torch.empty(2, dtype=torch.long)
    runner._exec_mask = torch.empty(2, dtype=torch.bool)
    runner._terminal_mask = torch.empty(2, dtype=torch.bool)
    runner._host_waveform = torch.empty((2, 4), dtype=torch.float32)
    runner._host_sample_lengths = torch.empty(2, dtype=torch.long)
    monkeypatch.setattr(torch.cuda, "device", lambda _device: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: stream)
    return runner, transition, graph, stream


def _run_scripted_runner(
    runner: _MingAudioStreamingRunner,
) -> tuple[torch.Tensor, ...]:
    return runner.run(
        slot_ids=(0,),
        patch_groups=((torch.ones((1, 3), dtype=torch.float32),),),
        terminal_flags=(False,),
    )


@pytest.mark.parametrize("fault_phase", ["replay", "synchronize"])
def test_runtime_graph_failure_drops_graph_and_next_call_runs_eager(
    monkeypatch: pytest.MonkeyPatch,
    fault_phase: str,
) -> None:
    runner, transition, graph, stream = _make_scripted_runner(monkeypatch)
    if fault_phase == "replay":
        graph.replay_error = RuntimeError("replay failed")
    else:
        stream.synchronize_error = RuntimeError("deferred graph failure")

    with pytest.raises(RuntimeError):
        _run_scripted_runner(runner)

    assert runner.is_ready
    assert runner._captured_graph is None
    assert graph.replay_calls == 1
    assert transition.decode_calls == 0

    transition.decode_actions.append(_runner_output())
    (waveform,) = _run_scripted_runner(runner)

    assert transition.decode_calls == 1
    assert graph.replay_calls == 1
    torch.testing.assert_close(waveform, torch.tensor([1.0, 2.0]))


def test_host_staging_error_does_not_disable_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, transition, graph, _stream = _make_scripted_runner(monkeypatch)
    captured = runner._captured_graph

    with pytest.raises(RuntimeError):
        runner.run(
            slot_ids=(0,),
            patch_groups=((torch.ones((3, 3), dtype=torch.float32),),),
            terminal_flags=(False,),
        )

    assert runner._captured_graph is captured
    assert graph.replay_calls == 0
    assert transition.decode_calls == 0


def test_graph_output_validation_failure_drops_graph_and_next_call_runs_eager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, transition, graph, _stream = _make_scripted_runner(monkeypatch)
    captured = _CapturedAudioVAEGraph(
        graph=graph,
        output=_runner_output(sample_lengths=(5, 0)),
    )
    runner._captured_graph = captured

    with pytest.raises(RuntimeError, match="invalid sample length"):
        _run_scripted_runner(runner)

    assert runner.is_ready
    assert runner._captured_graph is None
    assert graph.replay_calls == 1

    transition.decode_actions.append(_runner_output())
    (waveform,) = _run_scripted_runner(runner)

    assert transition.decode_calls == 1
    assert graph.replay_calls == 1
    torch.testing.assert_close(waveform, torch.tensor([1.0, 2.0]))


@pytest.mark.parametrize(
    "host_buffer_name",
    ["_host_waveform", "_host_sample_lengths"],
)
def test_graph_output_copy_failure_drops_graph_and_next_call_runs_eager(
    monkeypatch: pytest.MonkeyPatch,
    host_buffer_name: str,
) -> None:
    runner, transition, graph, _stream = _make_scripted_runner(monkeypatch)
    target = getattr(runner, host_buffer_name)
    original_copy = torch.Tensor.copy_
    failed = False

    def fail_target_once(
        destination: torch.Tensor,
        source: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        nonlocal failed
        if destination is target and not failed:
            failed = True
            raise RuntimeError(f"{host_buffer_name} copy failed")
        return original_copy(destination, source, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", fail_target_once)

    with pytest.raises(RuntimeError, match="copy failed"):
        _run_scripted_runner(runner)

    assert runner.is_ready
    assert runner._captured_graph is None
    assert graph.replay_calls == 1
    assert transition.decode_calls == 0

    transition.decode_actions.append(_runner_output())
    (waveform,) = _run_scripted_runner(runner)

    assert transition.decode_calls == 1
    assert graph.replay_calls == 1
    torch.testing.assert_close(waveform, torch.tensor([1.0, 2.0]))


def test_owned_waveform_clone_failure_does_not_disable_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, _transition, graph, _stream = _make_scripted_runner(monkeypatch)
    captured = runner._captured_graph

    def fail_clone(_tensor: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("owned waveform clone failed")

    monkeypatch.setattr(torch.Tensor, "clone", fail_clone)

    with pytest.raises(RuntimeError, match="owned waveform clone failed"):
        _run_scripted_runner(runner)

    assert runner.is_ready
    assert runner._captured_graph is captured
    assert graph.replay_calls == 1


def test_runner_close_invalidates_startup_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, _transition, graph, stream = _make_scripted_runner(monkeypatch)

    runner.close()

    assert not runner.is_ready
    assert runner._captured_graph is None
    assert graph.reset_calls == 1
    assert stream.synchronize_calls == 1
    with pytest.raises(RuntimeError, match="not prepared"):
        _run_scripted_runner(runner)


def _make_tiny_audio_vae() -> AudioVAE:
    backbone = {
        "_attn_implementation": "sdpa",
        "attention_dropout": 0.0,
        "hidden_act": "silu",
        "hidden_size": 8,
        "initializer_range": 0.02,
        "intermediate_size": 16,
        "max_position_embeddings": 256,
        "max_window_layers": 0,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10_000.0,
        "sliding_window": 64,
        "use_cache": False,
        "use_sliding_window": True,
        "vocab_size": 1,
    }
    config = AudioVAEconfig(
        sample_rate=44100,
        enc_kwargs={
            "backbone": {**backbone, "num_hidden_layers": 4},
            "input_dim": 4,
            "hop_size": 4,
            "latent_dim": 4,
        },
        dec_kwargs={
            "backbone": {**backbone, "num_hidden_layers": 1},
            "output_dim": 4,
            "latent_dim": 4,
        },
        patch_size=4,
    )
    return AudioVAE(config).eval()


def _decode_first_fixed_row_on_cpu(
    transition: _AudioVAEFixedStreamingTransition,
    latents: torch.Tensor,
    *,
    terminal: bool,
) -> torch.Tensor:
    envelope = torch.zeros(
        (
            transition.capacity,
            transition.max_step_latents,
            transition.latent_dim,
        )
    )
    envelope[0, : latents.shape[0]].copy_(latents)
    latent_lengths = torch.zeros(transition.capacity, dtype=torch.long)
    latent_lengths[0] = latents.shape[0]
    exec_mask = torch.zeros(transition.capacity, dtype=torch.bool)
    exec_mask[0] = True
    terminal_mask = torch.zeros(transition.capacity, dtype=torch.bool)
    terminal_mask[0] = terminal
    output = transition.decode(
        envelope,
        latent_lengths,
        exec_mask,
        terminal_mask,
    )
    return output.waveform[0, : int(output.sample_lengths[0])].clone()


def _snapshot_fixed_slot_state(
    transition: _AudioVAEFixedStreamingTransition,
    slot: int,
) -> dict[str, torch.Tensor]:
    return {
        name: tensor.select(row_dim, slot).clone()
        for name, tensor, row_dim in transition._state.slot_tensors()
    }


def _dynamic_stream_parts(
    audio_vae: AudioVAE,
    latents: torch.Tensor,
    chunk_patches: tuple[int, ...],
) -> list[torch.Tensor]:
    dynamic_cache = None
    stream_state = (None, None, None)
    parts = []
    patch_start = 0
    for index, patch_count in enumerate(chunk_patches):
        patch_end = patch_start + patch_count
        sequence = latents[patch_start:patch_end].flatten(0, 1).unsqueeze(0)
        waveform, stream_state, dynamic_cache = audio_vae.decode(
            sequence,
            past_key_values=dynamic_cache,
            use_cache=True,
            stream_state=stream_state,
            last_chunk=index == len(chunk_patches) - 1,
        )
        parts.append(waveform[0, 0].detach())
        patch_start = patch_end
    return parts


@pytest.mark.parametrize("chunk_patches", [(1,), (2, 4, 2), (2, 4, 4, 4)])
def test_fixed_streaming_transition_matches_dynamic_audio_vae_on_cpu(
    chunk_patches: tuple[int, ...],
) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        audio_vae = _make_tiny_audio_vae()
        latents = torch.randn(sum(chunk_patches), 4, 4)

    dynamic_parts = _dynamic_stream_parts(audio_vae, latents, chunk_patches)
    max_step_latents = max(chunk_patches) * 4
    zero_tail = _AudioVAEFixedStreamingTransition(
        audio_vae.decoder,
        capacity=1,
        max_step_latents=max_step_latents,
    )
    nonzero_tail = _AudioVAEFixedStreamingTransition(
        audio_vae.decoder,
        capacity=1,
        max_step_latents=max_step_latents,
    )

    patch_start = 0
    for index, (patch_count, dynamic) in enumerate(
        zip(chunk_patches, dynamic_parts, strict=True)
    ):
        patch_end = patch_start + patch_count
        current = latents[patch_start:patch_end].flatten(0, 1)
        terminal = index == len(chunk_patches) - 1
        envelopes = (
            torch.zeros((1, max_step_latents, 4)),
            torch.linspace(
                0.01,
                0.25,
                steps=max_step_latents * 4,
            ).reshape(1, max_step_latents, 4),
        )
        results = []
        for transition, envelope in zip(
            (zero_tail, nonzero_tail),
            envelopes,
            strict=True,
        ):
            if chunk_patches == (2, 4, 4, 4) and terminal:
                assert transition._state.qwen_positions.item() > transition._cache_size
            envelope[0, : current.shape[0]].copy_(current)
            output = transition.decode(
                envelope,
                torch.tensor([current.shape[0]]),
                torch.ones(1, dtype=torch.bool),
                torch.tensor([terminal]),
            )
            length = int(output.sample_lengths[0])
            results.append(output.waveform[0, :length].clone())

        torch.testing.assert_close(results[0], dynamic, rtol=1e-4, atol=1e-6)
        assert torch.equal(results[0], results[1])
        patch_start = patch_end


def test_fixed_streaming_transition_matches_independent_heterogeneous_slots() -> None:
    chunk_schedules = ((2, 4, 4), (1, 1), (2,))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3)
        audio_vae = _make_tiny_audio_vae()
        slot_latents = tuple(
            torch.randn(sum(schedule), 4, 4) for schedule in chunk_schedules
        )

    dynamic_parts = tuple(
        _dynamic_stream_parts(audio_vae, latents, schedule)
        for latents, schedule in zip(slot_latents, chunk_schedules, strict=True)
    )
    transition = _AudioVAEFixedStreamingTransition(
        audio_vae.decoder,
        capacity=3,
        max_step_latents=max(max(schedule) for schedule in chunk_schedules) * 4,
    )
    events = (
        ((0, 0), (2, 0)),
        ((0, 1), (1, 0)),
        ((0, 2), (1, 1)),
    )
    patch_offsets = [0, 0, 0]

    for step_events in events:
        envelope = torch.full(
            (transition.capacity, transition.max_step_latents, transition.latent_dim),
            0.25,
        )
        latent_lengths = torch.zeros(3, dtype=torch.long)
        exec_mask = torch.zeros(3, dtype=torch.bool)
        terminal_mask = torch.zeros(3, dtype=torch.bool)
        active_slots = set()
        for slot, chunk_index in step_events:
            patch_count = chunk_schedules[slot][chunk_index]
            patch_start = patch_offsets[slot]
            patch_end = patch_start + patch_count
            current = slot_latents[slot][patch_start:patch_end].flatten(0, 1)
            envelope[slot, : current.shape[0]].copy_(current)
            latent_lengths[slot] = current.shape[0]
            exec_mask[slot] = True
            terminal_mask[slot] = chunk_index == len(chunk_schedules[slot]) - 1
            patch_offsets[slot] = patch_end
            active_slots.add(slot)

        output = transition.decode(
            envelope,
            latent_lengths,
            exec_mask,
            terminal_mask,
        )
        event_by_slot = dict(step_events)
        for slot in range(transition.capacity):
            sample_length = int(output.sample_lengths[slot])
            if slot not in active_slots:
                assert sample_length == 0
                assert torch.count_nonzero(output.waveform[slot]).item() == 0
                continue
            expected = dynamic_parts[slot][event_by_slot[slot]]
            actual = output.waveform[slot, :sample_length]
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)


def test_fixed_terminal_transition_cleans_row_for_reuse_on_cpu() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1)
        audio_vae = _make_tiny_audio_vae()
        opening = torch.randn(4, 4)
        continuation = torch.randn(4, 4)
        terminal = torch.randn(4, 4)
        reuse = torch.randn(4, 4)

    transition = _AudioVAEFixedStreamingTransition(
        audio_vae.decoder,
        capacity=1,
        max_step_latents=4,
    )
    fresh = _AudioVAEFixedStreamingTransition(
        audio_vae.decoder,
        capacity=1,
        max_step_latents=4,
    )

    _decode_first_fixed_row_on_cpu(transition, opening, terminal=False)
    _decode_first_fixed_row_on_cpu(transition, continuation, terminal=False)
    _decode_first_fixed_row_on_cpu(transition, terminal, terminal=True)

    reused = _decode_first_fixed_row_on_cpu(transition, reuse, terminal=True)
    expected = _decode_first_fixed_row_on_cpu(fresh, reuse, terminal=True)
    torch.testing.assert_close(reused, expected, rtol=1e-4, atol=1e-6)


def test_fixed_transition_keeps_inactive_row_state_unchanged_on_cpu() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1)
        audio_vae = _make_tiny_audio_vae()
        opening = torch.randn(4, 4)
        continuation = torch.randn(4, 4)
        terminal = torch.randn(4, 4)

    transition = _AudioVAEFixedStreamingTransition(
        audio_vae.decoder,
        capacity=2,
        max_step_latents=4,
    )
    reference = _AudioVAEFixedStreamingTransition(
        audio_vae.decoder,
        capacity=1,
        max_step_latents=4,
    )
    envelope = torch.zeros((2, 4, 4))
    envelope[0].copy_(opening)
    transition.decode(
        envelope,
        torch.tensor([4, 0]),
        torch.tensor([True, False]),
        torch.tensor([False, False]),
    )
    envelope[0].copy_(continuation)
    transition.decode(
        envelope,
        torch.tensor([4, 0]),
        torch.tensor([True, False]),
        torch.tensor([False, False]),
    )
    inactive_before = _snapshot_fixed_slot_state(transition, 0)

    envelope.fill_(0.25)
    envelope[1].copy_(terminal)
    mixed = transition.decode(
        envelope,
        torch.tensor([4, 4]),
        torch.tensor([False, True]),
        torch.tensor([True, True]),
    )
    reference_output = reference.decode(
        terminal.reshape(1, 4, 4),
        torch.tensor([4]),
        torch.tensor([True]),
        torch.tensor([True]),
    )

    assert mixed.sample_lengths[0].item() == 0
    assert torch.count_nonzero(mixed.waveform[0]).item() == 0
    assert mixed.sample_lengths[1].item() == reference_output.sample_lengths[0].item()
    torch.testing.assert_close(
        mixed.waveform[1],
        reference_output.waveform[0],
        rtol=1e-4,
        atol=1e-6,
    )
    inactive_after = _snapshot_fixed_slot_state(transition, 0)
    assert inactive_after.keys() == inactive_before.keys()
    for name, expected in inactive_before.items():
        assert torch.equal(inactive_after[name], expected), name


def test_fixed_transition_reset_clears_only_selected_row_and_allows_reuse() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2)
        audio_vae = _make_tiny_audio_vae()
        reuse = torch.randn(4, 4)

    transition = _AudioVAEFixedStreamingTransition(
        audio_vae.decoder,
        capacity=2,
        max_step_latents=4,
    )
    fresh = _AudioVAEFixedStreamingTransition(
        audio_vae.decoder,
        capacity=1,
        max_step_latents=4,
    )
    for index, (name, tensor, row_dim) in enumerate(transition._state.slot_tensors()):
        selected_value, neighbor_value = (
            (1, 2) if name == "upsample_pending_lengths" else (index + 1, index + 11)
        )
        tensor.select(row_dim, 0).fill_(selected_value)
        tensor.select(row_dim, 1).fill_(neighbor_value)
    selected_before = _snapshot_fixed_slot_state(transition, 0)
    neighbor_before = _snapshot_fixed_slot_state(transition, 1)
    for name, value in selected_before.items():
        assert torch.count_nonzero(value).item() == value.numel(), name

    transition.reset_rows((0,))
    selected_after = _snapshot_fixed_slot_state(transition, 0)
    neighbor_after = _snapshot_fixed_slot_state(transition, 1)
    for name, value in selected_after.items():
        assert torch.count_nonzero(value).item() == 0, name
    for name, expected in neighbor_before.items():
        assert torch.equal(neighbor_after[name], expected), name

    reused = _decode_first_fixed_row_on_cpu(transition, reuse, terminal=True)
    expected = _decode_first_fixed_row_on_cpu(fresh, reuse, terminal=True)
    torch.testing.assert_close(reused, expected, rtol=1e-4, atol=1e-6)
    neighbor_after_reuse = _snapshot_fixed_slot_state(transition, 1)
    for name, expected_state in neighbor_before.items():
        assert torch.equal(neighbor_after_reuse[name], expected_state), name


class _RecordingPayloadDecoder:
    sample_rate = 44100

    def __init__(self, waveform: torch.Tensor) -> None:
        self.waveform = waveform
        self.calls: list[torch.Tensor] = []

    def decode_full(self, latents: torch.Tensor) -> torch.Tensor:
        self.calls.append(latents.clone())
        return self.waveform.clone()


@pytest.mark.parametrize("keep_latents", [False, True])
def test_ming_tts_full_payload_decodes_once(keep_latents: bool) -> None:
    latents = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
    state = MingTTSState(
        text="hello",
        prompt_tokens=3,
        completion_tokens=2,
        generated_latents=latents,
    )
    payload = StagePayload(
        request_id="full-payload",
        request=OmniRequest(inputs="hello"),
        data=state.to_dict(),
    )
    waveform = torch.tensor([0.25, -0.5, 0.75, -1.0], dtype=torch.float32)
    decoder = _RecordingPayloadDecoder(waveform)

    result = decode_ming_tts_audio_payload(
        payload,
        decoder,
        keep_latents=keep_latents,
    )

    assert len(decoder.calls) == 1
    torch.testing.assert_close(decoder.calls[0], latents)
    restored = MingTTSState.from_dict(result.data)
    if keep_latents:
        torch.testing.assert_close(restored.generated_latents, latents)
    else:
        assert restored.generated_latents is None
    assert restored.sample_rate == 44100
    assert restored.duration_s == pytest.approx(waveform.numel() / 44100)
    assert result.data["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
    }
    audio = np.frombuffer(result.data["audio_waveform"], dtype=np.float32)
    np.testing.assert_array_equal(audio, waveform.numpy())
