from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from transformers import WhisperConfig
from transformers import WhisperForConditionalGeneration as ReferenceWhisper

from sglang_omni.models.whisper_asr import sglang_model
from sglang_omni.models.whisper_asr.dense_prefill import (
    DensePrefillGraph,
    WhisperDensePrefillRunner,
)
from sglang_omni.models.whisper_asr.sglang_model import WhisperForConditionalGeneration


class RecordingPool:
    def __init__(self) -> None:
        self.writes: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def set_kv_buffer(
        self,
        attention: RadixAttention,
        locations: KVWriteLoc,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        self.writes[attention.layer_id] = (
            locations.loc.clone(),
            keys.clone(),
            values.clone(),
        )


class CPUReplay:
    def __init__(
        self,
        runner: WhisperDensePrefillRunner,
        request_count: int,
        logits: torch.Tensor,
    ) -> None:
        self.runner = runner
        self.request_count = request_count
        self.logits = logits

    def replay(self) -> None:
        with torch.inference_mode():
            self.logits.copy_(self.runner.forward_static(self.request_count, 4))


@pytest.mark.parametrize("request_count", [1, 3])
def test_dense_prefill_matches_reference_and_preserves_per_request_kv(
    monkeypatch: pytest.MonkeyPatch, request_count: int
) -> None:
    """Graph prefill must preserve causal attention and each request's KV slots."""
    torch.manual_seed(37)
    config = WhisperConfig(
        d_model=8,
        encoder_layers=1,
        decoder_layers=2,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=16,
        decoder_ffn_dim=16,
        vocab_size=32,
        max_source_positions=8,
        max_target_positions=8,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=3,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
    )
    reference = ReferenceWhisper(config).eval()
    monkeypatch.setattr(sglang_model, "LogitsProcessor", lambda config: None)
    model = WhisperForConditionalGeneration(config).eval()
    model.load_weights(reference.state_dict().items())
    pool = RecordingPool()
    runner = WhisperDensePrefillRunner(
        model, pool, encoder_token_count=8, max_requests=3, prompt_lengths=(4,)
    )
    input_ids = torch.randint(0, config.vocab_size, (request_count, 4))
    encoder_states = torch.randn(request_count, 8, config.d_model)
    prompt_locations = torch.randperm(request_count * 4) + 11
    encoder_locations = torch.randperm(request_count * 8) + 31
    runner.input_ids[: request_count * 4].copy_(input_ids.flatten())
    runner.positions[: request_count * 4].copy_(torch.arange(4).repeat(request_count))
    runner.encoder_states[: request_count * 8].copy_(encoder_states.flatten(0, 1))
    runner.prompt_cache_loc[: request_count * 4].copy_(prompt_locations)
    runner.encoder_cache_loc[: request_count * 8].copy_(encoder_locations)

    with torch.inference_mode():
        actual_logits = runner.forward_static(request_count, 4)
        hidden_states = reference.model.decoder(
            input_ids=input_ids, encoder_hidden_states=encoder_states, use_cache=False
        ).last_hidden_state
        expected_logits = reference.proj_out(hidden_states[:, -1])

    torch.testing.assert_close(actual_logits, expected_logits, rtol=1e-4, atol=1e-6)

    graph_logits = torch.empty_like(actual_logits)
    runner.graphs[(request_count, 4)] = DensePrefillGraph(
        graph=CPUReplay(runner, request_count, graph_logits), logits=graph_logits
    )
    batch = SimpleNamespace(
        batch_size=request_count,
        forward_mode=ForwardMode.EXTEND,
        input_ids=input_ids.flatten(),
        positions=torch.arange(4).repeat(request_count),
        out_cache_loc=prompt_locations,
        encoder_out_cache_loc=encoder_locations,
        encoder_cached=[False] * request_count,
        encoder_lens=torch.full((request_count,), 8),
        mm_inputs=[
            SimpleNamespace(mm_items=[SimpleNamespace(precomputed_embeddings=states)])
            for states in encoder_states
        ],
    )
    first_logits = runner.run(batch, 4).next_token_logits
    torch.testing.assert_close(first_logits, expected_logits, rtol=1e-4, atol=1e-6)
    saved_logits = first_logits.clone()
    batch.input_ids = (batch.input_ids + 1) % config.vocab_size
    second_logits = runner.run(batch, 4).next_token_logits
    assert not torch.equal(second_logits, first_logits)
    assert first_logits.data_ptr() != graph_logits.data_ptr()
    assert torch.equal(first_logits, saved_logits)

    assert len(pool.writes) == config.decoder_layers * 2
    for layer in model.model.decoder.layers:
        self_locations, self_keys, self_values = pool.writes[
            layer.self_attn.attn.layer_id
        ]
        cross_locations, cross_keys, cross_values = pool.writes[
            layer.encoder_attn.attn.layer_id
        ]
        assert torch.equal(self_locations, prompt_locations)
        assert torch.equal(cross_locations, encoder_locations)
        assert self_keys.shape == self_values.shape == (request_count * 4, 2, 4)
        expected_keys, expected_values = layer.encoder_attn.kv_proj(
            encoder_states.flatten(0, 1)
        ).chunk(2, dim=-1)
        torch.testing.assert_close(cross_keys, expected_keys.reshape(-1, 2, 4))
        torch.testing.assert_close(cross_values, expected_values.reshape(-1, 2, 4))


@pytest.mark.parametrize(
    "overrides",
    [
        {"extend_seq_lens_cpu": [4, 3]},
        {"extend_seq_lens_cpu": []},
        {"extend_prefix_lens_cpu": [1, 0]},
        {"extend_prefix_lens_cpu": None},
        {"encoder_cached": [False, True]},
        {"encoder_lens_cpu": [8, 7]},
        {"return_logprob": True},
        {"capture_hidden_mode": CaptureHiddenMode.FULL},
        {"forward_mode": ForwardMode.DECODE},
        {"forward_mode": ForwardMode.MIXED},
    ],
)
def test_dense_prefill_rejects_batches_outside_capture_contract(
    overrides: dict[str, object],
) -> None:
    """Cached prefixes, mixed lengths and extra outputs must use the original path."""
    runner = object.__new__(WhisperDensePrefillRunner)
    runner.max_requests = 2
    runner.encoder_token_count = 8
    runner.prompt_lengths = (4,)
    runner.graphs = {(2, 4): None}
    fields = {
        "batch_size": 2,
        "forward_mode": ForwardMode.EXTEND,
        "extend_seq_lens_cpu": [4, 4],
        "extend_prefix_lens_cpu": [0, 0],
        "encoder_cached": [False, False],
        "encoder_lens_cpu": [8, 8],
        "return_logprob": False,
        "capture_hidden_mode": CaptureHiddenMode.NULL,
    }
    assert runner.prompt_length_of(SimpleNamespace(**fields)) == 4
    fields.update(overrides)
    assert runner.prompt_length_of(SimpleNamespace(**fields)) is None


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("request_count", [1, 3])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_dense_prefill_cuda_capture_replay_kv_and_output_lifetime(
    monkeypatch: pytest.MonkeyPatch, request_count: int, dtype: torch.dtype
) -> None:
    """Real graph replay must write the selected KV slots and own its output."""
    torch.manual_seed(37)
    config = WhisperConfig(
        d_model=8,
        encoder_layers=1,
        decoder_layers=2,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=16,
        decoder_ffn_dim=16,
        vocab_size=32,
        max_source_positions=8,
        max_target_positions=8,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=3,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
    )
    reference = ReferenceWhisper(config).eval().to(device="cuda", dtype=dtype)
    monkeypatch.setattr(sglang_model, "LogitsProcessor", lambda config: None)
    model = (
        WhisperForConditionalGeneration(config).eval().to(device="cuda", dtype=dtype)
    )
    model.load_weights(reference.state_dict().items())

    class DevicePool:
        def __init__(self) -> None:
            self.buffers = {
                attention.layer_id: (
                    torch.full((96, 2, 4), float("nan"), device="cuda", dtype=dtype),
                    torch.full((96, 2, 4), float("nan"), device="cuda", dtype=dtype),
                )
                for layer in model.model.decoder.layers
                for attention in (layer.self_attn.attn, layer.encoder_attn.attn)
            }

        def set_kv_buffer(self, attention, locations, keys, values) -> None:
            key_buffer, value_buffer = self.buffers[attention.layer_id]
            key_buffer.index_copy_(0, locations.loc, keys)
            value_buffer.index_copy_(0, locations.loc, values)

    pool = DevicePool()
    runner = WhisperDensePrefillRunner(
        model, pool, encoder_token_count=8, max_requests=3, prompt_lengths=(4,)
    )
    runner.capture()
    assert len(runner.graphs) == 3
    assert all(
        isinstance(item.graph, torch.cuda.CUDAGraph) for item in runner.graphs.values()
    )
    encoder_states = torch.randn(request_count, 8, 8, device="cuda", dtype=dtype)
    batch = SimpleNamespace(
        batch_size=request_count,
        forward_mode=ForwardMode.EXTEND,
        input_ids=torch.randint(0, 32, (request_count * 4,), device="cuda"),
        positions=torch.arange(4, device="cuda").repeat(request_count),
        out_cache_loc=torch.randperm(request_count * 4, device="cuda") + 11,
        encoder_out_cache_loc=torch.randperm(request_count * 8, device="cuda") + 31,
        encoder_cached=[False] * request_count,
        encoder_lens=torch.full((request_count,), 8, device="cuda"),
        mm_inputs=[
            SimpleNamespace(mm_items=[SimpleNamespace(precomputed_embeddings=states)])
            for states in encoder_states
        ],
    )
    tolerance = 2e-3 if dtype == torch.float16 else 2e-2
    first_logits = None
    saved_logits = None
    for replay in range(2):
        if replay:
            batch.input_ids = (batch.input_ids + 1) % 32
            # Change both the values and slot ordering on the second replay.
            encoder_states.add_(0.1)
            batch.out_cache_loc = batch.out_cache_loc.flip(0)
            batch.encoder_out_cache_loc = batch.encoder_out_cache_loc.flip(0)
        logits = runner.run(batch, 4).next_token_logits
        hidden_states = reference.model.decoder(
            input_ids=batch.input_ids.view(request_count, 4),
            encoder_hidden_states=encoder_states,
            use_cache=False,
        ).last_hidden_state
        expected_logits = reference.proj_out(hidden_states[:, -1]).float()
        torch.testing.assert_close(
            logits, expected_logits, rtol=tolerance, atol=tolerance
        )
        # Independently record eager writes to verify every selected device KV slot.
        eager_pool = RecordingPool()
        runner.token_to_kv_pool = eager_pool
        eager_logits = runner.forward_static(request_count, 4)
        runner.token_to_kv_pool = pool
        torch.testing.assert_close(logits, eager_logits, rtol=tolerance, atol=tolerance)
        for layer_id, (locations, keys, values) in eager_pool.writes.items():
            key_buffer, value_buffer = pool.buffers[layer_id]
            torch.testing.assert_close(
                key_buffer[locations], keys, rtol=tolerance, atol=tolerance
            )
            torch.testing.assert_close(
                value_buffer[locations], values, rtol=tolerance, atol=tolerance
            )
            # Capture may use slot zero; unrelated request slots must remain untouched.
            assert torch.isnan(key_buffer[80:]).all()
            assert torch.isnan(value_buffer[80:]).all()
        if replay == 0:
            first_logits = logits
            saved_logits = logits.clone()
        else:
            assert first_logits.data_ptr() != logits.data_ptr()
            assert torch.equal(first_logits, saved_logits)
            assert not torch.equal(first_logits, logits)
